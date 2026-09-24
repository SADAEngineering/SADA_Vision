"""Der Vertrag auf dem Draht.

Die Zusicherungen hier sind die, auf die sich Unity und TraceForm verlassen.
Wer einen dieser Tests rot macht, bricht einen Aufrufer - nicht nur Code.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from sada_vision import CONTRACT_VERSION
from sada_vision.api.app import create_app
from sada_vision.models import registry
from tests.synthetic import as_png, blank, straight_crack, two_cracks, with_aruco


@pytest.fixture(scope="module")
def client():
    registry.reset_cache()
    with TestClient(create_app()) as c:
        yield c
    registry.reset_cache()


def _post(client, rgb, **form):
    return client.post(
        "/api/v1/detect/crack",
        files={"image": ("probe.png", as_png(rgb), "image/png")},
        data={k: str(v) for k, v in form.items()},
    )


# --------------------------------------------------------------- Gesundheit --


def test_live_antwortet_immer(client):
    assert client.get("/health/live").json() == {"status": "ok"}


def test_health_nennt_modell_und_vertrag(client):
    body = client.get("/health").json()
    assert body["contract_version"] == CONTRACT_VERSION
    assert body["service"] == "sada-vision"
    assert len(body["tasks"]["items"]) >= 1
    crack = body["tasks"]["items"][0]
    assert crack["task"] == "crack"
    # Ohne Gewichte laeuft der Notbehelf - und das muss sichtbar sein.
    assert crack["trained"] is False
    assert body["status"] == "degraded"


# ------------------------------------------------------------------ Vertrag --


def test_antwort_hat_alle_zugesicherten_felder(client):
    rgb, _ = straight_crack(width_px=6.0)
    body = _post(client, rgb).json()

    for key in (
        "contract_version", "task", "request_id", "image", "model",
        "scale", "summary", "instances", "warnings", "duration_ms",
    ):
        assert key in body, key
    assert body["task"] == "crack"
    assert body["contract_version"] == CONTRACT_VERSION
    assert body["image"]["width"] == rgb.shape[1]
    assert body["image"]["height"] == rgb.shape[0]


def test_listen_stehen_im_umschlag(client):
    """JsonUtility kann keine nackte Liste deserialisieren."""
    rgb, _ = straight_crack()
    body = _post(client, rgb).json()
    assert isinstance(body["instances"], dict) and "items" in body["instances"]
    assert isinstance(body["warnings"], dict) and "items" in body["warnings"]
    for inst in body["instances"]["items"]:
        assert isinstance(inst["paths"], dict) and "items" in inst["paths"]


def test_kein_null_in_der_ganzen_antwort(client):
    """JsonUtility kennt kein Optional - unbekannt ist -1 oder leer."""
    rgb, _ = straight_crack()
    body = _post(client, rgb).json()

    def walk(node, path=""):
        if node is None:
            raise AssertionError(f"null bei {path}")
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(body)


def test_verlauf_ist_flach_und_paarweise(client):
    rgb, _ = straight_crack(width_px=6.0)
    body = _post(client, rgb).json()
    instances = body["instances"]["items"]
    assert instances, "Der Notbehelf sollte den gezeichneten Riss finden"

    for inst in instances:
        for path in inst["paths"]["items"]:
            n = path["point_count"]
            assert len(path["path_yx"]) == 2 * n
            assert len(path["width_px"]) == n
            assert len(path["width_mm"]) == n


def test_koordinaten_liegen_im_bild(client):
    rgb, _ = straight_crack()
    body = _post(client, rgb).json()
    h, w = rgb.shape[:2]
    for inst in body["instances"]["items"]:
        for path in inst["paths"]["items"]:
            ys = path["path_yx"][0::2]
            xs = path["path_yx"][1::2]
            assert all(0 <= y <= h for y in ys)
            assert all(0 <= x <= w for x in xs)


def test_ohne_massstab_sind_alle_millimeter_minus_eins(client):
    rgb, _ = straight_crack()
    body = _post(client, rgb).json()
    assert body["scale"]["known"] is False
    assert body["scale"]["mm_per_px"] == -1.0
    for inst in body["instances"]["items"]:
        assert inst["width_max_mm"] == -1.0
        assert inst["severity"] == "unknown"


def test_mit_vorgegebenem_massstab_kommen_millimeter(client):
    rgb, _ = straight_crack(width_px=6.0)
    body = _post(client, rgb, mm_per_px=0.05).json()
    assert body["scale"]["known"] is True
    assert body["scale"]["source"] == "explicit"
    for inst in body["instances"]["items"]:
        assert inst["width_max_mm"] > 0
        assert inst["severity"] != "unknown"


def test_marker_im_bild_ergibt_den_massstab(client):
    rgb, _ = straight_crack(h=600, w=800, width_px=6.0)
    rgb = with_aruco(rgb, marker_id=7, size_px=120)
    body = _post(client, rgb, marker_size_mm=60.0).json()
    assert body["scale"]["source"] == "aruco"
    assert body["scale"]["mm_per_px"] == pytest.approx(0.5, rel=0.08)


def test_lidar_felder_ergeben_den_massstab(client):
    rgb, _ = straight_crack()
    body = _post(client, rgb, depth_mm=500.0, focal_px=1000.0).json()
    assert body["scale"]["source"] == "lidar"
    assert body["scale"]["mm_per_px"] == pytest.approx(0.5)


def test_notbehelf_wird_in_jeder_antwort_gemeldet(client):
    rgb, _ = straight_crack()
    body = _post(client, rgb).json()
    assert body["model"]["trained"] is False
    assert body["model"]["kind"] == "classic"
    assert any("nicht belastbar" in w for w in body["warnings"]["items"])


def test_ohne_verlauf_kommen_nur_kennzahlen(client):
    rgb, _ = straight_crack()
    voll = _post(client, rgb, include_paths=True).json()
    knapp = _post(client, rgb, include_paths=False).json()
    assert len(json.dumps(knapp)) < len(json.dumps(voll))
    for inst in knapp["instances"]["items"]:
        assert inst["paths"]["items"] == []
        assert inst["width_max_px"] > 0  # Kennzahlen bleiben


def test_zusammenfassung_passt_zu_den_befunden(client):
    body = _post(client, two_cracks(), mm_per_px=0.05).json()
    items = body["instances"]["items"]
    assert body["summary"]["instance_count"] == len(items)
    if items:
        assert body["summary"]["max_width_px"] == pytest.approx(
            max(i["width_max_px"] for i in items)
        )


def test_befunde_stehen_nach_breite_sortiert(client):
    body = _post(client, two_cracks(), mm_per_px=0.05).json()
    widths = [i["width_max_px"] for i in body["instances"]["items"]]
    assert widths == sorted(widths, reverse=True)
    ids = [i["instance_id"] for i in body["instances"]["items"]]
    assert ids == list(range(len(ids)))


# ------------------------------------------------------------------ Fehler --


def test_kaputtes_bild_wird_mit_422_abgelehnt(client):
    r = client.post(
        "/api/v1/detect/crack",
        files={"image": ("x.png", b"kein bild", "image/png")},
    )
    assert r.status_code == 422
    assert r.json()["error"] == "unprocessable_entity"


def test_fehlendes_bild_ist_ein_validierungsfehler(client):
    r = client.post("/api/v1/detect/crack", data={"mm_per_px": "0.1"})
    assert r.status_code == 422
    assert r.json()["error"] == "validation_failed"


def test_unbekanntes_breitenverfahren_wird_abgelehnt(client):
    rgb, _ = straight_crack()
    r = _post(client, rgb, width_method="raten")
    assert r.status_code == 422


def test_geplante_aufgaben_melden_501(client):
    r = client.post("/api/v1/detect/bolt", files={"image": ("x.png", b"x", "image/png")})
    assert r.status_code == 501
    assert "planned" in r.json()["detail"]


def test_unbekannte_aufgabe_meldet_404(client):
    r = client.post("/api/v1/detect/quatsch", files={"image": ("x.png", b"x", "image/png")})
    assert r.status_code == 404


def test_jede_antwort_traegt_korrelation_und_vertragsversion(client):
    rgb, _ = straight_crack()
    r = _post(client, rgb)
    assert r.headers["X-Contract-Version"] == CONTRACT_VERSION
    assert r.headers["X-Correlation-Id"]


def test_mitgegebene_korrelation_wird_zurueckgegeben(client):
    rgb, _ = straight_crack()
    r = client.post(
        "/api/v1/detect/crack",
        files={"image": ("p.png", as_png(rgb), "image/png")},
        headers={"X-Correlation-Id": "tf-12345"},
    )
    assert r.headers["X-Correlation-Id"] == "tf-12345"


# ----------------------------------------------------------------- Vorschau --


def test_vorschau_liefert_ein_png(client):
    rgb, _ = straight_crack()
    r = client.post(
        "/api/v1/preview/crack",
        files={"image": ("p.png", as_png(rgb), "image/png")},
        data={"mm_per_px": "0.05"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_leere_flaeche_bringt_den_dienst_nicht_aus_dem_tritt(client):
    r = _post(client, blank())
    assert r.status_code == 200
    assert r.json()["summary"]["instance_count"] >= 0


def test_openapi_beschreibt_beide_endpunkte(client):
    spec = client.get("/openapi.json").json()
    assert "/api/v1/detect/crack" in spec["paths"]
    assert "/api/v1/preview/crack" in spec["paths"]
