"""Der Durchstich: Bild rein, gepruefter Befund raus.

Damit hier die *Pipeline* geprueft wird und nicht die Qualitaet eines Netzes,
tritt ein Orakel-Segmentierer an seine Stelle: er kennt die wahre Maske. Was
dann noch schiefgehen kann, sind Koordinatenumrechnung, Zuschnitt,
Verkleinerung, Massstab und Einordnung - also genau das, was dieser Test
absichern soll.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from sada_vision.config import get_settings, reset_settings
from sada_vision.domain import ScaleInfo, SourceImage
from sada_vision.models import base as models_base
from sada_vision.models import registry
from sada_vision.pipeline import analyse_crack
from tests.synthetic import concrete_background, draw_crack


class OracleSegmenter:
    """Liefert die gezeichnete Maske als weiche Wahrscheinlichkeit."""

    info = models_base.SegmenterInfo(
        name="oracle", kind="onnx", trained=True, note="Testdoppel"
    )

    def __init__(self, prob: np.ndarray) -> None:
        self._prob = prob

    def probability(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        if (h, w) != self._prob.shape:
            return cv2.resize(self._prob, (w, h), interpolation=cv2.INTER_LINEAR)
        return self._prob


@pytest.fixture(autouse=True)
def _clean_registry():
    registry.reset_cache()
    reset_settings()
    yield
    registry.reset_cache()
    reset_settings()


def _install(monkeypatch, prob: np.ndarray) -> None:
    monkeypatch.setattr(registry, "_build", lambda task: OracleSegmenter(prob))


def _straight(h=400, w=700, thickness=6.0, angle_deg=0.0):
    """Bild, Wahrscheinlichkeitsfeld und wahre Punkte eines geraden Risses."""
    bg = concrete_background(h, w, seed=11)
    cy, cx = h / 2.0, w / 2.0
    half = min(h, w) * 0.4
    a = np.radians(angle_deg)
    t = np.linspace(-half, half, 400)
    points = np.stack([cy - np.sin(a) * t, cx + np.cos(a) * t], axis=1).astype(np.float32)
    img = draw_crack(bg, points, thickness)
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Wahres Wahrscheinlichkeitsfeld: Abstand zur Mittellinie, weich.
    canvas = np.zeros((h, w), dtype=np.uint8)
    prob = draw_crack(canvas, points, thickness, darkness=-255).astype(np.float32) / 255.0
    return rgb, np.clip(prob, 0, 1).astype(np.float32), points


def test_gerader_riss_wird_als_ein_befund_mit_einem_ast_gemeldet(monkeypatch):
    rgb, prob, _ = _straight()
    _install(monkeypatch, prob)

    source = SourceImage(rgb=rgb, width=rgb.shape[1], height=rgb.shape[0])
    result = analyse_crack(source, ScaleInfo(source="none"))

    assert len(result.instances) == 1
    inst = result.instances[0]
    assert len(inst.paths) == 1
    assert inst.pattern == "linear"
    assert inst.branch_count == 0
    assert inst.tortuosity < 1.05


@pytest.mark.parametrize("thickness", [4.0, 6.0, 10.0])
def test_breite_in_pixeln_trifft_die_gezeichnete(monkeypatch, thickness):
    rgb, prob, _ = _straight(thickness=thickness)
    _install(monkeypatch, prob)

    source = SourceImage(rgb=rgb, width=rgb.shape[1], height=rgb.shape[0])
    result = analyse_crack(source, ScaleInfo(source="none"))

    inst = result.instances[0]
    mid = np.concatenate([p.width_px for p in inst.paths])
    # Die Enden einer gezeichneten Linie laufen spitz aus, deshalb der Median
    # ueber das mittlere Drittel statt des Maximums.
    core = np.sort(mid)[len(mid) // 4 : -len(mid) // 4]
    assert abs(float(np.median(core)) - thickness) < 0.8


def test_ohne_massstab_bleiben_millimeter_bei_minus_eins(monkeypatch):
    rgb, prob, _ = _straight()
    _install(monkeypatch, prob)
    source = SourceImage(rgb=rgb, width=rgb.shape[1], height=rgb.shape[0])
    result = analyse_crack(source, ScaleInfo(source="none"))

    inst = result.instances[0]
    assert inst.width_max_mm == -1.0
    assert inst.length_mm == -1.0
    assert all(np.all(p.width_mm == -1.0) for p in inst.paths)


def test_mit_massstab_werden_millimeter_gerechnet(monkeypatch):
    thickness = 6.0
    mm_per_px = 0.05  # 6 px = 0,30 mm
    rgb, prob, _ = _straight(thickness=thickness)
    _install(monkeypatch, prob)

    source = SourceImage(rgb=rgb, width=rgb.shape[1], height=rgb.shape[0])
    scale = ScaleInfo(source="explicit", mm_per_px=mm_per_px, confidence=1.0)
    result = analyse_crack(source, scale)

    inst = result.instances[0]
    mm = np.concatenate([p.width_mm for p in inst.paths])
    core = np.sort(mm)[len(mm) // 4 : -len(mm) // 4]
    assert abs(float(np.median(core)) - thickness * mm_per_px) < 0.05
    assert inst.severity in {"moderate", "wide", "fine"}


@pytest.mark.parametrize(
    ("angle", "expected_class"),
    [(0.0, "horizontal"), (90.0, "vertical"), (45.0, "diagonal")],
)
def test_lage_wird_richtig_eingeordnet(monkeypatch, angle, expected_class):
    rgb, prob, _ = _straight(h=500, w=500, angle_deg=angle)
    _install(monkeypatch, prob)
    source = SourceImage(rgb=rgb, width=rgb.shape[1], height=rgb.shape[0])
    result = analyse_crack(source, ScaleInfo(source="none"))
    assert result.instances[0].orientation_class == expected_class


def test_verlauf_liegt_auf_der_gezeichneten_linie(monkeypatch):
    """Jeder gemeldete Punkt hoechstens 1,5 px von der Wahrheit entfernt."""
    rgb, prob, truth = _straight(thickness=5.0)
    _install(monkeypatch, prob)
    source = SourceImage(rgb=rgb, width=rgb.shape[1], height=rgb.shape[0])
    result = analyse_crack(source, ScaleInfo(source="none"))

    reported = np.concatenate([p.points_yx for p in result.instances[0].paths])
    # Abstand jedes gemeldeten Punktes zur naechsten wahren Stuetzstelle
    d = np.linalg.norm(reported[:, None, :] - truth[None, :, :], axis=2)
    assert float(np.max(np.min(d, axis=1))) < 1.5


def test_zwei_risse_werden_zwei_befunde(monkeypatch):
    h, w = 400, 700
    bg = concrete_background(h, w, seed=13)
    a = np.stack([np.full(300, 100.0), np.linspace(50, 650, 300)], axis=1).astype(np.float32)
    b = np.stack([np.full(300, 300.0), np.linspace(50, 650, 300)], axis=1).astype(np.float32)
    img = draw_crack(draw_crack(bg, a, 8.0), b, 3.0)
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    canvas = np.zeros((h, w), dtype=np.uint8)
    prob = draw_crack(draw_crack(canvas, a, 8.0, -255), b, 3.0, -255)
    prob = np.clip(prob.astype(np.float32) / 255.0, 0, 1)
    _install(monkeypatch, prob)

    source = SourceImage(rgb=rgb, width=w, height=h)
    result = analyse_crack(source, ScaleInfo(source="none"))

    assert len(result.instances) == 2
    # Der breitere steht oben.
    assert result.instances[0].width_max_px > result.instances[1].width_max_px


def test_verzweigter_riss_meldet_mehrere_aeste(monkeypatch):
    h, w = 400, 600
    canvas = np.zeros((h, w), dtype=np.uint8)
    trunk = np.stack(
        [np.linspace(50, 350, 300), np.full(300, 300.0)], axis=1
    ).astype(np.float32)
    branch = np.stack(
        [np.linspace(200, 330, 200), np.linspace(300, 560, 200)], axis=1
    ).astype(np.float32)
    prob = draw_crack(draw_crack(canvas, trunk, 6.0, -255), branch, 5.0, -255)
    prob = np.clip(prob.astype(np.float32) / 255.0, 0, 1)

    rgb = cv2.cvtColor(
        draw_crack(draw_crack(concrete_background(h, w, 17), trunk, 6.0), branch, 5.0),
        cv2.COLOR_GRAY2RGB,
    )
    _install(monkeypatch, prob)

    source = SourceImage(rgb=rgb, width=w, height=h)
    result = analyse_crack(source, ScaleInfo(source="none"))

    inst = result.instances[0]
    assert inst.branch_count >= 1
    assert len(inst.paths) == 3
    assert inst.pattern == "branched"


def test_leere_flaeche_liefert_keinen_befund(monkeypatch):
    h, w = 300, 300
    _install(monkeypatch, np.zeros((h, w), dtype=np.float32))
    rgb = cv2.cvtColor(concrete_background(h, w, 19), cv2.COLOR_GRAY2RGB)
    result = analyse_crack(SourceImage(rgb=rgb, width=w, height=h), ScaleInfo(source="none"))
    assert result.instances == []


def test_verkleinerung_rechnet_koordinaten_zurueck(monkeypatch):
    """Bei begrenzter Arbeitskante muessen die Punkte trotzdem im Original liegen."""
    h, w = 800, 1200
    rgb, prob, truth = _straight(h=h, w=w, thickness=10.0)
    _install(monkeypatch, prob)

    settings = get_settings()
    monkeypatch.setattr(settings, "max_edge_px", 600)

    source = SourceImage(rgb=rgb, width=w, height=h)
    result = analyse_crack(source, ScaleInfo(source="none"))

    assert result.image_width == w and result.image_height == h
    reported = np.concatenate([p.points_yx for p in result.instances[0].paths])
    assert reported[:, 0].max() <= h
    assert reported[:, 1].max() <= w
    d = np.linalg.norm(reported[:, None, :] - truth[None, :, :], axis=2)
    # Grober als ohne Verkleinerung - aber der Verlauf muss stimmen.
    assert float(np.median(np.min(d, axis=1))) < 4.0
    assert any("verkleinert" in warning for warning in result.warnings)


def test_gewoehnliches_handyfoto_wird_angenommen():
    """48 Megapixel sind bei heutigen Telefonen normal, keine Bombe."""
    from sada_vision.config import get_settings as _settings

    assert _settings().max_image_pixels >= 50_000_000


def test_viel_zu_grosses_bild_wird_abgelehnt(monkeypatch):
    """Der Bombenschutz muss trotzdem greifen."""
    import io as _io

    from PIL import Image as _Image

    from sada_vision.pipeline.preprocess import ImageRejected, decode

    settings = get_settings()
    monkeypatch.setattr(settings, "max_image_pixels", 1000)

    buffer = _io.BytesIO()
    _Image.new("RGB", (200, 200)).save(buffer, format="PNG")
    with pytest.raises(ImageRejected):
        decode(buffer.getvalue())


def test_winziges_bild_wird_abgelehnt():
    import io as _io

    from PIL import Image as _Image

    from sada_vision.pipeline.preprocess import ImageRejected, decode

    buffer = _io.BytesIO()
    _Image.new("RGB", (20, 20)).save(buffer, format="PNG")
    with pytest.raises(ImageRejected, match="zu klein"):
        decode(buffer.getvalue())
