"""Die drei Dinge, die eine Zahl vom Messwert trennen.

Der Dienst hat von Anfang an einen Verlauf und eine Breite geliefert. Was
lange fehlte, war die Unterscheidung zwischen einem Wert, der gilt, und
einem, der nur dasteht:

1. **An einer Verzweigung misst jedes Verfahren zu breit.** In eine Gabelung
   passt ein groesserer Kreis als in den Riss. Der Hoechstwert eines
   verzweigten Risses war deshalb regelmaessig ein Artefakt - und genau er
   bestimmt die Einstufung.
2. **Ein angeschnittener Riss laeuft weiter.** Seine Laenge ist dann eine
   Untergrenze, wurde aber als Zahl gemeldet.
3. **Die Breite wurde nur dort gemessen, wo die Formvereinfachung einen
   Punkt gelassen hat** - auf einem geraden Stueck also fast nirgends.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from sada_vision.config import get_settings, reset_settings
from sada_vision.domain import ScaleInfo, SourceImage
from sada_vision.models import base as models_base
from sada_vision.models import registry
from sada_vision.pipeline import analyse_crack, geometry, width
from tests.synthetic import concrete_background, draw_crack


class OracleSegmenter:
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
def _clean():
    registry.reset_cache()
    reset_settings()
    yield
    registry.reset_cache()
    reset_settings()


def _install(monkeypatch, prob):
    monkeypatch.setattr(registry, "_build", lambda task: OracleSegmenter(prob))


def _t_riss(h=400, w=600, dicke=6.0):
    """Ein T: waagerechter Stamm, senkrechter Ast, ueberall gleich breit."""
    canvas = np.zeros((h, w), dtype=np.uint8)
    stamm = np.stack(
        [np.full(400, 150.0), np.linspace(60, w - 60, 400)], axis=1
    ).astype(np.float32)
    ast = np.stack(
        [np.linspace(150, h - 60, 300), np.full(300, w / 2)], axis=1
    ).astype(np.float32)
    prob = draw_crack(draw_crack(canvas, stamm, dicke, -255), ast, dicke, -255)
    prob = np.clip(prob.astype(np.float32) / 255.0, 0, 1)

    rgb = cv2.cvtColor(
        draw_crack(draw_crack(concrete_background(h, w, 31), stamm, dicke), ast, dicke),
        cv2.COLOR_GRAY2RGB,
    )
    return rgb, prob


# ------------------------------------------------- 1 · Verzweigungen ------


def test_kreuzung_wird_gefunden():
    m = np.zeros((160, 160), dtype=bool)
    m[78:82, 20:140] = True
    m[80:140, 78:82] = True
    skelett = geometry.skeleton_branches(m)
    assert skelett.branch_count >= 1
    assert skelett.junctions.shape[0] == skelett.branch_count
    # Die Kreuzung liegt dort, wo sich Stamm und Ast treffen.
    y, x = skelett.junctions[0]
    assert abs(y - 80) < 8 and abs(x - 80) < 8


def test_ohne_verzweigung_ist_die_maske_leer():
    punkte = np.stack([np.zeros(50), np.arange(50.0)], axis=1)
    maske = width.junction_mask(punkte, np.zeros((0, 2)), np.full(50, 4.0))
    assert not maske.any()


def test_stuetzstellen_nahe_der_kreuzung_werden_markiert():
    punkte = np.stack([np.zeros(60), np.arange(60.0)], axis=1)
    kreuzung = np.array([[0.0, 30.0]])
    breiten = np.full(60, 4.0)          # Radius = 1,5 * 4 = 6 px
    maske = width.junction_mask(punkte, kreuzung, breiten, radius_factor=1.5)

    assert maske[30], "Der Punkt auf der Kreuzung muss markiert sein"
    assert maske[26] and maske[34], "Der Umkreis gehoert dazu"
    assert not maske[0] and not maske[59], "Weit weg nicht"


def test_radius_waechst_mit_der_breite():
    """Bei einem breiten Riss reicht die Verzerrung weiter."""
    punkte = np.stack([np.zeros(60), np.arange(60.0)], axis=1)
    kreuzung = np.array([[0.0, 30.0]])
    schmal = width.junction_mask(punkte, kreuzung, np.full(60, 2.0))
    breit = width.junction_mask(punkte, kreuzung, np.full(60, 10.0))
    assert breit.sum() > schmal.sum()


def test_hoechstwert_stammt_nicht_mehr_von_der_gabelung(monkeypatch):
    """Der Kern der Sache: gleich breites T, also kein Ausreisser.

    Der Riss ist ueberall 6 px breit. Ohne die Ausnahme meldet der Dienst
    an der Gabelung deutlich mehr - und stuft danach ein.
    """
    dicke = 6.0
    rgb, prob = _t_riss(dicke=dicke)
    _install(monkeypatch, prob)

    quelle = SourceImage(rgb=rgb, width=rgb.shape[1], height=rgb.shape[0])
    ergebnis = analyse_crack(quelle, ScaleInfo(source="none"))
    inst = ergebnis.instances[0]

    assert inst.branch_count >= 1
    assert inst.width_samples_excluded > 0, "An der Gabelung muss etwas wegfallen"

    # Was uebrig bleibt, liegt nahe an der gezeichneten Breite.
    assert abs(inst.width_max_px - dicke) < 2.5, (
        f"Hoechstwert {inst.width_max_px:.1f} statt {dicke} - "
        "die Gabelung ist offenbar mitgezaehlt"
    )

    # Gegenprobe: unter den ausgenommenen Stellen ist der Wert deutlich groesser.
    alle = np.concatenate([p.width_px for p in inst.paths])
    markiert = np.concatenate([p.at_junction for p in inst.paths])
    if markiert.any():
        assert alle[markiert].max() > inst.width_max_px


def test_unverzweigter_riss_nimmt_nichts_aus(monkeypatch):
    h, w = 300, 500
    canvas = np.zeros((h, w), dtype=np.uint8)
    linie = np.stack([np.full(300, 150.0), np.linspace(40, 460, 300)], axis=1).astype(
        np.float32
    )
    prob = np.clip(draw_crack(canvas, linie, 5.0, -255).astype(np.float32) / 255.0, 0, 1)
    rgb = cv2.cvtColor(
        draw_crack(concrete_background(h, w, 33), linie, 5.0), cv2.COLOR_GRAY2RGB
    )
    _install(monkeypatch, prob)

    inst = analyse_crack(
        SourceImage(rgb=rgb, width=w, height=h), ScaleInfo(source="none")
    ).instances[0]
    assert inst.branch_count == 0
    assert inst.width_samples_excluded == 0


# ------------------------------------------------- 2 · Bildrand -----------


def test_angeschnittener_riss_wird_gemeldet(monkeypatch):
    h, w = 300, 400
    canvas = np.zeros((h, w), dtype=np.uint8)
    # Von Rand zu Rand
    durch = np.stack([np.full(300, 150.0), np.linspace(0, w - 1, 300)], axis=1).astype(
        np.float32
    )
    prob = np.clip(draw_crack(canvas, durch, 6.0, -255).astype(np.float32) / 255.0, 0, 1)
    rgb = cv2.cvtColor(
        draw_crack(concrete_background(h, w, 35), durch, 6.0), cv2.COLOR_GRAY2RGB
    )
    _install(monkeypatch, prob)

    ergebnis = analyse_crack(
        SourceImage(rgb=rgb, width=w, height=h), ScaleInfo(source="none")
    )
    assert ergebnis.instances[0].touches_border is True
    assert any("Bildrand" in warnung for warnung in ergebnis.warnings)


def test_riss_mitten_im_bild_beruehrt_keinen_rand(monkeypatch):
    h, w = 300, 400
    canvas = np.zeros((h, w), dtype=np.uint8)
    innen = np.stack([np.full(300, 150.0), np.linspace(80, 320, 300)], axis=1).astype(
        np.float32
    )
    prob = np.clip(draw_crack(canvas, innen, 6.0, -255).astype(np.float32) / 255.0, 0, 1)
    rgb = cv2.cvtColor(
        draw_crack(concrete_background(h, w, 37), innen, 6.0), cv2.COLOR_GRAY2RGB
    )
    _install(monkeypatch, prob)

    ergebnis = analyse_crack(
        SourceImage(rgb=rgb, width=w, height=h), ScaleInfo(source="none")
    )
    assert ergebnis.instances[0].touches_border is False
    assert not any("Bildrand" in warnung for warnung in ergebnis.warnings)


# ------------------------------------------------- 3 · Abtastung ----------


def test_gleichmaessige_abtastung_haelt_den_abstand():
    t = np.linspace(0, 300, 301)
    punkte = np.stack([50 + 20 * np.sin(t / 40), t], axis=1).astype(np.float32)
    getastet = geometry.resample_uniform(punkte, 3.0)

    abstaende = np.hypot(
        np.diff(getastet[:, 0]), np.diff(getastet[:, 1])
    )
    assert abs(float(np.median(abstaende)) - 3.0) < 0.3
    assert float(abstaende.max()) < 4.0
    # Anfang und Ende bleiben stehen.
    assert np.allclose(getastet[0], punkte[0], atol=0.01)
    assert np.allclose(getastet[-1], punkte[-1], atol=0.01)


def test_gleichmaessige_abtastung_bleibt_auf_der_linie():
    """Anders als die Formvereinfachung darf sie keine Kurve abschneiden."""
    t = np.linspace(0, 2 * np.pi, 400)
    punkte = np.stack([60 + 40 * np.sin(t), 60 + 40 * np.cos(t)], axis=1).astype(
        np.float32
    )
    getastet = geometry.resample_uniform(punkte, 4.0)
    abstand = np.linalg.norm(
        getastet[:, None, :] - punkte[None, :, :], axis=2
    ).min(axis=1)
    assert float(abstand.max()) < 1.0


def test_gerades_stueck_bekommt_trotzdem_viele_stuetzstellen(monkeypatch):
    """Der eigentliche Fehler: RDP liess hier zwei Punkte stehen."""
    h, w = 200, 600
    canvas = np.zeros((h, w), dtype=np.uint8)
    gerade = np.stack([np.full(300, 100.0), np.linspace(40, 560, 300)], axis=1).astype(
        np.float32
    )
    prob = np.clip(
        draw_crack(canvas, gerade, 5.0, -255).astype(np.float32) / 255.0, 0, 1
    )
    rgb = cv2.cvtColor(
        draw_crack(concrete_background(h, w, 39), gerade, 5.0), cv2.COLOR_GRAY2RGB
    )
    _install(monkeypatch, prob)

    settings = get_settings()
    inst = analyse_crack(
        SourceImage(rgb=rgb, width=w, height=h), ScaleInfo(source="none")
    ).instances[0]

    pfad = max(inst.paths, key=lambda p: p.length_px)
    erwartet = pfad.length_px / settings.path_step_px
    assert pfad.point_count > erwartet * 0.8, (
        f"{pfad.point_count} Stuetzstellen auf {pfad.length_px:.0f} px - "
        "die Breite wird zu grob abgetastet"
    )


def test_abtastweite_ist_einstellbar(monkeypatch):
    t = np.linspace(0, 200, 201)
    punkte = np.stack([np.zeros_like(t), t], axis=1).astype(np.float32)
    eng = geometry.resample_uniform(punkte, 2.0)
    weit = geometry.resample_uniform(punkte, 10.0)
    assert eng.shape[0] > weit.shape[0] * 4


def test_sehr_kurzer_ast_behaelt_anfang_und_ende():
    punkte = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]], dtype=np.float32)
    getastet = geometry.resample_uniform(punkte, 10.0)
    assert getastet.shape[0] == 2
    assert np.allclose(getastet[0], punkte[0])
    assert np.allclose(getastet[-1], punkte[-1])
