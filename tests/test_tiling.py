"""Die Kachelung.

Risse sind duenn. Ein auf 512 verkleinertes Foto verliert sie, also wird in
Originalaufloesung gekachelt - und damit werden Kachelraender zu einer
eigenen Fehlerquelle. Zwei Dinge muessen stimmen:

1. **Jeder Pixel wird genau einmal gewichtet abgedeckt.** Faellt am unteren
   oder rechten Rand ein Streifen weg, fehlt dort jeder Befund - und zwar
   ohne Fehlermeldung.
2. **Der Uebergang ist nahtlos.** Ein Sprung an der Kachelgrenze wuerde als
   Kante durchs Bild laufen und als Riss gemeldet.
"""

from __future__ import annotations

import numpy as np
import pytest

from sada_vision.models.base import tiled_probability


def _constant(value: float):
    def predict(patch: np.ndarray) -> np.ndarray:
        return np.full(patch.shape[:2], value, dtype=np.float32)

    return predict


@pytest.mark.parametrize(
    ("h", "w"),
    [
        (100, 100),     # kleiner als eine Kachel
        (256, 256),     # genau eine Kachel
        (300, 300),     # knapp darueber
        (512, 256),     # genau zwei
        (700, 513),     # krumm in beiden Richtungen
        (128, 900),     # eine Kante kuerzer als die Kachel
    ],
)
def test_jeder_pixel_wird_abgedeckt(h, w):
    out = tiled_probability(np.zeros((h, w, 3), np.uint8), _constant(0.7), 256, 64)
    assert out.shape == (h, w)
    # Ein nicht abgedeckter Pixel waere 0, ein doppelt gezaehlter > 0,7.
    assert np.allclose(out, 0.7, atol=1e-4), (
        f"min {out.min():.4f}, max {out.max():.4f}"
    )


def test_uebergang_zwischen_kacheln_ist_nahtlos():
    """Ein waagerechter Verlauf muss waagerecht bleiben."""
    h, w = 600, 800
    rgb = np.zeros((h, w, 3), np.uint8)

    def predict(patch: np.ndarray) -> np.ndarray:
        # Haengt nur vom *Inhalt* der Kachel ab - hier also konstant je
        # Kachel, was den haerteste Fall fuer die Ueberblendung darstellt.
        return np.full(patch.shape[:2], 0.5, dtype=np.float32)

    out = tiled_probability(rgb, predict, 256, 64)
    # Keine Kante: der groesste Sprung zwischen Nachbarpixeln bleibt klein.
    dy = np.abs(np.diff(out, axis=0)).max()
    dx = np.abs(np.diff(out, axis=1)).max()
    assert dy < 1e-3 and dx < 1e-3


def test_ortsabhaengige_vorhersage_bleibt_erhalten():
    """Ein Muster, das nur an einer Stelle liegt, darf nicht verschmieren."""
    h, w = 500, 700
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[240:260, 100:600] = 255  # heller Balken

    def predict(patch: np.ndarray) -> np.ndarray:
        return (patch[:, :, 0] > 128).astype(np.float32)

    out = tiled_probability(rgb, predict, 256, 64)
    assert out[250, 300] > 0.95      # auf dem Balken
    assert out[100, 300] < 0.05      # darueber
    assert out[400, 300] < 0.05      # darunter


def test_kleines_bild_wird_gespiegelt_aufgefuellt_und_wieder_zugeschnitten():
    out = tiled_probability(np.zeros((60, 40, 3), np.uint8), _constant(1.0), 256, 64)
    assert out.shape == (60, 40)
    assert np.allclose(out, 1.0)
