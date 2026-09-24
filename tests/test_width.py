"""Die Messung gegen bekannte Breiten.

Das ist der Test, der zaehlt. Alles andere am Dienst ist Infrastruktur; der
Zweck ist eine Zahl in Millimetern, und ob die stimmt, zeigt sich nur an
einem Riss, dessen Breite man vorher kennt.
"""

from __future__ import annotations

import numpy as np
import pytest

from sada_vision.pipeline import width as width_mod
from tests.synthetic import draw_crack


def _band_mask(h: int, w: int, thickness: int) -> np.ndarray:
    """Waagerechtes Band exakter Dicke."""
    m = np.zeros((h, w), dtype=bool)
    y0 = h // 2 - thickness // 2
    m[y0 : y0 + thickness, :] = True
    return m


@pytest.mark.parametrize("thickness", [3, 5, 9, 15])
def test_distanztransformation_trifft_die_bandbreite(thickness):
    m = _band_mask(80, 160, thickness)
    dist = width_mod.distance_transform(m)
    ys = np.full(100, float(80 // 2 - thickness // 2 + thickness // 2))
    xs = np.linspace(20, 140, 100)
    points = np.stack([ys, xs], axis=1)
    got = width_mod.widths_from_distance(points, dist)
    # Die Distanztransformation misst vom Pixelmittelpunkt: bei ungerader
    # Dicke t ist der Mittelwert t, bei gerader liegt er dazwischen.
    assert abs(float(np.median(got)) - thickness) <= 1.0


@pytest.mark.parametrize("thickness", [2.0, 3.0, 4.5, 7.0, 11.0])
def test_lotrechte_messung_trifft_subpixelgenau(thickness):
    """Auf einem sauberen Wahrscheinlichkeitsbild unter 0,4 px Abweichung."""
    h, w = 80, 200
    yy = np.arange(h)[:, None].repeat(w, axis=1).astype(np.float64)
    centre = h / 2.0
    # Weiche Kante: 1 in der Mitte, 0 aussen, linearer Uebergang ueber 1 px.
    d = np.abs(yy - centre)
    prob = np.clip((thickness / 2.0 + 0.5 - d), 0.0, 1.0).astype(np.float32)

    mask = prob >= 0.5
    dist = width_mod.distance_transform(mask)
    points = np.stack(
        [np.full(60, centre), np.linspace(20, 180, 60)], axis=1
    )
    got = width_mod.widths_perpendicular(points, prob, dist, threshold=0.5)
    assert abs(float(np.median(got)) - thickness) < 0.4


def test_lotrechte_messung_ist_genauer_als_die_distanztransformation():
    """Der Grund, warum 'perpendicular' die Vorgabe ist."""
    h, w = 80, 200
    thickness = 3.4  # zwischen zwei ganzen Pixeln
    yy = np.arange(h)[:, None].repeat(w, axis=1).astype(np.float64)
    centre = h / 2.0
    d = np.abs(yy - centre)
    prob = np.clip((thickness / 2.0 + 0.5 - d), 0.0, 1.0).astype(np.float32)
    mask = prob >= 0.5
    dist = width_mod.distance_transform(mask)
    points = np.stack([np.full(60, centre), np.linspace(20, 180, 60)], axis=1)

    perp = float(np.median(width_mod.widths_perpendicular(points, prob, dist, 0.5)))
    dt = float(np.median(width_mod.widths_from_distance(points, dist)))
    assert abs(perp - thickness) < abs(dt - thickness)


def test_schraege_linie_wird_nicht_ueber_die_diagonale_gemessen():
    """Ohne Tangentenglaettung misst ein 45-Grad-Riss um Faktor 1,41 zu breit."""
    h = w = 240
    thickness = 6.0
    canvas = np.full((h, w), 200, dtype=np.uint8)
    t = np.linspace(-80, 80, 400)
    points = np.stack([h / 2 - t, w / 2 + t], axis=1).astype(np.float32)
    img = draw_crack(canvas, points, thickness, darkness=150)

    prob = (200.0 - img.astype(np.float32)) / 150.0
    prob = np.clip(prob, 0, 1).astype(np.float32)
    mask = prob >= 0.5
    dist = width_mod.distance_transform(mask)

    sample = points[100:300:4]
    got = width_mod.widths_perpendicular(sample, prob, dist, 0.5)
    assert abs(float(np.median(got)) - thickness) < 0.9


def test_tangenten_stehen_senkrecht_auf_dem_lot():
    points = np.stack([np.arange(50, dtype=np.float64), np.arange(50) * 2.0], axis=1)
    t = width_mod.tangents(points)
    assert np.allclose(np.hypot(t[:, 0], t[:, 1]), 1.0, atol=1e-6)
    normal = np.stack([t[:, 1], -t[:, 0]], axis=1)
    assert np.allclose(np.sum(t * normal, axis=1), 0.0, atol=1e-9)


def test_millimeter_bleiben_minus_eins_ohne_massstab():
    px = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    mm = width_mod.to_millimetres(px, np.array([-1.0, -1.0, -1.0]))
    assert np.all(mm == -1.0)


def test_millimeter_rechnen_punktweise():
    px = np.array([2.0, 4.0], dtype=np.float32)
    mm = width_mod.to_millimetres(px, np.array([0.5, 0.25]))
    assert mm[0] == pytest.approx(1.0)
    assert mm[1] == pytest.approx(1.0)


def test_laenge_in_millimetern_nutzt_den_ortsabhaengigen_massstab():
    points = np.array([[0.0, 0.0], [0.0, 10.0], [0.0, 20.0]])
    # Erste Haelfte 1 mm/px, zweite 2 mm/px -> 10 * 1,5 + 10 * 1,5 ... nein:
    # je Abschnitt das Mittel der Endpunkte.
    mm_per_px = np.array([1.0, 1.0, 3.0])
    got = width_mod.length_mm(points, mm_per_px)
    assert got == pytest.approx(10 * 1.0 + 10 * 2.0)


def test_laenge_ohne_massstab_ist_minus_eins():
    points = np.array([[0.0, 0.0], [0.0, 10.0]])
    assert width_mod.length_mm(points, np.array([-1.0, -1.0])) == -1.0


def test_kein_rand_gefunden_meldet_den_radius_statt_null():
    """Eine Flaeche ohne Rand darf nicht als Breite 0 durchgehen."""
    prob = np.ones((60, 60), dtype=np.float32)
    mask = prob >= 0.5
    dist = width_mod.distance_transform(mask)
    points = np.array([[30.0, 30.0]])
    got = width_mod.widths_perpendicular(points, prob, dist, 0.5)
    assert got[0] > 0.0
