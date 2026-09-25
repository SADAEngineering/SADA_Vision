"""Skelett, Aeste, Vereinfachung, Richtung."""

from __future__ import annotations

import numpy as np
import pytest

from sada_vision.pipeline import geometry


def _line_mask(h=100, w=200, thickness=5) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    y0 = h // 2
    m[y0 - thickness // 2 : y0 + thickness // 2 + 1, 10 : w - 10] = True
    return m


def test_gerade_linie_ergibt_einen_ast():
    skelett = geometry.skeleton_branches(_line_mask())
    branches, nodes = skelett.branches, skelett.branch_count
    assert nodes == 0
    # Enden zaehlen als Knoten, der Ast dazwischen ist einer.
    assert len(branches) == 1
    assert branches[0].shape[0] > 100


def test_t_form_ergibt_drei_aeste_und_einen_knoten():
    m = np.zeros((120, 120), dtype=bool)
    m[58:62, 10:110] = True   # Stamm waagerecht
    m[60:110, 58:62] = True   # Ast nach unten
    skelett = geometry.skeleton_branches(m)
    branches, nodes = skelett.branches, skelett.branch_count
    assert nodes >= 1
    assert len(branches) == 3


def test_leere_maske_liefert_nichts():
    skelett = geometry.skeleton_branches(np.zeros((50, 50), dtype=bool))
    branches, nodes = skelett.branches, skelett.branch_count
    assert branches == []
    assert nodes == 0


def test_ring_wird_geschlossen_zurueckgegeben():
    m = np.zeros((120, 120), dtype=bool)
    import cv2

    cv2.circle(m.astype(np.uint8), (60, 60), 40, 1, 4)
    ring = np.zeros((120, 120), dtype=np.uint8)
    cv2.circle(ring, (60, 60), 40, 1, 4)
    skelett = geometry.skeleton_branches(ring.astype(bool))
    branches, nodes = skelett.branches, skelett.branch_count
    assert len(branches) >= 1
    longest = max(branches, key=lambda b: b.shape[0])
    assert longest.shape[0] > 100
    # Ein Ring hat keine Enden: Anfang und Ende fallen zusammen.
    if nodes == 0:
        assert np.allclose(longest[0], longest[-1])


def test_rdp_behaelt_die_form_und_kuerzt_die_liste():
    t = np.linspace(0, 100, 400)
    points = np.stack([50 + 0.0 * t, t], axis=1).astype(np.float32)
    simplified = geometry.simplify_rdp(points, epsilon=1.0)
    # Eine Gerade braucht zwei Punkte.
    assert simplified.shape[0] == 2
    assert np.allclose(simplified[0], points[0])
    assert np.allclose(simplified[-1], points[-1])


def test_rdp_behaelt_den_knick():
    points = np.array(
        [[0, 0], [0, 10], [0, 20], [10, 30], [20, 40]], dtype=np.float32
    )
    simplified = geometry.simplify_rdp(points, epsilon=0.5)
    assert simplified.shape[0] >= 3


@pytest.mark.parametrize(
    ("angle", "expected"),
    [(0.0, 0.0), (45.0, 45.0), (90.0, 90.0), (-30.0, -30.0)],
)
def test_hauptrichtung_trifft_den_winkel(angle, expected):
    t = np.linspace(-50, 50, 200)
    a = np.radians(angle)
    points = np.stack([-np.sin(a) * t, np.cos(a) * t], axis=1).astype(np.float32)
    got = geometry.principal_orientation(points)
    # 90 und -90 sind dieselbe Richtung.
    diff = min(abs(got - expected), abs(abs(got - expected) - 180.0))
    assert diff < 2.0


def test_gewundenheit_einer_geraden_ist_eins():
    points = np.stack([np.zeros(50), np.arange(50)], axis=1).astype(np.float32)
    assert geometry.tortuosity(points) == pytest.approx(1.0, abs=1e-6)


def test_gewundenheit_eines_zickzacks_ist_groesser():
    x = np.arange(0, 100, 1.0)
    y = 10 * np.sin(x / 2.0)
    points = np.stack([y, x], axis=1).astype(np.float32)
    assert geometry.tortuosity(points) > 1.5


def test_komponenten_kommen_nach_flaeche_sortiert():
    m = np.zeros((100, 200), dtype=bool)
    m[10:20, 10:30] = True    # 200 px
    m[50:70, 50:100] = True   # 1000 px
    comps = geometry.components(m, min_area=10)
    assert len(comps) == 2
    assert comps[0][2] > comps[1][2]


def test_kleine_streusel_fallen_raus():
    m = np.zeros((100, 100), dtype=bool)
    m[10:12, 10:12] = True    # 4 px
    assert geometry.components(m, min_area=40) == []
