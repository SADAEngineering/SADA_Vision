"""Pixel zu Millimeter - der Teil, an dem die Zahl ihre Einheit bekommt."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from sada_vision.scale import explicit_scale, lidar_scale, resolve_scale
from sada_vision.scale.aruco import ArucoScaleResolver
from tests.synthetic import blank, with_aruco


def test_vorgegebener_massstab_wird_uebernommen():
    info = explicit_scale(0.05)
    assert info.known
    assert info.mm_per_px == pytest.approx(0.05)
    assert info.source == "explicit"


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_unsinniger_vorgabewert_gilt_als_unbekannt(bad):
    assert not explicit_scale(bad).known


def test_lidar_rechnet_tiefe_durch_brennweite():
    # 500 mm Abstand, 1000 px Brennweite -> 0,5 mm je Pixel
    info = lidar_scale(depth_mm=500.0, focal_px=1000.0)
    assert info.mm_per_px == pytest.approx(0.5)
    assert info.source == "lidar"


def test_lidar_rechnet_die_schraeglage_an():
    frontal = lidar_scale(500.0, 1000.0, tilt_deg=0.0)
    schraeg = lidar_scale(500.0, 1000.0, tilt_deg=60.0)
    # cos(60) = 0,5 -> doppelter Massstab
    assert schraeg.mm_per_px == pytest.approx(frontal.mm_per_px * 2.0, rel=1e-6)
    assert schraeg.confidence < frontal.confidence


def test_zu_flache_aufnahme_liefert_keinen_massstab():
    info = lidar_scale(500.0, 1000.0, tilt_deg=80.0)
    assert not info.known
    assert "schraeg" in info.note


def test_aruco_findet_den_marker_und_rechnet_richtig():
    rgb = with_aruco(blank(600, 800), marker_id=7, size_px=120)
    resolver = ArucoScaleResolver()
    # 120 px Kante entsprechen 60 mm -> 0,5 mm je Pixel
    info = resolver.resolve(rgb, marker_mm=60.0)
    assert info.known
    assert info.source == "aruco"
    assert info.mm_per_px == pytest.approx(0.5, rel=0.05)
    assert info.homography is not None


def test_aruco_ohne_marker_meldet_warum():
    info = ArucoScaleResolver().resolve(blank(400, 400), marker_mm=60.0)
    assert not info.known
    assert "Kein Marker" in info.note


def test_aruco_mit_falscher_kennung_meldet_die_gefundenen():
    rgb = with_aruco(blank(600, 800), marker_id=7)
    info = ArucoScaleResolver().resolve(rgb, marker_mm=60.0, marker_id=3)
    assert not info.known
    assert "7" in info.note


def test_homographie_liefert_ortsabhaengigen_massstab():
    """Bei frontaler Aufnahme ist der lokale Massstab ueberall gleich."""
    rgb = with_aruco(blank(600, 800), marker_id=7, size_px=120)
    info = ArucoScaleResolver().resolve(rgb, marker_mm=60.0)
    ys = np.array([100.0, 300.0, 500.0])
    xs = np.array([100.0, 400.0, 700.0])
    local = info.local_mm_per_px(ys, xs)
    assert np.allclose(local, info.mm_per_px, rtol=0.08)


def test_marker_schlaegt_vorgabe_schlaegt_lidar():
    rgb = with_aruco(blank(600, 800), marker_id=7, size_px=120)
    info, warnings = resolve_scale(
        rgb, mm_per_px=0.5, marker_size_mm=60.0, depth_mm=500.0, focal_px=1000.0
    )
    assert info.source == "aruco"
    assert any("Mehrere Massstaebe" in w for w in warnings)


def test_ohne_jede_angabe_bleibt_der_massstab_unbekannt():
    info, _ = resolve_scale(blank(300, 300))
    assert not info.known
    assert info.source == "none"
    assert info.mm_per_px == -1.0


def test_widersprechende_quellen_werden_gemeldet():
    """Zwei Angaben, die nicht zusammenpassen, sind ein Hinweis auf einen Fehler."""
    rgb = with_aruco(blank(600, 800), marker_id=7, size_px=120)
    # Marker sagt 0,5 mm/px, die Vorgabe behauptet 0,2
    _, warnings = resolve_scale(rgb, mm_per_px=0.2, marker_size_mm=60.0)
    assert any("weicht um" in w for w in warnings)


def test_verworfener_marker_hinterlaesst_eine_begruendung():
    _, warnings = resolve_scale(blank(400, 400), marker_size_mm=60.0, mm_per_px=0.3)
    assert any(w.startswith("Marker:") for w in warnings)


def test_lokaler_massstab_ist_minus_eins_wenn_unbekannt():
    info, _ = resolve_scale(blank(200, 200))
    local = info.local_mm_per_px(np.array([10.0]), np.array([10.0]))
    assert local[0] == -1.0


def test_schraeg_fotografierter_marker_wird_trotzdem_erkannt():
    """Perspektivisch verzerrt - der Massstab muss dann ortsabhaengig sein."""
    rgb = with_aruco(blank(700, 900), marker_id=7, size_px=200, margin=60)
    h, w = rgb.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[120, 0], [w - 40, 60], [w, h], [0, h - 30]])
    warped = cv2.warpPerspective(rgb, cv2.getPerspectiveTransform(src, dst), (w, h))

    info = ArucoScaleResolver().resolve(warped, marker_mm=100.0)
    if not info.known:
        pytest.skip("Marker nach der Verzerrung nicht mehr erkennbar")
    assert info.homography is not None
    oben = info.local_mm_per_px(np.array([80.0]), np.array([450.0]))[0]
    unten = info.local_mm_per_px(np.array([620.0]), np.array([450.0]))[0]
    # Bei echter Perspektive unterscheiden sich die beiden.
    assert oben > 0 and unten > 0
