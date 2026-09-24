"""Welcher Massstab gilt - und was, wenn mehrere angeboten werden."""

from __future__ import annotations

import numpy as np

from ..domain import ScaleInfo
from ..logging_setup import get_logger
from .explicit import explicit_scale
from .lidar import lidar_scale

log = get_logger(__name__)

# Reihenfolge = Rangfolge. Der Marker gewinnt, weil er als einziger im Bild
# selbst nachweisbar ist; ein vorgegebener Wert gewinnt gegen LiDAR, weil ihn
# jemand bewusst gesetzt hat.
_ORDER = ("aruco", "explicit", "lidar")


def resolve_scale(
    rgb: np.ndarray,
    *,
    mm_per_px: float = -1.0,
    marker_size_mm: float = -1.0,
    marker_id: int = -1,
    marker_dictionary: str = "",
    depth_mm: float = -1.0,
    focal_px: float = -1.0,
    tilt_deg: float = -1.0,
) -> tuple[ScaleInfo, list[str]]:
    """Probiert alle angebotenen Wege und nimmt den ranghoechsten.

    Gibt zusaetzlich die Hinweise der *verworfenen* Wege zurueck - wer einen
    Marker ins Bild gelegt hat und trotzdem keine Millimeter bekommt, soll
    erfahren warum.
    """
    found: dict[str, ScaleInfo] = {}
    warnings: list[str] = []

    if marker_size_mm and marker_size_mm > 0:
        try:
            from .aruco import DEFAULT_DICTIONARY, ArucoScaleResolver

            resolver = ArucoScaleResolver(marker_dictionary or DEFAULT_DICTIONARY)
            info = resolver.resolve(rgb, marker_size_mm, marker_id)
        except Exception as exc:  # pragma: no cover - cv2.aruco fehlt nur bei Fehlbau
            info = ScaleInfo(source="none", note=f"Markererkennung nicht moeglich: {exc}")
            log.warning("massstab.aruco.fehler", error=str(exc))
        if info.known:
            found["aruco"] = info
        else:
            warnings.append(f"Marker: {info.note}")

    if mm_per_px and mm_per_px > 0:
        info = explicit_scale(mm_per_px)
        if info.known:
            found["explicit"] = info
        else:
            warnings.append(f"mm_per_px: {info.note}")

    if depth_mm and depth_mm > 0 and focal_px and focal_px > 0:
        info = lidar_scale(depth_mm, focal_px, tilt_deg)
        if info.known:
            found["lidar"] = info
        else:
            warnings.append(f"LiDAR: {info.note}")

    for source in _ORDER:
        if source in found:
            chosen = found[source]
            others = [s for s in _ORDER if s in found and s != source]
            if others:
                warnings.append(
                    f"Mehrere Massstaebe angeboten ({', '.join(found)}); "
                    f"verwendet wird {source}."
                )
                _warn_on_disagreement(chosen, found, others, warnings)
            return chosen, warnings

    return (
        ScaleInfo(
            source="none",
            note=(
                "Kein Massstab. Breiten und Laengen werden nur in Pixeln "
                "gemeldet (mm-Felder = -1)."
            ),
        ),
        warnings,
    )


def _warn_on_disagreement(
    chosen: ScaleInfo,
    found: dict[str, ScaleInfo],
    others: list[str],
    warnings: list[str],
) -> None:
    """Weichen zwei Quellen stark ab, stimmt eine Annahme nicht."""
    for source in others:
        other = found[source]
        ratio = chosen.mm_per_px / max(other.mm_per_px, 1e-9)
        if ratio > 1.15 or ratio < 0.87:
            warnings.append(
                f"Massstab aus {source} weicht um {abs(1 - ratio) * 100:.0f} Prozent ab "
                f"({other.mm_per_px:.4f} statt {chosen.mm_per_px:.4f} mm/px) - "
                "eine der beiden Annahmen ist falsch."
            )
