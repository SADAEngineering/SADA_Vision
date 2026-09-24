"""Massstab aus Tiefe und Brennweite - der Weg der AR-App.

Bei einer Lochkamera bildet sich eine Strecke ``S`` im Abstand ``Z`` auf
``s = f * S / Z`` Pixel ab. Umgestellt:

    mm_per_px = Z / f

``Z`` liefert der LiDAR-Scanner, ``f`` die Intrinsik der Kamera (in Pixeln,
bezogen auf *dieselbe* Aufloesung wie das gesendete Bild - eine auf 1280
verkleinerte Aufnahme braucht eine auf 1280 umgerechnete Brennweite).

Steht die Flaeche schraeg zur Kamera, ist eine quer dazu gemessene Strecke im
Bild verkuerzt. Ist die Flaechennormale bekannt, korrigiert ``1 / cos(theta)``
das. Ab etwa 60 Grad wird die Korrektur unzuverlaessig - dann sinkt die
gemeldete Verlaesslichkeit, und ab 75 Grad gilt der Massstab als unbekannt.
"""

from __future__ import annotations

import math

from ..domain import ScaleInfo

_MAX_TILT_DEG = 75.0
_WARN_TILT_DEG = 45.0


def lidar_scale(
    depth_mm: float,
    focal_px: float,
    tilt_deg: float = -1.0,
) -> ScaleInfo:
    """``depth_mm`` Abstand zur Flaeche, ``focal_px`` Brennweite in Pixeln."""
    if depth_mm is None or depth_mm <= 0:
        return ScaleInfo(source="none", note="depth_mm muss groesser als 0 sein.")
    if focal_px is None or focal_px <= 0:
        return ScaleInfo(source="none", note="focal_px muss groesser als 0 sein.")

    mm_per_px = float(depth_mm) / float(focal_px)
    confidence = 0.8
    note = f"Aus Tiefe {depth_mm:.0f} mm und Brennweite {focal_px:.0f} px."

    if tilt_deg is not None and tilt_deg >= 0:
        if tilt_deg >= _MAX_TILT_DEG:
            return ScaleInfo(
                source="none",
                note=(
                    f"Flaeche steht {tilt_deg:.0f} Grad schraeg - zu flach fuer "
                    "eine belastbare Umrechnung. Frontaler aufnehmen."
                ),
            )
        cos_t = math.cos(math.radians(tilt_deg))
        mm_per_px /= max(cos_t, 1e-3)
        if tilt_deg >= _WARN_TILT_DEG:
            confidence = 0.4
            note += f" Schraeglage {tilt_deg:.0f} Grad angerechnet - grenzwertig."
        else:
            confidence = 0.7
            note += f" Schraeglage {tilt_deg:.0f} Grad angerechnet."
    else:
        note += " Schraeglage unbekannt, frontale Flaeche angenommen."
        confidence = 0.6

    return ScaleInfo(
        source="lidar",
        mm_per_px=mm_per_px,
        confidence=confidence,
        note=note,
    )
