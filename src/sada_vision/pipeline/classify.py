"""Einordnung des Befundes.

Was hier passiert, ist **abgeleitete Geometrie, kein gelerntes Urteil**. Aus
Skelett, Breite und Flaeche folgt das Muster (durchgehend, verzweigt,
netzartig), die Lage (waagerecht, senkrecht, schraeg) und ein Band der Breite.
Die Ursache eines Risses - Schwinden, Setzung, Bewehrungskorrosion, Zwang -
folgt daraus **nicht**; dafuer braucht es das Bauteil, die Belastung und einen
Menschen. TraceForm bekommt den Befund, nicht die Diagnose.

Ein gelernter Klassenkopf (Risstyp aus dem Bild) kommt, sobald eigene,
befundete Fotos vorliegen - siehe docs/Modell_und_Training.md.
"""

from __future__ import annotations

import numpy as np

from ..config import SEVERITY_BANDS_MM

# Ab dieser Verzweigungszahl gilt ein Riss nicht mehr als durchgehend.
_BRANCHED_MIN_NODES = 1
# Netzartig (Krakelee): viele Knoten auf wenig Laenge, Flaeche gut gefuellt.
_MAP_MIN_NODES = 6
_MAP_MIN_NODE_DENSITY = 0.004   # Knoten je Pixel Skelettlaenge
_MAP_MIN_FILL = 0.06            # Skelettlaenge je Pixel Kastenflaeche


def pattern_of(branch_count: int, total_length_px: float, bbox_area_px: float) -> str:
    """durchgehend / verzweigt / netzartig."""
    if total_length_px <= 0:
        return "unknown"
    density = branch_count / max(total_length_px, 1.0)
    fill = total_length_px / max(bbox_area_px, 1.0)
    if (
        branch_count >= _MAP_MIN_NODES
        and density >= _MAP_MIN_NODE_DENSITY
        and fill >= _MAP_MIN_FILL
    ):
        return "map"
    if branch_count >= _BRANCHED_MIN_NODES:
        return "branched"
    return "linear"


def orientation_class_of(angle_deg: float) -> str:
    """Lage im Bild. Bezug ist die Bildwaagerechte.

    Wer die Lage am *Bauteil* braucht - laengs oder quer zur Achse -, muss
    die Drehung des Bauteils im Bild kennen. Die kennt der Dienst nicht; sie
    kommt aus TraceForm oder aus der AR-App und wird dort angerechnet.
    """
    a = abs(((angle_deg + 90.0) % 180.0) - 90.0)
    if a <= 22.5:
        return "horizontal"
    if a >= 67.5:
        return "vertical"
    return "diagonal"


def severity_of(width_mm: float) -> str:
    """Band der groessten gemessenen Breite."""
    if width_mm is None or width_mm < 0:
        return "unknown"
    for name, upper in SEVERITY_BANDS_MM:
        if width_mm < upper:
            return name
    return SEVERITY_BANDS_MM[-1][0]


def summarise_widths(values: np.ndarray) -> tuple[float, float, float]:
    """(max, mittel, p95) - robust gegen leere Eingaben."""
    valid = values[values >= 0]
    if valid.size == 0:
        return -1.0, -1.0, -1.0
    return (
        float(np.max(valid)),
        float(np.mean(valid)),
        float(np.percentile(valid, 95)),
    )
