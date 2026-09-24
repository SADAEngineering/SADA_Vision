"""Interne Typen der Pipeline.

Bewusst getrennt von den DTOs in ``api/schemas.py``: hier darf sich alles
aendern, dort gilt der Vertrag. Koordinaten sind durchgehend ``(y, x)`` in
Pixeln des *Originalbildes* - nicht des verkleinerten Arbeitsbildes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class SourceImage:
    """Das dekodierte Bild samt dem, was zum Zurueckrechnen noetig ist."""

    rgb: np.ndarray            # (H, W, 3) uint8, EXIF-gedreht
    width: int
    height: int
    exif_rotated: bool = False

    @property
    def shape_hw(self) -> tuple[int, int]:
        return self.height, self.width


@dataclass(slots=True)
class ScaleInfo:
    """Umrechnung Pixel -> Millimeter.

    ``mm_per_px`` ist ein Mittelwert fuer das ganze Bild. Steht eine
    Homographie zur Verfuegung (Marker), ist der Massstab ortsabhaengig und
    ``local_mm_per_px`` liefert ihn punktweise.
    """

    source: str                       # "none" | "explicit" | "aruco" | "lidar"
    mm_per_px: float = -1.0           # -1 = unbekannt
    confidence: float = 0.0
    note: str = ""
    homography: np.ndarray | None = None   # Bild -> Ebene, 3x3
    # Die vier Ecken des erkannten Markers in (y, x), Originalbild. Wird
    # gebraucht, um ihn vor der Segmentierung auszublenden: seine harten
    # Schwarz-Weiss-Kanten sehen fuer jeden Kantenfilter wie Risse aus.
    marker_quad: np.ndarray | None = None
    _plane_mm_per_unit: float = 1.0

    @property
    def known(self) -> bool:
        return self.mm_per_px > 0.0

    def local_mm_per_px(self, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
        """Massstab je Punkt. Ohne Homographie konstant."""
        n = len(ys)
        if not self.known:
            return np.full(n, -1.0, dtype=np.float64)
        if self.homography is None:
            return np.full(n, self.mm_per_px, dtype=np.float64)
        return _homography_scale(self.homography, ys, xs) * self._plane_mm_per_unit


def _homography_scale(h: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Lokaler Flaechenmassstab einer Homographie, als Laenge je Pixel.

    Die Jacobi-Determinante der projektiven Abbildung an der Stelle (x, y);
    ihre Wurzel ist der isotrope Laengenfaktor.
    """
    x = xs.astype(np.float64)
    y = ys.astype(np.float64)
    a, b, c = h[0]
    d, e, f = h[1]
    g, i, j = h[2]
    w = g * x + i * y + j
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    # Jacobi der Abbildung (x,y) -> ((ax+by+c)/w, (dx+ey+f)/w)
    u = a * x + b * y + c
    v = d * x + e * y + f
    du_dx = (a * w - u * g) / w**2
    du_dy = (b * w - u * i) / w**2
    dv_dx = (d * w - v * g) / w**2
    dv_dy = (e * w - v * i) / w**2
    det = np.abs(du_dx * dv_dy - du_dy * dv_dx)
    return np.sqrt(np.maximum(det, 1e-18))


@dataclass(slots=True)
class Polyline:
    """Ein Ast des Risses: geordneter Verlauf mit Breite je Punkt."""

    points_yx: np.ndarray         # (N, 2) float32, Originalbild-Pixel
    width_px: np.ndarray          # (N,) float32, Breite quer zum Verlauf
    width_mm: np.ndarray          # (N,) float32, -1 wo unbekannt
    length_px: float = 0.0
    length_mm: float = -1.0
    is_loop: bool = False

    @property
    def point_count(self) -> int:
        return int(self.points_yx.shape[0])


@dataclass(slots=True)
class CrackInstance:
    """Ein zusammenhaengender Riss."""

    instance_id: int
    label: str = "crack"
    score: float = 0.0
    bbox_yxyx: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    paths: list[Polyline] = field(default_factory=list)

    # Aus der Geometrie abgeleitet
    pattern: str = "unknown"          # "linear" | "branched" | "map"
    orientation_deg: float = 0.0      # 0 = waagerecht, gegen den Uhrzeigersinn
    orientation_class: str = "unknown"
    tortuosity: float = 1.0
    branch_count: int = 0

    width_max_px: float = 0.0
    width_mean_px: float = 0.0
    width_p95_px: float = 0.0
    width_max_mm: float = -1.0
    width_mean_mm: float = -1.0
    width_p95_mm: float = -1.0

    length_px: float = 0.0
    length_mm: float = -1.0
    area_px: float = 0.0

    severity: str = "unknown"

    @property
    def total_points(self) -> int:
        return sum(p.point_count for p in self.paths)


@dataclass(slots=True)
class Analysis:
    """Ergebnis eines Laufs."""

    task: str
    image_width: int
    image_height: int
    scale: ScaleInfo
    instances: list[CrackInstance] = field(default_factory=list)
    model_name: str = ""
    model_kind: str = ""              # "onnx" | "classic"
    model_trained: bool = False
    warnings: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
