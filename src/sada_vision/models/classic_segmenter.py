"""Notbehelf ohne gelernte Gewichte: Ridge-Filter auf dem Grauwert.

Risse sind dunkle, duenne, langgestreckte Taeler im Grauwertgebirge - genau
das, wofuer die Hesse-Matrix-Filter von Frangi und Sato gebaut sind. Das
Ergebnis ist deutlich schlechter als ein trainiertes Netz (jede Fuge, jeder
Schattenriss, jedes Kabel wird mitgenommen), aber es macht den Dienst vom
ersten Tag an vollstaendig: gleiche Schnittstelle, gleiche Geometrie, gleiche
Breitenmessung.

Jede Antwort, die hierher kommt, traegt ``model.trained = false``. Wer das
ignoriert, misst Bildrauschen.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.filters import sato

from .base import SegmenterInfo, tiled_probability

_INFO = SegmenterInfo(
    name="classic-ridge",
    kind="classic",
    trained=False,
    note=(
        "Kein trainiertes Modell geladen. Ridge-Filter als Notbehelf - "
        "Befunde sind nicht belastbar."
    ),
)


class ClassicRidgeSegmenter:
    """Frangi/Sato-Ridge mit lokaler Normierung."""

    info = _INFO

    def __init__(self, sigmas: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 4.0)) -> None:
        self._sigmas = sigmas

    def probability(self, rgb: np.ndarray) -> np.ndarray:
        return tiled_probability(rgb, self._predict_tile, tile=512, overlap=64)

    def _predict_tile(self, rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # Beleuchtung herausrechnen: Betonflaechen sind selten gleichmaessig
        # ausgeleuchtet, und ein globaler Schwellwert wuerde die dunkle
        # Bildhaelfte komplett als Riss lesen.
        background = cv2.GaussianBlur(gray, (0, 0), sigmaX=25)
        flat = cv2.subtract(gray.astype(np.int16), background.astype(np.int16))
        flat = np.clip(flat + 128, 0, 255).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        flat = clahe.apply(flat)

        # Sato reagiert auf helle Grate, Risse sind dunkel -> invertieren.
        inv = (255 - flat).astype(np.float32) / 255.0
        ridge = sato(inv, sigmas=self._sigmas, black_ridges=False)

        if ridge.max() <= 1e-9:
            return np.zeros(gray.shape, dtype=np.float32)

        # Robuste Normierung: das 99,5-Perzentil statt des Maximums, sonst
        # bestimmt ein einzelner Ausreisser die ganze Skala.
        hi = float(np.percentile(ridge, 99.5))
        lo = float(np.percentile(ridge, 60.0))
        if hi - lo < 1e-9:
            return np.zeros(gray.shape, dtype=np.float32)
        prob = np.clip((ridge - lo) / (hi - lo), 0.0, 1.0)
        return prob.astype(np.float32)
