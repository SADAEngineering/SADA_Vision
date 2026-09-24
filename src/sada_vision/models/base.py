"""Was ein Segmentierer koennen muss - und die gemeinsame Kachelung."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class SegmenterInfo:
    name: str
    kind: str          # "onnx" | "classic"
    trained: bool      # False = Notbehelf ohne gelernte Gewichte
    note: str = ""


class Segmenter(Protocol):
    """Liefert je Pixel die Wahrscheinlichkeit, dass dort ein Riss ist."""

    info: SegmenterInfo

    def probability(self, rgb: np.ndarray) -> np.ndarray:
        """(H, W, 3) uint8 -> (H, W) float32 in [0, 1]."""
        ...


def _blend_window(h: int, w: int) -> np.ndarray:
    """Hann-Fenster, damit Kachelraender nicht als Kanten stehen bleiben."""
    wy = np.hanning(h + 2)[1:-1]
    wx = np.hanning(w + 2)[1:-1]
    win = np.outer(wy, wx).astype(np.float32)
    return np.maximum(win, 1e-3)


def tiled_probability(
    rgb: np.ndarray,
    predict: callable,
    tile: int,
    overlap: int,
) -> np.ndarray:
    """Kachelweise Inferenz mit ueberlappender, gewichteter Ueberblendung.

    ``predict`` bekommt eine Kachel (t, t, 3) uint8 und gibt (t, t) float32.
    Risse sind duenn; ein auf 512 verkleinertes Foto verliert sie. Deshalb
    wird in Originalaufloesung gekachelt statt das Bild zu skalieren.
    """
    h, w = rgb.shape[:2]
    if h <= tile and w <= tile:
        pad_h, pad_w = tile - h, tile - w
        if pad_h > 0 or pad_w > 0:
            padded = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            return predict(padded)[:h, :w]
        return predict(rgb)

    step = max(1, tile - overlap)
    acc = np.zeros((h, w), dtype=np.float32)
    wgt = np.zeros((h, w), dtype=np.float32)
    win = _blend_window(tile, tile)

    ys = list(range(0, max(1, h - tile + 1), step))
    xs = list(range(0, max(1, w - tile + 1), step))
    if not ys or ys[-1] != h - tile:
        ys.append(max(0, h - tile))
    if not xs or xs[-1] != w - tile:
        xs.append(max(0, w - tile))

    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            patch = rgb[y0:y1, x0:x1]
            ph, pw = patch.shape[:2]
            if ph < tile or pw < tile:
                patch = np.pad(
                    patch, ((0, tile - ph), (0, tile - pw), (0, 0)), mode="reflect"
                )
            prob = predict(patch)[:ph, :pw]
            acc[y0:y1, x0:x1] += prob * win[:ph, :pw]
            wgt[y0:y1, x0:x1] += win[:ph, :pw]

    return acc / np.maximum(wgt, 1e-6)
