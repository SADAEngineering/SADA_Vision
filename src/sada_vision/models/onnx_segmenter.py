"""Inferenz des trainierten Netzes ueber ONNX Runtime.

Trainiert wird mit PyTorch (siehe ``training/``), ausgeliefert wird ONNX.
Das haelt torch aus dem Dienst-Image heraus - rund 2 GB weniger je Container -
und macht die Laufzeit austauschbar.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .base import SegmenterInfo, tiled_probability

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class OnnxSegmenter:
    """Binaere Segmentierung, Ausgabe sind Logits in (1, 1, H, W)."""

    def __init__(self, path: Path, threads: int = 0) -> None:
        self._path = path
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads > 0:
            opts.intra_op_num_threads = threads
        self._session = ort.InferenceSession(
            str(path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

        shape = self._session.get_inputs()[0].shape
        self._tile = _static_dim(shape[2]) or _static_dim(shape[3]) or 512

        meta = _load_sidecar(path)
        self.info = SegmenterInfo(
            name=meta.get("name", path.stem),
            kind="onnx",
            trained=True,
            note=meta.get("note", ""),
        )
        self._mean = np.array(meta.get("mean", _IMAGENET_MEAN), dtype=np.float32)
        self._std = np.array(meta.get("std", _IMAGENET_STD), dtype=np.float32)
        self._overlap = int(meta.get("tile_overlap", max(32, self._tile // 8)))

    @property
    def tile(self) -> int:
        return self._tile

    def probability(self, rgb: np.ndarray) -> np.ndarray:
        return tiled_probability(rgb, self._predict_tile, self._tile, self._overlap)

    def _predict_tile(self, rgb: np.ndarray) -> np.ndarray:
        x = rgb.astype(np.float32) / 255.0
        x = (x - self._mean) / self._std
        x = np.transpose(x, (2, 0, 1))[None]  # NCHW
        out = self._session.run(None, {self._input_name: x.astype(np.float32)})[0]
        logits = np.squeeze(out)
        return _sigmoid(logits).astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Stabil fuer grosse Betraege - exp(710) ist inf.
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _static_dim(value: object) -> int | None:
    return value if isinstance(value, int) and value > 0 else None


def _load_sidecar(path: Path) -> dict:
    """Begleitdatei ``<modell>.json`` mit Normierung und Herkunft."""
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        return {}
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
