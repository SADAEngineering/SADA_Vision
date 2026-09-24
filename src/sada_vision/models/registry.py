"""Welches Modell fuer welche Aufgabe - und was tun, wenn keines da ist."""

from __future__ import annotations

import threading

from ..config import get_settings
from ..logging_setup import get_logger
from .base import Segmenter
from .classic_segmenter import ClassicRidgeSegmenter

log = get_logger(__name__)

_lock = threading.Lock()
_cache: dict[str, Segmenter] = {}


def get_segmenter(task: str = "crack") -> Segmenter:
    """Gibt den Segmentierer der Aufgabe zurueck, faellt sonst auf Ridge zurueck.

    Der Notbehelf ist Absicht: ein Dienst, der ohne Gewichte gar nicht
    startet, laesst sich nicht in Betrieb nehmen, bevor das Modell fertig ist.
    Erkennbar bleibt er an ``model.trained = false`` in jeder Antwort.
    """
    with _lock:
        if task in _cache:
            return _cache[task]
        seg = _build(task)
        _cache[task] = seg
        return seg


def _build(task: str) -> Segmenter:
    settings = get_settings()
    if task != "crack":
        raise KeyError(task)

    path = settings.model_path(settings.crack_model)
    if settings.crack_model and path.exists():
        try:
            from .onnx_segmenter import OnnxSegmenter

            seg = OnnxSegmenter(path, threads=settings.onnx_threads)
            log.info("modell.geladen", task=task, path=str(path), name=seg.info.name)
            return seg
        except Exception as exc:  # pragma: no cover - Ladefehler sind selten
            log.error("modell.ladefehler", task=task, path=str(path), error=str(exc))

    log.warning("modell.fehlt", task=task, path=str(path), fallback="classic-ridge")
    return ClassicRidgeSegmenter()


def registry_status() -> list[dict]:
    """Fuer /health: was ist geladen, was fehlt."""
    settings = get_settings()
    path = settings.model_path(settings.crack_model)
    seg = get_segmenter("crack")
    return [
        {
            "task": "crack",
            "model_name": seg.info.name,
            "model_kind": seg.info.kind,
            "trained": seg.info.trained,
            "weights_path": str(path),
            "weights_present": path.exists(),
        }
    ]


def reset_cache() -> None:
    """Nur fuer Tests."""
    with _lock:
        _cache.clear()
