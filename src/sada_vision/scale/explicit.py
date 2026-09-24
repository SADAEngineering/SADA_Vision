"""Der Aufrufer gibt den Massstab vor."""

from __future__ import annotations

from ..domain import ScaleInfo


def explicit_scale(mm_per_px: float) -> ScaleInfo:
    if mm_per_px is None or mm_per_px <= 0:
        return ScaleInfo(source="none", note="mm_per_px muss groesser als 0 sein.")
    return ScaleInfo(
        source="explicit",
        mm_per_px=float(mm_per_px),
        confidence=1.0,
        note="Vom Aufrufer vorgegeben - nicht geprueft.",
    )
