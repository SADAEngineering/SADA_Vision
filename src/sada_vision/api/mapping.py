"""Vom internen Ergebnis auf den Vertrag.

Die einzige Stelle, an der interne Typen zu DTOs werden. Alles, was sich am
Draht aendern koennte, aendert sich hier - die Pipeline weiss nichts vom
Vertrag, und der Vertrag nichts von numpy.
"""

from __future__ import annotations

import numpy as np

from .. import CONTRACT_VERSION, __version__
from ..config import get_settings
from ..domain import Analysis, CrackInstance, Polyline
from ..models import registry_status
from ..pipeline import classify
from .schemas import (
    DetectionResponseDto,
    HealthDto,
    ImageInfoDto,
    InstanceDto,
    InstanceListDto,
    ModelInfoDto,
    PathDto,
    PathListDto,
    ScaleDto,
    SummaryDto,
    TaskStatusDto,
    TaskStatusListDto,
    WarningListDto,
)

# So viele Nachkommastellen wie noetig, keine mehr: ein Pfad von 2.000 Punkten
# mit voller double-Darstellung blaeht die Antwort um ein Mehrfaches auf.
_COORD_DECIMALS = 2
_WIDTH_PX_DECIMALS = 3
_WIDTH_MM_DECIMALS = 4


def _round(values: np.ndarray, decimals: int) -> list[float]:
    return [float(v) for v in np.round(np.asarray(values, dtype=np.float64), decimals)]


def path_to_dto(path: Polyline) -> PathDto:
    flat = np.asarray(path.points_yx, dtype=np.float64).reshape(-1)
    return PathDto(
        point_count=path.point_count,
        path_yx=_round(flat, _COORD_DECIMALS),
        width_px=_round(path.width_px, _WIDTH_PX_DECIMALS),
        width_mm=_round(path.width_mm, _WIDTH_MM_DECIMALS),
        width_at_junction=(
            [bool(v) for v in path.at_junction]
            if path.at_junction is not None
            else [False] * path.point_count
        ),
        length_px=round(float(path.length_px), 2),
        length_mm=round(float(path.length_mm), 3),
        is_loop=path.is_loop,
    )


def instance_to_dto(inst: CrackInstance, include_paths: bool = True) -> InstanceDto:
    paths = PathListDto(
        items=[path_to_dto(p) for p in inst.paths] if include_paths else []
    )
    return InstanceDto(
        instance_id=inst.instance_id,
        label=inst.label,
        score=round(float(inst.score), 4),
        bbox_yxyx=[round(float(v), 2) for v in inst.bbox_yxyx],
        pattern=inst.pattern,
        orientation_deg=round(float(inst.orientation_deg), 2),
        orientation_class=inst.orientation_class,
        tortuosity=round(float(inst.tortuosity), 4),
        branch_count=inst.branch_count,
        touches_border=inst.touches_border,
        width_samples_excluded=inst.width_samples_excluded,
        severity=inst.severity,
        width_max_px=round(float(inst.width_max_px), 3),
        width_mean_px=round(float(inst.width_mean_px), 3),
        width_p95_px=round(float(inst.width_p95_px), 3),
        width_max_mm=round(float(inst.width_max_mm), 4),
        width_mean_mm=round(float(inst.width_mean_mm), 4),
        width_p95_mm=round(float(inst.width_p95_mm), 4),
        length_px=round(float(inst.length_px), 2),
        length_mm=round(float(inst.length_mm), 3),
        area_px=round(float(inst.area_px), 1),
        paths=paths,
    )


def _summary(analysis: Analysis) -> SummaryDto:
    if not analysis.instances:
        return SummaryDto(instance_count=0, severity="none")

    max_px = max(i.width_max_px for i in analysis.instances)
    mm_values = [i.width_max_mm for i in analysis.instances if i.width_max_mm >= 0]
    max_mm = max(mm_values) if mm_values else -1.0

    total_px = float(sum(i.length_px for i in analysis.instances))
    mm_lengths = [i.length_mm for i in analysis.instances]
    total_mm = float(sum(mm_lengths)) if all(v >= 0 for v in mm_lengths) else -1.0

    return SummaryDto(
        instance_count=len(analysis.instances),
        severity=classify.severity_of(max_mm),
        max_width_px=round(float(max_px), 3),
        max_width_mm=round(float(max_mm), 4),
        total_length_px=round(total_px, 2),
        total_length_mm=round(total_mm, 3),
    )


def analysis_to_dto(
    analysis: Analysis,
    request_id: str,
    include_paths: bool = True,
) -> DetectionResponseDto:
    return DetectionResponseDto(
        contract_version=CONTRACT_VERSION,
        task=analysis.task,
        request_id=request_id,
        image=ImageInfoDto(
            width=analysis.image_width,
            height=analysis.image_height,
        ),
        model=ModelInfoDto(
            name=analysis.model_name,
            kind=analysis.model_kind,
            trained=analysis.model_trained,
        ),
        scale=ScaleDto(
            source=analysis.scale.source,
            known=analysis.scale.known,
            mm_per_px=round(float(analysis.scale.mm_per_px), 6),
            confidence=round(float(analysis.scale.confidence), 3),
            note=analysis.scale.note,
        ),
        summary=_summary(analysis),
        instances=InstanceListDto(
            items=[instance_to_dto(i, include_paths) for i in analysis.instances]
        ),
        warnings=WarningListDto(items=analysis.warnings),
        duration_ms=round(float(analysis.duration_ms), 1),
    )


def health_to_dto() -> HealthDto:
    settings = get_settings()
    rows = registry_status()
    tasks = TaskStatusListDto(
        items=[
            TaskStatusDto(
                task=r["task"],
                model_name=r["model_name"],
                model_kind=r["model_kind"],
                trained=bool(r["trained"]),
                weights_present=bool(r["weights_present"]),
                # Der Dienst antwortet auch ohne Gewichte - deshalb ist
                # "ready" nicht dasselbe wie "trained". Wer auf belastbare
                # Befunde wartet, prueft "trained".
                ready=True,
            )
            for r in rows
        ]
    )
    degraded = any(not t.trained for t in tasks.items)
    return HealthDto(
        status="degraded" if degraded else "ok",
        service=settings.service_name,
        version=__version__,
        contract_version=CONTRACT_VERSION,
        tasks=tasks,
    )
