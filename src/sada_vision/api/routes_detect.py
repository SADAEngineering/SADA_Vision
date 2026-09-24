"""Der eine Endpunkt, der zaehlt: Bild rein, Befund raus."""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from ..logging_setup import get_logger
from ..pipeline import analyse_crack
from ..pipeline.preprocess import ImageRejected, decode
from ..scale import resolve_scale
from .mapping import analysis_to_dto
from .schemas import DetectionResponseDto
from .workload import run_heavy

log = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["detection"])

_SUPPORTED_TASKS = {"crack"}
_PLANNED_TASKS = {"bolt", "corrosion"}


@router.post(
    "/detect/crack",
    response_model=DetectionResponseDto,
    summary="Risse erkennen, vermessen und einordnen",
)
async def detect_crack(
    image: UploadFile = File(..., description="JPEG oder PNG des Bauteils"),
    # --- Massstab, alle drei Wege optional ---------------------------------
    mm_per_px: float = Form(-1.0, description="Bekannter Massstab"),
    marker_size_mm: float = Form(-1.0, description="Kantenlaenge des ArUco-Markers"),
    marker_id: int = Form(-1, description="Kennung; -1 = groesster Marker im Bild"),
    marker_dictionary: str = Form("", description="Vorgabe: DICT_4X4_50"),
    depth_mm: float = Form(-1.0, description="LiDAR: Abstand zur Flaeche"),
    focal_px: float = Form(-1.0, description="LiDAR: Brennweite in Pixeln des Bildes"),
    tilt_deg: float = Form(-1.0, description="LiDAR: Schraeglage der Flaeche"),
    # --- Feineinstellung ---------------------------------------------------
    threshold: float = Form(-1.0, description="Maskenschwelle, sonst Vorgabe"),
    min_area_px: int = Form(-1, description="Kleinster Befund in Pixeln"),
    max_instances: int = Form(-1, description="Obergrenze der gemeldeten Befunde"),
    width_method: str = Form("", description="perpendicular | distance_transform"),
    include_paths: bool = Form(True, description="false = nur Kennzahlen, kein Verlauf"),
) -> DetectionResponseDto:
    request_id = str(uuid.uuid4())
    structlog.contextvars.bind_contextvars(request_id=request_id)

    data = await image.read()
    try:
        source = decode(data)
    except ImageRejected as exc:
        log.warning("bild.abgelehnt", reason=str(exc), size=len(data))
        # 422 als Zahl: Starlette hat die Konstante umbenannt, die Zahl bleibt.
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    if width_method and width_method not in {"perpendicular", "distance_transform"}:
        # 422 als Zahl: Starlette hat die Konstante umbenannt, die Zahl bleibt.
        raise HTTPException(
            status_code=422,
            detail=f"Unknown width_method '{width_method}'.",
        )

    # Massstab und Analyse rechnen beide synchron und sekundenlang. In der
    # Ereignisschleife wuerde das den ganzen Prozess anhalten - siehe
    # workload.py.
    def compute():
        scale, scale_warnings = resolve_scale(
            source.rgb,
            mm_per_px=mm_per_px,
            marker_size_mm=marker_size_mm,
            marker_id=marker_id,
            marker_dictionary=marker_dictionary,
            depth_mm=depth_mm,
            focal_px=focal_px,
            tilt_deg=tilt_deg,
        )
        result = analyse_crack(
            source,
            scale,
            threshold=None if threshold < 0 else threshold,
            min_area_px=None if min_area_px < 0 else min_area_px,
            max_instances=None if max_instances < 0 else max_instances,
            width_method=width_method or None,
        )
        result.warnings = scale_warnings + result.warnings
        return result

    analysis = await run_heavy(compute)

    log.info(
        "befund",
        instances=len(analysis.instances),
        scale=analysis.scale.source,
        model=analysis.model_name,
        trained=analysis.model_trained,
        duration_ms=round(analysis.duration_ms, 1),
        width_px=source.width,
        height_px=source.height,
    )
    structlog.contextvars.clear_contextvars()
    return analysis_to_dto(analysis, request_id, include_paths=include_paths)


@router.post(
    "/detect/{task}",
    response_model=DetectionResponseDto,
    summary="Platzhalter fuer die noch nicht gebauten Aufgaben",
)
async def detect_other(task: str) -> DetectionResponseDto:
    """Der Pfad steht von Anfang an fest, damit Aufrufer ihn schon verdrahten.

    Schrauben und Korrosion sind die Anwendungsfaelle zwei und drei; beide
    liefern denselben Vertrag mit anderem ``label``.
    """
    if task in _SUPPORTED_TASKS:
        raise HTTPException(
            status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
            detail=f"Use /api/v1/detect/{task} with its own handler.",
        )
    if task in _PLANNED_TASKS:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"Task '{task}' is planned but not trained yet.",
        )
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=(
            f"Unknown task '{task}'. Known: "
            f"{', '.join(sorted(_SUPPORTED_TASKS | _PLANNED_TASKS))}."
        ),
    )
