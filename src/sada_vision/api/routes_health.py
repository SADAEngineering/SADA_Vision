"""Leben, Bereitschaft, Zustand.

Drei Endpunkte statt einem, weil sie drei verschiedene Fragen beantworten:

``/health/live``   Laeuft der Prozess? Fuer den Neustarter im Container.
``/health/ready``  Darf Verkehr kommen? Modell geladen, Speicher warm.
``/health``        Was ist los? Fuer Menschen und Ueberwachung.
"""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from ..models import registry_status
from .mapping import health_to_dto
from .schemas import HealthDto

router = APIRouter(tags=["health"])


@router.get("/health/live", summary="Prozess lebt")
async def live() -> dict:
    return {"status": "ok"}


@router.get("/health/ready", summary="Bereit fuer Verkehr")
async def ready(response: Response) -> dict:
    rows = registry_status()
    if not rows:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "no_tasks"}
    # Der Notbehelf zaehlt als bereit: der Dienst antwortet, er antwortet nur
    # schlechter. Wer das nicht will, prueft /health und liest "trained".
    return {"status": "ok"}


@router.get("/health", response_model=HealthDto, summary="Zustand im Klartext")
async def health() -> HealthDto:
    return health_to_dto()
