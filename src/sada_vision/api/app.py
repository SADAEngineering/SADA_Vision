"""Der Host."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import CONTRACT_VERSION, __version__
from ..config import get_settings
from ..logging_setup import configure_logging, get_logger
from ..models import get_segmenter
from . import routes_detect, routes_health, routes_preview, routes_tools
from .schemas import ErrorDto

log = get_logger(__name__)

_DESCRIPTION = """
Erkennung und Vermessung von Bauteilschaeden auf Fotos.

Risse sind gebaut, Schrauben und Korrosion folgen unter denselben Pfaden und
demselben Vertrag.

**Der Vertrag ist additiv.** Felder kommen hinzu, keines wird umbenannt oder
entfernt - die Unity-App deserialisiert mit `JsonUtility` und meldet einen
Tippfehler nicht, sie liefert `null`.

Koordinaten sind Pixel des **Originalbildes** in `(y, x)`, Ursprung links oben.
Millimeterwerte sind `-1`, solange kein Massstab feststeht.
""".strip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()
    # Modell beim Start laden, nicht beim ersten Aufruf: sonst laeuft der
    # erste Anwender in mehrere Sekunden Wartezeit und haelt den Dienst fuer
    # kaputt. Und /health/ready sagt erst dann die Wahrheit.
    segmenter = get_segmenter("crack")
    log.info(
        "start",
        version=__version__,
        contract=CONTRACT_VERSION,
        model=segmenter.info.name,
        trained=segmenter.info.trained,
        model_dir=str(settings.model_dir),
    )
    if not segmenter.info.trained:
        log.warning("start.ohne_modell", hinweis=segmenter.info.note)
    yield
    log.info("stop")


def create_app() -> FastAPI:
    app = FastAPI(
        title="SADA Vision",
        version=__version__,
        description=_DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    app.include_router(routes_health.router)
    app.include_router(routes_detect.router)
    app.include_router(routes_preview.router)
    app.include_router(routes_tools.router)

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Correlation-Id") or str(uuid.uuid4())
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Correlation-Id"] = request_id
        response.headers["X-Contract-Version"] = CONTRACT_VERSION
        if not request.url.path.startswith("/health"):
            log.info(
                "http",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
                request_id=request_id,
            )
        return response

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorDto(
                error=_slug(exc.status_code),
                detail=str(exc.detail),
                request_id=request.headers.get("X-Correlation-Id", ""),
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=ErrorDto(
                error="validation_failed",
                detail="; ".join(
                    f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
                    for e in exc.errors()[:8]
                ),
                request_id=request.headers.get("X-Correlation-Id", ""),
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        # Was hier ankommt, ist ein Fehler von uns. Nach aussen nur die
        # Korrelationskennung - Stapelspuren gehen ins Protokoll, nicht in
        # die Antwort.
        request_id = request.headers.get("X-Correlation-Id", str(uuid.uuid4()))
        log.exception("unbehandelt", path=request.url.path, request_id=request_id)
        return JSONResponse(
            status_code=500,
            content=ErrorDto(
                error="internal_error",
                detail="Unexpected failure. Quote the request id when reporting.",
                request_id=request_id,
            ).model_dump(),
        )

    return app


def _slug(status_code: int) -> str:
    return {
        400: "bad_request",
        404: "not_found",
        405: "method_not_allowed",
        413: "payload_too_large",
        422: "unprocessable_entity",
        500: "internal_error",
        501: "not_implemented",
        503: "unavailable",
    }.get(status_code, f"http_{status_code}")


app = create_app()
