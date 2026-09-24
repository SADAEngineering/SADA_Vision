"""Druckvorlage für den Maßstabsmarker.

Der Marker ist der genaueste Weg zu Millimetern — aber nur, wenn er in der
Größe gedruckt wird, die hinterher als ``marker_size_mm`` geschickt wird.
Genau da geht es in der Praxis schief: „An Seite anpassen" im Druckdialog
skaliert das Blatt, und die 60 mm sind dann 57,3 — ein Fehler von fünf
Prozent, der sich auf jede Rissbreite durchschlägt und nirgends auffällt.

Deshalb kommt die Vorlage aus dem Dienst selbst, mit einer aufgedruckten
Prüfstrecke daneben. Wer die nachmisst, weiß, ob der Druck stimmt.

Der Pfad liegt bewusst **außerhalb** von ``/api/v1``: das ist eine
Bequemlichkeit, kein Teil des Vertrags.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Query, Response
from PIL import Image

from ..scale.aruco import DEFAULT_DICTIONARY

router = APIRouter(prefix="/tools", tags=["tools"])

_MM_PER_INCH = 25.4


@router.get(
    "/marker.png",
    summary="Druckfertiger ArUco-Marker mit Prüfstrecke",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
async def marker_png(
    size_mm: float = Query(60.0, gt=5.0, le=400.0, description="Kantenlänge"),
    marker_id: int = Query(7, ge=0, le=49),
    dpi: int = Query(300, ge=72, le=1200),
    dictionary: str = Query(DEFAULT_DICTIONARY),
) -> Response:
    dict_id = getattr(cv2.aruco, dictionary, None)
    if dict_id is None:
        raise HTTPException(status_code=422, detail=f"Unknown dictionary '{dictionary}'.")

    px_per_mm = dpi / _MM_PER_INCH
    side = int(round(size_mm * px_per_mm))
    margin = int(round(12 * px_per_mm))
    caption = int(round(16 * px_per_mm))

    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, side)

    canvas = np.full(
        (side + 2 * margin + caption, side + 2 * margin), 255, dtype=np.uint8
    )
    canvas[margin : margin + side, margin : margin + side] = marker

    _ruler(canvas, px_per_mm, margin, side)
    _caption(canvas, size_mm, marker_id, dictionary, dpi, px_per_mm, margin, side)

    buffer = io.BytesIO()
    image = Image.fromarray(canvas)
    # dpi in die Datei schreiben, damit der Druckdialog die wahre Größe kennt.
    image.save(buffer, format="PNG", dpi=(dpi, dpi))
    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
        headers={
            "Content-Disposition": (
                f'inline; filename="aruco-{marker_id}-{size_mm:g}mm.png"'
            )
        },
    )


def _ruler(canvas: np.ndarray, px_per_mm: float, margin: int, side: int) -> None:
    """Prüfstrecke von 50 mm unter dem Marker - zum Nachmessen mit dem Lineal."""
    length_mm = 50.0
    y = margin + side + int(round(5 * px_per_mm))
    x0 = margin
    x1 = x0 + int(round(length_mm * px_per_mm))
    tick = int(round(2.0 * px_per_mm))

    cv2.line(canvas, (x0, y), (x1, y), 0, max(1, int(px_per_mm / 4)))
    for mm in range(0, int(length_mm) + 1, 10):
        x = x0 + int(round(mm * px_per_mm))
        cv2.line(canvas, (x, y - tick), (x, y + tick), 0, max(1, int(px_per_mm / 4)))


def _caption(
    canvas: np.ndarray,
    size_mm: float,
    marker_id: int,
    dictionary: str,
    dpi: int,
    px_per_mm: float,
    margin: int,
    side: int,
) -> None:
    scale = px_per_mm / 6.0
    thickness = max(1, int(round(px_per_mm / 5)))
    y = margin + side + int(round(13 * px_per_mm))
    lines = [
        f"{dictionary} id={marker_id}  edge={size_mm:g} mm  ->  marker_size_mm={size_mm:g}",
        "Print at 100% (no fit-to-page). Check: the bar above must span 50 mm.",
    ]
    for offset, text in enumerate(lines):
        cv2.putText(
            canvas,
            text,
            (margin, y + offset * int(round(5 * px_per_mm))),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            0,
            thickness,
            cv2.LINE_AA,
        )
