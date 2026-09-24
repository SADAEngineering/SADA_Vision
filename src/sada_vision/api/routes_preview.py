"""Ein Bild sagt mehr als 2.000 Koordinaten.

Der Vorschau-Endpunkt laesst denselben Lauf wie ``/detect/crack`` los und
zeichnet das Ergebnis ins Foto: Verlauf als Linie, die breiteste Stelle als
Kreuz, je Befund ein Schild mit Breite und Band. Er ist zum **Pruefen** da -
beim Einrichten, beim Vergleich zweier Modelle, beim Streit darueber, ob der
Dienst wirklich den Riss und nicht die Fuge gefunden hat.

Kein Endpunkt fuer die Fachlogik: TraceForm und die AR-App rechnen mit
Koordinaten, nicht mit Pixeln aus einem PNG.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from PIL import Image

from ..domain import Analysis
from ..pipeline import analyse_crack
from ..pipeline.preprocess import ImageRejected, decode
from ..scale import resolve_scale

router = APIRouter(prefix="/api/v1", tags=["detection"])

# BGR, weil OpenCV zeichnet.
_BAND_COLOURS = {
    "hairline": (180, 220, 120),
    "fine": (90, 200, 250),
    "moderate": (60, 150, 255),
    "wide": (50, 90, 250),
    "severe": (70, 60, 220),
    "unknown": (200, 200, 200),
}


@router.post(
    "/preview/crack",
    summary="Wie /detect/crack, aber als gezeichnetes PNG",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
async def preview_crack(
    image: UploadFile = File(...),
    mm_per_px: float = Form(-1.0),
    marker_size_mm: float = Form(-1.0),
    marker_id: int = Form(-1),
    depth_mm: float = Form(-1.0),
    focal_px: float = Form(-1.0),
    tilt_deg: float = Form(-1.0),
    threshold: float = Form(-1.0),
    line_width: int = Form(0, description="0 = aus der Rissbreite ableiten"),
    max_labels: int = Form(10, description="Wie viele Befunde beschriftet werden"),
) -> Response:
    data = await image.read()
    try:
        source = decode(data)
    except ImageRejected as exc:
        # 422 als Zahl: Starlette hat die Konstante umbenannt, die Zahl bleibt.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    scale, _ = resolve_scale(
        source.rgb,
        mm_per_px=mm_per_px,
        marker_size_mm=marker_size_mm,
        marker_id=marker_id,
        depth_mm=depth_mm,
        focal_px=focal_px,
        tilt_deg=tilt_deg,
    )
    analysis = analyse_crack(
        source, scale, threshold=None if threshold < 0 else threshold
    )

    canvas = draw_overlay(source.rgb, analysis, line_width, max_labels)
    buffer = io.BytesIO()
    Image.fromarray(canvas).save(buffer, format="PNG", optimize=True)
    return Response(content=buffer.getvalue(), media_type="image/png")


def draw_overlay(
    rgb: np.ndarray,
    analysis: Analysis,
    line_width: int = 0,
    max_labels: int = 10,
) -> np.ndarray:
    """Zeichnet Verlauf, Extremstelle und Beschriftung in eine Kopie.

    **Gezeichnet wird jeder Befund, beschriftet nur die ersten paar.** Ein
    Bild mit siebzig Schildern ist keine Pruefhilfe mehr - man sieht das
    Foto nicht mehr, und genau darum geht es hier. Die Befunde stehen nach
    Breite sortiert, die Beschriftung trifft also die schwersten.
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # Beschriftung bei 12-Megapixel-Fotos sonst unlesbar klein.
    scale_ui = max(0.5, min(rgb.shape[1], rgb.shape[0]) / 1400.0)

    for index, inst in enumerate(analysis.instances):
        colour = _BAND_COLOURS.get(inst.severity, _BAND_COLOURS["unknown"])
        thickness = line_width if line_width > 0 else max(1, int(round(2 * scale_ui)))

        for path in inst.paths:
            pts = path.points_yx[:, ::-1].round().astype(np.int32)  # (y,x) -> (x,y)
            cv2.polylines(bgr, [pts], False, colour, thickness, cv2.LINE_AA)

        _mark_widest(bgr, inst, colour, scale_ui)
        if index < max_labels:
            _label(bgr, inst, colour, scale_ui)

    hidden = max(0, len(analysis.instances) - max_labels)
    _banner(bgr, analysis, scale_ui, hidden)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _mark_widest(bgr, inst, colour, scale_ui) -> None:
    best_path, best_idx, best_val = None, -1, -1.0
    for path in inst.paths:
        if path.width_px.size == 0:
            continue
        idx = int(np.argmax(path.width_px))
        if float(path.width_px[idx]) > best_val:
            best_path, best_idx, best_val = path, idx, float(path.width_px[idx])
    if best_path is None:
        return
    y, x = best_path.points_yx[best_idx]
    r = max(6, int(round(10 * scale_ui)))
    cv2.drawMarker(
        bgr, (int(x), int(y)), colour, cv2.MARKER_TILTED_CROSS, r * 2,
        max(1, int(round(2 * scale_ui))), cv2.LINE_AA,
    )


def _label(bgr, inst, colour, scale_ui) -> None:
    if inst.width_max_mm >= 0:
        text = f"#{inst.instance_id} {inst.width_max_mm:.2f} mm ({inst.severity})"
    else:
        text = f"#{inst.instance_id} {inst.width_max_px:.1f} px"
    text += f" | {inst.pattern}, {inst.orientation_class}"

    height, width = bgr.shape[:2]
    y0, x0 = int(inst.bbox_yxyx[0]), int(inst.bbox_yxyx[1])
    font_scale = 0.5 * scale_ui
    thickness = max(1, int(round(scale_ui)))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    # Schild im Bild halten: ein Befund am rechten Rand haette sonst eine
    # halb abgeschnittene Beschriftung, und gerade die Zahl faellt weg.
    x0 = int(np.clip(x0, 0, max(0, width - tw - 8)))
    ty = int(np.clip(max(th + base + 2, y0 - 4), th + base + 2, height - 2))

    cv2.rectangle(bgr, (x0, ty - th - base - 2), (x0 + tw + 6, ty + 2), colour, -1)
    cv2.putText(
        bgr, text, (x0 + 3, ty - base + 1), cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, (20, 20, 20), thickness, cv2.LINE_AA,
    )


def _banner(bgr, analysis: Analysis, scale_ui, hidden: int = 0) -> None:
    """Oben eine Zeile: Modell, Massstab, Befundzahl - und die Warnung."""
    parts = [
        f"{analysis.model_name} ({analysis.model_kind})",
        f"scale: {analysis.scale.source}",
        f"{len(analysis.instances)} findings"
        + (f" ({hidden} unlabelled)" if hidden else ""),
    ]
    if not analysis.model_trained:
        parts.insert(0, "UNTRAINED FALLBACK")
    text = "  |  ".join(parts)

    font_scale = 0.55 * scale_ui
    thickness = max(1, int(round(scale_ui)))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    colour = (40, 40, 220) if not analysis.model_trained else (40, 40, 40)
    cv2.rectangle(bgr, (0, 0), (tw + 12, th + base + 10), colour, -1)
    cv2.putText(
        bgr, text, (6, th + 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        (255, 255, 255), thickness, cv2.LINE_AA,
    )
