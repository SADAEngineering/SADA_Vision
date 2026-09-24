"""Der Lauf: Bild rein, Befund raus.

    Bild -> Arbeitsbild -> Massstab -> Wahrscheinlichkeit -> Maske
         -> Instanzen -> Skelett -> Verlauf -> Breite -> Einordnung

Alle Koordinaten der Ausgabe sind Pixel des **Originalbildes** in ``(y, x)``,
auch wenn intern verkleinert gerechnet wurde. Wer das Bild mitschickt und die
Antwort darauf zeichnet, trifft.
"""

from __future__ import annotations

import time

import numpy as np

from ..config import get_settings
from ..domain import Analysis, CrackInstance, Polyline, ScaleInfo, SourceImage
from ..logging_setup import get_logger
from ..models import get_segmenter
from . import classify, geometry, width
from .preprocess import working_image

log = get_logger(__name__)


def analyse_crack(
    source: SourceImage,
    scale: ScaleInfo,
    *,
    threshold: float | None = None,
    min_area_px: int | None = None,
    max_instances: int | None = None,
    width_method: str | None = None,
) -> Analysis:
    settings = get_settings()
    started = time.perf_counter()

    min_area = settings.min_component_area_px if min_area_px is None else int(min_area_px)
    limit = settings.max_instances if max_instances is None else int(max_instances)
    method = width_method or settings.width_method

    segmenter = get_segmenter("crack")

    # Rangfolge der Schwelle: was der Aufrufer sagt, sonst was zu *diesem*
    # Modell gehoert, sonst die Vorgabe des Dienstes. Die mittlere Stufe ist
    # die wichtige: die modelleigene Schwelle wird beim Bewerten abgesucht
    # und korrigiert unter anderem die Breitenverzerrung. Faellt sie weg,
    # misst der Dienst mit einem fremden Wert - und zwar stillschweigend.
    if threshold is not None:
        threshold = float(threshold)
    elif segmenter.info.threshold > 0:
        threshold = segmenter.info.threshold
    else:
        threshold = settings.mask_threshold

    work, factor = working_image(source)
    warnings: list[str] = []
    if factor > 1.0:
        warnings.append(
            f"Bild fuer die Auswertung auf {work.shape[1]}x{work.shape[0]} verkleinert "
            f"(Faktor {factor:.2f}). Feine Risse koennen dabei verloren gehen - "
            f"SADAVISION_MAX_EDGE_PX anheben, wenn die Breiten zaehlen."
        )

    prob = segmenter.probability(work)

    if _blank_marker(prob, scale, factor):
        warnings.append(
            "Der erkannte Massstabsmarker wurde von der Auswertung "
            "ausgenommen - er liegt auf dem Bauteil, nicht darin."
        )

    mask = geometry.clean_mask(prob >= threshold, min_area)
    dist = width.distance_transform(mask)

    instances: list[CrackInstance] = []
    comps = geometry.components(mask, min_area)
    if len(comps) > limit:
        warnings.append(
            f"{len(comps)} Befunde gefunden, {limit} gemeldet (die flaechengroessten)."
        )
        comps = comps[:limit]

    for index, (sub_mask, bbox, area) in enumerate(comps):
        inst = _build_instance(
            index=index,
            sub_mask=sub_mask,
            bbox=bbox,
            area=area,
            prob=prob,
            dist=dist,
            factor=factor,
            scale=scale,
            threshold=threshold,
            method=method,
            settings=settings,
        )
        if inst is not None:
            instances.append(inst)

    # Nach Schwere ordnen: der breiteste Riss steht oben, nicht der groesste.
    instances.sort(key=lambda i: (i.width_max_px, i.length_px), reverse=True)
    for new_id, inst in enumerate(instances):
        inst.instance_id = new_id

    if not segmenter.info.trained:
        warnings.append(segmenter.info.note)
    if not scale.known:
        warnings.append(scale.note)

    return Analysis(
        task="crack",
        image_width=source.width,
        image_height=source.height,
        scale=scale,
        instances=instances,
        model_name=segmenter.info.name,
        model_kind=segmenter.info.kind,
        model_trained=segmenter.info.trained,
        warnings=warnings,
        duration_ms=(time.perf_counter() - started) * 1000.0,
    )


def _blank_marker(prob: np.ndarray, scale: ScaleInfo, factor: float) -> bool:
    """Blendet den Massstabsmarker aus dem Wahrscheinlichkeitsbild aus.

    Ein ArUco-Marker besteht aus harten Schwarz-Weiss-Kanten - fuer jeden
    Kantenfilter sieht das aus wie ein Netz feiner Risse, und auch ein
    trainiertes Netz hat so etwas im Training nie gesehen. Der Marker klebt
    auf dem Bauteil, er ist kein Befund. Also raus damit, mit etwas Rand.
    """
    if scale.marker_quad is None:
        return False

    quad = np.asarray(scale.marker_quad, dtype=np.float64) / max(factor, 1e-9)
    # (y, x) -> (x, y) fuer OpenCV
    poly = quad[:, ::-1]
    centre = poly.mean(axis=0)
    # Zehn Prozent Rand: die Erkennung liefert die Ecken der schwarzen
    # Umrandung, der weisse Rand darum gehoert ebenso wenig zum Bauteil.
    poly = centre + (poly - centre) * 1.15

    import cv2

    cv2.fillPoly(prob, [poly.round().astype(np.int32)], 0.0)
    return True


def _build_instance(
    *,
    index: int,
    sub_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    area: float,
    prob: np.ndarray,
    dist: np.ndarray,
    factor: float,
    scale: ScaleInfo,
    threshold: float,
    method: str,
    settings,
) -> CrackInstance | None:
    y0, x0, y1, x1 = bbox
    branches, branch_count = geometry.skeleton_branches(sub_mask)
    if not branches:
        return None

    prob_crop = prob[y0:y1, x0:x1]
    dist_crop = dist[y0:y1, x0:x1]

    paths: list[Polyline] = []
    all_widths: list[np.ndarray] = []
    total_length_px = 0.0

    for raw in branches:
        if geometry.polyline_length(raw) < settings.min_branch_length_px:
            continue

        simplified = geometry.simplify_rdp(raw, settings.rdp_epsilon_px)
        simplified = geometry.resample(simplified, settings.max_points_per_path)
        if simplified.shape[0] < 2:
            continue

        if method == "distance_transform":
            w_px_local = width.widths_from_distance(simplified, dist_crop)
        else:
            w_px_local = width.widths_perpendicular(
                simplified, prob_crop, dist_crop, threshold
            )

        # Zuschnitt -> Arbeitsbild -> Originalbild
        points = simplified + np.array([y0, x0], dtype=np.float32)
        points = points * factor
        w_px = w_px_local * factor

        mm_per_px = scale.local_mm_per_px(points[:, 0], points[:, 1])
        w_mm = width.to_millimetres(w_px, mm_per_px)
        length_px = geometry.polyline_length(points)

        paths.append(
            Polyline(
                points_yx=points.astype(np.float32),
                width_px=w_px.astype(np.float32),
                width_mm=w_mm.astype(np.float32),
                length_px=length_px,
                length_mm=width.length_mm(points, mm_per_px),
                is_loop=bool(np.allclose(points[0], points[-1])),
            )
        )
        all_widths.append(w_px)
        total_length_px += length_px

    if not paths:
        return None

    widths_all = np.concatenate(all_widths, axis=0)

    w_max_px, w_mean_px, w_p95_px = classify.summarise_widths(widths_all)
    mm_all = np.concatenate([p.width_mm for p in paths])
    w_max_mm, w_mean_mm, w_p95_mm = classify.summarise_widths(mm_all)

    lengths_mm = [p.length_mm for p in paths]
    length_mm_total = float(sum(lengths_mm)) if all(v >= 0 for v in lengths_mm) else -1.0

    # Der laengste Ast bestimmt Richtung und Gewundenheit - bei einem
    # verzweigten Riss ist das der Stamm, nicht die Summe aller Aeste.
    longest = max(paths, key=lambda p: p.length_px)

    bbox_o = (y0 * factor, x0 * factor, y1 * factor, x1 * factor)
    bbox_area = max((y1 - y0) * (x1 - x0), 1) * factor * factor

    orientation = geometry.principal_orientation(longest.points_yx)
    inst = CrackInstance(
        instance_id=index,
        label="crack",
        score=float(np.mean(prob_crop[sub_mask])) if sub_mask.any() else 0.0,
        bbox_yxyx=bbox_o,
        paths=paths,
        pattern=classify.pattern_of(branch_count, total_length_px, bbox_area),
        orientation_deg=orientation,
        orientation_class=classify.orientation_class_of(orientation),
        tortuosity=geometry.tortuosity(longest.points_yx),
        branch_count=branch_count,
        width_max_px=w_max_px,
        width_mean_px=w_mean_px,
        width_p95_px=w_p95_px,
        width_max_mm=w_max_mm,
        width_mean_mm=w_mean_mm,
        width_p95_mm=w_p95_mm,
        length_px=total_length_px,
        length_mm=length_mm_total,
        area_px=area * factor * factor,
        severity=classify.severity_of(w_max_mm),
    )
    return inst
