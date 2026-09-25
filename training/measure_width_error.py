"""Wie genau misst der Dienst - gegen die annotierte Wahrheit, auf echten Fotos.

IoU, F1 und tolerantes F1 beantworten, ob der Riss **gefunden** wurde. Keine
davon beantwortet die Frage, auf die es bei diesem Produkt ankommt:

    Wie weit liegt die gemeldete Rissbreite neben der wahren?

Das lässt sich messen, ohne ein einziges Foto nachzumessen. Die Masken des
Datensatzes sind von Menschen gezeichnet - sie **sind** die Wahrheit. Also:
dieselbe Geometrie- und Breitenmessung einmal auf der wahren Maske und
einmal auf der Vorhersage laufen lassen und die beiden Zahlen vergleichen.

Was dabei herauskommt, ist der Fehler der **ganzen Kette** - Netz, Schwelle,
Skelett, Lot, Interpolation - in Pixeln. Mit einem Maßstab von 0,08 mm/px
sind 0,5 px Fehler 0,04 mm, und das ist die Zahl, die in ein Protokoll
gehört: nicht "IoU 0,74", sondern "misst auf 0,04 mm genau".

Die Grenze des Verfahrens gehört dazu: der Vergleich misst gegen die
**gezeichnete** Maske, nicht gegen den Riss. Wo der Annotierende die Kante
anders gesetzt hat als die Physik, steckt der Fehler schon in der Wahrheit.
Deshalb wird auch die Streuung ausgegeben und nicht nur der Mittelwert.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from sada_vision.pipeline import geometry, width


def measure(mask: np.ndarray, prob: np.ndarray, threshold: float) -> dict | None:
    """Breite und Länge einer Maske - mit demselben Code wie der Dienst."""
    clean = geometry.clean_mask(mask, min_area=40)
    comps = geometry.components(clean, min_area=40)
    if not comps:
        return None

    dist = width.distance_transform(clean)
    widths: list[np.ndarray] = []
    total_length = 0.0

    for sub_mask, (y0, x0, y1, x1), _area in comps:
        branches = geometry.skeleton_branches(sub_mask).branches
        prob_crop = prob[y0:y1, x0:x1]
        dist_crop = dist[y0:y1, x0:x1]
        for raw in branches:
            if geometry.polyline_length(raw) < 12.0:
                continue
            points = geometry.simplify_rdp(raw, 1.0)
            if points.shape[0] < 2:
                continue
            widths.append(
                width.widths_perpendicular(points, prob_crop, dist_crop, threshold)
            )
            total_length += geometry.polyline_length(points)

    if not widths:
        return None
    all_widths = np.concatenate(widths)
    return {
        "width_max": float(np.max(all_widths)),
        "width_mean": float(np.mean(all_widths)),
        "width_p95": float(np.percentile(all_widths, 95)),
        "length": total_length,
        "instances": len(comps),
    }


def run(
    checkpoint: Path,
    data_root: Path,
    limit: int,
    threshold: float,
    max_annotation_width: float = 0.0,
) -> dict:
    import torch

    from training.data import filter_linear
    from training.evaluate import load_model, probability

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(checkpoint, device)
    tile = config.image_size
    overlap = max(32, tile // 8)

    names = sorted(p.stem for p in (data_root / "images").glob("*.png"))

    # Ohne Filter misst man ueberwiegend gegen Abplatzungen und Schlagloecher -
    # dort ist die "Breite" der Durchmesser einer Flaeche, nicht die eines
    # Risses. Fuer die Frage "wie genau misst der Dienst eine Rissbreite" ist
    # das die falsche Grundgesamtheit.
    if max_annotation_width > 0:
        vorher = len(names)
        names = filter_linear(
            data_root, names, max_annotation_width, keep_negatives=False
        )
        print(
            f"Nur linienhafte Annotationen (<= {max_annotation_width:g} px "
            f"mittlere Breite): {len(names)} von {vorher}"
        )

    if limit:
        names = names[:limit]

    rows: list[dict] = []
    missed = 0
    invented = 0

    for index, name in enumerate(names, start=1):
        bgr = cv2.imread(str(data_root / "images" / f"{name}.png"), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        truth_mask = (
            cv2.imread(str(data_root / "masks" / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
            > 127
        )

        prob = probability(model, rgb, tile, overlap, device)
        pred_mask = prob >= threshold

        # Die wahre Breite wird auf einem harten Wahrscheinlichkeitsbild
        # gemessen (die Maske selbst) - dieselbe Rechnung, andere Eingabe.
        truth = measure(truth_mask, truth_mask.astype(np.float32), threshold)
        pred = measure(pred_mask, prob, threshold)

        if truth is None:
            if pred is not None:
                invented += 1   # Riss gemeldet, wo keiner annotiert ist
            continue
        if pred is None:
            missed += 1         # annotierter Riss nicht gefunden
            continue

        rows.append(
            {
                "name": name,
                "truth_width_max": truth["width_max"],
                "pred_width_max": pred["width_max"],
                "truth_width_mean": truth["width_mean"],
                "pred_width_mean": pred["width_mean"],
                "truth_length": truth["length"],
                "pred_length": pred["length"],
            }
        )
        if index % 50 == 0:
            print(f"  {index}/{len(names)}", flush=True)

    return _summarise(
        rows, missed, invented, len(names), threshold, checkpoint, max_annotation_width
    )


def _summarise(
    rows: list[dict],
    missed: int,
    invented: int,
    total: int,
    threshold: float,
    checkpoint: Path,
    max_annotation_width: float = 0.0,
) -> dict:
    if not rows:
        return {"error": "Kein einziges Bildpaar vergleichbar."}

    def stats(key: str) -> dict:
        truth = np.array([r[f"truth_{key}"] for r in rows])
        pred = np.array([r[f"pred_{key}"] for r in rows])
        err = pred - truth
        rel = err / np.maximum(truth, 1e-6)
        return {
            "median_error_px": float(np.median(err)),
            "mean_error_px": float(np.mean(err)),
            "median_abs_error_px": float(np.median(np.abs(err))),
            "p90_abs_error_px": float(np.percentile(np.abs(err), 90)),
            "median_relative_error": float(np.median(rel)),
            "truth_median_px": float(np.median(truth)),
        }

    return {
        "checkpoint": str(checkpoint),
        "threshold": threshold,
        "max_annotation_width": max_annotation_width,
        "images_total": total,
        "images_compared": len(rows),
        "images_missed": missed,
        "images_invented": invented,
        "width_max": stats("width_max"),
        "width_mean": stats("width_mean"),
        "length": stats("length"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/crackseg9k/test"))
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--max-annotation-width", type=float, default=0.0,
        dest="max_annotation_width",
        help="Nur linienhafte Annotationen messen, z. B. 12 (Pixel)",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = run(
        args.checkpoint,
        args.data,
        args.limit,
        args.threshold,
        args.max_annotation_width,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)

    target = args.out or args.checkpoint.parent / "width_error.json"
    target.write_text(text, encoding="utf-8")
    print(f"\nGeschrieben: {target}")

    if "width_max" in report:
        w = report["width_max"]
        print(
            f"\nGroesste Breite je Bild: Median des Betragsfehlers "
            f"{w['median_abs_error_px']:.2f} px "
            f"(typische wahre Breite {w['truth_median_px']:.1f} px), "
            f"systematischer Versatz {w['median_error_px']:+.2f} px."
        )
        print(
            f"Bei 0,08 mm/px entspricht das rund "
            f"{w['median_abs_error_px'] * 0.08:.3f} mm."
        )
        print(
            f"{report['images_missed']} von {report['images_total']} annotierten "
            f"Rissen nicht gefunden, {report['images_invented']} erfunden."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
