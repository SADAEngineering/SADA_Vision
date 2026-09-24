"""Ein trainiertes Modell auf dem zurueckgehaltenen Teil bewerten.

Zwei Dinge unterscheiden diese Bewertung von der Zahl am Ende des Trainings:

1. Sie laeuft auf dem **Testteil**, den kein Trainingsschritt gesehen hat.
2. Sie laeuft auf dem **ganzen Bild** ueber die Kachelung des Dienstes - nicht
   auf 256er-Ausschnitten. Genau so kommt das Modell spaeter zum Einsatz, und
   Kachelraender sind eine eigene Fehlerquelle.

Zusaetzlich wird die Schwelle abgesucht: welcher Wert liefert das beste
tolerante F1, und wo steht die Breitenverzerrung dabei. Der Dienst nimmt
hinterher genau diese Schwelle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from training.data import IMAGENET_MEAN, IMAGENET_STD
from training.metrics import Running

THRESHOLDS = (0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7)


def load_model(checkpoint: Path, device: torch.device):
    from training.train import Config, build_model

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    config = Config(**{k: v for k, v in state["config"].items() if k != "extra"})
    model = build_model(config)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, config


@torch.no_grad()
def probability(model, rgb: np.ndarray, tile: int, overlap: int, device) -> np.ndarray:
    """Kachelweise Inferenz - dieselbe Ueberblendung wie im Dienst."""
    from sada_vision.models.base import tiled_probability

    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)

    def predict(patch: np.ndarray) -> np.ndarray:
        x = (patch.astype(np.float32) / 255.0 - mean) / std
        tensor = torch.from_numpy(np.transpose(x, (2, 0, 1))[None]).to(device)
        logits = model(tensor)
        return torch.sigmoid(logits)[0, 0].cpu().numpy().astype(np.float32)

    return tiled_probability(rgb, predict, tile, overlap)


def evaluate(
    checkpoint: Path,
    data_root: Path,
    limit: int = 0,
    tolerance: int = 2,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(checkpoint, device)
    tile = config.image_size
    overlap = max(32, tile // 8)

    names = sorted(p.stem for p in (data_root / "images").glob("*.png"))
    if limit:
        names = names[:limit]
    print(f"{len(names)} Bilder aus {data_root}")

    per_threshold = {t: Running(tolerance) for t in THRESHOLDS}
    for index, name in enumerate(names, start=1):
        bgr = cv2.imread(str(data_root / "images" / f"{name}.png"), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        truth = cv2.imread(str(data_root / "masks" / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        truth = truth > 127

        prob = probability(model, rgb, tile, overlap, device)
        for threshold, running in per_threshold.items():
            running.update(prob >= threshold, truth)

        if index % 50 == 0:
            print(f"  {index}/{len(names)}", flush=True)

    report = {
        "checkpoint": str(checkpoint),
        "data": str(data_root),
        "images": len(names),
        "tolerance_px": tolerance,
        "by_threshold": {
            f"{t:.2f}": running.summary() for t, running in per_threshold.items()
        },
    }
    best = max(
        report["by_threshold"].items(), key=lambda kv: kv[1]["tolerant_f1"]
    )
    report["best_threshold"] = float(best[0])
    report["best"] = best[1]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/crackseg9k/test"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = evaluate(args.checkpoint, args.data, args.limit, args.tolerance)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    target = args.out or args.checkpoint.parent / "evaluation.json"
    target.write_text(text, encoding="utf-8")
    print(f"\nGeschrieben: {target}")

    best = report["best"]
    print(
        f"\nBeste Schwelle {report['best_threshold']:.2f}: "
        f"IoU {best['iou']:.4f}, tolerantes F1 {best['tolerant_f1']:.4f}, "
        f"Breitenverzerrung {best['width_bias']:.3f}"
    )
    if best["width_bias"] > 1.2:
        print(
            "ACHTUNG: Das Netz malt Risse im Mittel deutlich zu breit. "
            "Gemeldete Breiten waeren zu gross - Schwelle anheben oder "
            "laenger trainieren."
        )
    return 0




if __name__ == "__main__":
    raise SystemExit(main())
