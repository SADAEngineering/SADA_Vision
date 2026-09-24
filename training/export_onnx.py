"""Vom Pruefpunkt zur Datei, die der Dienst laedt.

Der Dienst kennt kein PyTorch. Was ihn erreicht, sind zwei Dateien:

    models/crack_unet_r18.onnx    das Netz
    models/crack_unet_r18.json    Normierung, Kachelgroesse, Schwelle, Herkunft

Die Begleitdatei ist kein Beiwerk. Stehen dort andere Mittelwerte als beim
Training, rechnet der Dienst mit falsch normierten Bildern - und das faellt
nicht als Fehler auf, sondern als schlechte Erkennung. Deshalb wird sie hier
erzeugt und nicht von Hand geschrieben.

Geprueft wird der Export sofort: dasselbe Bild durch torch und durch
onnxruntime, und die Abweichung muss klein sein. Ein stummer Exportfehler
waere sonst erst im Betrieb zu sehen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from training.data import IMAGENET_MEAN, IMAGENET_STD


def export(
    checkpoint: Path,
    out_path: Path,
    tile: int | None = None,
    opset: int = 17,
    threshold: float | None = None,
) -> Path:
    from training.evaluate import load_model

    device = torch.device("cpu")
    model, config = load_model(checkpoint, device)
    size = tile or config.image_size

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, size, size, dtype=torch.float32)

    # Feste Kantenlaenge statt dynamischer Achsen: der Dienst kachelt ohnehin
    # auf genau diese Groesse, und ein statischer Graph laesst sich von
    # onnxruntime deutlich besser optimieren.
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["image"],
        output_names=["logits"],
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"Geschrieben: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    deviation = _verify(model, out_path, size)
    print(f"Groesste Abweichung torch gegen onnxruntime: {deviation:.2e}")
    if deviation > 1e-3:
        raise SystemExit(
            f"Export weicht zu stark ab ({deviation:.2e}). Nicht ausliefern."
        )

    sidecar = _sidecar(checkpoint, config, size, threshold)
    out_path.with_suffix(".json").write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Geschrieben: {out_path.with_suffix('.json')}")
    return out_path


def _verify(model, onnx_path: Path, size: int) -> float:
    import onnxruntime as ort

    rng = np.random.default_rng(7)
    sample = rng.standard_normal((1, 3, size, size)).astype(np.float32)

    with torch.no_grad():
        reference = model(torch.from_numpy(sample)).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = session.run(None, {session.get_inputs()[0].name: sample})[0]
    return float(np.max(np.abs(reference - got)))


def _sidecar(checkpoint: Path, config, size: int, threshold: float | None) -> dict:
    trained = {}
    run_meta = checkpoint.parent / "model.json"
    if run_meta.exists():
        trained = json.loads(run_meta.read_text(encoding="utf-8"))

    evaluation = {}
    eval_file = checkpoint.parent / "evaluation.json"
    if eval_file.exists():
        report = json.loads(eval_file.read_text(encoding="utf-8"))
        evaluation = report.get("best", {})
        if threshold is None:
            threshold = report.get("best_threshold")

    return {
        "name": config.name,
        "architecture": config.architecture,
        "encoder": config.encoder,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "tile": size,
        "tile_overlap": max(32, size // 8),
        "threshold": threshold if threshold is not None else config.threshold,
        "dataset": "CrackSeg9k (CC0 1.0), Kulkarni et al., arXiv:2208.13054",
        "checkpoint": str(checkpoint),
        "train_metrics": trained.get("metrics", {}),
        "test_metrics": evaluation,
        "note": _note(trained, evaluation),
    }


def _note(trained: dict, evaluation: dict) -> str:
    source = evaluation or trained.get("metrics", {})
    if not source:
        return "Trainiert auf CrackSeg9k."
    return (
        f"CrackSeg9k, tolerantes F1 {source.get('tolerant_f1', 0):.3f}, "
        f"IoU {source.get('iou', 0):.3f}, "
        f"Breitenverzerrung {source.get('width_bias', float('nan')):.2f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=Path("models/crack_unet_r18.onnx"))
    parser.add_argument("--tile", type=int)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()

    export(args.checkpoint, args.out, args.tile, args.opset, args.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
