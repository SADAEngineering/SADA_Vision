"""U-Net auf Risse trainieren.

Warum U-Net und nicht YOLO: Risse sind duenn, lang, verzweigt und laufen quer
durchs ganze Bild. Ein Kasten um sie herum sagt fast nichts, und die
Maskenkoepfe der YOLO-Segmentierer arbeiten intern auf grobem Raster - fuer
eine Breite, die bei 0,2 gegen 0,3 mm entscheidet, ist das zu grob. Ein
U-Net segmentiert in voller Aufloesung; die Instanzen entstehen danach aus
Zusammenhangskomponenten, was bei Rissen ohnehin die richtige Zerlegung ist.

Dazu kommt die Lizenz: ``segmentation_models_pytorch`` steht unter MIT, die
Encoder-Gewichte kommen aus torchvision (BSD). Nichts davon zwingt ein
Produkt zur Offenlegung.

Aufruf (CPU):

    python training/train.py --config training/configs/crack_unet_r18.yaml

Ein unterbrochener Lauf wird fortgesetzt:

    python training/train.py --config ... --resume runs/<name>/last.pt
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from training.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    CrackDataset,
    eval_transform,
    split_names,
    train_transform,
)
from training.metrics import Running

_stop_requested = False


def _request_stop(signum, frame) -> None:  # noqa: ARG001
    """SIGTERM sauber behandeln: laufende Epoche zu Ende, dann sichern.

    Ein Lauf ueber Stunden wird irgendwann abgebrochen - vom Betreiber, vom
    Neustart, vom Docker-Stop. Ohne das hier waere die Arbeit weg.
    """
    global _stop_requested
    _stop_requested = True
    print("\n[Abbruch angefordert - Epoche wird beendet und gesichert]", flush=True)


@dataclass
class Config:
    name: str = "crack_unet_r18"
    data_root: str = "data/crackseg9k/train"
    test_root: str = "data/crackseg9k/test"
    out_dir: str = "runs"

    architecture: str = "unet"
    encoder: str = "resnet18"
    encoder_weights: str = "imagenet"

    image_size: int = 256
    batch_size: int = 16
    epochs: int = 12
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    val_fraction: float = 0.1
    workers: int = 4
    threads: int = 0
    seed: int = 42

    # Risse machen wenige Prozent der Pixel aus. Ohne Gegengewicht lernt das
    # Netz "ueberall Hintergrund" und hat damit 97 Prozent Trefferquote.
    pos_weight: float = 4.0
    dice_weight: float = 1.0
    bce_weight: float = 1.0

    limit_train: int = 0
    limit_val: int = 0
    tolerance_px: int = 2
    threshold: float = 0.5
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None, overrides: dict) -> Config:
        values: dict = {}
        if path is not None:
            values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        values.update({k: v for k, v in overrides.items() if v is not None})
        known = {f for f in cls.__dataclass_fields__}
        extra = {k: v for k, v in values.items() if k not in known}
        return cls(**{k: v for k, v in values.items() if k in known}, extra=extra)


def build_model(config: Config) -> torch.nn.Module:
    import segmentation_models_pytorch as smp

    factory = {
        "unet": smp.Unet,
        "unetplusplus": smp.UnetPlusPlus,
        "deeplabv3plus": smp.DeepLabV3Plus,
    }[config.architecture.lower()]
    return factory(
        encoder_name=config.encoder,
        encoder_weights=config.encoder_weights or None,
        in_channels=3,
        classes=1,
    )


def build_loss(config: Config, device: torch.device):
    import segmentation_models_pytorch as smp

    bce = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([config.pos_weight], device=device)
    )
    dice = smp.losses.DiceLoss(mode="binary", from_logits=True)

    def loss_fn(logits, target):
        return config.bce_weight * bce(logits, target) + config.dice_weight * dice(
            logits, target
        )

    return loss_fn


@torch.no_grad()
def validate(model, loader, device, config: Config) -> dict:
    model.eval()
    running = Running(tolerance=config.tolerance_px)
    total_loss, batches = 0.0, 0
    loss_fn = build_loss(config, device)

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        total_loss += float(loss_fn(logits, masks).detach())
        batches += 1

        pred = (torch.sigmoid(logits) >= config.threshold).cpu().numpy()[:, 0]
        truth = masks.cpu().numpy()[:, 0] > 0.5
        for p, t in zip(pred, truth, strict=True):
            running.update(p, t)

    out = running.summary()
    out["loss"] = total_loss / max(batches, 1)
    return out


def train(config: Config, resume: Path | None) -> Path:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if config.threads > 0:
        torch.set_num_threads(config.threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Geraet: {device} | Threads: {torch.get_num_threads()}")

    root = Path(config.data_root)
    train_names, val_names = split_names(root, config.val_fraction, config.seed)
    if config.limit_train:
        train_names = train_names[: config.limit_train]
    if config.limit_val:
        val_names = val_names[: config.limit_val]

    train_set = CrackDataset(root, train_transform(config.image_size), train_names)
    val_set = CrackDataset(root, eval_transform(config.image_size), val_names)
    print(f"Training: {len(train_set)} Bilder | Pruefung: {len(val_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=config.workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        persistent_workers=config.workers > 0,
    )

    model = build_model(config).to(device)
    loss_fn = build_loss(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.learning_rate * 0.05
    )

    out_dir = Path(config.out_dir) / config.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(
        yaml.safe_dump(config.__dict__, allow_unicode=True), encoding="utf-8"
    )

    start_epoch, best = 0, -1.0
    history: list[dict] = []
    if resume and resume.exists():
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        best = state.get("best", -1.0)
        history = state.get("history", [])
        print(f"Fortgesetzt bei Epoche {start_epoch} (bestes tolerant_f1 {best:.4f})")

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    for epoch in range(start_epoch, config.epochs):
        model.train()
        started = time.time()
        total, seen = 0.0, 0

        for step, (images, masks) in enumerate(train_loader, start=1):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), masks)
            loss.backward()
            optimizer.step()

            total += float(loss.detach())
            seen += 1
            if step % 20 == 0:
                rate = (time.time() - started) / step
                left = (len(train_loader) - step) * rate
                print(
                    f"  Epoche {epoch + 1}/{config.epochs} "
                    f"Schritt {step}/{len(train_loader)} "
                    f"Verlust {total / seen:.4f} "
                    f"({rate:.2f} s/Schritt, noch ~{left / 60:.0f} min)",
                    flush=True,
                )

        scheduler.step()
        metrics = validate(model, val_loader, device, config)
        metrics.update(
            {
                "epoch": epoch + 1,
                "train_loss": total / max(seen, 1),
                "lr": scheduler.get_last_lr()[0],
                "minutes": (time.time() - started) / 60.0,
            }
        )
        history.append(metrics)
        print(
            f"Epoche {epoch + 1}: Verlust {metrics['train_loss']:.4f} | "
            f"IoU {metrics['iou']:.4f} | F1 {metrics['f1']:.4f} | "
            f"tolerantes F1 {metrics['tolerant_f1']:.4f} | "
            f"Breitenverzerrung {metrics['width_bias']:.3f} | "
            f"{metrics['minutes']:.1f} min",
            flush=True,
        )

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best": best,
            "history": history,
            "config": config.__dict__,
        }
        torch.save(state, out_dir / "last.pt")
        (out_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

        # Gewaehlt wird nach dem toleranten F1, nicht nach IoU: gefragt ist,
        # ob der Riss an der richtigen Stelle gefunden wurde.
        if metrics["tolerant_f1"] > best:
            best = metrics["tolerant_f1"]
            state["best"] = best
            torch.save(state, out_dir / "best.pt")
            _write_sidecar(out_dir, config, metrics)
            print(f"  -> neuer Bestwert ({best:.4f}), gesichert", flush=True)

        if _stop_requested:
            print("Abbruch nach Sicherung.", flush=True)
            break

    return out_dir / "best.pt"


def _write_sidecar(out_dir: Path, config: Config, metrics: dict) -> None:
    """Begleitdatei, die spaeter neben das ONNX gelegt wird."""
    (out_dir / "model.json").write_text(
        json.dumps(
            {
                "name": config.name,
                "architecture": config.architecture,
                "encoder": config.encoder,
                "mean": list(IMAGENET_MEAN),
                "std": list(IMAGENET_STD),
                "tile": config.image_size,
                "tile_overlap": max(32, config.image_size // 8),
                "threshold": config.threshold,
                "dataset": "CrackSeg9k (CC0 1.0)",
                "metrics": {
                    k: (None if isinstance(v, float) and np.isnan(v) else v)
                    for k, v in metrics.items()
                },
                "note": f"CrackSeg9k, tolerantes F1 {metrics['tolerant_f1']:.3f}",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, dest="batch_size")
    parser.add_argument("--image-size", type=int, dest="image_size")
    parser.add_argument("--encoder")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--limit-train", type=int, dest="limit_train")
    parser.add_argument("--limit-val", type=int, dest="limit_val")
    parser.add_argument("--name")
    args = parser.parse_args()

    overrides = {
        k: v for k, v in vars(args).items() if k not in {"config", "resume"}
    }
    config = Config.load(args.config, overrides)
    if config.threads <= 0:
        config.threads = max(1, (os.cpu_count() or 4))

    best = train(config, args.resume)
    print(f"Fertig. Bestes Modell: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
