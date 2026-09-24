"""Datensatz und Verstaerkung.

Die Verstaerkung ist bewusst zurueckhaltend. Ein Riss ist ein paar Pixel
breit; was seine Form verbiegt oder seine Kanten verschmiert, lehrt das Netz
etwas Falsches - und zwar genau an der Stelle, an der spaeter gemessen wird.
Deshalb: spiegeln, drehen in rechten Winkeln, Helligkeit und Rauschen. Kein
elastisches Verzerren, kein starkes Weichzeichnen, kein Zuschneiden auf
Briefmarkengroesse.
"""

from __future__ import annotations

import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform(size: int) -> A.Compose:
    return A.Compose(
        [
            # Zufaelliger Ausschnitt statt Verkleinerung: die Aufloesung des
            # Risses bleibt die des Originals.
            A.PadIfNeeded(size, size, border_mode=cv2.BORDER_REFLECT_101),
            A.RandomCrop(size, size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.25, contrast_limit=0.25, p=0.7
            ),
            # Schattenwurf und ungleiche Ausleuchtung sind der Normalfall auf
            # der Baustelle, nicht die Ausnahme.
            A.RandomGamma(gamma_limit=(70, 140), p=0.3),
            A.GaussNoise(p=0.25),
            A.MotionBlur(blur_limit=3, p=0.15),
            A.ImageCompression(quality_range=(55, 100), p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def eval_transform(size: int) -> A.Compose:
    return A.Compose(
        [
            A.PadIfNeeded(size, size, border_mode=cv2.BORDER_REFLECT_101),
            A.CenterCrop(size, size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class CrackDataset(Dataset):
    """Bild-/Maskenpaare aus ``<root>/images`` und ``<root>/masks``."""

    def __init__(
        self,
        root: Path,
        transform: A.Compose,
        names: list[str] | None = None,
    ) -> None:
        self.images_dir = root / "images"
        self.masks_dir = root / "masks"
        if names is None:
            names = sorted(p.stem for p in self.images_dir.glob("*.png"))
        self.names = [n for n in names if (self.masks_dir / f"{n}.png").exists()]
        if not self.names:
            raise FileNotFoundError(f"Keine Paare unter {root}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        name = self.names[index]
        image = cv2.imread(str(self.images_dir / f"{name}.png"), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.masks_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        out = self.transform(image=image, mask=mask)
        image_t = np.transpose(out["image"], (2, 0, 1)).astype(np.float32)
        mask_t = out["mask"][None].astype(np.float32)
        return image_t, mask_t


def split_names(root: Path, val_fraction: float, seed: int = 42) -> tuple[list, list]:
    """Zufaellige, aber wiederholbare Aufteilung in Training und Pruefung.

    Die Trennung geht ueber *Dateien*, nicht ueber Ausschnitte: zwei Kacheln
    desselben Fotos duerfen nicht auf beiden Seiten landen, sonst misst die
    Pruefung das Auswendiglernen mit.
    """
    names = sorted(p.stem for p in (root / "images").glob("*.png"))
    rng = random.Random(seed)
    rng.shuffle(names)
    cut = int(len(names) * (1.0 - val_fraction))
    return names[:cut], names[cut:]
