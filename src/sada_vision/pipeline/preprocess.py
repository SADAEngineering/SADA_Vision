"""Bild hereinnehmen: dekodieren, drehen, bei Bedarf verkleinern."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from ..config import get_settings
from ..domain import SourceImage


class ImageRejected(ValueError):
    """Das Bild ist unbrauchbar - mit Grund fuer den Aufrufer."""


def decode(data: bytes) -> SourceImage:
    """Bytes -> RGB-Feld, aufrecht gedreht.

    Handyfotos liegen fast immer quer im Datenstrom und stehen nur durch das
    EXIF-Feld ``Orientation`` richtig. Wer das ueberspringt, misst senkrechte
    Risse als waagerechte.
    """
    settings = get_settings()
    if not data:
        raise ImageRejected("Leerer Upload.")
    if len(data) > settings.max_upload_bytes:
        raise ImageRejected(
            f"Bild groesser als erlaubt ({len(data)} > {settings.max_upload_bytes} Bytes)."
        )

    # Zip-Bomben-Schutz: PIL rechnet die Pixelzahl vor dem Dekodieren aus.
    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = settings.max_image_pixels
    try:
        with Image.open(io.BytesIO(data)) as img:
            original_size = img.size
            img = ImageOps.exif_transpose(img)
            rotated = img.size != original_size
            rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    except UnidentifiedImageError as exc:
        raise ImageRejected("Format nicht erkannt.") from exc
    except Image.DecompressionBombError as exc:
        raise ImageRejected("Bild zu gross.") from exc
    except OSError as exc:
        raise ImageRejected(f"Bild nicht lesbar: {exc}") from exc
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit

    h, w = rgb.shape[:2]
    if h < 32 or w < 32:
        raise ImageRejected(f"Bild zu klein ({w}x{h}).")

    return SourceImage(rgb=rgb, width=w, height=h, exif_rotated=rotated)


def working_image(source: SourceImage) -> tuple[np.ndarray, float]:
    """Arbeitsbild und der Faktor, mit dem zurueckgerechnet wird.

    Verkleinert wird nur, wenn ein Foto ueber die eingestellte Kante geht.
    Der Faktor ist ``original / arbeit``: Koordinaten und Breiten aus dem
    Arbeitsbild werden damit multipliziert.
    """
    settings = get_settings()
    limit = settings.max_edge_px
    long_edge = max(source.height, source.width)
    if limit <= 0 or long_edge <= limit:
        return source.rgb, 1.0

    import cv2

    ratio = limit / long_edge
    new_w = max(32, int(round(source.width * ratio)))
    new_h = max(32, int(round(source.height * ratio)))
    # AREA verliert duenne dunkle Linien am wenigsten.
    small = cv2.resize(source.rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return small, source.width / new_w
