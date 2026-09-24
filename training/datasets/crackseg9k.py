"""CrackSeg9k holen und in Bilder und Masken zerlegen.

CrackSeg9k ist die groesste konsolidierte Sammlung fuer Risssegmentierung:
9.255 Bilder zu 400x400 aus zehn Einzeldatensaetzen (Crack500, DeepCrack,
CrackTree, GAPs, Volker/Rissbilder, SDNET, Masonry, Ceramic - und *Noncrack*,
die rissfreien Gegenbeispiele, ohne die ein Netz jede Fuge fuer einen Riss
haelt).

    Kulkarni et al., CrackSeg9k, arXiv:2208.13054
    Harvard Dataverse, doi:10.7910/DVN/EGIEBY
    Lizenz: CC0 1.0 - Public Domain

**Die Lizenz ist der Grund fuer diese Wahl.** CC0 heisst: kein Copyleft, keine
Namensnennungspflicht, kein Risiko fuer ein Produkt, das bei Kunden auf deren
Blech laeuft. Datensaetze mit "nur fuer Forschung" waeren hier eine Zeitbombe.

Gespeichert wird als PNG-Paare, nicht als Parquet: das Original traegt die
Bilder base64-kodiert und braucht 4 GB, die entpackten PNGs ein Bruchteil -
und der Trainingslauf liest sie ohne jede Dekodierstufe.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import io
import json
import sys
from pathlib import Path

REPO = "rimvydasrub/crackseg9k"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main"
FILES = {"train": "data/train.parquet", "test": "data/test.parquet"}

# Bildnamen mit diesem Wortstamm stammen aus den rissfreien Teilmengen.
NONCRACK_HINTS = ("noncrack", "non_crack", "nocrack")


def download(url: str, target: Path, chunk: int = 1 << 20) -> Path:
    """Laedt mit Fortsetzung - 3 GB ueber eine Bueroleitung reissen sonst ab."""
    import requests

    target.parent.mkdir(parents=True, exist_ok=True)
    done = target.stat().st_size if target.exists() else 0

    head = requests.head(url, allow_redirects=True, timeout=60)
    total = int(head.headers.get("content-length", 0))
    if total and done == total:
        print(f"  schon da: {target.name} ({done / 1e6:.0f} MB)")
        return target

    headers = {"Range": f"bytes={done}-"} if done else {}
    with requests.get(url, stream=True, headers=headers, timeout=300) as response:
        response.raise_for_status()
        mode = "ab" if done else "wb"
        with target.open(mode) as handle:
            for block in response.iter_content(chunk_size=chunk):
                handle.write(block)
                done += len(block)
                if total:
                    pct = 100.0 * done / total
                    print(f"\r  {target.name}: {done / 1e6:7.0f} / {total / 1e6:.0f} MB "
                          f"({pct:5.1f} %)", end="", flush=True)
    print()
    return target


def _decode(value) -> bytes | None:
    """Die Spalten stehen base64-kodiert in Parquet, teils schon als Bytes."""
    if value is None:
        return None
    if isinstance(value, bytes):
        # Entweder schon PNG/JPEG oder base64 in Bytes.
        if value[:8] == b"\x89PNG\r\n\x1a\n" or value[:2] == b"\xff\xd8":
            return value
        value = value.decode("ascii", errors="ignore")
    if isinstance(value, str):
        try:
            return base64.b64decode(value, validate=False)
        except (binascii.Error, ValueError):
            return None
    return None


def convert(parquet: Path, out_dir: Path, split: str, limit: int = 0) -> dict:
    """Parquet -> ``<out>/<split>/images/*.png`` und ``.../masks/*.png``."""
    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image

    images_dir = out_dir / split / "images"
    masks_dir = out_dir / split / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    stats = {"written": 0, "skipped": 0, "noncrack": 0, "empty_mask": 0}
    parquet_file = pq.ParquetFile(parquet)
    index = 0

    for batch in parquet_file.iter_batches(batch_size=64):
        table = batch.to_pydict()
        names = table.get("head") or [None] * len(table["image"])
        for raw_image, raw_mask, raw_name in zip(
            table["image"], table["mask"], names, strict=True
        ):
            index += 1
            if limit and stats["written"] >= limit:
                return stats

            image_bytes = _decode(raw_image)
            mask_bytes = _decode(raw_mask)
            if not image_bytes or not mask_bytes:
                stats["skipped"] += 1
                continue

            try:
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
            except Exception:
                stats["skipped"] += 1
                continue

            if mask.size != image.size:
                mask = mask.resize(image.size, Image.NEAREST)

            # Masken kommen mit Zwischenwerten (JPEG-Artefakte an den Raendern).
            # Ein Riss ist binaer: Mitte der Skala als Schnitt.
            arr = np.asarray(mask)
            binary = (arr > 127).astype(np.uint8) * 255
            if binary.max() == 0:
                stats["empty_mask"] += 1

            name = _name_of(raw_name, image_bytes, index)
            if any(hint in name.lower() for hint in NONCRACK_HINTS):
                stats["noncrack"] += 1

            image_path = images_dir / f"{name}.png"
            mask_path = masks_dir / f"{name}.png"
            if image_path.exists() and mask_path.exists():
                # Fortsetzen: ein abgebrochener Lauf faengt nicht von vorn an.
                stats["written"] += 1
                continue

            # ``optimize=True`` sucht die beste Filterkombination und kostet
            # ein Vielfaches der Schreibzeit. Bei 9.000 Paaren sind das
            # Stunden - fuer Dateien, die ohnehin nur lokal liegen.
            image.save(image_path, compress_level=1)
            Image.fromarray(binary).save(mask_path, compress_level=1)
            stats["written"] += 1

            if stats["written"] % 250 == 0:
                print(f"\r  {split}: {stats['written']} Paare", end="", flush=True)

    print(f"\r  {split}: {stats['written']} Paare")
    return stats


def _name_of(raw_name, image_bytes: bytes, index: int) -> str:
    """Stabiler Dateiname - moeglichst der Originalname, sonst ein Hash.

    Die Spalte ``head`` traegt bei diesem Datensatz **keinen** Dateinamen,
    sondern Binaerdaten. Wer sie blind dekodiert, bekommt Dateinamen voller
    Steuerzeichen - unter Windows unbrauchbar und in jedem Protokoll
    unlesbar. Uebernommen wird sie deshalb nur, wenn dabei wirklich Text
    herauskommt.
    """
    decoded = _decode(raw_name)
    if decoded:
        text = decoded.decode("utf-8", errors="ignore").strip()
        text = Path(text).stem
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in text)
        stripped = safe.strip("_")
        # Mindestens vier verwertbare Zeichen, und nicht ueberwiegend
        # Ersatzstriche - sonst war es kein Name.
        if len(stripped) >= 4 and stripped.count("_") < len(stripped) / 2:
            return f"{index:06d}_{stripped[:60]}"

    digest = hashlib.sha1(image_bytes).hexdigest()[:10]
    return f"{index:06d}_{digest}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/crackseg9k"))
    parser.add_argument("--cache", type=Path, default=Path("data/_download"))
    parser.add_argument(
        "--limit", type=int, default=0, help="Nur so viele Paare je Teil (0 = alle)"
    )
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    summary = {}
    for split, remote in FILES.items():
        local = args.cache / Path(remote).name
        print(f"[{split}]")
        if not args.skip_download:
            download(f"{BASE_URL}/{remote}", local)
        if not local.exists():
            print(f"  fehlt: {local}", file=sys.stderr)
            return 1
        summary[split] = convert(local, args.out, split, args.limit)

    args.out.mkdir(parents=True, exist_ok=True)
    meta = {
        "source": REPO,
        "citation": "Kulkarni et al., CrackSeg9k, arXiv:2208.13054",
        "license": "CC0 1.0 Universal (Public Domain)",
        "splits": summary,
    }
    (args.out / "dataset.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
