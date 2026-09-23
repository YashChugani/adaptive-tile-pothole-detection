"""Zip the Component-1 output (images/ + labels/) into one archive for a ONE-TIME
upload to Google Drive, from which Colab copies it to local SSD before training.

The archive stores entries as `images/<name>.jpg` and `labels/<name>.txt`, so that
`extractall(<dataset_root>)` recreates the Ultralytics-friendly layout directly.

Usage:
    python src/detection/package_dataset.py            # -> data/rdd2022_india.zip
    python src/detection/package_dataset.py --src data/rdd2022_india --out data/rdd2022_india.zip
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def package(src_root: Path, out_zip: Path) -> None:
    images = sorted((src_root / "images").glob("*"))
    labels = sorted((src_root / "labels").glob("*.txt"))
    if not images or not labels:
        raise SystemExit(f"expected images/ and labels/ under {src_root}")

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in images:
            z.write(f, arcname=f"images/{f.name}")
            n += 1
        for f in labels:
            z.write(f, arcname=f"labels/{f.name}")
            n += 1

    size = out_zip.stat().st_size
    print(f"images={len(images)}  labels={len(labels)}  entries={n}")
    print(f"wrote {out_zip}  ({size:,} bytes = {size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Zip images/+labels/ for Drive upload.")
    parser.add_argument("--src", default="data/rdd2022_india", help="dataset root (has images/ and labels/)")
    parser.add_argument("--out", default="data/rdd2022_india.zip", help="output zip (gitignored under data/)")
    args = parser.parse_args()
    package(Path(args.src), Path(args.out))


if __name__ == "__main__":
    main()
