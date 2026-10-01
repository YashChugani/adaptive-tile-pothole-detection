"""Zip the severity crops (Component 6.3B) for a ONE-TIME upload to Google Drive, from which
Colab copies them to local SSD before training the severity CNN.

Stores entries as `crops/pothrgbd/{split}/{tier}/*.jpg` and `crops/pothole440/{tier}/*.jpg`,
so that `extractall(data/severity)` recreates the ImageFolder-friendly layout directly.

Usage:
    python src/severity/package_crops.py            # -> data/severity_crops.zip
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def package(src_root: Path, out_zip: Path) -> None:
    files = [p for p in src_root.rglob("*") if p.is_file()]
    if not files:
        raise SystemExit(f"no crops under {src_root} (run crops.py first)")
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in files:
            z.write(f, arcname=str(Path("crops") / f.relative_to(src_root)))
    size = out_zip.stat().st_size
    print(f"crops={len(files)}  wrote {out_zip}  ({size:,} bytes = {size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Zip severity crops for Drive upload.")
    parser.add_argument("--src", default="data/severity/crops", help="crops root")
    parser.add_argument("--out", default="data/severity_crops.zip", help="output zip (gitignored)")
    args = parser.parse_args()
    package(Path(args.src), Path(args.out))


if __name__ == "__main__":
    main()
