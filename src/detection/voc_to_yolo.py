"""PASCAL VOC XML -> YOLO detection-label converter for RDD2022.

Component 1 of the build (dataset prep). This module ONLY converts one country
subset's VOC annotations into YOLO `.txt` labels, mirrors the images into an
Ultralytics-friendly layout, verifies the geometry, and reports class statistics.
It does NOT carve a train/val/test split, write a data.yaml, or train anything.

Design notes (see project.md 21 and CLAUDE.md):
- The class-index map is built from the classes ACTUALLY present in the XML files
  (not hardcoded), and printed. All classes are kept faithfully; detector-class
  scoping is a later component.
- Boxes are normalized using the image size from the XML <size> tag. Boxes that
  exceed image bounds are clipped and counted; malformed boxes are skipped and
  counted. Nothing crashes on bad input.
- Images are mirrored via hardlink (same volume, no admin) with a copy fallback,
  leaving the source (data/raw/) untouched.

Usage:
    python src/detection/voc_to_yolo.py --src data/raw/.../India --dst data/rdd2022_india
    python src/detection/voc_to_yolo.py --src ... --dst ... --inspect   # report only
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
POTHOLE_CLASS = "D40"  # RDD2022 pothole code; verified against actual data at runtime

# The 4 official RDD scored classes, remapped to a clean contiguous index.
# The detector trains on these only; the other codes present in the XML
# (D01, D0w0, D11, D43, D44, D50) are dropped box-wise (images are kept).
SCORED_CLASSES = ["D00", "D10", "D20", "D40"]
SCORED_MAP = {name: i for i, name in enumerate(SCORED_CLASSES)}  # D00:0 D10:1 D20:2 D40:3


def find_xml_files(src_root: Path) -> list[Path]:
    """All annotation XMLs under src_root (train split; test has no annotations)."""
    return sorted(src_root.rglob("*.xml"))


def index_images_by_stem(src_root: Path) -> dict[str, Path]:
    """Map image filename stem -> image path, for matching XMLs to their images."""
    index: dict[str, Path] = {}
    for ext in IMAGE_EXTS:
        for img in src_root.rglob("*" + ext):
            index.setdefault(img.stem, img)
    return index


def _object_names(xml_path: Path) -> list[str]:
    """Class names in one XML (best-effort; empty on parse failure)."""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return []
    names = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name:
            names.append(name)
    return names


def collect_classes(xml_files: list[Path]) -> dict[str, int]:
    """Deterministic {class_name: index} map from all classes present in the XMLs."""
    seen: set[str] = set()
    for xml_path in xml_files:
        seen.update(_object_names(xml_path))
    return {name: i for i, name in enumerate(sorted(seen))}


def count_classes(xml_files: list[Path]) -> Counter:
    """Full per-class instance counts over ALL classes present (provenance)."""
    counts: Counter[str] = Counter()
    for xml_path in xml_files:
        counts.update(_object_names(xml_path))
    return counts


def _read_size(root: ET.Element) -> tuple[int, int]:
    """Image (width, height) from the XML <size> tag; (0, 0) if missing/invalid."""
    size = root.find("size")
    if size is None:
        return 0, 0
    try:
        return int(float(size.findtext("width") or 0)), int(float(size.findtext("height") or 0))
    except ValueError:
        return 0, 0


def _read_box(obj: ET.Element) -> tuple[float, float, float, float] | None:
    """(xmin, ymin, xmax, ymax) from an <object>, or None if unparseable."""
    box = obj.find("bndbox")
    if box is None:
        return None
    try:
        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))
    except (TypeError, ValueError):
        return None
    return xmin, ymin, xmax, ymax


def _mirror_image(src_img: Path, dst_img: Path) -> None:
    """Hardlink src into dst (same volume, no admin); fall back to a copy."""
    if dst_img.exists():
        return
    try:
        os.link(src_img, dst_img)
    except OSError:
        shutil.copy2(src_img, dst_img)


def convert(src_root: Path, dst_root: Path, class_map: dict[str, int]) -> dict:
    """Convert VOC XML under src_root into YOLO labels + mirrored images under dst_root.

    Returns a stats dict with per-class instance counts and box-quality counters.
    """
    images_dir = dst_root / "images"
    labels_dir = dst_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_index = index_images_by_stem(src_root)
    xml_files = find_xml_files(src_root)

    class_counts: Counter[str] = Counter()
    images_with_pothole = 0
    clipped_boxes = 0
    dropped_boxes = 0           # valid box of a class not in class_map (out of scope)
    skipped_boxes = 0            # malformed geometry / unparseable box
    skipped_size = 0            # XML missing a usable <size>
    xml_without_image = 0
    images_written = 0
    labels_written = 0

    for xml_path in xml_files:
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            skipped_size += 1
            continue

        img_path = image_index.get(xml_path.stem)
        if img_path is None:
            xml_without_image += 1
            continue

        width, height = _read_size(root)
        if width <= 0 or height <= 0:
            skipped_size += 1
            continue

        lines: list[str] = []
        has_pothole = False
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            if not name:
                skipped_boxes += 1
                continue
            if name not in class_map:
                dropped_boxes += 1  # e.g. non-scored code when class_map is SCORED_MAP
                continue
            coords = _read_box(obj)
            if coords is None:
                skipped_boxes += 1
                continue
            xmin, ymin, xmax, ymax = coords

            # Clip to image bounds; count if anything was actually out of bounds.
            cxmin = min(max(xmin, 0.0), width)
            cymin = min(max(ymin, 0.0), height)
            cxmax = min(max(xmax, 0.0), width)
            cymax = min(max(ymax, 0.0), height)
            if (cxmin, cymin, cxmax, cymax) != (xmin, ymin, xmax, ymax):
                clipped_boxes += 1

            if cxmax <= cxmin or cymax <= cymin:
                skipped_boxes += 1
                continue

            cx = (cxmin + cxmax) / 2.0 / width
            cy = (cymin + cymax) / 2.0 / height
            bw = (cxmax - cxmin) / width
            bh = (cymax - cymin) / height
            lines.append(f"{class_map[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            class_counts[name] += 1
            if name == POTHOLE_CLASS:
                has_pothole = True

        # Write one label file per image (empty file = valid YOLO background).
        (labels_dir / (xml_path.stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
        labels_written += 1
        _mirror_image(img_path, images_dir / img_path.name)
        images_written += 1
        if has_pothole:
            images_with_pothole += 1

    return {
        "class_counts": class_counts,
        "images_with_pothole": images_with_pothole,
        "clipped_boxes": clipped_boxes,
        "dropped_boxes": dropped_boxes,
        "skipped_boxes": skipped_boxes,
        "skipped_size": skipped_size,
        "xml_without_image": xml_without_image,
        "images_written": images_written,
        "labels_written": labels_written,
        "num_xml": len(xml_files),
    }


def draw_verification(dst_root: Path, class_map: dict[str, int], target: str, count: int) -> list[Path]:
    """Round-trip check: draw the WRITTEN YOLO labels back onto their images."""
    index_to_name = {i: n for n, i in class_map.items()}
    verify_dir = dst_root / "_verify"
    verify_dir.mkdir(parents=True, exist_ok=True)
    images_dir = dst_root / "images"
    labels_dir = dst_root / "labels"
    target_idx = class_map.get(target)

    saved: list[Path] = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        if len(saved) >= count:
            break
        rows = [r.split() for r in label_path.read_text(encoding="utf-8").splitlines() if r.strip()]
        if target_idx is not None and not any(int(r[0]) == target_idx for r in rows):
            continue
        img_path = next((images_dir / (label_path.stem + e) for e in IMAGE_EXTS
                         if (images_dir / (label_path.stem + e)).exists()), None)
        if img_path is None:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        for r in rows:
            cls = int(r[0])
            cx, cy, bw, bh = (float(v) for v in r[1:5])
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            color = (0, 0, 255) if cls == target_idx else (0, 200, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, index_to_name.get(cls, str(cls)), (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        out = verify_dir / (label_path.stem + "_verify.jpg")
        cv2.imwrite(str(out), img)
        saved.append(out)
    return saved


def _print_report(stats: dict, class_map: dict[str, int]) -> None:
    counts = stats["class_counts"]
    print("\n=== Class instance counts (whole India set) ===")
    print(f"{'class':<8}{'index':<7}{'instances':>10}")
    for name in sorted(class_map, key=lambda n: class_map[n]):
        print(f"{name:<8}{class_map[name]:<7}{counts.get(name, 0):>10}")
    print(f"{'TOTAL':<15}{sum(counts.values()):>10}")
    print("\n=== Box quality ===")
    print(f"XML files scanned        : {stats['num_xml']}")
    print(f"labels written           : {stats['labels_written']}")
    print(f"images mirrored          : {stats['images_written']}")
    print(f"clipped boxes            : {stats['clipped_boxes']}")
    print(f"dropped boxes (off-scope): {stats['dropped_boxes']}")
    print(f"skipped boxes (malformed): {stats['skipped_boxes']}")
    print(f"skipped (no usable size) : {stats['skipped_size']}")
    print(f"XML without image        : {stats['xml_without_image']}")
    print("\n=== GATE ===")
    print(f"images with >=1 {POTHOLE_CLASS}     : {stats['images_with_pothole']}")
    print(f"total {POTHOLE_CLASS} instances     : {counts.get(POTHOLE_CLASS, 0)}  <-- GATE number")


def main() -> None:
    parser = argparse.ArgumentParser(description="RDD2022 VOC XML -> YOLO converter (one country subset).")
    parser.add_argument("--src", required=True, help="country root, e.g. data/raw/.../RDD2022/India")
    parser.add_argument("--dst", required=True, help="output root, e.g. data/rdd2022_india")
    parser.add_argument("--inspect", action="store_true", help="report classes/counts only; no conversion")
    parser.add_argument("--verify-count", type=int, default=5, help="number of verification images to draw")
    parser.add_argument("--all-classes", action="store_true",
                        help="export ALL discovered classes instead of the 4 scored classes (default: scored only)")
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    if not src_root.exists():
        raise SystemExit(f"source not found: {src_root}")

    xml_files = find_xml_files(src_root)
    class_map = collect_classes(xml_files)               # full discovered map (provenance)
    full_counts = count_classes(xml_files)               # full per-class counts (provenance)
    print(f"Found {len(xml_files)} XML files under {src_root}")
    print(f"Discovered class-index map (from actual data): {class_map}")
    if POTHOLE_CLASS in class_map:
        print(f"Confirmed pothole class {POTHOLE_CLASS!r} present at index {class_map[POTHOLE_CLASS]}.")
    else:
        print(f"WARNING: pothole class {POTHOLE_CLASS!r} NOT found in this subset.")

    if args.inspect:
        return

    # Output label set: the 4 scored classes by default, or all discovered with --all-classes.
    out_map = class_map if args.all_classes else SCORED_MAP
    print(f"\nExporting with {'ALL discovered' if args.all_classes else 'SCORED (4-class)'} "
          f"map -> {out_map}")

    stats = convert(src_root, dst_root, out_map)
    _print_report(stats, out_map)

    # Before/after: full 10-class (provenance) vs exported (scored) counts.
    print("\n=== Before -> after (discovered vs exported) ===")
    print(f"{'class':<8}{'discovered':>12}{'exported':>10}")
    for name in sorted(class_map, key=lambda n: class_map[n]):
        exported = stats["class_counts"].get(name, 0) if name in out_map else 0
        print(f"{name:<8}{full_counts.get(name, 0):>12}{exported:>10}")
    print(f"{'TOTAL':<8}{sum(full_counts.values()):>12}{sum(stats['class_counts'].values()):>10}")

    # Invariant: a scored class is never dropped, so its exported count == discovered count.
    d40_exported = stats["class_counts"].get(POTHOLE_CLASS, 0)
    d40_full = full_counts.get(POTHOLE_CLASS, 0)
    assert d40_exported == d40_full, f"{POTHOLE_CLASS} changed: {d40_exported} != {d40_full}"
    print(f"\n{POTHOLE_CLASS} unchanged: exported={d40_exported} == discovered={d40_full}  (index {out_map.get(POTHOLE_CLASS)})")

    # Empty labels after this export (images whose kept-class box count is 0).
    empty_labels = sum(1 for p in (dst_root / "labels").glob("*.txt") if p.stat().st_size == 0)
    print(f"empty label files (no kept-class boxes): {empty_labels}")

    # Provenance sidecar (retains the full 10-class stats; under data/, gitignored).
    sidecar = dst_root / "class_stats.json"
    sidecar.write_text(json.dumps({
        "source": str(src_root),
        "num_xml": stats["num_xml"],
        "images": stats["images_written"],
        "labels": stats["labels_written"],
        "empty_labels": empty_labels,
        "discovered_class_map": class_map,
        "discovered_class_counts": dict(full_counts),
        "discovered_total_instances": sum(full_counts.values()),
        "scored_class_map": SCORED_MAP,
        "scored_class_counts": {n: stats["class_counts"].get(n, 0) for n in SCORED_CLASSES},
        "scored_total_instances": sum(stats["class_counts"].values()),
        "dropped_boxes": stats["dropped_boxes"],
        "images_with_pothole": stats["images_with_pothole"],
        "d40_instances": d40_full,
    }, indent=2), encoding="utf-8")
    print(f"wrote provenance sidecar: {sidecar}")

    saved = draw_verification(dst_root, out_map, POTHOLE_CLASS, args.verify_count)
    print(f"\nWrote {len(saved)} verification images to {dst_root / '_verify'}:")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
