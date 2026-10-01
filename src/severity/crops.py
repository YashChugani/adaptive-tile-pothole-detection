"""Extract severity training crops from the frozen depth-tier labels (Component 6.3A).

PothRGBD: CLEAN subset only (snr_ratio>2), using the frozen manifest + splits.
Pothole440: all of it (cross-check). For each pothole: tight bbox from the mask, expand by
MARGIN, crop the RGB, save under data/severity/crops/..., and record a crops manifest
(stem, split, tier, tight box, image size, dbs_scalar, snr_ratio). CPU only, no CNN.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.severity.inspect_datasets import POTHOLE440, load_mask_png, pothrgbd_root, rasterize_yoloseg

MARGIN = 0.15            # expand bbox by this fraction of box size on each side
MIN_BOX_PX = 4           # boxes smaller than this (w or h) are degenerate -> skipped
CROPS = Path("data/severity/crops")
PREVIEW = Path("data/severity/_crops_preview")
LABELS_PR = Path("data/severity/labels_pothrgbd.csv")
LABELS_P4 = Path("data/severity/labels_pothole440.csv")


def mask_bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def expand_clip(box, w, h):
    x0, y0, x1, y1 = box
    mx, my = int(MARGIN * (x1 - x0)), int(MARGIN * (y1 - y0))
    return (max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my))


def _save_crop(rgb, box, out: Path):
    x0, y0, x1, y1 = box
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), rgb[y0:y1, x0:x1])


def extract_pothrgbd():
    root = pothrgbd_root()
    rows = [r for r in csv.DictReader(open(LABELS_PR))
            if r["tier"] and _is_clean(r)]
    out_rows, degenerate = [], 0
    for r in rows:
        stem = r["stem"]
        rgb = cv2.imread(str(root / "images" / (stem + ".jpg")))
        if rgb is None:
            degenerate += 1
            continue
        h, w = rgb.shape[:2]
        mask, _ = rasterize_yoloseg(root / "labels" / (stem + ".txt"), h, w)
        box = mask_bbox(mask)
        if box is None or (box[2] - box[0]) < MIN_BOX_PX or (box[3] - box[1]) < MIN_BOX_PX:
            degenerate += 1
            continue
        _save_crop(rgb, expand_clip(box, w, h),
                   CROPS / "pothrgbd" / r["split"] / r["tier"] / (stem + ".jpg"))
        out_rows.append({"stem": stem, "split": r["split"], "tier": r["tier"],
                         "bx0": box[0], "by0": box[1], "bx1": box[2], "by1": box[3],
                         "img_w": w, "img_h": h, "dbs_scalar": r["dbs_scalar"], "snr_ratio": r["snr_ratio"]})
    return out_rows, degenerate


def extract_pothole440():
    rows = [r for r in csv.DictReader(open(LABELS_P4)) if r["tier"]]
    out_rows, degenerate = [], 0
    for r in rows:
        stem, split = r["stem"], r["src_split"]
        base = POTHOLE440 / split
        name = stem + ".png"
        rgb = cv2.imread(str(base / "rgb" / name))
        mask = load_mask_png(base / "mask" / name)
        if rgb is None or mask is None:
            degenerate += 1
            continue
        h, w = rgb.shape[:2]
        box = mask_bbox(mask)
        if box is None or (box[2] - box[0]) < MIN_BOX_PX or (box[3] - box[1]) < MIN_BOX_PX:
            degenerate += 1
            continue
        _save_crop(rgb, expand_clip(box, w, h),
                   CROPS / "pothole440" / r["tier"] / (stem + ".jpg"))
        out_rows.append({"stem": stem, "src_split": split, "tier": r["tier"],
                         "bx0": box[0], "by0": box[1], "bx1": box[2], "by1": box[3],
                         "img_w": w, "img_h": h, "dbs_scalar": r["dbs_scalar"], "snr_ratio": ""})
    return out_rows, degenerate


def _is_clean(r):
    try:
        return float(r["snr_ratio"]) > 2
    except (ValueError, KeyError):
        return False


def _preview(rows, crop_dir_fn, out_prefix, n_each, splits_span=True):
    rng = random.Random(42)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    for tier in ("low", "medium", "high"):
        picks = [r for r in rows if r["tier"] == tier]
        rng.shuffle(picks)
        for r in picks[:n_each]:
            cp = crop_dir_fn(r)
            img = cv2.imread(str(cp))
            if img is None:
                continue
            img = cv2.resize(img, (256, 256))
            cv2.rectangle(img, (0, 0), (256, 28), (0, 0, 0), -1)
            cv2.putText(img, f"{tier} {r['stem'][:14]}", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imwrite(str(PREVIEW / f"{out_prefix}_{tier}_{r['stem'][:18]}.jpg"), img)


def main():
    pr_rows, pr_deg = extract_pothrgbd()
    p4_rows, p4_deg = extract_pothole440()

    with open("data/severity/crops_pothrgbd.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "split", "tier", "bx0", "by0", "bx1", "by1",
                                          "img_w", "img_h", "dbs_scalar", "snr_ratio"])
        w.writeheader(); w.writerows(pr_rows)
    with open("data/severity/crops_pothole440.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "src_split", "tier", "bx0", "by0", "bx1", "by1",
                                          "img_w", "img_h", "dbs_scalar", "snr_ratio"])
        w.writeheader(); w.writerows(p4_rows)

    def pr_cp(r):
        return CROPS / "pothrgbd" / r["split"] / r["tier"] / (r["stem"] + ".jpg")

    def p4_cp(r):
        return CROPS / "pothole440" / r["tier"] / (r["stem"] + ".jpg")

    _preview(pr_rows, pr_cp, "pothrgbd", 3)
    _preview(p4_rows, p4_cp, "pothole440", 2)

    print("=== PothRGBD crops (clean snr>2) ===")
    for sp in ("train", "val", "test"):
        cnt = {t: sum(1 for r in pr_rows if r["split"] == sp and r["tier"] == t) for t in ("low", "medium", "high")}
        print(f"  {sp}: {cnt}  total={sum(cnt.values())}")
    print(f"  degenerate skipped: {pr_deg}   grand total crops: {len(pr_rows)}")
    print("\n=== Pothole440 crops (all) ===")
    cnt4 = {t: sum(1 for r in p4_rows if r["tier"] == t) for t in ("low", "medium", "high")}
    print(f"  tiers: {cnt4}  total={len(p4_rows)}  degenerate skipped: {p4_deg}")
    print(f"\ncrops -> {CROPS}/ ; preview -> {PREVIEW}/ ; manifests -> data/severity/crops_*.csv")


if __name__ == "__main__":
    main()
