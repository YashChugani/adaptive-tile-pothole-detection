"""Size-stratified D40 evaluator on the FROZEN test split.

This is the reference scorer for the tiling ablation. Its scoring core is
**prediction-source independent**: `score_d40()` takes a dict of predicted D40 boxes
per image and a dict of GT D40 boxes (with frozen size buckets) per image, and computes
per-bucket recall/precision with OUR OWN greedy matcher. The baseline path feeds it
plain `best.pt` predictions; the later tiling path feeds it SAHI-merged predictions —
both scored by the identical matcher, which is the whole point.

Buckets are the frozen COCO-absolute definition from Component 2B, computed on the
imgsz=640 letterboxed input (r=640/720, no padding since all images are 720x720).
Matching (IoU) is done at native pixel resolution; bucket assignment uses the 640-scaled
GT area so it matches the frozen manifest exactly.

Run on Colab against best.pt:
    python src/detection/eval_stratified.py --weights /content/drive/MyDrive/pothole/checkpoints/baseline/weights/best.pt
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---- fixed scoring constants (reused verbatim by the tiling path later) ----
D40_IDX = 3
IMGSZ = 640
ORIG = 720
R = IMGSZ / ORIG          # 0.8889 letterbox scale (uniform; no padding, square images)
CONF_AP = 0.001           # low conf for AP / recall-ceiling
CONF_OP = 0.25            # deployment operating point
NMS_IOU = 0.7             # NMS IoU used at inference (explicit; not the Ultralytics default)
MATCH_IOU = 0.5           # our pred<->GT matching threshold

TEST_LIST = Path("configs/splits/test.txt")
META = Path("configs/splits/meta.json")
OUT_DIR = Path("data/eval")   # gitignored


def _coco_bucket(area640: float, small: float, medium: float) -> str:
    if area640 < small:
        return "small"
    if area640 <= medium:
        return "medium"
    return "large"


def load_thresholds() -> tuple[float, float, int]:
    """COCO small/medium thresholds + the frozen small-D40 test count, from Component 2B."""
    meta = json.loads(META.read_text())
    small = float(meta["coco_small"])
    medium = float(meta["coco_medium"])
    frozen_test_small = int(meta["primary_split"]["test"]["small"])
    assert small == 32 ** 2 and medium == 96 ** 2, f"unexpected COCO thresholds: {small},{medium}"
    return small, medium, frozen_test_small


def load_gt_d40(stems: list[str], small: float, medium: float) -> dict[str, list[tuple]]:
    """GT D40 boxes per image as (x1,y1,x2,y2,bucket) at native pixels; bucket by 640 area."""
    gt: dict[str, list[tuple]] = {}
    for stem in stems:
        lab = Path(f"data/rdd2022_india/labels/{stem}.txt")
        boxes = []
        if lab.exists() and lab.stat().st_size:
            for line in lab.read_text().splitlines():
                c, cx, cy, bw, bh = line.split()
                if int(c) != D40_IDX:
                    continue
                cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)
                x1 = (cx - bw / 2) * ORIG
                y1 = (cy - bh / 2) * ORIG
                x2 = (cx + bw / 2) * ORIG
                y2 = (cy + bh / 2) * ORIG
                area640 = bw * IMGSZ * bh * IMGSZ   # == bw*ORIG*R * bh*ORIG*R
                boxes.append((x1, y1, x2, y2, _coco_bucket(area640, small, medium)))
        gt[stem] = boxes
    return gt


def iou_xyxy(a: tuple, b: tuple) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _pred_bucket(x1, y1, x2, y2, small, medium) -> str:
    area640 = (x2 - x1) * R * (y2 - y1) * R
    return _coco_bucket(area640, small, medium)


def _match_one_conf(preds_by_image, gt_by_image, conf, small, medium):
    """Greedy match (desc conf, IoU>=MATCH_IOU, one pred per GT). Returns per-bucket
    tp/fp/gt. TP is credited to the matched GT's bucket; FP to the pred's own size bucket."""
    buckets = ("small", "medium", "large")
    gt_count = {b: 0 for b in buckets}
    tp = {b: 0 for b in buckets}
    fp = {b: 0 for b in buckets}
    for stem, gts in gt_by_image.items():
        for *_box, b in gts:
            gt_count[b] += 1
    for stem, gts in gt_by_image.items():
        preds = [p for p in preds_by_image.get(stem, []) if p[4] >= conf]
        preds.sort(key=lambda p: p[4], reverse=True)
        matched = [False] * len(gts)
        for px1, py1, px2, py2, pc in preds:
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gts):
                if matched[j]:
                    continue
                v = iou_xyxy((px1, py1, px2, py2), g[:4])
                if v >= MATCH_IOU and v > best_iou:
                    best_iou, best_j = v, j
            if best_j >= 0:
                matched[best_j] = True
                tp[gts[best_j][4]] += 1
            else:
                fp[_pred_bucket(px1, py1, px2, py2, small, medium)] += 1
    return gt_count, tp, fp


def score_d40(preds_by_image, gt_by_image, small, medium,
              conf_ap=CONF_AP, conf_op=CONF_OP):
    """Per-bucket table + overall recall/AP. Prediction-source independent."""
    gt_count, tp_ap, fp_ap = _match_one_conf(preds_by_image, gt_by_image, conf_ap, small, medium)
    _, tp_op, fp_op = _match_one_conf(preds_by_image, gt_by_image, conf_op, small, medium)

    rows = []
    for b in ("small", "medium", "large"):
        rec_ap = tp_ap[b] / gt_count[b] if gt_count[b] else float("nan")
        rec_op = tp_op[b] / gt_count[b] if gt_count[b] else float("nan")
        prec_op = tp_op[b] / (tp_op[b] + fp_op[b]) if (tp_op[b] + fp_op[b]) else float("nan")
        rows.append({"bucket": b, "gt": gt_count[b],
                     "recall@%.3f" % conf_ap: rec_ap,
                     "recall@%.2f" % conf_op: rec_op,
                     "precision@%.2f" % conf_op: prec_op})
    total_gt = sum(gt_count.values())
    total_tp_ap = sum(tp_ap.values())
    overall_recall_ap = total_tp_ap / total_gt if total_gt else float("nan")
    ap50 = _ap50(preds_by_image, gt_by_image, conf_ap)
    return rows, {"overall_recall@%.3f" % conf_ap: overall_recall_ap,
                  "overall_AP@50": ap50, "total_gt": total_gt}


def _ap50(preds_by_image, gt_by_image, conf):
    """Overall D40 AP@50 (COCO 101-point interpolation) as a cross-check vs Ultralytics."""
    total_gt = sum(len(v) for v in gt_by_image.values())
    dets = []  # (conf, is_tp)
    for stem, gts in gt_by_image.items():
        preds = [p for p in preds_by_image.get(stem, []) if p[4] >= conf]
        preds.sort(key=lambda p: p[4], reverse=True)
        matched = [False] * len(gts)
        for px1, py1, px2, py2, pc in preds:
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gts):
                if matched[j]:
                    continue
                v = iou_xyxy((px1, py1, px2, py2), g[:4])
                if v >= MATCH_IOU and v > best_iou:
                    best_iou, best_j = v, j
            if best_j >= 0:
                matched[best_j] = True
                dets.append((pc, 1))
            else:
                dets.append((pc, 0))
    if total_gt == 0 or not dets:
        return float("nan")
    dets.sort(key=lambda d: d[0], reverse=True)
    tps = np.array([d[1] for d in dets])
    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(1 - tps)
    recall = tp_cum / total_gt
    precision = tp_cum / (tp_cum + fp_cum)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = precision[recall >= t].max() if np.any(recall >= t) else 0.0
        ap += p / 101
    return float(ap)


def predict_baseline(weights: str, stems: list[str], conf: float, iou: float, imgsz: int):
    """Plain best.pt inference -> D40 preds per image at native pixels. (Runs on Colab GPU.)"""
    from ultralytics import YOLO
    model = YOLO(weights)
    preds_by_image: dict[str, list[tuple]] = defaultdict(list)
    for stem in stems:
        img = f"data/rdd2022_india/images/{stem}.jpg"
        r = model.predict(img, conf=conf, iou=iou, imgsz=imgsz, device=0, verbose=False)[0]
        for b in r.boxes:
            if int(b.cls) == D40_IDX:
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                preds_by_image[stem].append((x1, y1, x2, y2, float(b.conf)))
    return preds_by_image


def print_and_save(rows, summary, out_stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    width = {c: max(len(c), 10) for c in cols}
    print("\n=== D40 size-stratified (BASELINE row of the ablation) ===")
    print("  ".join(c.ljust(width[c]) for c in cols))
    for row in rows:
        print("  ".join((f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c])).ljust(width[c]) for c in cols))
    print("\nsummary:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()})
    (OUT_DIR / f"{out_stem}.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))
    with (OUT_DIR / f"{out_stem}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"saved -> {OUT_DIR / (out_stem + '.json')} and .csv")


def main() -> None:
    ap = argparse.ArgumentParser(description="Size-stratified D40 evaluator on the frozen test split.")
    ap.add_argument("--weights", required=True, help="path to best.pt (e.g. on Drive)")
    ap.add_argument("--conf-ap", type=float, default=CONF_AP)
    ap.add_argument("--conf-op", type=float, default=CONF_OP)
    ap.add_argument("--iou", type=float, default=NMS_IOU, help="NMS IoU at inference")
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--out", default="baseline_stratified")
    ap.add_argument("--ultra-ref", type=float, default=0.404, help="Ultralytics D40 mAP@50 to cross-check")
    args = ap.parse_args()

    small, medium, frozen_small = load_thresholds()
    stems = [Path(l).stem for l in TEST_LIST.read_text().splitlines() if l.strip()]
    gt = load_gt_d40(stems, small, medium)

    recomputed_small = sum(1 for v in gt.values() for *_b, bk in v if bk == "small")
    assert recomputed_small == frozen_small == 67, \
        f"small-D40 mismatch: recomputed={recomputed_small} frozen={frozen_small}"
    print(f"test images={len(stems)}  D40 GT={sum(len(v) for v in gt.values())}  "
          f"small={recomputed_small} (matches frozen manifest)")

    preds = predict_baseline(args.weights, stems, args.conf_ap, args.iou, args.imgsz)
    rows, summary = score_d40(preds, gt, small, medium, args.conf_ap, args.conf_op)
    print_and_save(rows, summary, args.out)

    delta = summary["overall_AP@50"] - args.ultra_ref
    print(f"\ncross-check: our overall D40 AP@50={summary['overall_AP@50']:.4f} vs "
          f"Ultralytics {args.ultra_ref} -> delta={delta:+.4f} "
          f"(differences expected from our fixed conf/iou + custom greedy matching)")


if __name__ == "__main__":
    main()
