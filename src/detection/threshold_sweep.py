"""D40 operating-threshold analysis (no retraining).

Reuses eval_stratified.py's scoring core (same matcher, MATCH_IOU, bucket defs), so the
numbers are directly comparable to the ablation. Threshold is selected on VAL and reported
on TEST (never selected on test).

Pipeline:
  1. best.pt D40 predictions at conf 0.001 on VAL and TEST (one inference pass each).
  2. VAL sweep conf 0.05..0.50 step 0.05: overall + per-bucket P/R/F1.
  3. pick two VAL operating points: F1-optimal, and recall-favoring (max recall with
     overall precision >= 0.50).
  4. apply both VAL-chosen thresholds to TEST; report vs the 0.25 baseline row.
  5. emit overall D40 PR-curve points (from conf 0.001 preds) for VAL and TEST as CSV.

Run on Colab:
    python src/detection/threshold_sweep.py --weights /content/drive/MyDrive/pothole/checkpoints/baseline/weights/best.pt
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for the import below
from src.detection import eval_stratified as E

PREC_FLOOR = 0.50          # recall-favoring point requires overall precision >= this
SWEEP = [round(t, 2) for t in np.arange(0.05, 0.5001, 0.05)]
VAL_LIST = Path("configs/splits/val.txt")


def overall_and_buckets(preds, gt, thr, small, medium) -> dict:
    """Overall + per-bucket P/R/F1 at a single confidence threshold."""
    gc, tp, fp = E._match_one_conf(preds, gt, thr, small, medium)
    o_tp, o_fp, o_gt = sum(tp.values()), sum(fp.values()), sum(gc.values())
    o_p = o_tp / (o_tp + o_fp) if (o_tp + o_fp) else float("nan")
    o_r = o_tp / o_gt if o_gt else float("nan")
    o_f1 = 2 * o_p * o_r / (o_p + o_r) if (o_p + o_r) else float("nan")
    row = {"conf": thr, "P": o_p, "R": o_r, "F1": o_f1}
    for b in ("small", "medium", "large"):
        row[f"{b}_R"] = tp[b] / gc[b] if gc[b] else float("nan")
        row[f"{b}_P"] = tp[b] / (tp[b] + fp[b]) if (tp[b] + fp[b]) else float("nan")
    return row


def pr_curve(preds, gt):
    """Overall D40 PR points from the conf-0.001 preds (same greedy matcher as AP)."""
    total_gt = sum(len(v) for v in gt.values())
    dets = []
    for stem, gts in gt.items():
        p = sorted(preds.get(stem, []), key=lambda d: d[4], reverse=True)
        matched = [False] * len(gts)
        for px1, py1, px2, py2, pc in p:
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gts):
                if matched[j]:
                    continue
                v = E.iou_xyxy((px1, py1, px2, py2), g[:4])
                if v >= E.MATCH_IOU and v > best_iou:
                    best_iou, best_j = v, j
            if best_j >= 0:
                matched[best_j] = True
                dets.append((pc, 1))
            else:
                dets.append((pc, 0))
    dets.sort(key=lambda d: d[0], reverse=True)
    tp = np.cumsum([d[1] for d in dets])
    fp = np.cumsum([1 - d[1] for d in dets])
    conf = [d[0] for d in dets]
    rows = []
    for i in range(len(dets)):
        r = tp[i] / total_gt if total_gt else float("nan")
        p = tp[i] / (tp[i] + fp[i])
        rows.append({"conf": conf[i], "precision": p, "recall": r})
    return rows


def _fmt(row, cols):
    return "  ".join((f"{row[c]:.3f}" if isinstance(row[c], float) else str(row[c])).ljust(8) for c in cols)


def main() -> None:
    ap = argparse.ArgumentParser(description="D40 operating-threshold analysis (VAL-selected, TEST-reported).")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--iou", type=float, default=E.NMS_IOU)
    ap.add_argument("--imgsz", type=int, default=E.IMGSZ)
    args = ap.parse_args()

    small, medium, _ = E.load_thresholds()
    val_stems = [Path(l).stem for l in VAL_LIST.read_text().splitlines() if l.strip()]
    test_stems = [Path(l).stem for l in E.TEST_LIST.read_text().splitlines() if l.strip()]
    gt_val = E.load_gt_d40(val_stems, small, medium)
    gt_test = E.load_gt_d40(test_stems, small, medium)

    preds_val = E.predict_baseline(args.weights, val_stems, E.CONF_AP, args.iou, args.imgsz)
    preds_test = E.predict_baseline(args.weights, test_stems, E.CONF_AP, args.iou, args.imgsz)

    # ---- VAL sweep ----
    cols = ["conf", "P", "R", "F1", "small_R", "small_P", "medium_R", "medium_P", "large_R", "large_P"]
    print("\n=== VAL sweep (threshold selection happens here) ===")
    print("  ".join(c.ljust(8) for c in cols))
    val_rows = []
    for thr in SWEEP:
        row = overall_and_buckets(preds_val, gt_val, thr, small, medium)
        val_rows.append(row)
        print(_fmt(row, cols))

    # ---- select two operating points on VAL ----
    f1_row = max(val_rows, key=lambda r: (r["F1"] if r["F1"] == r["F1"] else -1))
    f1_thr = f1_row["conf"]
    qualifying = [r for r in val_rows if r["P"] >= PREC_FLOOR]
    if qualifying:
        rec_row = max(qualifying, key=lambda r: r["R"])
        rec_thr = rec_row["conf"]
        rec_note = f"highest VAL recall with precision>= {PREC_FLOOR}: conf={rec_thr} (P={rec_row['P']:.3f}, R={rec_row['R']:.3f})"
    else:
        rec_thr = None
        rec_note = f"NO sweep point reached precision>= {PREC_FLOOR} on VAL; recall-favoring point undefined."
    print("\n=== chosen VAL operating points ===")
    print(f"F1-optimal      : conf={f1_thr}  (VAL F1={f1_row['F1']:.3f}, P={f1_row['P']:.3f}, R={f1_row['R']:.3f})")
    print(f"recall-favoring : {rec_note}")

    # ---- apply to TEST + 0.25 baseline ----
    print("\n=== TEST comparison (VAL-chosen thresholds; do NOT select on test) ===")
    print("  ".join(c.ljust(8) for c in cols))
    test_points = [("baseline", E.CONF_OP), ("F1-opt", f1_thr)]
    if rec_thr is not None:
        test_points.append(("recall-fav", rec_thr))
    for label, thr in test_points:
        row = overall_and_buckets(preds_test, gt_test, thr, small, medium)
        print(f"{label:<11}" + _fmt(row, cols))

    # ---- PR curves -> CSV ----
    E.OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, preds, gt in (("val", preds_val, gt_val), ("test", preds_test, gt_test)):
        rows = pr_curve(preds, gt)
        out = E.OUT_DIR / f"pr_curve_{name}.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["conf", "precision", "recall"])
            w.writeheader()
            w.writerows(rows)
        print(f"saved PR curve -> {out}  ({len(rows)} points)")


if __name__ == "__main__":
    main()
