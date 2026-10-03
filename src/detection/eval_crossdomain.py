"""Per-domain test eval: India-only baseline vs cross-domain detector (Component 10.5).

Pothole-vs-pothole on the cross-domain TEST split (802 imgs), scored with the EXISTING stratified
evaluator machinery (src/detection/eval_stratified.py: same greedy matcher, same IoU/conf) — reused,
not reimplemented. Both models scored at conf=0.20, IoU-match=0.5. Baseline keeps only D40 (idx 3)
remapped to pothole. GT boxes are deduped per image (float tol 1e-6) to remove Roboflow duplicate rows.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.detection import eval_stratified as E   # reuse matcher/AP/IoU + constants

UNIFIED = Path("data/crossdomain/unified")
MANIFEST = Path("configs/crossdomain/manifest_split.csv")
CROSSDOMAIN_W = "data/weights/crossdomain_best.pt"
BASELINE_W = "data/weights/baseline_best.pt"
CONF_OP = 0.20          # baseline's established operating point (Component 3), applied to BOTH
MATCH_IOU = E.MATCH_IOU  # 0.5
DOMAINS = ["far_dashcam", "close_up", "tiny_far", "mid"]
D40_IDX = 3             # baseline pothole class
OUT = Path("data/crossdomain/eval/per_domain_results.json")
TOL = 1e-6


def load_gt():
    """Per test image: deduped GT pothole boxes as (x1,y1,x2,y2,'all') at native pixels.
    Returns gt_by_image, domain_of, and before/after dedup instance counts per domain."""
    rows = [r for r in csv.DictReader(open(MANIFEST)) if r["split"] == "test"]
    gt, domain_of = {}, {}
    before, after = defaultdict(int), defaultdict(int)
    for r in rows:
        fn = r["filename"]
        dom = r["domain"]
        domain_of[fn] = dom
        with Image.open(UNIFIED / "images" / fn) as im:
            w, h = im.size
        lab = UNIFIED / "labels" / (Path(fn).stem + ".txt")
        raw = []
        if lab.exists() and lab.stat().st_size:
            for line in lab.read_text().splitlines():
                t = line.split()
                if not t:
                    continue
                cx, cy, bw, bh = (float(v) for v in t[1:5])
                raw.append((cx, cy, bw, bh))
        before[dom] += len(raw)
        # dedup identical normalized boxes to float tolerance
        uniq = []
        for b in raw:
            if not any(all(abs(b[i] - u[i]) <= TOL for i in range(4)) for u in uniq):
                uniq.append(b)
        after[dom] += len(uniq)
        boxes = []
        for cx, cy, bw, bh in uniq:
            x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
            x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
            boxes.append((x1, y1, x2, y2, "small"))   # valid bucket key; unused (overall sums only)
        gt[fn] = boxes
    return gt, domain_of, before, after


def predict(weights, stems, keep_cls=None):
    """conf=0.001 inference -> {stem: [(x1,y1,x2,y2,conf)]}. keep_cls filters a class (baseline D40)."""
    from ultralytics import YOLO
    model = YOLO(weights)
    out = defaultdict(list)
    for stem in stems:
        img = str(UNIFIED / "images" / stem)
        r = model.predict(img, conf=E.CONF_AP, iou=E.NMS_IOU, imgsz=E.IMGSZ, device="cpu", verbose=False)[0]
        for b in r.boxes:
            if keep_cls is not None and int(b.cls) != keep_cls:
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            out[stem].append((x1, y1, x2, y2, float(b.conf)))
    return out


def score(preds, gt):
    """Overall P/R/F1/AP/TP/FP/FN/nGT/nPred for one image set, via the reused evaluator.
    small/medium are only bucket LABELS inside _match_one_conf; we read TOTALS, so literals are fine."""
    gc, tp, fp = E._match_one_conf(preds, gt, CONF_OP, 1024.0, 9216.0)
    TP, FP, GT = sum(tp.values()), sum(fp.values()), sum(gc.values())
    FN = GT - TP
    P = TP / (TP + FP) if (TP + FP) else float("nan")
    R = TP / GT if GT else float("nan")
    F1 = 2 * P * R / (P + R) if (P == P and R == R and (P + R)) else float("nan")
    ap = E._ap50(preds, gt, E.CONF_AP)
    n_pred = sum(1 for s in preds for p in preds[s] if p[4] >= CONF_OP)
    return {"P": P, "R": R, "F1": F1, "AP50": ap, "nGT": GT, "nPred": n_pred,
            "TP": TP, "FP": FP, "FN": FN}


def main():
    gt, domain_of, before, after = load_gt()
    stems = list(gt)
    print(f"test images: {len(stems)}")

    print("\n=== GT instances per domain: BEFORE -> AFTER dedup (tol {:.0e}) ===".format(TOL))
    tb = ta = 0
    for d in DOMAINS:
        print(f"  {d:<12} {before[d]:>6} -> {after[d]:>6}   (removed {before[d]-after[d]})")
        tb += before[d]; ta += after[d]
    print(f"  {'OVERALL':<12} {tb:>6} -> {ta:>6}   (removed {tb-ta})")

    print("\npredicting (cross-domain)...")
    preds_cd = predict(CROSSDOMAIN_W, stems, keep_cls=None)
    print("predicting (baseline, keep D40 only)...")
    preds_bl = predict(BASELINE_W, stems, keep_cls=D40_IDX)

    def by_domain(preds):
        res = {}
        for d in DOMAINS + ["OVERALL"]:
            sel = [s for s in stems if (d == "OVERALL" or domain_of[s] == d)]
            res[d] = score({s: preds[s] for s in sel}, {s: gt[s] for s in sel})
        return res

    cd = by_domain(preds_cd)
    bl = by_domain(preds_bl)

    # image & GT context per domain
    img_count = {d: sum(1 for s in stems if domain_of[s] == d) for d in DOMAINS}
    img_count["OVERALL"] = len(stems)

    print("\n=== PER-DOMAIN TEST EVAL (conf=0.20, IoU=0.5) — baseline(India-only) vs cross-domain ===")
    print(f"{'domain':<12}{'imgs':>5}{'GT':>6} | {'BL_P':>6}{'BL_R':>6}{'BL_F1':>6}{'BL_AP':>6} | "
          f"{'CD_P':>6}{'CD_R':>6}{'CD_F1':>6}{'CD_AP':>6}")
    for d in DOMAINS + ["OVERALL"]:
        b, c = bl[d], cd[d]
        print(f"{d:<12}{img_count[d]:>5}{b['nGT']:>6} | "
              f"{b['P']:>6.3f}{b['R']:>6.3f}{b['F1']:>6.3f}{b['AP50']:>6.3f} | "
              f"{c['P']:>6.3f}{c['R']:>6.3f}{c['F1']:>6.3f}{c['AP50']:>6.3f}")

    print("\n=== TP/FP/FN (conf=0.20) ===")
    print(f"{'domain':<12} | {'BL TP/FP/FN (nPred)':>22} | {'CD TP/FP/FN (nPred)':>22}")
    for d in DOMAINS + ["OVERALL"]:
        b, c = bl[d], cd[d]
        bstr = "{}/{}/{} ({})".format(b["TP"], b["FP"], b["FN"], b["nPred"])
        cstr = "{}/{}/{} ({})".format(c["TP"], c["FP"], c["FN"], c["nPred"])
        print(f"{d:<12} | {bstr:>22} | {cstr:>22}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "operating_point": {"conf": CONF_OP, "iou_match": MATCH_IOU},
        "gt_dedup": {"before": dict(before), "after": dict(after)},
        "image_counts": img_count,
        "baseline": bl, "crossdomain": cd,
    }, indent=2), encoding="utf-8")
    print(f"\nresults -> {OUT}")


if __name__ == "__main__":
    main()
