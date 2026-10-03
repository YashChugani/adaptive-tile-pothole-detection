"""Severity-calibration gate diagnostic (data-driven threshold). REPORT ONLY.

Severity tiers are trained on PothRGBD close-up crops. A box-area-fraction gate marks a detection
"calibrated" only if the pothole fills enough of the frame to resemble those training crops. Here we
measure the box-area-fraction distribution of (1) the PothRGBD severity TRAINING crops and (2) the
cross-domain TEST set per domain, and propose candidate thresholds T.

box-area-fraction = (box_w * box_h) / (img_w * img_h); for YOLO-normalized labels this is just w*h.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

CROPS_PR = Path("data/severity/crops_pothrgbd.csv")        # severity training crops (box px + img dims)
MANIFEST = Path("configs/crossdomain/manifest_split.csv")   # cross-domain split
LABELS = Path("data/crossdomain/unified/labels")
DOMAINS = ["far_dashcam", "close_up", "tiny_far", "mid"]
PCTS = [5, 10, 25, 50, 75, 95]
TOL = 1e-6


def pct_line(name, arr):
    a = np.asarray(arr, float)
    p = {k: float(np.percentile(a, k)) for k in PCTS}
    print(f"  {name:<14} n={len(a):<5} min={a.min():.4f}  " +
          "  ".join(f"p{k}={p[k]:.4f}" for k in PCTS))
    return p


def pothrgbd_train_fracs():
    fr = []
    for r in csv.DictReader(open(CROPS_PR)):
        if r["split"] != "train":
            continue
        w = (int(r["bx1"]) - int(r["bx0"])) / int(r["img_w"])
        h = (int(r["by1"]) - int(r["by0"])) / int(r["img_h"])
        fr.append(w * h)
    return fr


def test_fracs_by_domain():
    rows = [r for r in csv.DictReader(open(MANIFEST)) if r["split"] == "test"]
    by = defaultdict(list)
    for r in rows:
        lab = LABELS / (Path(r["filename"]).stem + ".txt")
        if not lab.exists() or not lab.stat().st_size:
            continue
        seen = []
        for line in lab.read_text().splitlines():
            t = line.split()
            if not t:
                continue
            cx, cy, w, h = (float(v) for v in t[1:5])
            box = (cx, cy, w, h)
            if any(all(abs(box[i] - s[i]) <= TOL for i in range(4)) for s in seen):  # dedup Roboflow dups
                continue
            seen.append(box)
            by[r["domain"]].append(w * h)   # normalized w*h == area fraction
    return by


def main():
    print("=== 1. PothRGBD severity TRAINING crops: box-area-fraction ===")
    pr = pothrgbd_train_fracs()
    pr_p = pct_line("PothRGBD(train)", pr)

    print("\n=== 2. Cross-domain TEST box-area-fraction per domain (dups removed) ===")
    by = test_fracs_by_domain()
    for d in DOMAINS:
        pct_line(d, by[d])

    print("\n=== 3. Candidate thresholds T (from PothRGBD train percentiles) ===")
    candidates = {"PothRGBD_p5": pr_p[5], "PothRGBD_p10": float(np.percentile(pr, 10)), "PothRGBD_p25": pr_p[25]}
    hdr = f"{'T (source=value)':<26}{'PR(train)':>10}" + "".join(f"{d:>13}" for d in DOMAINS)
    print(hdr)
    print("  % of potholes ABOVE T (= 'calibrated'):")
    for name, T in candidates.items():
        prpct = 100 * np.mean(np.asarray(pr) >= T)
        row = f"{name}={T:.4f}".ljust(26) + f"{prpct:>9.1f}%"
        for d in DOMAINS:
            a = np.asarray(by[d], float)
            row += f"{100*np.mean(a >= T):>12.1f}%"
        print("  " + row)
    print("\nGoal: a T where PR(train) stays high (close-ups calibrated) and far/tiny drop low "
          "(correctly marked uncalibrated). close_up should track PR(train); far_dashcam/tiny_far should fall.")


if __name__ == "__main__":
    main()
