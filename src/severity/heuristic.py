"""Box-area heuristic severity baseline (Component 6.3A) — the baseline the CNN must beat.

Predicts low/med/high from bounding-box AREA alone (normalized by image area so it is
resolution-fair across datasets). Tertile thresholds are fit on the PothRGBD TRAIN clean
split and FROZEN, then applied to PothRGBD val/test and to Pothole440. Reports accuracy,
per-class F1, confusion matrices, and the area-vs-depth-tier correlation (CNN headroom).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

TIERS = ["low", "medium", "high"]
ORD = {t: i for i, t in enumerate(TIERS)}
PR = Path("data/severity/crops_pothrgbd.csv")
P4 = Path("data/severity/crops_pothole440.csv")


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["area_frac"] = (int(r["bx1"]) - int(r["bx0"])) * (int(r["by1"]) - int(r["by0"])) \
            / (int(r["img_w"]) * int(r["img_h"]))
    return rows


def predict(area, a1, a2):
    return "low" if area <= a1 else ("medium" if area <= a2 else "high")


def report(name, rows, a1, a2):
    y_true = [r["tier"] for r in rows]
    y_pred = [predict(r["area_frac"], a1, a2) for r in rows]
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, labels=TIERS, average="macro", zero_division=0)
    f1c = f1_score(y_true, y_pred, labels=TIERS, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=TIERS)
    print(f"\n=== {name} (n={len(rows)}) ===")
    print(f"  accuracy={acc:.3f}  macro-F1={f1m:.3f}  per-class F1 low/med/high={[round(float(x),3) for x in f1c]}")
    print("  confusion (rows=true, cols=pred low/med/high):")
    for t, row in zip(TIERS, cm):
        print(f"    {t:<7}{list(map(int,row))}")
    return acc, f1m


def main():
    pr = load(PR)
    p4 = load(P4)
    train = [r for r in pr if r["split"] == "train"]
    a1, a2 = (float(x) for x in np.percentile([r["area_frac"] for r in train], [100 / 3, 200 / 3]))
    print(f"PothRGBD TRAIN-frozen area-fraction tertile cuts: a1={a1:.5f}  a2={a2:.5f}")
    print("(normalized box_area/image_area so PothRGBD 640x480 cuts can apply to Pothole440 1200x800)")

    report("PothRGBD TEST (clean)", [r for r in pr if r["split"] == "test"], a1, a2)
    report("PothRGBD VAL (clean)", [r for r in pr if r["split"] == "val"], a1, a2)
    report("Pothole440 (cross-dataset, PothRGBD-train cuts)", p4, a1, a2)

    # area vs depth-tier correlation (CNN headroom) on PothRGBD clean
    areas = np.array([r["area_frac"] for r in pr])
    tiers_ord = np.array([ORD[r["tier"]] for r in pr])
    dbs = np.array([float(r["dbs_scalar"]) for r in pr])
    rho_t, p_t = spearmanr(areas, tiers_ord)
    rho_d, p_d = spearmanr(areas, dbs)
    print("\n=== Area vs depth signal (PothRGBD clean, n=%d) ===" % len(pr))
    print(f"  Spearman(box-area, depth TIER)   rho={rho_t:.3f} (p={p_t:.1e})")
    print(f"  Spearman(box-area, depth SCALAR) rho={rho_d:.3f} (p={p_d:.1e})")
    print("  -> low|moderate correlation = depth carries signal area does NOT => CNN has headroom;")
    print("     high correlation would mean area already explains depth (little headroom).")


if __name__ == "__main__":
    main()
