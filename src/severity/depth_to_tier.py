"""Depth -> severity-tier labeling for the severity head (Component 6.2). LABELING ONLY.

Produces, per pothole, a depth-below-surface scalar (mm for PothRGBD, raw units for
Pothole440) and a low/med/high tier, with visual verification panels. No CNN, no training.

Sign note (IMPORTANT): the task wrote `depth_below_surface = plane - depth`, but PothRGBD
depth is RealSense camera-distance where in-pothole pixels read LARGER (confirmed in 6.1),
so a pothole floor is FARTHER than the road plane. To keep the stated semantic
"larger = deeper, deep -> high", we use **depth_below_surface = depth - plane** for PothRGBD
(positive = below road). Pothole440 depth is already surface-referenced (larger = deeper), so
its depth_below_surface is the raw in-mask depth. Both end up positive-deeper -> 'high'.

Outputs (all gitignored under data/):
  data/severity/_tiers/*.jpg  verification panels
  data/severity/labels_pothrgbd.csv , labels_pothole440.csv  (stem, dbs_scalar, tier, split)
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.severity.inspect_datasets import (POTHOLE440, load_depth, load_mask_png,
                                           pothrgbd_root, rasterize_yoloseg)

SEED = 42
SPLIT = (0.70, 0.15, 0.15)       # PothRGBD train/val/test
PLANE_MODE = "annulus"            # "annulus" (local ring) | "global" (whole-image road)
CLIP_MIN_MM = 200                 # PothRGBD depth clip: drop pixels outside this physical range
CLIP_MAX_MM = 3000                # (RealSense mm; kills uint16 sensor garbage / spikes)
ANNULUS_INNER_PX = 15             # dilate mask by this to skip the rim before the ring starts
ANNULUS_OUTER_PX = 61             # outer extent of the surrounding road ring
GLOBAL_EXCLUDE_PX = 21            # (global mode) exclude this neighborhood around potholes
ROBUST_PCT = 90                   # per-pothole robust depth statistic
FIT_SAMPLE = 6000                 # road pixels subsampled for RANSAC speed
MIN_INLIER_FRAC = 0.50            # flag plane fits below this
MIN_MASK_PX = 20                  # need at least this many valid in-mask pixels
MIN_RING_PX = 200                 # need at least this many road-ring pixels to fit a local plane
TIERS = ["low", "medium", "high"]
TIER_OUT = Path("data/severity/_tiers")


def _valid(depth: np.ndarray) -> np.ndarray:
    if np.issubdtype(depth.dtype, np.floating):
        return np.isfinite(depth) & (depth != 0)
    return depth > 0


def clip_depth(depth: np.ndarray) -> tuple[np.ndarray, int]:
    """Zero out PothRGBD pixels outside the sane physical range; return (clipped, n_removed)."""
    out = depth.astype(np.float32).copy()
    bad = (out > 0) & ((out < CLIP_MIN_MM) | (out > CLIP_MAX_MM))
    out[bad] = 0
    return out, int(bad.sum())


def fit_plane(depth: np.ndarray, fit_mask: np.ndarray, rng: random.Random):
    """RANSAC plane z=ax+by+c fit on the given pixel set; return (regressor, inlier_frac, ok)."""
    ys, xs = np.where(fit_mask)
    if len(xs) < 50:
        return None, 0.0, False
    z = depth[fit_mask].astype(float)
    if len(xs) > FIT_SAMPLE:
        idx = np.array(rng.sample(range(len(xs)), FIT_SAMPLE))
        xs_f, ys_f, z_f = xs[idx], ys[idx], z[idx]
    else:
        xs_f, ys_f, z_f = xs, ys, z
    try:
        r = RANSACRegressor(LinearRegression(), random_state=0).fit(
            np.column_stack([xs_f, ys_f]).astype(float), z_f)
    except Exception:
        return None, 0.0, False
    return r, float(r.inlier_mask_.mean()), True


def plane_at(reg, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    return reg.predict(np.column_stack([xs, ys]).astype(float))


def plane_full(reg, h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return reg.predict(np.column_stack([xx.ravel(), yy.ravel()]).astype(float)).reshape(h, w)


def road_fit_mask(mask: np.ndarray, valid: np.ndarray, mode: str) -> np.ndarray:
    """Pixels used to fit the road plane: a local ring (annulus) around the pothole, or the
    whole-image road (global). Excludes the pothole itself and invalid pixels in both modes."""
    if mode == "global":
        excl = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (GLOBAL_EXCLUDE_PX,) * 2)) > 0
        return valid & ~excl
    inner = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ANNULUS_INNER_PX,) * 2)) > 0
    outer = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ANNULUS_OUTER_PX,) * 2)) > 0
    return outer & ~inner & valid


# ----------------------------- PothRGBD -----------------------------
def process_pothrgbd(mode: str = PLANE_MODE):
    root = pothrgbd_root()
    rng = random.Random(SEED)
    imgs = sorted((root / "images").glob("*.jpg"))
    depth_by_ts = {"_".join(p.name.split("_")[:2]): p for p in (root / "depths").glob("*.npy")}

    records = []
    inlier_fracs = []
    n_ransac_fail = n_mask_fail = n_lowinlier = n_ring_fail = 0
    clipped_total = 0

    for p in imgs:
        ts = "_".join(p.name.split("_")[:2])
        dep_p = depth_by_ts.get(ts)
        lab = root / "labels" / (p.stem + ".txt")
        if not dep_p or not lab.exists():
            continue
        depth, n_clip = clip_depth(load_depth(dep_p))   # CLIP first
        clipped_total += n_clip
        mask, ninst = rasterize_yoloseg(lab, depth.shape[0], depth.shape[1])
        valid = _valid(depth)
        in_mask_valid = (mask > 0) & valid
        fit_mask = road_fit_mask(mask, valid, mode)

        rec = {"stem": p.stem, "ts": ts, "dbs": float("nan"), "road_std": float("nan"),
               "ratio": float("nan"), "inlier_frac": 0.0, "ninst": ninst, "flag": None}
        if in_mask_valid.sum() < MIN_MASK_PX:
            n_mask_fail += 1; rec["flag"] = "mask_fail"; records.append(rec); continue
        if fit_mask.sum() < MIN_RING_PX:
            n_ring_fail += 1; rec["flag"] = "ring_fail"; records.append(rec); continue
        reg, inlier_frac, ok = fit_plane(depth, fit_mask, rng)
        if not ok:
            n_ransac_fail += 1; rec["flag"] = "ransac_fail"; records.append(rec); continue
        inlier_fracs.append(inlier_frac)
        rec["inlier_frac"] = inlier_frac
        if inlier_frac < MIN_INLIER_FRAC:
            n_lowinlier += 1; rec["flag"] = "low_inlier"

        ys_m, xs_m = np.where(in_mask_valid)
        dbs_map = depth[in_mask_valid] - plane_at(reg, ys_m, xs_m)   # depth - plane (deeper = larger +)
        rec["dbs"] = float(np.percentile(dbs_map, ROBUST_PCT))
        ys_r, xs_r = np.where(fit_mask)
        res_ring = depth[fit_mask] - plane_at(reg, ys_r, xs_r)       # road residuals on the ring
        rec["road_std"] = float(np.std(res_ring))
        rec["ratio"] = rec["dbs"] / rec["road_std"] if rec["road_std"] > 0 else float("inf")
        records.append(rec)
    return records, inlier_fracs, {"ransac_fail": n_ransac_fail, "low_inlier": n_lowinlier,
                                   "mask_fail": n_mask_fail, "ring_fail": n_ring_fail,
                                   "clipped_px": clipped_total, "total": len(records)}


# ----------------------------- Pothole440 -----------------------------
def process_pothole440():
    records = []
    n_mask_fail = 0
    for split in ("train", "val", "test"):
        base = POTHOLE440 / split
        for r in sorted((base / "rgb").glob("*.png")):
            dep = base / "depth" / r.name
            mp = base / "mask" / r.name
            if not dep.exists() or not mp.exists():
                continue
            depth = load_depth(dep)
            mask = load_mask_png(mp)
            valid = _valid(depth)
            in_mask_valid = (mask > 0) & valid
            if in_mask_valid.sum() < MIN_MASK_PX:
                n_mask_fail += 1
                records.append({"stem": r.stem, "src_split": split, "dbs": float("nan"), "flag": "mask_fail"})
                continue
            dbs = float(np.percentile(depth[in_mask_valid].astype(float), ROBUST_PCT))  # already surface-ref
            records.append({"stem": r.stem, "src_split": split, "dbs": dbs, "flag": None})
    return records, {"mask_fail": n_mask_fail, "total": len(records)}


# ----------------------------- tiers + split -----------------------------
def assign_tiers(scalars: list[float], c1: float, c2: float) -> list[str]:
    out = []
    for s in scalars:
        if s != s:
            out.append("")
        elif s <= c1:
            out.append("low")
        elif s <= c2:
            out.append("medium")
        else:
            out.append("high")
    return out


def tertile_cuts(values: list[float]) -> tuple[float, float]:
    v = np.array([x for x in values if x == x])
    return float(np.percentile(v, 100 / 3)), float(np.percentile(v, 200 / 3))


def split_pothrgbd(records, rng: random.Random):
    """70/15/15 over UNIQUE timestamps (group key). Report whether aug variants exist."""
    ts_list = sorted({r["ts"] for r in records})
    unique = len(ts_list) == len(records)
    rng.shuffle(ts_list)
    n = len(ts_list)
    n_tr = int(SPLIT[0] * n)
    n_va = int(SPLIT[1] * n)
    assign = {}
    for i, ts in enumerate(ts_list):
        assign[ts] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return assign, unique


# ----------------------------- panels -----------------------------
def _colormap(depth, mask_invalid=None):
    valid = _valid(depth)
    vis = np.zeros(depth.shape, np.uint8)
    if valid.any():
        lo, hi = np.percentile(depth[valid], [2, 98])
        vis = (np.clip((depth.astype(float) - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    c = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    c[~valid] = 0
    return c


def _label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def _draw_contour(img, mask):
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
    return img


def panel(rgb, depth, mask, residual, tier, dbs, out: Path, crop=None):
    if crop is not None:
        y0, y1, x0, x1 = crop
        rgb, depth, mask = rgb[y0:y1, x0:x1], depth[y0:y1, x0:x1], mask[y0:y1, x0:x1]
        if residual is not None:
            residual = residual[y0:y1, x0:x1]
    cols = []
    cols.append(_label(_draw_contour(rgb.copy(), mask), "RGB + mask"))
    cols.append(_label(_draw_contour(_colormap(depth), mask), "depth"))
    if residual is not None:
        # scale the residual colormap to the IN-MASK depth-below-surface (not frame edges),
        # so the pothole's depression is visible rather than plane-mismatch at the corners.
        in_valid = (mask > 0) & _valid(depth)
        vmax = max(float(np.percentile(residual[in_valid], 95)) if in_valid.any() else 40.0, 20.0)
        res_vis = np.clip(residual, 0, vmax)
        rr = (res_vis / vmax * 255).astype(np.uint8)
        rr[~_valid(depth)] = 0
        cmap = cv2.applyColorMap(rr, cv2.COLORMAP_INFERNO)
        cols.append(_label(_draw_contour(cmap, mask), f"depth-below-surf (0-{vmax:.0f})"))
    else:
        cols.append(_label(_draw_contour(_colormap(depth), mask), "depth (surf-ref)"))
    tile = np.full_like(cols[0], 30)
    cols.append(_label(tile, f"{tier.upper()}  dbs={dbs:.0f}"))
    h = cols[0].shape[0]
    panel = np.hstack([cv2.resize(c, (int(c.shape[1] * h / c.shape[0]), h)) for c in cols])
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), panel)


def pick_examples(records, tier_key, n_each=3):
    out = []
    for t in TIERS:
        picks = [r for r in records if r.get(tier_key) == t][:n_each]
        out += picks
    return out


def main():
    TIER_OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    # ---------- PothRGBD ----------
    print("########## PothRGBD ##########")
    recs, inliers, fails = process_pothrgbd()
    assign, unique_ts = split_pothrgbd(recs, random.Random(SEED))
    for r in recs:
        r["split"] = assign[r["ts"]]
    train_scalars = [r["dbs"] for r in recs if r["split"] == "train"]
    c1, c2 = tertile_cuts(train_scalars)
    for r in recs:
        r["tier"] = assign_tiers([r["dbs"]], c1, c2)[0]

    valid_dbs = np.array([r["dbs"] for r in recs if r["dbs"] == r["dbs"]])
    ratios = np.array([r["ratio"] for r in recs if np.isfinite(r["ratio"])])
    print(f"plane mode={PLANE_MODE}  potholes={len(recs)}  valid_scalars={len(valid_dbs)}")
    print(f"CLIP {CLIP_MIN_MM}-{CLIP_MAX_MM} mm removed {fails['clipped_px']:,} pixels total")
    print(f"augmentation: unique timestamps={sum(1 for _ in set(r['ts'] for r in recs))}/{len(recs)} "
          f"(all-unique={unique_ts}); shared-ts pairs grouped in split")
    print(f"depth-below-surface mm: min={valid_dbs.min():.0f} p25={np.percentile(valid_dbs,25):.0f} "
          f"median={np.median(valid_dbs):.0f} p75={np.percentile(valid_dbs,75):.0f} max={valid_dbs.max():.0f}")
    print(f"RANSAC inlier frac: mean={np.mean(inliers):.2f} min={np.min(inliers):.2f}")
    print(f"flags: ransac_fail={fails['ransac_fail']} low_inlier={fails['low_inlier']} "
          f"mask_fail={fails['mask_fail']} ring_fail={fails['ring_fail']}")
    clean = int((ratios > 2).sum()); marg = int(((ratios >= 1) & (ratios <= 2)).sum()); noise = int((ratios < 1).sum())
    tot = len(ratios)
    print(f"\n  SNR gate ratio = dbs / road-residual-std  (n={tot}):")
    print(f"    percentiles p10/50/90 = {np.percentile(ratios,[10,50,90]).round(2)}")
    print(f"    CLEAN  (>2)  : {clean} ({100*clean/tot:.1f}%)")
    print(f"    MARGINAL(1-2): {marg} ({100*marg/tot:.1f}%)")
    print(f"    NOISE  (<1)  : {noise} ({100*noise/tot:.1f}%)")
    print(f"\nTRAIN-frozen tertile cuts (mm): c1={c1:.1f}  c2={c2:.1f}")
    for sp in ("train", "val", "test"):
        cnt = {t: sum(1 for r in recs if r["split"] == sp and r["tier"] == t) for t in TIERS}
        print(f"  {sp}: {cnt}  (unlabeled={sum(1 for r in recs if r['split']==sp and not r['tier'])})")

    # ---------- Pothole440 ----------
    print("\n########## Pothole440 (inference-only) ##########")
    recs4, fails4 = process_pothole440()
    v4 = [r["dbs"] for r in recs4 if r["dbs"] == r["dbs"]]
    d1, d2 = tertile_cuts(v4)
    for r in recs4:
        r["tier"] = assign_tiers([r["dbs"]], d1, d2)[0]
    print(f"potholes={len(recs4)}  valid={len(v4)}  mask_fail={fails4['mask_fail']}")
    print(f"depth-below-surface (raw uint8): min={min(v4):.0f} median={np.median(v4):.0f} max={max(v4):.0f}")
    print(f"OWN tertile cuts (raw): d1={d1:.1f}  d2={d2:.1f}  (stored separately; never shared with PothRGBD)")
    cnt4 = {t: sum(1 for r in recs4 if r["tier"] == t) for t in TIERS}
    print(f"  tiers: {cnt4}")

    # ---------- panels ----------
    root = pothrgbd_root()
    depth_by_ts = {"_".join(p.name.split("_")[:2]): p for p in (root / "depths").glob("*.npy")}
    for r in pick_examples([x for x in recs if x["tier"] and x["flag"] != "mask_fail"], "tier"):
        p = root / "images" / (r["stem"] + ".jpg")
        depth, _ = clip_depth(load_depth(depth_by_ts[r["ts"]]))
        mask, _ = rasterize_yoloseg(root / "labels" / (r["stem"] + ".txt"), depth.shape[0], depth.shape[1])
        fm = road_fit_mask(mask, _valid(depth), PLANE_MODE)
        reg, _, ok = fit_plane(depth, fm, random.Random(SEED))
        residual = (depth.astype(float) - plane_full(reg, depth.shape[0], depth.shape[1])) if ok else None
        # crop to the local annulus neighborhood (what is actually measured), not the whole frame
        outer = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ANNULUS_OUTER_PX + 20,) * 2)) > 0
        ys, xs = np.where(outer)
        crop = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
        panel(cv2.imread(str(p)), depth, mask, residual, r["tier"], r["dbs"],
              TIER_OUT / f"pothrgbd_{r['tier']}_{r['ts']}.jpg", crop=crop)
    for r in pick_examples([x for x in recs4 if x["tier"]], "tier"):
        base = POTHOLE440 / r["src_split"]
        name = r["stem"] + ".png"
        depth = load_depth(base / "depth" / name)
        mask = load_mask_png(base / "mask" / name)
        panel(cv2.imread(str(base / "rgb" / name)), depth, mask, None, r["tier"], r["dbs"],
              TIER_OUT / f"pothole440_{r['tier']}_{r['stem']}.jpg")

    # ---------- manifests ----------
    with open("data/severity/labels_pothrgbd.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "dbs_scalar", "road_std", "snr_ratio",
                                          "tier", "split", "inlier_frac", "flag"])
        w.writeheader()
        for r in recs:
            w.writerow({"stem": r["stem"], "dbs_scalar": r["dbs"], "road_std": r["road_std"],
                        "snr_ratio": r["ratio"], "tier": r["tier"], "split": r["split"],
                        "inlier_frac": round(r["inlier_frac"], 3), "flag": r["flag"] or ""})
    with open("data/severity/labels_pothole440.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "dbs_scalar", "tier", "src_split", "flag"])
        w.writeheader()
        for r in recs4:
            w.writerow({"stem": r["stem"], "dbs_scalar": r["dbs"], "tier": r["tier"],
                        "src_split": r["src_split"], "flag": r["flag"] or ""})
    print(f"\ncut sets -> PothRGBD(mm): [{c1:.1f},{c2:.1f}]  Pothole440(raw): [{d1:.1f},{d2:.1f}]")
    print(f"panels -> {TIER_OUT}/ ; manifests -> data/severity/labels_*.csv")


if __name__ == "__main__":
    main()
