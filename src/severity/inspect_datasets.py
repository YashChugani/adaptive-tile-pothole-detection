"""Inspect the two severity datasets (PothRGBD + Pothole440). INSPECTION ONLY.

No tier logic, no labels, no crops, no model. Reads everything from the files and reports:
directory tree + counts, RGB/depth/mask formats + pairing, depth dtype/shape/range/units/
invalid-pixel encoding, mask format, visual panels (RGB | depth | mask overlay), and the
depth-inside-vs-outside-mask comparison with an EXPLICIT sign (do in-pothole pixels read
LARGER or SMALLER depth than the road, and by how much). Panels go to data/severity/_inspect/.

Usage:
    python src/severity/inspect_datasets.py --dataset both [--limit 3]
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

POTHRGBD = Path("data/severity/pothrgbd")
POTHOLE440 = Path("data/severity/pothole440")
INSPECT_OUT = Path("data/severity/_inspect")


# ----------------------------- generic helpers -----------------------------
def tree_counts(root: Path) -> None:
    dirs = sorted({p.parent for p in root.rglob("*") if p.is_file()})
    print(f"  tree under {root}:")
    for d in dirs:
        rel = d.relative_to(root)
        exts = Counter(p.suffix.lower() for p in d.glob("*") if p.is_file())
        print(f"    {rel}/ : " + ", ".join(f"{n}x*{e}" for e, n in exts.items()))


def load_depth(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)  # preserves 16-bit / float


def depth_report(arr: np.ndarray, name: str) -> dict:
    valid = arr[arr > 0] if not np.issubdtype(arr.dtype, np.floating) else arr[np.isfinite(arr) & (arr != 0)]
    info = {
        "dtype": str(arr.dtype), "shape": tuple(arr.shape),
        "min": float(np.nanmin(arr)), "max": float(np.nanmax(arr)),
        "zeros": int((arr == 0).sum()), "nan": int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0,
        "valid_pct": round(100 * valid.size / arr.size, 1),
    }
    if valid.size:
        info["valid_p1_50_99"] = [round(float(x), 1) for x in np.percentile(valid, [1, 50, 99])]
    print(f"    {name}: dtype={info['dtype']} shape={info['shape']} range=[{info['min']:.1f},{info['max']:.1f}] "
          f"zeros={info['zeros']} nan={info['nan']} valid={info['valid_pct']}% p1/50/99={info.get('valid_p1_50_99')}")
    return info


def rasterize_yoloseg(label_path: Path, h: int, w: int) -> tuple[np.ndarray, int]:
    """YOLO-seg polygons (normalized) -> binary mask; returns (mask, n_instances)."""
    mask = np.zeros((h, w), np.uint8)
    n = 0
    for line in label_path.read_text().splitlines():
        toks = line.split()
        if len(toks) < 7:
            continue
        coords = np.array(toks[1:], float).reshape(-1, 2)
        pts = (coords * [w, h]).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
        n += 1
    return mask, n


def load_mask_png(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0).astype(np.uint8)


def inside_outside(depth: np.ndarray, mask: np.ndarray) -> dict | None:
    """Depth stats inside vs outside the mask, over VALID pixels, with explicit sign."""
    valid = (depth > 0)
    if np.issubdtype(depth.dtype, np.floating):
        valid = np.isfinite(depth) & (depth != 0)
    din = depth[(mask == 1) & valid]
    dout = depth[(mask == 0) & valid]
    if din.size == 0 or dout.size == 0:
        return None
    in_med, out_med = float(np.median(din)), float(np.median(dout))
    delta = in_med - out_med
    sign = "LARGER" if delta > 0 else "SMALLER"
    return {"in_med": in_med, "out_med": out_med, "delta": delta, "sign": sign,
            "in_n": int(din.size), "out_n": int(dout.size)}


def save_panel(rgb: np.ndarray, depth: np.ndarray, mask: np.ndarray, out: Path) -> None:
    valid = depth > 0
    if np.issubdtype(depth.dtype, np.floating):
        valid = np.isfinite(depth) & (depth != 0)
    vis = np.zeros(depth.shape, np.uint8)
    if valid.any():
        lo, hi = np.percentile(depth[valid], [2, 98])
        norm = np.clip((depth.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
        vis = (norm * 255).astype(np.uint8)
    depth_c = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    depth_c[~valid] = (0, 0, 0)
    overlay = rgb.copy()
    overlay[mask == 1] = (0.5 * overlay[mask == 1] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    h = min(rgb.shape[0], depth_c.shape[0], overlay.shape[0])
    panel = np.hstack([cv2.resize(x, (int(x.shape[1] * h / x.shape[0]), h)) for x in (rgb, depth_c, overlay)])
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), panel)


# ----------------------------- PothRGBD -----------------------------
def pothrgbd_root() -> Path:
    cand = list(POTHRGBD.rglob("depths"))
    return cand[0].parent if cand else POTHRGBD


def inspect_pothrgbd(limit: int) -> dict:
    print("\n########## A. PothRGBD ##########")
    tree_counts(POTHRGBD)
    root = pothrgbd_root()
    imgs = sorted((root / "images").glob("*.jpg"))
    depths = sorted((root / "depths").glob("*.npy"))
    labels = sorted((root / "labels").glob("*.txt"))
    print(f"  counts: images={len(imgs)} depths={len(depths)} labels={len(labels)}")

    def ts(name: str) -> str:
        return "_".join(name.split("_")[:2])  # 20250227_135438

    depth_by_ts = {ts(p.name): p for p in depths}
    paired = [(p, depth_by_ts.get(ts(p.name)), (root / "labels" / (p.stem + ".txt"))) for p in imgs]
    matched = [t for t in paired if t[1] and t[2].exists()]
    print(f"  pairing: RGB.stem==label.stem; depth by timestamp prefix. matched triples={len(matched)}/{len(imgs)}")
    print("  mask format: YOLO-seg polygons in labels/*.txt (normalized, class 0) -> rasterized to binary")

    signs = []
    for rgb_p, dep_p, lab_p in matched[:limit]:
        rgb = cv2.imread(str(rgb_p)); depth = load_depth(dep_p)
        print(f"\n  [{rgb_p.name[:28]}...]")
        depth_report(depth, "depth")
        mask, ninst = rasterize_yoloseg(lab_p, depth.shape[0], depth.shape[1])
        print(f"    mask: {ninst} polygon instance(s), coverage={100*mask.mean():.1f}%")
        io = inside_outside(depth, mask)
        if io:
            signs.append(io["sign"])
            print(f"    DEPTH in-pothole vs road: in_median={io['in_med']:.0f} vs out_median={io['out_med']:.0f} "
                  f"-> in-pothole reads {io['sign']} by {abs(io['delta']):.0f} (raw uint16 ~= mm)")
        save_panel(rgb, depth, mask, INSPECT_OUT / f"pothrgbd_{rgb_p.stem[:20]}.jpg")
    return {"count": len(matched), "depth_dtype": "uint16", "invalid": "0", "units": "mm (RealSense D415)",
            "signs": signs}


# ----------------------------- Pothole440 -----------------------------
def inspect_pothole440(limit: int) -> dict:
    print("\n########## B. Pothole440 ##########")
    if not any(POTHOLE440.rglob("*")):
        print("  (empty / clone not present yet)")
        return {}
    tree_counts(POTHOLE440)
    # find split dirs with rgb/depth/mask
    splits = {}
    for d in POTHOLE440.rglob("rgb"):
        splits[d.parent.name] = d.parent
    print(f"  splits found: {sorted(splits)}")

    depth_dtype = invalid = "?"
    signs = []
    shown = 0
    for split, base in sorted(splits.items()):
        rgbs = sorted((base / "rgb").glob("*"))
        deps = {p.stem: p for p in (base / "depth").glob("*")}
        masks = {p.stem: p for p in (base / "mask").glob("*")}
        matched = [r for r in rgbs if r.stem in deps and r.stem in masks]
        print(f"  [{split}] rgb={len(rgbs)} depth={len(deps)} mask={len(masks)} matched={len(matched)}")
        for r in matched:
            if shown >= limit:
                break
            depth = load_depth(deps[r.stem]); rgb = cv2.imread(str(r)); mask = load_mask_png(masks[r.stem])
            print(f"\n  [{split}/{r.name}]")
            info = depth_report(depth, "depth"); depth_dtype = info["dtype"]
            invalid = "0" if info["zeros"] > 0 else ("NaN" if info["nan"] > 0 else "none-seen")
            uniq = np.unique(cv2.imread(str(masks[r.stem]), cv2.IMREAD_UNCHANGED))
            print(f"    mask: values={uniq[:6]}{'...' if len(uniq)>6 else ''} -> {'binary' if len(uniq)<=2 else 'multi-value'}, coverage={100*mask.mean():.1f}%")
            io = inside_outside(depth, mask)
            if io:
                signs.append(io["sign"])
                print(f"    DEPTH in-pothole vs road: in_median={io['in_med']:.1f} vs out_median={io['out_med']:.1f} "
                      f"-> in-pothole reads {io['sign']} by {abs(io['delta']):.1f} (raw units)")
            save_panel(rgb, depth, mask, INSPECT_OUT / f"pothole440_{split}_{r.stem}.jpg")
            shown += 1
    return {"depth_dtype": depth_dtype, "invalid": invalid, "signs": signs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["pothrgbd", "pothole440", "both"], default="both")
    ap.add_argument("--limit", type=int, default=3, help="example potholes per dataset to visualize")
    args = ap.parse_args()

    a = b = {}
    if args.dataset in ("pothrgbd", "both"):
        a = inspect_pothrgbd(args.limit)
    if args.dataset in ("pothole440", "both"):
        b = inspect_pothole440(args.limit)

    if a and b:
        print("\n########## CROSS-DATASET MISMATCHES ##########")
        print(f"  depth dtype : PothRGBD={a.get('depth_dtype')}  vs  Pothole440={b.get('depth_dtype')}")
        print(f"  invalid enc : PothRGBD={a.get('invalid')}  vs  Pothole440={b.get('invalid')}")
        print(f"  units       : PothRGBD={a.get('units')}  vs  Pothole440=(scanner; confirm)")
        sa = set(a.get("signs", [])); sb = set(b.get("signs", []))
        print(f"  DEPTH SIGN  : PothRGBD in-pothole reads {sa or '?'}  vs  Pothole440 {sb or '?'} than road")
        if sa and sb and sa != sb:
            print("  >>> OPPOSITE SIGN CONVENTIONS — must be reconciled before any shared tier logic! <<<")
        elif sa and sb:
            print("  >>> same sign convention (good for a shared tier definition)")
    print(f"\npanels saved under {INSPECT_OUT}/")


if __name__ == "__main__":
    main()
