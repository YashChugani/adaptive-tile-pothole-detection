"""Cross-domain single-class 'pothole' inspection (Component 10.1). INSPECTION ONLY.

No conversion, no split, no training. Pools 4 sources that each contribute ONE class (pothole)
and reports: per-source counts/resolution/label-format/classes/which=pothole/domain proxy;
a unification plan with background counts; a cross-source pHash near-dup leakage check; and
projected pooled totals + domain mix. Reuses the Component 2B pHash.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.detection.build_eval_split import phash_u64, single_linkage_groups  # reuse pHash + clustering

# (name, [(images_dir, labels_dir)], pothole_class_index, format, note)
SOURCES = {
    "A_RDD_India": dict(
        dirs=[("data/rdd2022_india/images", "data/rdd2022_india/labels")],
        pothole=3, fmt="yolo_box", note="dashcam/far (D40 only; D00/D10/D20 dropped)"),
    "B_PothRGBD": dict(
        dirs=[("data/severity/pothrgbd/PUBLIC POTHOLE DATASET/images",
               "data/severity/pothrgbd/PUBLIC POTHOLE DATASET/labels")],
        pothole=0, fmt="yolo_seg", note="close-up (seg polygon class 0 -> bbox)"),
    "C_RoadDamage": dict(
        dirs=[(f"data/crossdomain/road_damage/Data Y12 Final/{s}/images",
               f"data/crossdomain/road_damage/Data Y12 Final/{s}/labels") for s in ("train", "valid", "test")],
        pothole=1, fmt="yolo_box", note="Roboflow multi-class [Crack,Pothole,Surface Erosion]; keep class 1"),
    "D_PotholeKaggle": dict(
        dirs=[(f"data/crossdomain/pothole_kaggle/{s}/images",
               f"data/crossdomain/pothole_kaggle/{s}/labels") for s in ("train", "valid", "test")],
        pothole=0, fmt="yolo_box", note="Roboflow single-class pothole (GoPro G00xxx)"),
}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def list_images(images_dir: Path):
    out = []
    for e in IMG_EXTS:
        out += list(images_dir.glob("*" + e))
    return sorted(set(out))


def pothole_boxes_frac(label_path: Path, cls: int, fmt: str):
    """Return list of pothole box area-fractions (w*h normalized) for this label file."""
    if not label_path.exists() or not label_path.stat().st_size:
        return []
    fracs = []
    for line in label_path.read_text().splitlines():
        t = line.split()
        if not t or int(float(t[0])) != cls:
            continue
        if fmt == "yolo_box":
            _, _, _, w, h = (float(v) for v in t[:5])
            fracs.append(w * h)
        else:  # yolo_seg polygon -> bbox
            xy = np.array(t[1:], float).reshape(-1, 2)
            fracs.append(float((xy[:, 0].max() - xy[:, 0].min()) * (xy[:, 1].max() - xy[:, 1].min())))
    return fracs


def domain_label(mean_frac: float) -> str:
    if mean_frac < 0.02:
        return "far/dashcam"
    if mean_frac < 0.08:
        return "mid"
    return "close-up"


def main():
    pooled = []   # (source, path, phash)
    per_source = {}
    for name, cfg in SOURCES.items():
        imgs, sizes, classes = [], [], Counter()
        n_pothole_boxes = 0
        imgs_with_pothole = 0
        all_fracs = []
        for images_dir, labels_dir in cfg["dirs"]:
            idir, ldir = Path(images_dir), Path(labels_dir)
            if not idir.exists():
                continue
            for ip in list_images(idir):
                imgs.append(ip)
                lp = ldir / (ip.stem + ".txt")
                if lp.exists() and lp.stat().st_size:
                    for line in lp.read_text().splitlines():
                        if line.split():
                            classes[int(float(line.split()[0]))] += 1
                fr = pothole_boxes_frac(lp, cfg["pothole"], cfg["fmt"])
                n_pothole_boxes += len(fr)
                all_fracs += fr
                if fr:
                    imgs_with_pothole += 1
                with Image.open(ip) as im:
                    sizes.append(im.size)
                    pooled.append((name, str(ip), phash_u64(im)))
        res = Counter(sizes)
        mean_frac = float(np.mean(all_fracs)) if all_fracs else 0.0
        per_source[name] = dict(
            n_images=len(imgs), n_pothole_boxes=n_pothole_boxes, imgs_with_pothole=imgs_with_pothole,
            backgrounds=len(imgs) - imgs_with_pothole, classes=dict(classes),
            res_common=res.most_common(3), res_unique=len(res),
            mean_pothole_frac=mean_frac, domain=domain_label(mean_frac), cfg=cfg)

    print("=" * 70)
    print("PER-SOURCE INSPECTION")
    for name, s in per_source.items():
        print(f"\n### {name}  ({s['cfg']['note']})")
        print(f"  images={s['n_images']}  format={s['cfg']['fmt']}  pothole_class_idx={s['cfg']['pothole']}")
        print(f"  classes present (idx:count): {s['classes']}")
        print(f"  resolutions: {s['res_unique']} unique; common={s['res_common']}")
        print(f"  pothole boxes={s['n_pothole_boxes']}  imgs_with_pothole={s['imgs_with_pothole']}  "
              f"backgrounds(no pothole)={s['backgrounds']}")
        print(f"  mean pothole box area-frac={s['mean_pothole_frac']:.4f} -> domain: {s['domain']}")

    # ---- unification plan ----
    print("\n" + "=" * 70)
    print("UNIFICATION PLAN (report only) -> single class 0 = pothole")
    print("  A_RDD_India    : D40(idx3) -> 0 ; drop D00/D10/D20")
    print("  B_PothRGBD     : seg polygon(idx0) -> bbox -> 0")
    print("  C_RoadDamage   : Pothole(idx1) -> 0 ; drop Crack/Surface Erosion")
    print("  D_PotholeKaggle: pothole(idx0) -> 0")
    print("  backgrounds (0 pothole boxes, kept as negatives) per source:")
    for name, s in per_source.items():
        print(f"    {name}: {s['backgrounds']} / {s['n_images']}")

    # ---- cross-source near-dup (pHash Hamming<=3) ----
    print("\n" + "=" * 70)
    print("CROSS-SOURCE NEAR-DUPLICATE CHECK (pHash Hamming<=3)")
    hashes = np.array([p[2] for p in pooled], dtype=np.uint64)
    print(f"  pooled images: {len(hashes)}  (computing all-pairs clusters...)")
    groups = single_linkage_groups(hashes, 3)
    members = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)
    cross_pairs = Counter()
    cross_clusters = 0
    examples = []
    for g, idxs in members.items():
        srcs = {pooled[i][0] for i in idxs}
        if len(idxs) >= 2 and len(srcs) >= 2:
            cross_clusters += 1
            for a in sorted(srcs):
                for b in sorted(srcs):
                    if a < b:
                        cross_pairs[(a, b)] += 1
            if len(examples) < 8:
                examples.append([Path(pooled[i][1]).name[:34] + f"({pooled[i][0][0]})" for i in idxs[:4]])
    print(f"  cross-source near-dup clusters: {cross_clusters}")
    print("  source-pair cluster counts:")
    for pair, c in cross_pairs.most_common():
        print(f"    {pair[0]} <-> {pair[1]}: {c}")
    print("  example cross-source clusters:")
    for ex in examples:
        print(f"    {ex}")

    # ---- projected pooled totals + domain mix ----
    print("\n" + "=" * 70)
    print("PROJECTED POOLED TOTALS (pothole images + instances)")
    tot_img = tot_box = 0
    dom_imgs = Counter()
    for name, s in per_source.items():
        tot_img += s["imgs_with_pothole"]; tot_box += s["n_pothole_boxes"]
        dom_imgs[s["domain"]] += s["imgs_with_pothole"]
        print(f"  {name}: pothole_imgs={s['imgs_with_pothole']}  pothole_boxes={s['n_pothole_boxes']}  domain={s['domain']}")
    print(f"  COMBINED: pothole_imgs={tot_img}  pothole_boxes={tot_box}")
    print("  domain mix (by pothole images):")
    for d, c in dom_imgs.most_common():
        print(f"    {d}: {c} ({100*c/tot_img:.1f}%)")


if __name__ == "__main__":
    main()
