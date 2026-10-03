"""Build the UNIFIED single-class ('pothole'=0) cross-domain dataset (Component 10.2).

Convert + remap-to-class-0 + dedup (cross-source C-vs-A and within-source) + domain-tag + pool
+ cap backgrounds. Outputs to data/crossdomain/unified/ (gitignored). No split / data.yaml (10.3).

Sources -> class 0:
  rdd (far_dashcam)  data/rdd2022_india           D40(3)->0, drop D00/D10/D20
  pothrgbd (close_up) data/severity/pothrgbd        seg(0) polygon -> bbox -> 0
  roaddmg (mid)      data/crossdomain/road_damage  Pothole(1)->0 ; drop images that pHash<=3-dup rdd
  kaggle (tiny_far)  data/crossdomain/pothole_kaggle pothole(0)->0
Within-source pHash<=3 pairs: keep one, drop the rest. Backgrounds capped at BACKGROUND_FRACTION.
"""
from __future__ import annotations

import csv
import hashlib
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.detection.build_eval_split import phash_u64  # reuse pHash

SEED = 42
HAMMING = 3
BACKGROUND_FRACTION = 0.30          # backgrounds <= this share of final pooled images
OUT = Path("data/crossdomain/unified")
VERIFY = Path("data/crossdomain/_verify")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

SRC = {
    "rdd": dict(domain="far_dashcam", pothole=3, fmt="box",
                dirs=[("data/rdd2022_india/images", "data/rdd2022_india/labels")]),
    "pothrgbd": dict(domain="close_up", pothole=0, fmt="seg",
                     dirs=[("data/severity/pothrgbd/PUBLIC POTHOLE DATASET/images",
                            "data/severity/pothrgbd/PUBLIC POTHOLE DATASET/labels")]),
    "roaddmg": dict(domain="mid", pothole=1, fmt="box",
                    dirs=[(f"data/crossdomain/road_damage/Data Y12 Final/{s}/images",
                           f"data/crossdomain/road_damage/Data Y12 Final/{s}/labels") for s in ("train", "valid", "test")]),
    "kaggle": dict(domain="tiny_far", pothole=0, fmt="box",
                   dirs=[(f"data/crossdomain/pothole_kaggle/{s}/images",
                          f"data/crossdomain/pothole_kaggle/{s}/labels") for s in ("train", "valid", "test")]),
}


def _clip01(v):
    return min(max(v, 0.0), 1.0)


def remap_boxes(label_path: Path, cls: int, fmt: str):
    """Pothole boxes -> normalized (cx,cy,w,h) at class 0, clipped to [0,1]."""
    out = []
    if not label_path.exists() or not label_path.stat().st_size:
        return out
    for line in label_path.read_text().splitlines():
        t = line.split()
        if not t or int(float(t[0])) != cls:
            continue
        if fmt == "box":
            cx, cy, w, h = (float(v) for v in t[1:5])
        else:  # seg polygon -> bbox
            xy = np.array(t[1:], float).reshape(-1, 2)
            x0, y0, x1, y1 = xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()
            cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)
        cx, cy, w, h = _clip01(cx), _clip01(cy), _clip01(w), _clip01(h)
        if w > 0 and h > 0:
            out.append((cx, cy, w, h))
    return out


def list_images(d: Path):
    out = []
    for e in IMG_EXTS:
        out += list(d.glob("*" + e))
    return sorted(set(out))


def gather():
    """Per source: list of record dicts {src,domain,img,stem,phash,boxes}."""
    recs = defaultdict(list)
    for name, cfg in SRC.items():
        for idir, ldir in cfg["dirs"]:
            idir, ldir = Path(idir), Path(ldir)
            if not idir.exists():
                continue
            for ip in list_images(idir):
                with Image.open(ip) as im:
                    ph = phash_u64(im)
                boxes = remap_boxes(ldir / (ip.stem + ".txt"), cfg["pothole"], cfg["fmt"])
                recs[name].append(dict(src=name, domain=cfg["domain"], img=ip, stem=ip.stem,
                                       phash=ph, boxes=boxes))
    return recs


def _greedy_keep(rs, thr):
    """Greedy keep-first dedup (compare each image only to already-KEPT representatives, so
    near-dup chains do not collapse distinct images)."""
    kept, kh = [], []
    for r in rs:
        if kh and int(np.bitwise_count(np.uint64(r["phash"]) ^ np.array(kh, dtype=np.uint64)).min()) <= thr:
            continue
        kept.append(r); kh.append(r["phash"])
    return kept


# PothRGBD is close-up: pHash is non-discriminative (even Hamming==0 collides across ~182 distinct
# potholes), so it uses EXACT FILE-HASH dedup. The far/mid video-sourced sets use greedy pHash<=3.
FILEHASH_SOURCES = {"pothrgbd"}


def _filehash_keep(rs):
    """Drop only true byte-identical file duplicates (md5), keeping distinct images."""
    seen, kept = set(), []
    for r in rs:
        h = hashlib.md5(Path(r["img"]).read_bytes()).hexdigest()
        if h in seen:
            continue
        seen.add(h); kept.append(r)
    return kept


def dedup(recs):
    report = {}
    # cross-source: drop roaddmg records that pHash<=3-dup any rdd image (unchanged)
    a_hashes = np.array([r["phash"] for r in recs["rdd"]], dtype=np.uint64)
    kept_c, dropped_c = [], 0
    for r in recs["roaddmg"]:
        if int(np.bitwise_count(np.uint64(r["phash"]) ^ a_hashes).min()) <= HAMMING:
            dropped_c += 1
        else:
            kept_c.append(r)
    recs["roaddmg"] = kept_c
    report["roaddmg_cross_vs_rdd_dropped"] = dropped_c

    # within-source: PothRGBD = exact file-hash; others = greedy keep-first pHash<=3 (no chaining)
    for name in SRC:
        rs = recs[name]
        if not rs:
            continue
        if name in FILEHASH_SOURCES:
            kept = _filehash_keep(rs)
            report[f"{name}_within_dropped(filehash)"] = len(rs) - len(kept)
        else:
            kept = _greedy_keep(rs, HAMMING)
            report[f"{name}_within_dropped(phash<={HAMMING})"] = len(rs) - len(kept)
        recs[name] = kept
    return report


def draw_overlay(rec, out_path: Path):
    img = cv2.imread(str(rec["img"]))
    if img is None:
        return
    h, w = img.shape[:2]
    for cx, cy, bw, bh in rec["boxes"]:
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, f"{rec['src']} {rec['domain']} boxes={len(rec['boxes'])}", (4, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def mirror(src_img: Path, dst: Path):
    if dst.exists():
        return
    try:
        os.link(src_img, dst)
    except OSError:
        shutil.copy2(src_img, dst)


def main():
    rng = random.Random(SEED)
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)

    recs = gather()
    pre = {k: len(v) for k, v in recs.items()}
    dedrep = dedup(recs)

    # partition
    pothole_recs, bg_recs = [], []
    for name in SRC:
        for r in recs[name]:
            (pothole_recs if r["boxes"] else bg_recs).append(r)
    n_pothole = len(pothole_recs)

    # cap backgrounds at BACKGROUND_FRACTION of final pooled count
    max_bg = int(BACKGROUND_FRACTION / (1 - BACKGROUND_FRACTION) * n_pothole)
    rng.shuffle(bg_recs)
    kept_bg = bg_recs[:max_bg]
    kept_bg_set = {id(r) for r in kept_bg}
    bg_kept_by_src = Counter(r["src"] for r in kept_bg)
    bg_drop_by_src = Counter(r["src"] for r in bg_recs if id(r) not in kept_bg_set)

    final = pothole_recs + kept_bg

    # write images + labels + manifest
    man = []
    for r in final:
        fn = f"{r['src']}_{r['stem']}{r['img'].suffix}"
        mirror(r["img"], OUT / "images" / fn)
        (OUT / "labels" / (Path(fn).stem + ".txt")).write_text(
            "\n".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cx, cy, w, h in r["boxes"]), encoding="utf-8")
        man.append({"filename": fn, "source": r["src"], "domain": r["domain"],
                    "n_pothole_boxes": len(r["boxes"]), "is_background": int(not r["boxes"])})
    with (OUT / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "source", "domain", "n_pothole_boxes", "is_background"])
        w.writeheader(); w.writerows(man)

    # overlays: 3 pothole-bearing per source
    for name in SRC:
        picks = [r for r in recs[name] if r["boxes"]][:3]
        for i, r in enumerate(picks):
            draw_overlay(r, VERIFY / f"{name}_{i}_{r['stem'][:16]}.jpg")

    # ---- report ----
    print("=== dedup ===")
    for k, v in dedrep.items():
        print(f"  {k}: {v}")
    print("  pre-dedup image counts:", pre)
    print("\n=== final pooled ===")
    by = defaultdict(lambda: [0, 0, 0])  # src -> [images, pothole_imgs, boxes]
    for r in final:
        b = by[r["src"]]; b[0] += 1; b[1] += 1 if r["boxes"] else 0; b[2] += len(r["boxes"])
    print(f"{'source':<10}{'domain':<12}{'images':>8}{'pothole_imgs':>14}{'boxes':>8}{'bg':>6}")
    for name in SRC:
        b = by[name]
        print(f"{name:<10}{SRC[name]['domain']:<12}{b[0]:>8}{b[1]:>14}{b[2]:>8}{b[0]-b[1]:>6}")
    tot_img = len(final); tot_pot = n_pothole; tot_box = sum(len(r['boxes']) for r in final); tot_bg = tot_img - tot_pot
    print(f"{'TOTAL':<22}{tot_img:>8}{tot_pot:>14}{tot_box:>8}{tot_bg:>6}")
    print(f"\nbackground cap: BACKGROUND_FRACTION={BACKGROUND_FRACTION} -> max_bg={max_bg}; "
          f"kept={len(kept_bg)} ({100*len(kept_bg)/tot_img:.1f}% of pooled)")
    print(f"  bg kept per source : {dict(bg_kept_by_src)}")
    print(f"  bg dropped per src : {dict(bg_drop_by_src)}")
    note = ("NOTE: within-source dedup here is HYGIENE only (remove exact/near-identical frame "
            "duplicates). Train/test LEAKAGE is prevented in 10.3 by pHash-grouping the full pool "
            "across the split, not by this dedup.")
    (OUT / "DEDUP_NOTE.txt").write_text(note + "\n", encoding="utf-8")
    print("\n" + note)
    print(f"\noutputs -> {OUT}/images, {OUT}/labels, {OUT}/manifest.csv ; overlays -> {VERIFY}/")


if __name__ == "__main__":
    main()
