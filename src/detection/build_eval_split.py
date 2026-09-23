"""Freeze the evaluation spine for the 4-class RDD2022 India detection set.

Component 2B. Deterministic + seeded. Builds:
  - pHash single-linkage near-dup groups (Hamming <= 3), kept wholly within a split/fold.
  - a grouped, stratified 70/15/15 train/val/test split (the FIXED primary spine).
  - a grouped, stratified 5-fold CV assignment over the SAME scheme (every image tests once)
    used ONLY for the size-stratified small-recall CI.
  - per-D40-box size buckets at imgsz=640 (COCO-absolute primary, equal-count tertiles secondary).

Stratification unit = pHash group. Stratum key = (group_has_D40, small_D40_band), where
small = COCO area < 32^2 on the 640 letterboxed input and band = min(small_count, 3).

Nothing here trains. Data artifacts go under data/splits/ (gitignored); the small frozen
manifests that DEFINE the split are also written to configs/splits/ (tracked) for provenance.
"""
from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

# --- frozen configuration (implements the approved decisions) ---
IMGSZ = 640
SCORED_CLASSES = ["D00", "D10", "D20", "D40"]  # mirrors voc_to_yolo.SCORED_MAP
D40_IDX = 3
COCO_SMALL = 32 ** 2   # 1024
COCO_MEDIUM = 96 ** 2  # 9216
HAMMING_THR = 3
PRIMARY_SEED = 42
CV_SEED = 1337
N_FOLDS = 5
SPLIT_TARGETS = {"train": 0.70, "val": 0.15, "test": 0.15}

ROOT = Path("data/rdd2022_india")
IMAGES = ROOT / "images"
LABELS = ROOT / "labels"
DATA_SPLITS = Path("data/splits")            # gitignored (full artifacts)
CFG_SPLITS = Path("configs/splits")          # tracked (frozen manifests)
DATA_YAML = Path("configs/rdd_india.yaml")
IMG_LIST_PREFIX = "data/rdd2022_india/images"  # repo-relative, used inside data.yaml lists
REPO_ROOT_ABS = Path.cwd().resolve()


def phash_u64(im: Image.Image, hs: int = 8, hf: int = 32) -> np.uint64:
    g = im.convert("L").resize((hf, hf), Image.BILINEAR)
    px = np.asarray(g, dtype=np.float32)

    def dct1(x):
        N = x.shape[0]
        k = np.arange(N)
        M = np.cos(np.pi * (2 * k[:, None] + 1) * k[None, :] / (2 * N))
        return M @ x

    d = dct1(dct1(px).T).T
    low = d[:hs, :hs]
    med = np.median(low[1:, 1:])
    return np.packbits((low > med).flatten()).view(">u8")[0]


def single_linkage_groups(hashes: np.ndarray, thr: int) -> list[int]:
    """Union-find over all-pairs Hamming <= thr; returns a group id per image."""
    n = len(hashes)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        d = np.bitwise_count(hashes[i] ^ hashes)
        for j in np.where(d <= thr)[0]:
            j = int(j)
            if j > i:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb
    # normalize root -> contiguous group id
    roots = [find(i) for i in range(n)]
    remap = {r: g for g, r in enumerate(sorted(set(roots)))}
    return [remap[r] for r in roots]


def assign_units(units: list, sizes: dict, strata: dict, targets: dict, seed: int) -> dict:
    """Water-filling grouped assignment, done independently within each stratum.

    Each unit (pHash group) is placed whole into the target bucket that is currently
    most under-served relative to its weight, keeping proportions close per stratum.
    """
    rng = random.Random(seed)
    counts = {k: 0 for k in targets}
    order = list(targets)
    assignment: dict = {}
    by_stratum = defaultdict(list)
    for u in units:
        by_stratum[strata[u]].append(u)
    for stratum in sorted(by_stratum, key=str):
        group_list = sorted(by_stratum[stratum], key=str)
        rng.shuffle(group_list)
        for u in group_list:
            k = min(order, key=lambda name: (counts[name] / targets[name], order.index(name)))
            assignment[u] = k
            counts[k] += sizes[u]
    return assignment


def coco_bucket(area: float) -> str:
    if area < COCO_SMALL:
        return "small"
    if area <= COCO_MEDIUM:
        return "medium"
    return "large"


def main() -> None:
    imgs = sorted(IMAGES.glob("*.jpg"), key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    n = len(imgs)
    stems = [p.stem for p in imgs]

    # --- pass 1: pHash + per-image D40 boxes/areas ---
    hashes = np.empty(n, dtype=np.uint64)
    d40_boxes = {}          # stem -> list[(area640, coco_bucket)]
    all_areas = []          # every D40 area640 (for tertile cuts)
    for i, p in enumerate(imgs):
        with Image.open(p) as im:
            W, H = im.size
            hashes[i] = phash_u64(im)
        r = IMGSZ / max(W, H)
        boxes = []
        lab = LABELS / (p.stem + ".txt")
        if lab.exists() and lab.stat().st_size:
            for line in lab.read_text().splitlines():
                c, cx, cy, bw, bh = line.split()
                if int(c) == D40_IDX:
                    area = float(bw) * W * r * float(bh) * H * r
                    boxes.append((area, coco_bucket(area)))
                    all_areas.append(area)
        d40_boxes[p.stem] = boxes

    d40_total = len(all_areas)

    # tertile cut points (secondary bucketing), stored exactly
    tert_c1, tert_c2 = (float(x) for x in np.percentile(all_areas, [100 / 3, 200 / 3]))

    def tertile_bucket(area: float) -> str:
        if area <= tert_c1:
            return "small"
        if area <= tert_c2:
            return "medium"
        return "large"

    # --- groups + per-group stratum ---
    group_of = single_linkage_groups(hashes, HAMMING_THR)  # index -> group id
    group_stems = defaultdict(list)
    for i, g in enumerate(group_of):
        group_stems[g].append(stems[i])
    groups = sorted(group_stems)

    group_size = {g: len(group_stems[g]) for g in groups}
    group_has_d40 = {}
    group_small_band = {}
    for g in groups:
        has = False
        small = 0
        for s in group_stems[g]:
            boxes = d40_boxes[s]
            if boxes:
                has = True
            small += sum(1 for _, b in boxes if b == "small")
        group_has_d40[g] = has
        group_small_band[g] = min(small, 3)
    group_stratum = {g: (group_has_d40[g], group_small_band[g]) for g in groups}

    # --- primary 70/15/15 split (grouped + stratified) ---
    split_of_group = assign_units(groups, group_size, group_stratum, SPLIT_TARGETS, PRIMARY_SEED)
    split_of_stem = {s: split_of_group[g] for g in groups for s in group_stems[g]}

    # --- grouped 5-fold CV (same scheme, own seed) ---
    fold_targets = {f: 1.0 / N_FOLDS for f in range(N_FOLDS)}
    fold_of_group = assign_units(groups, group_size, group_stratum, fold_targets, CV_SEED)
    fold_of_stem = {s: fold_of_group[g] for g in groups for s in group_stems[g]}

    # ---------- verification table ----------
    def d40_bucket_counts(members: list[str]) -> tuple[int, dict]:
        total = 0
        bc = Counter()
        for s in members:
            for area, b in d40_boxes[s]:
                total += 1
                bc[b] += 1
        return total, bc

    print("=== PLAN ===")
    print(f"images={n}  D40 boxes={d40_total}  groups(pHash<= {HAMMING_THR})={len(groups)}")
    print(f"primary split targets={SPLIT_TARGETS} seed={PRIMARY_SEED}; {N_FOLDS}-fold CV seed={CV_SEED}")
    print(f"COCO buckets: small<{COCO_SMALL} medium<= {COCO_MEDIUM} large>; "
          f"tertile cuts={tert_c1:.1f},{tert_c2:.1f}")

    header = f"{'split':<8}{'images':>8}{'D40':>7}{'small':>7}{'medium':>8}{'large':>7}"
    print("\n=== PRIMARY SPLIT ===")
    print(header)
    split_members = {k: [s for s in stems if split_of_stem[s] == k] for k in SPLIT_TARGETS}
    for k in ("train", "val", "test"):
        m = split_members[k]
        tot, bc = d40_bucket_counts(m)
        print(f"{k:<8}{len(m):>8}{tot:>7}{bc['small']:>7}{bc['medium']:>8}{bc['large']:>7}")

    print("\n=== 5-FOLD CV (test partition per fold) ===")
    print(f"{'fold':<8}{'images':>8}{'D40':>7}{'small':>7}{'medium':>8}{'large':>7}")
    fold_members = {f: [s for s in stems if fold_of_stem[s] == f] for f in range(N_FOLDS)}
    for f in range(N_FOLDS):
        m = fold_members[f]
        tot, bc = d40_bucket_counts(m)
        print(f"{f:<8}{len(m):>8}{tot:>7}{bc['small']:>7}{bc['medium']:>8}{bc['large']:>7}")

    # ---------- asserts ----------
    for g in groups:
        splits = {split_of_stem[s] for s in group_stems[g]}
        assert len(splits) == 1, f"group {g} spans splits {splits}"
        folds = {fold_of_stem[s] for s in group_stems[g]}
        assert len(folds) == 1, f"group {g} spans folds {folds}"
    assert sum(len(v) for v in split_members.values()) == n
    assert len(set().union(*[set(v) for v in split_members.values()])) == n
    fold_union = [fold_of_stem[s] for s in stems]
    assert len(fold_union) == n and set(fold_union) == set(range(N_FOLDS))
    assert sum(len(fold_members[f]) for f in range(N_FOLDS)) == n
    split_d40 = sum(d40_bucket_counts(split_members[k])[0] for k in SPLIT_TARGETS)
    assert split_d40 == d40_total == 3187, f"D40 total mismatch: {split_d40} vs {d40_total}"
    print("\nAll asserts passed "
          "(no group spans splits/folds; every image tests once; D40 total == 3187).")

    # ---------- write artifacts ----------
    DATA_SPLITS.mkdir(parents=True, exist_ok=True)
    CFG_SPLITS.mkdir(parents=True, exist_ok=True)

    # tracked: image-path lists (repo-relative), used by data.yaml AND defining the split
    for k in ("train", "val", "test"):
        (CFG_SPLITS / f"{k}.txt").write_text(
            "\n".join(f"{IMG_LIST_PREFIX}/{s}.jpg" for s in split_members[k]) + "\n", encoding="utf-8")
    # tracked: fold + group maps + meta (seeds, cuts, counts)
    (CFG_SPLITS / "folds.json").write_text(json.dumps(fold_of_stem, indent=0), encoding="utf-8")
    (CFG_SPLITS / "phash_groups.json").write_text(
        json.dumps({s: group_of[i] for i, s in enumerate(stems)}, indent=0), encoding="utf-8")

    def counts_block(members):
        tot, bc = d40_bucket_counts(members)
        return {"images": len(members), "d40": tot,
                "small": bc["small"], "medium": bc["medium"], "large": bc["large"]}

    meta = {
        "images": n, "d40_total": d40_total, "num_groups": len(groups),
        "hamming_threshold": HAMMING_THR, "imgsz": IMGSZ, "letterbox_r": IMGSZ / 720,
        "coco_small": COCO_SMALL, "coco_medium": COCO_MEDIUM,
        "tertile_cuts": [tert_c1, tert_c2],
        "primary_seed": PRIMARY_SEED, "cv_seed": CV_SEED,
        "split_targets": SPLIT_TARGETS, "n_folds": N_FOLDS,
        "primary_split": {k: counts_block(split_members[k]) for k in SPLIT_TARGETS},
        "cv_folds": {str(f): counts_block(fold_members[f]) for f in range(N_FOLDS)},
    }
    (CFG_SPLITS / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # gitignored: full per-box bucket assignments + convenience copies
    with (DATA_SPLITS / "d40_boxes.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "box_index", "area640", "coco_bucket", "tertile_bucket"])
        for s in stems:
            for bi, (area, cb) in enumerate(d40_boxes[s]):
                w.writerow([s, bi, f"{area:.2f}", cb, tertile_bucket(area)])
    (DATA_SPLITS / "split_assignment.json").write_text(json.dumps(split_of_stem, indent=0), encoding="utf-8")
    (DATA_SPLITS / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # data.yaml (4-class), pointing at the frozen image lists
    yaml_text = (
        "# Frozen 4-class RDD2022 India detection set (Component 2B).\n"
        "# 'path' is the absolute repo root on this machine; edit it if the repo moves.\n"
        f"path: {REPO_ROOT_ABS.as_posix()}\n"
        "train: configs/splits/train.txt\n"
        "val: configs/splits/val.txt\n"
        "test: configs/splits/test.txt\n"
        "nc: 4\n"
        "names:\n"
        "  0: D00\n  1: D10\n  2: D20\n  3: D40\n"
    )
    DATA_YAML.write_text(yaml_text, encoding="utf-8")
    print(f"\nWrote data.yaml -> {DATA_YAML}")
    print(f"Tracked manifests -> {CFG_SPLITS}/  (train/val/test.txt, folds.json, phash_groups.json, meta.json)")
    print(f"Full artifacts   -> {DATA_SPLITS}/  (d40_boxes.csv, split_assignment.json, meta.json)")


if __name__ == "__main__":
    main()
