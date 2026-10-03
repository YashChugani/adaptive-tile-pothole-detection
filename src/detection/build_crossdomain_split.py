"""Frozen cross-domain split for the unified single-class pothole pool (Component 10.3).

Domain-aware leakage guard: group near-duplicates BEFORE splitting, assign whole groups.
  - far/varied sources (rdd, roaddmg, kaggle): greedy keep-first pHash<=3 (seed keeps; images
    within 3 of a seed join it; joined images do NOT recruit further). No single-linkage.
  - close_up (pothrgbd): group by file-content md5 ONLY (pHash is non-discriminative on close-ups).
Then 70/15/15 (seed 42), whole groups, stratified by (source, domain, is_background) cells.

Writes manifests under configs/crossdomain/ (NOT committed here). Read-only over data/.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.detection.build_eval_split import assign_units, phash_u64  # reuse assigner + pHash

UNIFIED = Path("data/crossdomain/unified")
OUT = Path("configs/crossdomain")
SEED = 42
HAMMING = 3
TARGETS = {"train": 0.70, "val": 0.15, "test": 0.15}
FAR_SOURCES = {"rdd", "roaddmg", "kaggle"}   # pHash-groupable
CLOSEUP_SOURCES = {"pothrgbd"}               # md5 exact only
DOMAINS = ["far_dashcam", "close_up", "tiny_far", "mid"]


def load_manifest():
    rows = list(csv.DictReader(open(UNIFIED / "manifest.csv")))
    for r in rows:
        r["n_pothole_boxes"] = int(r["n_pothole_boxes"])
        r["is_background"] = int(r["is_background"])
    return rows


def group_records(rows):
    """Assign a group_id to every image. Returns {filename: group_id}."""
    group_of = {}
    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)

    for source, recs in by_source.items():
        recs = sorted(recs, key=lambda r: r["filename"])   # stable order
        if source in CLOSEUP_SOURCES:
            md5_to_gid = {}
            for i, r in enumerate(recs):
                h = hashlib.md5((UNIFIED / "images" / r["filename"]).read_bytes()).hexdigest()
                gid = md5_to_gid.setdefault(h, f"{source}_{len(md5_to_gid)}")
                group_of[r["filename"]] = gid
        else:  # greedy keep-first pHash<=3
            seed_hashes = []   # parallel arrays
            seed_gids = []
            for r in recs:
                ph = phash_u64(Image.open(UNIFIED / "images" / r["filename"]))
                if seed_hashes:
                    d = np.bitwise_count(np.uint64(ph) ^ np.array(seed_hashes, dtype=np.uint64))
                    j = int(d.argmin())
                    if int(d[j]) <= HAMMING:
                        group_of[r["filename"]] = seed_gids[j]
                        continue
                gid = f"{source}_{len(seed_gids)}"
                seed_hashes.append(ph); seed_gids.append(gid)
                group_of[r["filename"]] = gid
    return group_of


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_manifest()
    for r in rows:
        if not (UNIFIED / "images" / r["filename"]).exists():
            raise SystemExit(f"missing image {r['filename']}")
    group_of = group_records(rows)

    # group -> members, size, cell, has_pothole
    gmembers = defaultdict(list)
    for r in rows:
        gmembers[group_of[r["filename"]]].append(r)
    groups = sorted(gmembers)
    gsize = {g: len(gmembers[g]) for g in groups}
    g_has_pothole = {g: any(m["n_pothole_boxes"] > 0 for m in gmembers[g]) for g in groups}
    gsource = {g: gmembers[g][0]["source"] for g in groups}
    gdomain = {g: gmembers[g][0]["domain"] for g in groups}
    gstratum = {g: (gsource[g], gdomain[g], 0 if g_has_pothole[g] else 1) for g in groups}

    # stratified whole-group assignment (seeded water-fill per cell)
    split_of_group = assign_units(groups, gsize, gstratum, TARGETS, SEED)
    split_of_file = {r["filename"]: split_of_group[group_of[r["filename"]]] for r in rows}

    # ---- write manifests ----
    for sp in ("train", "val", "test"):
        files = sorted(r["filename"] for r in rows if split_of_file[r["filename"]] == sp)
        (OUT / f"{sp}.txt").write_text(
            "\n".join(f"data/crossdomain/unified/images/{f}" for f in files) + "\n", encoding="utf-8")
    with (OUT / "manifest_split.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "source", "domain", "split",
                                          "n_pothole_boxes", "is_background", "group_id"])
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["filename"]):
            w.writerow({"filename": r["filename"], "source": r["source"], "domain": r["domain"],
                        "split": split_of_file[r["filename"]], "n_pothole_boxes": r["n_pothole_boxes"],
                        "is_background": r["is_background"], "group_id": group_of[r["filename"]]})
    (OUT / "data.yaml").write_text(
        "# Single-class cross-domain pothole dataset (Component 10.3).\n"
        "# Paths are RELATIVE to the repo root (run from there). Colab-local ABSOLUTE paths are\n"
        "# generated at runtime by the setup step and are NEVER committed.\n"
        "path: .\n"
        "train: configs/crossdomain/train.txt\n"
        "val: configs/crossdomain/val.txt\n"
        "test: configs/crossdomain/test.txt\n"
        "nc: 1\n"
        "names: [pothole]\n", encoding="utf-8")

    # ---- report tables ----
    def table(is_bg):
        tab = {sp: Counter() for sp in ("train", "val", "test")}
        for r in rows:
            if r["is_background"] == is_bg:
                tab[r["split"] if False else split_of_file[r["filename"]]][r["domain"]] += 1
        return tab

    def counts_block(is_bg, title):
        print(f"\n{title}")
        print(f"{'split':<7}" + "".join(f"{d:>13}" for d in DOMAINS) + f"{'total':>8}")
        for sp in ("train", "val", "test"):
            row = Counter(r["domain"] for r in rows
                          if split_of_file[r["filename"]] == sp and r["is_background"] == is_bg)
            print(f"{sp:<7}" + "".join(f"{row[d]:>13}" for d in DOMAINS) + f"{sum(row.values()):>8}")

    meta_counts = {}
    print("=== GROUPS ===")
    print(f"total groups: {len(groups)}")
    by_dom_groups = Counter(gdomain[g] for g in groups)
    print("groups by domain:", dict(by_dom_groups))
    for dom in DOMAINS:
        sizes = [gsize[g] for g in groups if gdomain[g] == dom]
        print(f"  {dom}: groups={len(sizes)} largest_group={max(sizes) if sizes else 0} "
              f"images={sum(sizes)}")

    counts_block(0, "=== POTHOLE images: split x domain ===")
    counts_block(1, "=== BACKGROUND images: split x domain ===")

    print("\n=== per-split totals ===")
    for sp in ("train", "val", "test"):
        n = sum(1 for r in rows if split_of_file[r["filename"]] == sp)
        nbg = sum(1 for r in rows if split_of_file[r["filename"]] == sp and r["is_background"] == 1)
        meta_counts[sp] = {"images": n, "background": nbg}
        print(f"  {sp}: images={n}  background={nbg} ({100*nbg/n:.1f}%)")

    # ---- asserts ----
    print("\n=== SANITY ASSERTS ===")
    # no image in >1 split (split_of_file is a dict -> inherently 1); check coverage
    assigned = set(split_of_file)
    ok_unique = len(assigned) == len(rows)
    print(f"  [{'PASS' if ok_unique else 'FAIL'}] every image in exactly one split")
    # no group spans splits
    span = [g for g in groups if len({split_of_file[m['filename']] for m in gmembers[g]}) > 1]
    print(f"  [{'PASS' if not span else 'FAIL'}] no group spans splits (offenders={len(span)})")
    # every test domain cell > 0 (pothole)
    test_dom = Counter(r["domain"] for r in rows
                       if split_of_file[r["filename"]] == "test" and r["is_background"] == 0)
    ok_cells = all(test_dom[d] > 0 for d in DOMAINS)
    print(f"  [{'PASS' if ok_cells else 'FAIL'}] every test domain has pothole images: {dict(test_dom)}")
    total = sum(meta_counts[sp]["images"] for sp in ("train", "val", "test"))
    ok_sum = total == len(rows) == 5347
    print(f"  [{'PASS' if ok_sum else 'FAIL'}] split sum == {len(rows)} (==5347): {total}")

    (OUT / "split_meta.json").write_text(json.dumps({
        "seed": SEED, "targets": TARGETS, "hamming": HAMMING,
        "grouping_policy": {"far_sources": sorted(FAR_SOURCES), "far_method": "greedy keep-first pHash<=3",
                            "closeup_sources": sorted(CLOSEUP_SOURCES), "closeup_method": "md5 file-content exact"},
        "total_images": len(rows), "total_groups": len(groups),
        "groups_by_domain": dict(by_dom_groups), "per_split": meta_counts,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote -> {OUT}/ (train/val/test.txt, manifest_split.csv, data.yaml, split_meta.json)")


if __name__ == "__main__":
    main()
