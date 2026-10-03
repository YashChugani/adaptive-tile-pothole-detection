"""Close-up near-duplicate leakage check (Component 10.3 pre-commit). Diagnostic only.

Concern: PothRGBD close_up uses md5-exact dedup (pHash unreliable on close-ups). Non-byte-identical
near-dup frames of the same scene could straddle train/test and inflate close_up test metrics.
Here we use PIXEL nearest-neighbor (32x32 gray, L2-normalized) over the 996 close_up images, report
the NN-distance distribution + the 10 closest pairs (with their splits), and save a visual panel.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

UNIFIED = Path("data/crossdomain/unified")
SPLIT_MANIFEST = Path("configs/crossdomain/manifest_split.csv")
PANEL = Path("data/crossdomain/_verify/closeup_nn_pairs.png")


def main():
    split_of = {r["filename"]: r["split"] for r in csv.DictReader(open(SPLIT_MANIFEST))
                if r["source"] == "pothrgbd"}
    files = sorted((UNIFIED / "images").glob("pothrgbd_*.jpg"))
    files = [f for f in files if f.name in split_of]
    print(f"close_up images: {len(files)}")

    # 1. filename structure
    print("\n=== 1. filename structure (15 samples) ===")
    for f in files[:15]:
        print("  ", f.name)
    ts = [m.group(1) for f in files if (m := re.search(r"pothrgbd_(\d{8}_\d{6})", f.name))]
    print(f"  parsed timestamps: {len(ts)} / {len(files)}  unique_timestamps={len(set(ts))}")
    print("  pattern: pothrgbd_<YYYYMMDD_HHMMSS>_color_png.rf.<roboflow_hash>.jpg  "
          "-> timestamp-based IDs (capture time) + Roboflow export hash; check uniqueness above.")

    # 2. pixel NN (32x32 gray, L2-normalized)
    vecs = np.zeros((len(files), 1024), np.float32)
    for i, f in enumerate(files):
        g = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2GRAY)
        v = cv2.resize(g, (32, 32)).astype(np.float32).flatten()
        vecs[i] = v / (np.linalg.norm(v) + 1e-8)
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(vecs)
    dist, idx = nn.kneighbors(vecs)
    nn_dist = dist[:, 1]       # nearest OTHER
    nn_idx = idx[:, 1]
    print("\n=== 2. nearest-neighbor L2 distance distribution ===")
    pct = {k: float(np.percentile(nn_dist, k)) for k in (0, 1, 5, 25, 50)}
    print(f"  min={nn_dist.min():.4f}  p1={pct[1]:.4f}  p5={pct[5]:.4f}  p25={pct[25]:.4f}  median={pct[50]:.4f}")
    for thr in (0.05, 0.10, 0.20, 0.30):
        print(f"  images with NN-dist < {thr:.2f}: {int((nn_dist < thr).sum())}")

    # 3. 10 closest unique pairs
    pairs = {}
    for i in range(len(files)):
        a, b = sorted((i, int(nn_idx[i])))
        pairs[(a, b)] = min(pairs.get((a, b), 9e9), float(nn_dist[i]))
    top = sorted(pairs.items(), key=lambda kv: kv[1])[:10]
    print("\n=== 3. 10 closest close_up pairs (fileA, fileB, dist, splits) ===")
    straddle = 0
    for (a, b), d in top:
        sa, sb = split_of[files[a].name], split_of[files[b].name]
        flag = "  <<< STRADDLES SPLIT" if sa != sb else ""
        if sa != sb:
            straddle += 1
        print(f"  {d:.4f}  {files[a].name[:34]} [{sa}]  <->  {files[b].name[:34]} [{sb}]{flag}")
    print(f"\n  pairs straddling different splits (of top 10): {straddle}")

    # visual panel
    fig, axes = plt.subplots(10, 2, figsize=(6, 28))
    for row, ((a, b), d) in enumerate(top):
        for col, k in ((0, a), (1, b)):
            ax = axes[row, col]
            ax.imshow(cv2.cvtColor(cv2.imread(str(files[k])), cv2.COLOR_BGR2RGB))
            ax.set_title(f"d={d:.3f} [{split_of[files[k].name]}]", fontsize=8)
            ax.axis("off")
    PANEL.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(PANEL, dpi=90); plt.close()
    print(f"\nsaved panel -> {PANEL}")


if __name__ == "__main__":
    main()
