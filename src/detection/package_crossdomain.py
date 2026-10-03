"""Package the cross-domain training set into a self-contained YOLO tree + zip (Component 10.4A).

Reads the committed split manifests, copies each split's images + their EXISTING unified-set
labels (copied as-is, never re-derived) into data/crossdomain/pkg/, verifies, writes a relative-path
data.yaml, and zips to data/crossdomain/crossdomain_pkg.zip. Local only; nothing committed.

Background label convention (found in the unified set): every image has a .txt label; backgrounds
are EMPTY .txt files. Preserved as-is.
"""
from __future__ import annotations

import csv
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

UNIFIED = Path("data/crossdomain/unified")
CFG = Path("configs/crossdomain")
PKG = Path("data/crossdomain/pkg")
ZIP = Path("data/crossdomain/crossdomain_pkg.zip")
EXPECT = {"train": 3743, "val": 802, "test": 802}
EXPECT_BOXES = 8234


def main():
    man = {r["filename"]: r for r in csv.DictReader(open(CFG / "manifest_split.csv"))}

    for sp in ("train", "val", "test"):
        (PKG / "images" / sp).mkdir(parents=True, exist_ok=True)
        (PKG / "labels" / sp).mkdir(parents=True, exist_ok=True)

    per_split = defaultdict(lambda: {"img": 0, "lab": 0, "bg": 0})
    orphans = []
    class_ids = set()
    total_boxes = 0
    spot = defaultdict(list)   # (split,domain) -> [(fn, nboxes, is_bg)]

    for sp in ("train", "val", "test"):
        lines = [l for l in (CFG / f"{sp}.txt").read_text().splitlines() if l.strip()]
        for ln in lines:
            fn = Path(ln).name
            src_img = UNIFIED / "images" / fn
            src_lab = UNIFIED / "labels" / (Path(fn).stem + ".txt")
            if not src_img.exists():
                orphans.append(f"image-missing {fn}"); continue
            shutil.copy2(src_img, PKG / "images" / sp / fn)
            per_split[sp]["img"] += 1
            nboxes = 0
            is_bg = man[fn]["is_background"] == "1"
            if src_lab.exists():
                shutil.copy2(src_lab, PKG / "labels" / sp / (Path(fn).stem + ".txt"))
                per_split[sp]["lab"] += 1
                for line in src_lab.read_text().splitlines():
                    if line.split():
                        cid = int(float(line.split()[0])); class_ids.add(cid)
                        nboxes += 1; total_boxes += 1
            else:
                orphans.append(f"label-missing {fn}")
            if nboxes == 0:
                per_split[sp]["bg"] += 1
            if not is_bg and nboxes == 0:
                orphans.append(f"non-bg-but-empty-label {fn}")
            dom = man[fn]["domain"]
            if len(spot[(sp, dom)]) < 2:
                spot[(sp, dom)].append((fn, nboxes, int(is_bg)))

    # data.yaml (relative, self-describing)
    (PKG / "data.yaml").write_text(
        "# Self-contained cross-domain pothole package (relative paths). Colab-runtime data.yaml\n"
        "# with absolute paths is generated at runtime, never committed.\n"
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 1\nnames: [pothole]\n",
        encoding="utf-8")

    # ---- report ----
    print("=== per-split counts ===")
    for sp in ("train", "val", "test"):
        s = per_split[sp]
        ok = "OK" if s["img"] == EXPECT[sp] else "MISMATCH"
        print(f"  {sp}: images={s['img']} labels={s['lab']} backgrounds={s['bg']}  "
              f"[expect {EXPECT[sp]} -> {ok}]")
    assert all(per_split[sp]["img"] == EXPECT[sp] for sp in EXPECT), "image count mismatch"

    print("\n=== orphans (want none) ===")
    print(f"  {len(orphans)} orphan(s)" + ("" if not orphans else ": " + "; ".join(orphans[:10])))
    assert not orphans, "orphans found"

    print("\n=== spot-check (2 imgs per domain per split) ===")
    for sp in ("train", "val", "test"):
        for dom in ("far_dashcam", "close_up", "tiny_far", "mid"):
            for fn, nb, isbg in spot[(sp, dom)]:
                print(f"  {sp:<5} {dom:<12} boxes={nb} is_bg={isbg}  {fn[:40]}")

    print("\n=== class sanity ===")
    print(f"  class IDs present: {sorted(class_ids)}  (MUST be [0])")
    print(f"  total boxes: {total_boxes}  (expect {EXPECT_BOXES})")
    assert class_ids == {0}, f"non-single-class ids: {sorted(class_ids)}"
    assert total_boxes == EXPECT_BOXES, f"box count {total_boxes} != {EXPECT_BOXES}"

    # ---- zip ----
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(PKG.rglob("*")):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(PKG.parent)))  # pkg/... inside zip
    size = ZIP.stat().st_size
    print(f"\n=== zip ===")
    print(f"  {ZIP}  ({size:,} bytes = {size/1e6:.1f} MB)")
    with zipfile.ZipFile(ZIP) as z:
        tops = Counter(Path(n).parts[1] + "/" + Path(n).parts[2] if len(Path(n).parts) > 2 else Path(n).name
                       for n in z.namelist())
        print("  top-level structure (dir : file count):")
        for k, v in sorted(tops.items()):
            print(f"    pkg/{k}: {v}")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
