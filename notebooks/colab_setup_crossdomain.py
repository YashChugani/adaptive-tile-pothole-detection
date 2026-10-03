"""One-call Colab setup for the CROSS-DOMAIN single-class detector (Component 10.4B).

Mirrors notebooks/colab_setup.py. Run on Colab after cloning the repo + installing deps:

    import colab_setup_crossdomain as cs
    cfg = cs.setup()      # edit the CONFIG block below if your paths differ
    # cfg["data_yaml"] -> /content runtime yaml; cfg["project"] -> Drive checkpoint dir

setup(): mounts Drive, copies crossdomain_pkg.zip to /content (local SSD), unzips to
/content/crossdomain_pkg/, GENERATES an absolute-path runtime data.yaml there (nc:1,
names:[pothole]; never committed), pre-places Arial.ttf offline-first, returns the Drive
checkpoint dir, and prints a summary. Train off the /content SSD copy, not off Drive.
"""
from __future__ import annotations

import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

# ============================ CONFIG (edit this block only) ============================
DRIVE_BASE = "/content/drive/MyDrive/pothole"              # your Drive project folder
REPO_DIR = "/content/Pothole-Detection-System"            # where the repo is cloned
PKG_ZIP_DRIVE = f"{DRIVE_BASE}/crossdomain_pkg.zip"       # the one-time uploaded package zip
CHECKPOINT_DIR = f"{DRIVE_BASE}/checkpoints/crossdomain"  # Drive checkpoints (NOT baseline's dir)
LOCAL_ROOT = "/content/crossdomain_pkg"                   # local SSD dataset root
# =====================================================================================

ARIAL_SYSTEM_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]
ARIAL_URLS = [
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf",
    "https://ultralytics.com/assets/Arial.ttf",
]


def _mount_drive() -> None:
    from google.colab import drive
    drive.mount("/content/drive")


def _copy_and_unzip() -> Path:
    """Copy the zip to local SSD and unzip, stripping the leading 'pkg/' so the dataset root is
    LOCAL_ROOT (images/{train,val,test}, labels/{train,val,test} directly under it)."""
    root = Path(LOCAL_ROOT)
    local_zip = Path("/content/crossdomain_pkg.zip")
    if not Path(PKG_ZIP_DRIVE).exists():
        raise FileNotFoundError(
            f"package zip not found on Drive: {PKG_ZIP_DRIVE}\n"
            f"Upload data/crossdomain/crossdomain_pkg.zip (from package_crossdomain.py) to that path first.")
    print(f"copying {PKG_ZIP_DRIVE} -> {local_zip}")
    shutil.copy(PKG_ZIP_DRIVE, local_zip)
    root.mkdir(parents=True, exist_ok=True)
    print(f"unzipping into {root}")
    with zipfile.ZipFile(local_zip) as z:
        for m in z.namelist():
            if m.endswith("/"):
                continue
            rel = m[len("pkg/"):] if m.startswith("pkg/") else m
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    return root


def _generate_yaml(root: Path) -> Path:
    """Runtime data.yaml with ABSOLUTE /content paths (never committed)."""
    out = root / "data_colab.yaml"
    out.write_text(
        "# GENERATED at runtime on Colab -- absolute /content paths, never committed.\n"
        f"train: {root}/images/train\n"
        f"val: {root}/images/val\n"
        f"test: {root}/images/test\n"
        "nc: 1\n"
        "names: [pothole]\n", encoding="utf-8")
    return out


def _ensure_arial() -> Path:
    """Offline-first Arial.ttf (primary: copy a preinstalled system font; fallback: download)."""
    from ultralytics.utils import USER_CONFIG_DIR
    dest = Path(USER_CONFIG_DIR) / "Arial.ttf"
    if dest.exists():
        print(f"Arial.ttf: already present at {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for fnt in ARIAL_SYSTEM_FONTS:
        if Path(fnt).exists():
            shutil.copy(fnt, dest)
            print(f"Arial.ttf: copied system font {fnt} (PRIMARY, offline)")
            return dest
    for url in ARIAL_URLS:
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"Arial.ttf: downloaded from {url} (fallback)")
            return dest
        except Exception as e:  # noqa: BLE001
            print(f"  Arial download failed ({url}): {e}")
    print(f"WARNING: could not provide Arial.ttf; upload one to {dest} if needed.")
    return dest


def _commit_hash(repo: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", repo, "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def setup(checkpoint_dir: str = CHECKPOINT_DIR) -> dict:
    _mount_drive()
    root = _copy_and_unzip()
    data_yaml = _generate_yaml(root)
    _ensure_arial()
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    counts = {sp: len(list((root / "images" / sp).glob("*"))) for sp in ("train", "val", "test")}
    commit = _commit_hash(REPO_DIR)
    print("\n=== cross-domain setup summary ===")
    print(f"repo commit    : {commit}")
    print(f"dataset root   : {root}  (local SSD)")
    print(f"images/split   : {counts}")
    print(f"data.yaml      : {data_yaml}")
    print(f"checkpoint dir : {checkpoint_dir}  (Drive; separate from baseline)")
    return {"data_yaml": str(data_yaml), "project": checkpoint_dir, "root": str(root),
            "commit": commit, "counts": counts}


if __name__ == "__main__":
    setup()
