"""One-call Colab setup for the severity CNN (Component 6.3B).

Run on Colab after cloning the repo and installing deps:

    import severity_setup
    cfg = severity_setup.setup()     # edit the CONFIG block below if your paths differ
    # cfg["crops_root"] -> ImageFolder parents; cfg["project"] -> Drive checkpoint dir

setup() mounts Drive, copies the crops zip to /content and unzips into <repo>/data/severity
(training reads from local SSD, not Drive), pre-places Arial.ttf offline-first (same fix as
the detector), creates/returns a Drive checkpoint dir, and prints a summary. Nothing here
is committed except this file; crops/weights stay gitignored.
"""
from __future__ import annotations

import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

# ============================ CONFIG (edit this block only) ============================
DRIVE_BASE = "/content/drive/MyDrive/pothole"          # your Drive project folder
REPO_DIR = "/content/Pothole-Detection-System"         # where the repo is cloned
CROPS_ZIP_DRIVE = f"{DRIVE_BASE}/severity_crops.zip"   # the one-time uploaded crops zip
CHECKPOINT_DIR = f"{DRIVE_BASE}/severity_checkpoints"  # severity-head checkpoints (on Drive)
# =====================================================================================

CROPS_SUBDIR = "data/severity"                         # crops/ lands here -> data/severity/crops
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


def _copy_and_unzip(repo: Path) -> Path:
    dst = repo / CROPS_SUBDIR
    local_zip = Path("/content/severity_crops.zip")
    if not Path(CROPS_ZIP_DRIVE).exists():
        raise FileNotFoundError(
            f"crops zip not found on Drive: {CROPS_ZIP_DRIVE}\n"
            f"Upload data/severity_crops.zip (from package_crops.py) to that path first.")
    print(f"copying {CROPS_ZIP_DRIVE} -> {local_zip}")
    shutil.copy(CROPS_ZIP_DRIVE, local_zip)
    dst.mkdir(parents=True, exist_ok=True)
    print(f"unzipping into {dst}")
    with zipfile.ZipFile(local_zip) as z:
        z.extractall(dst)
    return dst / "crops"


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
            print(f"Arial.ttf: copied preinstalled system font {fnt} (PRIMARY, offline)")
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


def _commit_hash(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _counts(crops_root: Path) -> dict:
    out = {}
    for split in ("pothrgbd/train", "pothrgbd/val", "pothrgbd/test", "pothole440"):
        base = crops_root / split
        if base.exists():
            out[split] = {t.name: len(list(t.glob("*.jpg"))) for t in sorted(base.iterdir()) if t.is_dir()}
    return out


def setup(repo_dir: str = REPO_DIR, checkpoint_dir: str = CHECKPOINT_DIR) -> dict:
    repo = Path(repo_dir)
    _mount_drive()
    crops_root = _copy_and_unzip(repo)
    _ensure_arial()
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    counts = _counts(crops_root)
    commit = _commit_hash(repo)
    print("\n=== Severity setup summary ===")
    print(f"repo commit    : {commit}")
    print(f"crops root     : {crops_root}")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"checkpoint dir : {checkpoint_dir}")
    return {"crops_root": str(crops_root), "project": checkpoint_dir, "commit": commit, "counts": counts}


if __name__ == "__main__":
    setup()
