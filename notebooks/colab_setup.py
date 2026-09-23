"""One-call Colab environment setup for the RDD2022-India YOLO11s baseline.

Run on Colab (after cloning the repo and pip-installing requirements):

    import colab_setup
    cfg = colab_setup.setup()          # edit the CONFIG block below if your paths differ
    # cfg["data_yaml"], cfg["project"] feed straight into model.train(...)

What setup() does, in order:
  a. mounts Google Drive;
  b. copies the data zip from Drive to /content and unzips into <repo>/data/rdd2022_india
     (training reads from local SSD, never from Drive);
  c. generates a Colab-local data.yaml (configs/_rdd_india_colab.yaml, gitignored) from the
     committed class names + frozen split lists, with paths pointing at the local data;
  d. pre-places Arial.ttf in the Ultralytics config dir (avoids the check_font download hang);
  e. sets/returns an Ultralytics project= dir on Drive (checkpoint persistence + resume);
  f. prints a summary (commit hash, resolved image/label counts, yaml path, checkpoint dir).

Nothing here is committed except this file; the generated yaml and data stay gitignored.
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
DATA_ZIP_DRIVE = f"{DRIVE_BASE}/rdd2022_india.zip"     # the one-time uploaded data zip
CHECKPOINT_DIR = f"{DRIVE_BASE}/checkpoints"           # Ultralytics project= (on Drive)
# =====================================================================================

COMMITTED_YAML = "configs/rdd_india.yaml"              # class names come from here
COLAB_YAML = "configs/_rdd_india_colab.yaml"          # generated, gitignored
SPLIT_DIR = "configs/splits"                           # frozen train/val/test lists
DATASET_SUBDIR = "data/rdd2022_india"                 # images/ + labels/ land here
ARIAL_URLS = [
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf",
    "https://ultralytics.com/assets/Arial.ttf",
]
ARIAL_FALLBACK_FONTS = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _mount_drive() -> None:
    from google.colab import drive  # only available on Colab
    drive.mount("/content/drive")


def _copy_and_unzip(repo: Path) -> Path:
    dst_root = repo / DATASET_SUBDIR
    local_zip = Path("/content/rdd2022_india.zip")
    if not Path(DATA_ZIP_DRIVE).exists():
        raise FileNotFoundError(
            f"data zip not found on Drive: {DATA_ZIP_DRIVE}\n"
            f"Upload data/rdd2022_india.zip (from package_dataset.py) to that path first.")
    print(f"copying {DATA_ZIP_DRIVE} -> {local_zip}")
    shutil.copy(DATA_ZIP_DRIVE, local_zip)
    dst_root.mkdir(parents=True, exist_ok=True)
    print(f"unzipping into {dst_root}")
    with zipfile.ZipFile(local_zip) as z:
        z.extractall(dst_root)
    return dst_root


def _generate_yaml(repo: Path) -> Path:
    import yaml  # provided by ultralytics/pyyaml
    names = yaml.safe_load((repo / COMMITTED_YAML).read_text())["names"]
    out = repo / COLAB_YAML
    lines = [
        "# GENERATED on Colab by colab_setup.py -- do not commit (gitignored).",
        f"path: {repo.as_posix()}",
        f"train: {SPLIT_DIR}/train.txt",
        f"val: {SPLIT_DIR}/val.txt",
        f"test: {SPLIT_DIR}/test.txt",
        f"nc: {len(names)}",
        "names:",
    ]
    lines += [f"  {k}: {v}" for k, v in names.items()]
    out.write_text("\n".join(lines) + "\n")
    return out


def _ensure_arial() -> Path:
    from ultralytics.utils import USER_CONFIG_DIR
    dest = Path(USER_CONFIG_DIR) / "Arial.ttf"
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in ARIAL_URLS:
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"fetched Arial.ttf from {url}")
            return dest
        except Exception as e:  # noqa: BLE001
            print(f"  Arial download failed ({url}): {e}")
    for fnt in ARIAL_FALLBACK_FONTS:
        if Path(fnt).exists():
            shutil.copy(fnt, dest)
            print(f"copied fallback font {fnt} -> {dest} (metric-compatible stand-in)")
            return dest
    print(f"WARNING: could not provide Arial.ttf; upload one to {dest} if check_font fails.")
    return dest


def _count(dst_root: Path) -> tuple[int, int]:
    return (len(list((dst_root / "images").glob("*"))),
            len(list((dst_root / "labels").glob("*.txt"))))


def _commit_hash(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def setup(repo_dir: str = REPO_DIR, checkpoint_dir: str = CHECKPOINT_DIR) -> dict:
    repo = Path(repo_dir)
    _mount_drive()
    dst_root = _copy_and_unzip(repo)
    data_yaml = _generate_yaml(repo)
    _ensure_arial()
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    n_img, n_lab = _count(dst_root)
    commit = _commit_hash(repo)
    print("\n=== Colab setup summary ===")
    print(f"repo commit    : {commit}")
    print(f"images/labels  : {n_img} / {n_lab}  (under {dst_root})")
    print(f"data.yaml      : {data_yaml}")
    print(f"checkpoint dir : {checkpoint_dir}  (Ultralytics project=)")
    return {
        "data_yaml": str(data_yaml),
        "project": checkpoint_dir,
        "images": n_img,
        "labels": n_lab,
        "commit": commit,
        "dataset_root": str(dst_root),
    }


if __name__ == "__main__":
    setup()
