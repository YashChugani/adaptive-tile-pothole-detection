# Adaptive Tile-Based Real-Time Pothole Detection and Learned Severity Prediction

Final-year Computer Vision project. A two-stage, decoupled system that (1) detects
potholes in road imagery with **YOLO11 + adaptive tiling** (built on SAHI) to
recover small/distant potholes, and (2) scores each detected pothole into a
**severity tier** with a learned CNN head.

See [project.md](project.md) for the full design and [CLAUDE.md](CLAUDE.md) for
working conventions.

## Structure

```
src/tiling/      adaptive tiling controller (SAHI-based)
src/detection/   YOLO11 detector + dataset prep
src/severity/    learned severity head (CNN)
src/pipeline.py  end-to-end inference
configs/         config.yaml (paths, model settings)
data/            datasets (gitignored)
notebooks/       exploration
tests/           tests
app/             demo UI (TBD)
```

## Setup (Windows)

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Call the venv interpreter directly (`.venv\Scripts\python.exe`) rather than
relying on shell activation.

## Status

Scaffolding + environment only. No datasets downloaded and nothing trained yet.
Build order: dataset prep → baseline detector → adaptive tiling → severity head →
end-to-end pipeline + demo.
