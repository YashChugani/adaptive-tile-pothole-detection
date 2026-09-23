# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current State

This repository is **pre-implementation**. The only file is [project.md](project.md) — a detailed design/planning document for a final-year Computer Vision project. There is no code, no build system, no tests, and it is not yet a git repository. When implementation begins, update this file with the actual build/train/test/run commands (see "Commands" below).

`project.md` is the source of truth for scope and decisions. Read it before doing substantive work; it is the primary source that should be updated when decisions are finalized (see "Working principle" below).

## What This Project Is

**Title:** *Adaptive Tile-Based Real-Time Pothole Detection and Learned Severity Prediction using Deep Learning.*

A research-oriented extension of the base paper — Wu et al., *Object Detection Model Design for Tiny Road Surface Damage*, Scientific Reports 2025 (DOI `10.1038/s41598-025-95502-z`) — not a generic pothole classifier. The project's two confirmed contributions over the base paper are what everything else serves:

1. **Adaptive tiling** — conditionally/content-triggered slicing to detect small/distant potholes, built on top of SAHI. Framing: *uniform SAHI = baseline; adaptive/triggered tiling = the contribution.*
2. **Learned severity prediction** — a CNN head that scores each detected pothole into a tier (low/medium/high).

## Architecture (the one decision that shapes everything)

**Two-stage, decoupled design — not an end-to-end multi-task model.** This is deliberate and data-forced, and code should preserve it:

- **Detector** (YOLO11, Ultralytics) trains on **RDD2022** (boxes, no severity labels).
- **Severity head** (lightweight CNN, e.g. ResNet-18/MobileNet) trains on a **separate mask/depth dataset** (Pothole Mix / SHREC2022, or PothRGBD).

The two datasets are disjoint (RDD2022 has boxes but no severity; the severity set has masks/depth but isn't RDD imagery), so a fused model would need a dataset labeled for both, which doesn't exist. Each component must train on the dataset that actually holds its labels. Neither training dataset is present at inference.

**Inference pipeline:** image/frame → adaptive tiling *controller* (decides whether/how to tile) → YOLO11 on full image + tiles → cross-tile NMS merge → severity head per detected crop → boxes + severity + aggregate summary. Adaptive triggering is both an accuracy technique *and* what keeps the system real-time (tile only when needed rather than slicing every frame).

## Confirmed Stack (do not silently swap these)

- **Language/framework:** Python, PyTorch.
- **Detection + tiling:** Ultralytics (YOLO11), SAHI, OpenCV + NumPy.
- **Severity head:** PyTorch/torchvision; scikit-learn for tier metrics (accuracy, confusion matrix, F1).
- **Data prep:** RDD2022 ships PASCAL VOC XML; YOLO needs normalized `.txt` — a one-time conversion is required (Roboflow or a script).
- YOLO26 is an *optional* modern baseline to also report; YOLO11 is primary.

## Environment & Workflow

- Use a local virtual environment at `.venv`. Install and run everything through it. On Windows, call the venv's interpreter directly (`.venv\Scripts\python.exe`, `.venv\Scripts\python.exe -m pip ...`) rather than relying on shell activation persisting between commands.
- **Build one component at a time**, in this order: dataset prep → baseline detector → adaptive tiling → severity head → end-to-end pipeline + demo. Do NOT scaffold or implement the whole system in a single pass.
- Keep **datasets and model weights out of git** (large, re-downloadable). `.gitignore` covers `.venv/`, `data/`, `*.pt`, `runs/`, caches.
- After writing any script, **run it on 1–2 sample inputs before scaling up**. Heavy training runs on Colab/Kaggle (free GPU); the local machine is for development, inference, and tests.

## Code Conventions

- **Preserve the author's own variable names and code structure** when editing existing code — do not rename or restructure without a clearly stated reason.
- Keep modules small and mirroring the architecture (`tiling` / `detection` / `severity` / `pipeline`); prefer readable code over cleverness.

## Non-Negotiable Constraints

- **Preserve the confirmed research direction.** Do not replace it with a simpler generic detector without a strong, stated reason.
- **Pothole (D40) is a minority class in RDD2022** (~2,914 pothole instances in a typical 10k sample). Any detector work must account for this: pothole-heavy country subsets, class-aware augmentation/oversampling, optionally topping up from HRP4K.
- **Novelty must be framed honestly.** The gap is the *integration* (adaptive tiling for tiny potholes + learned severity in one real-time pipeline), NOT claiming severity estimation or tiny-object detection is first-ever. ASAHI (2026) and YOLO26 already narrow the isolated "adaptive tiling" claim — cite them as related work to differentiate from.
- **Evaluation is the point, not just detection.** Success = measurable small-pothole recall gain from adaptive tiling (size-stratified ablation vs no-tiling and vs uniform SAHI, keeping inference near real-time) AND higher severity accuracy than a heuristic baseline (e.g. box-area thresholds — beating this answers the "isn't this just box area?" objection). Report expected results as targets to validate, not as results already obtained.

## Deliberately Not Yet Decided

Do **not** assume these are settled (see `project.md` §19): exact tile size/overlap and trigger policy; exact severity labels/annotation source/thresholds; training hyperparameters; compute/environment; deployment platform and UI framework (Streamlit/Gradio/Flask/FastAPI candidates); final folder structure. Decide these during implementation based on data and results — and record the decision in `project.md`.

## Working Principle

When a currently-open item is finalized, record it in `project.md` (it already tracks this via its Section 12 → Section 19 status pattern) so the decision survives across sessions. Preserve confirmed decisions; build on them rather than re-litigating them.

## Commands

_To be filled in once implemented. Placeholders:_

- **Env setup:** `python -m venv .venv` then install `requirements.txt`
- **Convert dataset (VOC → YOLO):** _TBD_
- **Train detector:** _TBD_
- **Run inference / adaptive tiling:** _TBD_
- **Train severity head:** _TBD_
- **Run end-to-end demo:** _TBD_
- **Tests:** _TBD_
