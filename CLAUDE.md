# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current State

This repository is **pre-implementation**. The only file is [project.md](project.md) — a detailed design/planning document for a final-year Computer Vision project. There is no code, no build system, no tests, and it is not yet a git repository. When implementation begins, update this file with the actual build/train/test/run commands (see "Commands" below).

`project.md` is the source of truth for scope and decisions. Read it before doing substantive work; it is the primary source that should be updated when decisions are finalized (see "Working principle" below).

## What This Project Is

**Title:** *Adaptive Tile-Based Real-Time Pothole Detection and Learned Severity Prediction using Deep Learning.*

A research-oriented extension of the base paper — Wu et al., *Object Detection Model Design for Tiny Road Surface Damage*, Scientific Reports 2025 (DOI `10.1038/s41598-025-95502-z`) — not a generic pothole classifier. The project's two contributions:

1. **Minority-class D40 detection** — improve pothole (D40) detection via operating-threshold analysis, class-aware sampling/loss weighting, and optional confidence calibration, measured on the frozen spine vs the baseline. *(Reframed 2026-09-24 — see `project.md` §22.)* Originally this was **adaptive tiling**, dropped after diagnostics showed the D40 bottleneck on RDD2022 India is confidence/imbalance, not resolution (upscaled 1280 inference did not lift small-D40 recall). The tiling work is kept as a **reported negative result**, not deleted.
2. **Learned severity prediction** — a CNN head that scores each detected pothole into a tier (low/medium/high). **Unchanged, and now the primary novelty.**

## Architecture (the one decision that shapes everything)

**Two-stage, decoupled design — not an end-to-end multi-task model.** This is deliberate and data-forced, and code should preserve it:

- **Detector** (YOLO11, Ultralytics) trains on **RDD2022** (boxes, no severity labels).
- **Severity head** (lightweight CNN, e.g. ResNet-18/MobileNet) trains on a **separate mask/depth dataset** (Pothole Mix / SHREC2022, or PothRGBD).

The two datasets are disjoint (RDD2022 has boxes but no severity; the severity set has masks/depth but isn't RDD imagery), so a fused model would need a dataset labeled for both, which doesn't exist. Each component must train on the dataset that actually holds its labels. Neither training dataset is present at inference.

**Inference pipeline:** image/frame → YOLO11 detector → severity head per detected crop → boxes + severity + aggregate summary. (No tiling stage — see the Contribution 1 pivot above.)

## Confirmed Stack (do not silently swap these)

- **Language/framework:** Python, PyTorch.
- **Detection:** Ultralytics (YOLO11), OpenCV + NumPy. *(SAHI is no longer a build dependency — tiling was dropped; see the Contribution 1 pivot.)*
- **Severity head:** PyTorch/torchvision; scikit-learn for tier metrics (accuracy, confusion matrix, F1).
- **Data prep:** RDD2022 ships PASCAL VOC XML; YOLO needs normalized `.txt` — a one-time conversion is required (Roboflow or a script).
- YOLO26 is an *optional* modern baseline to also report; YOLO11 is primary.

## Environment & Workflow

- Use a local virtual environment at `.venv`. Install and run everything through it. On Windows, call the venv's interpreter directly (`.venv\Scripts\python.exe`, `.venv\Scripts\python.exe -m pip ...`) rather than relying on shell activation persisting between commands.
- **Build one component at a time**, in this order: dataset prep (done) → baseline detector (done) → minority-class detection improvements → severity head → end-to-end pipeline → demo → results. Do NOT scaffold or implement the whole system in a single pass.
- Keep **datasets and model weights out of git** (large, re-downloadable). `.gitignore` covers `.venv/`, `data/`, `*.pt`, `runs/`, caches.
- After writing any script, **run it on 1–2 sample inputs before scaling up**. Heavy training runs on Colab/Kaggle (free GPU); the local machine is for development, inference, and tests.

## Code Conventions

- **Preserve the author's own variable names and code structure** when editing existing code — do not rename or restructure without a clearly stated reason.
- Keep modules small and mirroring the architecture (`detection` / `severity` / `pipeline`); prefer readable code over cleverness.

## Non-Negotiable Constraints

- **Preserve the confirmed research direction.** Do not replace it with a simpler generic detector without a strong, stated reason.
- **Pothole (D40) is a minority class in RDD2022** (~2,914 pothole instances in a typical 10k sample). Any detector work must account for this: pothole-heavy country subsets, class-aware augmentation/oversampling, optionally topping up from HRP4K.
- **Novelty must be framed honestly.** The primary novelty is the **learned severity head**, NOT claiming severity estimation is first-ever — the contribution is its integration into a practical pothole-inspection pipeline. For Contribution 1, the honest framing is *minority-class D40 detection improvement over a measured baseline*, plus the **tiling negative result** as a genuine finding (see `project.md` §22). Do not re-introduce adaptive tiling as a contribution without new evidence.
- **Evaluation is the point, not just detection.** Success = measurable D40 detection improvement over the **0.391 AP@50 baseline** on the frozen spine (operating-threshold / class-aware sampling / calibration, size-stratified via `eval_stratified.py`) AND higher severity accuracy than a heuristic baseline (e.g. box-area thresholds — beating this answers the "isn't this just box area?" objection). Report expected results as targets to validate, not as results already obtained.

## Deliberately Not Yet Decided

Do **not** assume these are settled (see `project.md` §19): exact severity labels/annotation source/thresholds; training hyperparameters; the specific minority-class technique(s) to adopt; compute/environment; deployment platform and UI framework (Streamlit/Gradio/Flask/FastAPI candidates); final folder structure. Decide these during implementation based on data and results — and record the decision in `project.md`.

## Working Principle

When a currently-open item is finalized, record it in `project.md` (it already tracks this via its Section 12 → Section 19 status pattern) so the decision survives across sessions. Preserve confirmed decisions; build on them rather than re-litigating them.

## Commands

_To be filled in once implemented. Placeholders:_

- **Env setup:** `python -m venv .venv` then install `requirements.txt`
- **Convert dataset (VOC → YOLO):** `.venv\Scripts\python.exe src\detection\voc_to_yolo.py --src data\raw\India --dst data\rdd2022_india`
- **Build/freeze eval spine:** `.venv\Scripts\python.exe src\detection\build_eval_split.py`
- **Train baseline detector:** Colab — `notebooks/train_baseline.ipynb` (T4)
- **Stratified eval (frozen test):** `python src\detection\eval_stratified.py --weights <best.pt>`
- **Minority-class detection experiments:** _TBD_
- **Train severity head:** _TBD_
- **Run end-to-end demo:** _TBD_
- **Tests:** _TBD_
