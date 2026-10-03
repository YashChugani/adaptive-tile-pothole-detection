# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. It is **working guidance**, not a results doc — consolidated results live in [docs/RESULTS.md](docs/RESULTS.md).

## Current State

This is a **fully implemented codebase** in an active **git repository with history** (pushed to GitHub). Both contributions are complete:

1. **Cross-domain single-class pothole detection** (Contribution 1).
2. **Depth-grounded, calibration-aware severity** (Contribution 2).

Three docs, by role:
- **[project.md](project.md)** — the dated decision log and **source of truth** for scope, decisions, and every recorded number. Read it before substantive work; record new decisions there.
- **[docs/RESULTS.md](docs/RESULTS.md)** — consolidated results (claim → evidence → caveat, incl. negatives).
- **[README.md](README.md)** — orientation / front door (install, run, one headline result).

## What This Project Is

**Title:** *Robust Cross-Domain Pothole Detection with Depth-Grounded, Calibration-Aware Severity.*

> The **old title** — *"Adaptive Tile-Based Real-Time Pothole Detection and Learned Severity Prediction"* — is **superseded**. Adaptive tiling was **dropped on evidence** (project.md §22: resolution was not the bottleneck) and is now a **reported negative result**, NOT a current component. Do not re-introduce tiling/SAHI as a contribution without new evidence.

A research-oriented project (base paper: Wu et al., *Object Detection Model Design for Tiny Road Surface Damage*, Scientific Reports 2025, DOI `10.1038/s41598-025-95502-z`). Two contributions:

1. **Cross-domain detection** — a single YOLO11s detector, single-class (`pothole`=0), trained across **4 capture domains** (far/dashcam, close-up, tiny/distant, mid) and evaluated per-domain. Robust where a single-dataset (India-only) detector collapses (see RESULTS.md §1).
2. **Depth-grounded, calibration-aware severity** — a ResNet-18 head grades each detected pothole crop into a tier (low/med/high) from depth-derived labels, beats a box-area heuristic, and is shown **only in its validated close-range regime** via a calibration gate (out-of-range detections marked uncalibrated, not faked).

## Architecture (the one decision that shapes everything)

**Two-stage, decoupled design — not an end-to-end multi-task model.** This is deliberate and data-forced, and code should preserve it:

- **Detector** (YOLO11s, Ultralytics) trains on the **pooled 4-domain single-class set** (boxes, no severity labels).
- **Severity head** (ResNet-18) trains on **PothRGBD** (depth-derived tier labels; RGB-crop-only input).

The two training datasets are disjoint (the detection pool has boxes but no severity; PothRGBD has depth/severity but isn't the detection pool), so a fused model would need a dataset labeled for both, which doesn't exist. Each component trains on the dataset that holds its labels. **Neither training set is present at inference.**

**Inference pipeline (current):**
image/frame → **cross-domain YOLO11s** (single-class) → per detection: crop (0.15 margin) → 224×224 → **ResNet-18 severity head** → low/med/high → **calibration gate** (box-area-fraction ≥ `0.123` → calibrated tier in green/amber/red; else **uncalibrated**, cyan box, `* uncal`, confidence dropped) → annotated image + summary (calibrated vs uncalibrated tier counts). **No tiling stage.**

## Confirmed Stack (do not silently swap these)

- **Language/framework:** Python, PyTorch.
- **Detection:** Ultralytics (YOLO11s), OpenCV + NumPy — **single-class** cross-domain. *(SAHI is NOT a build dependency — tiling was dropped.)*
- **Severity head:** PyTorch/torchvision (ResNet-18); scikit-learn for tier metrics + Spearman.
- **Demo:** Gradio (`app/demo.py`).

## Environment & Workflow

- Use a local virtual environment at `.venv`. Install and run everything through it. On Windows, call the venv's interpreter directly (`.venv\Scripts\python.exe`, `.venv\Scripts\python.exe -m pip ...`) rather than relying on shell activation persisting between commands.
- **Local torch is CPU-only.** The local machine is for development, inference, light tests, and the small CPU severity train; **heavy detector training runs on Colab (T4)** — see `notebooks/`.
- **Colab gotchas (enforced in the notebooks — keep them):** install **only** `ultralytics==8.4.160 sahi==0.12.6` and keep Colab's GPU torch (pinning torch risks a CUDA mismatch → silent CPU); assert `torch.cuda.is_available()`; pre-place **Arial.ttf offline-first** (copy a system TTF into Ultralytics' config dir, download only as fallback — the download hangs/fails and otherwise aborts even training).
- After writing any script, **run it on 1–2 sample inputs before scaling up.**
- Keep **datasets and model weights out of git** (large, re-downloadable). `.gitignore` covers `.venv/`, `data/`, `*.pt`, `runs/`, `configs/_*.yaml`, caches. Only **code + small frozen manifests** (`configs/splits/`, `configs/crossdomain/`) are tracked.

## Code Conventions

- **Preserve the author's own variable names and code structure** when editing existing code — do not rename or restructure without a clearly stated reason.
- Keep modules small and mirroring the architecture (`detection` / `severity` / `pipeline`); prefer readable code over cleverness.
- pHash: discriminative on far/varied/dashcam scenes; NON-discriminative whenever the pothole is uniform or a tiny fraction of the frame (close-ups = "dark blob on gray"). For close-up datasets (PothRGBD) use file-content hash (md5) for exact-dup, NOT pHash — pHash Hamming==0 still collapsed 182 distinct close-ups. For far/varied/dashcam within-source use greedy keep-first pHash<=3, never single-linkage (chaining).

## Non-Negotiable Constraints

- **Preserve the two confirmed contributions** (cross-domain detection + depth-grounded calibration-aware severity) and the decoupled two-stage design. Don't collapse to a simpler generic detector without a strong, stated reason.
- **Honesty-first.** Frame negatives as findings, not apologies (tiling, oversampling, cross-sensor severity are reported, not hidden — RESULTS.md §4). Never present an out-of-validated-range severity as a confident grade (that's what the calibration gate enforces). Don't claim severity estimation is first-ever — the contribution is its integration + honest calibration.
- **Evaluation is the point.** Comparisons use our own stratified matcher applied identically to both sides (`eval_stratified.py`, `eval_crossdomain.py`), GT deduplicated before scoring, per-domain breakdowns with n and caveats.

## Working Principle

When a decision is made or an open item finalized, **record it in `project.md`** (dated-section decision log) so it survives across sessions. Consolidate results into `docs/RESULTS.md` (pull figures from project.md; don't recompute). Preserve confirmed decisions; build on them rather than re-litigating them.

## Commands

All via the venv interpreter; heavy training on Colab.

**Env setup**
- `python -m venv .venv` then `.venv\Scripts\python.exe -m pip install -r requirements.txt`

**India detector spine (Components 1–5, still real)**
- Convert VOC→YOLO: `.venv\Scripts\python.exe src\detection\voc_to_yolo.py --src data\raw\India --dst data\rdd2022_india`
- Build/freeze eval spine: `.venv\Scripts\python.exe src\detection\build_eval_split.py`
- Size-stratified eval: `.venv\Scripts\python.exe src\detection\eval_stratified.py --weights <best.pt>`

**Cross-domain detector (Component 10)**
- Inspect sources: `.venv\Scripts\python.exe src\detection\inspect_crossdomain.py`
- Build unified pool: `.venv\Scripts\python.exe src\detection\build_crossdomain.py`
- Freeze split: `.venv\Scripts\python.exe src\detection\build_crossdomain_split.py`
- Package for Colab: `.venv\Scripts\python.exe src\detection\package_crossdomain.py`
- Train: Colab — `notebooks/train_crossdomain.ipynb` (+ `notebooks/colab_setup_crossdomain.py`), T4
- Per-domain eval: `.venv\Scripts\python.exe src\detection\eval_crossdomain.py`

**Severity head (Component 6)**
- Depth→tier labels: `.venv\Scripts\python.exe src\severity\depth_to_tier.py`
- Crops + heuristic: `src\severity\crops.py`, `src\severity\heuristic.py`
- Train: Colab `notebooks/train_severity.ipynb`, or local CPU `.venv\Scripts\python.exe src\severity\train_severity_local.py`
- Calibration-gate threshold diagnostic: `.venv\Scripts\python.exe src\detection\diag_severity_gate.py`

**End-to-end**
- Run pipeline on an image: `.venv\Scripts\python.exe src\pipeline.py --image <path>`
- Run demo: `.venv\Scripts\python.exe app\demo.py` → open http://127.0.0.1:7860

> Running the pipeline/demo needs the gitignored weights in place: `data/weights/crossdomain_best.pt` (Colab `checkpoints/crossdomain`) and `data/severity/cpu_ckpts/severity_best_finetune_all.pt` (severity training).
