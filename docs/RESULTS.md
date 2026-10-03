# Results

Consolidated results for the project. Every figure here is copied from the recorded sections of
`project.md` (the source of truth) — nothing is recomputed. Each result is stated as **Claim →
Evidence → Caveat**; negatives are reported as findings, not footnotes.

## Overview

Two contributions, built as a **two-stage decoupled pipeline** (detector and severity head trained
on different datasets, run together at inference):

1. **Cross-domain detection** — one single-class pothole detector trained across 4 capture domains,
   robust where a single-domain detector collapses.
2. **Depth-grounded, calibration-aware severity** — a CNN grades severity from depth-derived labels,
   beats a box-area heuristic, and is shown only within its validated close-range regime.

Evaluation is honesty-first: a custom stratified matcher used identically for every comparison, GT
deduplicated before scoring, and limitations (including dropped ideas) reported in full.

---

## 1. Cross-domain detection

**Claim.** A detector pooled over 4 capture domains is robust across all of them, including the
close-up and tiny/distant regimes where the single-domain (India-only) baseline essentially fails.

**Evidence.** Per-domain **test AP@50**, India-only baseline vs cross-domain, on the same 802-image
test split, same evaluator, conf 0.20, IoU 0.5 (project.md §10.5):

| domain | baseline | cross-domain |
|---|---|---|
| far_dashcam | 0.551 | 0.445 |
| close_up | 0.023 | **0.919** |
| tiny_far | 0.045 | **0.539** |
| mid | 0.035 | **0.518** |
| OVERALL | 0.236 | **0.566** |

Overall recall rises **0.226 → 0.575** at roughly equal precision (~0.52). The close-up jump
(0.023 → 0.919; the baseline found 2 of 162) is the demo-failure case the cross-domain set was built
to fix.

**Caveat.** The baseline is *better* on **far_dashcam** (0.551 vs 0.445) — its sole training
domain; this is an honest trade-off for the large gains elsewhere. **mid** is only n=64 test images
(noisy — directional). The far_dashcam figures here are **not** comparable to the single-domain
detector's Component-3 AP@50 of 0.389 (different split, images, and single-class remap); only the
within-table baseline-vs-cross-domain comparison is apples-to-apples.

### Data handling (why the numbers are trustworthy)

The pool is **5,347 images** (3,743 pothole + 1,604 background = 30.0%), **8,234 boxes**, remapped
to a single `pothole` class (project.md §10.2):

| domain | source | pothole imgs |
|---|---|---|
| far_dashcam | RDD2022 India | 1,509 |
| close_up | PothRGBD | 996 |
| tiny_far | Pothole (Kaggle) | 937 |
| mid | Road Damage (Roboflow) | 301 |

Dedup is **domain-aware**: close-up uses file-content hash (md5) because pHash is non-discriminative
on close-ups (it collapsed ~590/1000 distinct potholes), while far/mid/tiny use greedy pHash≤3.
Cross-source, 163 Road-Damage images that duplicate RDD were dropped. Train/test **leakage** is
guarded by grouping near-duplicates across the split, not by dedup alone. GT boxes are deduplicated
before scoring — the Kaggle (tiny_far) labels carried exact-duplicate rows, cutting its GT from
**541 → 389** (other domains: 0), which prevents an artificially low tiny_far recall.

---

## 2. Depth-grounded severity head

**Claim.** A learned CNN grades pothole severity better than the obvious box-area heuristic — i.e.
severity is real appearance signal, not just a proxy for size.

**Evidence.** PothRGBD clean test set, n=134 (project.md §25.3):

| model | accuracy | macro-F1 |
|---|---|---|
| box-area heuristic | 0.41 | 0.42 |
| ResNet-18 severity head | **0.672** | **0.680** |

Box area vs depth-tier correlate only weakly (**Spearman ρ = 0.26**, ~7% of variance), so the head
is capturing something size alone cannot. The head is RGB-crop-only (depth was a *label* source),
resized to 224 so box size can't leak in; trained on 618 clean crops (SNR-gated 879/996).

**Caveat.** It **overfits by ~epoch 9** (train macro-F1 0.97 vs val ~0.55); the reported number
relies on early-stopping on val macro-F1. At n=134 the test figure carries roughly **±8 pp**. The
**medium** tier is the hardest (it bleeds into low/high); low is near-perfect.

**Ablation.** A frozen-backbone linear probe scores **0.33 macro-F1 (chance)** — fine-tuning the
backbone is necessary; generic ImageNet features are insufficient here (project.md §25.4).

---

## 3. Calibration-aware severity display

**Claim.** Severity is shown only where it is validated. The detector fires on far/tiny/mid potholes
that are out of the severity head's close-range training regime; those get a tier but it is marked
**uncalibrated** rather than presented as a confident grade.

**Evidence.** Gate = box-area-fraction ≥ **T = 0.123** (the PothRGBD-train p5), set from data
(project.md "Component 10 complete"). The threshold separates the domains cleanly:

| domain | median box-area-frac | % calibrated (≥ T) |
|---|---|---|
| close_up | 0.254 | ~85% |
| far_dashcam | 0.0087 | ~4% |
| tiny_far | 0.0003 | 0% |
| mid | 0.037 | ~17% |

Calibrated detections render in a tier color with confidence; uncalibrated ones render in a distinct
(cyan) box marked `* uncal` with the confidence dropped, and the summary counts the two groups
**separately** so validated and indicative tiers are never mixed.

**Caveat.** Conservative by design: it will occasionally flag a genuine close-up as uncalibrated
(the close-up left tail falls below T), but never the reverse — it never presents an out-of-range
guess as validated.

---

## 4. Negative results & limitations

Reported as findings — each was a hypothesis we measured and let the data settle.

**Adaptive tiling — dropped on evidence (project.md §22).** We hypothesized that tiling would
recover small/distant potholes lost to downscaling. Measured on the frozen India spine: upscaled
1280 inference did **not** lift small-D40 recall@0.25 (**0.403 → 0.388**; overall recall even fell
0.340 → 0.183), ~45% of small-D40 misses were **confidence** failures vs ~15% localization, and the
test set had **zero <16px** potholes (smallest ~21px). The bottleneck was confidence/class imbalance,
not resolution, so tiling was dropped (kept as this reported negative). Scope: specific to RDD2022
India at 720px/640 inference.

**Oversampling — regressed small-pothole recall (project.md §23.3).** Replicating D40-bearing train
images ×3 **lowered** small-bucket recall **0.427 → 0.373** (trading recall for precision
0.448 → 0.610). D40 is already an instance-level plurality (46.5% of train boxes), so the model was
not box-starved; duplicate same-distribution positives didn't help. Kept as an honest ablation row.

**Cross-sensor severity transfer — does not hold (project.md §25.6).** On the Pothole440 cross-check
(n=434, different depth sensor), the severity head scored **0.318 vs the 0.28 heuristic** — both near
chance. Learned severity is **sensor/domain-specific**; this is a stated limitation, and it is exactly
why the calibration gate (§3) restricts display to the validated close-range regime.

---

## 5. Reproducibility

- **Frozen splits/manifests** are committed: `configs/crossdomain/` (cross-domain split, data.yaml,
  per-image domain tags) and `configs/splits/` (the India detector spine + `severity_meta.json`).
- **Build/eval scripts** are committed under `src/detection/` and `src/severity/` (cross-domain
  build, split, packager; severity labeling, crops, training, heuristic).
- **Weights are gitignored** — `data/weights/crossdomain_best.pt` (Colab `checkpoints/crossdomain`)
  and `data/severity/cpu_ckpts/severity_best_finetune_all.pt` (severity training).
- **Evaluation** is reproducible via `src/detection/eval_stratified.py` (India spine, size-stratified)
  and `src/detection/eval_crossdomain.py` (per-domain, cross-domain vs baseline).

## 6. Key references

- **Base paper:** Wu et al., *Object Detection Model Design for Tiny Road Surface Damage*,
  Scientific Reports, 2025.
- **Datasets:** RDD2022 (Arya et al., CC BY) · PothRGBD (RGB-D, RealSense D415) · Pothole Detection
  (Kaggle) · Road Damage (Roboflow). Pothole440 (Go!SCAN 3D) used as a severity cross-check.

Full citations, dataset handling, and the dated decision log are in `project.md`.
