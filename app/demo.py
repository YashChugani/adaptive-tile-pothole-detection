"""Gradio demo for the end-to-end pothole pipeline (Component 8). Local CPU.

Thin UI over src.pipeline.run_pipeline (detection + severity) — detection/severity are NOT
reimplemented here. Upload a road image (or click an example) -> annotated image + summary.

Run:
    .venv\\Scripts\\python.exe app/demo.py
then open the printed local URL (http://127.0.0.1:7860). share=False by default; set
SHARE=True below (or env GRADIO_SHARE=1) for a temporary public link.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from src.pipeline import (DET_CONF, DETECTOR_WEIGHTS, RDD2022_SEVERITY_CAVEAT,  # reuse constants
                          SEVERITY_CALIB_MIN_AREA_FRAC, SEVERITY_WEIGHTS, run_pipeline)

SHARE = os.environ.get("GRADIO_SHARE", "0") == "1"   # temporary public link if True

# example images (local, gitignored data/) — only those present are shown
_EXAMPLE_CANDIDATES = [
    "data/rdd2022_india/images/India_000209.jpg",   # RDD: many potholes
    "data/rdd2022_india/images/India_000105.jpg",   # RDD: a couple potholes
    "data/rdd2022_india/images/India_000162.jpg",   # RDD: cracks only (not tiered)
    *[str(p) for p in sorted(Path("data/severity/pothrgbd/PUBLIC POTHOLE DATASET/images").glob("*.jpg"))[:1]],
]
EXAMPLES = [[p] for p in _EXAMPLE_CANDIDATES if Path(p).exists()]

CAVEATS_MD = f"""
### Caveats (please read)
- **Detector domain:** YOLO11s trained across **4 capture domains** (far/dashcam, close-up,
  tiny/far, mid) and evaluated per-domain — it performs across all four. Measured trade-off: a
  far-domain-only baseline does slightly better on far/dashcam alone.
- **Severity is VALIDATED only on close-range (PothRGBD-like) potholes** (0.67 test accuracy).
  On any other domain (far/dashcam, tiny/far, mid) the tier is **qualitative/indicative only.**
- **Calibration gate:** severity is validated only for potholes large in frame (box-area-fraction
  ≥ {SEVERITY_CALIB_MIN_AREA_FRAC}, the PothRGBD p5). Smaller/distant detections show the tier marked
  **`* uncal`** in a grey box and should be read as indicative only.
- **Operating point:** detector confidence threshold **conf = {DET_CONF}** (locked); imgsz 640.
- Single-class detector: every detection is a **pothole**, boxed and colored by severity tier
  (**green=low / amber=medium / red=high**) with the tier shown on the box.
"""


def _summary_md(summary: dict) -> str:
    d = summary["detections_by_class"]
    cal = summary["potholes_by_tier_calibrated"]
    unc = summary["potholes_by_tier_uncalibrated"]
    lines = ["**Detections by class**", "", "| class | count |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in d.items()]
    lines += ["", "**Potholes by severity tier — calibrated (close-range)**", "", "| tier | count |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in cal.items()]
    lines += ["", "**Potholes by severity tier — uncalibrated (out of validated size range, indicative only)**",
              "", "| tier | count |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in unc.items()]
    return "\n".join(lines)


def predict(image_path):
    if not image_path:
        return None, "Upload an image or pick an example."
    try:
        annotated, summary = run_pipeline(image_path)
    except FileNotFoundError as e:
        return None, f"**Error:** {e}"
    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return rgb, _summary_md(summary)


def build_demo():
    with gr.Blocks(title="Pothole Detection + Severity") as demo:
        gr.Markdown("# Pothole Detection + Learned Severity\n"
                    "Upload a road image (or click an example). "
                    "Pipeline: YOLO11s detector → per-pothole severity CNN → annotated image + summary.")
        gr.Markdown(CAVEATS_MD)
        with gr.Row():
            inp = gr.Image(type="filepath", label="Road image", sources=["upload"])
            out_img = gr.Image(label="Annotated output")
        gr.Markdown("*Legend: colored box = calibrated severity (green=low / amber=medium / red=high); "
                    "grey box + `* uncal` = severity shown but uncalibrated (pothole too small/distant).*")
        run_btn = gr.Button("Run pipeline", variant="primary")
        out_md = gr.Markdown(label="Summary")
        run_btn.click(predict, inputs=inp, outputs=[out_img, out_md])
        if EXAMPLES:
            gr.Examples(EXAMPLES, inputs=inp, label="Examples (RDD India test + PothRGBD)")
    return demo


if __name__ == "__main__":
    print(f"detector: {DETECTOR_WEIGHTS} | severity: {SEVERITY_WEIGHTS} | conf={DET_CONF} | share={SHARE}")
    build_demo().launch(share=SHARE)
