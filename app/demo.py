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
                          SEVERITY_WEIGHTS, run_pipeline)

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
- **Detector domain:** YOLO11s trained on **RDD2022 India** (dashcam road scenes) — works best
  on that domain and may detect little on close-up or other-domain images (e.g. PothRGBD).
- **Severity on RDD is QUALITATIVE:** no depth ground truth and cross-domain from the severity
  training set. **Validated severity (0.67 test accuracy) is on the PothRGBD test split.**
- **Operating point:** detector confidence threshold **conf = {DET_CONF}** (locked); imgsz 640.
- Pothole (D40) boxes are colored by tier: **green=low / amber=medium / red=high**. Other damage
  classes (D00/D10/D20) are detected but **severity is n/a (potholes only)** — shown as
  "severity: n/a" on the box and noted in the summary.
"""


def _summary_md(summary: dict) -> str:
    d = summary["detections_by_class"]; t = summary["potholes_by_tier"]
    lines = ["**Detections by class**", "", "| class | count |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in d.items()]
    lines += ["", "**Potholes (D40) by severity tier**", "", "| tier | count |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in t.items()]
    if summary.get("severity_note"):
        lines += ["", f"_{summary['severity_note']}_"]
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
        run_btn = gr.Button("Run pipeline", variant="primary")
        out_md = gr.Markdown(label="Summary")
        run_btn.click(predict, inputs=inp, outputs=[out_img, out_md])
        if EXAMPLES:
            gr.Examples(EXAMPLES, inputs=inp, label="Examples (RDD India test + PothRGBD)")
    return demo


if __name__ == "__main__":
    print(f"detector: {DETECTOR_WEIGHTS} | severity: {SEVERITY_WEIGHTS} | conf={DET_CONF} | share={SHARE}")
    build_demo().launch(share=SHARE)
