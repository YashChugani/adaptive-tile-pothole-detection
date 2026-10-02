"""End-to-end pothole inference pipeline (Component 7). Inference only, local CPU.

image -> YOLO11s detector (4-class) -> per-D40 severity CNN -> annotated image + summary.

Reuses the SEVERITY training crop/preprocess verbatim (no reimplementation):
  - margin + box expansion: src.severity.crops.MARGIN / expand_clip
  - 224 resize + ImageNet norm: src.severity.train_severity_local.eval_tf
  - model architecture: src.severity.train_severity_local.build_model

Two-stage, decoupled (project.md architecture): the detector trains on RDD2022, the severity
head on PothRGBD; here they are only run, never fused.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root for the reuse imports
from src.severity.crops import MARGIN, expand_clip               # reuse: 0.15 margin + box expand
from src.severity.train_severity_local import build_model, eval_tf  # reuse: arch + 224/ImageNet transform

# ---------------- top constants ----------------
DETECTOR_WEIGHTS = "data/weights/baseline_best.pt"   # <-- place your baseline best.pt here (gitignored)
SEVERITY_WEIGHTS = "data/severity/cpu_ckpts/severity_best_finetune_all.pt"
DET_CONF = 0.20          # locked operating point (Component 5.1)
DET_IMGSZ = 640
D40_IDX = 3
DET_CLASSES = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
TIER_COLORS = {"low": (0, 200, 0), "medium": (0, 180, 255), "high": (0, 0, 255)}  # BGR
OUT_DIR = Path("data/_pipeline_test")
FONT = cv2.FONT_HERSHEY_SIMPLEX
RDD2022_SEVERITY_CAVEAT = (
    "NOTE: severity tiers on RDD2022 are QUALITATIVE only (no depth GT; cross-domain from the "
    "PothRGBD severity training set). Validated severity metrics are on the PothRGBD test split.")
# -----------------------------------------------


def load_severity():
    ckpt = torch.load(SEVERITY_WEIGHTS, map_location="cpu")
    classes = ckpt["classes"]
    model = build_model(ckpt.get("mode", "finetune_all"), len(classes))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, classes


def predict_tier(model, classes, crop_bgr):
    """Severity tier + softmax confidence for one pothole crop, using the SAME preprocess
    (eval_tf) as training. crop_bgr is an OpenCV BGR array."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    x = eval_tf(Image.fromarray(rgb)).unsqueeze(0)
    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1)[0]
    idx = int(prob.argmax())
    return classes[idx], float(prob[idx])


def run_pipeline(image_path: str):
    """image -> (annotated_image, summary). Non-pothole classes are drawn but not tiered."""
    from ultralytics import YOLO

    if not Path(DETECTOR_WEIGHTS).exists():
        raise FileNotFoundError(
            f"detector weights not found: {DETECTOR_WEIGHTS}\n"
            f"Place your baseline best.pt there (e.g. copy baseline/weights/best.pt).")
    det = YOLO(DETECTOR_WEIGHTS)
    sev_model, sev_classes = load_severity()

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    h, w = img.shape[:2]
    res = det.predict(str(image_path), conf=DET_CONF, imgsz=DET_IMGSZ, device="cpu", verbose=False)[0]

    summary = {"image": str(image_path),
               "detections_by_class": {c: 0 for c in DET_CLASSES.values()},
               "potholes_by_tier": {"low": 0, "medium": 0, "high": 0},
               "severity_note": "Non-pothole classes (D00/D10/D20) are detected but severity is "
                                "n/a (potholes only); tiers apply to D40."}
    annotated = img.copy()
    for b in res.boxes:
        cls, conf = int(b.cls), float(b.conf)
        x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
        name = DET_CLASSES.get(cls, str(cls))
        summary["detections_by_class"][name] = summary["detections_by_class"].get(name, 0) + 1
        if cls == D40_IDX:
            cx0, cy0, cx1, cy1 = expand_clip((x1, y1, x2, y2), w, h)   # reuse 0.15 margin expansion
            crop = img[cy0:cy1, cx0:cx1]
            if crop.size == 0:
                continue
            tier, tconf = predict_tier(sev_model, sev_classes, crop)
            summary["potholes_by_tier"][tier] += 1
            color = TIER_COLORS[tier]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"D40 {conf:.2f} | {tier} {tconf:.2f}", (x1, max(y1 - 6, 14)),
                        FONT, 0.5, color, 2)
        else:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv2.putText(annotated, f"{name} {conf:.2f} | severity: n/a (potholes only)",
                        (x1, max(y1 - 6, 14)), FONT, 0.45, (255, 255, 255), 1)
    return annotated, summary


def main():
    ap = argparse.ArgumentParser(description="End-to-end pothole detection + severity pipeline.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default=None, help="output annotated image path (default under data/_pipeline_test/)")
    args = ap.parse_args()

    annotated, summary = run_pipeline(args.image)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else OUT_DIR / (Path(args.image).stem + "_annotated.jpg")
    cv2.imwrite(str(out), annotated)

    print(f"\nimage: {summary['image']}")
    print(f"detections by class : {summary['detections_by_class']}")
    print(f"potholes by tier    : {summary['potholes_by_tier']}")
    print(f"severity note       : {summary['severity_note']}")
    print(f"annotated -> {out}")
    print(RDD2022_SEVERITY_CAVEAT)


if __name__ == "__main__":
    main()
