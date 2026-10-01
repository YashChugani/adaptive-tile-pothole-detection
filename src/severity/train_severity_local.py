"""Local CPU training for the severity CNN (same logic as train_severity.ipynb cell 3/4).

ResNet-18 ImageNet-pretrained, Dropout(0.4) + weight_decay, resize-224 (size not a feature),
light depth-preserving aug, RGB-crop-only. Runs BOTH modes (finetune_all, frozen_backbone),
early-stops each on VAL macro-F1, selects the winner, and prints the full cell-4 eval head-to-head
vs the box-area heuristic. Reads crops from data/severity/crops/; checkpoints to data/severity/
(gitignored). CPU only — slow but fine.

    .venv\\Scripts\\python.exe src/severity/train_severity_local.py
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ---------------- PARAMS (mirror the notebook) ----------------
CROPS = Path("data/severity/crops")
PROJECT = Path("data/severity/cpu_ckpts"); PROJECT.mkdir(parents=True, exist_ok=True)
TRAIN_MODES = ("finetune_all", "frozen_backbone")
EPOCHS = 40
PATIENCE = 8
LR = 1e-4
WD = 1e-4
DROPOUT = 0.4
BATCH = 32
IMG = 224
DEVICE = "cpu"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
HEUR_TEST_ACC = 0.41        # box-area heuristic baselines (Component 6.3A)
HEUR_P440_ACC = 0.28
# --------------------------------------------------------------

train_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def build_model(mode, n_classes):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if mode == "frozen_backbone":
        for p in m.parameters():
            p.requires_grad = False
    m.fc = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(m.fc.in_features, n_classes))
    return m.to(DEVICE)


def eval_f1_acc(model, dl):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for x, y in dl:
            ps += model(x.to(DEVICE)).argmax(1).cpu().tolist(); ys += y.tolist()
    return f1_score(ys, ps, average="macro"), accuracy_score(ys, ps)


def train_one(mode, train_dl, val_dl, n_train, classes):
    print(f"\n========== TRAIN_MODE = {mode} ==========", flush=True)
    model = build_model(mode, len(classes))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    crit = nn.CrossEntropyLoss()
    ckpt_path = PROJECT / f"severity_best_{mode}.pt"
    best_f1, best_epoch, bad = -1, -1, 0
    for ep in range(1, EPOCHS + 1):
        model.train(); tot = 0; ys, ps = [], []
        for x, y in train_dl:
            opt.zero_grad()
            out = model(x.to(DEVICE)); loss = crit(out, y.to(DEVICE))
            loss.backward(); opt.step(); tot += loss.item() * x.size(0)
            ps += out.argmax(1).detach().cpu().tolist(); ys += y.tolist()
        tr_f1, tr_acc = f1_score(ys, ps, average="macro"), accuracy_score(ys, ps)
        va_f1, va_acc = eval_f1_acc(model, val_dl)
        print(f"  ep {ep:02d}  loss={tot/n_train:.3f} | TRAIN acc={tr_acc:.3f} mF1={tr_f1:.3f} | "
              f"VAL acc={va_acc:.3f} mF1={va_f1:.3f} | gap(mF1)={tr_f1-va_f1:+.3f}", flush=True)
        if va_f1 > best_f1:
            best_f1, best_epoch, bad = va_f1, ep, 0
            torch.save({"model": model.state_dict(), "classes": classes, "mode": mode}, ckpt_path)
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"  early stop at {ep} (best val mF1={best_f1:.3f} @ ep {best_epoch})", flush=True); break
    print(f"  {mode}: best VAL macro-F1 = {best_f1:.3f} -> {ckpt_path}", flush=True)
    return best_f1


def evaluate(model, classes, folder, name, heuristic_acc):
    ds = datasets.ImageFolder(folder, eval_tf)
    dl = DataLoader(ds, BATCH, shuffle=False, num_workers=0)
    ys, ps = [], []
    with torch.no_grad():
        for x, y in dl:
            ps += model(x.to(DEVICE)).argmax(1).cpu().tolist(); ys += y.tolist()
    acc = accuracy_score(ys, ps); f1m = f1_score(ys, ps, average="macro")
    f1c = f1_score(ys, ps, average=None); cm = confusion_matrix(ys, ps)
    print(f"\n=== {name} (n={len(ds)}) classes={classes} ===")
    print(f"  CNN      : acc={acc:.3f}  macro-F1={f1m:.3f}  per-class F1={[round(float(v),3) for v in f1c]}")
    print(f"  heuristic: acc={heuristic_acc:.3f}  (box-area baseline)")
    print(f"  delta    : acc {acc-heuristic_acc:+.3f}")
    print("  confusion (rows=true, cols=pred):")
    for c, row in zip(classes, cm):
        print(f"    {c:<7}{list(map(int,row))}")


def main():
    train_ds = datasets.ImageFolder(CROPS / "pothrgbd" / "train", train_tf)
    val_ds = datasets.ImageFolder(CROPS / "pothrgbd" / "val", eval_tf)
    classes = train_ds.classes
    print(f"classes: {classes} | train {len(train_ds)} val {len(val_ds)} | device={DEVICE}", flush=True)
    train_dl = DataLoader(train_ds, BATCH, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, BATCH, shuffle=False, num_workers=0)

    results = {mode: train_one(mode, train_dl, val_dl, len(train_ds), classes) for mode in TRAIN_MODES}
    winner = max(results, key=results.get)
    print(f"\nVAL macro-F1 by mode: {results}")
    print(f"WINNER (by val macro-F1): {winner}  (val mF1={results[winner]:.3f})", flush=True)

    model = build_model(winner, len(classes))
    ckpt = torch.load(PROJECT / f"severity_best_{winner}.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model"]); model.eval()
    evaluate(model, classes, CROPS / "pothrgbd" / "test", "PothRGBD TEST (clean)", HEUR_TEST_ACC)
    evaluate(model, classes, CROPS / "pothole440", "Pothole440 (cross-dataset)", HEUR_P440_ACC)
    print("\nNote: CNN input is RGB crop ONLY; crops resized 224x224 so box-area is not a shortcut.", flush=True)


if __name__ == "__main__":
    main()
