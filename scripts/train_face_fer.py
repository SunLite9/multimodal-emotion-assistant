# scripts/train_face_fer.py
# Fine-tune MobileNetV3-Small on FER-2013 folder splits (train/, test/).
# Example:
#   python scripts/train_face_fer.py --train_root "C:\...\FER-2013\train" --val_root "C:\...\FER-2013\test" --out weights\face_mobilenetv3.pt
import os, sys, argparse, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.append(ROOT)

from models_face import load_face_model  # reuses the same head definition (7 classes)

LABELS = ["angry","disgust","fear","happy","neutral","sad","surprise"]

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", required=True, help="FER-2013 train folder with 7 subfolders")
    ap.add_argument("--val_root", required=True, help="FER-2013 val/test folder with 7 subfolders")
    ap.add_argument("--out", required=True, help="Output weights path, e.g., weights\\face_mobilenetv3.pt")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    return ap.parse_args()

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tfm_tr = transforms.Compose([
        transforms.Resize((240, 240), antialias=True),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    tfm_va = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    ds_tr = datasets.ImageFolder(args.train_root, transform=tfm_tr)
    ds_va = datasets.ImageFolder(args.val_root, transform=tfm_va)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=(device=="cuda"))
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=(device=="cuda"))

    # Build a MobileNetV3-Small, replace head to 7 classes (same as app)
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_feats = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feats, len(LABELS))
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    best = 0.0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for epoch in range(1, args.epochs+1):
        model.train()
        tot, correct = 0, 0
        for x, y in dl_tr:
            x = x.to(device); y = y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            tot += y.size(0)
        acc_tr = correct / max(1, tot)

        model.eval()
        tot, correct = 0, 0
        with torch.inference_mode():
            for x, y in dl_va:
                x = x.to(device); y = y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                tot += y.size(0)
        acc_va = correct / max(1, tot)
        sched.step()

        print(f"Epoch {epoch:02d} | train_acc={acc_tr:.4f} | val_acc={acc_va:.4f}")
        if acc_va >= best:
            best = acc_va
            torch.save(model.state_dict(), args.out)
            print(f"  [saved] {args.out} (best_val_acc={best:.4f})")

    print(f"Done. Best val acc = {best:.4f}. Weights -> {args.out}")

if __name__ == "__main__":
    main()
