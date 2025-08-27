# scripts/eval_face.py
# Evaluate the face CNN on your FER2013 test folder structure.
# Expected folder layout (ImageFolder):
#   <test_root>/
#     angry/ ...png|jpg
#     disgust/ ...
#     fear/ ...
#     happy/ ...
#     neutral/ ...
#     sad/ ...
#     surprise/ ...
#
# Usage:
#   python scripts/eval_face.py --test_root "D:\FER2013\test" --weights weights\face_mobilenetv3.pt
import os, sys, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Make local imports work when run from project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.append(ROOT)

from models_face import load_face_model

LABELS = ["angry","disgust","fear","happy","neutral","sad","surprise"]  # expected head order

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_root", required=True, help="FER2013 test root folder (with 7 class subfolders)")
    ap.add_argument("--weights", required=True, help="Path to face model weights .pt")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)  # 0 is safest on Windows
    return ap.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Transforms: resize to 224 and normalize like ImageNet (matches your model)
    tfm = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    ds = datasets.ImageFolder(args.test_root, transform=tfm)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=(device=="cuda"))

    # Sanity: class order must match LABELS
    classes = [c.lower() for c in ds.classes]
    if classes != LABELS:
        print(f"[warn] Dataset classes {classes} differ from expected {LABELS}. If you trained with a different order, accuracy may be off.")

    model = load_face_model(args.weights)
    model.eval()

    C = len(LABELS)
    conf = np.zeros((C, C), dtype=np.int64)
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            top1 = probs.argmax(dim=1)
            top5 = torch.topk(probs, k=min(5,C), dim=1).indices

            correct_top1 += (top1 == y).sum().item()
            correct_top5 += sum([int(y[i].item() in top5[i].tolist()) for i in range(y.size(0))])

            # Confusion
            preds = top1.cpu().numpy()
            trues = y.cpu().numpy()
            for t, p in zip(trues, preds):
                conf[t, p] += 1

            total += y.size(0)

    acc1 = correct_top1 / total
    acc5 = correct_top5 / total
    print(f"\nOverall:  top1={acc1:.4f}   top5={acc5:.4f}   (N={total})")

    print("\nPer-class accuracy:")
    for i, lab in enumerate(LABELS):
        support = conf[i].sum()
        correct = conf[i, i]
        acc = correct / support if support > 0 else 0.0
        print(f"  {lab:>8s}: acc={acc:.4f}   support={support}")

    print("\nConfusion matrix (rows=true, cols=pred):")
    with np.printoptions(linewidth=140, suppress=True):
        print(conf)

if __name__ == "__main__":
    main()
