# scripts/train_audio_ravdess.py
# Train MFCC -> BiLSTM on RAVDESS speech. Default = 7 classes (calm merged into neutral).
# Example:
#   python scripts/train_audio_ravdess.py --root "C:\...\RAVDESS Emotional speech audio" --out weights\audio_bilstm.pt --epochs 25
import os, sys, argparse, math, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.append(ROOT)

from preprocess import mfcc_from_pcm_int16, ravdess_emotion_from_filename
from models_audio import load_audio_model

LABELS_7 = ["angry","disgust","fear","happy","neutral","sad","surprise"]
LABELS_8 = ["angry","disgust","fear","happy","neutral","calm","sad","surprise"]

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Folder that directly contains Actor_01..Actor_24")
    ap.add_argument("--out", required=True, help="Output weights path, e.g. weights\\audio_bilstm.pt")
    ap.add_argument("--classes", type=int, choices=[7,8], default=7, help="7 (merge calm->neutral) or 8 classes")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--duration", type=float, default=1.6, help="Seconds per training window (fixed length)")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1337)
    return ap.parse_args()

def list_ravdess(root, merge_calm=True):
    rows = []
    for d in sorted(os.listdir(root)):
        if not d.startswith("Actor_"): continue
        adir = os.path.join(root, d)
        for fn in os.listdir(adir):
            if not fn.lower().endswith(".wav"): continue
            parts = os.path.splitext(fn)[0].split("-")
            if len(parts) < 3: continue
            if parts[0] != "03" or parts[1] != "01":  # audio-only, speech
                continue
            path = os.path.join(adir, fn)
            lab = ravdess_emotion_from_filename(fn, merge_calm_to_neutral=merge_calm)
            actor_id = int(parts[6]) if len(parts) >= 7 else int(d.split("_")[-1])
            rows.append((path, lab, actor_id))
    return rows

class RAVDESSAudioDS(Dataset):
    def __init__(self, items, labels, sr=16000, seconds=1.6, train=True):
        self.items = items
        self.labels = labels
        self.lab2idx = {l:i for i,l in enumerate(labels)}
        self.sr = sr
        self.N = int(seconds * sr)
        self.train = train

    def __len__(self):
        return len(self.items)

    def _load_fixed(self, path):
        y, sr = librosa.load(path, sr=self.sr, mono=True)
        if len(y) < self.N:
            pad = self.N - len(y)
            y = np.pad(y, (0, pad), mode="constant")
        elif len(y) > self.N:
            # random crop for train; center crop for val
            if self.train:
                start = np.random.randint(0, len(y) - self.N + 1)
            else:
                start = (len(y) - self.N) // 2
            y = y[start:start+self.N]
        return (y * 32767.0).astype(np.int16)

    def _spec_augment(self, mfcc):
        # Simple SpecAugment (train only): 1 freq mask, 1 time mask
        if not self.train: return mfcc
        m = mfcc.copy()
        T, F = m.shape
        # freq mask
        f = np.random.randint(0, max(1, F//8))
        f0 = np.random.randint(0, F - f + 1)
        m[:, f0:f0+f] = 0
        # time mask
        t = np.random.randint(0, max(1, T//8))
        t0 = np.random.randint(0, T - t + 1)
        m[t0:t0+t, :] = 0
        return m

    def __getitem__(self, i):
        path, lab = self.items[i]
        pcm16 = self._load_fixed(path)
        mfcc = mfcc_from_pcm_int16(pcm16, sr=self.sr, n_mfcc=40)  # [T,40]
        mfcc = self._spec_augment(mfcc)
        x = torch.from_numpy(mfcc).float()       # [T,40]
        y = torch.tensor(self.lab2idx[lab], dtype=torch.long)
        return x, y

def collate_pad(batch):
    # Pad variable-T MFCCs to max T in the batch
    xs, ys = zip(*batch)
    lens = [x.size(0) for x in xs]
    T = max(lens)
    F = xs[0].size(1)
    out = torch.zeros(len(xs), T, F, dtype=torch.float32)
    for i, x in enumerate(xs):
        out[i, :x.size(0), :] = x
    y = torch.stack(ys)
    return out, y

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def main():
    args = get_args()
    set_seed(args.seed)

    labels = LABELS_7 if args.classes == 7 else LABELS_8
    merge_calm = (args.classes == 7)

    all_rows = list_ravdess(args.root, merge_calm=merge_calm)
    # Speaker split: use Actors 21-24 as validation (no speaker leakage)
    train_items, val_items = [], []
    for path, lab, actor in all_rows:
        if actor >= 21: val_items.append((path, lab))
        else:           train_items.append((path, lab))

    print(f"[info] Train clips: {len(train_items)} | Val clips: {len(val_items)} | classes={args.classes}: {labels}")

    ds_tr = RAVDESSAudioDS(train_items, labels, sr=args.sr, seconds=args.duration, train=True)
    ds_va = RAVDESSAudioDS(val_items, labels, sr=args.sr, seconds=args.duration, train=False)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=0, collate_fn=collate_pad)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=collate_pad)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_audio_model("", num_classes=args.classes)  # start from random head
    model.train()
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for epoch in range(1, args.epochs+1):
        # ---- train
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

        # ---- val
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

        if acc_va >= best_acc:
            best_acc = acc_va
            torch.save(model.state_dict(), args.out)
            print(f"  [saved] {args.out}  (best_val_acc={best_acc:.4f})")

    print(f"Done. Best val acc = {best_acc:.4f}. Weights -> {args.out}")

if __name__ == "__main__":
    main()
