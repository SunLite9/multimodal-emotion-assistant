# RUN THIS

# python scripts\eval_audio.py --root "C:\Users\rmadi\multimodal-emotion-assistant\data\RAVDESS Emotional speech audio" --weights .\weights\audio_bilstm.pt --summary_only

import os, sys, csv, argparse, json
import numpy as np
import librosa

# Make local imports work when run from project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.append(ROOT)

from models_audio import load_audio_model, infer_audio
from preprocess import mfcc_from_pcm_int16, ravdess_emotion_from_filename

# Default 7-class labels (calm merged to neutral)
LABELS_7 = ["angry","disgust","fear","happy","neutral","sad","surprise"]
LABELS_8 = ["angry","disgust","fear","happy","neutral","calm","sad","surprise"]  # if you trained 8-class

def parse_args():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--root", help="RAVDESS actors root (contains Actor_01..Actor_24)")
    g.add_argument("--manifest", help="CSV with columns: path,label")
    ap.add_argument("--weights", required=True, help="Path to audio model weights .pt")
    ap.add_argument("--sr", type=int, default=16000, help="Audio sample rate for eval")
    ap.add_argument("--classes", type=int, default=7, choices=[7,8], help="Num classes in your audio head")
    ap.add_argument("--batch_windows", type=int, default=5,
                    help="If >0, chunk long clips into N windows and average probs (0 = whole-clip)")

    # NEW:
    ap.add_argument("--summary_only", action="store_true",
                    help="Only print a single accuracy line at the end.")
    ap.add_argument("--metrics_json", help="Optional path to write metrics as JSON.")
    return ap.parse_args()

def scan_ravdess(root, merge_calm=True):
    rows = []
    for actor in sorted(os.listdir(root)):
        if not actor.startswith("Actor_"): continue
        adir = os.path.join(root, actor)
        for fn in os.listdir(adir):
            if not fn.lower().endswith(".wav"): continue
            # Require audio-only=03, speech=01
            parts = os.path.splitext(fn)[0].split("-")
            if len(parts) < 3: continue
            if parts[0] != "03" or parts[1] != "01":  # modality=03, vchannel=01
                continue
            path = os.path.join(adir, fn)
            lab = ravdess_emotion_from_filename(fn, merge_calm_to_neutral=merge_calm)
            rows.append((path, lab))
    return rows

def load_manifest(manifest):
    rows = []
    with open(manifest, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((row["path"], row["label"]))
    return rows

def int16_from_float(x):
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)

def main():
    args = parse_args()
    labels = LABELS_7 if args.classes == 7 else LABELS_8
    lab2idx = {lab:i for i, lab in enumerate(labels)}

    # Collect eval files
    if args.root:
        rows = scan_ravdess(args.root, merge_calm=(args.classes==7))
    else:
        rows = load_manifest(args.manifest)

    # Filter to labels we expect
    data = [(p, lab) for (p, lab) in rows if lab in lab2idx]
    if not data:
        print("No files found. Check --root/--manifest and label mapping.")
        return

    if not args.summary_only:
        print(f"[info] Evaluating {len(data)} clips, classes={args.classes}: {labels}")

    # Load model
    model = load_audio_model(args.weights, num_classes=args.classes)

    # Metrics
    C = len(labels)
    conf = np.zeros((C, C), dtype=np.int64)
    correct_top1 = 0
    correct_top5 = 0

    for i, (path, lab) in enumerate(data, 1):
        y, sr = librosa.load(path, sr=args.sr, mono=True)
        if args.batch_windows and args.batch_windows > 0:
            # Chunk the audio into N equal windows, average probs
            probs_list = []
            L = len(y)
            step = max(1, L // args.batch_windows)
            for w in range(args.batch_windows):
                a = w * step
                b = L if w == args.batch_windows - 1 else (w+1)*step
                pcm16 = int16_from_float(y[a:b])
                mfcc = mfcc_from_pcm_int16(pcm16, sr=args.sr, n_mfcc=40)
                _, p = infer_audio(model, mfcc)
                probs_list.append(p)
            probs = np.mean(np.stack(probs_list, axis=0), axis=0)
            pred = int(np.argmax(probs))
        else:
            # Single pass on full clip
            pcm16 = int16_from_float(y)
            mfcc = mfcc_from_pcm_int16(pcm16, sr=args.sr, n_mfcc=40)
            pred, probs = infer_audio(model, mfcc)

        true = lab2idx[lab]
        conf[true, pred] += 1
        correct_top1 += int(pred == true)
        # top-5
        top5 = np.argsort(probs)[::-1][:5]
        correct_top5 += int(true in top5)

    total = len(data)
    acc1 = correct_top1 / total
    acc5 = correct_top5 / total

    # Optional JSON dump
    if args.metrics_json:
        metrics = {
            "top1": float(acc1),
            "top5": float(acc5),
            "n": int(total),
            "labels": labels,
            "confusion": conf.tolist(),
        }
        os.makedirs(os.path.dirname(args.metrics_json) or ".", exist_ok=True)
        with open(args.metrics_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        if not args.summary_only:
            print(f"[info] Wrote metrics JSON -> {args.metrics_json}")

    # Print results
    if args.summary_only:
        print(f"ACCURACY top1={acc1*100:.2f}% top5={acc5*100:.2f}% N={total}")
        return

    print(f"\nOverall:  top1={acc1:.4f}   top5={acc5:.4f}   (N={total})")

    # Per-class
    print("\nPer-class accuracy:")
    for i, lab in enumerate(labels):
        support = conf[i].sum()
        correct = conf[i, i]
        acc = correct / support if support > 0 else 0.0
        print(f"  {lab:>8s}: acc={acc:.4f}   support={support}")

    # Confusion matrix
    print("\nConfusion matrix (rows=true, cols=pred):")
    with np.printoptions(linewidth=140, suppress=True):
        print(conf)

    # Extra compact line at the very end (easy to grep/parse)
    print(f"\nACCURACY top1={acc1*100:.2f}% top5={acc5*100:.2f}% N={total}")

if __name__ == "__main__":
    main()
