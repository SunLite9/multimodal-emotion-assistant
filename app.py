import os, sys, time, base64, json
from collections import deque
from typing import Optional, Dict, Any

# --- Strict Python version guard (enforce 3.11.9 by default)
STRICT = os.getenv("STRICT_PY3119", "1") == "1"
if STRICT:
    if not (sys.version_info.major == 3 and sys.version_info.minor == 11 and sys.version_info.micro == 9):
        raise RuntimeError(
            f"This project is pinned to Python 3.11.9. Detected {sys.version.split()[0]}.\n"
            "Set STRICT_PY3119=0 to relax."
        )
else:
    if sys.version_info[:2] != (3, 11):
        print(f"[warn] Recommended Python 3.11.x; detected {sys.version.split()[0]}")

import numpy as np
import cv2
import librosa
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from models_face import load_face_model
from models_audio import load_audio_model
from preprocess import rms_level, softmax, mfcc_from_pcm_int16

from pipeline import run_face_inference, run_audio_inference, fuse
from llm_client import get_empathetic_reply

# ---------- App setup
load_dotenv()
app = FastAPI()
app.mount("/ui", StaticFiles(directory="ui"), name="ui")

# ---------- Config: smoothing & fusion (matches your eval-5 windows)
AUDIO_SMOOTH_K = int(os.getenv("AUDIO_SMOOTH_K", "5"))   # rolling audio window count
FACE_SMOOTH_K  = int(os.getenv("FACE_SMOOTH_K",  "10"))  # frames to smooth
AUDIO_VAD_THRESH = float(os.getenv("AUDIO_VAD_THRESH", "0.01"))
FUSE_W_VOICE = float(os.getenv("FUSE_W_VOICE", "0.6"))

# ---------- Config: prelaunch eval (prints to terminal on startup)
PRELAUNCH_EVAL = os.getenv("PRELAUNCH_EVAL", "0") == "1"
EVAL_BATCH_WINDOWS = int(os.getenv("EVAL_BATCH_WINDOWS", "5"))   # default 5
EVAL_MAX_PER_CLASS = int(os.getenv("EVAL_MAX_PER_CLASS", "20"))  # per-class cap for speed
FUSED_SAMPLES_PER_CLASS = int(os.getenv("FUSED_SAMPLES_PER_CLASS", "15"))

# ---------- Data paths (optional, for prelaunch eval)
RAVDESS_ROOT = os.getenv("RAVDESS_ROOT", "").strip()
FER_TEST_ROOT = os.getenv("FER_TEST_ROOT", "").strip()

# ---------- Rolling buffers (server-side)
AUDIO_SR = 16000
AUDIO_SECONDS = 1.0
AUDIO_MAX_SAMPLES = int(AUDIO_SR * AUDIO_SECONDS)
audio_buffer = deque(maxlen=AUDIO_MAX_SAMPLES)  # int16 samples
latest_frame_bgr: Optional[np.ndarray] = None

# Rolling probability histories for smoothing (live app)
audio_probs_hist: deque = deque(maxlen=AUDIO_SMOOTH_K)
face_probs_hist: deque  = deque(maxlen=FACE_SMOOTH_K)

# Models (loaded at startup)
face_model = None
audio_model = None
labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_AUDIO_CLASSES = len(labels)  # 7 (calm merged to neutral)

# ======================================================================================
#                               PRELAUNCH QUICK EVAL
# ======================================================================================

def _ravdess_emotion_from_filename(filename: str, merge_calm_to_neutral: bool = True) -> str:
    """
    Parse RAVDESS 7-part filename, e.g. 03-01-06-01-02-01-12.wav
    parts[2] is the emotion ID: 01 neutral, 02 calm, 03 happy, 04 sad,
    05 angry, 06 fearful, 07 disgust, 08 surprised.
    """
    import os
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("-")
    if len(parts) < 3:
        return "neutral"
    emo_id = parts[2]
    mapping = {
        "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
        "05": "angry",   "06": "fear", "07": "disgust", "08": "surprise",
    }
    lab = mapping.get(emo_id, "neutral")
    if merge_calm_to_neutral and lab == "calm":
        lab = "neutral"
    return lab


def _sample_audio_paths_by_label(root: str, max_per_class: int):
    """Scan RAVDESS (audio-only speech) -> dict[label] = list(paths)."""
    if not root or not os.path.isdir(root):
        return {}
    by = {lab: [] for lab in labels}
    for actor in sorted(os.listdir(root)):
        if not actor.startswith("Actor_"):
            continue
        adir = os.path.join(root, actor)
        if not os.path.isdir(adir):
            continue
        for fn in os.listdir(adir):
            if not fn.lower().endswith(".wav"):
                continue
            parts = os.path.splitext(fn)[0].split("-")
            if len(parts) < 3:
                continue
            if parts[0] != "03" or parts[1] != "01":  # audio-only=03, speech=01
                continue
            lab = _ravdess_emotion_from_filename(fn, merge_calm_to_neutral=True)
            if lab in by:
                by[lab].append(os.path.join(adir, fn))
    # cap per class for speed (shuffle for randomness)
    rng = np.random.RandomState(1337)
    for lab in by:
        rng.shuffle(by[lab])
        by[lab] = by[lab][:max_per_class]
    return by

def _sample_face_paths_by_label(test_root: str, max_per_class: int):
    """Scan FER2013 test -> dict[label] = list(image_paths)."""
    if not test_root or not os.path.isdir(test_root):
        return {}
    by = {lab: [] for lab in labels}
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    for lab in labels:
        d = os.path.join(test_root, lab)
        if not os.path.isdir(d):
            continue
        files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(exts)]
        rng = np.random.RandomState(1337)
        rng.shuffle(files)
        by[lab] = files[:max_per_class]
    return by

def _infer_audio_avg_windows(wav_path: str, windows: int):
    """Return averaged probability vector over N windows."""
    y, _ = librosa.load(wav_path, sr=AUDIO_SR, mono=True)
    if windows and windows > 0:
        L = len(y)
        step = max(1, L // windows)
        probs_list = []
        for w in range(windows):
            a = w * step
            b = L if w == windows - 1 else (w + 1) * step
            pcm16 = (np.clip(y[a:b], -1.0, 1.0) * 32767.0).astype(np.int16)
            out = run_audio_inference(audio_model, pcm16, sr=AUDIO_SR, labels=labels)
            probs_list.append(np.array(out["probs"], dtype=np.float32))
        return np.mean(np.stack(probs_list, axis=0), axis=0)
    # single pass
    pcm16 = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16)
    out = run_audio_inference(audio_model, pcm16, sr=AUDIO_SR, labels=labels)
    return np.array(out["probs"], dtype=np.float32)

def _infer_face_img(img_path: str):
    """Return probability vector for one FER image."""
    img = cv2.imread(img_path)  # BGR
    if img is None:
        return None
    out = run_face_inference(face_model, img, labels)
    return np.array(out["probs"], dtype=np.float32)

def _acc_from_probs(true_idx: int, probs: np.ndarray):
    top1 = int(np.argmax(probs) == true_idx)
    # top-5
    k = min(5, len(probs))
    top5 = int(true_idx in np.argsort(probs)[-k:])
    return top1, top5

def _prelaunch_eval():
    """Quick accuracy printouts before serving requests."""
    # Audio quick eval
    try:
        if RAVDESS_ROOT and os.path.isdir(RAVDESS_ROOT):
            by_aud = _sample_audio_paths_by_label(RAVDESS_ROOT, EVAL_MAX_PER_CLASS)
            total = sum(len(v) for v in by_aud.values())
            if total > 0:
                correct1 = correct5 = 0
                for lab, paths in by_aud.items():
                    t = labels.index(lab)
                    for p in paths:
                        pr = _infer_audio_avg_windows(p, EVAL_BATCH_WINDOWS)
                        a1, a5 = _acc_from_probs(t, pr)
                        correct1 += a1; correct5 += a5
                print(f"[prelaunch] audio ACCURACY top1={100*correct1/total:.2f}% top5={100*correct5/total:.2f}% N={total} windows={EVAL_BATCH_WINDOWS}")
            else:
                print("[prelaunch] audio: no files found under RAVDESS_ROOT; skipping.")
        else:
            print("[prelaunch] audio: RAVDESS_ROOT not set/exists; skipping.")
    except Exception as e:
        print(f"[prelaunch] audio eval error: {e}")

    # Face quick eval
    try:
        if FER_TEST_ROOT and os.path.isdir(FER_TEST_ROOT):
            by_face = _sample_face_paths_by_label(FER_TEST_ROOT, EVAL_MAX_PER_CLASS)
            total = sum(len(v) for v in by_face.values())
            if total > 0:
                correct1 = correct5 = 0
                for lab, paths in by_face.items():
                    t = labels.index(lab)
                    for p in paths:
                        pr = _infer_face_img(p)
                        if pr is None: 
                            continue
                        a1, a5 = _acc_from_probs(t, pr)
                        correct1 += a1; correct5 += a5
                print(f"[prelaunch] face  ACCURACY top1={100*correct1/total:.2f}% top5={100*correct5/total:.2f}% N={total}")
            else:
                print("[prelaunch] face: no files found under FER_TEST_ROOT; skipping.")
        else:
            print("[prelaunch] face: FER_TEST_ROOT not set/exists; skipping.")
    except Exception as e:
        print(f"[prelaunch] face eval error: {e}")

    # Fused quick eval (synthetic pairs per label)
    try:
        if RAVDESS_ROOT and FER_TEST_ROOT and os.path.isdir(RAVDESS_ROOT) and os.path.isdir(FER_TEST_ROOT):
            by_aud = _sample_audio_paths_by_label(RAVDESS_ROOT, FUSED_SAMPLES_PER_CLASS)
            by_face = _sample_face_paths_by_label(FER_TEST_ROOT, FUSED_SAMPLES_PER_CLASS)
            # Make same-count pairs per label
            total = 0; correct = 0
            for lab in labels:
                aud_list = by_aud.get(lab, [])
                face_list = by_face.get(lab, [])
                n = min(len(aud_list), len(face_list))
                n = min(n, FUSED_SAMPLES_PER_CLASS)
                for i in range(n):
                    pa = _infer_audio_avg_windows(aud_list[i], EVAL_BATCH_WINDOWS)
                    pf = _infer_face_img(face_list[i])
                    if pf is None:
                        continue
                    # fuse like the app (after smoothing; here single-step)
                    p_fused = FUSE_W_VOICE * pa + (1.0 - FUSE_W_VOICE) * pf
                    if labels[int(np.argmax(p_fused))] == lab:
                        correct += 1
                    total += 1
            if total > 0:
                print(f"[prelaunch] fused ACCURACY (synthetic pairs) top1={100*correct/total:.2f}% N={total}  [note: not a true multimodal dataset]")
            else:
                print("[prelaunch] fused: insufficient per-label pairs; skipping.")
        else:
            print("[prelaunch] fused: need both RAVDESS_ROOT and FER_TEST_ROOT; skipping.")
    except Exception as e:
        print(f"[prelaunch] fused eval error: {e}")

# ======================================================================================

@app.on_event("startup")
def _load_models():
    global face_model, audio_model
    face_model = load_face_model(os.getenv("FACE_WEIGHTS", ""))
    audio_model = load_audio_model(os.getenv("AUDIO_WEIGHTS", ""), num_classes=NUM_AUDIO_CLASSES)
    print("[startup] Models ready.")

    if PRELAUNCH_EVAL:
        print("[prelaunch] running quick accuracy checks...")
        _prelaunch_eval()
        print("[prelaunch] done.")

@app.get("/")
async def root():
    with open("ui/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.post("/reply")
async def reply(payload: Dict[str, Any]):
    emotion = payload.get("emotion", "neutral")
    recent_transcript = payload.get("context", "")
    topk = payload.get("topk", [])
    text = await get_empathetic_reply(emotion, topk, recent_transcript, model="gpt-4o-mini")
    return JSONResponse({"reply": text})

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    last_emit = 0.0
    global latest_frame_bgr
    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)

            # --- Receive audio chunk (base64 int16) and frame (base64 JPEG)
            if "audio_b16" in data and data["audio_b16"]:
                pcm = np.frombuffer(base64.b64decode(data["audio_b16"]), dtype=np.int16)
                audio_buffer.extend(pcm.tolist())

            if "jpeg_b64" in data and data["jpeg_b64"]:
                jpg = base64.b64decode(data["jpeg_b64"])
                nparr = np.frombuffer(jpg, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    latest_frame_bgr = frame

            # --- Emit ~2 Hz
            now = time.time()
            if now - last_emit >= 0.5:
                last_emit = now

                # Audio level (RMS)
                if len(audio_buffer) > 0:
                    audio_chunk = np.array(audio_buffer, dtype=np.int16)
                    level = float(rms_level(audio_chunk))
                else:
                    audio_chunk = np.zeros(AUDIO_MAX_SAMPLES, dtype=np.int16)
                    level = 0.0

                # Raw inference
                face_out = {"label": "neutral", "probs": [0.0]*len(labels)}
                audio_out = {"label": "neutral", "probs": [0.0]*len(labels)}

                if latest_frame_bgr is not None:
                    face_out = run_face_inference(face_model, latest_frame_bgr, labels)

                if len(audio_buffer) >= int(0.5 * AUDIO_SR):  # need at least ~0.5s
                    audio_out = run_audio_inference(audio_model, audio_chunk, sr=AUDIO_SR, labels=labels)

                # ---- Smoothing (rolling)
                if face_out.get("probs"):
                    face_probs_hist.append(np.asarray(face_out["probs"], dtype=np.float32))
                p_face = np.mean(np.stack(face_probs_hist, axis=0), axis=0) if len(face_probs_hist) else np.asarray(face_out["probs"], dtype=np.float32)

                if audio_out.get("probs"):
                    if level > AUDIO_VAD_THRESH:  # skip near-silence to reduce neutral bias
                        audio_probs_hist.append(np.asarray(audio_out["probs"], dtype=np.float32))
                p_audio = np.mean(np.stack(audio_probs_hist, axis=0), axis=0) if len(audio_probs_hist) else np.asarray(audio_out["probs"], dtype=np.float32)

                face_smoothed = {"label": labels[int(np.argmax(p_face))] if p_face.size else face_out["label"], "probs": p_face.tolist() if p_face.size else face_out["probs"]}
                audio_smoothed = {"label": labels[int(np.argmax(p_audio))] if p_audio.size else audio_out["label"], "probs": p_audio.tolist() if p_audio.size else audio_out["probs"]}

                # ---- Fuse AFTER smoothing
                fused = fuse(face_smoothed, audio_smoothed, labels=labels, mode="weighted", w_voice=FUSE_W_VOICE)

                # Top-5 for UI
                probs = np.array(fused["probs"], dtype=float)
                top_idx = probs.argsort()[::-1][:5]
                topk = [{"label": labels[i], "prob": float(probs[i])} for i in top_idx]

                await ws.send_text(json.dumps({
                    "audio_level": level,
                    "face": face_smoothed,
                    "voice": audio_smoothed,
                    "fused": fused,
                    "topk": topk
                }))
    except WebSocketDisconnect:
        pass
