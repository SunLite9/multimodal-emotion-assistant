import os, sys, time, base64, io, json
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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from models_face import load_face_model
from models_audio import load_audio_model
from preprocess import rms_level, softmax
from pipeline import run_face_inference, run_audio_inference, fuse
from llm_client import get_empathetic_reply

# --- App setup
load_dotenv()
app = FastAPI()
app.mount("/ui", StaticFiles(directory="ui"), name="ui")

# Rolling buffers (server-side)
AUDIO_SR = 16000
AUDIO_SECONDS = 1.0
AUDIO_MAX_SAMPLES = int(AUDIO_SR * AUDIO_SECONDS)
audio_buffer = deque(maxlen=AUDIO_MAX_SAMPLES)  # int16 samples
latest_frame_bgr: Optional[np.ndarray] = None

# Models (loaded at startup)
face_model = None
audio_model = None
labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

@app.on_event("startup")
def _load_models():
    global face_model, audio_model
    face_model = load_face_model(os.getenv("FACE_WEIGHTS", ""))
    audio_model = load_audio_model(os.getenv("AUDIO_WEIGHTS", ""))
    print("[startup] Models ready.")

@app.get("/")
async def root():
    # Serve UI
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
    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)

            # --- Receive audio chunk (base64 int16) and frame (base64 JPEG)
            if "audio_b16" in data and data["audio_b16"]:
                pcm = np.frombuffer(base64.b64decode(data["audio_b16"]), dtype=np.int16)
                for s in pcm:
                    audio_buffer.append(int(s))

            if "jpeg_b64" in data and data["jpeg_b64"]:
                jpg = base64.b64decode(data["jpeg_b64"])
                nparr = np.frombuffer(jpg, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    # Keep last frame
                    global latest_frame_bgr
                    latest_frame_bgr = frame

            # --- Compute and send updates ~2 Hz to keep UI smooth but light
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

                # Inference
                face_out = {"label": "neutral", "probs": [0]*len(labels)}
                audio_out = {"label": "neutral", "probs": [0]*len(labels)}

                if latest_frame_bgr is not None:
                    face_out = run_face_inference(face_model, latest_frame_bgr, labels)

                if len(audio_buffer) >= int(0.5 * AUDIO_SR):  # need at least ~0.5s for MFCC
                    audio_out = run_audio_inference(audio_model, audio_chunk, sr=AUDIO_SR, labels=labels)

                fused = fuse(face_out, audio_out, mode="weighted", w_voice=0.6)

                # Prepare top3 details
                probs = np.array(fused["probs"], dtype=float)
                top_idx = probs.argsort()[::-1][:3]
                topk = [{"label": labels[i], "prob": float(probs[i])} for i in top_idx]

                await ws.send_text(json.dumps({
                    "audio_level": level,
                    "face": face_out,
                    "voice": audio_out,
                    "fused": fused,
                    "topk": topk
                }))
    except WebSocketDisconnect:
        pass
