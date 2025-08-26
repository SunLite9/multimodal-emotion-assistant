import numpy as np
from typing import Dict, List
from models_face import infer_face
from models_audio import infer_audio
from preprocess import mfcc_from_pcm_int16, late_fusion

def run_face_inference(model, frame_bgr, labels: List[str]) -> Dict:
    pred, probs = infer_face(model, frame_bgr)
    if probs is None:
        # no face; fall back neutral low confidence
        probs = np.zeros(len(labels), dtype=np.float32)
        idx = labels.index("neutral")
        probs[idx] = 1.0
        pred = idx
    return {"label": labels[pred], "probs": [float(p) for p in probs]}

def run_audio_inference(model, pcm_int16: np.ndarray, sr: int, labels: List[str]) -> Dict:
    mfcc = mfcc_from_pcm_int16(pcm_int16, sr=sr, n_mfcc=40)
    pred, probs = infer_audio(model, mfcc)
    return {"label": labels[pred], "probs": [float(p) for p in probs]}

def fuse(face_out: Dict, audio_out: Dict, mode="weighted", w_voice=0.6) -> Dict:
    if mode == "weighted":
        p = late_fusion(face_out["probs"], audio_out["probs"], w_face=1.0 - w_voice, w_voice=w_voice)
    else:
        p = audio_out["probs"]
    idx = int(np.argmax(p))
    labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    return {"label": labels[idx], "probs": [float(x) for x in p]}

