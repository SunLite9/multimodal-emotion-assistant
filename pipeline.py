import numpy as np
from typing import Dict, List
from models_face import infer_face
from models_audio import infer_audio
from preprocess import mfcc_from_pcm_int16, late_fusion
 
def run_face_inference(face_model, frame_bgr, labels):
    """
    Returns dict: {"label": str, "probs": [float]*len(labels)}
    """
    probs = face_model.predict_probs(frame_bgr)  # <— uses HSEmotion backend
    # Align to the labels passed in (should already match)
    if probs.shape[0] != len(labels):
        # Simple guard if labels mismatch
        probs = np.resize(probs, len(labels))
        probs = probs / (probs.sum() + 1e-8)
    idx = int(np.argmax(probs))
    return {"label": labels[idx], "probs": probs.tolist()}


def run_audio_inference(model, pcm_int16: np.ndarray, sr: int, labels: List[str]) -> Dict:
    mfcc = mfcc_from_pcm_int16(pcm_int16, sr=sr, n_mfcc=40)
    pred, probs = infer_audio(model, mfcc)
    return {"label": labels[pred], "probs": [float(p) for p in probs]}

def fuse(face_out: Dict, audio_out: Dict, labels: List[str], mode="weighted", w_voice=0.6) -> Dict:
    if mode == "weighted":
        p = late_fusion(face_out["probs"], audio_out["probs"], w_face=1.0 - w_voice, w_voice=w_voice)
    else:
        p = audio_out["probs"]
    idx = int(np.argmax(p))
    return {"label": labels[idx], "probs": [float(x) for x in p]}


