# models_face.py 

import os
import numpy as np
import cv2
import mediapipe as mp
import torch

# ---------------------------------------------------------------------
# Make torch.load behave for HSEmotion checkpoints:
# - allow full unpickling (weights_only=False) so timm classes can unpickle
# - map to CPU unless FACE_DEVICE=cuda and CUDA is available
# ---------------------------------------------------------------------
_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    want_cuda = os.getenv("FACE_DEVICE", "").lower() == "cuda"
    if "map_location" not in kwargs:
        if not (want_cuda and torch.cuda.is_available()):
            kwargs["map_location"] = torch.device("cpu")
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat
# ---------------------------------------------------------------------

from hsemotion.facial_emotions import HSEmotionRecognizer

LABELS_7 = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Single global MediaPipe detector
_mp_fd = mp.solutions.face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
)

def _pick_device() -> str:
    want = os.getenv("FACE_DEVICE", "").lower()
    if want == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"

def detect_face_bgr(frame_bgr: np.ndarray):
    """Returns (x1,y1,x2,y2) or None."""
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = _mp_fd.process(frame_rgb)
    if res.detections:
        det = res.detections[0]
        bbox = det.location_data.relative_bounding_box
        x1 = max(int(bbox.xmin * w), 0)
        y1 = max(int(bbox.ymin * h), 0)
        x2 = min(int((bbox.xmin + bbox.width) * w), w - 1)
        y2 = min(int((bbox.ymin + bbox.height) * h), h - 1)
        if x2 > x1 and y2 > y1:
            return (x1, y1, x2, y2)
    return None

def _crop_face(frame_bgr: np.ndarray, bbox):
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size > 0:
            return crop
    # fallback: center square
    h, w = frame_bgr.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return frame_bgr[y0:y0+s, x0:x0+s]

class FaceModel:
    """
    Thin wrapper around HSEmotion (enet_b2_7).
    Returns 7-dim probabilities aligned to LABELS_7.
    """
    def __init__(self, device="cpu"):
        self.device = device
        # This downloads/loads a pre-trained FER model. No local .pt needed.
        self.fer = HSEmotionRecognizer(model_name="enet_b2_7", device=device)

    def predict_probs(self, bgr_image: np.ndarray) -> np.ndarray:
        bbox = detect_face_bgr(bgr_image)
        face_bgr = _crop_face(bgr_image, bbox)

        # logits=False -> probabilities in 7-class order:
        # [Anger, Disgust, Fear, Happiness, Neutral, Sadness, Surprise]
        _, probs = self.fer.predict_emotions(face_bgr, logits=False)

        probs = np.asarray(probs, dtype=np.float32).reshape(-1)
        if probs.shape[0] != 7:
            p = probs[:7].astype(np.float32)
            p = np.exp(p - np.max(p)); p = p / (p.sum() + 1e-8)
            return p
        return probs

def load_face_model(_weights_path: str = ""):
    device = _pick_device()
    print(f"[face] HSEmotion enet_b2_7 on device={device} (no local .pt required)")
    return FaceModel(device=device)

def infer_face(model: FaceModel, frame_bgr: np.ndarray):
    probs = model.predict_probs(frame_bgr)
    pred = int(np.argmax(probs))
    return pred, probs
