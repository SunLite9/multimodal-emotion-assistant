import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
import mediapipe as mp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MediaPipe face detector
_mp_fd = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Preprocess for ImageNet backbones
_img_tf = transforms.Compose([
    transforms.ToTensor(),                              # HWC [0,255] -> CHW [0,1]
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def build_face_model(num_classes=7):
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_feats = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_feats, num_classes)
    return m.to(DEVICE).eval()

def load_face_model(weights_path: str):
    model = build_face_model(num_classes=7)
    if weights_path and os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(state)
        print(f"[face] loaded weights: {weights_path}")
    else:
        print("[face] no weights provided; using ImageNet-pretrained backbone with random head")
    return model

def detect_face_bgr(frame_bgr):
    # Returns (x1,y1,x2,y2) in pixel coords or None
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
        return (x1, y1, x2, y2)
    return None

def preprocess_face(frame_bgr, bbox):
    x1,y1,x2,y2 = bbox
    crop = frame_bgr[y1:y2, x1:x2].copy()
    if crop.size == 0:
        crop = frame_bgr
    img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tens = _img_tf(img).unsqueeze(0)  # [1,3,224,224]
    return tens.to(DEVICE)

@torch.inference_mode()
def infer_face(model, frame_bgr):
    bbox = detect_face_bgr(frame_bgr)
    if bbox is None:
        return None, None
    x = preprocess_face(frame_bgr, bbox)
    logits = model(x)
    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    pred = int(probs.argmax())
    return pred, probs
