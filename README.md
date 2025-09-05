# Real-Time Multimodal Emotion Recognition AI

Real-time, browser→server demo that infers **audio**, **visual**, and **fused** emotions. The frontend streams mic/webcam to a **FastAPI** backend (drop-in swappable with Flask). The server runs lightweight inference and streams predictions back to the UI.

---

## How it works

1. **Capture & streaming**  
   When you click **Start**, the browser grabs your camera and microphone.  
   It sends video frames as base64 JPEGs and audio as Int16 chunks over a WebSocket to the FastAPI server (a few msgs/second).  

2. **Audio pipeline (voice)**  
   - The server keeps a short rolling 1-second buffer at 16 kHz.  
   - Each update, the buffer is converted to MFCCs and passed into the BiLSTM audio classifier.  
   - That returns a 7-value probability vector over emotions:  
     `Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise`.  

3. **Face pipeline (visual)**  
   - From the latest frame, MediaPipe finds a face box.  
   - The face crop is fed into **HSEmotion enet_b2_7** (pretrained).  
   - That model outputs its own 7-class probability vector (same label order as audio).  

4. **Smoothing & fusion**  
   - To reduce jitter, the app keeps short histories of recent audio and face probabilities and averages them (window sizes are tunable).  
   - If the mic level is near-silence, the app skips adding that audio step to avoid biasing to *Neutral*.  
   - After smoothing, the two probability vectors are fused:  

     ```
     final = w_voice * audio + (1 - w_voice) * face
     ```

     *(voice weight is configurable, default = 0.6)*  

5. **Model outputs shown in the UI**  
   - The top emotion from **Voice**, **Face**, and **Fused** are shown.  
   - A **Top-5 list** displays the highest fused probabilities with labels (first letter capitalized).  

6. **Server & UI loop**  
   - This loop repeats ~2× per second:  
     *receive data → run models → smooth → fuse → send results back to the page*.  
   - The page updates labels immediately.  
   - **Start** button = green, **Stop** button = red.  

7. **Pre-launch quick accuracy (optional)**  
   On startup (if enabled in env), the server quickly samples:  
   - **RAVDESS** clips for audio  
   - **FER-2013** test images for face  
   - Builds simple synthetic audio+face pairs for fused  
   - Prints **Top-1 / Top-5 accuracies** to your terminal before the app begins serving.  


---

## Models used

- **Audio (speech)**
  - Input: 16 kHz mono, MFCC(40).
  - Head: **BiLSTM** classifier.
  - Labels (7): `angry, disgust, fear, happy, neutral, sad, surprise` (*calm merged*).
  - Weights: `weights/audio_bilstm.pt`

- **Face (visual)**
  - Detector: **MediaPipe**.
  - FER head: **HSEmotion** `enet_b2_7` (auto-download; no local `.pt` required).
  - Labels mapped to app order: `angry, disgust, fear, happy, neutral, sad, surprise`.
  - Weights: `weights\face_mobilenetv3.pt`

- **Fusion**
  - Late fusion of softmax probabilities; weight toward audio by default (`w_voice=0.6`).

---

## Accuracy expectations

> Ballpark from the built-in quick checks (your numbers will vary with mic, lighting, and datasets).

- **Audio (RAVDESS, 7-class, windowed avg):** **~35–40% top-1**, **~90% top-5**.
- **Face (FER-2013/test):** **~50% top-1**, **~95% top-5**.
- **Fused (synthetic pairing):** **~55–60% top-1**  
  *(pairs audio & face of same label from different datasets; not a true multimodal benchmark).*

---

## Quick start

```powershell
# 1) Create venv (Python 3.11.x recommended)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 2) Install deps
pip install -r requirements.txt

# 3) Pin versions that interoperate for the face path
pip install --upgrade --force-reinstall "hsemotion==0.3.0" "timm==0.9.16"

# 4) Ensure NumPy < 2.3 for numba/librosa stacks
pip install "numpy<2.3"   # e.g., 2.2.6

# 5) Put your audio model weights in place
#    (file should be tens of MB; not an LFS pointer)
#    weights/audio_bilstm.pt

# 6) Run the server
uvicorn app:app --reload
# Open http://127.0.0.1:8000/
```

### Credits

- **RAVDESS Emotional Speech Audio** — Ryerson Audio-Visual Database of Emotional Speech and Song.
- **FER-2013** — Facial Expression Recognition dataset (Kaggle).
- **HSEmotion** — Pretrained facial emotion models (`enet_b2_7`).
- **MediaPipe** — Face detection.
- **FastAPI / Uvicorn** — WebSocket server runtime.
- **NumPy / Librosa / PyTorch / timm** — DSP & model tooling.

