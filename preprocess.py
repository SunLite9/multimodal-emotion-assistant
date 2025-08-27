import numpy as np
import librosa

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-9)

def rms_level(pcm_int16: np.ndarray) -> float:
    # 0..1 scale; treat silence as ~0
    if pcm_int16.size == 0:
        return 0.0
    x = pcm_int16.astype(np.float32) / 32768.0
    rms = np.sqrt((x**2).mean())
    return float(np.clip(rms * 3.0, 0.0, 1.0))  # a little gain for visual effect

def mfcc_from_pcm_int16(pcm_int16: np.ndarray, sr=16000, n_mfcc=40, win_length=400, hop_length=160):
    # Convert to float, pre-emphasis optional
    x = pcm_int16.astype(np.float32) / 32768.0
    m = librosa.feature.mfcc(
        y=x, sr=sr, n_mfcc=n_mfcc,
        n_fft=512, hop_length=hop_length, win_length=win_length, center=True
    ).T  # shape [T, n_mfcc]
    # Optional: add deltas for better performance
    # d1 = librosa.feature.delta(m.T).T
    # d2 = librosa.feature.delta(m.T, order=2).T
    # m = np.concatenate([m, d1, d2], axis=1)
    return m.astype(np.float32)

def late_fusion(face_probs, voice_probs, w_face=0.4, w_voice=0.6):
    face = np.array(face_probs, dtype=np.float32)
    voice = np.array(voice_probs, dtype=np.float32)
    face = face / (face.sum() + 1e-8)
    voice = voice / (voice.sum() + 1e-8)
    p = w_face * face + w_voice * voice
    p = p / (p.sum() + 1e-8)
    return p.tolist()

def ravdess_emotion_from_filename(filename: str, merge_calm_to_neutral: bool = True) -> str:
    """
    Parse RAVDESS 7-part filename, e.g. 03-01-06-01-02-01-12.wav
    Emotion code (3rd field):
      01 neutral, 02 calm, 03 happy, 04 sad, 05 angry, 06 fearful, 07 disgust, 08 surprised.
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
