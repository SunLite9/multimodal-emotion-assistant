import os
import torch
import torch.nn as nn
import numpy as np

# ... keep imports ...

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class BiLSTMClassifier(nn.Module):
    def __init__(self, n_mfcc=40, hidden=128, layers=2, num_classes=7, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_mfcc, hidden_size=hidden, num_layers=layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden*2, num_classes)

    def forward(self, x):  # x: [B, T, n_mfcc]
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)

def build_audio_bilstm(n_mfcc=40, hidden=128, layers=2, num_classes=7):
    m = BiLSTMClassifier(n_mfcc=n_mfcc, hidden=hidden, layers=layers, num_classes=num_classes)
    return m.to(DEVICE).eval()

def load_audio_model(weights_path: str, num_classes: int = 7):
    model = build_audio_bilstm(n_mfcc=40, hidden=128, layers=2, num_classes=num_classes)
    if weights_path and os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(state)
        print(f"[audio] loaded weights: {weights_path}")
    else:
        print(f"[audio] no weights provided; using randomly-initialized BiLSTM head (num_classes={num_classes})")
    return model

@torch.inference_mode()
def infer_audio(model, mfcc: np.ndarray):
    x = torch.from_numpy(mfcc).float().unsqueeze(0).to(DEVICE)  # [1,T,F]
    logits = model(x)
    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    pred = int(probs.argmax())
    return pred, probs
