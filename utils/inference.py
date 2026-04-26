import os
import numpy as np
from sklearn.ensemble import IsolationForest

import torch
import streamlit as st

from model.vit_classifier import ViTClassifier

MODEL_DIR = "model"
BEST_MODEL_FILE = "best_model_phase2_sslvit.pth"
BACKBONE_FILE = "vitbackbone.pth"   # optional

BEST_MODEL_PATH = os.path.join(MODEL_DIR, BEST_MODEL_FILE)
BACKBONE_PATH = os.path.join(MODEL_DIR, BACKBONE_FILE)

ANOMALY_MODEL: IsolationForest | None = None
ANOMALY_FIT_DONE: bool = False


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone_path = BACKBONE_PATH if os.path.exists(BACKBONE_PATH) else None
    model = ViTClassifier(num_classes=6, backbone_path=backbone_path)

    if not os.path.exists(BEST_MODEL_PATH):
        st.error(f"Model checkpoint not found: {BEST_MODEL_PATH}")
        return None

    ckpt = torch.load(BEST_MODEL_PATH, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)
    return model, device


def get_anomaly_score(feature_vec: np.ndarray) -> tuple[float, float]:
    """
    Returns (raw_score, anomaly_percentage).
    Higher percentage = more anomalous.
    """
    global ANOMALY_MODEL, ANOMALY_FIT_DONE

    if ANOMALY_MODEL is None or not ANOMALY_FIT_DONE:
        ANOMALY_MODEL = IsolationForest(
            contamination=0.05,
            random_state=42,
        )
        baseline = feature_vec.reshape(1, -1)
        ANOMALY_MODEL.fit(baseline)
        ANOMALY_FIT_DONE = True

    score = -ANOMALY_MODEL.decision_function(feature_vec.reshape(1, -1))[0]
    anomaly_pct = float(100.0 * (1.0 / (1.0 + np.exp(-5 * (score - 0.5)))))
    return float(score), anomaly_pct
