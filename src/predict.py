import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from src.model import Net
from src.preprocess import normalize_log


ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "log_model.pt"
META_PATH = ARTIFACTS_DIR / "meta.json"


def load_model() -> tuple[Net, SentenceTransformer, float]:
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    input_size = meta["input_size"]
    threshold = meta["threshold"]

    model = Net(input_size)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    embedder = SentenceTransformer(meta["model_name"], device="cpu")

    return model, embedder, threshold


def predict_log(text: str) -> tuple[str, float, str]:
    model, embedder, threshold = load_model()

    normalized = normalize_log(text)
    vec = torch.tensor(embedder.encode([normalized]), dtype=torch.float32)

    with torch.no_grad():
        pred = model(vec)
        score = pred.item()

    label = "ATTACK" if score > threshold else "NORMAL"
    return label, score, normalized