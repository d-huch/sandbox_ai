import json
import csv
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from src.model import Net
from src.preprocess import normalize_log


ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "log_model.pt"
META_PATH = ARTIFACTS_DIR / "meta.json"


def resolve_input_path(path: str) -> Path:
    file_path = Path(path).expanduser()
    if not file_path.is_absolute():
        file_path = Path.cwd() / file_path
    return file_path.resolve(strict=False)


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


def predict_log(
    text: str,
    model: Net | None = None,
    embedder: SentenceTransformer | None = None,
    threshold: float | None = None,
) -> tuple[str, float, str]:
    if model is None or embedder is None or threshold is None:
        model, embedder, threshold = load_model()

    normalized = normalize_log(text)
    vec = torch.tensor(embedder.encode([normalized]), dtype=torch.float32)

    with torch.no_grad():
        pred = model(vec)
        score = pred.item()

    label = "ATTACK" if score > threshold else "NORMAL"
    return label, score, normalized


def predict_logs(texts: list[str]) -> list[dict[str, Any]]:
    model, embedder, threshold = load_model()
    results: list[dict[str, Any]] = []

    for text in texts:
        label, score, normalized = predict_log(text, model, embedder, threshold)
        results.append(
            {
                "text": text,
                "normalized": normalized,
                "score": score,
                "prediction": label,
            }
        )

    return results


def predict_txt_file(path: str) -> list[dict[str, Any]]:
    file_path = resolve_input_path(path)

    if not file_path.is_file():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    return predict_logs(texts)


def predict_csv_file(path: str) -> list[dict[str, Any]]:
    file_path = resolve_input_path(path)

    if not file_path.is_file():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError("CSV file must contain a 'text' column.")

        texts = [row["text"].strip() for row in reader if row.get("text", "").strip()]

    return predict_logs(texts)


def predict_file(path: str) -> list[dict[str, Any]]:
    suffix = resolve_input_path(path).suffix.lower()

    if suffix == ".txt":
        return predict_txt_file(path)
    if suffix == ".csv":
        return predict_csv_file(path)

    raise ValueError("Supported file types: .txt, .csv")
