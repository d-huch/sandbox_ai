import json
import csv
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from src.embeddings import encode_with_cache, get_embedder
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


def load_model() -> tuple[Net, SentenceTransformer, float, str]:
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model_name = meta["model_name"]
    input_size = meta["input_size"]
    threshold = meta["threshold"]

    model = Net(input_size)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    embedder = get_embedder(model_name)

    return model, embedder, threshold, model_name


def predict_log(
    text: str,
    model: Net | None = None,
    embedder: SentenceTransformer | None = None,
    threshold: float | None = None,
    model_name: str | None = None,
) -> tuple[str, float, str]:
    if model is None or embedder is None or threshold is None or model_name is None:
        model, embedder, threshold, model_name = load_model()

    normalized = normalize_log(text)
    vec = encode_with_cache([normalized], model_name)

    with torch.no_grad():
        pred = model(vec)
        score = pred.item()

    label = "ATTACK" if score > threshold else "NORMAL"
    return label, score, normalized


def predict_logs(texts: list[str]) -> list[dict[str, Any]]:
    if not texts:
        return []

    model, embedder, threshold, model_name = load_model()
    normalized_texts = [normalize_log(text) for text in texts]
    vectors = encode_with_cache(normalized_texts, model_name)

    with torch.no_grad():
        scores = model(vectors).squeeze(1).tolist()

    results: list[dict[str, Any]] = []

    for text, normalized, score in zip(texts, normalized_texts, scores):
        label = "ATTACK" if score > threshold else "NORMAL"
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
