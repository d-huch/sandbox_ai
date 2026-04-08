import hashlib
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


ARTIFACTS_DIR = Path("artifacts")
EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"
_EMBEDDER_CACHE: dict[str, SentenceTransformer] = {}


def _cache_file_for_model(model_name: str) -> Path:
    model_hash = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:16]
    return EMBEDDINGS_DIR / f"{model_hash}.pt"


def get_embedder(model_name: str) -> SentenceTransformer:
    embedder = _EMBEDDER_CACHE.get(model_name)
    if embedder is None:
        embedder = SentenceTransformer(model_name, device="cpu")
        _EMBEDDER_CACHE[model_name] = embedder
    return embedder


def load_embedding_cache(model_name: str) -> dict[str, list[float]]:
    cache_path = _cache_file_for_model(model_name)
    if not cache_path.exists():
        return {}

    cache = torch.load(cache_path, map_location="cpu")
    if isinstance(cache, dict):
        return cache
    return {}


def save_embedding_cache(model_name: str, cache: dict[str, list[float]]) -> None:
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_file_for_model(model_name)
    torch.save(cache, cache_path)


def encode_with_cache(texts: list[str], model_name: str) -> torch.Tensor:
    embedder = get_embedder(model_name)
    cache = load_embedding_cache(model_name)

    missing_texts = [text for text in texts if text not in cache]
    if missing_texts:
        new_vectors = embedder.encode(missing_texts)
        for text, vector in zip(missing_texts, new_vectors):
            cache[text] = [float(value) for value in vector]
        save_embedding_cache(model_name, cache)

    vectors = [cache[text] for text in texts]
    return torch.tensor(vectors, dtype=torch.float32)
