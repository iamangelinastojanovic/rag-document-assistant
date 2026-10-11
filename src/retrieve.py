from __future__ import annotations

import json

import numpy as np
from sentence_transformers import SentenceTransformer

from src.paths import CHUNKS_PATH, EMBEDDING_MODEL_ID, EMBEDDINGS_PATH

_model = None
_chunks = None
_embeddings = None


def reset_cache() -> None:
    global _model, _chunks, _embeddings
    _model = None
    _chunks = None
    _embeddings = None


def load_index():
    global _chunks, _embeddings
    if _chunks is None:
        if not CHUNKS_PATH.exists() or not EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(
                "Index not found. Run: python -m src.ingest"
            )
        _chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        embeddings = np.load(EMBEDDINGS_PATH)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        _embeddings = embeddings.astype(np.float32, copy=False)
        if len(_chunks) != len(_embeddings):
            raise ValueError(
                "Index is corrupt: chunk count does not match embedding rows. "
                "Rebuild with: python -m src.ingest"
            )
        if len(_chunks) == 0:
            raise ValueError("Index is empty. Add documents and run python -m src.ingest")
    return _chunks, _embeddings


def get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_ID)
    return _model


def retrieve(question: str, top_k: int = 4, min_score: float = 0.25) -> list[dict]:
    question = (question or "").strip()
    if not question:
        return []
    chunks, embeddings = load_index()
    query = get_embedder().encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0].astype(np.float32, copy=False)
    scores = embeddings @ query
    order = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in order:
        score = float(scores[idx])
        if score < min_score:
            continue
        item = dict(chunks[int(idx)])
        item["score"] = round(score, 3)
        results.append(item)
    return results
