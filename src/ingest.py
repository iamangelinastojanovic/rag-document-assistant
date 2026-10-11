"""Read .md/.txt/.pdf files, chunk them, embed, and save a local index."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from src.chunk import chunk_text
from src.paths import (
    CHUNKS_PATH,
    EMBEDDING_MODEL_ID,
    EMBEDDINGS_PATH,
    INDEX_DIR,
    KNOWLEDGE_DIR,
)

SUPPORTED = {".md", ".txt", ".pdf"}


def read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            pages.append(f"[page {i}] {page_text}")
        return "\n".join(pages)
    raise ValueError(f"Unsupported file type: {path}")


def collect_chunks() -> list[dict]:
    if not KNOWLEDGE_DIR.exists():
        raise FileNotFoundError(f"Missing knowledge folder: {KNOWLEDGE_DIR}")

    chunks: list[dict] = []
    files = sorted(
        p for p in KNOWLEDGE_DIR.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED
    )
    if not files:
        raise FileNotFoundError(f"No .md/.txt/.pdf files in {KNOWLEDGE_DIR}")

    for path in files:
        relative = str(path.relative_to(KNOWLEDGE_DIR)).replace("\\", "/")
        text = read_file(path)
        chunks.extend(chunk_text(text, source=relative))
    return chunks


def build_index() -> None:
    chunks = collect_chunks()
    if not chunks:
        raise ValueError(
            f"No text could be extracted from files in {KNOWLEDGE_DIR}"
        )
    model = SentenceTransformer(EMBEDDING_MODEL_ID)
    embeddings = model.encode(
        [item["text"] for item in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(EMBEDDINGS_PATH, embeddings.astype(np.float32))
    print(f"Indexed {len(chunks)} chunks from {KNOWLEDGE_DIR}")
    print(f"Saved {CHUNKS_PATH.name} and {EMBEDDINGS_PATH.name}")


if __name__ == "__main__":
    build_index()
