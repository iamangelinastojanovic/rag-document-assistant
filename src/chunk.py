from __future__ import annotations


def chunk_text(
    text: str,
    *,
    source: str,
    chunk_size: int = 450,
    overlap: int = 80,
) -> list[dict]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    overlap = min(overlap, max(0, chunk_size - 1))
    chunks: list[dict] = []
    start = 0
    index = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(
                {
                    "id": f"{source}::{index}",
                    "source": source,
                    "text": piece,
                }
            )
            index += 1
        if end == len(cleaned):
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks
