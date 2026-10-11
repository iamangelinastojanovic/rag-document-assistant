from __future__ import annotations

from src.generate import answer_from_chunks
from src.retrieve import retrieve


def ask(question: str, top_k: int = 4, min_score: float = 0.28) -> dict:
    question = (question or "").strip()
    if not question:
        return {"answer": "Ask a question first.", "sources": []}

    chunks = retrieve(question, top_k=top_k, min_score=min_score)
    answer = answer_from_chunks(question, chunks)
    return {"answer": answer, "sources": chunks}
