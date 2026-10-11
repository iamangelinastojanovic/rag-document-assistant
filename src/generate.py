"""Grounded answers: generate only from retrieved chunks, or refuse."""

from __future__ import annotations

from transformers import pipeline

from src.paths import GENERATOR_MODEL_ID

_generator = None

REFUSAL = (
    "I do not know based on the documents. "
    "The retrieved passages are not similar enough to the question."
)


def get_generator():
    global _generator
    if _generator is None:
        _generator = pipeline(
            "text2text-generation",
            model=GENERATOR_MODEL_ID,
        )
    return _generator


def _context_for_t5(chunks: list[dict], budget: int = 1400) -> str:
    """FLAN-T5-small truncates around 512 tokens; keep context short."""
    parts: list[str] = []
    used = 0
    for item in chunks:
        block = f"Source {item['source']}: {item['text']}"
        if used + len(block) > budget:
            remaining = budget - used
            if remaining > 80:
                parts.append(block[:remaining])
            break
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts)


def answer_from_chunks(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return REFUSAL

    context = _context_for_t5(chunks)
    prompt = (
        "Answer the question using only the context. "
        "If the context does not contain the answer, say you do not know. "
        "Be short and factual.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    output = get_generator()(
        prompt,
        max_new_tokens=80,
        truncation=True,
    )[0]["generated_text"].strip()
    return output or REFUSAL
