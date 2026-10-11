"""Retrieval evaluation: does the right document appear in top-k?"""

from __future__ import annotations

import json
import sys

from src.paths import EVAL_PATH
from src.retrieve import retrieve


def evaluate(top_k: int = 4) -> None:
    if not EVAL_PATH.exists():
        raise FileNotFoundError(f"Missing {EVAL_PATH}")

    items = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    if not items:
        raise ValueError(f"{EVAL_PATH} has no questions")

    hits = 0
    print(f"Evaluating {len(items)} questions (hit@{top_k})\n")

    for row in items:
        question = row["question"]
        expected = row["expected_source"]
        results = retrieve(question, top_k=top_k, min_score=0.0)
        sources = [item["source"] for item in results]
        ok = expected in sources
        hits += int(ok)
        mark = "HIT " if ok else "MISS"
        print(f"{mark} | {question}")
        print(f"       expected: {expected}")
        print(f"       got: {sources[:top_k]}\n")

    print(f"hit@{top_k}: {hits}/{len(items)} = {hits / len(items):.2f}")


if __name__ == "__main__":
    try:
        evaluate()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
