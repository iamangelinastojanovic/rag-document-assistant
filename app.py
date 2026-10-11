"""Gradio demo: ask a question over the local document folder."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr

from src.ingest import build_index
from src.pipeline import ask
from src.retrieve import reset_cache


def format_sources(sources: list[dict]) -> str:
    if not sources:
        return "No passage passed the similarity threshold."
    lines = []
    for item in sources:
        lines.append(
            f"- {item['source']} (score {item['score']})\n  {item['text'][:400]}"
        )
    return "\n\n".join(lines)


def respond(question: str, top_k: float, min_score: float):
    try:
        result = ask(question, top_k=int(top_k), min_score=float(min_score))
        return result["answer"], format_sources(result["sources"])
    except Exception as exc:
        return f"Error: {exc}", ""


def rebuild():
    try:
        reset_cache()
        build_index()
        reset_cache()
        return "Index rebuilt. You can ask questions again."
    except Exception as exc:
        return f"Rebuild failed: {exc}"


with gr.Blocks(title="RAG asistent nad dokumentima") as demo:
    gr.Markdown(
        """
# RAG asistent nad dokumentima

Pitanje ide kroz **embeddings + pretragu pasusa**, pa mali **FLAN-T5** model
odgovara samo iz pronađenog teksta. Ako sličnost nije dovoljna, sistem kaže da ne zna.

Prvo izgradi indeks: `python -m src.ingest` (ili dugme ispod).
        """
    )
    with gr.Row():
        question = gr.Textbox(
            label="Question",
            lines=3,
            placeholder="When do applications close?",
        )
    with gr.Row():
        top_k = gr.Slider(1, 8, value=4, step=1, label="top_k passages")
        min_score = gr.Slider(0.0, 0.8, value=0.28, step=0.01, label="min cosine score")
    ask_btn = gr.Button("Ask", variant="primary")
    rebuild_btn = gr.Button("Rebuild index from data/knowledge")
    answer = gr.Textbox(label="Answer", lines=4)
    sources = gr.Textbox(label="Retrieved sources", lines=12)
    status = gr.Textbox(label="Index status", lines=1)

    ask_btn.click(
        fn=respond,
        inputs=[question, top_k, min_score],
        outputs=[answer, sources],
    )
    question.submit(
        fn=respond,
        inputs=[question, top_k, min_score],
        outputs=[answer, sources],
    )
    rebuild_btn.click(fn=rebuild, outputs=status)

    gr.Examples(
        examples=[
            ["When do applications close?"],
            ["How many students are admitted to the AI track?"],
            ["What should a RAG system do if the answer is not in the documents?"],
            ["Do kog datuma se šalju prijave?"],
            ["Who won the 1994 World Cup?"],
        ],
        inputs=question,
    )


if __name__ == "__main__":
    demo.launch()
