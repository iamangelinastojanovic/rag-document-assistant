# RAG Document Assistant

A retrieval-augmented question answering system over a local document collection
(`.md`, `.txt`, `.pdf`). It retrieves the passages most similar to a question,
answers **only** from those passages, and cites them. When nothing in the
documents is similar enough, it says so instead of guessing.

Runs entirely locally on CPU — no API keys, no external services.

## Design decisions

**Grounded answers or none at all.** The generator is instructed to use only the
retrieved context, and retrieval applies a minimum similarity threshold. If no
passage clears it, the system returns an explicit refusal rather than an
invented answer. In a document assistant, a confident wrong answer is worse than
"I don't know" — the user cannot tell the two apart without re-reading the
source themselves.

**Every answer carries its sources.** Each returned passage includes its origin
file and similarity score, so an answer can be verified against the document it
came from.

**Multilingual retrieval.** The embedding model
(`paraphrase-multilingual-MiniLM-L12-v2`) places English and Serbian text in the
same vector space, so a question in one language can retrieve a passage written
in the other.

**A flat index, deliberately.** Embeddings are a normalised NumPy matrix and
similarity is a single dot product. For a corpus this size that is exact,
instant, and has no moving parts — a vector database would add operational
complexity without improving results. It would become necessary somewhere in the
hundreds of thousands of chunks.

## Pipeline

```
documents (.md / .txt / .pdf)
   │  src/ingest.py
   ▼
chunks — 450 characters, 80-character overlap
   │  sentence-transformers, normalised
   ▼
data/index/  →  embeddings.npy + chunks.json
   │  src/retrieve.py — cosine similarity, top-k, min-score cutoff
   ▼
retrieved passages
   │  src/generate.py — FLAN-T5, context-only prompt
   ▼
answer + cited sources
```

Overlapping chunks exist so that a sentence spanning a boundary is still fully
present in at least one chunk.

## Retrieval evaluation

Generation quality is subjective; retrieval is not. `src/evaluate.py` measures
**hit@k** — for each question, does the document that actually contains the
answer appear in the top *k* retrieved passages?

The question set (`data/eval_questions.json`) has 7 questions across 4 source
documents covering sensor maintenance, data logging, alarm handling, and
equipment servicing — the last of these written in Serbian. Two questions cross
the language boundary in opposite directions (a Serbian question and an English
question both targeting the Serbian document), which is what exercises the
multilingual embedding rather than merely matching shared vocabulary.

```bash
python -m src.evaluate
```

This separates the two failure modes: if retrieval misses, no generator can
recover; if retrieval hits and the answer is still wrong, the problem is
generation.

## Limitations

- `flan-t5-small` is a small model. Retrieval works well in both languages, but
  generated phrasing is noticeably weaker in Serbian than in English.
- The index is held in memory — appropriate for this corpus size, not for a
  large one.
- No reranking stage; retrieval is a single similarity pass.
- The evaluation set is small (6 questions). It catches gross retrieval
  failures, not fine differences between configurations.

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
# source .venv/bin/activate       # Linux / macOS

pip install -r requirements.txt

python -m src.ingest       # build the index
python -m src.evaluate     # retrieval hit@k
python app.py              # Gradio interface
```

The first run downloads the embedding and generation models (~500 MB) and needs
an internet connection. Everything after that is local.

To use your own documents, put `.md`, `.txt` or `.pdf` files in
`data/knowledge/` and re-run `python -m src.ingest`.

## Layout

```
app.py                     Gradio interface
src/ingest.py              read, chunk, embed, save index
src/chunk.py               chunking with overlap
src/retrieve.py            cosine similarity search
src/generate.py            grounded generation, refusal on low similarity
src/evaluate.py            hit@k retrieval evaluation
src/pipeline.py            retrieve + generate
data/knowledge/            source documents
data/eval_questions.json   evaluation set
```

## Stack

Python · sentence-transformers · Transformers (FLAN-T5) · NumPy · pypdf · Gradio
