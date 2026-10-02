# NixOS Docs Copilot

A retrieval-augmented assistant for the [NixOS manual](https://nixos.org/manual/nixos/stable/).
Ask a NixOS question in plain English. The answer comes only from the manual, with a deep link to every
section it cites, and the assistant says so when the manual doesn't cover the question.

**Live demo:** _coming soon_

## How it works

```
question ─► retrieve ─► grade ──relevant──► generate (cited answer)
               ▲          │
               │       nothing relevant
               └─ rewrite ◄┘   (at most once, then "not in the manual")
```

- **Ingestion** (`app/ingest.py`): fetches the single-page NixOS manual, splits it at every anchored
  heading, and builds a `Chapter › Section › Subsection` breadcrumb from the DocBook nesting, so a chunk
  titled "Quick Start" still says which service it belongs to. Sections are chunked with
  LangChain's `RecursiveCharacterTextSplitter` and embedded locally with `BAAI/bge-small-en-v1.5` (fastembed /
  ONNX, so no embedding API key is needed) into a persistent **Chroma** collection.
- **Agent** (`app/graph.py`): a corrective-RAG state machine in **LangGraph**. An LLM grader filters the
  retrieved passages with structured output. If nothing relevant survives, the agent rewrites the query
  into the manual's vocabulary (option names, commands) and retries once. The answer prompt only allows
  claims from the numbered passages and requires `[n]` citations.
- **API** (`app/main.py`): **FastAPI** with `POST /api/ask`, `GET /api/health`, request validation,
  per-request ids in the logs, and a 502 when the LLM provider fails, with no internals leaked.
- **Evals** (`evals/`): 22 hand-written questions (19 answerable, each with a reference answer and the
  manual section it comes from, plus 3 out-of-scope). The harness compares a plain-RAG baseline against
  the agent, scoring retrieval hit@k, whether the right section is cited, LLM-as-judge correctness (1-5)
  and faithfulness to the passages, and refusal on out-of-scope questions.

## Results

Measured on the eval set in `evals/dataset.jsonl` (NixOS manual 26.05, 970 chunks):

| metric | value |
|---|---|
| retrieval hit@3 / hit@6 (embedding search only) | 95% / 95% (18 of 19) |

<!-- full baseline-vs-agent table is added from a real `python -m evals.run_evals` run -->

Run the evals yourself. Raw per-question results are written to `evals/results/`.

```bash
python -m evals.run_evals --retrieval-only   # no API key needed
python -m evals.run_evals                    # baseline vs agent, LLM judge
```

## Run locally

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env        # add a GROQ_API_KEY (free) or switch to OpenAI
python -m app.ingest        # ~5 min: fetch, chunk, embed, index
uvicorn app.main:app --reload --port 7860
```

Open http://localhost:7860. Tests and lint, no API key needed:

```bash
pytest -q && ruff check .
```

## Deploy

The `Dockerfile` builds the index into the image, so the container starts ready to serve. It listens
on `$PORT` (default 7860) and needs `GROQ_API_KEY` (or `LLM_PROVIDER=openai` + `OPENAI_API_KEY`) as a secret.

## Stack

Python 3.13 · LangGraph · LangChain (core, text splitters, Chroma/Groq/OpenAI integrations) · Chroma ·
fastembed (bge-small-en-v1.5) · FastAPI · Docker · GitHub Actions

## License

MIT. The NixOS manual content is fetched at build time from nixos.org (MIT-licensed, © NixOS contributors).
