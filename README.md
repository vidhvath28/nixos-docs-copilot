---
title: NixOS Docs Copilot
emoji: ❄️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Corrective-RAG assistant for the NixOS manual
---

# NixOS Docs Copilot

A retrieval-augmented assistant for the [NixOS manual](https://nixos.org/manual/nixos/stable/).
Ask a NixOS question in plain English. The answer comes only from the manual, with a deep link to every
section it cites, and the assistant says so when the manual doesn't cover the question.

**Live demo:** [huggingface.co/spaces/Vidhvath/Nixos](https://huggingface.co/spaces/Vidhvath/Nixos)

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

Measured on `evals/dataset.jsonl` (NixOS manual 26.05, 970 chunks). Answers come from `openai/gpt-oss-120b`
on Groq and are judged by a **different** model (`qwen/qwen3.8-27b`), so the model isn't grading its own output.
The raw per-question output is in [`evals/results/20261002T120422Z.json`](evals/results/20261002T120422Z.json).

| metric | plain RAG baseline | LangGraph agent |
|---|---|---|
| correctness, judge score 1-5 | 4.74 | **4.95** |
| answers scored ≥ 4 | 95% | **100%** |
| faithful to the retrieved passages | 74% | **84%** |
| right manual section cited | 95% | **100%** |
| refuses the 3 out-of-scope questions | 100% | 100% |
| p50 latency | 1.2 s | 2.2 s |

Retrieval alone (embedding search, hit@6) finds the right section for 18 of 19 answerable questions.

What the eval found along the way: on the first run the agent refused "how do I open ports 80 and 443". Search
had returned another service's firewall page, the grader passed it, and the generator correctly found no answer
in it. Because the grader had passed the page, the rewrite step never ran. The fix sends "not found" answers
from graded passages through one query rewrite too (`route_after_generate` in `app/graph.py`, covered by a test).

Caveats: the set is small (19 answerable questions, so one answer moves a rate by about 5 points), and LLM-judge
scores vary between runs. Agent faithfulness scored 90% on an earlier run of the previous code and 84% on this one.
The baseline was 74% both times. Latency is on Groq's free tier, with a pause between questions to stay under its
rate limit.

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

The `Dockerfile` builds the index into the image, so the container starts ready to serve
(about 360 MB RSS, which fits a 512 MB free instance). `render.yaml` is a Render blueprint: New → Blueprint →
pick this repo → enter `GROQ_API_KEY`. It listens
on `$PORT` (default 7860) and needs `GROQ_API_KEY` (or `LLM_PROVIDER=openai` + `OPENAI_API_KEY`) as a secret.

## Stack

Python 3.13 · LangGraph · LangChain (core, text splitters, Chroma/Groq/OpenAI integrations) · Chroma ·
fastembed (bge-small-en-v1.5) · FastAPI · Docker · GitHub Actions

## License

MIT. The NixOS manual content is fetched at build time from nixos.org (MIT-licensed, © NixOS contributors).
