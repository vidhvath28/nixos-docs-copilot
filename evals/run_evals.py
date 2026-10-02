"""Eval harness: plain RAG baseline vs. the corrective LangGraph agent.

    python -m evals.run_evals --retrieval-only   # hit@k, no API key needed
    python -m evals.run_evals                    # full run with an LLM judge

Metrics
  hit@k        a section the reference answer comes from is among the top-k retrieved (in-scope questions)
  cited-hit    that section survives into the sources the answer actually cites from
  correctness  judge score 1-5 against the reference answer
  faithful     judge: every claim in the answer is supported by the passages it was given
  refusal      out-of-scope questions answered with the "not in the manual" message
"""

import argparse
import json
import logging
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.config import get_settings
from app.graph import build_graph, chroma_search, format_passages, is_not_found
from app.llm import get_vectorstore, make_chat_model

HERE = Path(__file__).parent
log = logging.getLogger("evals")


class Judgement(BaseModel):
    correctness: int = Field(
        ge=1, le=5, description="1 = wrong or missing the key point, 5 = fully matches the reference"
    )
    faithful: bool = Field(description="true if every factual claim in the answer is supported by the passages")
    reason: str = Field(description="one sentence")


JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict grader for a NixOS documentation assistant. Compare the ANSWER with the "
            "REFERENCE for correctness (does it give the same option names, commands and key facts?). "
            "Separately decide whether every claim in the ANSWER is supported by the PASSAGES it was given; "
            "claims not in the passages make it unfaithful even if they happen to be true.",
        ),
        ("human", "QUESTION: {question}\n\nREFERENCE: {reference}\n\nANSWER: {answer}\n\nPASSAGES:\n{passages}"),
    ]
)


def load_dataset() -> list[dict]:
    return [json.loads(line) for line in (HERE / "dataset.jsonl").read_text().splitlines() if line.strip()]


def anchors_of(docs) -> list[str]:
    return [d.metadata.get("anchor") for d in docs]


def with_retry(fn, *args, attempts: int = 5):
    """Free-tier LLM APIs rate-limit hard; back off instead of failing the whole run."""
    for i in range(attempts):
        try:
            return fn(*args)
        except Exception as exc:  # provider-specific RateLimitError classes differ
            if i == attempts - 1 or "rate" not in str(exc).lower() and "429" not in str(exc):
                raise
            wait = 2**i * 5
            log.warning("rate limited, retrying in %ss", wait)
            time.sleep(wait)


def retrieval_eval(items: list[dict], k: int) -> dict:
    vs = get_vectorstore()
    rows = []
    for it in items:
        if not it["anchors"]:
            continue
        got = anchors_of(vs.similarity_search(it["question"], k=k))
        rows.append({"id": it["id"], "hit": bool(set(got) & set(it["anchors"])), "retrieved": got})
    return {"k": k, "hit_rate": sum(r["hit"] for r in rows) / len(rows), "n": len(rows), "rows": rows}


def full_eval(items: list[dict], pause: float = 0.0) -> dict:
    s = get_settings()
    search = chroma_search(get_vectorstore())
    llm = make_chat_model()
    judge = JUDGE_PROMPT | make_chat_model(s.judge_model).with_structured_output(Judgement)
    report = {}
    for mode in ("baseline", "agent"):
        graph = build_graph(llm, search, top_k=s.top_k, max_rewrites=s.max_rewrites, mode=mode)
        rows = []
        for it in items:
            time.sleep(pause)  # stay under free-tier tokens/minute so latency isn't mostly throttling
            t0 = time.perf_counter()
            out = with_retry(graph.invoke, {"question": it["question"]})
            row = {
                "id": it["id"],
                "ms": int((time.perf_counter() - t0) * 1000),
                "steps": out["steps"],
                "answer": out["answer"],
            }
            refused = is_not_found(out["answer"])
            if it["reference"] is None:
                row["refused"] = refused
            else:
                cited = anchors_of(out["documents"])
                row["cited_hit"] = bool(set(cited) & set(it["anchors"]))
                if refused:
                    row.update(correctness=1, faithful=True, reason="refused an answerable question")
                else:
                    j = with_retry(
                        judge.invoke,
                        {
                            "question": it["question"],
                            "reference": it["reference"],
                            "answer": out["answer"],
                            "passages": format_passages(out["documents"]),
                        },
                    )
                    row.update(correctness=j.correctness, faithful=j.faithful, reason=j.reason)
            log.info("%s %s %s", mode, it["id"], {k: v for k, v in row.items() if k not in ("answer", "steps")})
            rows.append(row)
        scored = [r for r in rows if "correctness" in r]
        oos = [r for r in rows if "refused" in r]
        report[mode] = {
            "correctness_avg": round(statistics.mean(r["correctness"] for r in scored), 2),
            "correct_rate": round(sum(r["correctness"] >= 4 for r in scored) / len(scored), 3),
            "faithful_rate": round(sum(r["faithful"] for r in scored) / len(scored), 3),
            "cited_hit_rate": round(sum(r["cited_hit"] for r in scored) / len(scored), 3),
            "refusal_rate": round(sum(r["refused"] for r in oos) / len(oos), 3) if oos else None,
            "p50_ms": int(statistics.median(r["ms"] for r in rows)),
            "rows": rows,
        }
    return report


def summary_table(retrieval: dict, full: dict | None) -> str:
    lines = [f"Retrieval hit@{retrieval['k']}: {retrieval['hit_rate']:.0%} ({retrieval['n']} in-scope questions)"]
    if full:
        lines += ["", "| metric | baseline (plain RAG) | agent (LangGraph CRAG) |", "|---|---|---|"]
        for key, label in [
            ("correctness_avg", "correctness (1-5)"),
            ("correct_rate", "answers scored >= 4"),
            ("faithful_rate", "faithful to passages"),
            ("cited_hit_rate", "right section cited"),
            ("refusal_rate", "refuses out-of-scope"),
            ("p50_ms", "p50 latency (ms)"),
        ]:
            b, a = full["baseline"][key], full["agent"][key]
            fmt = (lambda v: f"{v:.0%}") if isinstance(b, float) and key.endswith("rate") else str
            lines.append(f"| {label} | {fmt(b)} | {fmt(a)} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--pause", type=float, default=0.0, help="seconds to wait between questions")
    ap.add_argument("--k", type=int, default=get_settings().top_k)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    s = get_settings()
    items = load_dataset()
    retrieval = retrieval_eval(items, args.k)
    full = None if args.retrieval_only else full_eval(items, args.pause)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    meta = {
        "timestamp": stamp,
        "llm": f"{s.llm_provider}/{s.llm_model}",
        "judge": s.judge_model or s.llm_model,
        "embedding_model": s.embedding_model,
        "top_k": args.k,
    }
    (out_dir / f"{stamp}.json").write_text(json.dumps({"meta": meta, "retrieval": retrieval, "full": full}, indent=2))
    table = summary_table(retrieval, full)
    print(f"\n{json.dumps(meta)}\n\n{table}\n\nsaved evals/results/{stamp}.json")


if __name__ == "__main__":
    main()
