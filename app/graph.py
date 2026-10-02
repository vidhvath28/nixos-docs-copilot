"""Corrective-RAG agent as a LangGraph state machine.

    retrieve -> grade -> generate
                  \\-> rewrite -> retrieve   (when nothing relevant came back)

`mode="baseline"` skips grading/rewriting (plain retrieve -> generate) so the
eval harness can measure what the extra steps buy.
"""

import logging
import time
from collections.abc import Callable
from typing import Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

log = logging.getLogger("graph")

Search = Callable[[str, int], list[Document]]
NOT_FOUND = "I couldn't find this in the NixOS manual."


class RagState(TypedDict, total=False):
    question: str
    query: str
    documents: list[Document]
    rewrites: int
    answer: str
    sources: list[dict]
    steps: list[str]


class Relevant(BaseModel):
    """Which passages help answer the question."""

    relevant: list[int] = Field(description="1-based numbers of the passages that help answer the question")


class Rewrite(BaseModel):
    query: str = Field(description="a search query more likely to match the NixOS manual's wording")


GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You grade search results for a NixOS documentation assistant. Return the numbers of the "
            "passages that contain information useful for answering the question. Return an empty list if none do.",
        ),
        ("human", "Question: {question}\n\nPassages:\n{passages}"),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "The search for the user's question returned nothing relevant from the NixOS manual. "
            "Rewrite it as a short search query using the terms the manual is likely to use "
            "(module names, option names such as services.<name>.enable, command names).",
        ),
        ("human", "Question: {question}\nPrevious query: {query}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions about NixOS using ONLY the numbered passages from the NixOS manual. "
            "Cite every claim with its passage number like [1] or [2][3]. Include configuration snippets "
            "when the passages have them. If the passages do not contain the answer, reply exactly: "
            f'"{NOT_FOUND}"',
        ),
        ("human", "Question: {question}\n\nPassages:\n{passages}"),
    ]
)


def format_passages(docs: list[Document]) -> str:
    return "\n\n".join(f"[{i}] ({d.metadata.get('title', '')})\n{d.page_content}" for i, d in enumerate(docs, 1))


def is_not_found(answer: str) -> bool:
    return answer.strip().strip('"') == NOT_FOUND


def build_graph(
    llm: BaseChatModel,
    search: Search,
    *,
    top_k: int = 6,
    max_rewrites: int = 1,
    mode: Literal["agent", "baseline"] = "agent",
):
    grader = GRADE_PROMPT | llm.with_structured_output(Relevant)
    rewriter = REWRITE_PROMPT | llm.with_structured_output(Rewrite)
    answerer = ANSWER_PROMPT | llm

    def retrieve(state: RagState) -> RagState:
        query = state.get("query") or state["question"]
        t0 = time.perf_counter()
        docs = search(query, top_k)
        log.info("retrieve query=%r hits=%d ms=%.0f", query, len(docs), (time.perf_counter() - t0) * 1000)
        return {"query": query, "documents": docs, "steps": [*state.get("steps", []), f"retrieve: {query}"]}

    def grade(state: RagState) -> RagState:
        docs = state["documents"]
        if not docs:
            return {"steps": [*state["steps"], "grade: 0 hits"]}
        result = grader.invoke({"question": state["question"], "passages": format_passages(docs)})
        keep = [docs[i - 1] for i in sorted(set(result.relevant)) if 1 <= i <= len(docs)]
        log.info("grade kept=%d/%d", len(keep), len(docs))
        return {"documents": keep, "steps": [*state["steps"], f"grade: kept {len(keep)}/{len(docs)}"]}

    def route_after_grade(state: RagState) -> str:
        if state["documents"]:
            return "generate"
        if state.get("rewrites", 0) < max_rewrites:
            return "rewrite"
        return "generate"

    def rewrite(state: RagState) -> RagState:
        result = rewriter.invoke({"question": state["question"], "query": state["query"]})
        log.info("rewrite %r -> %r", state["query"], result.query)
        return {
            "query": result.query,
            "rewrites": state.get("rewrites", 0) + 1,
            "steps": [*state["steps"], f"rewrite: {result.query}"],
        }

    def generate(state: RagState) -> RagState:
        docs = state["documents"]
        if not docs:
            return {"answer": NOT_FOUND, "sources": [], "steps": [*state["steps"], "generate: no context"]}
        msg = answerer.invoke({"question": state["question"], "passages": format_passages(docs)})
        if is_not_found(msg.content):
            # The grader passed these passages but they didn't hold the answer - nothing to cite.
            return {"answer": NOT_FOUND, "sources": [], "steps": [*state["steps"], "generate: not in passages"]}
        sources = [
            {"n": i, "title": d.metadata.get("title"), "url": d.metadata.get("url")} for i, d in enumerate(docs, 1)
        ]
        return {"answer": msg.content, "sources": sources, "steps": [*state["steps"], "generate"]}

    def route_after_generate(state: RagState) -> str:
        # Passages that looked relevant but didn't answer the question get one more search, too.
        if state["answer"] == NOT_FOUND and state["documents"] and state.get("rewrites", 0) < max_rewrites:
            return "rewrite"
        return END

    g = StateGraph(RagState)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_edge(START, "retrieve")
    if mode == "baseline":
        g.add_edge("retrieve", "generate")
    else:
        g.add_node("grade", grade)
        g.add_node("rewrite", rewrite)
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges("grade", route_after_grade, {"generate": "generate", "rewrite": "rewrite"})
        g.add_edge("rewrite", "retrieve")
        g.add_conditional_edges("generate", route_after_generate, {"rewrite": "rewrite", END: END})
    if mode == "baseline":
        g.add_edge("generate", END)
    return g.compile()


def chroma_search(vectorstore) -> Search:
    def search(query: str, k: int) -> list[Document]:
        return vectorstore.similarity_search(query, k=k)

    return search
