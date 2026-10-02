"""Graph routing tests with a scripted fake LLM - no network, no API key."""

from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.graph import NOT_FOUND, Relevant, Rewrite, build_graph


class ScriptedLLM(FakeListChatModel):
    """Returns queued structured outputs for with_structured_output() and text for plain calls."""

    structured: list[Any] = []

    def with_structured_output(self, schema, **kwargs):
        from langchain_core.runnables import RunnableLambda

        def pop(_):
            item = self.structured.pop(0)
            assert isinstance(item, schema), f"expected {schema.__name__}, got {type(item).__name__}"
            return item

        return RunnableLambda(pop)


def doc(title: str) -> Document:
    return Document(page_content=f"{title}\n\nbody", metadata={"title": title, "url": f"https://x/#{title}"})


def test_relevant_docs_go_straight_to_generate():
    llm = ScriptedLLM(responses=["Set services.openssh.enable = true; [1]"], structured=[Relevant(relevant=[2])])
    calls = []

    def search(q, k):
        calls.append(q)
        return [doc("IPv4"), doc("Secure Shell Access")]

    out = build_graph(llm, search).invoke({"question": "enable ssh?"})
    assert calls == ["enable ssh?"]
    assert out["answer"].startswith("Set services.openssh.enable")
    assert [s["title"] for s in out["sources"]] == ["Secure Shell Access"]  # irrelevant doc dropped


def test_irrelevant_results_trigger_one_rewrite_then_answer():
    llm = ScriptedLLM(
        responses=["Use nixos-rebuild switch --rollback [1]"],
        structured=[Relevant(relevant=[]), Rewrite(query="nixos-rebuild --rollback"), Relevant(relevant=[1])],
    )
    calls = []

    def search(q, k):
        calls.append(q)
        return [doc("Rolling Back Configuration Changes")]

    out = build_graph(llm, search).invoke({"question": "undo my last upgrade"})
    assert calls == ["undo my last upgrade", "nixos-rebuild --rollback"]
    assert out["steps"][-1] == "generate"
    assert "rollback" in out["answer"]


def test_gives_up_after_max_rewrites():
    llm = ScriptedLLM(
        responses=["unused"], structured=[Relevant(relevant=[]), Rewrite(query="q2"), Relevant(relevant=[])]
    )
    out = build_graph(llm, lambda q, k: [doc("Unrelated")], max_rewrites=1).invoke(
        {"question": "who won the 1998 world cup"}
    )
    assert out["answer"] == NOT_FOUND
    assert out["sources"] == []


def test_baseline_mode_skips_grading():
    llm = ScriptedLLM(responses=["answer [1]"], structured=[])
    out = build_graph(llm, lambda q, k: [doc("A")], mode="baseline").invoke({"question": "q?"})
    assert out["steps"] == ["retrieve: q?", "generate"]
