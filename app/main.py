"""FastAPI service: POST /api/ask, GET /api/health, and a small web UI at /."""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import get_settings
from app.graph import build_graph, chroma_search
from app.llm import get_vectorstore, make_chat_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("api")
STATIC = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class Source(BaseModel):
    n: int
    title: str | None
    url: str | None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    steps: list[str]
    latency_ms: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    app.state.graph = build_graph(
        make_chat_model(), chroma_search(get_vectorstore()), top_k=s.top_k, max_rewrites=s.max_rewrites
    )
    log.info("ready provider=%s model=%s", s.llm_provider, s.llm_model)
    yield


app = FastAPI(title="NixOS Docs Copilot", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "chunks": get_vectorstore()._collection.count()}


@app.post("/api/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request):
    rid = uuid.uuid4().hex[:8]
    t0 = time.perf_counter()
    try:
        result = await request.app.state.graph.ainvoke({"question": body.question})
    except Exception as exc:  # surface provider errors (rate limits, bad key) without leaking internals
        log.exception("ask failed rid=%s", rid)
        raise HTTPException(status_code=502, detail=f"LLM call failed (request {rid})") from exc
    ms = int((time.perf_counter() - t0) * 1000)
    log.info("ask rid=%s ms=%d steps=%s", rid, ms, result["steps"])
    return AskResponse(answer=result["answer"], sources=result["sources"], steps=result["steps"], latency_ms=ms)
