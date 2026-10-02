"""Fetch the NixOS manual, split it by section, embed it and persist a Chroma index.

Run:  python -m app.ingest
"""

import logging
import re
import shutil
import time

import httpx
from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings, get_settings
from app.embeddings import FastEmbedEmbeddings

log = logging.getLogger("ingest")

_MARK = "\x00SEC\x00"
_HEADINGS = ["h1", "h2", "h3", "h4"]
# DocBook output reuses h2 for chapters and their sections, so depth comes from the wrapping divs.
_BLOCKS = ["p", "pre", "li", "dt", "dd", "div", "tr", "br", "h1", "h2", "h3", "h4", "h5", "h6"]
_CONTAINERS = {"part", "chapter", "appendix", "preface", "section"}


def _depth(tag) -> int:
    return sum(1 for p in tag.parents if p.name == "div" and _CONTAINERS & set(p.get("class") or []))


def parse_sections(html: str, base_url: str) -> list[Document]:
    """Turn the single-page manual into one Document per titled section.

    Every heading that carries an anchor id starts a new section, so each chunk
    can cite a deep link back into the manual.
    """
    soup = BeautifulSoup(html, "html.parser")
    for toc in soup.select("div.toc"):
        toc.decompose()

    version = ""
    subtitle = soup.find("h2", class_="subtitle")
    if subtitle:
        version = subtitle.get_text(strip=True).removeprefix("Version").strip()

    for h in soup.find_all(_HEADINGS):
        anchor = h.get("id") or (h.find("a", id=True) or {}).get("id")
        title = " ".join(h.get_text(" ", strip=True).split())
        if not anchor or not title or anchor.startswith("book-"):
            continue  # admonition titles like "Note" have no id; keep them inline
        h.replace_with(f"\n{_MARK}{_depth(h)}|{anchor}{_MARK}{title}{_MARK}\n")

    # Break lines only at block elements so inline <code>/<a> stay in their sentence.
    for block in soup.find_all(_BLOCKS):
        block.insert_after("\n")
    text = (soup.body or soup).get_text()
    parts = text.split(_MARK)
    docs: list[Document] = []
    trail: dict[int, str] = {}  # heading level -> title, for "Chapter › Section" breadcrumbs
    # parts = [preamble, "level|anchor", title, body, "level|anchor", title, body, ...]
    for i in range(1, len(parts) - 2, 3):
        level_anchor, title, body = parts[i], parts[i + 1], parts[i + 2]
        level_s, anchor = level_anchor.split("|", 1)
        level = int(level_s)
        trail = {lvl: t for lvl, t in trail.items() if lvl < level}
        trail[level] = title
        crumb = " › ".join(t for _, t in sorted(trail.items()))
        body = _clean(body)
        if len(body) < 80:
            continue  # heading-only sections (parts, chapter wrappers)
        docs.append(
            Document(
                page_content=f"{crumb}\n\n{body}",
                metadata={
                    "title": crumb,
                    "anchor": anchor,
                    "url": f"{base_url}#{anchor}",
                    "version": version,
                },
            )
        )
    return docs


def _clean(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def build_index(settings: Settings | None = None) -> int:
    s = settings or get_settings()
    t0 = time.perf_counter()
    log.info("fetching %s", s.manual_url)
    resp = httpx.get(s.manual_url, follow_redirects=True, timeout=60)
    resp.raise_for_status()

    sections = parse_sections(resp.text, str(resp.url))
    splitter = RecursiveCharacterTextSplitter(chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap)
    chunks = splitter.split_documents(sections)
    for c in chunks:
        # Keep the section title on every chunk so retrieval doesn't lose context mid-section.
        if not c.page_content.startswith(c.metadata["title"]):
            c.page_content = f"{c.metadata['title']}\n\n{c.page_content}"
    log.info("parsed %d sections -> %d chunks", len(sections), len(chunks))

    if s.chroma_dir.exists():
        shutil.rmtree(s.chroma_dir)
    Chroma.from_documents(
        chunks,
        FastEmbedEmbeddings(s.embedding_model),
        collection_name=s.collection,
        persist_directory=str(s.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )
    log.info("indexed %d chunks into %s in %.1fs", len(chunks), s.chroma_dir, time.perf_counter() - t0)
    return len(chunks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    build_index()
