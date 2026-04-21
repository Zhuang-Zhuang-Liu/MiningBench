"""BM25 search tool for controlled business documents."""

import json
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence, Tuple

from mint.tools.base import Tool


DEFAULT_TOP_K = 3
DEFAULT_SNIPPET_CHARS = 400
SUPPORTED_SUFFIXES = (".md", ".txt")
ASCII_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
CJK_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+")
HEADING_PATTERN = re.compile(r"^\s*#+\s*(.+?)\s*$", re.MULTILINE)
DOC_SPLIT_PATTERN = re.compile(r"\n\s*\n+")


@dataclass(frozen=True)
class DocumentChunk:
    """A searchable paragraph-level chunk."""

    source_path: str
    title: str
    chunk_id: str
    text: str
    tokens: Tuple[str, ...]


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _workspace_doc_dir() -> str:
    return os.path.join(_repo_root(), "workspace", "work_docl")


def _tokenize(text: str) -> Tuple[str, ...]:
    tokens: List[str] = [token.lower() for token in ASCII_TOKEN_PATTERN.findall(text)]

    for block in CJK_TOKEN_PATTERN.findall(text):
        cleaned = block.strip()
        if not cleaned:
            continue
        tokens.append(cleaned)
        if len(cleaned) == 1:
            continue
        for idx in range(len(cleaned)):
            tokens.append(cleaned[idx])
        for idx in range(len(cleaned) - 1):
            tokens.append(cleaned[idx : idx + 2])

    return tuple(tokens)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _extract_title(path: str, content: str) -> str:
    heading = HEADING_PATTERN.search(content)
    if heading:
        return heading.group(1).strip()
    return os.path.splitext(os.path.basename(path))[0].replace("_", " ")


def _paragraphs(content: str) -> Iterable[str]:
    for block in DOC_SPLIT_PATTERN.split(content):
        cleaned = block.strip()
        if len(cleaned) >= 20:
            yield cleaned


def _chunk_document(path: str) -> List[DocumentChunk]:
    content = _read_text(path)
    title = _extract_title(path, content)
    paragraphs = list(_paragraphs(content))
    if not paragraphs:
        paragraphs = [content.strip()]

    chunks: List[DocumentChunk] = []
    for idx, paragraph in enumerate(paragraphs, start=1):
        tokens = _tokenize(paragraph)
        if not tokens:
            continue
        chunks.append(
            DocumentChunk(
                source_path=path,
                title=title,
                chunk_id=f"{os.path.basename(path)}#{idx}",
                text=paragraph,
                tokens=tokens,
            )
        )
    return chunks


def _collect_doc_paths(doc_dir: str) -> Tuple[str, ...]:
    paths: List[str] = []
    if not os.path.isdir(doc_dir):
        return ()

    for root, _, files in os.walk(doc_dir):
        for filename in sorted(files):
            if filename.lower().endswith(SUPPORTED_SUFFIXES):
                paths.append(os.path.join(root, filename))
    return tuple(sorted(paths))


@lru_cache(maxsize=8)
def _load_chunks_cached(doc_dir: str, file_state: Tuple[Tuple[str, float], ...]) -> Tuple[DocumentChunk, ...]:
    del file_state
    chunks: List[DocumentChunk] = []
    for path in _collect_doc_paths(doc_dir):
        chunks.extend(_chunk_document(path))
    return tuple(chunks)


def _load_chunks(doc_dir: str) -> Tuple[DocumentChunk, ...]:
    file_state = tuple((path, os.path.getmtime(path)) for path in _collect_doc_paths(doc_dir))
    return _load_chunks_cached(doc_dir, file_state)


class BM25Index:
    """A light-weight BM25 index for local paragraph chunks."""

    def __init__(self, chunks: Sequence[DocumentChunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.doc_count = len(self.chunks)
        self.avgdl = (
            sum(len(chunk.tokens) for chunk in self.chunks) / self.doc_count
            if self.doc_count
            else 0.0
        )
        self.term_doc_freq: Dict[str, int] = {}
        self.term_freqs: List[Dict[str, int]] = []
        for chunk in self.chunks:
            tf: Dict[str, int] = {}
            for token in chunk.tokens:
                tf[token] = tf.get(token, 0) + 1
            self.term_freqs.append(tf)
            for token in tf:
                self.term_doc_freq[token] = self.term_doc_freq.get(token, 0) + 1

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Tuple[float, DocumentChunk]]:
        query_terms = _tokenize(query)
        if not query_terms or not self.chunks:
            return []

        scores: List[Tuple[float, DocumentChunk]] = []
        for idx, chunk in enumerate(self.chunks):
            score = self._score(query_terms, idx, chunk)
            if score > 0:
                scores.append((score, chunk))
        scores.sort(key=lambda item: item[0], reverse=True)
        return scores[:top_k]

    def _score(self, query_terms: Sequence[str], idx: int, chunk: DocumentChunk) -> float:
        score = 0.0
        doc_len = len(chunk.tokens)
        tf = self.term_freqs[idx]
        for term in query_terms:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            doc_freq = self.term_doc_freq.get(term, 0)
            idf = math.log(1 + (self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5))
            denom = freq + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1e-6))
            score += idf * (freq * (self.k1 + 1)) / denom
        return score


def _snippet(text: str, max_chars: int = DEFAULT_SNIPPET_CHARS) -> str:
    condensed = " ".join(text.split())
    if len(condensed) <= max_chars:
        return condensed
    return condensed[: max_chars - 3] + "..."


def _format_results(results: Sequence[Tuple[float, DocumentChunk]], doc_dir: str) -> str:
    payload = {
        "results": [],
        "message": "",
        "document_directory": doc_dir,
        "hint": "",
    }

    if not results:
        payload["message"] = "No relevant business documents found."
        payload["hint"] = "Try using product names, metric names, workflow terms, or policy keywords."
        return json.dumps(payload, ensure_ascii=False, indent=2)

    payload["message"] = "Relevant business documents found."
    for rank, (score, chunk) in enumerate(results, start=1):
        rel_path = os.path.relpath(chunk.source_path, _repo_root())
        payload["results"].append(
            {
                "rank": rank,
                "title": chunk.title,
                "source": rel_path,
                "chunk": chunk.chunk_id,
                "score": round(score, 3),
                "snippet": _snippet(chunk.text),
            }
        )
    return json.dumps(payload, ensure_ascii=False, indent=2)


class BusinessDocSearch(Tool):
    """Tool that searches local business documents with BM25."""

    name = "business_doc_search"
    signature = f"{name}(query: str) -> str"
    description = (
        "- 动作：搜索业务信息（business_doc_search）"
        "\n    如果你遇到你不了解的必要信息，你可以使用这个工具来查找相关信息（包括但不限于各种计算口径、历史经验、业务信息等）"
        "\n    请使用 <execute>business_doc_search(query: str)</execute> 作为搜索工具(already imported in <execute> environment)"
        "\n    例如： <execute>business_doc_search('转化率')</execute> "
    )

    def __init__(self, doc_dir: str = None):
        self.doc_dir = doc_dir or _workspace_doc_dir()

    def __call__(self, query: str) -> str:
        chunks = _load_chunks(self.doc_dir)
        index = BM25Index(chunks)
        results = index.search(query)
        return _format_results(results, self.doc_dir)
