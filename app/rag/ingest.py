"""
RAG ingestion + retrieval orchestration (Part 6.2).

Ingestion:  text → parent chunks → child chunks → contextual enrichment →
            Gemini embeddings → Qdrant upsert (with metadata payload).
Retrieval:  query → embed → metadata-filtered child search → **auto-merge** up
            to the parent sections (deduped) so the LLM gets full context.

Ingestion is designed to run in the background (see the /rag routes), so the
admin can add documents from the UI without blocking.
"""

from __future__ import annotations

import os
import re
import uuid
from typing import Any, Dict, List

import requests
from qdrant_client.models import PointStruct

from .chunking import chunk_document
from .embeddings import embed_query, embed_texts
from .enrichment import enrich
from .vector_store import delete_by_metadata, search, upsert_points


def ingest_text(
    text: str,
    metadata: Dict[str, Any],
    replace: bool = True,
) -> Dict[str, Any]:
    """
    Chunk, enrich, embed and store a document.

    `metadata` should carry at least: year, module, type (+ optional course,
    source). `replace=True` clears prior chunks for the same `source` so
    re-ingesting a document doesn't duplicate it.
    """
    children = chunk_document(text)
    if not children:
        return {"chunks": 0, "parents": 0}

    return _store(children, metadata, replace)


def ingest_ocr_response(
    raw: Dict[str, Any],
    metadata: Dict[str, Any],
    replace: bool = True,
) -> Dict[str, Any]:
    """
    Index a cached Mistral OCR 4 response — the second ingestion path.

    Takes the RAW response (the cached JSON), so re-chunking after a rule change
    costs nothing and never re-calls the paid endpoint. Everything after this
    function is shared with the native path.
    """
    from .ocr4.chunker import chunk_ocr_document
    from .ocr4.parser import parse_response

    children = chunk_ocr_document(parse_response(raw))
    if not children:
        return {"chunks": 0, "parents": 0}
    return _store(children, metadata, replace)


def _store(children, metadata: Dict[str, Any], replace: bool) -> Dict[str, Any]:
    """
    Embed and store chunks — shared by BOTH ingestion paths.

    The OCR path and the native path converge here, on one collection and one
    payload shape. Retrieval must never learn which produced a chunk; the extra
    keys below are simply absent on native chunks.
    """
    if replace and metadata.get("source"):
        delete_by_metadata({"source": metadata["source"]})

    enriched = [enrich(c, c.parent_text, metadata) for c in children]
    vectors = embed_texts(enriched)

    points: List[PointStruct] = []
    for child, vector in zip(children, vectors):
        payload: Dict[str, Any] = {
            # Original child text (what matched) + parent (what we return)
            "text": child.text,
            "parent_id": child.parent_id,
            "parent_text": child.parent_text,
            # Real location in the source document — what makes a cited page
            # verifiable instead of invented.
            **({"page": child.page} if child.page is not None else {}),
            **({"section": child.section} if child.section else {}),
            **{k: v for k, v in metadata.items() if v not in (None, "")},
        }
        # ── OCR path only (additive; native chunks simply lack these) ──
        kind = getattr(child, "kind", None)
        if kind and kind != "prose":
            payload["kind"] = kind
        for attr in ("section_path", "page_label", "html", "bbox"):
            value = getattr(child, attr, None)
            if value:
                payload[attr] = value
        points.append(
            PointStruct(id=uuid.uuid4().hex, vector=vector, payload=payload)
        )

    upsert_points(points)
    return {
        "chunks": len(points),
        "parents": len({c.parent_id for c in children}),
    }


# What a question is really asking for. A dosing question is answered by a
# table, not by the paragraph that mentions the drug — but a dense vector alone
# rarely surfaces the table, because tables read nothing like questions.
_KIND_SIGNALS = [
    (
        re.compile(
            r"\b(dose|posologie|mg\b|mg/kg|combien|µg|UI\b|crit[èe]res?|"
            r"classification|score|stades?|seuils?|valeurs?\s+normales?)",
            re.IGNORECASE,
        ),
        "table",
    ),
    (
        re.compile(
            r"\b(algorithme|conduite\s+[àa]\s+tenir|CAT\b|d[ée]marche|"
            r"quand\s+(?:faut-il|doit)|arbre\s+d[ée]cisionnel|[àa]\s+retenir)",
            re.IGNORECASE,
        ),
        "callout",
    ),
]


# How many parents to hand the reranker. Beyond this the extra latency buys
# little: the dense recall curve is already flat by 25 on this corpus.
CANDIDATE_POOL = int(os.getenv("RERANK_CANDIDATES", "25"))


def _reranker_url() -> str:
    return os.getenv("RERANKER_URL", "").rstrip("/")


def _rerank(
    query: str, candidates: List[Dict[str, Any]], top_k: int
) -> List[Dict[str, Any]]:
    """
    Reorder candidates with the cross-encoder, or keep the dense order.

    Optional by design: with no RERANKER_URL nothing is called and the behaviour
    is exactly what it was. A reranker that is down, slow or wrong degrades the
    ranking — it must never break a correction, so every failure falls back.
    """
    url = _reranker_url()
    if not url or len(candidates) <= 1:
        return candidates[:top_k]

    try:
        # The PARENT text is what the model will receive, so it is what must be
        # scored — ranking on the child excerpt would rank something else.
        documents = [(c.get("context") or c.get("matched_chunk") or "") for c in candidates]
        resp = requests.post(
            f"{url}/rerank",
            json={"query": query, "documents": documents, "top_k": top_k},
            timeout=(2, 10),
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
        if not results:
            return candidates[:top_k]

        ordered = []
        for r in results:
            i = r.get("index")
            if isinstance(i, int) and 0 <= i < len(candidates):
                hit = dict(candidates[i])
                hit["rerank_score"] = r.get("score")
                ordered.append(hit)
        return ordered or candidates[:top_k]
    except Exception as exc:  # noqa: BLE001 - never fail a correction on this
        print(f"[rag] rerank unavailable, keeping dense order: {exc}")
        return candidates[:top_k]


def _boosted_kind(query: str) -> str | None:
    """The chunk kind this question is really asking for, if any."""
    for pattern, kind in _KIND_SIGNALS:
        if pattern.search(query):
            return kind
    return None


def retrieve(
    query: str,
    metadata: Dict[str, Any] | None = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Metadata-filtered semantic search, auto-merged to parent sections.

    We over-fetch children (several may share a parent), then dedupe by
    parent_id keeping the best-scoring hit order.
    """
    query_vector = embed_query(query)
    # Over-fetch when a reranker is available: it can only reorder what it is
    # given, and the right passage is sometimes outside the dense top-k. We
    # widen the CANDIDATE pool, not what the model finally receives.
    pool = CANDIDATE_POOL if _reranker_url() else top_k
    hits = search(query_vector, metadata or {}, limit=max(pool, top_k) * 4)

    # Kind-aware branch: give the asked-for kind its own budget so a dosing
    # table can surface alongside prose instead of being outranked by it.
    kind = _boosted_kind(query)
    if kind:
        extra = search(
            query_vector, {**(metadata or {}), "kind": kind}, limit=max(4, top_k)
        )
        seen = {h.get("parent_id") for h in hits}
        hits = [h for h in extra if h.get("parent_id") not in seen] + hits

    merged: Dict[str, Dict[str, Any]] = {}
    for hit in hits:
        parent_id = hit.get("parent_id")
        if parent_id and parent_id not in merged:
            merged[parent_id] = {
                "score": hit.get("score"),
                "matched_chunk": hit.get("text"),
                "context": hit.get("parent_text"),
                "year": hit.get("year"),
                "module": hit.get("module"),
                "course": hit.get("course"),
                "type": hit.get("type"),
                "source": hit.get("source"),
                "page": hit.get("page"),
                "section": hit.get("section"),
                "file_url": hit.get("file_url"),
                # OCR path extras — absent on native chunks, which is fine:
                # the caller treats them as optional enrichment.
                "kind": hit.get("kind", "prose"),
                "section_path": hit.get("section_path"),
                "page_label": hit.get("page_label"),
                "html": hit.get("html"),
                "bbox": hit.get("bbox"),
            }
        if len(merged) >= pool:
            break

    candidates = list(merged.values())
    return _rerank(query, candidates, top_k)
