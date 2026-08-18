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


# A line carrying a copyright mark is an image credit, never content.
_CREDIT_LINE = re.compile(r"(?m)^.*©.*$")
# Below this many informative characters a section says nothing a student could
# use, and every one of its tokens is waste in the prompt.
MIN_INFORMATIVE_CHARS = int(os.getenv("RAG_MIN_INFORMATIVE_CHARS", "15"))


def _informative_text(text: str) -> str:
    """
    What is left of a section once the packaging is removed.

    Strips the section-path heading, the "Figure :" / "Tableau :" marker and any
    credit line, so what remains is the part a reader would actually learn from.
    """
    body = text.split("\n\n", 1)[-1] if "\n\n" in text else text
    body = re.sub(r"^\s*(figure|tableau)\s*:\s*", "", body, flags=re.IGNORECASE)
    body = _CREDIT_LINE.sub("", body)
    return re.sub(r"\s+", " ", body).strip()


def _is_empty_section(text: str) -> bool:
    """
    True for a section that costs tokens and teaches nothing.

    Measured on the indexed cardiology book, this drops 5 sections out of 141 —
    "Figure : Illustration © mgundj", "Figure : 蛸志", "Figure : 1" — and keeps
    every real one. The credit must be removed BEFORE measuring, not merely
    detected: several genuine sections (including the one holding the atheroma
    answer) end with a credit line, and rejecting on the mere presence of "©"
    threw the best content in the corpus away.

    Deliberately NOT a length test. "Figure : Classification de Stanford ou de
    De Bakey" is 71 characters and is exactly what a question about aortic
    dissection needs; "25 mm/s 1 carreau = 0,04 s" is shorter still and is the
    ECG calibration. Short is not the same as empty.
    """
    body = _informative_text(text)
    if len(body) < MIN_INFORMATIVE_CHARS:
        return True
    return sum(ch.isalpha() for ch in body) < 3


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


def _retrieve_with_vector(
    query: str,
    query_vector: List[float],
    metadata: Dict[str, Any] | None,
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    The search itself, once the query is already embedded.

    Split out from `retrieve` so `retrieve_many` can embed every question in ONE
    batched call and still run the identical search per question.
    """
    # Over-fetch when a reranker is available: it can only reorder what it is
    # given, and the right passage is sometimes outside the dense top-k. We
    # widen the CANDIDATE pool, not what the model finally receives.
    #
    # Without a reranker the pool used to equal top_k, so the merge loop stopped
    # at the first 3 parents it saw. Widening it changes nothing on its own —
    # `_rerank` still returns the first top_k in dense order — but it stops the
    # merge from being the narrowest step in the chain, and it is what the
    # reranker will need the day it is deployed.
    pool = CANDIDATE_POOL if _reranker_url() else max(top_k * 3, 10)
    hits = search(query_vector, metadata or {}, limit=max(pool, top_k) * 4)

    # Kind-aware branch: a dosing table can sit outside the dense limit because
    # tables read nothing like questions, so we fetch the asked-for kind
    # separately and let it COMPETE — merged by score, never prepended.
    #
    # Prepending was a bug with teeth. "Quelle classification est utilisée dans
    # l'AOMI ?" matches `classification`, so a table-only search was forced to
    # the front and the question came back with IDM complications and
    # conduction disorders, while "Stades de Leriche et Fontaine" — a FIGURE on
    # page 14 — never got a chance. Measured on this corpus, only 1 of the 26
    # sections carrying a trigger word is a table: forcing the kind picked the
    # right intent and the wrong 3% of the book.
    #
    # Both searches score against the same query vector, so their scores are
    # directly comparable. A table now wins when it deserves to.
    kind = _boosted_kind(query)
    if kind:
        extra = search(
            query_vector, {**(metadata or {}), "kind": kind}, limit=max(4, top_k)
        )
        seen = {h.get("parent_id") for h in hits}
        new = [h for h in extra if h.get("parent_id") not in seen]
        if new:
            hits = sorted(
                hits + new, key=lambda h: h.get("score") or 0.0, reverse=True
            )

    merged: Dict[str, Dict[str, Any]] = {}
    for hit in hits:
        parent_id = hit.get("parent_id")
        if parent_id and parent_id not in merged:
            merged[parent_id] = {
                # Carried through so a caller merging SEVERAL searches can tell
                # that two questions matched the same section (retrieve_many).
                "parent_id": parent_id,
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

    # Drop the empty sections BEFORE the final cut, not after: filtering later
    # would return fewer chunks than asked, whereas here the freed slot goes to
    # the next real section.
    candidates = [
        c for c in merged.values()
        if not _is_empty_section(c.get("context") or c.get("matched_chunk") or "")
    ]
    return _rerank(query, candidates, top_k)


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
    return _retrieve_with_vector(query, embed_query(query), metadata, top_k)


def retrieve_many(
    queries: List[str],
    metadata: Dict[str, Any] | None = None,
    top_k: int = 3,
    max_total: int = 8,
) -> List[Dict[str, Any]]:
    """
    One search PER question, merged round-robin.

    A paste of five questions used to be embedded as a single string, so the
    vector landed on the batch's centre of gravity and the same three chunks
    were handed to all five. Measured on a real failure: a question about
    atheroma plaque formation came back with IDM complications, chest pain and
    rhythm disorders — the other questions' subjects.

    The merge is ROUND-ROBIN, not concatenation: the best chunk of question 1,
    then the best of question 2, and only later the second of question 1. That
    is the property that was missing — no question can be left with nothing
    while another takes three slots.

    Embedding is batched into a single request (`embed_texts` already speaks
    `batchEmbedContents`), so N questions cost one network round-trip, not N.
    The N Qdrant searches that follow are local to the Docker network.
    """
    queries = [q for q in (queries or []) if q and q.strip()]
    if not queries:
        return []
    if len(queries) == 1:
        hits = _retrieve_with_vector(
            queries[0], embed_query(queries[0]), metadata, top_k
        )
        for h in hits:
            h["query_index"] = 0
        return hits

    vectors = embed_texts(queries, task_type="RETRIEVAL_QUERY")
    per_query = [
        _retrieve_with_vector(q, v, metadata, top_k)
        for q, v in zip(queries, vectors)
    ]
    # Which question pulled this section in. The merge below flattens every
    # question's results into one list, and without this tag a caller could no
    # longer tell — which is exactly what makes the debug console readable.
    for qi, hits in enumerate(per_query):
        for h in hits:
            h["query_index"] = qi

    out: List[Dict[str, Any]] = []
    seen: set = set()
    for rank in range(top_k):
        for hits in per_query:
            if rank >= len(hits):
                continue
            hit = hits[rank]
            # Two questions on the same subject legitimately match the same
            # section; it must not occupy two slots.
            key = hit.get("parent_id")
            if key is None or key in seen:
                continue
            seen.add(key)
            out.append(hit)
            if len(out) >= max_total:
                return out
    return out
