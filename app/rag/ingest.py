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

import uuid
from typing import Any, Dict, List

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

    if replace and metadata.get("source"):
        delete_by_metadata({"source": metadata["source"]})

    enriched = [enrich(c.text, c.parent_text, metadata) for c in children]
    vectors = embed_texts(enriched)

    points: List[PointStruct] = []
    for child, vector in zip(children, vectors):
        points.append(
            PointStruct(
                id=uuid.uuid4().hex,
                vector=vector,
                payload={
                    # Original child text (what matched) + parent (what we return)
                    "text": child.text,
                    "parent_id": child.parent_id,
                    "parent_text": child.parent_text,
                    **{k: v for k, v in metadata.items() if v not in (None, "")},
                },
            )
        )

    upsert_points(points)
    return {
        "chunks": len(points),
        "parents": len({c.parent_id for c in children}),
    }


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
    hits = search(query_vector, metadata or {}, limit=top_k * 4)

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
            }
        if len(merged) >= top_k:
            break
    return list(merged.values())
