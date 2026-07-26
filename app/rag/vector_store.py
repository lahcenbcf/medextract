"""
Qdrant vector store wrapper (Part 6.2 #4 — strict metadata filtering).

We store **child** chunks as points (vector = embedding of the enriched child)
with the parent section denormalized into the payload. At query time we filter
by metadata FIRST (Qdrant applies the filter during search, so non-matching
years/modules can never be returned) and then rank by semantic similarity.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from .embeddings import EMBED_DIM

COLLECTION = os.getenv("QDRANT_COLLECTION", "nobles_kb")

# Metadata keys eligible for strict filtering.
FILTERABLE_KEYS = ("year", "module", "type", "course", "source")

_client: Optional[QdrantClient] = None


def get_client() -> QdrantClient:
    """Lazily create the Qdrant client (so imports don't require a live server)."""
    global _client
    if _client is None:
        url = os.getenv("QDRANT_URL", "http://qdrant:6333")
        api_key = os.getenv("QDRANT_API_KEY") or None
        _client = QdrantClient(url=url, api_key=api_key, timeout=60)
    return _client


def ensure_collection() -> None:
    """Create the collection on first use (idempotent)."""
    client = get_client()
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )


def build_filter(metadata: Dict[str, Any]) -> Optional[Filter]:
    """Build a strict AND filter from the provided metadata keys."""
    conditions = [
        FieldCondition(key=key, match=MatchValue(value=metadata[key]))
        for key in FILTERABLE_KEYS
        if metadata.get(key) not in (None, "")
    ]
    return Filter(must=conditions) if conditions else None


def upsert_points(points: List[PointStruct]) -> int:
    if not points:
        return 0
    ensure_collection()
    get_client().upsert(collection_name=COLLECTION, points=points)
    return len(points)


def search(
    query_vector: List[float],
    metadata: Dict[str, Any] | None = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Metadata-filtered semantic search over child chunks."""
    ensure_collection()
    hits = get_client().search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        query_filter=build_filter(metadata or {}),
        limit=limit,
        with_payload=True,
    )
    return [{"score": h.score, **(h.payload or {})} for h in hits]


def delete_by_metadata(metadata: Dict[str, Any]) -> None:
    """Remove all chunks matching the given metadata (e.g. re-ingesting a doc)."""
    flt = build_filter(metadata)
    if flt is None:
        return
    ensure_collection()
    get_client().delete(collection_name=COLLECTION, points_selector=flt)
