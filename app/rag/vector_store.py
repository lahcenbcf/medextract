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
    MatchAny,
    PointStruct,
    VectorParams,
)

from .embeddings import EMBED_DIM

COLLECTION = os.getenv("QDRANT_COLLECTION", "nobles_kb")

# Metadata keys eligible for strict filtering.
FILTERABLE_KEYS = ("year", "module", "type", "course", "source", "kind")

_client: Optional[QdrantClient] = None


def get_client() -> QdrantClient:
    """Lazily create the Qdrant client (so imports don't require a live server)."""
    global _client
    if _client is None:
        url = os.getenv("QDRANT_URL", "http://qdrant:6333")
        api_key = os.getenv("QDRANT_API_KEY") or None
        _client = QdrantClient(url=url, api_key=api_key, timeout=60)
    return _client


# Fields worth an index. Without them every filtered query degenerates into a
# full scan — the difference between 20 ms and 2 s on a book-scale corpus.
_INDEXED_FIELDS = {
    "source": "keyword",
    "module": "keyword",
    "year": "keyword",
    "course": "keyword",
    "type": "keyword",
    "kind": "keyword",
    "page": "integer",
}


def ensure_collection() -> None:
    """Create the collection on first use (idempotent)."""
    client = get_client()
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
    _ensure_payload_indexes(client)


def _ensure_payload_indexes(client) -> None:
    """
    Create the payload indexes, tolerating the ones that already exist.

    Idempotent by design: this runs on every ingestion, and an existing index
    must not be an error. Failures are logged, never raised — a missing index
    makes queries slower, not wrong.
    """
    for field, schema in _INDEXED_FIELDS.items():
        try:
            client.create_payload_index(
                collection_name=COLLECTION, field_name=field, field_schema=schema
            )
        except Exception as exc:  # noqa: BLE001 - already-exists is the common case
            msg = str(exc).lower()
            if "already exists" not in msg and "conflict" not in msg:
                print(f"[vector_store] payload index on {field!r} skipped: {exc}")


def build_filter(metadata: Dict[str, Any]) -> Optional[Filter]:
    """Build a strict AND filter from the provided metadata keys (case-insensitive fallback)."""
    conditions = []
    for key in FILTERABLE_KEYS:
        val = metadata.get(key)
        if val not in (None, ""):
            if isinstance(val, str):
                # Search for exactly what was passed, plus common case variations
                # because Qdrant keyword matching is strictly case-sensitive.
                variants = list({val, val.upper(), val.lower(), val.capitalize()})
                conditions.append(FieldCondition(key=key, match=MatchAny(any=variants)))
            else:
                conditions.append(FieldCondition(key=key, match=MatchValue(value=val)))
                
    return Filter(must=conditions) if conditions else None


# Qdrant refuses a body over `max_request_size_mb` — 32 MB by default — and it
# answers with a JSON error, not a transport failure, so the client surfaces it
# as a plain 400 and a whole book's indexing dies at the last step. Measured on
# a real one: 50 224 506 bytes in a single upsert.
#
# Half the ceiling, because the estimate below is an estimate.
QDRANT_MAX_BATCH_BYTES = int(
    os.getenv("QDRANT_MAX_BATCH_BYTES", str(16 * 1024 * 1024))
)
# A second ceiling on count: a thousand tiny chunks make a slow request long
# before they make a large one.
QDRANT_MAX_BATCH_POINTS = int(os.getenv("QDRANT_MAX_BATCH_POINTS", "256"))
# One 768-float vector costs roughly this much once serialised — and on a book
# of short chunks the VECTORS, not the text, are what fills the request.
_BYTES_PER_DIMENSION = 20


def _point_bytes(point: PointStruct) -> int:
    """Rough serialised size of one point. Cheap on purpose: it runs per point."""
    payload = point.payload or {}
    text = sum(len(str(v)) for v in payload.values())
    vector = point.vector
    dims = len(vector) if isinstance(vector, (list, tuple)) else 0
    return text + dims * _BYTES_PER_DIMENSION


def upsert_points(points: List[PointStruct]) -> int:
    """Write points to Qdrant in batches small enough to be accepted."""
    if not points:
        return 0
    ensure_collection()
    client = get_client()

    batch: List[PointStruct] = []
    size = 0
    sent = 0

    def flush() -> None:
        nonlocal batch, size, sent
        if not batch:
            return
        client.upsert(collection_name=COLLECTION, points=batch)
        sent += len(batch)
        batch, size = [], 0

    for point in points:
        n = _point_bytes(point)
        if n > QDRANT_MAX_BATCH_BYTES:
            # Cannot happen with the current chunker (2 400 characters a chunk),
            # so if it ever does, say which point rather than letting Qdrant
            # answer with a byte count nobody can trace back to a document.
            page = (point.payload or {}).get("page")
            src = (point.payload or {}).get("source")
            print(
                f"[vector-store] WARNING: single point of {n / 1048576:.1f} MB "
                f"exceeds the batch ceiling (source={src!r} page={page}) — "
                "sending it alone; Qdrant may still refuse it",
                flush=True,
            )
        # A single point over the ceiling still has to go alone: nothing here
        # can split it, but it must not drag its neighbours past the limit.
        if batch and (
            size + n > QDRANT_MAX_BATCH_BYTES
            or len(batch) >= QDRANT_MAX_BATCH_POINTS
        ):
            flush()
        batch.append(point)
        size += n
    flush()
    return sent


def search(
    query_vector: List[float],
    metadata: Dict[str, Any] | None = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Metadata-filtered semantic search over child chunks."""
    ensure_collection()
    resp = get_client().query_points(
        collection_name=COLLECTION,
        query=query_vector,
        query_filter=build_filter(metadata or {}),
        limit=limit,
        with_payload=True,
    )
    return [{"score": h.score, **(h.payload or {})} for h in resp.points]


def file_urls_for_source(source: str) -> List[str]:
    """
    The stored original(s) behind one document, read from the chunk payloads.

    The uploaded filename and the `source` tag can differ (the admin may rename
    the source at import), so the URL cannot be rebuilt from the name — it has
    to be read back from what was indexed. Called just before deleting the
    chunks, while they still exist.
    """
    client = get_client()
    if not client.collection_exists(COLLECTION):
        return []

    urls: List[str] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=build_filter({"source": source}),
            with_payload=True,
            with_vectors=False,
            limit=256,
            offset=offset,
        )
        for p in points:
            url = (p.payload or {}).get("file_url")
            if url and url not in urls:
                urls.append(url)
        if offset is None:
            break
    return urls


def delete_by_metadata(metadata: Dict[str, Any]) -> None:
    """Remove all chunks matching the given metadata (e.g. re-ingesting a doc)."""
    flt = build_filter(metadata)
    if flt is None:
        return
    ensure_collection()
    get_client().delete(collection_name=COLLECTION, points_selector=flt)


def list_sources() -> List[Dict[str, Any]]:
    """
    List indexed documents grouped by their `source` (the uploaded filename),
    with a chunk count and the tags they carry. Scrolls the whole collection —
    fine for the modest Knowledge Base sizes we deal with.
    """
    client = get_client()
    if not client.collection_exists(COLLECTION):
        return []

    grouped: Dict[str, Dict[str, Any]] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION,
            with_payload=True,
            with_vectors=False,
            limit=256,
            offset=offset,
        )
        for p in points:
            payload = p.payload or {}
            source = payload.get("source") or "(sans nom)"
            entry = grouped.get(source)
            if entry is None:
                entry = {
                    "source": source,
                    "chunks": 0,
                    "module": payload.get("module"),
                    "year": payload.get("year"),
                    "type": payload.get("type"),
                    "course": payload.get("course"),
                }
                grouped[source] = entry
            entry["chunks"] += 1
        if offset is None:
            break

    return sorted(grouped.values(), key=lambda e: e["source"].lower())
