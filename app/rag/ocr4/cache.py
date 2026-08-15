"""
Permanent cache for raw OCR responses, on Bunny storage.

OCR is the only expensive, slow, non-deterministic step in the pipeline.
Everything after it — parsing, chunking, embedding — is a pure function over
this JSON and may be re-run freely at zero cost. That is why the cache is keyed
by CONTENT hash: the same book re-uploaded under a different filename resolves
to the same entry and is never paid for twice.

Kept on Bunny rather than on disk because the medextract container has no
volume: anything written to its filesystem dies on the next `up --build`, which
would silently re-bill every deploy.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import requests

STORAGE_API = "https://storage.bunnycdn.com"
CACHE_DIR = "kb-ocr"


def _zone() -> str:
    return os.getenv("BUNNY_ZONE_STORAGE", "nobles")


def _cdn() -> str:
    return os.getenv("CDN_HOST", "https://ziania-storage.b-cdn.net")


def _key() -> str:
    key = os.getenv("BUNNY_STORAGE_API_KEY", "")
    if not key:
        raise RuntimeError("BUNNY_STORAGE_API_KEY not configured")
    return key


def cache_uri(sha256: str) -> str:
    """The public URI of a cached response (what `ocrRawUri` records)."""
    return f"{_cdn()}/{CACHE_DIR}/{sha256}.json"


def _storage_url(sha256: str) -> str:
    return f"{STORAGE_API}/{_zone()}/{CACHE_DIR}/{sha256}.json"


def load(sha256: str) -> Optional[Dict[str, Any]]:
    """
    The cached response, or None. Never raises — a cache miss is normal.

    Read from STORAGE, not from the CDN. The CDN serves this path with
    `max-age=2592000`: after an update, the edge kept returning a 30-day-old
    copy, so freshly written figure annotations were invisible and a whole book
    stayed blind while every log said success. A cache is internal
    infrastructure — it must never be read through an edge that can hold a
    stale copy (and this also keeps a copyrighted book's OCR off a public URL).
    """
    try:
        resp = requests.get(
            _storage_url(sha256), headers={"AccessKey": _key()}, timeout=180
        )
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code != 404:
            print(f"[ocr-cache] unexpected {resp.status_code} for {sha256}")
    except Exception as exc:  # noqa: BLE001 - a miss must not break ingestion
        print(f"[ocr-cache] read failed for {sha256}: {exc}")
    return None


def store(sha256: str, raw: Dict[str, Any]) -> Optional[str]:
    """
    Persist a response. Returns its URI, or None if storage failed.

    A storage failure is logged but not raised: the OCR has already been paid
    for and the caller can still index from the in-memory copy. Losing the cache
    costs money on the NEXT run, not this one.
    """
    body = json.dumps(raw, ensure_ascii=False).encode("utf-8")
    url = _storage_url(sha256)
    try:
        resp = requests.put(
            url,
            data=body,
            headers={"AccessKey": _key(), "Content-Type": "application/json"},
            timeout=180,
        )
        if 200 <= resp.status_code < 300:
            return cache_uri(sha256)
        print(f"[ocr-cache] store failed ({resp.status_code}) for {sha256}")
    except Exception as exc:  # noqa: BLE001
        print(f"[ocr-cache] store failed for {sha256}: {exc}")
    return None
