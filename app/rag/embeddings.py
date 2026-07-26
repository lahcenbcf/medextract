"""
Gemini embeddings client (gemini-embedding-001, 768-dim) over raw REST.

Used to embed enriched child chunks at ingestion time and queries at
retrieval time. Uses the batch endpoint to embed many chunks in one call.

Note: the legacy `text-embedding-004` model was retired by Google (its
endpoint now 404s), so we use `gemini-embedding-001` and request a 768-dim
output (`outputDimensionality`) to keep the vector size stable.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List

import requests

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = 768  # requested output dimensionality (gemini-embedding-001 supports 768/1536/3072)
# Smaller batches keep each request well under the API's per-call limits — big
# documents produce many chunks, and an oversized batch is more likely to 4xx.
_BATCH = int(os.getenv("EMBED_BATCH", "50"))
# Transient statuses worth retrying (rate limit + upstream hiccups).
_RETRYABLE = {429, 500, 502, 503, 504}
_MAX_RETRIES = int(os.getenv("EMBED_MAX_RETRIES", "4"))


def _api_key() -> str:
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not configured")
    return key


def _post_batch(url: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """POST one embedding batch, retrying rate limits / transient errors with
    exponential backoff so large documents don't fail on a single 429."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = requests.post(
                url, json=body, headers={"Content-Type": "application/json"},
                timeout=120,
            )
            if resp.status_code in _RETRYABLE and attempt < _MAX_RETRIES:
                last_exc = requests.HTTPError(f"HTTP {resp.status_code}")
                time.sleep(2 ** attempt * 1.5)  # 1.5s, 3s, 6s, 12s
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                time.sleep(2 ** attempt * 1.5)
                continue
            raise
    raise last_exc or RuntimeError("embedding request failed")


def embed_texts(texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
    """Embed a list of texts, batching requests. Returns one vector per text."""
    if not texts:
        return []
    key = _api_key()
    model_path = f"models/{EMBED_MODEL}"
    url = f"{GEMINI_ENDPOINT}/{EMBED_MODEL}:batchEmbedContents?key={key}"
    vectors: List[List[float]] = []

    for start in range(0, len(texts), _BATCH):
        batch = texts[start : start + _BATCH]
        body = {
            "requests": [
                {
                    "model": model_path,
                    "content": {"parts": [{"text": t}]},
                    "taskType": task_type,
                    "outputDimensionality": EMBED_DIM,
                }
                for t in batch
            ]
        }
        data = _post_batch(url, body)
        vectors.extend(e["values"] for e in data.get("embeddings", []))

    return vectors


def embed_query(text: str) -> List[float]:
    """Embed a single query string (RETRIEVAL_QUERY task type)."""
    return embed_texts([text], task_type="RETRIEVAL_QUERY")[0]
