"""
Gemini embeddings client (text-embedding-004, 768-dim) over raw REST.

Used to embed enriched child chunks at ingestion time and queries at
retrieval time. Uses the batch endpoint to embed many chunks in one call.
"""

from __future__ import annotations

import os
from typing import List

import requests

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")
EMBED_DIM = 768  # text-embedding-004 output dimensionality
_BATCH = 100


def _api_key() -> str:
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not configured")
    return key


def embed_texts(texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
    """Embed a list of texts, batching requests. Returns one vector per text."""
    if not texts:
        return []
    key = _api_key()
    model_path = f"models/{EMBED_MODEL}"
    vectors: List[List[float]] = []

    for start in range(0, len(texts), _BATCH):
        batch = texts[start : start + _BATCH]
        body = {
            "requests": [
                {
                    "model": model_path,
                    "content": {"parts": [{"text": t}]},
                    "taskType": task_type,
                }
                for t in batch
            ]
        }
        resp = requests.post(
            f"{GEMINI_ENDPOINT}/{EMBED_MODEL}:batchEmbedContents?key={key}",
            json=body,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        vectors.extend(e["values"] for e in data.get("embeddings", []))

    return vectors


def embed_query(text: str) -> List[float]:
    """Embed a single query string (RETRIEVAL_QUERY task type)."""
    return embed_texts([text], task_type="RETRIEVAL_QUERY")[0]
