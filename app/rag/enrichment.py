"""
Contextual enrichment (Part 6.2 #3).

Before embedding, every child chunk is prefixed with a context string so an
isolated sentence keeps its overarching medical subject. E.g.:

    [Module: Cardiology | Course: Antihypertensives] The primary side effect
    is severe hypotension.

The metadata prefix is always applied (deterministic and cheap). Optionally a
small LLM adds a one-line situating sentence derived from the parent section —
enable with RAG_LLM_ENRICH=1. The enriched text is what gets embedded; the
original child text is what we store and return to the LLM.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import requests

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"

# Metadata keys surfaced in the prefix, in display order.
_PREFIX_KEYS = [
    ("module", "Module"),
    ("course", "Course"),
    ("year", "Year"),
    ("type", "Type"),
]


def build_metadata_prefix(metadata: Dict[str, object]) -> str:
    parts = [
        f"{label}: {metadata[key]}"
        for key, label in _PREFIX_KEYS
        if metadata.get(key)
    ]
    return f"[{' | '.join(parts)}]" if parts else ""


def build_context_prefix(child, metadata: Dict[str, object]) -> str:
    """
    The line that situates a chunk INSIDE its document.

    The metadata prefix alone is identical on every chunk of a book, so it
    separates documents — which the metadata filter already does — and tells
    nothing apart within one. The section path does: measured on a real book,
    32% of children were embedded with no trace of their section, so an excerpt
    like "Le choc électrique externe (CEE)" carried nothing linking it to
    "FIBRILLATION AURICULAIRE" and no question about AF could reach it.

    The section path comes FIRST so it weighs most in the vector.
    """
    bits = []
    path = " > ".join(p for p in (getattr(child, "section_path", None) or []) if p)
    # The parent text already opens with the section path, so the FIRST window
    # of a parent inherits it. Repeating it would double those tokens in the
    # vector for no gain.
    text = getattr(child, "text", "") or ""
    if path and not text.lstrip().startswith(path[:40]):
        bits.append(path)
    label = getattr(child, "page_label", None)
    if label:
        bits.append(f"p. {label}")
    head = " — ".join(bits)
    meta = build_metadata_prefix(metadata)
    return f"{head} {meta}".strip() if head else meta


def _llm_context(child_text: str, parent_text: str) -> Optional[str]:
    """Ask a small Gemini model for a one-line situating context (best-effort)."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return None
    model = os.getenv("GEMINI_ENRICH_MODEL", "gemini-2.0-flash")
    prompt = (
        "In one short sentence, situate the following excerpt within its "
        "section so it is understandable on its own. Reply with the sentence "
        f"only.\n\nSECTION:\n{parent_text[:1500]}\n\nEXCERPT:\n{child_text}"
    )
    try:
        resp = requests.post(
            f"{GEMINI_ENDPOINT}/{model}:generateContent?key={api_key}",
            json={
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.0, "maxOutputTokens": 60},
            },
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        parts = (
            resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
        )
        text = "".join(p.get("text", "") for p in parts).strip()
        return text or None
    except Exception:
        return None


def enrich(child, parent_text: str, metadata: Dict[str, object]) -> str:
    """
    Return the text to EMBED: context prefix (+ optional LLM ctx) + chunk.

    `child` is the ChildChunk — the prefix needs its section path and page, not
    just its text. What is stored and returned to the model stays `child.text`,
    untouched: the prefix exists only to steer the vector.
    """
    child_text = getattr(child, "text", child)
    prefix = build_context_prefix(child, metadata)
    pieces = [prefix] if prefix else []

    if os.getenv("RAG_LLM_ENRICH", "0") == "1":
        ctx = _llm_context(child_text, parent_text)
        if ctx:
            pieces.append(ctx)

    pieces.append(child_text)
    return " ".join(pieces).strip()
