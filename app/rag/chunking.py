"""
Semantic (recursive) + parent-child chunking (Part 6.2).

Strategy:
  1. Recursive/semantic split into **parent** chunks along natural boundaries
     (paragraph → line → sentence), never fixed token counts.
  2. Split each parent into tiny **child** chunks (1-2 sentences). Children are
     what we embed & search (precise vector match); the parent is what we
     return to the LLM (full clinical context) — i.e. auto-merging retrieval.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import List

# Natural boundaries tried in order (recursive character splitting).
SEPARATORS = ["\n\n", "\n", ". "]
PARENT_MAX_CHARS = 1500
CHILD_SENTENCES = 2  # sentences per child chunk


@dataclass
class ParentChunk:
    text: str
    index: int
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class ChildChunk:
    text: str
    parent_id: str
    parent_text: str
    index: int
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


def _recursive_split(text: str, separators: List[str], max_chars: int) -> List[str]:
    """Split text along the first separator; recurse into over-long pieces."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars or not separators:
        return [text]

    sep, rest = separators[0], separators[1:]
    parts = text.split(sep)
    chunks: List[str] = []
    buf = ""
    for part in parts:
        candidate = f"{buf}{sep}{part}" if buf else part
        if len(candidate) <= max_chars:
            buf = candidate
            continue
        if buf:
            chunks.append(buf)
            buf = ""
        if len(part) > max_chars:
            chunks.extend(_recursive_split(part, rest, max_chars))
        else:
            buf = part
    if buf.strip():
        chunks.append(buf)
    return [c.strip() for c in chunks if c.strip()]


def _split_sentences(text: str) -> List[str]:
    """Naive sentence splitter on ., !, ? followed by whitespace."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def build_parent_chunks(text: str) -> List[ParentChunk]:
    sections = _recursive_split(text, SEPARATORS, PARENT_MAX_CHARS)
    return [ParentChunk(text=s, index=i) for i, s in enumerate(sections)]


def build_child_chunks(parent: ParentChunk) -> List[ChildChunk]:
    sentences = _split_sentences(parent.text) or [parent.text]
    children: List[ChildChunk] = []
    for i in range(0, len(sentences), CHILD_SENTENCES):
        window = " ".join(sentences[i : i + CHILD_SENTENCES]).strip()
        if window:
            children.append(
                ChildChunk(
                    text=window,
                    parent_id=parent.id,
                    parent_text=parent.text,
                    index=len(children),
                )
            )
    return children


def chunk_document(text: str) -> List[ChildChunk]:
    """Full pipeline: text → parent chunks → child chunks (with parent context)."""
    children: List[ChildChunk] = []
    for parent in build_parent_chunks(text):
        children.extend(build_child_chunks(parent))
    return children
