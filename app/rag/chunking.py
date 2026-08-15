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

# Children are sized in CHARACTERS, not in sentences. Measured on a real medical
# revision book: splitting on `.!?` left pieces of up to 2 950 characters,
# because the corpus is bullet lists with almost no sentence-ending punctuation.
# Splitting on the finest boundary and packing back up gives predictable sizes.
CHILD_TARGET_CHARS = 450
CHILD_OVERLAP_UNITS = 1  # one unit of overlap so a notion cut at a boundary
                         # stays findable from both sides


@dataclass
class ParentChunk:
    text: str
    index: int
    # ── Set by the OCR path only; the native path leaves them None so both
    #    emit the same shape and retrieval never learns which produced a chunk.
    kind: str = "prose"  # prose | table | figure | callout
    section_path: List[str] | None = None
    page_label: str | None = None  # PRINTED page number
    html: str | None = None  # tables: the faithful version
    bbox: List[float] | None = None  # normalized 0..1, for the source crop
    # Where this passage sits in the source document — a PDF page, or a .docx
    # heading. Carried into the Qdrant payload so a grounded answer can cite a
    # REAL location instead of guessing one.
    page: int | None = None
    section: str | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class ChildChunk:
    text: str
    parent_id: str
    parent_text: str
    index: int
    page: int | None = None
    section: str | None = None
    kind: str = "prose"
    section_path: List[str] | None = None
    page_label: str | None = None
    html: str | None = None
    bbox: List[float] | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


# Emitted by the extractors, stripped here: a marker must never reach an
# embedding nor the model's context.
LOC_RE = re.compile(r"<!--loc:(page|section)=(.*?)-->")


def split_by_location(text: str) -> List[tuple[int | None, str | None, str]]:
    """
    Cut the document at its location markers.

    Returns [(page, section, segment_text)]. Chunking each segment separately
    means a chunk never straddles two pages — so the page we cite is the page
    the sentence is actually on.
    """
    segments: List[tuple[int | None, str | None, str]] = []
    page: int | None = None
    section: str | None = None
    cursor = 0
    for m in LOC_RE.finditer(text):
        chunk = text[cursor : m.start()]
        if chunk.strip():
            segments.append((page, section, chunk))
        kind, value = m.group(1), m.group(2).strip()
        if kind == "page":
            page = int(value) if value.isdigit() else None
            section = None  # a new page invalidates the previous heading
        else:
            section = value or None
        cursor = m.end()
    tail = text[cursor:]
    if tail.strip():
        segments.append((page, section, tail))
    return segments or [(None, None, text)]


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


# The finest natural boundary: a line break, a bullet, a numbered item, or a
# sentence end. Measured separator yields on the same corpus — lines: 12 pieces
# per parent (median 53 chars), bullets: 3 (median 94), sentences: 3 (median
# 131). None is usable alone; together they give a fine, regular grain.
_UNIT_SPLIT = re.compile(
    r"\n+"                       # line breaks
    r"|(?<=[.!?])\s+"            # sentence ends
    r"|(?=\s*[•▪◦⇒⇨]\s)"        # bullets
    r"|(?=\s*[①-⑳]\s)"          # the circled numbers this publisher uses
)


def _split_units(text: str) -> List[str]:
    """Split into the smallest meaningful pieces, before packing them back."""
    parts = _UNIT_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p and p.strip()]


def pack_units(
    units: List[str],
    target: int = CHILD_TARGET_CHARS,
    overlap: int = CHILD_OVERLAP_UNITS,
) -> List[str]:
    """
    Group units into windows of ~`target` characters.

    A unit longer than the target goes out on its own rather than being cut
    mid-line: a half-sentence embeds to nothing useful. `overlap` repeats the
    last unit(s) at the start of the next window.
    """
    if not units:
        return []
    windows: List[str] = []
    current: List[str] = []
    size = 0
    for unit in units:
        if current and size + len(unit) > target:
            windows.append("\n".join(current))
            # Carry the tail over — but never a unit so long it would dominate
            # the next window. An oversized unit already went out on its own;
            # repeating it would just duplicate it.
            tail = current[-overlap:] if overlap else []
            if sum(len(u) for u in tail) > target // 2:
                tail = []
            current = tail
            size = sum(len(u) for u in current)
        current.append(unit)
        size += len(unit)
    if current:
        windows.append("\n".join(current))
    return windows


def build_parent_chunks(text: str) -> List[ParentChunk]:
    out: List[ParentChunk] = []
    for page, section, segment in split_by_location(text):
        for piece in _recursive_split(segment, SEPARATORS, PARENT_MAX_CHARS):
            out.append(
                ParentChunk(
                    text=piece, index=len(out), page=page, section=section
                )
            )
    return out


def build_child_chunks(parent: ParentChunk) -> List[ChildChunk]:
    windows = pack_units(_split_units(parent.text)) or [parent.text]
    children: List[ChildChunk] = []
    for window in windows:
        if window.strip():
            children.append(
                ChildChunk(
                    text=window,
                    parent_id=parent.id,
                    parent_text=parent.text,
                    index=len(children),
                    page=parent.page,
                    section=parent.section,
                    kind=parent.kind,
                    section_path=parent.section_path,
                    page_label=parent.page_label,
                    html=parent.html,
                    bbox=parent.bbox,
                )
            )
    return children


def chunk_document(text: str) -> List[ChildChunk]:
    """Full pipeline: text → parent chunks → child chunks (with parent context)."""
    children: List[ChildChunk] = []
    for parent in build_parent_chunks(text):
        children.extend(build_child_chunks(parent))
    return children
