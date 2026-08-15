"""
Structure-aware chunking for the OCR path.

The native path splits on characters and separators. This one splits on what
the document actually is: sections, tables, figures and boxed clinical content.
Both emit the same `ChildChunk`, so retrieval never learns which produced a
chunk — that is the whole integration contract.

Four rules carry most of the value:

  * **Section boundaries are hard cuts.** A chunk spanning the end of
    "Diagnostic" and the start of "Traitement" answers neither question well.
  * **Tables are atomic.** A header row separated from its data rows yields two
    chunks and neither is answerable. Dosing tables and classifications are the
    highest-value targets in a clinical corpus.
  * **Callouts are their own kind.** This corpus concentrates its best content
    in boxes ("Clinical pearls", "Attention", numbered key points); they are
    already written as self-contained answers.
  * **Every chunk is prefixed with its section path.** Standalone, "titrer
    jusqu'à la dose maximale tolérée" could be about any condition in medicine.
"""

from __future__ import annotations

import re
from typing import List

from ..chunking import ChildChunk, ParentChunk, build_child_chunks

# `<table><tr><td rowspan="3">` means nothing to an embedding model. Measured on
# a real book: 21% of a table chunk's embedded text was markup.
_TAG = re.compile(r"<[^>]+>")


def _strip_markup(html_text: str) -> str:
    """Cell text only, cells separated so they do not run together."""
    text = _TAG.sub(" ", html_text)
    return re.sub(r"\s+", " ", text).strip()
from .parser import HEADING, PLACEHOLDER, Page, Region

# Sized in characters, not tokens: there is no local tokenizer for the Gemini
# embedder. ~4 chars/token on French medical prose, so this targets ~400 tokens
# — comfortably inside the model's window with room for the section prefix.
TARGET_CHARS = 1600
MAX_CHARS = 2400

# This book's boxed content. `①-⑳` matter: the publisher numbers its key points
# with circled digits, and those blocks are exactly the ones worth isolating.
CALLOUT_CUES = re.compile(
    r"^\s*(?:[①-⑳]|(?:points?\s+cl[ée]s?|[àa]\s+retenir|attention|piège|remarque|"
    r"NB|rappel|en\s+pratique|conduite\s+[àa]\s+tenir|CAT)\b)",
    re.IGNORECASE,
)


def _page_number(page: Page) -> int:
    """
    The page number as a PDF viewer shows it.

    `Page.index` keeps whatever the OCR API returns (0-based) because that is
    its raw contract; the shift happens HERE, once, on the way into a chunk.
    Do not "correct" this back: the native path already emits `page_num + 1`
    (pdf_extractor), and a viewer told to open page 5 for what it calls page 6
    lands one page early — which it silently did before this.
    """
    return page.index + 1


def _prefix(section_path: List[str]) -> str:
    """The heading trail, prepended to every chunk (Rule 6)."""
    return " > ".join(p for p in section_path if p)


# A line shorter than this matches too many regions to be evidence of anything.
_BBOX_MIN_LINE = 15


def _lines_bbox(lines: List[str], page: Page) -> List[float] | None:
    """
    Where on the page these lines came from, as one normalized box.

    The markdown spine carries no coordinates, so a prose chunk would otherwise
    have nothing to show the admin. Tying each line back to the region that
    contains it recovers that: measured on a real book, 663 of 667 lines match,
    and the union covers 10% of the page against 7% for the regions themselves —
    tight enough that one rectangle is honest, and far simpler than shipping a
    list of boxes through the whole chain.
    """
    boxes = []
    for line in lines:
        line = line.strip()
        if len(line) < _BBOX_MIN_LINE:
            continue
        for r in page.regions:
            if r.type not in ("text", "list", "title"):
                continue
            text = (r.text or "").strip()
            if text and (line in text or text in line):
                boxes.append((r.x0, r.y0, r.x1, r.y1))
                break
    if not boxes:
        return None
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def _flush(
    buf: List[str],
    page: Page,
    section_path: List[str],
    out: List[ParentChunk],
    kind: str = "prose",
) -> None:
    body = "\n".join(buf).strip()
    lines = list(buf)
    buf.clear()
    if len(body) < 20:  # a stray bullet is not a chunk
        return
    head = _prefix(section_path)
    out.append(
        ParentChunk(
            text=f"{head}\n\n{body}" if head else body,
            index=len(out),
            kind=kind,
            section_path=list(section_path),
            # 1-based, like every PDF viewer — see _page_number().
            page=_page_number(page),
            page_label=page.label,
            bbox=_lines_bbox(lines, page),
        )
    )


def _region_chunk(
    region: Region,
    page: Page,
    section_path: List[str],
    caption: str,
    out: List[ParentChunk],
) -> None:
    """A table or figure: never split, always self-describing."""
    head = _prefix(section_path)
    if region.type == "table":
        # The caption leads because queries match captions far more often than
        # cell values; the faithful HTML rides in the payload for rendering.
        body = f"Tableau : {caption}".strip() if caption else "Tableau"
        text = f"{head}\n\n{body}\n\n{region.html or ''}".strip()
        kind = "table"
    else:
        # A figure is retrievable through its caption AND through the text the
        # annotation pass read inside it. On this corpus that transcription is
        # frequently the ONLY copy of the content: entire boxed sections are
        # classified as images by the OCR.
        label = caption or region.caption_hint or ""
        if not region.transcribed and not label:
            # Nothing to match on. Measured on a real book, 41 of 148 chunks
            # were "(SECTION) Figure" placeholders — a quarter of the index
            # competing for the top-k while answering nothing. The region stays
            # in the parser (bbox + image kept for display); it just never
            # becomes a vector.
            return
        body = f"Figure : {label}".strip() if label else "Figure"
        parts = [head, body]
        if region.transcribed:
            parts.append(region.transcribed)
        text = "\n\n".join(p for p in parts if p).strip()
        # A transcribed box is prose, not a picture: typing it "figure" would
        # exclude it from the default retrieval kinds.
        kind = "callout" if region.transcribed else "figure"
    # A figure/table is atomic and carries payload value (bbox, HTML, image), so
    # it is NOT held to the prose length floor. A figure with no caption and no
    # heading is still poorly retrievable — that is what the figure-annotation
    # pass is for — but losing it entirely is worse.
    out.append(
        ParentChunk(
            text=text,
            index=len(out),
            kind=kind,
            section_path=list(section_path),
            page=_page_number(page),
            page_label=page.label,
            html=region.html,
            bbox=[region.x0, region.y0, region.x1, region.y1],
        )
    )


def chunk_pages(pages: List[Page]) -> List[ParentChunk]:
    """Walk the markdown spine, cutting on structure."""
    out: List[ParentChunk] = []
    section_path: List[str] = []
    buf: List[str] = []
    # What the buffer currently is: a callout cue turns the run that follows it
    # into a callout chunk rather than ordinary prose.
    pending_kind = "prose"

    for page in pages:
        for line in page.markdown.splitlines():
            stripped = line.strip()

            heading = HEADING.match(stripped)
            if heading:
                # A title flushes what came before, then re-roots the trail at
                # its own level.
                _flush(buf, page, section_path, out, kind=pending_kind)
                pending_kind = "prose"
                level = len(heading.group(1))
                title = heading.group(2).strip()
                del section_path[level - 1 :]
                section_path.append(title)
                continue

            placeholder = PLACEHOLDER.fullmatch(stripped)
            if placeholder:
                region = page.region(placeholder.group(2))
                if region is not None:
                    # The line before the figure is its caption far more often
                    # than the line after, in this corpus.
                    caption = buf[-1].strip() if buf else ""
                    _flush(buf, page, section_path, out, kind=pending_kind)
                    pending_kind = "prose"
                    _region_chunk(region, page, section_path, caption, out)
                continue

            if not stripped:
                continue

            # A callout starts its own chunk and never merges into prose.
            if CALLOUT_CUES.match(stripped):
                if buf:
                    _flush(buf, page, section_path, out, kind=pending_kind)
                pending_kind = "callout"

            buf.append(stripped)
            if sum(len(x) for x in buf) >= TARGET_CHARS:
                _flush(buf, page, section_path, out, kind=pending_kind)
                pending_kind = "prose"

        # Never let prose run across a page boundary silently: flush per page so
        # `page`/`page_label` on the chunk stay truthful for citation.
        _flush(buf, page, section_path, out, kind=pending_kind)
        pending_kind = "prose"

    return out


# Below this, a prose chunk gives the model too little to work with. Measured
# on a real book: median parent was 256 characters, 39 of 148 under 100.
MIN_PROSE_CHARS = 400


def consolidate(parents: List[ParentChunk]) -> List[ParentChunk]:
    """
    Merge consecutive prose fragments that belong together.

    Cutting on every heading and every page boundary keeps chunks honest but
    leaves them thin. Two prose parents are merged only when they share the SAME
    page and the SAME section path — so a merge can never blur two subjects, and
    `page`/`page_label` stay truthful for citation.

    Tables, figures and callouts are never merged: their atomicity is exactly
    what makes them answerable.
    """
    out: List[ParentChunk] = []
    for parent in parents:
        prev = out[-1] if out else None
        mergeable = (
            prev is not None
            and prev.kind == "prose"
            and parent.kind == "prose"
            and prev.page == parent.page
            and (prev.section_path or []) == (parent.section_path or [])
            and len(prev.text) < MIN_PROSE_CHARS
            and len(prev.text) + len(parent.text) <= TARGET_CHARS
        )
        if mergeable:
            # Drop the repeated section prefix from the tail before joining.
            head = _prefix(parent.section_path or [])
            body = parent.text
            if head and body.startswith(head):
                body = body[len(head) :].lstrip("\n")
            prev.text = f"{prev.text}\n\n{body}".strip()
            continue
        out.append(parent)
    # Re-number so `index` stays contiguous after merging.
    for i, parent in enumerate(out):
        parent.index = i
    return out


def chunk_ocr_document(pages: List[Page]) -> List[ChildChunk]:
    """
    OCR pages → child chunks, the same shape the native path produces.

    Children are what gets embedded (precise match); the parent is what comes
    back to the model (full context) — the auto-merging retrieval the existing
    pipeline already relies on.
    """
    children: List[ChildChunk] = []
    for parent in consolidate(chunk_pages(pages)):
        if parent.kind in ("table", "figure"):
            # Atomic: splitting a table into sentences would embed its header
            # row apart from its data. The EMBEDDED text drops the markup; the
            # parent and the `html` payload keep it intact for display and for
            # what the model finally reads.
            embed_text = (
                _strip_markup(parent.text) if parent.kind == "table" else parent.text
            )
            children.append(
                ChildChunk(
                    text=embed_text[:2000],
                    parent_id=parent.id,
                    parent_text=parent.text,
                    index=0,
                    page=parent.page,
                    section=parent.section,
                    kind=parent.kind,
                    section_path=parent.section_path,
                    page_label=parent.page_label,
                    html=parent.html,
                    bbox=parent.bbox,
                )
            )
        else:
            children.extend(build_child_chunks(parent))
    return children
