"""
Parse a Mistral OCR 4 response into a semantic, chunkable document.

**The page `markdown` is the text spine, not the blocks.** OCR 4 has already
resolved reading order, heading levels and list structure into that markdown;
rebuilding the text from `blocks[]` would undo that work and hand the chunker a
bag of spatial fragments. The blocks are kept as an *overlay* — they carry the
region types and bounding boxes we need to show a source crop beside an answer.

Markdown references figures and tables by placeholder, which is what makes the
two layers line up cleanly:

    ![img-0.jpeg](img-0.jpeg)      → an `image` block, with its bbox
    [tbl-0.html](tbl-0.html)       → a `table`, whose faithful HTML is in
                                     `page.tables[]` (markdown pipe tables
                                     cannot express rowspan, so we never use
                                     them for tables)

Every field name here was verified against `fixtures/ocr4_medical.json`, taken
from a real scanned cardiology book. Several differ from the documentation:
there is no `bbox` array (four PIXEL fields instead), the block text is in
`content`, and `block.confidence_scores` is always null.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# `![img-0.jpeg](img-0.jpeg)` / `[tbl-0.html](tbl-0.html)`
PLACEHOLDER = re.compile(r"!?\[([^\]]+)\]\(([^)]+)\)")
# "# Title" / "## Sub"
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Region:
    """A typed region of the page — the geometric overlay on the markdown."""

    id: Optional[str]  # img-0.jpeg / tbl-0.html, when the markdown refers to it
    type: str  # text | title | list | table | image | footer | header
    text: str
    html: Optional[str] = None
    image_base64: Optional[str] = None
    # Text transcribed FROM the picture by the annotation pass. On this corpus
    # whole boxed sections are classified as images, so this is often the only
    # place their content exists.
    transcribed: Optional[str] = None
    figure_type: Optional[str] = None
    caption_hint: Optional[str] = None
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0

    @property
    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)


@dataclass
class Page:
    index: int  # 0-based physical page in the PDF
    label: Optional[str]  # PRINTED page number ("194"), what a reader verifies
    markdown: str  # the semantic spine, placeholders resolved
    regions: List[Region] = field(default_factory=list)
    conf_avg: Optional[float] = None
    conf_min: Optional[float] = None
    width: int = 0
    height: int = 0

    def region(self, rid: str) -> Optional[Region]:
        return next((r for r in self.regions if r.id == rid), None)


def _page_label(page: Dict[str, Any], blocks: List[Dict[str, Any]]) -> Optional[str]:
    """
    The PRINTED page number — "Mikbook p. 194", not "PDF page 11".

    Read before headers/footers are dropped. A clinician verifying a dose checks
    the printed number; the PDF index is meaningless to them.
    """
    for candidate in (page.get("footer"), page.get("header")):
        if candidate and re.fullmatch(r"\s*(?:\d{1,4}|[ivxlcdm]{1,7})\s*", str(candidate), re.I):
            return str(candidate).strip()
    for b in blocks:
        if (b.get("type") or "").lower() in ("footer", "header"):
            t = (b.get("content") or "").strip()
            if re.fullmatch(r"(?:\d{1,4}|[ivxlcdm]{1,7})", t, re.I):
                return t
    return None


def parse_page(page: Dict[str, Any]) -> Page:
    dims = page.get("dimensions") or {}
    width = float(dims.get("width") or 1) or 1.0
    height = float(dims.get("height") or 1) or 1.0
    raw_blocks = page.get("blocks") or []
    conf = page.get("confidence_scores") or {}

    tables = {t.get("id"): t for t in (page.get("tables") or []) if isinstance(t, dict)}
    images = {i.get("id"): i for i in (page.get("images") or []) if isinstance(i, dict)}

    out = Page(
        index=int(page.get("index", 0)),
        label=_page_label(page, raw_blocks),
        markdown=(page.get("markdown") or "").strip(),
        conf_avg=conf.get("average_page_confidence_score"),
        conf_min=conf.get("minimum_page_confidence_score"),
        width=int(width),
        height=int(height),
    )

    def _annotation(rid: Optional[str]) -> Dict[str, Any]:
        """The vision transcription attached to an image, if the pass ran."""
        ann = (images.get(rid) or {}).get("image_annotation")
        if not ann:
            return {}
        if isinstance(ann, str):
            try:
                import json as _json

                return _json.loads(ann)
            except Exception:  # noqa: BLE001 - a malformed annotation is ignorable
                return {}
        return ann if isinstance(ann, dict) else {}

    for b in raw_blocks:
        btype = (b.get("type") or "text").lower()
        content = (b.get("content") or "").strip()

        # A block whose content IS a placeholder is the geometric counterpart of
        # that markdown reference — that is how the two layers are joined.
        rid = b.get("image_id") or b.get("table_id")
        if not rid:
            m = PLACEHOLDER.fullmatch(content)
            if m:
                rid = m.group(2)

        html = None
        if btype == "table":
            entry = tables.get(rid) or {}
            # Never the markdown pipe version: it loses rowspan/colspan.
            html = entry.get("content") or (content if "<" in content else None)

        ann = _annotation(rid) if btype == "image" else {}
        out.regions.append(
            Region(
                id=rid,
                type=btype,
                text=content,
                html=html,
                image_base64=(images.get(rid) or {}).get("image_base64"),
                transcribed=(ann.get("transcribed_text") or "").strip() or None,
                figure_type=ann.get("figure_type"),
                caption_hint=(ann.get("description") or "").strip() or None,
                # Pixels → 0..1. Scanned books arrive at inconsistent DPI (200,
                # 300, 600 within one volume when assembled from several
                # sessions), so every downstream heuristic is a page fraction.
                x0=float(b.get("top_left_x", 0)) / width,
                y0=float(b.get("top_left_y", 0)) / height,
                x1=float(b.get("bottom_right_x", 0)) / width,
                y1=float(b.get("bottom_right_y", 0)) / height,
            )
        )

    # Drop the running head/foot from the SPINE only — they are ~1 per page and
    # would otherwise become thousands of near-identical vectors all saying
    # "CHAPTER 12 — HEART FAILURE". Their regions are kept (the page label came
    # from them, and a crop may still be wanted).
    out.markdown = _strip_running_heads(out.markdown, out.regions)
    return out


def _strip_running_heads(markdown: str, regions: List[Region]) -> str:
    """Remove header/footer lines from the markdown spine."""
    drop = {
        r.text.strip()
        for r in regions
        if r.type in ("header", "footer") and r.text.strip()
    }
    if not drop:
        return markdown
    kept = [ln for ln in markdown.splitlines() if ln.strip() not in drop]
    return "\n".join(kept).strip()


def parse_response(raw: Dict[str, Any]) -> List[Page]:
    """Parse a whole OCR 4 response, in page order."""
    return [parse_page(p) for p in (raw.get("pages") or [])]
