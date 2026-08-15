"""
Free, local analysis that proposes an ingestion route for a PDF.

No network, no API cost: everything here is PyMuPDF over the file the admin
just uploaded. The point is that the admin decides whether to spend money on
OCR — but decides with evidence rather than blind.

The case that matters most for a medical corpus is NOT the obvious scan. It is
the book that was already OCR'd once, badly, by a scanner appliance: it *has* a
text layer, so a naive `chars_per_page > 0` check routes it to the native
pipeline and silently indexes garbage. `garbage_ratio` and `word_like_ratio`
are what catch that.
"""

from __future__ import annotations

import base64
import io
import re
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import fitz  # PyMuPDF — already a dependency (see extractors/pdf_extractor.py)

# Replacement char + control range: what a failed OCR leaves behind.
GARBAGE = re.compile(r"[�\x00-\x08\x0b\x0c\x0e-\x1f]")
WORD = re.compile(r"[A-Za-zÀ-ɏ]{3,}")

# "Heart failure ................ 412"
LEADER = re.compile(r"\.{4,}\s*\d+\s*$")
# "Amiodarone, 214, 337-339"
INDEXY = re.compile(r"^[A-Za-z][\w\s\-,'()]{2,60},\s*\d+(\s*[,–-]\s*\d+)*\s*$")


@dataclass
class Metrics:
    page_count: int
    sampled_pages: List[int]
    chars_per_page_median: float
    chars_per_page_p90: float
    image_area_ratio_median: float
    pages_with_no_text: float
    garbage_ratio: float
    word_like_ratio: float
    has_font_embedding: bool


def sample_pages(n: int, k: int = 18) -> List[int]:
    """
    First 3, last 3, and an even spread of the middle.

    A book's front matter is often a scanned cover with zero text; sampling the
    first pages only would call every book "scanned".
    """
    if n <= k:
        return list(range(n))
    head, tail = [0, 1, 2], [n - 3, n - 2, n - 1]
    step = max(1, (n - 6) // (k - 6))
    mid = list(range(3, n - 3, step))[: k - 6]
    return sorted(set(head + mid + tail))


def _line_ratio(page, pattern: re.Pattern) -> float:
    """Fraction of the page's non-empty lines matching `pattern`."""
    lines = [l for l in page.get_text("text").splitlines() if l.strip()]
    if not lines:
        return 0.0
    return sum(1 for l in lines if pattern.search(l.strip())) / len(lines)


def guess_toc(doc, upto: int = 60) -> List[int]:
    """Pages that look like a table of contents (dot leaders)."""
    return [
        i
        for i in range(min(upto, doc.page_count))
        if _line_ratio(doc[i], LEADER) > 0.30
    ]


def guess_index(doc, tail: int = 90) -> List[int]:
    """Pages that look like a back-of-book index (term, page numbers)."""
    start = max(0, doc.page_count - tail)
    return [i for i in range(start, doc.page_count) if _line_ratio(doc[i], INDEXY) > 0.40]


def _thumbnails(doc, dpi: int = 100, count: int = 5) -> List[Dict[str, Any]]:
    """
    A few pages rendered as base64 PNG.

    On a scanned book the text layer is empty, so numbers alone tell the admin
    nothing — they can only judge by looking. Rendered at 100 DPI rather than
    the plan's 150: these travel through two JSON hops to the browser.
    """
    n = doc.page_count
    if n == 0:
        return []
    picks = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})[:count]
    out: List[Dict[str, Any]] = []
    for i in picks:
        try:
            pix = doc[i].get_pixmap(dpi=dpi)
            out.append(
                {
                    "page": i,
                    "png": base64.b64encode(pix.tobytes("png")).decode("ascii"),
                }
            )
        except Exception:
            continue  # a broken page must not sink the whole inspection
    return out


def _decide(m: Metrics) -> Tuple[str, float]:
    """Ordered decision rules — first match wins."""
    if m.chars_per_page_median < 40 and m.image_area_ratio_median > 0.55:
        return "scanned", 0.95  # classic scanned book
    if m.chars_per_page_median < 40:
        return "scanned", 0.80  # no text and no big image: odd — OCR anyway
    if m.garbage_ratio > 0.02 or (
        m.word_like_ratio < 0.45 and m.chars_per_page_median > 200
    ):
        return "scanned", 0.70  # a text layer exists, but it is junk
    if m.pages_with_no_text > 0.30:
        return "mixed", 0.60  # e.g. scanned plates inside a digital book
    if m.chars_per_page_median > 400 and m.image_area_ratio_median < 0.40:
        return "native", 0.92
    return "unknown", 0.40


def inspect_pdf(file_bytes: bytes, with_thumbnails: bool = True) -> Dict[str, Any]:
    """Analyse a PDF and propose a route. Never raises on page-level oddities."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        n = doc.page_count
        if n == 0:
            raise ValueError("PDF has no pages")
        idx = sample_pages(n)

        chars: List[int] = []
        img_ratios: List[float] = []
        empties = garbage_chars = total_chars = words = 0

        for i in idx:
            p = doc[i]
            txt = p.get_text("text")
            c = len(txt.strip())
            chars.append(c)
            if c < 20:
                empties += 1
            total_chars += len(txt)
            garbage_chars += len(GARBAGE.findall(txt))
            words += len(WORD.findall(txt))

            page_area = abs(p.rect.width * p.rect.height) or 1.0
            covered = 0.0
            for blk in p.get_image_info(hashes=False):
                r = fitz.Rect(blk["bbox"])
                covered += abs(r.width * r.height)
            img_ratios.append(min(covered / page_area, 1.0))

        m = Metrics(
            page_count=n,
            sampled_pages=idx,
            chars_per_page_median=statistics.median(chars),
            chars_per_page_p90=sorted(chars)[max(0, int(len(chars) * 0.9) - 1)],
            image_area_ratio_median=statistics.median(img_ratios),
            pages_with_no_text=empties / len(idx),
            garbage_ratio=(garbage_chars / total_chars) if total_chars else 0.0,
            # ~5 chars per word: how much of the text layer is real words rather
            # than symbol soup.
            word_like_ratio=(words * 5 / total_chars) if total_chars else 0.0,
            has_font_embedding=any(bool(doc[i].get_fonts()) for i in idx[:5]),
        )

        verdict, confidence = _decide(m)
        return {
            "verdict": verdict,
            "confidence": confidence,
            "metrics": asdict(m),
            "pageCount": n,
            "tocPages": guess_toc(doc),
            "indexPages": guess_index(doc),
            "thumbnails": _thumbnails(doc) if with_thumbnails else [],
        }
    finally:
        doc.close()
