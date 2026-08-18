"""
The Mistral OCR 4 call — the only place in the codebase that spends money.

Upload + signed URL rather than an inline base64 document: base64 inflates the
file by a third, holds raw + encoded + JSON body in memory at once, cannot
resume a dropped transfer, and is unusable with the Batch API (one JSONL line
per document would be hundreds of MB). base64 is for pages; upload is for books.

The uploaded copy is deleted once the response is in hand — medical material
should not sit on a third party longer than the call needs it.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

# Verified against a real scanned book (see fixtures/ocr4_medical.json).
OCR_MODEL = "mistral-ocr-latest"
OCR_PARAMS: Dict[str, Any] = {
    "model": OCR_MODEL,
    # The entire reason for OCR 4 over a cheap engine: typed regions + bboxes.
    "include_blocks": True,
    # Page granularity: word scores turned out to be dominated by the
    # publisher's typography (⇨ ② Ⓛ at 0.24-0.34), not by text quality.
    "confidence_scores_granularity": "page",
    # Markdown pipe tables cannot express rowspan/colspan — on this corpus's ECG
    # table the cell governing three rows collapsed into three empty cells.
    "table_format": "html",
}

# Figures here are ECGs, echo views and treatment algorithms, so the image is
# worth having — but inline base64 does not scale to a book. Five pages already
# weigh 194 KB; a 1,000-page volume would return hundreds of MB to hold in
# memory, cache and re-read on every re-chunk. Above this many pages the images
# are left out and figures are recovered by the targeted annotation pass.
INLINE_IMAGE_PAGE_LIMIT = 120

PRICE_SYNC_PER_PAGE = 0.004
PRICE_BATCH_PER_PAGE = 0.002  # batch halves it — the reason books go through it


def ocr_params(page_count: int | None = None) -> Dict[str, Any]:
    """Request parameters, with image inlining decided by document size."""
    params = dict(OCR_PARAMS)
    params["include_image_base64"] = (
        page_count is None or page_count <= INLINE_IMAGE_PAGE_LIMIT
    )
    return params


def _signed_url(cl, file_id: str, expiry: int = 24, attempts: int = 5) -> str:
    """
    Mint a temporary link to an uploaded file, tolerating propagation lag.

    Observed on a real 18 MB upload: asking for the URL immediately after the
    upload returns 404 "No file matches the given query" — the file exists (it
    is listed and retrievable) but is not yet queryable for signing. Backing off
    a moment fixes it; not retrying makes book ingestion fail at random.
    """
    import time

    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return cl.files.get_signed_url(file_id=file_id, expiry=expiry).url
        except Exception as exc:  # noqa: BLE001 - retried below
            last = exc
            if "404" not in str(exc) and "no file matches" not in str(exc).lower():
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Signed URL unavailable for {file_id}: {last}")


def _client():
    key = os.getenv("MISTRAL_API_KEY", "")
    if not key:
        raise RuntimeError("MISTRAL_API_KEY not configured")
    # SDK 2.x moved the entry point; the documented `from mistralai import
    # Mistral` now resolves to a namespace package with no exports.
    try:
        from mistralai.client import Mistral  # SDK >= 2.x
    except ImportError:  # pragma: no cover - older SDKs
        from mistralai import Mistral  # type: ignore[no-redef]
    return Mistral(api_key=key)


def page_spec(
    start: Optional[int], end: Optional[int], skip: Optional[List[int]] = None
) -> Optional[List[int]]:
    """
    The pages to bill for.

    Front matter and the back-of-book index are not merely paid-for waste: an
    index page is a dense list of every term in the book, so it weakly matches
    every query ever run. Trimming is a retrieval fix as much as a cost one.
    """
    if start is None and end is None and not skip:
        return None  # whole document
    lo = start or 0
    hi = end if end is not None else lo + 5000
    excluded = set(skip or [])
    return [p for p in range(lo, hi + 1) if p not in excluded]


def run_ocr(
    file_bytes: bytes,
    filename: str,
    pages: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    OCR one document synchronously and return the raw response.

    For SMALL documents only. A book must go through `submit_batch` — a
    thousand pages in one blocking call will exhaust any timeout, and costs
    twice the batch rate. Callers must have checked the cache first: this
    function always spends.
    """
    cl = _client()
    up = cl.files.upload(
        file={"file_name": filename, "content": file_bytes}, purpose="ocr"
    )
    file_id = up.id
    try:
        # 24h, not 1h: a queued job that reaches a dead link is an expensive way
        # to learn about expiry.
        url = _signed_url(cl, file_id)
        kwargs = ocr_params(len(pages) if pages else None)
        if pages:
            kwargs["pages"] = pages
        resp = cl.ocr.process(
            document={"type": "document_url", "document_url": url}, **kwargs
        )
        import json as _json

        return _json.loads(resp.model_dump_json())
    finally:
        try:
            cl.files.delete(file_id=file_id)
        except Exception as exc:  # noqa: BLE001 - cleanup must not mask the result
            print(f"[ocr] could not delete uploaded file {file_id}: {exc}")


def billed_pages(raw: Dict[str, Any]) -> int:
    usage = raw.get("usage_info") or {}
    return int(usage.get("pages_processed") or len(raw.get("pages") or []))


def estimated_cost(raw: Dict[str, Any]) -> float:
    return round(billed_pages(raw) * PRICE_SYNC_PER_PAGE, 4)


# ═══ Batch path — the one books go through ═════════════════════════════
#
# A thousand-page volume cannot go through `run_ocr`: one blocking call would
# exhaust every timeout, and sync costs twice the batch rate. Batch turns it
# into submit → poll → fetch, which is also what makes the whole ingestion
# resumable: the state that matters lives in z_api's KbDocument row, not in
# this process.

TERMINAL_OK = {"SUCCESS"}
TERMINAL_BAD = {"FAILED", "CANCELLED", "CANCELLATION_REQUESTED", "TIMEOUT_EXCEEDED"}


def submit_batch(
    file_bytes: bytes,
    filename: str,
    custom_id: str,
    pages: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Upload the book and queue a batch OCR job.

    Returns the ids the caller must persist: without `file_id` a crashed worker
    could never delete our copy from Mistral's storage.
    """
    import json as _json
    import tempfile

    cl = _client()
    up = cl.files.upload(
        file={"file_name": filename, "content": file_bytes}, purpose="ocr"
    )
    # 24h: batch jobs queue, and a link that dies before the worker reaches it
    # is an expensive way to learn about expiry.
    url = _signed_url(cl, up.id)

    body: Dict[str, Any] = {
        "document": {"type": "document_url", "document_url": url},
        **ocr_params(len(pages) if pages else None),
    }
    if pages:
        body["pages"] = pages
    # One JSONL line per request — ~400 bytes here. This is precisely what an
    # inline base64 document would make impossible.
    line = _json.dumps({"custom_id": custom_id, "body": body}, ensure_ascii=False)

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        fh.write(line + "\n")
        batch_input_path = fh.name
    with open(batch_input_path, "rb") as fh:
        batch_file = cl.files.upload(
            file={"file_name": "ocr_batch.jsonl", "content": fh.read()},
            purpose="batch",
        )

    job = cl.batch.jobs.create(
        input_files=[batch_file.id],
        model=OCR_MODEL,
        endpoint="/v1/ocr",
        metadata={"pipeline": "nobles-kb"},
    )
    return {
        "batch_job_id": job.id,
        "file_id": up.id,
        "batch_file_id": batch_file.id,
        "status": str(getattr(job, "status", "QUEUED")),
    }


def get_batch(batch_job_id: str) -> Dict[str, Any]:
    """Current state of a batch job — what the poller reads."""
    job = _client().batch.jobs.get(job_id=batch_job_id)
    return {
        "id": job.id,
        "status": str(job.status),
        "total": getattr(job, "total_requests", None),
        "succeeded": getattr(job, "succeeded_requests", None),
        "failed": getattr(job, "failed_requests", None),
        "output_file": getattr(job, "output_file", None),
        "error_file": getattr(job, "error_file", None),
        "done": str(job.status) in TERMINAL_OK | TERMINAL_BAD,
        "ok": str(job.status) in TERMINAL_OK,
    }


def fetch_batch_result(output_file_id: str) -> Dict[str, Any]:
    """
    Download a finished batch's output and return the OCR response itself.

    The output is JSONL: one line per request, each wrapping the real response
    under `response.body`. We submit one document per job, so there is one line.
    """
    import json as _json

    cl = _client()
    raw = cl.files.download(file_id=output_file_id)
    content = raw.read() if hasattr(raw, "read") else bytes(raw)
    text = content.decode("utf-8") if isinstance(content, bytes) else str(content)

    for line in text.splitlines():
        if not line.strip():
            continue
        entry = _json.loads(line)
        body = (entry.get("response") or {}).get("body")
        if body:
            return body
        if entry.get("pages"):  # tolerate a bare response shape
            return entry
    raise RuntimeError("Batch output contained no OCR response")


def cleanup_batch(file_id: Optional[str], batch_file_id: Optional[str] = None) -> None:
    """
    Delete our copies from Mistral's storage.

    Called only once the job reports SUCCESS and the results are pulled — never
    in a `finally` that would fire while the job is still queued.
    """
    cl = _client()
    for fid in (file_id, batch_file_id):
        if not fid:
            continue
        try:
            cl.files.delete(file_id=fid)
        except Exception as exc:  # noqa: BLE001
            print(f"[ocr] could not delete file {fid}: {exc}")


def batch_cost(page_count: int) -> float:
    return round(page_count * PRICE_BATCH_PER_PAGE, 4)


# ═══ Figure annotation — recovering text locked inside image regions ════
#
# Measured on a real Mikbook: 5 of 39 pages returned barely 350 characters
# because their boxed clinical content was classified as an `image` region, not
# as text. The "ANGOR DE PRINZMETAL" box — a complete answer to an exam question
# — never existed as text and so could never be retrieved. This pass sends those
# regions to a vision model and transcribes them back.

# Page fraction below which a picture is assumed decorative.
#
# ZERO, i.e. annotate every figure. It was 0.10, on the theory that a small
# image is a publisher logo — and that theory cost this corpus most of its
# figures: 30 of 41 images sat below the bar, among them the 6.4% region
# holding stages I and II of the Leriche-Fontaine classification. Only 3% of
# the images below the threshold ever got a transcription, against 91% above.
#
# The bar existed to control spend. Measured on the real book, annotating
# EVERY figure costs 0.125 USD — 25 pages at Document AI pricing. There was no
# spend to control. Raise it via the environment for a corpus whose images
# really are decorative; never raise it to save money.
FIGURE_PASS_MIN_AREA = float(os.getenv("OCR4_FIGURE_MIN_AREA", "0"))
# `image_limit` caps images PER CALL and drops the overflow WITHOUT an error.
# Batches are therefore built by image count, not by page count: three pages
# carrying four figures each would silently lose four of them, and the loss is
# invisible — the response simply comes back short.
IMAGES_PER_CALL = 8
PRICE_DOCAI_PER_PAGE = 0.005


def figure_pages(
    raw: Dict[str, Any],
    min_area: float = FIGURE_PASS_MIN_AREA,
    skip_annotated: bool = True,
) -> List[int]:
    """
    Pages carrying an image region big enough to hide real content.

    Targeted on purpose: annotation moves a page to Document AI pricing and adds
    a vision call per image. Without the area filter we would fire one at every
    publisher logo and section rule.

    `skip_annotated` makes the pass IDEMPOTENT: a page whose large images already
    carry a transcription is not re-sent. That is what lets the caller run this
    on a cached response — a response stored before the pass existed has figures
    and no transcriptions, and would otherwise stay blind forever.
    """
    out: List[int] = []
    for page in raw.get("pages") or []:
        dims = page.get("dimensions") or {}
        w = float(dims.get("width") or 1) or 1.0
        h = float(dims.get("height") or 1) or 1.0
        for img in page.get("images") or []:
            area = (
                abs(float(img.get("bottom_right_x", 0)) - float(img.get("top_left_x", 0)))
                * abs(float(img.get("bottom_right_y", 0)) - float(img.get("top_left_y", 0)))
            ) / (w * h)
            if area < min_area:
                continue
            if skip_annotated and img.get("image_annotation"):
                continue
            out.append(int(page.get("index", 0)))
            break
    return out


def _annotation_batches(
    pages: List[int], images_per_page: Dict[int, int] | None
) -> List[List[int]]:
    """
    Group pages so no single call can exceed `IMAGES_PER_CALL`.

    Batching by PAGE count was a coin toss: three pages carrying four figures
    each is twelve images against a cap of eight, and the four that overflow are
    dropped with no error. Counting the images we already know about removes the
    guesswork. Without a count (caller did not supply one) we assume the
    worst — one page per call — rather than gamble.
    """
    out: List[List[int]] = []
    batch: List[int] = []
    load = 0
    for page in pages:
        n = max(1, (images_per_page or {}).get(page, IMAGES_PER_CALL))
        # A single page with more images than the cap still has to go alone;
        # nothing here can rescue it, but it must not drag neighbours down.
        if batch and load + n > IMAGES_PER_CALL:
            out.append(batch)
            batch, load = [], 0
        batch.append(page)
        load += n
    if batch:
        out.append(batch)
    return out


def annotate_figures(
    file_bytes: bytes,
    filename: str,
    pages: List[int],
    images_per_page: Dict[int, int] | None = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Transcribe the text inside figure regions, page by page.

    Returns {page_index: {image_id: annotation}} so the caller can merge it into
    the cached OCR response — after which re-chunking stays free.

    `images_per_page` lets the caller size the batches from the response it
    already has; see `_annotation_batches`.
    """
    if not pages:
        return {}

    from pydantic import BaseModel, Field
    from mistralai.extra import response_format_from_pydantic_model

    class FigureContent(BaseModel):
        figure_type: str = Field(
            ...,
            description="algorithm | flowchart | ecg | imaging | graph | photo | diagram | text_box",
        )
        transcribed_text: str = Field(
            ...,
            description=(
                "Verbatim transcription of ALL text visible in the image, in French, "
                "preserving reading order and line breaks. Empty string if none."
            ),
        )
        description: str = Field(
            ..., description="One clinical sentence describing what this figure shows."
        )

    cl = _client()
    up = cl.files.upload(
        file={"file_name": filename, "content": file_bytes}, purpose="ocr"
    )
    try:
        url = _signed_url(cl, up.id)
        import json as _json

        out: Dict[int, Dict[Any, Any]] = {}
        # In BATCHES, because `image_limit` caps the images processed per call
        # and silently drops the rest: asking for ten pages at once returned
        # annotations for the first eight images only — the page we actually
        # needed was among the dropped ones, with no error anywhere.
        for batch in _annotation_batches(pages, images_per_page):
            resp = cl.ocr.process(
                model=OCR_MODEL,
                document={"type": "document_url", "document_url": url},
                pages=batch,
                bbox_annotation_format=response_format_from_pydantic_model(
                    FigureContent
                ),
                # REQUIRED: the crop has to be sent to the vision model.
                include_image_base64=True,
                # No `image_min_size`: the base OCR call sets none, so any floor
                # here would return a DIFFERENT set of regions than the cached
                # response holds — small figures would simply never come back,
                # and the ones that did would be cropped on other boundaries.
                # The two calls must see the same page the same way.
                image_limit=IMAGES_PER_CALL,
            )
            raw = _json.loads(resp.model_dump_json())
            _collect_annotations(raw, out)
        return out
    finally:
        try:
            cl.files.delete(file_id=up.id)
        except Exception as exc:  # noqa: BLE001
            print(f"[ocr] could not delete {up.id}: {exc}")


def _collect_annotations(raw: Dict[str, Any], out: Dict[int, Dict[Any, Any]]) -> None:
    """Accumulate one response's annotations, keyed by page and position."""
    for page in raw.get("pages") or []:
        idx = int(page.get("index", 0))
        for img in page.get("images") or []:
            ann = img.get("image_annotation")
            if ann:
                    # Keyed by POSITION, not by id: image ids are numbered per
                    # RESPONSE, so a pass over pages [3,5,7…] restarts at
                    # "img-0" while the cached full-document response calls the
                    # same region "img-3". Matching on ids silently merged
                    # nothing — the coordinates are the only stable identity.
                out.setdefault(idx, {})[_bbox_key(img)] = ann


def _bbox_key(img: Dict[str, Any]) -> tuple:
    """A region's stable identity across calls: its pixel coordinates."""
    return (
        int(img.get("top_left_x", 0)),
        int(img.get("top_left_y", 0)),
        int(img.get("bottom_right_x", 0)),
        int(img.get("bottom_right_y", 0)),
    )


# How much of the SMALLER box must fall inside the other for the two to be the
# same content. Measured against the real responses, this covers both ways the
# region detector disagrees with itself between calls:
#
#   drift — the same figure comes back 1-5 px off (ratio ≈ 0.99);
#   split — one cached image comes back as two stacked halves (ratio 1.0 for
#           each half, while their mutual IoU with the whole is only 0.49).
#
# Two genuinely different figures on a page score 0.
BBOX_MATCH_MIN_OVERLAP = 0.7


def _overlap_ratio(a: tuple, b: tuple) -> float:
    """Intersection as a fraction of the SMALLER of the two boxes."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


def _combine_annotations(parts: List[Any]) -> Any:
    """
    Fuse the annotations of one region that came back split into several.

    Keeping only one half would leave the figure half-read — and on this corpus
    a figure IS the content, so half a transcription is a wrong answer waiting
    to happen.
    """
    if len(parts) == 1:
        return parts[0]
    import json as _json

    texts, descs, ftype = [], [], ""
    for part in parts:
        try:
            d = _json.loads(part) if isinstance(part, str) else part
        except (ValueError, TypeError):
            continue
        if not isinstance(d, dict):
            continue
        ftype = ftype or (d.get("figure_type") or "")
        if (d.get("transcribed_text") or "").strip():
            texts.append(d["transcribed_text"].strip())
        if (d.get("description") or "").strip():
            descs.append(d["description"].strip())
    if not texts and not descs:
        return parts[0]
    return _json.dumps(
        {
            "figure_type": ftype or "diagram",
            "transcribed_text": "\n".join(texts),
            "description": " ".join(descs),
        },
        ensure_ascii=False,
    )


def merge_annotations(
    raw: Dict[str, Any], annotations: Dict[int, Dict[Any, Any]]
) -> Dict[str, Any]:
    """
    Fold annotations into the OCR response, so the CACHE carries them too.

    Matched by OVERLAP, not by equality. Coordinates are the only stable
    identity across calls — image ids restart per response — but they are not
    stable to the pixel: measured on this book, the same region came back as
    (85, 175, 622, 572) from the base call and (85, 174, 617, 568) from the
    annotation pass. Exact-integer keys dropped that transcription on the floor,
    and the page kept a figure nobody could read. A one-pixel wobble in a region
    detector must never cost a page its content.
    """
    merged = 0
    for page in raw.get("pages") or []:
        # Copied, and entries removed as they are used: one annotation must not
        # be claimed by two regions.
        page_ann = dict(annotations.get(int(page.get("index", 0))) or {})
        if not page_ann:
            continue
        for img in page.get("images") or []:
            key = _bbox_key(img)
            exact = page_ann.pop(key, None)  # the common case
            parts = [exact] if exact is not None else []
            if not parts and page_ann:
                # EVERY overlapping region, not the best one: a figure that came
                # back split into halves must be reassembled, not halved.
                for candidate in [
                    k
                    for k in page_ann
                    if _overlap_ratio(key, k) >= BBOX_MATCH_MIN_OVERLAP
                ]:
                    parts.append(page_ann.pop(candidate))
            if parts:
                img["image_annotation"] = _combine_annotations(parts)
                merged += len(parts)
    expected = sum(len(v) for v in annotations.values())
    if merged < expected:
        # Loud on purpose: a silent partial merge is what cost a whole book its
        # annotations once already.
        print(f"[ocr] WARNING: merged {merged}/{expected} annotations")
    return raw
