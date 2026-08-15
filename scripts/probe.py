"""
Pin the Mistral OCR 4 response contract against a REAL document.

Run once per corpus type, then commit the output as a fixture. The API
reference documents every request parameter but does not fully specify the
`pages[].blocks[]` shape, and field names have moved between OCR versions —
writing the parser first is how you silently produce empty chunks.

    MISTRAL_API_KEY=... python3 scripts/probe.py <book.pdf> [page ...]

Five things this must resolve before any parser is written:
  1. Coordinate space — pixels relative to `dimensions`, or already 0..1?
  2. The text field name — `content`, `text`, or `markdown`?
  3. Confidence shape at "word" granularity — per block, or flat per page?
  4. Do header/footer arrive as blocks, as page fields, or both?
  5. Is `tables` a separate page array duplicating the `table` blocks?
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "ocr4_medical.json"

# The request configuration for scanned medical books. `include_blocks` is the
# entire reason for using OCR 4 rather than a cheap engine: it is what carries
# the typed regions and their bounding boxes.
OCR_PARAMS = dict(
    model="mistral-ocr-latest",
    include_blocks=True,
    # Page granularity, not word: the block text is not what we chunk on (the
    # page `markdown` is), and word scores turned out to be dominated by the
    # publisher's typography (⇨ ② Ⓛ at 0.24-0.34) rather than by text quality.
    confidence_scores_granularity="page",
    # Markdown pipe tables cannot express rowspan/colspan: on the ECG table of
    # this book the cell governing three rows collapsed into three empty cells.
    # The page markdown stays our text spine; table REGIONS use this faithful
    # HTML instead.
    table_format="html",
    # Figures in this corpus are ECGs, algorithms and echo views — the image
    # itself has to travel so it can be stored and shown beside an answer.
    include_image_base64=True,
)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    pdf = Path(sys.argv[1])
    if not pdf.exists():
        print(f"Not found: {pdf}")
        return 2
    # Pick pages that actually exercise the structure: one with a table, one
    # with a formula, one with a figure. A page of plain prose proves nothing.
    pages = [int(a) for a in sys.argv[2:]] or [40, 41, 42, 43, 44]

    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        print("MISTRAL_API_KEY is not set")
        return 2

    # SDK 2.x moved the entry point: `from mistralai import Mistral` (the form
    # in every doc and in the plan) now resolves to a namespace package with no
    # exports. Accept both so this script survives the next move too.
    try:
        from mistralai.client import Mistral  # SDK >= 2.x
    except ImportError:  # pragma: no cover - older SDKs
        from mistralai import Mistral  # type: ignore[no-redef]

    cl = Mistral(api_key=key)
    # Upload + signed URL rather than an inline base64 document. base64 inflates
    # the file by a third, holds raw + encoded + JSON body in RAM at once, cannot
    # resume a dropped transfer — and, decisively, is unusable with the Batch API
    # (one JSONL line per document would be hundreds of MB, which is where the
    # 50% discount lives). base64 is for pages; upload is for books.
    up = cl.files.upload(
        file={"file_name": pdf.name, "content": pdf.open("rb")}, purpose="ocr"
    )
    # 24h, not 1h: batch jobs queue, and a dead link when the worker gets there
    # is an expensive way to learn about expiry.
    url = cl.files.get_signed_url(file_id=up.id, expiry=24).url

    try:
        resp = cl.ocr.process(
            document={"type": "document_url", "document_url": url},
            pages=pages,
            **OCR_PARAMS,
        )
    finally:
        # This request is synchronous, so the file is done with here. For BATCH
        # jobs the delete must wait until the job reports SUCCESS and the results
        # are pulled — never in a finally that fires while it is still queued.
        # Medical material should not sit on a third party longer than needed.
        try:
            cl.files.delete(file_id=up.id)
        except Exception as exc:  # noqa: BLE001 - cleanup must not mask the result
            print(f"warning: could not delete uploaded file {up.id}: {exc}")

    raw = json.loads(resp.model_dump_json())

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")

    pg = (raw.get("pages") or [{}])[0]
    blocks = pg.get("blocks") or []
    print(f"fixture      : {FIXTURE}")
    print("page keys    :", sorted(pg.keys()))
    print("block keys   :", sorted(blocks[0].keys()) if blocks else "NO BLOCKS")
    print("dimensions   :", pg.get("dimensions"))
    print("first bbox   :", (blocks[0] if blocks else {}).get("bbox"))
    print("conf shape   :", json.dumps(pg.get("confidence_scores"))[:400])
    print(
        "types seen   :",
        sorted({b.get("type") for p in raw.get("pages", []) for b in p.get("blocks", [])}),
    )
    print("page arrays  :", [k for k, v in pg.items() if isinstance(v, list)])
    print("usage        :", raw.get("usage_info"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
