"""
MedExtract-IA: PDF Layout Extractor

Extracts structured markdown text + images from .pdf files using PyMuPDF (fitz).
Preserves:
  - Bold text via span-level bitwise flag inspection
  - Tables via PyMuPDF table extraction
  - Images via xref → in-memory BytesIO buffers
"""

import io
import re
import fitz  # PyMuPDF


def extract_pdf(file_bytes: bytes) -> tuple[str, dict[str, io.BytesIO]]:
    """
    Extract text and images from a .pdf file.

    Args:
        file_bytes: Raw bytes of the .pdf file

    Returns:
        Tuple of (markdown_text, image_buffers)
        - markdown_text: Clean markdown with **bold** preserved and [[IMG_N]] placeholders
        - image_buffers: Dict mapping image key (e.g. 'img_1') to BytesIO buffer
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    lines: list[str] = []
    image_buffers: dict[str, io.BytesIO] = {}
    image_counter = 0

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Location marker consumed by the RAG chunker (app/rag/chunking.py) and
        # stripped before indexing. Without it the page number is lost here for
        # good, and a grounded answer can only cite a page it invented.
        lines.append(f"<!--loc:page={page_num + 1}-->")

        # ─── Extract images from the page ────────────────────────────
        image_list = page.get_images(full=True)
        for img_info in image_list:
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                if base_image and base_image.get("image"):
                    image_counter += 1
                    key = f"img_{image_counter}"
                    image_buffers[key] = io.BytesIO(base_image["image"])
            except Exception:
                pass

        # ─── Extract text with formatting via dict blocks ────────────
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        page_lines: list[str] = []
        img_placeholder_count = 0

        for block in blocks:
            if block["type"] == 1:  # Image block
                img_placeholder_count += 1
                # Find matching image counter
                img_idx = image_counter - len(image_list) + img_placeholder_count
                if img_idx > 0:
                    page_lines.append(f"[[IMG_{img_idx}]]")
                continue

            if block["type"] != 0:  # text block
                continue

            for line_data in block.get("lines", []):
                parts: list[str] = []
                for span in line_data.get("spans", []):
                    text = span.get("text", "")
                    if not text.strip():
                        parts.append(text)
                        continue

                    # Check bold via flags (bit 4 = bold in PyMuPDF)
                    flags = span.get("flags", 0)
                    is_bold = bool(flags & (1 << 4))  # bit 4

                    if is_bold:
                        parts.append(f"**{text}**")
                    else:
                        parts.append(text)

                line_text = "".join(parts).strip()
                if line_text:
                    # Clean adjacent bold markers
                    line_text = re.sub(r'\*\*\*\*', ' ', line_text)
                    page_lines.append(line_text)

        # ─── Extract tables ──────────────────────────────────────────
        try:
            tables = page.find_tables()
            for table in tables:
                table_data = table.extract()
                if table_data and len(table_data) > 0:
                    md_lines = []
                    for i, row in enumerate(table_data):
                        cells = [str(cell or "").strip() for cell in row]
                        md_lines.append("| " + " | ".join(cells) + " |")
                        if i == 0:
                            md_lines.append("| " + " | ".join(["---"] * len(cells)) + " |")
                    page_lines.append("\n".join(md_lines))
                    page_lines.append("")
        except Exception:
            pass  # Table extraction may not be available in all PyMuPDF versions

        if page_lines:
            lines.extend(page_lines)
            lines.append("")  # Page separator

    doc.close()

    markdown = "\n".join(lines)
    markdown = re.sub(r'\n{3,}', '\n\n', markdown)

    return markdown, image_buffers
