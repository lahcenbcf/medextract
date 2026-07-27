# Discussion: Generating Fully Corrected Exams

This document outlines the current limitations, technical challenges, and proposed solutions for injecting correction content directly under each question of an exam file after it passes through the 5-stage pipeline (`RECEIVED` -> `PUBLISHED`).

## 🚨 The Core Technical Challenge: PDF "Reflow"
The biggest hurdle in modifying existing exam files is the **PDF format**. 
Unlike HTML or Word documents, PDFs are **not flow-based**. They are a fixed-layout format where every piece of text is drawn at absolute `(x, y)` coordinates. 
If you try to insert a 5-line paragraph of correction text under Question 1, **Question 2 will not automatically get pushed down**. The correction text will simply overlap and render on top of Question 2.

### Current Implementation Limitations
1. **Loss of Spatial Data:** Our ingestion pipeline (`IngestionJob` via `medextract`) extracts the text and creates `Question` records in the database, but we do not store the exact coordinates (bounding boxes) or page numbers of where that text originally lived.
2. **Missing Source Formatting:** If we reconstruct the document entirely, we risk losing headers, footers, university logos, and specific spatial formatting present in the original source file.

---

## 🛠️ Proposed Solutions & Open-Source Frameworks

Here are the 3 main approaches we can take, ranked from most recommended to least recommended:

### Approach 1: Re-generate a Fresh Document from the Database (Highly Recommended)
Instead of trying to "hack" the original uploaded file, we treat the database as the single source of truth. Once the exam is in the `PUBLISHED` stage, we generate a brand new, beautifully formatted PDF or DOCX that contains the question, the choices, and the correction.

*   **How it works:** We create a standard "Ziania / NoblesQcm Corrected Exam" template. We pull the `Question` array from the DB and render it into the template.
*   **Pros:** 100% reliable, zero overlapping text, consistent branding, easy to include images.
*   **Cons:** Doesn't perfectly match the *exact* visual layout of the original professor's upload (but usually, students prefer a cleaner, standardized format anyway).
*   **Open-Source Frameworks:**
    *   **`WeasyPrint` (Python):** (Best for PDF) Write the exam in HTML/CSS (using a Jinja2 template) and WeasyPrint converts it to a perfect PDF. 
    *   **`docxtpl` (Python):** (Best for DOCX) Create a Word document template with tags like `{{ question.body }}`, and it populates it with DB data.
    *   **`ReportLab` (Python):** Powerful but complex programmatic PDF generation.

### Approach 2: Native `.docx` Modification (Viable if source is DOCX)
If the original uploaded file is a Microsoft Word `.docx` file, we *can* edit it natively because Word documents **do** support text reflow.

*   **How it works:** We open the `.docx` file, search for paragraphs that match the question text in our DB, and insert a new styled paragraph (the correction) immediately following it. The rest of the document naturally pushes down.
*   **Pros:** Keeps the exact original formatting and logos.
*   **Cons:** Only works for `.docx` uploads. Searching for exact text matches can fail if the LLM slightly altered the text during the `CORRECTION` pipeline stage.
*   **Open-Source Frameworks:**
    *   **`python-docx`:** The standard library for reading, querying, and modifying Word documents.

### Approach 3: PDF to Markdown/HTML Conversion (The "Reflow" Hack)
If we absolutely must use the original PDF but need to insert text, we have to convert the PDF into a flow-based format, insert our text, and then convert it back to PDF.

*   **How it works:** Convert PDF -> Markdown/HTML. Find the questions using string matching. Inject the correction markdown. Render Markdown/HTML -> PDF.
*   **Pros:** Works on PDFs.
*   **Cons:** Very high risk of destroying complex layouts (tables, side-by-side columns, header images) during the round-trip conversion.
*   **Open-Source Frameworks:**
    *   **`PyMuPDF` (specifically `pymupdf4llm`):** Excellent at extracting PDF to Markdown.
    *   **`pdf2docx`:** Converts PDF to a Word document using Python, which we can then edit using `python-docx` and export back to PDF (requires LibreOffice headless for the final export).

---

## 🎯 Recommended Next Steps for Implementation

1. **Standardize on Generation (Approach 1):** 
   I recommend we build a route in `z_api` (or a worker in `medextract`) that fetches the `Exam` and its `Question`s, then uses an HTML template to render a fresh "Corrected Version" PDF using `WeasyPrint` or `Puppeteer`. 
2. **Store it alongside the original:** 
   We can add a new field to the `Exam` model: `correctedFileUrl String?`. 
   When the pipeline hits `PUBLISHED`, an async job generates this PDF, uploads it to the CDN, and saves the link.
3. **Dashboard UI:**
   In the Kanban board (or Exam details view), admins can click "Download Original" or "Download Corrected".

Let me know which direction you feel fits the product vision best! We can go deep into native DOCX editing if your users mostly upload Word files, or we can build a robust PDF generator.
