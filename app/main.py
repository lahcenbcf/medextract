"""
MedExtract-IA: FastAPI Microservice

Accepts .docx/.pdf files via HTTP, extracts QCM questions using
a deterministic layout parser + LLM semantic layer, streams images
to Bunny.net, and posts the structured result to the NestJS callback.
"""

import os
import io
import time
import re
import socket
import traceback
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.responses import JSONResponse

from app.extractors.docx_extractor import extract_docx
from app.extractors.pdf_extractor import extract_pdf
from app.llm.structured_output import extract_questions
from app.upload.bunny_uploader import upload_all_images
from app.schemas import (
    ExtractionResult,
    ParsedQuestion,
    ParseMetadata,
    ClinicalCaseGroup,
    ClinicalCasePart,
    Choice,
)
from pydantic import BaseModel
from fastapi import BackgroundTasks
from app.llm.correction_graph import (
    run_correction,
    run_classification,
    run_module_classification,
)
from app.rag.ingest import ingest_ocr_response, ingest_text, retrieve
from app.rag.jobs import create_job, get_job, list_jobs, update_job
from app.rag.inspector import inspect_pdf
from app.rag.vector_store import (
    delete_by_metadata,
    file_urls_for_source,
    list_sources,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 MedExtract-IA service started")
    yield
    print("🛑 MedExtract-IA service shutting down")


app = FastAPI(
    title="MedExtract-IA",
    description="QCM extraction microservice for medical documents",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "medextract-ia"}


@app.get("/llm/models")
def get_llm_models():
    """
    Fetch dynamically the available models from the Gemini API.
    Only models supporting 'generateContent' are returned.
    """
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return {"models": []}
    
    try:
        resp = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}", timeout=10)
        if resp.status_code == 200:
            data = resp.json().get("models", [])
            valid_models = [m["name"].replace("models/", "") for m in data if "generateContent" in m.get("supportedGenerationMethods", [])]
            return {"models": valid_models}
    except Exception as exc:
        print(f"[llm/models] failed to fetch from Gemini: {exc}")
    
    return {"models": []}


# ─── Correction chat (Phase 3 / 4) ──────────────────────────────────────
class ChatCorrectionRequest(BaseModel):
    message: str
    history: list[dict] = []
    systemPrompt: str = ""
    profile: str = "standard"
    useWebSearch: bool = False
    useLocalKb: bool = False
    # Strict RAG filter forwarded by z_api (e.g. {"module": …, "year": …})
    metadata: dict = {}
    # Candidate courses for the classify node — [{id, name}], from z_api
    candidateCourses: list[dict] = []
    # Clinical-case memory (partial submissions), managed by z_api
    clinicalCase: bool = False
    caseContext: str = ""
    examId: int | None = None
    # Gemini model id (from z_api's global setting); env fallback if absent
    model: str | None = None


@app.post("/chat/correction")
def chat_correction(req: ChatCorrectionRequest):
    """
    LangGraph-powered correction chat, proxied here by z_api.

    Structured mode returns {"questions": [...]} — one entry per pasted question,
    each corrected and annotated with a suggested course. Web-search mode returns
    {"reply": "<markdown>"}.
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message is required")
    try:
        return run_correction(
            message=req.message,
            history=req.history,
            system_prompt=req.systemPrompt,
            use_web_search=req.useWebSearch,
            use_local_kb=req.useLocalKb,
            metadata=req.metadata,
            candidate_courses=req.candidateCourses,
            clinical_case=req.clinicalCase,
            case_context=req.caseContext,
            model=req.model,
        )
    except Exception as exc:
        print(f"[chat/correction] failed: {exc}")
        raise HTTPException(status_code=502, detail="AI pipeline failed") from exc


class ClassifyRequest(BaseModel):
    # Already-structured questions to classify — [{description, caseDescription?}]
    questions: list[dict] = []
    # Course catalogue — [{id, name, group?}] (group = "module · année" for residanat)
    candidateCourses: list[dict] = []
    # False for residanat: pick existing ids only, never invent a course name
    allowNewCourse: bool = True
    model: str | None = None


@app.post("/classify")
def classify(req: ClassifyRequest):
    """
    Classify already-corrected questions against a course catalogue in ONE LLM
    call. Returns {"questions": [...]} where each entry gains suggestedCourId /
    suggestedCourName / classifyConfidence. Persists nothing (HITL: admin approves).
    """
    if not req.questions:
        return {"questions": []}
    try:
        return run_classification(
            questions=req.questions,
            candidate_courses=req.candidateCourses,
            allow_new_course=req.allowNewCourse,
            model=req.model,
        )
    except Exception as exc:
        print(f"[classify] failed: {exc}")
        raise HTTPException(status_code=502, detail="Classification failed") from exc


class TranslateRequest(BaseModel):
    # The stored question: { description, caseDescription?, explanation?,
    # choices[], propositions[] } — labels are echoed back untouched.
    question: dict
    language: str = "en"
    model: str | None = None
    # FR→EN pairs, ALREADY filtered by z_api to those present in this question.
    glossary: list[dict] = []


@app.post("/translate")
def translate(req: TranslateRequest):
    """
    Produce the English version of one question.

    Not a literal translation: the prompt asks for medical English as written
    for medical students, while labels, K-type combinations and HTML markup are
    preserved verbatim.
    """
    if not req.question:
        raise HTTPException(status_code=400, detail="question is required")
    try:
        from app.llm.translation import translate_question

        return translate_question(
            req.question,
            language=req.language,
            model=req.model,
            glossary=req.glossary,
        )
    except Exception as exc:
        print(f"[translate] failed: {exc}")
        raise HTTPException(status_code=502, detail="Translation failed") from exc


class ClassifyModuleRequest(BaseModel):
    # Already-structured questions — [{description, caseDescription?}]
    questions: list[dict] = []
    # Module catalogue — [{id, name, group?}] (group = "année")
    candidateModules: list[dict] = []
    model: str | None = None


@app.post("/classify-module")
def classify_module(req: ClassifyModuleRequest):
    """
    Résidanat: pick the best-fitting MODULE (from the DB modules list) for each
    already-structured question in ONE LLM call. Returns {"questions": [...]} where
    each entry has suggestedModuleId / suggestedModuleName / confidence.
    """
    if not req.questions:
        return {"questions": []}
    try:
        return run_module_classification(
            questions=req.questions,
            candidate_modules=req.candidateModules,
            model=req.model,
        )
    except Exception as exc:
        print(f"[classify-module] failed: {exc}")
        raise HTTPException(status_code=502, detail="Module classification failed") from exc


# ─── Knowledge Base RAG ingestion (Part 6.2) ────────────────────────────
def _run_ingest(job_id: str, text: str, metadata: dict) -> None:
    """Background worker: chunk → enrich → embed → store."""
    update_job(job_id, status="processing")
    try:
        result = ingest_text(text, metadata)
        update_job(
            job_id,
            status="done",
            chunks=result["chunks"],
            parents=result["parents"],
        )
        print(f"[rag/ingest] job {job_id} done: {result}")
    except Exception as exc:
        print(f"[rag/ingest] job {job_id} failed: {exc}")
        update_job(job_id, status="failed", error=_friendly_ingest_error(exc))


def _run_ingest_file(job_id: str, file_bytes: bytes, ext: str, metadata: dict) -> None:
    """Background worker for uploads: extract text FIRST (large PDFs can be slow,
    so it must not block the HTTP request), then chunk → enrich → embed → store."""
    update_job(job_id, status="processing")
    try:
        if ext == "docx":
            document_text, _ = extract_docx(file_bytes)
        else:  # pdf (already validated at the endpoint)
            document_text, _ = extract_pdf(file_bytes)
        if not document_text.strip():
            update_job(job_id, status="failed", error="Aucun texte extractible du document.")
            return
        result = ingest_text(document_text, metadata)
        update_job(
            job_id,
            status="done",
            chunks=result["chunks"],
            parents=result["parents"],
        )
        print(f"[rag/ingest] job {job_id} done: {result}")
    except Exception as exc:
        print(f"[rag/ingest] job {job_id} failed: {exc}")
        update_job(job_id, status="failed", error=_friendly_ingest_error(exc))


def _friendly_ingest_error(exc: Exception) -> str:
    """Turn low-level network/DNS errors into an admin-readable message."""
    raw = str(exc)
    if "name resolution" in raw or "gaierror" in raw or isinstance(exc, socket.gaierror):
        if "qdrant" in raw.lower():
            return "Base vectorielle injoignable — le service Qdrant est-il démarré ?"
        return "Service IA injoignable (réseau/DNS). Vérifiez la connectivité du serveur."
    if isinstance(exc, (ConnectionError, requests.ConnectionError)):
        return "Connexion au service échouée. Réessayez ou vérifiez le réseau."
    return raw


# ═══ Batch OCR — three stateless steps, driven by z_api ════════════════
#
# The durable state lives in z_api's KbDocument row, NOT here: a batch on a
# thousand-page book can run for hours, and this service's job registry is
# in-memory. These endpoints are deliberately thin.


@app.post("/rag/ocr/submit")
async def rag_ocr_submit(
    file: UploadFile = File(...),
    sha256: str = Form(...),
    content_page_start: int | None = Form(default=None),
    content_page_end: int | None = Form(default=None),
    skip_pages: str = Form(default=""),
):
    """
    Queue a batch OCR job. Returns immediately with the ids to persist.

    Cache-first: an already-OCR'd book returns `cached: true` and no job — the
    caller then goes straight to /rag/ocr/finalize, free of charge.
    """
    from app.rag.ocr4 import cache as ocr_cache
    from app.rag.ocr4.client import page_spec, submit_batch

    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="OCR applies to .pdf only")
    if not sha256.strip():
        raise HTTPException(status_code=400, detail="sha256 is required (cache key)")

    if ocr_cache.load(sha256.strip()) is not None:
        return {"cached": True, "ocrRawUri": ocr_cache.cache_uri(sha256.strip())}

    data = await file.read()
    skip = [int(x) for x in skip_pages.split(",") if x.strip().isdigit()]
    pages = page_spec(content_page_start, content_page_end, skip)
    try:
        return {"cached": False, **submit_batch(data, file.filename, sha256.strip(), pages)}
    except Exception as exc:
        print(f"[rag/ocr/submit] failed: {exc}")
        raise HTTPException(status_code=502, detail=f"OCR submission failed: {exc}")


@app.get("/rag/ocr/batch/{batch_job_id}")
def rag_ocr_batch_status(batch_job_id: str):
    """Poll a batch job. Cheap and safe to call on a schedule."""
    from app.rag.ocr4.client import get_batch

    try:
        return get_batch(batch_job_id)
    except Exception as exc:
        print(f"[rag/ocr/batch] failed: {exc}")
        raise HTTPException(status_code=502, detail=f"Batch status failed: {exc}")


def _annotate_figures_if_needed(
    raw: dict, metadata: dict, pdf_bytes: bytes | None = None
) -> tuple[dict, bool]:
    """
    Recover the text locked inside image regions.

    Measured on a real book: 5 of 39 pages came back with barely 350 characters
    because their boxed clinical content was typed as an `image`. A complete
    answer to an exam question existed only as a picture, so retrieval could
    never find it. Targeted at pages with a large figure — annotation is billed
    per page and a vision call at every publisher logo would be waste.

    Best-effort: losing the annotations costs recall, losing the whole ingestion
    costs the OCR too.
    """
    from app.rag.ocr4.client import annotate_figures, figure_pages, merge_annotations

    pages = figure_pages(raw)
    if not pages:
        return raw, False

    try:
        data = pdf_bytes
        if data is None:
            # Fallback only. z_api normally streams the file with the request:
            # it already holds it, and this download crosses the public internet
            # from inside a container — it timed out on a real ingestion and
            # silently cost us the annotations of a whole book.
            file_url = metadata.get("file_url")
            if not file_url:
                print("[rag/ocr] figures need annotation but no file — skipped")
                return raw, False
            import requests

            # Separate connect/read timeouts: 300s to merely OPEN a socket turns
            # an unreachable host into a five-minute hang.
            resp = requests.get(file_url, timeout=(10, 180))
            resp.raise_for_status()
            data = resp.content

        annotations = annotate_figures(
            data, metadata.get("source") or "document.pdf", pages
        )
        if not annotations:
            return raw, False
        print(f"[rag/ocr] annotated {len(annotations)} page(s) of figures")
        # `merge_annotations` mutates in place, so identity cannot signal a
        # change — the caller needs an explicit flag to know it must re-cache.
        return merge_annotations(raw, annotations), True
    except Exception as exc:  # noqa: BLE001
        print(f"[rag/ocr] figure annotation failed (continuing): {exc}")
        return raw, False


def _run_batch_finalize(
    job_id: str,
    sha256: str,
    output_file_id: str | None,
    file_id: str | None,
    batch_file_id: str | None,
    metadata: dict,
    pdf_bytes: bytes | None = None,
) -> None:
    """Background worker: fetch the batch output → cache → chunk → index."""
    from app.rag.ocr4 import cache as ocr_cache
    from app.rag.ocr4.client import batch_cost, billed_pages, cleanup_batch, fetch_batch_result

    update_job(job_id, status="processing")
    try:
        raw = ocr_cache.load(sha256)
        cached = raw is not None
        if raw is None:
            if not output_file_id:
                raise RuntimeError("No cached result and no batch output file")
            raw = fetch_batch_result(output_file_id)
            # Only now, with the results in hand, is it safe to delete our
            # copies from Mistral's storage.
            cleanup_batch(file_id, batch_file_id)

        # Annotation runs on BOTH paths, not just on a cache miss. A response
        # cached before this pass existed has figures and no transcriptions —
        # re-indexing it would silently keep the book blind, which is exactly
        # what happened to the first book. `figure_pages` skips already
        # annotated images, so a second run over a complete cache costs nothing.
        raw, newly_annotated = _annotate_figures_if_needed(raw, metadata, pdf_bytes)
        if newly_annotated or not cached:
            # Cache BEFORE indexing, annotations included: if chunking then
            # crashes, both paid steps are banked and the retry costs nothing.
            ocr_cache.store(sha256, raw)

        result = ingest_ocr_response(raw, metadata)
        pages_billed = 0 if cached else billed_pages(raw)
        update_job(
            job_id,
            status="done",
            chunks=result["chunks"],
            parents=result["parents"],
            ocrCached=cached,
            ocrRawUri=ocr_cache.cache_uri(sha256),
            ocrPagesBilled=pages_billed,
            ocrCostUsd=0.0 if cached else batch_cost(pages_billed),
        )
        print(f"[rag/ocr/finalize] job {job_id} done: {result}")
    except Exception as exc:
        print(f"[rag/ocr/finalize] job {job_id} failed: {exc}")
        update_job(job_id, status="failed", error=str(exc))


@app.post("/rag/ocr/finalize", status_code=202)
async def rag_ocr_finalize(
    background: BackgroundTasks,
    file: UploadFile | None = File(default=None),
    sha256: str = Form(...),
    output_file_id: str = Form(default=""),
    file_id: str = Form(default=""),
    batch_file_id: str = Form(default=""),
    year: str = Form(default=""),
    module: str = Form(default=""),
    course: str = Form(default=""),
    source: str = Form(default=""),
    doc_type: str = Form(default="reference", alias="type"),
    file_url: str = Form(default=""),
):
    """Turn a finished (or cached) OCR result into indexed chunks."""
    metadata = {
        k: v
        for k, v in {
            "year": year,
            "module": module,
            "course": course,
            "type": doc_type,
            "source": source,
            "file_url": file_url,
        }.items()
        if v
    }
    # The original PDF, streamed by z_api over the internal network so the
    # figure-annotation pass never depends on reaching the CDN itself.
    pdf_bytes = await file.read() if file is not None else None

    job_id = create_job(metadata)
    background.add_task(
        _run_batch_finalize,
        job_id,
        sha256.strip(),
        output_file_id or None,
        file_id or None,
        batch_file_id or None,
        metadata,
        pdf_bytes,
    )
    return {"jobId": job_id, "status": "queued"}


@app.post("/rag/inspect")
async def rag_inspect(
    file: UploadFile = File(...),
    thumbnails: bool = Form(default=True),
):
    """
    Propose an ingestion route for a PDF — free, local, no API call.

    Deliberately does NOT touch the index or the job registry: it only reports.
    z_api owns the registry row and the approval gate; this endpoint just gives
    the admin the evidence to decide with.
    """
    name = (file.filename or "").lower()
    if not name.endswith(".pdf"):
        # .docx is always a text format — there is nothing to OCR.
        raise HTTPException(
            status_code=400, detail="Inspection only applies to .pdf files"
        )
    try:
        data = await file.read()
        return inspect_pdf(data, with_thumbnails=thumbnails)
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[rag/inspect] failed: {exc}")
        raise HTTPException(status_code=422, detail=f"Could not inspect PDF: {exc}")


@app.post("/rag/ingest", status_code=202)
async def rag_ingest(
    background: BackgroundTasks,
    file: UploadFile | None = File(default=None),
    text: str = Form(default=""),
    year: str = Form(default=""),
    module: str = Form(default=""),
    course: str = Form(default=""),
    source: str = Form(default=""),
    doc_type: str = Form(default="course", alias="type"),
    file_url: str = Form(default=""),
):
    """
    Add a document to the Local Knowledge Base.

    Accepts either an uploaded .pdf/.docx or raw `text`, plus the metadata used
    for strict filtering at retrieval time, and an optional `file_url` pointing
    at the stored original (so the admin can open the cited page). Returns immediately with a jobId —
    the chunk/enrich/embed/store work runs in the background so the admin UI
    never blocks. Poll GET /rag/status/{job_id}.
    """
    resolved_source = source
    file_bytes: bytes | None = None
    ext = ""

    if file is not None and file.filename:
        # Read the bytes here, but defer extraction to the background worker so a
        # large (100-200MB) document never blocks the request or times it out.
        file_bytes = await file.read()
        file_name = file.filename
        ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
        if ext not in ("pdf", "docx"):
            raise HTTPException(
                status_code=400, detail=f"Unsupported file type: .{ext}"
            )
        resolved_source = source or file_name
    elif not text.strip():
        raise HTTPException(status_code=400, detail="Provide a file or text")

    metadata = {
        "year": year,
        "module": module,
        "course": course,
        "type": doc_type,
        "source": resolved_source,
        # Where the ORIGINAL file lives (z_api uploaded it to the CDN before
        # forwarding). Carried into every chunk so a grounded answer can be
        # opened at its page. Not a filter key — purely informational.
        "file_url": file_url,
    }
    metadata = {k: v for k, v in metadata.items() if v}

    job_id = create_job(metadata)
    if file_bytes is not None:
        background.add_task(_run_ingest_file, job_id, file_bytes, ext, metadata)
    else:
        background.add_task(_run_ingest, job_id, text, metadata)
    return {"jobId": job_id, "status": "queued", "metadata": metadata}


@app.get("/rag/status/{job_id}")
def rag_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/rag/jobs")
def rag_jobs():
    """List all ingestion jobs (newest first) so the UI can always show pending
    work — even after the admin navigates away and back."""
    return {"jobs": list_jobs()}


class RagSearchRequest(BaseModel):
    query: str
    year: str | None = None
    module: str | None = None
    course: str | None = None
    type: str | None = None
    topK: int = 5


@app.post("/rag/search")
def rag_search(req: RagSearchRequest):
    """Metadata-filtered semantic search, auto-merged to parent sections."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query is required")
    metadata = {
        k: v
        for k, v in {
            "year": req.year,
            "module": req.module,
            "course": req.course,
            "type": req.type,
        }.items()
        if v
    }
    try:
        return {"results": retrieve(req.query, metadata, top_k=req.topK)}
    except Exception as exc:
        print(f"[rag/search] failed: {exc}")
        raise HTTPException(status_code=502, detail="Search failed") from exc


@app.get("/rag/sources")
def rag_sources():
    """List indexed documents (grouped by source) for the KB management UI."""
    try:
        return {"sources": list_sources()}
    except Exception as exc:
        print(f"[rag/sources] failed: {exc}")
        raise HTTPException(status_code=502, detail="Could not list sources") from exc


class RagDeleteRequest(BaseModel):
    source: str


@app.delete("/rag/sources")
def rag_delete_source(req: RagDeleteRequest):
    """Remove all chunks of one document from the Knowledge Base."""
    source = (req.source or "").strip()
    if not source:
        raise HTTPException(status_code=400, detail="source is required")
    try:
        # Read the stored originals BEFORE dropping the chunks — afterwards the
        # only record of where the file lives is gone. z_api uses these to clean
        # up the CDN.
        file_urls = file_urls_for_source(source)
        delete_by_metadata({"source": source})
        return {"deleted": source, "fileUrls": file_urls}
    except Exception as exc:
        print(f"[rag/sources delete] failed: {exc}")
        raise HTTPException(status_code=502, detail="Could not delete source") from exc


@app.post("/extract")
def extract_document(
    file: UploadFile = File(...),
    job_id: int = Form(...),
    callback_url: str = Form(...),
    bunny_storage_key: str = Form(default=""),
    bunny_storage_zone: str = Form(default="nobles"),
    bunny_cdn_host: str = Form(default="https://ziania-storage.b-cdn.net"),
    x_api_key: str = Header(default="", alias="X-API-Key"),
):
    """
    Main extraction endpoint.

    1. Reads the uploaded file into memory
    2. Extracts layout (markdown + images) using the appropriate parser
    3. Sends markdown to LLM for semantic extraction
    4. Streams images to Bunny.net CDN
    5. Posts the final structured payload to the NestJS callback URL
    """
    start_time = time.time()

    try:
        # ─── Read file into memory ──────────────────────────────────
        file_bytes = file.file.read()
        file_name = file.filename or "unknown"
        file_ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

        print(f"[Job #{job_id}] Processing: {file_name} ({len(file_bytes)} bytes)")

        # ─── Step 1: Deterministic Layout Extraction ────────────────
        if file_ext == "docx":
            markdown_text, image_buffers = extract_docx(file_bytes)
        elif file_ext == "pdf":
            markdown_text, image_buffers = extract_pdf(file_bytes)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: .{file_ext}",
            )

        print(f"[Job #{job_id}] Extracted: {len(markdown_text)} chars, {len(image_buffers)} images")

        # ─── Step 2: Upload images to Bunny.net (zero-disk) ────────
        image_url_map: dict[str, str] = {}
        if image_buffers and bunny_storage_key:
            from app.llm.structured_output import _find_question_boundaries
            boundaries = _find_question_boundaries(markdown_text)

            def slugify(text: str) -> str:
                # Remove markdown asterisks and non-alphanumeric chars
                text = text.replace('*', '')
                text = re.sub(r'[^a-zA-Z0-9\s-]', '', text).strip()
                return re.sub(r'[-\s]+', '_', text)[:60]

            # We use the filename (without extension) as the base course name
            course_base = file_name.rsplit(".", 1)[0]
            course_slug = slugify(course_base)

            new_image_buffers = {}
            for key, buffer in image_buffers.items():
                idx = key.replace("img_", "")
                placeholder = f"[[IMG_{idx}]]"
                pos = markdown_text.find(placeholder)
                
                question_slug = "context_general"
                if pos != -1 and boundaries:
                    valid_b = [b for b in boundaries if b <= pos]
                    if valid_b:
                        start_b = valid_b[-1]
                        end_b = markdown_text.find("\n", start_b)
                        if end_b == -1: end_b = len(markdown_text)
                        q_line = markdown_text[start_b:end_b].strip()
                        question_slug = slugify(q_line)
                
                # Create the new descriptive key
                new_key = f"{course_slug}_{question_slug}_{key}"
                new_image_buffers[new_key] = buffer
                
                # Update placeholder in markdown so it matches the new key
                markdown_text = markdown_text.replace(placeholder, f"[[IMG_{new_key}]]")

            image_url_map = upload_all_images(
                image_buffers=new_image_buffers,
                storage_key=bunny_storage_key,
                course_name=course_slug,
                job_id=job_id,
                storage_zone=bunny_storage_zone,
                cdn_host=bunny_cdn_host,
            )
            print(f"[Job #{job_id}] Uploaded {len(image_url_map)} images to Bunny.net")

            # Replace placeholders with actual URLs in markdown
            for key, url in image_url_map.items():
                markdown_text = markdown_text.replace(f"[[IMG_{key}]]", f"![image]({url})")

        # ─── Step 3: LLM Semantic Extraction ────────────────────────
        llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
        # Extract Groq keys if using Groq
        groq_keys = []
        if llm_provider == "groq":
            for k, v in os.environ.items():
                if k.startswith("GROQ_API_KEY") and v.strip():
                    groq_keys.append(v.strip())
            if not groq_keys:
                raise HTTPException(status_code=500, detail="No GROQ_API_KEY found")
        if llm_provider == "groq":
            llm_api_keys = groq_keys
        elif llm_provider == "gemini":
            # extract_with_gemini reads GEMINI_API_KEY itself; keep the guard
            # below meaningful by surfacing the same key here.
            llm_api_keys = [os.getenv("GEMINI_API_KEY")]
        else:
            llm_api_keys = [os.getenv("OPENAI_API_KEY")] if llm_provider == "openai" else [os.getenv("ANTHROPIC_API_KEY")]

        if not llm_api_keys or not llm_api_keys[0]:
            raise HTTPException(
                status_code=500,
                detail=f"LLM API key not configured for provider: {llm_provider}",
            )

        llm_result = extract_questions(
            markdown_text=markdown_text,
            file_name=file_name,
            provider=llm_provider,
            api_keys=llm_api_keys,
        )

        print(f"[Job #{job_id}] LLM extracted: {len(llm_result.questions)} questions, {len(llm_result.clinical_cases)} clinical cases")

        # ─── Step 4: Transform LLM output to NestJS payload ────────
        questions_by_course: dict[str, list[dict]] = {}
        clinical_case_groups: list[dict] = []

        # Map clinical cases
        case_map: dict[int, dict] = {}
        for i, case in enumerate(llm_result.clinical_cases):
            parts = [
                {
                    "type": "CASE_INTRO",
                    "description": case.intro_text,
                    "imageUrl": "",
                    "fromIndex": -1, # Set when first question is found
                }
            ]
            case_map[i] = {
                "name": case.name,
                "parts": parts,
                "questionIndices": [],
                "_last_context": case.intro_text,
            }

        # Normalize course names so they are grouped together
        if llm_result.questions:
            # Find the most common course name to prevent slight variations
            course_counts = {}
            for q in llm_result.questions:
                if q.course_name:
                    course_counts[q.course_name] = course_counts.get(q.course_name, 0) + 1
            if course_counts:
                dominant_course = max(course_counts, key=course_counts.get)
                for q in llm_result.questions:
                    if q.course_name:
                        q.course_name = dominant_course

        # Map questions
        global_index = 0
        for q in llm_result.questions:
            course = q.course_name or "Non classé"
            if course not in questions_by_course:
                questions_by_course[course] = []

            # Resolve image URLs in question
            question_image_url = ""
            if q.is_clinical_case_child and q.clinical_case_id is not None:
                if q.clinical_case_id in case_map:
                    cmap = case_map[q.clinical_case_id]
                    if not cmap["questionIndices"]:
                        # This is the first question of the case, set intro fromIndex
                        cmap["parts"][0]["fromIndex"] = global_index
                    
                    # Dynamically detect CASE_UPDATEs from accumulated context
                    current_ctx = (q.context or "").strip()
                    last_ctx = cmap["_last_context"].strip()
                    
                    # Normalize for comparison
                    def _normalize(t):
                        return "".join(t.split()).lower()
                        
                    norm_current = _normalize(current_ctx)
                    norm_last = _normalize(last_ctx)
                    
                    if norm_current and norm_current != norm_last:
                        new_text = current_ctx.lstrip("\n -+*")

                        if new_text:
                            # Prevent duplicates using normalized text
                            norm_new = _normalize(new_text)
                            already_exists = False
                            for p in cmap["parts"]:
                                norm_p = _normalize(p["description"])
                                # If one is a large substring of the other, consider it a duplicate
                                if norm_new in norm_p or norm_p in norm_new:
                                    already_exists = True
                                    break
                                    
                            if not already_exists:
                                cmap["parts"].append({
                                    "type": "CASE_UPDATE",
                                    "description": new_text,
                                    "imageUrl": "",
                                    "fromIndex": global_index,
                                })
                                print(f"  → ADDED CASE_UPDATE: {new_text[:80]}...")
                        cmap["_last_context"] = current_ctx
                    
                    cmap["questionIndices"].append(global_index)

            # Build the question payload matching NestJS DTO
            question_payload = {
                "type": q.type.value,
                "description": q.description,
                "choices": [
                    {
                        "label": ch.label,
                        "text": ch.text,
                        "isCorrect": ch.is_correct,
                    }
                    for ch in q.choices
                ],
                "propositions": [
                    {
                        "label": p.label,
                        "text": p.text,
                        "isCorrect": p.is_correct,
                    }
                    for p in q.propositions
                ] if q.propositions else [],
                "isKtype": q.is_ktype,
                "correctAnswers": q.correct_answers,
                "explanation": q.explanation,
                "explanationUrls": [],
                "imageUrl": question_image_url,
                "context": q.context,
                "courseName": course,
                "whereIsMentioned": q.where_is_mentioned,
                "indication": q.indication,
                "logicType": q.logic_type.value if q.logic_type else None,
            }

            questions_by_course[course].append(question_payload)
            global_index += 1

        # Collect clinical case groups
        clinical_case_groups = []
        for v in case_map.values():
            if v["questionIndices"]:
                # Strip out internal state to satisfy NestJS strict validation
                group = {
                    "name": v["name"],
                    "parts": v["parts"],
                    "questionIndices": v["questionIndices"],
                }
                clinical_case_groups.append(group)

        elapsed_ms = (time.time() - start_time) * 1000

        # ─── Step 5: POST result to NestJS callback ─────────────────
        callback_payload = {
            "jobId": job_id,
            "questions": questions_by_course,
            "clinicalCaseGroups": clinical_case_groups,
            "metadata": {
                "totalQuestions": global_index,
                "totalImages": len(image_url_map),
                "parsingDurationMs": round(elapsed_ms, 2),
            },
        }

        ingestion_api_key = os.getenv("INGESTION_API_KEY", "")

        try:
            cb_response = requests.post(
                callback_url,
                json=callback_payload,
                headers={"X-API-Key": ingestion_api_key},
                timeout=30,
            )
            print(f"[Job #{job_id}] Callback response: {cb_response.status_code}")
            if cb_response.status_code >= 400:
                print(f"[Job #{job_id}] Callback error details: {cb_response.text}")
        except Exception as cb_err:
            print(f"[Job #{job_id}] Callback failed: {cb_err}")
            # Don't fail the request — the payload is still returned

        return {
            "success": True,
            "jobId": job_id,
            "totalQuestions": global_index,
            "totalImages": len(image_url_map),
            "parsingDurationMs": round(elapsed_ms, 2),
        }

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[Job #{job_id}] ERROR: {error_msg}")
        traceback.print_exc()

        # Try to notify NestJS of failure
        try:
            ingestion_api_key = os.getenv("INGESTION_API_KEY", "")
            requests.post(
                callback_url,
                json={
                    "jobId": job_id,
                    "error": error_msg,
                },
                headers={"X-API-Key": ingestion_api_key},
                timeout=10,
            )
        except Exception:
            pass

        raise HTTPException(status_code=500, detail=error_msg)
