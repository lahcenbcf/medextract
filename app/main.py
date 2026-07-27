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
from app.rag.ingest import ingest_text, retrieve
from app.rag.jobs import create_job, get_job, list_jobs, update_job
from app.rag.vector_store import delete_by_metadata, list_sources


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
):
    """
    Add a document to the Local Knowledge Base.

    Accepts either an uploaded .pdf/.docx or raw `text`, plus the metadata used
    for strict filtering at retrieval time. Returns immediately with a jobId —
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
        delete_by_metadata({"source": source})
        return {"deleted": source}
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
