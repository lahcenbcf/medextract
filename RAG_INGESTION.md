# Local Knowledge Base — RAG Ingestion Runbook

How the Local Knowledge Base is chunked, embedded, stored and queried, and how
to run it. Implements **Part 6.2** of `implementations/04_Phase3_correction.md`.

**Design goal:** ingestion is *not* a one-off script. An admin adds a document
from the frontend UI, the request returns **immediately**, and the heavy work
(chunk → enrich → embed → store) runs in the **background**. The UI polls a job
status. Adding documents later is the normal path, not a migration.

---

## 1. Architecture

```
Admin UI (NoblesQcm)
      │  POST /ingestion/kb/documents  (multipart: file + year/module/course/type)
      ▼
z_api (NestJS gateway, :8000)          ← CORS-allowed origin for the browser
      │  forwards multipart → MEDEXTRACT_SERVICE_URL
      ▼
medextract (FastAPI, :8001)
      │  POST /rag/ingest → returns { jobId, status: "queued" }  (202, instant)
      │  BackgroundTasks:
      │     extract text (PyMuPDF / python-docx)
      │        ↓
      │     semantic recursive chunking  → PARENT chunks (\n\n → \n → ". ")
      │        ↓
      │     1–2 sentence CHILD chunks (parent kept alongside)
      │        ↓
      │     contextual enrichment  → "[Module: X | Course: Y] <chunk>"
      │        ↓
      │     Gemini embeddings (text-embedding-004, 768-dim)
      │        ↓
      ▼     Qdrant upsert (vector = child, payload = child + parent + metadata)
Qdrant (:6333)
```

### The four strategies (as fixed in the plan)

| # | Strategy | Where | Notes |
|---|----------|-------|-------|
| 1 | **Semantic (recursive) chunking** | `app/rag/chunking.py` | Splits on natural boundaries `\n\n` → `\n` → `". "`, never fixed token counts. `PARENT_MAX_CHARS = 1500`. |
| 2 | **Parent-child (auto-merging)** | `app/rag/chunking.py` + `ingest.py` | Embeds tiny 1–2 sentence **children** for precise vector match; retrieval returns the surrounding **parent** section for full clinical context. |
| 3 | **Contextual enrichment** | `app/rag/enrichment.py` | Prepends `[Module: … \| Course: … \| Year: … \| Type: …]` before embedding. Optional LLM one-liner via `RAG_LLM_ENRICH=1`. |
| 4 | **Strict metadata filtering** | `app/rag/vector_store.py` | Qdrant applies a `must` filter on `year`/`module`/`course`/`type`/`source` **during** search, so a 3rd-year chunk can never surface for a 6th-year query. |

> **Why enriched text is embedded but original text is stored:** the enrichment
> only exists to make the *vector* carry its subject. The LLM receives the clean
> original parent section.

---

## 2. Prerequisites

```bash
# 1. Start Qdrant (defined in the root docker-compose.yml)
docker compose up -d qdrant          # exposes :6333, data in the qdrant_data volume

# 2. Install the new Python deps
cd medextract && pip install -r requirements.txt   # adds langgraph, qdrant-client
```

Set in `medextract/.env`:

```ini
GEMINI_API_KEY=<your key>          # REQUIRED — embeddings + enrichment
GEMINI_EMBED_MODEL=text-embedding-004
QDRANT_URL=http://qdrant:6333      # use http://localhost:6333 if running uvicorn on the host
QDRANT_API_KEY=
QDRANT_COLLECTION=nobles_kb
RAG_LLM_ENRICH=0                   # 1 = also add an LLM-generated situating sentence
```

The collection is created automatically on first ingest (768-dim, cosine).

Run the service:

```bash
docker compose up -d medextract
# or, on the host:
cd medextract && uvicorn app.main:app --reload --port 8001
```

---

## 3. Adding documents (the normal path — via the UI/API)

### Through the gateway (what the frontend calls)

```bash
curl -X POST http://localhost:8000/ingestion/kb/documents \
  -F "file=@cardio_antihypertenseurs.pdf" \
  -F "year=3" \
  -F "module=Cardiology" \
  -F "course=Antihypertensives" \
  -F "type=course"
# → { "jobId": "a1b2…", "status": "queued", "metadata": { … } }
```

Poll until done:

```bash
curl http://localhost:8000/ingestion/kb/status/a1b2…
# → { "status": "processing" }
# → { "status": "done", "chunks": 142, "parents": 23 }
# → { "status": "failed", "error": "…" }
```

### Directly against medextract (debugging)

```bash
curl -X POST http://localhost:8001/rag/ingest \
  -F "file=@cours.docx" -F "year=3" -F "module=Cardiology" -F "type=course"

curl http://localhost:8001/rag/status/<jobId>
```

Raw text instead of a file (no upload needed):

```bash
curl -X POST http://localhost:8001/rag/ingest \
  -F "text=Les IEC provoquent une toux sèche…" \
  -F "year=3" -F "module=Cardiology" -F "course=Antihypertensives"
```

**Re-ingesting is safe.** `ingest_text(..., replace=True)` deletes prior chunks
with the same `source` before inserting, so re-uploading an updated document
replaces it instead of duplicating it.

---

## 4. Querying (metadata-filtered + auto-merged)

```bash
curl -X POST http://localhost:8001/rag/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"effet secondaire des IEC","year":"3","module":"Cardiology","topK":5}'
```

Returns parent sections, deduped and ranked:

```json
{ "results": [
    { "score": 0.87,
      "matched_chunk": "La toux sèche est l'effet indésirable le plus fréquent.",
      "context": "<the whole parent section, for the LLM>",
      "year": "3", "module": "Cardiology", "course": "Antihypertensives",
      "type": "course", "source": "cardio_antihypertenseurs.pdf" } ] }
```

Any of `year` / `module` / `course` / `type` may be omitted; every one you pass
is applied as a **hard filter** before similarity ranking.

---

## 5. API reference

| Method | Route | Service | Purpose |
|--------|-------|---------|---------|
| POST | `/ingestion/kb/documents` | z_api | Frontend entry point. Multipart file + metadata → 202 + `jobId`. |
| GET | `/ingestion/kb/status/:jobId` | z_api | Poll ingestion progress. |
| POST | `/rag/ingest` | medextract | Background ingest (file **or** `text`) → `jobId`. |
| GET | `/rag/status/{job_id}` | medextract | `queued` → `processing` → `done` \| `failed`. |
| POST | `/rag/search` | medextract | Metadata-filtered, auto-merged retrieval. |

---

## 6. Tuning

All in `app/rag/chunking.py`:

| Knob | Default | Effect |
|------|---------|--------|
| `PARENT_MAX_CHARS` | `1500` | Bigger = more context per hit, less precise filtering. |
| `CHILD_SENTENCES` | `2` | Sentences per embedded child. Lower = more precise, more vectors. |
| `SEPARATORS` | `["\n\n", "\n", ". "]` | Boundary priority for the recursive splitter. |

---

## 7. How the correction chat consumes the KB

The `[📚 Local Knowledge Base]` toggle in the chat panel is live. When enabled,
the LangGraph in `app/llm/correction_graph.py` branches through a retrieval
node before generating:

```
entry ─┬─ (use_local_kb) ─→ retrieve ─→ generate ─→ END
       └────────────────────────────────→ generate ─→ END
```

- **z_api** looks up the exam and forwards `useLocalKb` plus
  `metadata: { module, year }` taken from the exam record.
- **`_retrieve_context`** calls `retrieve()` with that metadata, so an exam can
  only ever be grounded on its own module/year material.
- **`_generate`** appends the retrieved parent sections to the system prompt
  under a `# LOCAL KNOWLEDGE BASE CONTEXT` heading, instructing the model to
  prefer them and to say so when they don't cover the question.
- Retrieval is **best-effort**: if Qdrant is down or empty, the node logs and
  returns empty context so the chat still answers (ungrounded) instead of
  failing.

> ⚠️ **Tag values must match exactly.** Retrieval filters on the exam's
> `module.name` and `year.year`. The frontend KB upload form
> (`/dashboard/knowledge-base`) picks year → module → course from the same
> lists the exam flow uses (courses come from `/admin/getCoursPerModule/:id`)
> for this reason — free text or a different label is invisible to the chat.
>
> All three tags are **optional**: an untagged document is still chunked,
> embedded and searchable via `/rag/search`, but the exam-scoped chat won't
> retrieve it, because the `must` filter on module+year can't match a document
> that has no such field. The upload form warns about this inline.

---

## 8. Known limits / next steps

- **Job registry is in-memory** (`app/rag/jobs.py`), so it's per-process. With
  multiple uvicorn workers or replicas, a poll can hit a worker that doesn't
  know the job. Move it to Redis/Postgres before scaling out — the payload is
  deliberately small and serialisable so the swap is mechanical.
- **The KB job list is per-session.** `/dashboard/knowledge-base` tracks job ids
  in page state, so a refresh clears the *list* (the indexed documents persist
  in Qdrant). A "documents in the KB" listing needs a Qdrant scroll endpoint or
  a documents table in Postgres.
- **No delete-from-UI yet.** `delete_by_metadata()` exists in
  `vector_store.py` (and re-ingesting the same `source` replaces it), but it
  isn't exposed as a route.
- `RAG_LLM_ENRICH=1` costs one LLM call per child chunk — fine for a few
  documents, slow for bulk. Batch it if you enable it broadly.
