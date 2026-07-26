# MedExtract-IA — Execution Pipeline & Testing Guide

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│  Admin Panel │────▶│   NestJS API     │────▶│  medextract (Docker) │
│  (Next.js)   │     │  POST /ingestion │     │  FastAPI :8001       │
│              │     │    /upload       │     │                      │
└──────────────┘     └──────┬───────────┘     └──────────┬───────────┘
                            │                            │
                     ┌──────▼──────┐              ┌──────▼──────────┐
                     │  BullMQ     │              │ 1. DOCX/PDF     │
                     │  (Redis)    │              │    extractor    │
                     │  'ingestion'│              │ 2. LLM layer    │
                     │  queue      │              │ 3. Bunny.net    │
                     └─────────────┘              │    uploader     │
                                                  └──────┬──────────┘
                                                         │
                                              ┌──────────▼──────────┐
                                              │ POST /ingestion/    │
                                              │   ingest-payload    │
                                              │ (callback to NestJS)│
                                              └──────────┬──────────┘
                                                         │
                                              ┌──────────▼──────────┐
                                              │ Admin Reviews       │
                                              │ GET /ingestion/     │
                                              │   review/:jobId     │
                                              └──────────┬──────────┘
                                                         │
                                              ┌──────────▼──────────┐
                                              │ POST /ingestion/    │
                                              │   approve/:jobId    │
                                              │ → Bulk DB Insert    │
                                              └─────────────────────┘
```

---

## Status: Service Health

```bash
# Check all containers
docker compose ps

# Check medextract health
docker exec medextract_service python -c \
  "import requests; r = requests.get('http://localhost:8001/health'); print(r.json())"
# Expected: {'status': 'healthy', 'service': 'medextract-ia'}
```

---

## Test 1: Standalone DOCX Extraction (No LLM)

Test just the layout extractor without the LLM to verify text + image extraction works.

```bash
# Create a minimal test .docx inside the container
docker exec medextract_service python -c "
from docx import Document
from docx.shared import Pt
import io

doc = Document()
doc.add_heading('QCM - Module Endocrinologie', level=1)

# Question 1 with bold
p = doc.add_paragraph()
run1 = p.add_run('1. Parmi les propositions suivantes concernant le diabète de type 2, laquelle est ')
run2 = p.add_run('FAUSSE')
run2.bold = True
run3 = p.add_run(' ?')

# Choices
doc.add_paragraph('A. Insulinorésistance centrale')
doc.add_paragraph('B. Metformine en 1ère intention')
doc.add_paragraph('C. Survient exclusivement après 60 ans')
doc.add_paragraph('D. HbA1c cible < 7%')
doc.add_paragraph('E. Obésité facteur de risque')

# Question 2 — Clinical case
doc.add_heading('Cas Clinique 1', level=2)
p2 = doc.add_paragraph('Patient de 55 ans, hypertendu sous amlodipine 5mg, diabétique de type 2 sous metformine.')
p3 = doc.add_paragraph('Glycémie à jeun: 2.3 g/L, HbA1c: 9.2%')

doc.add_paragraph('Question 1: Quel est le diagnostic le plus probable ?')
doc.add_paragraph('A. Diabète décompensé')
doc.add_paragraph('B. Hypoglycémie')
doc.add_paragraph('C. Acidocétose diabétique')
doc.add_paragraph('D. Coma hyperosmolaire')

doc.save('/tmp/test_qcm.docx')
print('✅ Test DOCX created: /tmp/test_qcm.docx')
"
```

```bash
# Test the extractor directly
docker exec medextract_service python -c "
from app.extractors.docx_extractor import extract_docx

with open('/tmp/test_qcm.docx', 'rb') as f:
    file_bytes = f.read()

markdown, images = extract_docx(file_bytes)
print('=== EXTRACTED MARKDOWN ===')
print(markdown)
print()
print(f'=== IMAGES FOUND: {len(images)} ===')
for key in images:
    print(f'  {key}: {images[key].getbuffer().nbytes} bytes')
"
```

**Expected output:**
- The word `FAUSSE` should appear wrapped in `**FAUSSE**`
- Clinical case text should be preserved
- Choice labels (A, B, C, D, E) should be intact

---

## Test 2: Full Pipeline via HTTP (with LLM)

> [!WARNING]  
> This test requires a valid `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` env var 
> set inside the medextract container.

```bash
# Set the LLM API key (choose one)
docker exec medextract_service bash -c 'echo "export OPENAI_API_KEY=sk-..." >> /etc/environment'

# Or pass it via docker-compose env_file (.env.production should have):
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-...
# INGESTION_API_KEY=your-secret-key
```

```bash
# Full pipeline test via the /extract endpoint
docker exec medextract_service python -c "
import requests, json

# Read the test file
with open('/tmp/test_qcm.docx', 'rb') as f:
    file_bytes = f.read()

# Call the extract endpoint (use a fake callback URL since we just want the response)
response = requests.post(
    'http://localhost:8001/extract',
    files={'file': ('test_qcm.docx', file_bytes)},
    data={
        'job_id': '999',
        'callback_url': 'http://localhost:9999/fake-callback',  # will fail silently
        'bunny_storage_key': '',
        'bunny_storage_zone': 'nobles',
        'bunny_cdn_host': 'https://ziania-storage.b-cdn.net',
    },
)

print(f'Status: {response.status_code}')
print(json.dumps(response.json(), indent=2, ensure_ascii=False))
"
```

---

## Test 3: NestJS End-to-End (Full Stack)

> [!IMPORTANT]  
> This requires the NestJS backend running with Prisma migration applied.

### Step 1: Apply the Prisma migration

```bash
cd /home/lahcen/ziania/z_api
npx prisma db push
# or
npx prisma migrate dev --name add-ingestion-job
```

### Step 2: Start the NestJS backend

```bash
cd /home/lahcen/ziania/z_api
npm run start:dev
```

### Step 3: Upload a file via the NestJS endpoint

```bash
curl -X POST http://localhost:8000/ingestion/upload \
  -F "file=@/path/to/your/exam.docx" \
  -F "yearId=1" \
  -F "moduleId=5"
```

**Expected response:**
```json
{
  "jobId": 1,
  "status": "PENDING",
  "message": "File \"exam.docx\" queued for processing"
}
```

### Step 4: Monitor job status

```bash
# Poll until status changes from PENDING → PROCESSING → PARSED
curl http://localhost:8000/ingestion/status/1
```

### Step 5: Review parsed questions

```bash
curl http://localhost:8000/ingestion/review/1 | python3 -m json.tool
```

### Step 6: Approve (insert into DB)

```bash
curl -X POST http://localhost:8000/ingestion/approve/1
```

**Expected:** Questions are now in the `Question` table with `sourceFile` set to the uploaded filename.

---

## Environment Variables Required

Add these to `.env.production` (or z_api `.env`):

```env
# ── MedExtract-IA ─────────────────────────────────
MEDEXTRACT_SERVICE_URL=http://medextract:8001
INGESTION_API_KEY=change-me-to-a-secure-random-string

# LLM (choose one provider)
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...

# Bunny.net (already existing)
BUNNY_STORAGE_API_KEY=<existing>
BUNNY_ZONE_STORAGE=nobles
CDN_HOST=https://ziania-storage.b-cdn.net
```

---

## Docker Commands Reference

```bash
# Build + start the medextract service
docker compose up -d --build medextract

# View medextract logs
docker compose logs -f medextract

# Restart medextract after code changes
docker compose restart medextract

# Rebuild after requirements.txt change
docker compose build --no-cache medextract && docker compose up -d medextract

# Shell into the container for debugging
docker exec -it medextract_service bash
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `500: LLM API key not configured` | Missing env var | Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in `.env.production` |
| `Connection refused` to medextract | Container not on same network | Verify `docker network inspect ziania_ziania-network` |
| `FAILED` job status | Python extraction error | Check `docker compose logs medextract` |
| `parsedPayload is null` in review | Callback URL wrong | Ensure `BACKEND_URL` resolves inside Docker network |
| Large file timeout | File > 20MB / complex PDF | Increase `timeout` in `ingestion.processor.ts` |
