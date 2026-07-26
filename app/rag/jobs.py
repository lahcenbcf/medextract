"""
Lightweight in-process job registry for background KB ingestion.

Keeps the admin UI responsive: /rag/ingest returns a job_id immediately and the
UI polls /rag/status/{job_id}.

NOTE: this is in-memory, so it is per-process. With multiple uvicorn workers or
replicas, move this to Redis/Postgres (the payload is intentionally tiny and
serialisable so that swap is mechanical).
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Dict, Optional

_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}


def create_job(metadata: Dict[str, Any]) -> str:
    job_id = uuid.uuid4().hex
    with _lock:
        _jobs[job_id] = {
            "id": job_id,
            "status": "queued",  # queued → processing → done | failed
            "metadata": metadata,
            "chunks": 0,
            "parents": 0,
            "error": None,
            "createdAt": time.time(),
            "updatedAt": time.time(),
        }
    return job_id


def update_job(job_id: str, **fields: Any) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job:
            job.update(fields)
            job["updatedAt"] = time.time()


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None
