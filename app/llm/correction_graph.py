"""
Correction chat pipeline for the "Correction" stage of the NoblesQCM pipeline.

z_api (the NestJS gateway) proxies admin chat requests here. This module runs a
LangGraph state graph:

  entry ─┬─(use_local_kb)→ retrieve ─→ generate ─→ classify ─→ END
         └───────────────────────────→ generate ─→ classify ─→ END

  - Web-Search mode binds Google Search grounding and returns grounded markdown
    (Gemini forbids Search + a forced JSON schema in the same call). No classify.
  - Otherwise the admin may paste SEVERAL questions; ``generate`` returns a LIST
    of strictly-typed question objects, and ``classify`` suggests the best-fitting
    course for each from a candidate list supplied by z_api.

Gemini is called over raw REST (``requests``) to avoid pinning a LangChain
provider version; LangGraph orchestrates so the graph can grow later.
"""

from __future__ import annotations

import os
import re
import json
import textwrap
import time
from typing import Any, Dict, List, Optional, TypedDict

import requests
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)
from langgraph.graph import StateGraph, END

from app.rag.ingest import retrieve_many


GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"
# A PER-QUESTION budget: every question in a paste gets its own search.
#
# ONE. Three became two when the third slot started holding a figure whose whole
# content was "Illustration © mgundj"; two became one because per-question
# retrieval made the first hit reliably the right one, and every extra section
# is prose the model must read before answering.
#
# This is the tightest setting that still grounds an answer, and it has no
# margin: if the best section is wrong, the question has nothing else to fall
# back on. Raise it with RAG_TOP_K=2 — no redeploy needed — if answers start
# citing the right page but the wrong passage.
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "1"))
# What the model finally receives, across every question. Without a ceiling a
# ten-question paste would carry thirty sections into the prompt and give back
# in prefill what per-question retrieval gains in accuracy.
RAG_MAX_CHUNKS = int(os.getenv("RAG_MAX_CHUNKS", "8"))
# Beyond this many questions in one paste, retrieval stops splitting: the tail
# questions would get one chunk each at best, and the caps above bind anyway.
RAG_MAX_QUESTIONS = int(os.getenv("RAG_MAX_QUESTIONS", "8"))


# ─── Debug console ─────────────────────────────────────────────────────
# One place for everything this pipeline writes to stdout, so
# `docker compose logs -f medextract` reads as a trace rather than as scattered
# prints. Set CORRECTION_DEBUG=0 to silence it without touching the code.
#
# `flush=True` throughout: Python buffers stdout when it is not a TTY, and under
# Docker that means the trace of a request arrives long after the request did.
DEBUG = os.getenv("CORRECTION_DEBUG", "1").lower() not in ("0", "false", "no")
_WIDTH = 78


def _log(msg: str = "") -> None:
    if DEBUG:
        print(msg, flush=True)


def _rule(title: str = "", char: str = "─") -> None:
    """A separator, optionally carrying a title."""
    if not DEBUG:
        return
    if not title:
        print(char * _WIDTH, flush=True)
        return
    head = f"{char * 3} {title} "
    print(head + char * max(3, _WIDTH - len(head)), flush=True)


def _field(label: str, value: Any, limit: int = 0) -> None:
    """
    A labelled block of text, whitespace-collapsed, wrapped and indented.

    Collapsing the whitespace matters: chunk text is full of newlines, and one
    section printed raw swamps the twenty lines around it.
    """
    if not DEBUG:
        return
    raw = "" if value is None else str(value)
    text = " ".join(raw.split())
    if limit and len(text) > limit:
        text = f"{text[:limit]}… (+{len(text) - limit} car.)"
    print(f"  {label}", flush=True)
    for line in textwrap.wrap(text, _WIDTH - 6) or ["(vide)"]:
        print(f"      {line}", flush=True)


# ─── Pydantic models (structured-output validation) ────────────────────
class CorrectionChoice(BaseModel):
    # Lenient: in web-search mode (no enforced schema) the model often omits
    # the label and embeds it in the text — we normalise it after parsing.
    # Tolerate model drift: some models emit `letter` instead of `label`.
    model_config = ConfigDict(populate_by_name=True)
    label: str = Field("", validation_alias=AliasChoices("label", "letter"))
    text: str = ""
    isCorrect: bool = False
    # Amboss-style per-choice commentary (why this option is true/false, + source).
    explanation: str = ""
    source: str | None = None
    sourceRef: dict | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data):
        """Tolerate model drift: None → "" for strings, and the common case where
        the model puts the WHOLE option in `label`/`letter` with no `text`."""
        if not isinstance(data, dict):
            return data
        d = dict(data)
        for k in ("label", "letter", "text", "explanation", "source", "sourceRef"):
            if d.get(k) is None:
                d[k] = "" if k not in ("source", "sourceRef") else None
        if not (d.get("text") or "").strip():
            alt = (d.get("label") or d.get("letter") or "").strip()
            if alt:
                d["text"] = alt  # _normalize_labels will re-extract the letter
                d["label"] = ""
                d.pop("letter", None)
        return d


class CorrectionQuestion(BaseModel):
    description: str
    isKtype: bool = False
    choices: List[CorrectionChoice] = Field(default_factory=list)
    propositions: List[CorrectionChoice] | None = None
    correctAnswers: str = ""
    # Reviewer-facing ALERTS only ("⚠️ CONFLIT CLÉ", "⚠️ AMBIGUË"…) — the
    # per-option commentary lives on each choice, the student-facing recap in
    # `globalComment`.
    explanation: str = ""
    # Student-facing recap of the whole question. Any comparative table is
    # emitted as MARKDOWN and converted to HTML at the display boundary
    # (NoblesQcm/lib/html.ts) — HTML here would cost tokens for nothing.
    globalComment: str = ""
    # Clinical case: the shared/preceding context for THIS question (patient
    # intro, new lab results…) — empty for a standalone question — and the
    # question's index within the case.
    caseDescription: str = ""
    caseIndex: Optional[int] = None
    # Résidanat only: which of the 3 fixed épreuves this question belongs to —
    # "biologie" | "chirurgie" | "medecine". Empty for non-résidanat exams.
    epreuve: str = ""

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data):
        """Tolerate model drift so a single odd field never drops the question:
        None → "" for string fields, and `correctAnswers` as a list → "A,B"."""
        if not isinstance(data, dict):
            return data
        d = dict(data)
        for k in (
            "description",
            "explanation",
            "globalComment",
            "caseDescription",
            "epreuve",
            "correctAnswers",
        ):
            if d.get(k) is None:
                d[k] = ""
        ca = d.get("correctAnswers")
        if isinstance(ca, list):
            d["correctAnswers"] = ",".join(
                str(x).strip() for x in ca if str(x).strip()
            )
        return d


# Gemini responseSchema (OpenAPI 3.0 subset) for ONE question object.
QUESTION_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "description": {"type": "STRING"},
        "isKtype": {"type": "BOOLEAN"},
        "choices": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "label": {"type": "STRING"},
                    "text": {"type": "STRING"},
                    "isCorrect": {"type": "BOOLEAN"},
                    "explanation": {"type": "STRING"},
                },
                "required": ["label", "text", "isCorrect", "explanation"],
            },
        },
        "propositions": {
            "type": "ARRAY",
            "nullable": True,
            "items": {
                "type": "OBJECT",
                "properties": {
                    "label": {"type": "STRING"},
                    "text": {"type": "STRING"},
                    "isCorrect": {"type": "BOOLEAN"},
                    "explanation": {"type": "STRING"},
                    "source": {"type": "STRING", "nullable": True},
                },
                "required": ["label", "text", "isCorrect", "explanation"],
            },
        },
        "correctAnswers": {"type": "STRING"},
        # Reviewer alerts only; the per-option commentary is on each choice.
        "explanation": {"type": "STRING"},
        # Student-facing recap. Comparative tables are Markdown (converted to HTML
        # for display); see toDisplayHtml in NoblesQcm/lib/html.ts.
        "globalComment": {"type": "STRING"},
        "caseDescription": {"type": "STRING"},
        "caseIndex": {"type": "INTEGER", "nullable": True},
        # Résidanat only: "biologie" | "chirurgie" | "medecine" (empty otherwise).
        "epreuve": {"type": "STRING", "nullable": True},
    },
    # `caseIndex` is required-but-nullable ON PURPOSE: the model must decide,
    # for every question, whether it belongs to a clinical case (0, 1, 2…) or
    # stands alone (null). Since the case text is now written only once and
    # redistributed afterwards, that flag is the ONLY membership signal — left
    # optional, a model that simply omitted it would strand the later questions
    # of a case with no context at all. Three tokens per question for a signal
    # we cannot reconstruct.
    "required": [
        "description",
        "isKtype",
        "choices",
        "correctAnswers",
        "explanation",
        "caseIndex",
    ],
}

# The admin may paste several questions at once → return a LIST.
QUESTIONS_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "ARRAY",
    "items": QUESTION_RESPONSE_SCHEMA,
}

# Classification: one entry per question, in the same order.
CLASSIFY_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "suggestedCourId": {"type": "STRING", "nullable": True},
            "suggestedCourName": {"type": "STRING"},
            "confidence": {"type": "NUMBER"},
        },
        "required": ["suggestedCourName"],
    },
}


# ─── LangGraph state ───────────────────────────────────────────────────
class CorrectionState(TypedDict, total=False):
    message: str
    history: List[Dict[str, str]]
    system_prompt: str
    use_web_search: bool
    use_local_kb: bool
    metadata: Dict[str, Any]
    candidate_courses: List[Dict[str, Any]]  # [{id, name, group?}]
    allow_new_course: bool  # False for residanat (existing courses only)
    context: str
    kb_sources: List[Dict[str, Any]]  # retrieved KB docs [{source, course, snippet}]
    questions: List[Dict[str, Any]]  # structured output (+ suggestions after classify)
    reply: str  # web-search markdown (structured mode leaves this empty)
    # Clinical-case memory (partial submissions)
    clinical_case: bool
    case_context: str  # remembered patient context (in), updated by _detect_case (out)
    is_new_case: bool  # set by _detect_case
    model: str  # Gemini model id (from z_api's global setting); env fallback if absent


_RETRYABLE = {429, 500, 502, 503, 504}


def _post_gemini(url: str, body: Dict[str, Any], timeout: int = 110):
    """POST to Gemini with ONE retry on transient failures (timeout / 5xx / 429)."""
    last_exc = None
    for attempt in range(2):
        try:
            resp = requests.post(
                url, json=body, headers={"Content-Type": "application/json"},
                timeout=timeout,
            )
            if resp.status_code in _RETRYABLE and attempt == 0:
                last_exc = requests.HTTPError(f"HTTP {resp.status_code}")
                time.sleep(1.5)
                continue
            resp.raise_for_status()
            return resp
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            if attempt == 0:
                time.sleep(1.5)
                continue
            raise
    raise last_exc  # type: ignore[misc]


def thinking_config(model: str) -> Optional[Dict[str, Any]]:
    """
    Cap the thinking budget on the models that have one, nothing elsewhere.

    Correction latency is dominated by the dead time BEFORE the first output
    token, and `gemini-flash-latest` — the default — thinks by default: measured
    against the live API, a trivial "Dis OK." prompt burned 248 thinking tokens
    unprompted, 41 with a zero budget. On a full correction prompt that gap is
    what the corrector spends staring at a spinner.

    Two guards, both measured rather than assumed:

      * The 1.5/2.0 series has no thinking mode — the field is meaningless there.
      * Pro REFUSES a zero budget: `gemini-pro-latest` answers
        `400 Budget 0 is invalid. This model only works in thinking mode.`
        128 is its floor and still roughly halves what it spends (423 → 228).

    Without the Pro guard, correction would break outright the moment an admin
    switched model in the settings.
    """
    if "2.5" not in model and "latest" not in model:
        return None
    return {"thinkingBudget": 128 if "pro" in model else 0}


def _gemini_generate(
    system_prompt: str,
    contents: List[Dict[str, Any]],
    *,
    web_search: bool = False,
    response_schema: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
) -> str:
    """Low-level Gemini REST call. Returns the concatenated text of part 0..n."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not configured")
    model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    body: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": {"temperature": 0.2},
    }
    thinking = thinking_config(model)
    if thinking:
        body["generationConfig"]["thinkingConfig"] = thinking
    _log(
        f"  → Gemini · {model} · thinking="
        f"{thinking['thinkingBudget'] if thinking else 'défaut'}"
        f"{' · web_search' if web_search else ''}"
        f"{' · schéma JSON' if response_schema is not None else ''}"
    )
    if system_prompt:
        body["system_instruction"] = {"parts": [{"text": system_prompt}]}
    if web_search:
        body["tools"] = [{"google_search": {}}]
    elif response_schema is not None:
        body["generationConfig"]["responseMimeType"] = "application/json"
        body["generationConfig"]["responseSchema"] = response_schema

    resp = _post_gemini(
        f"{GEMINI_ENDPOINT}/{model}:generateContent?key={api_key}", body
    )
    data = resp.json()
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts)


def _gemini_grounded(
    system_prompt: str,
    contents: List[Dict[str, Any]],
    *,
    model: Optional[str] = None,
):
    """Grounded (Google Search) call. Returns (text, sources) where sources are
    the REAL citation URLs from groundingMetadata — which the plain text drops."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not configured")
    model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    body: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": {"temperature": 0.2},
        "tools": [{"google_search": {}}],
    }
    thinking = thinking_config(model)
    if thinking:
        body["generationConfig"]["thinkingConfig"] = thinking
    if system_prompt:
        body["system_instruction"] = {"parts": [{"text": system_prompt}]}

    resp = _post_gemini(
        f"{GEMINI_ENDPOINT}/{model}:generateContent?key={api_key}", body
    )
    cand = resp.json().get("candidates", [{}])[0]
    text = "".join(
        p.get("text", "") for p in cand.get("content", {}).get("parts", [])
    )
    sources: List[Dict[str, str]] = []
    for ch in (cand.get("groundingMetadata", {}) or {}).get("groundingChunks", []) or []:
        web = ch.get("web") or {}
        if web.get("uri"):
            sources.append({"title": web.get("title") or web["uri"], "uri": web["uri"]})
    return text, sources


def _retrieve_context(state: CorrectionState) -> CorrectionState:
    """
    RAG node: metadata-filtered grounding from the Local Knowledge Base.

    One search PER question. The whole paste used to be embedded as a single
    string — after an LLM pass distilled it to keywords — so five questions
    shared one vector and one set of three chunks. That vector pointed at the
    batch's centre of gravity and therefore at none of the questions: asked how
    an atheroma plaque forms, the KB returned IDM complications, chest pain and
    rhythm disorders, which were the OTHER questions' subjects.

    The keyword-distillation pass is gone with it: it cost an LLM call on the
    critical path, truncated the original question to 200 characters, and its
    output mixed vocabulary across questions — one question's "Classification"
    was enough to divert every other question's search toward tables.
    """
    message = state["message"]
    # `_split_questions` prefers the preamble the message carries itself and
    # falls back to this one, so a partial submission ("3. …" alone) still gets
    # its patient without the two ever being prepended twice.
    remembered = state.get("case_context") or "" if state.get("clinical_case") else ""
    segments = _split_questions(message, remembered)

    _log()
    _rule("RAG · BASE DE CONNAISSANCES", "═")
    _log(f"  {len(segments)} question(s) · filtre : {state.get('metadata') or '(aucun)'}")
    _log(f"  budget : {RAG_TOP_K} extrait(s)/question, {RAG_MAX_CHUNKS} au total")
    try:
        hits = retrieve_many(
            queries=segments,
            metadata=state.get("metadata") or {},
            top_k=RAG_TOP_K,
            max_total=RAG_MAX_CHUNKS,
        )
    except Exception as exc:  # noqa: BLE001 - never fail the chat on KB issues
        _rule("RAG ÉCHEC", "═")
        _log(f"  {exc}")
        _log("  → correction poursuivie SANS contexte")
        return {"context": ""}

    if not hits:
        _log("  → AUCUN extrait trouvé — correction sans contexte")
        _rule("", "═")
        return {"context": ""}

    blocks = []
    kb_sources: List[Dict[str, Any]] = []
    positions: Dict[int, int] = {}
    for i, hit in enumerate(hits, start=1):
        positions[id(hit)] = i
        source = hit.get("source") or "unknown"
        course = hit.get("course") or ""
        ctx = hit.get("context", "") or ""
        page = hit.get("page")
        section = hit.get("section") or ""
        # The localisation goes in the header so the model can cite it verbatim
        # — it is the only page/section it will ever legitimately know.
        header = f"[{i}] source: {source}"
        if page is not None:
            header += f" | page: {page}"
        if section:
            header += f" | section: {section}"
        if course:
            header += f" | course: {course}"
        blocks.append(f"{header}\n{ctx}")
        # Keep the top chunk of each doc so the UI can show it on hover.
        kb_sources.append(
            {
                "source": source,
                "course": course,
                "page": page,
                "section": section,
                # The URL never enters the model's context — it would just be
                # tokens. It travels to the UI only.
                "fileUrl": hit.get("file_url") or "",
                "snippet": ctx.strip()[:400],
                "bbox": hit.get("bbox"),
            }
        )

    # Grouped BY QUESTION, not by rank: the merge hands back one flat list, and
    # what a reader needs is "what did THIS question ask, and what did it get".
    for qi, seg in enumerate(segments):
        _rule(f"Q{qi + 1}/{len(segments)}")
        _field("REQUÊTE ENVOYÉE À LA KB", seg, 500)
        mine = [h for h in hits if h.get("query_index") == qi]
        if not mine:
            _log("  ⚠ aucun extrait retenu pour cette question")
            continue
        for hit in mine:
            pos = positions[id(hit)]
            page = hit.get("page")
            _log(
                f"  ── extrait [{pos}] · {hit.get('source') or 'unknown'}"
                f" · page {page} · {hit.get('kind') or 'prose'}"
            )
            # These two are DIFFERENT texts and the distinction matters: the
            # child excerpt is what the vector search scored, the parent is what
            # the model actually reads. A parent can hold the answer while the
            # excerpt that ranked it does not.
            _field("· extrait qui a matché (sert au CLASSEMENT)",
                   hit.get("matched_chunk"), 220)
            _field("· contexte ENVOYÉ AU MODÈLE (section parente)",
                   hit.get("context"), 700)

    context = "\n\n---\n\n".join(blocks)
    _rule("", "═")
    _log(f"  {len(hits)} extrait(s) retenu(s) · {len(context)} caractères de contexte")
    _rule("", "═")
    _log()
    return {"context": context, "kb_sources": kb_sources}


# ─── Clinical-case detection + memory (partial submissions) ────────────
_CASE_HEADER_RE = re.compile(
    r"^\s*(cas\s+clinique|observation\s+clinique|cas\s+n[°o]|"
    r"(?:une?\s+)?patient[e]?\b)",
    re.IGNORECASE,
)
_FIRST_Q_RE = re.compile(r"(?m)^\s*(\d{1,2})\s*[.)]\s")


def _first_question_number(message: str) -> Optional[int]:
    m = _FIRST_Q_RE.search(message)
    return int(m.group(1)) if m else None


def _extract_leading_context(message: str) -> str:
    """The text before the first 'N.' question = the (new) case context block."""
    m = _FIRST_Q_RE.search(message)
    return message[: m.start()].strip() if (m and m.start() > 0) else ""


# A question stem announces itself — a question mark, or an interrogative cue.
_Q_CUE = re.compile(
    r"\?|\b(quelles?|quels?|parmi|cochez|indiquez|citez|donnez|pr[ée]cisez|"
    r"concernant|laquelle|lequel|lesquell?es?|d[ée]finir|d[ée]finissez)\b",
    re.IGNORECASE,
)
# An answer option: "A- …", "B) …", "1. …" — a letter OR a digit label.
_OPTION_LABEL = re.compile(r"^\s*(?:[A-Za-z]|\d{1,2})\s*[.)\-–:]\s+\S")
# A LETTER label is ALWAYS an option: no question stem starts with "A- ".
_LETTER_LABEL = re.compile(r"^\s*[A-Za-z]\s*[.)\-–:]\s+\S")
# A digit label opens either a numbered question ("1. Quelle est…") or a K-type
# sub-proposition ("1. L'oxydation des LDL…"). Only the cue tells them apart.
_NUM_LABEL = re.compile(r"^\s*\d{1,2}\s*[.)]\s+")


def _is_question_start(line: str) -> bool:
    """Does this line open a new question?"""
    s = line.strip()
    if not s or _LETTER_LABEL.match(s):
        return False
    m = _NUM_LABEL.match(s)
    # Strip the number before looking for the cue, so "1. Quelle est…" reads as
    # a question while "1. L'oxydation des LDL…" reads as a sub-proposition.
    return bool(_Q_CUE.search(s[m.end() :] if m else s))


def _has_option(segment: str) -> bool:
    """Does this segment carry answer options of its own?"""
    return any(_OPTION_LABEL.match(l) for l in segment.splitlines()[1:])


def _split_questions(message: str, remembered_context: str = "") -> List[str]:
    """
    Cut a paste into one segment per question, so each gets its own search.

    Boundaries are found by CONTENT, not by numbering. Numbering was the first
    attempt and it failed on the most ordinary paste there is — questions typed
    one after another with no "1." at all. Everything then landed in a single
    query again, which is the batch-centroid problem the split exists to remove.

    Three rules, each earning its place:

      * a LETTER label ("A- …") is always an option, never a stem;
      * a DIGIT label is ambiguous, so the cue is tested on what follows it —
        that is what keeps a K-type question ("1. L'oxydation des LDL…") from
        exploding into one search per sub-proposition;
      * a real question carries options, so a stem split across two lines
        ("Concernant les marqueurs…" / "quelle proposition est exacte ?")
        merges FORWARD into the piece that does.

    `remembered_context` is the case a partial submission continues; it is used
    only when the message carries no preamble of its own, since the two overlap.

    Falls back to the whole message whenever fewer than two questions are found,
    keeping the single-question case — the common one — exactly as it was.
    """
    def whole() -> List[str]:
        if remembered_context:
            return [f"{remembered_context}\n\n{message}".strip()]
        return [message]

    lines = message.splitlines()
    starts = [i for i, l in enumerate(lines) if _is_question_start(l)]
    if len(starts) < 2:
        return whole()

    bounds = starts + [len(lines)]
    segments: List[str] = []
    pending = ""
    for i in range(len(bounds) - 1):
        piece = "\n".join(lines[bounds[i] : bounds[i + 1]]).strip()
        if not piece:
            continue
        if pending:
            piece = f"{pending}\n{piece}"
            pending = ""
        if _has_option(piece):
            segments.append(piece)
        else:
            pending = piece          # a stem still waiting for its options
    if pending:
        if segments:
            segments[-1] = f"{segments[-1]}\n{pending}"
        else:
            segments.append(pending)

    if len(segments) < 2:
        return whole()

    # The preamble before the first question — a clinical vignette's patient.
    lead = "\n".join(lines[: starts[0]]).strip() or remembered_context
    if lead:
        segments = [f"{lead}\n\n{s}" for s in segments]
    return segments[:RAG_MAX_QUESTIONS]


def _llm_is_new_case(
    prev_context: str, message: str, model: Optional[str] = None
) -> bool:
    """Small classifier: does `message` START a new clinical case?"""
    prompt = (
        "Décide si le texte suivant DÉBUTE un NOUVEAU cas clinique, ou s'il "
        "CONTINUE le cas clinique précédent (mêmes patient/contexte). Réponds "
        "STRICTEMENT par un seul mot: NEW ou CONTINUATION.\n\n"
        f"CONTEXTE PRÉCÉDENT:\n{prev_context or '(aucun)'}\n\n"
        f"NOUVEAU TEXTE:\n{message}"
    )
    try:
        out = _gemini_generate(
            "", [{"role": "user", "parts": [{"text": prompt}]}], model=model
        ).strip().upper()
        return "NEW" in out and "CONTINU" not in out
    except Exception as exc:  # noqa: BLE001 - best effort
        print(f"[correction_graph] case-detect LLM failed: {exc}")
        return not prev_context  # no memory yet → must be new


def _detect_case(state: CorrectionState) -> CorrectionState:
    """Decide NEW vs CONTINUATION (regex fast-path → LLM) and update the memory."""
    message = state["message"]
    prev = state.get("case_context") or ""
    leading = _extract_leading_context(message)
    first_q = _first_question_number(message)

    if _CASE_HEADER_RE.search(message):
        is_new = True                      # explicit "Cas clinique …" header
    elif not prev:
        is_new = True                      # nothing remembered yet
    elif first_q is not None and first_q > 1:
        is_new = False                     # starts at "3." → clearly continues
    else:
        is_new = _llm_is_new_case(prev, message, state.get("model"))  # ambiguous → ask the model

    if is_new:
        new_context = leading
    else:
        # Continuation: keep prior context + append any new block (lab results…).
        new_context = (prev + ("\n\n" + leading if leading else "")).strip()

    return {"is_new_case": is_new, "case_context": new_context}


def _route_entry(state: CorrectionState) -> str:
    """Clinical mode detects the case first; else straight to KB/generate."""
    if state.get("clinical_case"):
        return "detect"
    return "retrieve" if state.get("use_local_kb") else "generate"


def _route_after_detect(state: CorrectionState) -> str:
    return "retrieve" if state.get("use_local_kb") else "generate"


def _build_contents(
    history: List[Dict[str, str]], message: str
) -> List[Dict[str, Any]]:
    """Map our {role, content} history to Gemini's user/model turn format."""
    contents: List[Dict[str, Any]] = []
    for m in history or []:
        role = "user" if m.get("role") == "user" else "model"
        contents.append({"role": role, "parts": [{"text": m.get("content", "")}]})
    contents.append({"role": "user", "parts": [{"text": message}]})
    return contents


def _fanout_case_description(
    questions: List[Dict[str, Any]],
    seed: str = "",
    clinical: bool = False,
) -> List[Dict[str, Any]]:
    """
    Give every question of a clinical case its context back.

    The model now writes the case text ONCE — on the first question of the case
    — and leaves `caseDescription` empty on the rest. Re-typing a 150-word
    patient presentation for each of five questions was pure decode time, and
    decode is what the corrector actually waits on.

    Storage is deliberately unchanged: each row still carries its own
    `clinicalCasePartDescription`, so the student app and the translation table
    see exactly what they saw before. Only the GENERATION is deduplicated —
    copying a string server-side costs nothing.

    Runs before `classify`, not after: that node ranks each question on
    `caseDescription + description`, so parts 2..n stripped of their context
    would get worse course suggestions. Those re-injected tokens are prefill,
    which is not what makes the corrector wait.

    `seed` is the remembered context of a case already in progress, so a partial
    submission ("3. …" on its own) inherits it too.
    """
    last = (seed or "").strip()
    for q in questions:
        current = (q.get("caseDescription") or "").strip()
        if current:
            last = current
            continue
        if not last:
            continue
        # Clinical mode: the whole submission belongs to the remembered case.
        # Otherwise demand the model's explicit membership flag — without it a
        # standalone question pasted after a case would inherit someone else's
        # patient presentation, and students would see it.
        if clinical or q.get("caseIndex") is not None:
            q["caseDescription"] = last
    return questions


def _generate(state: CorrectionState) -> CorrectionState:
    """Correct the pasted question(s). Web-search → markdown; else → question list."""
    use_web_search = bool(state.get("use_web_search"))
    model = state.get("model")

    # Ground on retrieved course material when the KB node ran.
    system_prompt = state.get("system_prompt", "")
    context = state.get("context") or ""
    mode = "recherche web" if use_web_search else "structuré"
    if state.get("clinical_case"):
        mode += " · cas clinique"
    kb = f"contexte KB {len(context)} car." if context else "AUCUN contexte KB"
    _rule("CORRECTION", "═")
    _log(f"  mode : {mode} · {kb}")
    if context:
        system_prompt += (
            "\n\nInstruction de source · Base de connaissances\n"
            "# CITATION DE LA SOURCE — MODE BASE DE CONNAISSANCES\n"
            "Des extraits de documents te sont fournis dans le contexte ci-dessous "
            "(préfixés « source: <nom du document> »). Pour chaque option qui s'appuie "
            "directement sur ces extraits, tu DOIS remplir le champ `source` avec : "
            "<nom du document réel> (le document dont provient l'information), puis, "
            "entre parenthèses, cite le PASSAGE le plus pertinent de cet extrait. "
            "N'invente JAMAIS d'URL ni de document ; si aucun extrait ne couvre la "
            "question, laisse le champ `source` VIDE ou NULL.\n\n"
            f"{context}"
        )
        _log(f"  prompt système + contexte KB : {len(system_prompt)} car. au total")

    # Clinical mode: prepend the remembered case context so a partial submission
    # (e.g. just "3. …") is corrected WITH the full patient background.
    effective_message = state["message"]
    case_context = state.get("case_context") or ""
    if state.get("clinical_case") and case_context:
        effective_message = (
            "CONTEXTE DU CAS CLINIQUE (à conserver pour TOUTES les questions ; "
            "ne le recopie pas dans « description ») :\n"
            f"{case_context}\n\n{state['message']}"
        )

    contents = _build_contents(state.get("history", []), effective_message)

    if use_web_search:
        # Google Search grounding can't be combined with a forced responseSchema,
        # and free-form grounded JSON is often malformed. So we do TWO passes:
        #   A) grounded research (with citations) as free text — reliable grounding;
        #   B) restructure that analysis into strict JSON via responseSchema —
        #      guaranteed-valid structure, carrying the citations into each choice.
        research_prompt = system_prompt + (
            "\n\nUtilise la recherche web pour vérifier chaque question. Pour CHAQUE "
            "proposition, dis brièvement si elle est correcte ou fausse et indique la "
            "source (avec son URL). Sois concis."
        )
        research, sources = _gemini_grounded(research_prompt, contents, model=model)

        # Feed the REAL grounding URLs to the structuring pass so it can cite
        # concrete links (Gemini keeps these in groundingMetadata, not the text).
        if sources:
            research += "\n\nSOURCES WEB (URLs réelles à citer):\n" + "\n".join(
                f"- {s['title']}: {s['uri']}" for s in sources
            )

        struct_system = system_prompt + (
            "\n\nStructure l'analyse fournie en objets question JSON. CHAQUE option "
            "porte SA PROPRE explication dans SON champ « explanation » : pourquoi "
            "CETTE option est vraie ou fausse. Ne produis PAS de commentaire global — "
            "le champ « explanation » au niveau de la question reste VIDE (\"\"). "
            "Pour une question à choix composés (K-type), l'explication se place sur "
            "CHAQUE SOUS-PROPOSITION NUMÉROTÉE de « choices » (1, 2, 3…) : pourquoi CE "
            "point précis est vrai ou faux. Les combinaisons lettrées de "
            "« propositions » (« A. 1+2 ») ne portent AUCUNE explication — laisse leur "
            "champ « explanation » VIDE (\"\"), elles ne font qu'assembler des "
            "sous-propositions déjà justifiées. "
            "Chaque explication fait 2 à 4 phrases et contient TOUJOURS l'élément "
            "discriminant : le chiffre, le seuil, le critère ou le mécanisme qui permet "
            "de trancher. Pas de paraphrase de l'option, pas de justification "
            "circulaire. "
            "Renseigne aussi « globalComment » : une SYNTHÈSE pédagogique destinée à "
            "l'étudiant (affichée après sa réponse) — le raisonnement qui permet de "
            "trancher, et si c'est pertinent un petit tableau comparatif en MARKDOWN "
            "(ligne d'en-tête « | Critère | A | B | », puis « |---|---|---| », puis les "
            "lignes de données). N'utilise JAMAIS de balises HTML : elles sont "
            "converties automatiquement à l'affichage. "
            "N'y recopie pas les explications option par option. "
            "Laisse-le vide (\"\") si la question ne s'y prête pas. Le champ "
            "« explanation » au niveau de la question reste réservé aux ALERTES "
            "(« ⚠️ CONFLIT CLÉ : », « ⚠️ AMBIGUË : », « ⚠️ INCERTAIN : », « ⚠️ TEXTE : ») "
            "et vaut \"\" s'il n'y a rien à signaler. "
            # No source line, and no `source` field either. Grounded search
            # returned URLs too generic to be worth reading — a publisher's home
            # page rather than the passage backing the point — and this
            # instruction wrote them INTO the explanation, so every student saw
            # them. Said explicitly rather than merely omitted: the research
            # pass above still carries a "SOURCES WEB" list, and left to itself
            # the model appends citations from it out of habit.
            "N'ajoute AUCUNE mention de source : ni ligne « Source : … » à la fin "
            "d'une explication, ni URL dans le texte. Laisse le champ « source » "
            "de chaque option VIDE."
        )
        struct_contents = [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            f"QUESTIONS ORIGINALES:\n{effective_message}\n\n"
                            f"ANALYSE SOURCÉE:\n{research}\n\n"
                            "Produis le tableau JSON structuré (un objet par question, "
                            "dans l'ordre)."
                        )
                    }
                ],
            }
        ]
        raw = _gemini_generate(
            struct_system,
            struct_contents,
            response_schema=QUESTIONS_RESPONSE_SCHEMA,
            model=model,
        )
        questions = _parse_questions(raw)
        if questions:
            _log(f"  ✓ {len(questions)} question(s) structurée(s)")
            _rule("", "═")
            return {
                "questions": _fanout_case_description(
                    questions, case_context, bool(state.get("clinical_case"))
                ),
                "reply": "",
            }
        # Structuring failed → show the grounded analysis as markdown.
        _log("  ⚠ structuration impossible — renvoi de l'analyse en markdown")
        _rule("", "═")
        return {"questions": [], "reply": research or "No response generated"}

    # No web search: enforce the JSON array with responseSchema (most reliable).
    # The mode-specific source rule (KB document / established knowledge) is
    # appended to the system prompt by z_api/NoblesQcm per the active mode.
    system_prompt += (
        "\n\nL'administrateur peut coller UNE ou PLUSIEURS questions à la fois. "
        "Renvoie un TABLEAU JSON avec un objet par question, dans l'ordre."
    )
    raw = _gemini_generate(
        system_prompt, contents, response_schema=QUESTIONS_RESPONSE_SCHEMA, model=model
    )
    questions = _parse_questions(raw)
    if questions:
        _log(f"  ✓ {len(questions)} question(s) structurée(s)")
        _rule("", "═")
        return {
            "questions": _fanout_case_description(
                questions, case_context, bool(state.get("clinical_case"))
            ),
            "reply": "",
        }
    _log("  ⚠ échec de l'analyse JSON — réponse brute renvoyée")
    _rule("", "═")
    return {"questions": [], "reply": raw or "No response generated"}


# Leading choice label like "A.", "A)", "1 -", "B:" at the start of a text.
_LABEL_PREFIX = re.compile(r"^\s*([A-Za-z]|\d{1,2})\s*[.)\-–:]\s*")


def _normalize_labels(choices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure every choice has an authoritative label + clean text.

    Tolerates model drift: honours an explicit `label` OR `letter` field, and
    ALWAYS strips a redundant leading label prefix from the text ("A. foo" → foo),
    so the label is never both a field AND embedded in the text. Falls back to the
    text prefix, then to A, B, C… by order."""
    for i, c in enumerate(choices):
        # The model sometimes calls it `letter` instead of `label`.
        label = (c.get("label") or c.get("letter") or "").strip()
        text = (c.get("text") or "").strip()
        m = _LABEL_PREFIX.match(text)
        if m:
            # Always remove the "A." / "A)" prefix from the visible text.
            text = text[m.end():].strip()
            if not label:
                label = m.group(1)
        if not label:
            label = chr(65 + i)  # A, B, C, …
        c["label"] = label.upper()
        c["text"] = text
        c.pop("letter", None)
    return choices


def _parse_questions(raw: str) -> List[Dict[str, Any]]:
    """Extract + validate a list of question objects from a raw model reply."""
    try:
        items = _extract_json(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"[correction_graph] JSON extract failed: {exc}")
        return []
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        return []
    out: List[Dict[str, Any]] = []
    for it in items:
        try:
            q = CorrectionQuestion.model_validate(it).model_dump()
        except ValidationError as exc:
            print(f"[correction_graph] question validation failed: {exc}")
            continue
        q["choices"] = _normalize_labels(q.get("choices") or [])
        if q.get("propositions"):
            q["propositions"] = _normalize_labels(q["propositions"])
        out.append(q)
    return out


def _match_bracket(t: str, start: int) -> int:
    """Index of the bracket that closes the one at `start`, string-aware.
    Ignores brackets inside JSON strings and any trailing text after the match
    (e.g. a "SOURCES WEB" block the model appends after the array)."""
    open_ch = t[start]
    close_ch = "]" if open_ch == "[" else "}"
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return -1


def _extract_json(text: str):
    """Pull the FIRST complete JSON array/object from model text, ignoring any
    trailing prose (grounding "SOURCES WEB" blocks, notes) that the model appends
    after it — matching brackets by depth so trailing brackets never corrupt it."""
    t = (text or "").strip()
    if t.startswith("```"):
        # drop the opening fence (+ optional language tag) and closing fence
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()
    # Prefer an array; else a single object — whichever opens first.
    arr, obj = t.find("["), t.find("{")
    candidates = [p for p in (arr, obj) if p != -1]
    if not candidates:
        raise ValueError("no JSON found")
    start = min(candidates)
    end = _match_bracket(t, start)
    if end == -1:
        raise ValueError("no balanced JSON found")
    return json.loads(t[start : end + 1])


def _classify(state: CorrectionState) -> CorrectionState:
    """Suggest the best-fitting course for each question (HITL: admin approves)."""
    questions = state.get("questions") or []
    if not questions:
        return {}

    candidates = state.get("candidate_courses") or []
    # Residanat exams span many modules/years and cannot create new courses, so
    # each candidate may carry a "group" label (module · année) for disambiguation.
    def _cat_line(c: Dict[str, Any]) -> str:
        group = c.get("group")
        return f"- id={c.get('id')} | {c.get('name')}" + (f" ({group})" if group else "")

    catalogue = (
        "\n".join(_cat_line(c) for c in candidates)
        if candidates
        else "(aucun cours existant)"
    )
    listing = "\n\n".join(
        f"[{i}] {q.get('caseDescription', '')} {q.get('description', '')}".strip()
        for i, q in enumerate(questions)
    )
    # When new courses aren't allowed (e.g. residanat), the model must pick an
    # existing id or leave the suggestion empty — never invent a name.
    allow_new = state.get("allow_new_course", True)
    if allow_new:
        no_match_rule = (
            "Si aucun cours de la liste ne convient, mets suggestedCourId à null "
            "et propose un nom de cours court et pertinent dans suggestedCourName."
        )
    else:
        no_match_rule = (
            "Choisis UNIQUEMENT un id existant du catalogue. Si vraiment aucun ne "
            "convient, mets suggestedCourId à null et suggestedCourName à \"\" "
            "(n'invente jamais de nouveau cours)."
        )
    system = (
        "Tu es un classificateur de curriculum médical. Pour CHAQUE question, "
        "détermine le sujet médical puis choisis le cours le plus pertinent du "
        "catalogue et renvoie son id dans suggestedCourId. "
        f"{no_match_rule} "
        "Donne aussi un score de confiance entre 0 et 1 dans confidence. "
        "Réponds avec exactement une entrée par question, dans le même ordre."
    )
    user = f"CATALOGUE DES COURS:\n{catalogue}\n\nQUESTIONS:\n{listing}"

    by_id = {str(c.get("id")): c.get("name", "") for c in candidates}
    try:
        raw = _gemini_generate(
            system,
            [{"role": "user", "parts": [{"text": user}]}],
            response_schema=CLASSIFY_RESPONSE_SCHEMA,
            model=state.get("model"),
        )
        suggestions = json.loads(raw)
    except Exception as exc:  # noqa: BLE001 - classification is best-effort
        print(f"[correction_graph] classification failed: {exc}")
        suggestions = []

    for i, q in enumerate(questions):
        sug = suggestions[i] if i < len(suggestions) else {}
        raw_id = sug.get("suggestedCourId")
        cour_id = int(raw_id) if raw_id not in (None, "") and str(raw_id).isdigit() else None
        # Only honor ids that are actually in the catalogue.
        if cour_id is not None and str(cour_id) not in by_id:
            cour_id = None
        # Prefer the catalogue's canonical name when we matched an existing id.
        if cour_id is not None:
            name = by_id.get(str(cour_id), "")
        else:
            name = (sug.get("suggestedCourName") or "") if allow_new else ""
        q["suggestedCourId"] = cour_id
        q["suggestedCourName"] = name or ""
        conf = sug.get("confidence")
        q["classifyConfidence"] = float(conf) if isinstance(conf, (int, float)) else None

    return {"questions": questions}


# Compile the graph once at import time.
#   entry ─(clinical?)→ detect ─┬─(kb?)→ retrieve → generate → classify → END
#         └───────────────────── ┴──────────────── generate → classify → END
_graph = StateGraph(CorrectionState)
_graph.add_node("detect", _detect_case)
_graph.add_node("retrieve", _retrieve_context)
_graph.add_node("generate", _generate)
_graph.add_node("classify", _classify)
_graph.set_conditional_entry_point(
    _route_entry,
    {"detect": "detect", "retrieve": "retrieve", "generate": "generate"},
)
_graph.add_conditional_edges(
    "detect", _route_after_detect,
    {"retrieve": "retrieve", "generate": "generate"},
)
_graph.add_edge("retrieve", "generate")
_graph.add_edge("generate", "classify")
_graph.add_edge("classify", END)
correction_graph = _graph.compile()


def run_correction(
    message: str,
    history: List[Dict[str, str]] | None = None,
    system_prompt: str = "",
    use_web_search: bool = False,
    use_local_kb: bool = False,
    metadata: Dict[str, Any] | None = None,
    candidate_courses: List[Dict[str, Any]] | None = None,
    clinical_case: bool = False,
    case_context: str = "",
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Invoke the correction graph.

    Returns { questions | reply } plus, for clinical mode, the updated
    `caseContext` and `isNewCase` so z_api can persist the case memory.
    """
    result = correction_graph.invoke(
        {
            "message": message,
            "history": history or [],
            "system_prompt": system_prompt,
            "use_web_search": use_web_search,
            "use_local_kb": use_local_kb,
            "metadata": metadata or {},
            "candidate_courses": candidate_courses or [],
            "clinical_case": clinical_case,
            "case_context": case_context,
            "model": model,
        }
    )
    questions = result.get("questions") or []

    # ── Programmatically populate the `source` + `sourceRef` on each choice ─
    # The LLM rarely fills the JSON `source` field reliably, so we set it
    # from the KB retrieval metadata we already have (document + pages).
    # `sourceRef` carries the full hit (page, fileUrl, snippet) so the
    # frontend can open the PDF at the exact page with passage highlighting.
    kb_sources = result.get("kb_sources") or []
    if questions and kb_sources:
        # Build a concise source label: group pages per document.
        from collections import OrderedDict
        doc_pages: OrderedDict[str, list] = OrderedDict()
        for ks in kb_sources:
            doc = ks.get("source") or "unknown"
            pg = ks.get("page")
            doc_pages.setdefault(doc, [])
            if pg is not None and pg not in doc_pages[doc]:
                doc_pages[doc].append(pg)
        source_labels = []
        for doc, pages in doc_pages.items():
            if pages:
                pages_str = ", ".join(str(p) for p in sorted(pages))
                source_labels.append(f"{doc} (pages {pages_str})")
            else:
                source_labels.append(doc)
        source_label = " | ".join(source_labels)

        # For each choice, find the KB hit whose snippet best overlaps
        # with the choice's explanation text (simple word-overlap score).
        def _best_hit(explanation: str) -> dict | None:
            if not explanation or not kb_sources:
                return None
            exp_words = set(explanation.lower().split())
            best, best_score = None, 0
            for ks in kb_sources:
                snip = (ks.get("snippet") or "").lower()
                score = len(exp_words & set(snip.split()))
                if score > best_score:
                    best_score = score
                    best = ks
            return best if best_score > 2 else (kb_sources[0] if kb_sources else None)

        for q in questions:
            for choice in q.get("choices") or []:
                if not choice.get("source"):
                    choice["source"] = source_label
                hit = _best_hit(choice.get("explanation") or "")
                if hit:
                    choice["sourceRef"] = {
                        "source": hit.get("source") or "",
                        "page": hit.get("page"),
                        "fileUrl": hit.get("fileUrl") or "",
                        "snippet": hit.get("snippet") or "",
                        "bbox": hit.get("bbox"),
                    }
            for prop in q.get("propositions") or []:
                if not prop.get("source"):
                    prop["source"] = source_label

    out: Dict[str, Any] = (
        {"questions": questions}
        if questions
        else {"reply": result.get("reply") or "No response generated"}
    )
    if clinical_case:
        out["caseContext"] = result.get("case_context", case_context)
        out["isNewCase"] = bool(result.get("is_new_case"))
    if kb_sources:
        out["kbSources"] = kb_sources
    return out


def run_classification(
    questions: List[Dict[str, Any]],
    candidate_courses: List[Dict[str, Any]] | None = None,
    allow_new_course: bool = True,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Classify already-structured questions against a course catalogue in ONE call.
    Reuses the `_classify` node directly (no generate/retrieve). Each returned
    question carries suggestedCourId / suggestedCourName / classifyConfidence.
    """
    result = _classify(
        {
            "questions": questions,
            "candidate_courses": candidate_courses or [],
            "allow_new_course": allow_new_course,
            "model": model,
        }
    )
    return {"questions": result.get("questions") or questions}


def run_module_classification(
    questions: List[Dict[str, Any]],
    candidate_modules: List[Dict[str, Any]] | None = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Résidanat: pick the best-fitting MODULE (from the DB modules list) for each
    already-structured question, in ONE call. Modules are fixed, so the model must
    choose an existing id — never invent one. Reuses CLASSIFY_RESPONSE_SCHEMA
    (suggestedCourId/Name repurposed as the module), returning per question:
    { suggestedModuleId, suggestedModuleName, confidence }.
    """
    mods = candidate_modules or []
    if not questions:
        return {"questions": []}

    def _cat_line(m: Dict[str, Any]) -> str:
        group = m.get("group")
        return f"- id={m.get('id')} | {m.get('name')}" + (f" ({group})" if group else "")

    catalogue = "\n".join(_cat_line(m) for m in mods) if mods else "(aucun module)"
    listing = "\n\n".join(
        f"[{i}] {q.get('caseDescription', '')} {q.get('description', '')}".strip()
        for i, q in enumerate(questions)
    )
    system = (
        "Tu es un classificateur de curriculum médical. Pour CHAQUE question, "
        "détermine le domaine médical puis choisis le MODULE le plus pertinent du "
        "catalogue et renvoie son id dans suggestedCourId. Choisis UNIQUEMENT un id "
        "existant du catalogue ; si vraiment aucun ne convient, mets suggestedCourId "
        "à null et suggestedCourName à \"\" (n'invente jamais de module). Donne un "
        "score de confiance entre 0 et 1 dans confidence. Réponds avec exactement une "
        "entrée par question, dans le même ordre."
    )
    user = f"CATALOGUE DES MODULES:\n{catalogue}\n\nQUESTIONS:\n{listing}"

    by_id = {str(m.get("id")): m.get("name", "") for m in mods}
    try:
        raw = _gemini_generate(
            system,
            [{"role": "user", "parts": [{"text": user}]}],
            response_schema=CLASSIFY_RESPONSE_SCHEMA,
            model=model,
        )
        suggestions = json.loads(raw)
    except Exception as exc:  # noqa: BLE001 - classification is best-effort
        print(f"[correction_graph] module classification failed: {exc}")
        suggestions = []

    out: List[Dict[str, Any]] = []
    for i in range(len(questions)):
        sug = suggestions[i] if i < len(suggestions) else {}
        raw_id = sug.get("suggestedCourId")
        mod_id = int(raw_id) if raw_id not in (None, "") and str(raw_id).isdigit() else None
        if mod_id is not None and str(mod_id) not in by_id:
            mod_id = None
        conf = sug.get("confidence")
        out.append(
            {
                "suggestedModuleId": mod_id,
                "suggestedModuleName": by_id.get(str(mod_id), "") if mod_id is not None else "",
                "confidence": float(conf) if isinstance(conf, (int, float)) else None,
            }
        )
    return {"questions": out}
