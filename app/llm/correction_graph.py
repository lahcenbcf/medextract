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
import time
from typing import Any, Dict, List, Optional, TypedDict

import requests
from pydantic import BaseModel, Field, ValidationError
from langgraph.graph import StateGraph, END

from app.rag.ingest import retrieve


GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))


# ─── Pydantic models (structured-output validation) ────────────────────
class CorrectionChoice(BaseModel):
    # Lenient: in web-search mode (no enforced schema) the model often omits
    # the label and embeds it in the text — we normalise it after parsing.
    label: str = ""
    text: str
    isCorrect: bool = False


class CorrectionQuestion(BaseModel):
    description: str
    isKtype: bool = False
    choices: List[CorrectionChoice] = Field(default_factory=list)
    propositions: List[CorrectionChoice] | None = None
    correctAnswers: str = ""
    # ONE consolidated commentary for the whole question (why each option /
    # composed K-type answer is correct/false, with sources).
    explanation: str = ""
    # Clinical case: the shared/preceding context for THIS question (patient
    # intro, new lab results…) — empty for a standalone question — and the
    # question's index within the case.
    caseDescription: str = ""
    caseIndex: Optional[int] = None
    # Résidanat only: which of the 3 fixed épreuves this question belongs to —
    # "biologie" | "chirurgie" | "medecine". Empty for non-résidanat exams.
    epreuve: str = ""


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
                },
                "required": ["label", "text", "isCorrect"],
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
                },
                "required": ["label", "text", "isCorrect"],
            },
        },
        "correctAnswers": {"type": "STRING"},
        # One consolidated commentary for the whole question.
        "explanation": {"type": "STRING"},
        "caseDescription": {"type": "STRING"},
        "caseIndex": {"type": "INTEGER", "nullable": True},
        # Résidanat only: "biologie" | "chirurgie" | "medecine" (empty otherwise).
        "epreuve": {"type": "STRING", "nullable": True},
    },
    "required": ["description", "isKtype", "choices", "correctAnswers", "explanation"],
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
    """RAG node: metadata-filtered grounding from the Local Knowledge Base."""
    try:
        hits = retrieve(
            query=state["message"],
            metadata=state.get("metadata") or {},
            top_k=RAG_TOP_K,
        )
    except Exception as exc:  # noqa: BLE001 - never fail the chat on KB issues
        print(f"[correction_graph] KB retrieval failed, continuing without: {exc}")
        return {"context": ""}

    if not hits:
        return {"context": ""}

    blocks = []
    for i, hit in enumerate(hits, start=1):
        source = hit.get("source") or "unknown"
        course = hit.get("course") or ""
        header = f"[{i}] source: {source}" + (f" | course: {course}" if course else "")
        blocks.append(f"{header}\n{hit.get('context', '')}")
    return {"context": "\n\n---\n\n".join(blocks)}


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


def _generate(state: CorrectionState) -> CorrectionState:
    """Correct the pasted question(s). Web-search → markdown; else → question list."""
    use_web_search = bool(state.get("use_web_search"))
    model = state.get("model")

    # Ground on retrieved course material when the KB node ran.
    system_prompt = state.get("system_prompt", "")
    context = state.get("context") or ""
    if context:
        system_prompt += (
            "\n\n# LOCAL KNOWLEDGE BASE CONTEXT\n"
            "The following excerpts come from the course material for this "
            "exam's module and year. Prefer them over your own recollection "
            "when they are relevant, and do not contradict them. If they do "
            "not cover the question, rely on your medical knowledge and say so "
            "in the explanation.\n\n"
            f"{context}"
        )

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
            "\n\nStructure l'analyse fournie en objets question JSON. Le champ "
            "« explanation » est UN SEUL commentaire consolidé pour toute la question : "
            "traite CHAQUE option (« A. … B. … », pourquoi vraie/fausse) — ou, pour une "
            "question à choix composés (K-type), traite chaque réponse composée "
            "(« A. (1+2) vraie car… ») et NON les sous-propositions individuelles. "
            "Termine par « Source : <URL> » avec une URL RÉELLE de la liste SOURCES WEB "
            "ci-dessus (jamais inventée) ; si la liste est vide, écris "
            "« Source : connaissances médicales établies »."
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
            return {"questions": questions, "reply": ""}
        # Structuring failed → show the grounded analysis as markdown.
        return {"questions": [], "reply": research or "No response generated"}

    # No web search: enforce the JSON array with responseSchema (most reliable).
    system_prompt += (
        "\n\nL'administrateur peut coller UNE ou PLUSIEURS questions à la fois. "
        "Renvoie un TABLEAU JSON avec un objet par question, dans l'ordre."
    )
    raw = _gemini_generate(
        system_prompt, contents, response_schema=QUESTIONS_RESPONSE_SCHEMA, model=model
    )
    questions = _parse_questions(raw)
    if questions:
        return {"questions": questions, "reply": ""}
    print("[correction_graph] structured parse failed")
    return {"questions": [], "reply": raw or "No response generated"}


# Leading choice label like "A.", "A)", "1 -", "B:" at the start of a text.
_LABEL_PREFIX = re.compile(r"^\s*([A-Za-z]|\d{1,2})\s*[.)\-–:]\s*")


def _normalize_labels(choices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure every choice has a label. If missing, pull it from the text
    prefix ("A. foo" → label A, text foo); otherwise assign A, B, C… by order."""
    for i, c in enumerate(choices):
        label = (c.get("label") or "").strip()
        text = (c.get("text") or "").strip()
        if not label:
            m = _LABEL_PREFIX.match(text)
            if m:
                label = m.group(1).upper()
                text = text[m.end():].strip()
            else:
                label = chr(65 + i)  # A, B, C, …
        c["label"] = label
        c["text"] = text
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


def _extract_json(text: str):
    """Best-effort: pull the outermost JSON array/object from model text."""
    t = (text or "").strip()
    if t.startswith("```"):
        # drop the opening fence (+ optional language tag) and closing fence
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()
    start = t.find("[")
    end = t.rfind("]")
    if start != -1 and end > start:
        return json.loads(t[start : end + 1])
    # fall back to a single object
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end > start:
        return json.loads(t[start : end + 1])
    raise ValueError("no JSON found")


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
    out: Dict[str, Any] = (
        {"questions": questions}
        if questions
        else {"reply": result.get("reply") or "No response generated"}
    )
    if clinical_case:
        out["caseContext"] = result.get("case_context", case_context)
        out["isNewCase"] = bool(result.get("is_new_case"))
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
