"""
MedExtract-IA: LLM Structured Output Layer

Uses OpenAI or Anthropic LLMs to semantically segment unstructured
medical QCM text into structured question objects.

Handles:
  C1 — Logic type flagging (POSITIVE/NEGATIVE)
  C2 — K-type structure splitting (sub_propositions + combinations)
  C3 — Context propagation for clinical cases

Large Document Strategy:
  - Documents are split into chunks at natural question boundaries
  - Each chunk carries a "context window" from the previous chunk
    (the last active clinical case narrative) to ensure C3 continuity
  - Results from all chunks are merged and de-duplicated
"""

import os
import re
import json
import time
from typing import Any, Dict, Optional
from app.schemas import LLMExtractionOutput, LLMQuestionOutput, LLMClinicalCase
from app.llm.correction_graph import thinking_config

# ─── Configuration ───────────────────────────────────────────────────────

# Max characters per chunk sent to the LLM (conservative to stay under token limits)
# gpt-4o: aggressive chunking prevents LLM laziness and skipping questions
MAX_CHUNK_CHARS = 2000
OVERLAP_CHARS = 500     # Context window carried to next chunk (~last clinical case)

# ─── System prompt for the LLM ──────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un expert en extraction de QCM médicaux. Tu reçois un document médical formaté en Markdown contenant des questions à choix multiples (QCM) et potentiellement des cas cliniques.

## Ta mission:
1. **Identifier chaque question** avec son type (UNIQUE_CHOICE, CLINIC_CASE)
2. **Extraire les propositions** (A, B, C, D, E...) et identifier les réponses correctes
3. **Détecter le type logique (C1)**: POSITIVE si la question demande ce qui est VRAI/CORRECT, NEGATIVE si elle demande ce qui est FAUX/INCORRECT
4. **Propager le contexte clinique (C3)**: Si un cas clinique est présenté suivi de plusieurs questions, attribuer le contexte du cas à chaque question dépendante
5. **Identifier les cas cliniques**: Détecter les narratifs cliniques (présentation patient, examens...) et les regrouper avec leurs questions
6. **Détecter les questions K-type (C2)**: Questions avec sous-propositions numérotées ET combinaisons lettrées

## CORRECTION ET RÉPONSES JUSTES:
Le document contient les réponses correctes, souvent sous la forme d'une "Grille des réponses", d'un tableau à la fin, ou directement après la question (ex: "Réponse: A").

**Consignes importantes**:
1. Il faut chercher explicitement la réponse donnée dans le texte ou dans la grille des réponses.
2. Veuillez ne pas deviner la réponse avec des connaissances externes. La tâche est d'extraire la réponse indiquée dans le document.
3. Si le document indique que la réponse est "E", le champ `correct_answers` doit être "E", même si une autre réponse semble plus logique.
4. Assure-toi de faire correspondre exactement la lettre (A, B, C, D, E) avec les propositions de la question.

## CAS CLINIQUES — DÉTECTION ET TYPAGE:
Un **cas clinique** est un narratif patient (âge, antécédents, examen, bilan...) suivi de questions.

**Indicateurs de cas clinique:**
- "Cas clinique N°", "Monsieur/Madame X, âgé(e) de..."
- Présentation de données cliniques: poids, taille, tension, bilan sanguin
- Texte narratif inséré ENTRE les questions (mises à jour: "Quelques jours après...", "Le patient a été mis sous...")

**RÈGLES DU CONTEXTE (CRITIQUE):**
- `is_clinical_case_child`: true pour chaque question du cas.
- `clinical_case_id`: l'index du cas (0 pour le 1er). **IMPORTANT**: Un cas clinique peut avoir plusieurs textes intercalés (Mises à jour). Toutes les questions et mises à jour liées au même patient appartiennent au MÊME cas clinique et doivent partager le MÊME `clinical_case_id`. Ne crée PAS un nouveau cas clinique dans ta réponse pour une mise à jour.
- `context`: Doit contenir le texte narratif (intro ou mise à jour).
- **MISE À JOUR DU CONTEXTE**: Si un nouveau texte (bilan, évolution, examens complémentaires...) est inséré *juste avant* une question, tu DOIS utiliser CE NOUVEAU TEXTE comme `context` pour cette question.
- Si aucun texte n'est inséré avant la question, recopie simplement le `context` de la question précédente.

**Exemple de contexte ÉVOLUTIF:**
```
Texte: "Mr AB, 55 ans..."
→ Q13: context = "Mr AB, 55 ans..."
Texte intercalé: "Quelques jours après, bilan..."  ← NOUVEAU
→ Q14: context = "Quelques jours après, bilan..."
→ Q15: context = "Quelques jours après, bilan..."
Texte intercalé: "Le patient a été pris en charge..." ← 2ème NOUVEAU
→ Q17: context = "Le patient a été pris en charge..."
```

## Questions K-type (C2):
Certaines questions ont DEUX niveaux de choix:

**Exemple K-type:**
```
25. L'état hyperosmolaire : La ou les réponses justes (P2 2024 T35)
1- Est défini par une hyperglycémie majeure associée à une osmolarité > 350 mosm/kg
2- Survient surtout chez le sujet âgé grabataire
3- La présence de cétose est obligatoire pour le diagnostic
4- Survient surtout chez les patients traités par insulinothérapie
5- L'existence d'une insulinopénie absolue est nécessaire
A: 1-2  B: 1-4  C: 2-3  D: 2-4  E: 4-5
Réponse: A
```

Pour cette question K-type, tu dois remplir:
- `is_ktype`: true
- `choices`: les sous-propositions numérotées → [{label:"1", text:"Est défini par...", is_correct:true}, {label:"2", text:"Survient surtout...", is_correct:true}, ...]. Le is_correct des choices est déterminé par les numéros présents dans la combinaison correcte (A=1+2, donc choices 1 et 2 sont correctes).
- `propositions`: les combinaisons lettrées → [{label:"A", text:"1+2", is_correct:true}, {label:"B", text:"1+4", is_correct:false}, ...]
- `correct_answers`: "A" (la lettre de la combinaison correcte)

**Question standard (NON K-type):**
```
1. Quel est le diagnostic ? A. Diabète  B. Hypothyroïdie  C. Basedow  Réponse: C
```
- `is_ktype`: false
- `choices`: [{label:"A", text:"Diabète", is_correct:false}, ...]
- `propositions`: [] (vide)

## Règles de formatage:
- Préserver le formatage **gras** (texte entre ** **)
- Les réponses correctes viennent de la GRILLE DE RÉPONSES si elle existe, sinon du texte "Réponse: X"
- Si une question contient "cochez la/les réponse(s) FAUSSE(s)" ou "SAUF" ou "NE...PAS" ou "(Réponse fausse)", le logic_type est NEGATIVE
- Si une question contient "cochez la/les réponse(s) JUSTE(s)" ou "parmi les propositions suivantes" ou "(Les réponses justes)", le logic_type est POSITIVE
- Chaque question doit avoir un course_name (la matière médicale). Il doit être EXACTEMENT le même pour toutes les questions du même document/chapitre.
- Les symboles médicaux (μmol/L, mmHg, etc.) doivent être préservés en UTF-8
- Les mots clés négatifs en gras (**SAUF**, **FAUX**, **NE PAS**) doivent rester en gras
- Les références de source comme "(P2 2024 T35)", "(Constantine 2023)" doivent être extraites dans where_is_mentioned

## Consignes pour l'explication (explanation) :
Il est important de ne pas résumer ou reformuler l'explication.
Si le document contient un long paragraphe ou un tableau d'explication pour une question, veuillez le copier mot pour mot (Copier-Coller exact), peu importe sa longueur.
La préservation du texte exact (avec le Markdown `**`, les puces `•` et `<br>`) est requise.
Si le texte est résumé, l'extraction sera considérée comme erronée."""

CHUNK_CONTEXT_PREFIX = """

## IMPORTANT — Continuité du Cas Clinique:
Le texte ci-dessous est la SUITE d'un document plus grand. 
L'ancien texte narratif actif était:
"{context}"

Instruction de continuité:
1. Les questions qui suivent appartiennent à ce cas clinique en cours. Tu DOIS régler `is_clinical_case_child: true` pour ces questions.
2. Si aucun nouveau texte narratif n'est présent avant la prochaine question, continue d'utiliser le texte ci-dessus comme `context`.
3. Dès qu'une nouvelle phrase narrative est identifiée (ex: "Le patient a été pris en charge..."), utilise uniquement cette nouvelle phrase comme `context`. Elle remplace l'ancienne.
4. CRITIQUE: Si tu rencontres un indicateur de questions isolées (ex: "QCS", "QCM", "Questions isolées", "Questions indépendantes", "Questions à choix simple/unique"), cela signifie que le cas clinique est terminé. Pour toutes les questions qui suivent cet indicateur, tu DOIS régler `is_clinical_case_child: false` et `clinical_case_id: null`.

--- DÉBUT DU NOUVEAU CONTENU ---
"""


# ─── Chunking Logic ─────────────────────────────────────────────────────

def _find_question_boundaries(text: str) -> list[int]:
    """
    Find line positions where new questions or sections start.
    These are safe split points that won't break a question mid-text.
    """
    boundaries = [0]
    lines = text.split('\n')
    pos = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Detect question starts: "1.", "Question 1:", "QCM 1:", headings
        is_boundary = False

        # Numbered question patterns (allowing optional markdown bold/italic ** or *, and optional space before punctuation)
        if re.match(r'^(\*\*|\*)?\d+\s*[\.\)\-:]\s', stripped):
            is_boundary = True
        # "Question N" pattern
        elif re.match(r'^(\*\*|\*)?Question\s+\d+', stripped, re.IGNORECASE):
            is_boundary = True
        # Markdown headings (## Cas Clinique, # QCM, etc.)
        elif re.match(r'^#{1,3}\s', stripped):
            is_boundary = True
        # "Cas Clinique" pattern
        elif re.match(r'^(\*\*|\*)?Cas\s+[Cc]linique', stripped, re.IGNORECASE):
            is_boundary = True

        # If it's a number pattern but NOT bolded, ONLY treat as boundary if preceded by empty line or at start
        # This prevents splitting on K-type sub-propositions (1., 2., 3.)
        if is_boundary and re.match(r'^\d+\s*[\.\)\-:]\s', stripped):
            if i > 0 and lines[i-1].strip() != '':
                is_boundary = False

        if is_boundary and pos > 0:
            boundaries.append(pos)

        pos += len(line) + 1  # +1 for the \n

    return boundaries


def _find_active_clinical_context(text: str) -> str:
    """
    Extract the last active clinical case narrative from a text chunk.
    This is the context that needs to be carried to the next chunk for C3 continuity.
    """
    lines = text.split('\n')
    
    # 1. Check for trailing narrative text after the last question/answer
    last_q_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r'^(\*\*|\*)?\d+\s*[\.\)\-:]\s', stripped) or re.match(r'^(\*\*|\*)?Question\s+\d+', stripped, re.IGNORECASE):
            last_q_idx = i
            
    if last_q_idx != -1:
        trailing_text = []
        # Scan backwards from the end of the chunk to the last question
        for i in range(len(lines) - 1, last_q_idx, -1):
            stripped = lines[i].strip()
            if not stripped:
                continue
                
            # Break if we hit a "Réponse" line
            if re.match(r'^(\*\*|\*)?R[ée]ponse\s*[:\.]', stripped, re.IGNORECASE):
                break
                
            # Break if we hit a standard choice line (A., B., C., A-, B), etc.
            # Handle cases with or without space: e.g. "A. Texte", "A.Texte", "A-Texte"
            if re.match(r'^(\*\*|\*)?[A-F][\.\)\-:]\s*', stripped, re.IGNORECASE):
                # To prevent falsely breaking on sentences like "A-t-il...", we can check if it's strictly a choice format.
                # However, since we are inside a QCM document, lines starting with A-, B., etc. are overwhelmingly choices.
                # If it's "A-t-il", it will break. To be slightly safer, we ensure the next char is not a lower-case letter 
                # UNLESS it's a known typo like "E-Diabète" (capital letter). But actually, "E-diabète" is also possible.
                # Let's just match any letter A-F followed by punctuation, as it's 99% a choice in this context.
                if not re.match(r'^(\*\*|\*)?[Aa]-t-il\b', stripped, re.IGNORECASE):
                    break
                
            # Break if we hit a line containing combinations like "A. 1 B. 2"
            if re.match(r'^(\*\*|\*)?A[\.\)\-:]', stripped, re.IGNORECASE) and re.search(r'\bB[\.\)\-:]', stripped, re.IGNORECASE):
                break
                
            # Break if we hit a sub-proposition line (1., 2., 3., 1-, 2-)
            if re.match(r'^(\*\*|\*)?\d+[\.\)\-:]\s', stripped):
                break
                
            # Skip tables and images, but don't break
            if stripped.startswith('|') or stripped.startswith('[['):
                continue
                
            trailing_text.insert(0, lines[i])
            
        # Filter out "Fin du cas clinique" which is a marker, not narrative
        filtered_trailing = [t for t in trailing_text if "fin du cas" not in t.lower()]
        
        # Filter out short section headers (like "**Oncologie**" or "Hématologie")
        final_trailing = []
        for t in filtered_trailing:
            clean_t = re.sub(r'[\*\#\_]', '', t).strip()
            # If it's a very short line without punctuation, it's a heading, not narrative
            if len(clean_t) < 30 and not re.search(r'[.,;:]', clean_t):
                continue
            final_trailing.append(t)
            
        if final_trailing:
            return '\n'.join(final_trailing)

    # 2. Fallback: Detect clinical case start (intro)
    context_lines = []
    in_clinical_case = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (re.match(r'^#{1,3}\s*(\*\*|\*)?Cas\s+[Cc]linique', stripped, re.IGNORECASE) or
            re.match(r'^(\*\*|\*)?Cas\s+[Cc]linique', stripped, re.IGNORECASE)):
            in_clinical_case = True
            context_lines = [line]
            continue

        if in_clinical_case:
            if re.match(r'^(\*\*|\*)?\d+\s*[\.\)\-:]\s', stripped) or re.match(r'^(\*\*|\*)?Question\s+\d+', stripped, re.IGNORECASE):
                break
            elif re.match(r'^#{1,3}\s', stripped) and not re.match(r'^#{1,3}\s*(\*\*|\*)?Cas', stripped, re.IGNORECASE):
                in_clinical_case = False
                context_lines = []
            else:
                if stripped:
                    context_lines.append(line)

    return '\n'.join(context_lines) if context_lines else ''


def _should_stop_clinical_context(text: str) -> bool:
    """Detect if the chunk starts with standalone question section headers (e.g. QCS, QCM)."""
    # Find the position of the first question
    first_q_pos = len(text)
    q_match = re.search(r'^(\*\*|\*)?\d+\s*[\.\)\-:]\s', text, re.MULTILINE) or re.search(r'^(\*\*|\*)?Question\s+\d+', text, re.IGNORECASE | re.MULTILINE)
    if q_match:
        first_q_pos = q_match.start()
        
    text_before_q = text[:first_q_pos]
    
    stop_patterns = [
        r'\bQCS\b',
        r'\bQCM\b',
        r'Questions?\s+isol[eé]es?',
        r'Questions?\s+ind[eé]pendantes?',
        r'Questions?\s+à\s+choix\s+simples?',
        r'Questions?\s+à\s+choix\s+multiples?',
    ]
    
    for pat in stop_patterns:
        if re.search(pat, text_before_q, re.IGNORECASE):
            return True
            
    return False


def chunk_document(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[dict]:
    """
    Split a large document into LLM-friendly chunks at natural question boundaries.

    Returns list of dicts:
      - 'text': the chunk text to send to LLM
      - 'context': the clinical case context from previous chunk (for C3)
      - 'is_continuation': whether this chunk continues from a previous one
      - 'chunk_index': 0-based chunk number
    """
    if len(text) <= max_chars:
        return [{'text': text, 'context': '', 'is_continuation': False, 'chunk_index': 0}]

    boundaries = _find_question_boundaries(text)
    chunks = []
    current_start = 0
    chunk_index = 0
    prev_context = ''

    while current_start < len(text):
        # Find the furthest boundary that fits within max_chars
        chunk_end = current_start + max_chars
        if chunk_end >= len(text):
            chunk_end = len(text)
        else:
            # Find the last boundary before chunk_end
            valid_boundaries = [b for b in boundaries if current_start < b <= chunk_end]
            if valid_boundaries:
                chunk_end = valid_boundaries[-1]
            else:
                # If a single question is larger than max_chars, find the NEXT boundary
                # This guarantees we NEVER split inside a question or its explanation table!
                next_boundaries = [b for b in boundaries if b > chunk_end]
                if next_boundaries:
                    chunk_end = next_boundaries[0]
                else:
                    chunk_end = len(text)

        chunk_text = text[current_start:chunk_end].strip()

        if chunk_text:
            context_to_pass = prev_context
            is_cont = chunk_index > 0
            
            # Check if this new chunk starts with standalone question indicators
            if _should_stop_clinical_context(chunk_text):
                print(f"[Chunking] Stop marker (QCS/QCM) detected in Chunk {chunk_index}. Resetting clinical context.")
                context_to_pass = ""
                is_cont = False
                prev_context = ""
                
            chunks.append({
                'text': chunk_text,
                'context': context_to_pass,
                'is_continuation': is_cont,
                'chunk_index': chunk_index,
            })

            # Extract the active clinical context for the next chunk
            prev_context = _find_active_clinical_context(chunk_text)
            chunk_index += 1

        current_start = chunk_end

    print(f"[Chunking] Document split into {len(chunks)} chunks "
          f"({len(text)} chars total, max {max_chars}/chunk)")
    for i, c in enumerate(chunks):
        ctx_info = f", context: {len(c['context'])} chars" if c['context'] else ""
        print(f"  Chunk {i}: {len(c['text'])} chars{ctx_info}")

    return chunks


# ─── LLM Extraction Functions ───────────────────────────────────────────

def _build_prompt(chunk: dict, file_name: str, dynamic_context: str = "") -> str:
    """Build the user prompt for a chunk, including context if it's a continuation."""
    file_context = f"Le nom du fichier d'origine est '{file_name}'. Utilise ce nom de fichier (ou sa matière sous-jacente) pour remplir le champ 'course_name' si tu n'es pas sûr.\n\n"
    
    # Give priority to chunk['context'] (trailing text) over dynamic_context (last question's context)
    # because trailing text appears AFTER the last question in the document flow!
    ctx = chunk['context'] if chunk['context'] else dynamic_context
    
    if chunk['is_continuation'] and ctx:
        return (
            file_context
            + CHUNK_CONTEXT_PREFIX.format(context=ctx)
            + chunk['text']
        )
    return f"{file_context}Extrais toutes les questions QCM du document suivant:\n\n{chunk['text']}"


# ─── Gemini extraction (REST, no SDK pin — same approach as correction_graph) ──

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"

_GEMINI_CHOICE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "label": {"type": "STRING"},
        "text": {"type": "STRING"},
        "is_correct": {"type": "BOOLEAN"},
    },
    "required": ["label", "text", "is_correct"],
}

# Mirrors LLMExtractionOutput so the JSON validates straight into Pydantic.
GEMINI_EXTRACTION_SCHEMA: dict = {
    "type": "OBJECT",
    "properties": {
        "questions": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "type": {
                        "type": "STRING",
                        "enum": ["UNIQUE_CHOICE", "CLINIC_CASE", "QROC"],
                    },
                    "description": {"type": "STRING"},
                    "choices": {"type": "ARRAY", "items": _GEMINI_CHOICE_SCHEMA},
                    "propositions": {
                        "type": "ARRAY",
                        "items": _GEMINI_CHOICE_SCHEMA,
                    },
                    "is_ktype": {"type": "BOOLEAN"},
                    "correct_answers": {"type": "STRING"},
                    "explanation": {"type": "STRING"},
                    "logic_type": {
                        "type": "STRING",
                        "enum": ["POSITIVE", "NEGATIVE"],
                    },
                    "course_name": {"type": "STRING"},
                    "context": {"type": "STRING"},
                    "is_clinical_case_child": {"type": "BOOLEAN"},
                    "clinical_case_id": {"type": "INTEGER", "nullable": True},
                    "where_is_mentioned": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                    "indication": {"type": "ARRAY", "items": {"type": "STRING"}},
                },
                "required": [
                    "type",
                    "description",
                    "choices",
                    "correct_answers",
                    "logic_type",
                ],
            },
        },
        "clinical_cases": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "intro_text": {"type": "STRING"},
                },
                "required": ["name", "intro_text"],
            },
        },
    },
    "required": ["questions"],
}


def _remap_clinical_cases(
    result: LLMExtractionOutput,
    chunk: dict,
    dynamic_context: str,
    all_cases: list,
) -> str:
    """
    Map each question's chunk-local `clinical_case_id` onto a document-global
    case index, so a case split across chunks stays one case.

    Returns the (possibly reset) dynamic_context for the next chunk.
    """
    if not chunk['is_continuation']:
        dynamic_context = ""
    active_ctx = chunk['context'] if chunk['context'] else dynamic_context

    def normalize(text):
        n = str(text).lower()
        n = re.sub(r'n[°0o]', '', n)
        return re.sub(r'[^a-z0-9]', '', n)

    for q in result.questions:
        if not q.is_clinical_case_child:
            continue
        if q.clinical_case_id is not None and 0 <= q.clinical_case_id < len(result.clinical_cases):
            case_obj = result.clinical_cases[q.clinical_case_id]
            matched_global_idx = -1
            t2 = normalize(case_obj.intro_text)

            # 0. Continuation of the currently active context
            if active_ctx and len(t2) > 15:
                norm_active = normalize(active_ctx)
                if t2[:30] in norm_active or norm_active[:30] in t2:
                    if all_cases:
                        matched_global_idx = len(all_cases) - 1

            # 1. Text similarity against already-seen cases
            if matched_global_idx == -1 and len(t2) > 15:
                for idx, gc in enumerate(all_cases):
                    t1 = normalize(gc.intro_text)
                    if len(t1) > 15 and (t2[:30] in t1 or t1[:30] in t2):
                        matched_global_idx = idx
                        break

            # 2. Normalized name (generic "cas clinique N" names don't count)
            if matched_global_idx == -1:
                norm_name = normalize(case_obj.name)
                if norm_name and not re.match(r'^casclinique\d*$', norm_name):
                    for idx, gc in enumerate(all_cases):
                        if normalize(gc.name) == norm_name:
                            matched_global_idx = idx
                            break

            if matched_global_idx == -1:
                matched_global_idx = len(all_cases)
                all_cases.append(case_obj)

            q.clinical_case_id = matched_global_idx
        else:
            # No local case object → fall back to the last active one.
            if all_cases and active_ctx:
                q.clinical_case_id = len(all_cases) - 1
            else:
                q.clinical_case_id = None
                q.is_clinical_case_child = False

    return dynamic_context


def extract_with_gemini(
    markdown_text: str,
    file_name: str,
    model: str = "gemini-2.0-flash",
) -> LLMExtractionOutput:
    """
    Extract QCM questions with Gemini, over raw REST — the same approach the
    correction graph already uses, so no SDK version is pinned.

    Gemini is used strictly as an EXTRACTOR here: the system prompt forbids it
    from reasoning about answers, and `responseSchema` forces the exact JSON
    shape, so there is no reasoning-token budget to exhaust.
    """
    import requests

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not configured")

    chunks = chunk_document(markdown_text)
    all_questions: list[LLMQuestionOutput] = []
    all_cases: list[LLMClinicalCase] = []
    dynamic_context = ""

    for chunk in chunks:
        prompt = _build_prompt(chunk, file_name, dynamic_context)
        print(
            f"[LLM] Processing chunk {chunk['chunk_index']+1}/{len(chunks)} "
            f"({len(chunk['text'])} chars)..."
        )

        generation_config: Dict[str, Any] = {
            "temperature": 0.1,
            "responseMimeType": "application/json",
            "responseSchema": GEMINI_EXTRACTION_SCHEMA,
            # Generous ceiling: a dense chunk with per-choice explanations
            # produces a lot of JSON, and truncation loses the whole chunk.
            "maxOutputTokens": 32768,
        }
        thinking = thinking_config(model)
        if thinking:
            generation_config["thinkingConfig"] = thinking

        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "generationConfig": generation_config,
        }

        result: Optional[LLMExtractionOutput] = None
        for attempt in range(4):
            try:
                resp = requests.post(
                    f"{GEMINI_ENDPOINT}/{model}:generateContent?key={api_key}",
                    json=body,
                    timeout=180,
                )
                if resp.status_code >= 400:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                data = resp.json()
                parts = (
                    data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [])
                )
                raw = "".join(p.get("text", "") for p in parts)
                if not raw.strip():
                    raise RuntimeError("Empty response from Gemini")
                result = LLMExtractionOutput.model_validate(json.loads(raw))
                break
            except Exception as e:
                wait = 2 ** (attempt + 1)
                print(f"[LLM] Error on chunk {chunk['chunk_index']+1}: {e}")
                if attempt == 3:
                    # One bad chunk must not sink the whole document.
                    print(
                        f"[LLM] Chunk {chunk['chunk_index']+1} skipped after 4 attempts"
                    )
                    break
                print(f"[LLM] retrying in {wait}s...")
                time.sleep(wait)

        if result is None:
            continue

        dynamic_context = _remap_clinical_cases(
            result, chunk, dynamic_context, all_cases
        )
        all_questions.extend(result.questions)
        print(
            f"[LLM] Chunk {chunk['chunk_index']+1}: {len(result.questions)} questions, "
            f"{len(result.clinical_cases)} cases"
        )

        if result.questions:
            last_q = result.questions[-1]
            dynamic_context = last_q.context or ""

    print(
        f"[LLM] Gemini extraction done: {len(all_questions)} questions, "
        f"{len(all_cases)} clinical case(s)"
    )
    return LLMExtractionOutput(questions=all_questions, clinical_cases=all_cases)


def extract_with_openai(
    markdown_text: str,
    api_keys: list[str],
    file_name: str,
    model: str = "gpt-4o",
) -> LLMExtractionOutput:
    """
    Use OpenAI structured output to extract QCM questions from markdown text.
    Supports both direct OpenAI keys (sk-...) and GitHub Models tokens (github_pat_...).
    Handles large documents via chunking with context continuity.
    """
    from openai import OpenAI

    # GitHub Models tokens use a different base URL
    base_url = None
    if api_keys[0].startswith("github_"):
        base_url = "https://models.inference.ai.azure.com"
    elif api_keys[0].startswith("sk-or-"):
        base_url = "https://openrouter.ai/api/v1"
    elif api_keys[0].startswith("gsk_"):
        base_url = "https://api.groq.com/openai/v1"

    current_key_idx = 0
    client = OpenAI(api_key=api_keys[current_key_idx], **({"base_url": base_url} if base_url else {}))

    chunks = chunk_document(markdown_text)

    all_questions: list[LLMQuestionOutput] = []
    all_cases: list[LLMClinicalCase] = []
    
    dynamic_context = ""

    for chunk in chunks:
        prompt = _build_prompt(chunk, file_name, dynamic_context)
        print(f"[LLM] Processing chunk {chunk['chunk_index']+1}/{len(chunks)} "
              f"({len(prompt)} chars)...")

        retry_count = 0
        max_retries = 5
        while retry_count < max_retries:
            try:
                is_groq = api_keys[current_key_idx].startswith("gsk_")
                
                if is_groq:
                    import json
                    schema_str = json.dumps(LLMExtractionOutput.model_json_schema())
                    groq_sys = f"{SYSTEM_PROMPT}\n\nYou MUST return only valid JSON matching this schema:\n{schema_str}"
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": groq_sys},
                            {"role": "user", "content": prompt},
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.1,
                        max_tokens=8192,
                    )
                    result = LLMExtractionOutput.model_validate_json(response.choices[0].message.content)
                else:
                    kwargs = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        "response_format": LLMExtractionOutput,
                    }
                    if model.startswith("o") and not model.startswith("o-") and "gpt-4o" not in model:
                        kwargs["max_completion_tokens"] = 8192
                    else:
                        kwargs["temperature"] = 0.1
                        kwargs["max_tokens"] = 8192
                        
                    response = client.beta.chat.completions.parse(**kwargs)
                    result = response.choices[0].message.parsed
                # --- FIX: Remap local chunk clinical_case_id to a global index ---
                if not chunk['is_continuation']:
                    dynamic_context = ""
                active_ctx = chunk['context'] if chunk['context'] else dynamic_context
                
                def normalize(text):
                    import re
                    n = str(text).lower()
                    n = re.sub(r'n[°0o]', '', n)
                    return re.sub(r'[^a-z0-9]', '', n)

                for q in result.questions:
                    if q.is_clinical_case_child:
                        if q.clinical_case_id is not None and 0 <= q.clinical_case_id < len(result.clinical_cases):
                            case_obj = result.clinical_cases[q.clinical_case_id]
                            matched_global_idx = -1
                            
                            t2 = normalize(case_obj.intro_text)
                            
                            # 0. Match against active_ctx (if it's a continuation of the active context)
                            if active_ctx and len(t2) > 15:
                                norm_active = normalize(active_ctx)
                                if t2[:30] in norm_active or norm_active[:30] in t2:
                                    if all_cases:
                                        matched_global_idx = len(all_cases) - 1
                            
                            # 1. Match by text similarity (substring search of first 30 chars)
                            if matched_global_idx == -1 and len(t2) > 15:
                                for idx, gc in enumerate(all_cases):
                                    t1 = normalize(gc.intro_text)
                                    if len(t1) > 15:
                                        if t2[:30] in t1 or t1[:30] in t2:
                                            matched_global_idx = idx
                                            break
                            
                            # 2. Match by normalized name (if text match failed)
                            if matched_global_idx == -1:
                                norm_name = normalize(case_obj.name)
                                # Prevent matching generic names like 'casclinique', 'casclinique1'
                                if norm_name and not re.match(r'^casclinique\d*$', norm_name):
                                    for idx, gc in enumerate(all_cases):
                                        if normalize(gc.name) == norm_name:
                                            matched_global_idx = idx
                                            break
                                    
                            if matched_global_idx == -1:
                                matched_global_idx = len(all_cases)
                                all_cases.append(case_obj)
                                
                            q.clinical_case_id = matched_global_idx
                        else:
                            # Fallback to last active case if missing local case_obj
                            if all_cases and active_ctx:
                                q.clinical_case_id = len(all_cases) - 1
                            else:
                                q.clinical_case_id = None
                                q.is_clinical_case_child = False
                            
                all_questions.extend(result.questions)
                print(f"[LLM] Chunk {chunk['chunk_index']+1}: "
                      f"{len(result.questions)} questions, {len(result.clinical_cases)} cases")
                
                # Update dynamic context for the next chunk
                if result.questions:
                    last_q = result.questions[-1]
                    if last_q.context:
                        dynamic_context = last_q.context
                    else:
                        dynamic_context = ""
                
                # Pace requests to avoid GitHub Models burst limits
                if chunk['chunk_index'] + 1 < len(chunks):
                    time.sleep(10)
                break
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "rate_limit" in error_msg.lower() or "too many requests" in error_msg.lower() or "organization_restricted" in error_msg.lower():
                    print(f"[LLM] Key restricted or rate limited (index {current_key_idx}).")
                    current_key_idx += 1
                    if current_key_idx < len(api_keys):
                        print(f"[LLM] Switching to API key index {current_key_idx}...")
                        client = OpenAI(api_key=api_keys[current_key_idx], **({"base_url": base_url} if base_url else {}))
                        continue # retry immediately with new key
                    else:
                        print("[LLM] All API keys exhausted! Waiting before retry...")
                        current_key_idx = 0 # wrap around
                        client = OpenAI(api_key=api_keys[current_key_idx], **({"base_url": base_url} if base_url else {}))
                        retry_count += 1
                        wait = min(15 * retry_count, 60)
                        print(f"[LLM] Retrying in {wait}s...")
                        time.sleep(wait)
                else:
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise
                    wait = 2 ** retry_count
                    print(f"[LLM] Error on chunk {chunk['chunk_index']+1}: {error_msg}, "
                          f"retrying in {wait}s...")
                    time.sleep(wait)

    return LLMExtractionOutput(
        questions=all_questions,
        clinical_cases=all_cases,
    )


def extract_with_anthropic(
    markdown_text: str,
    api_key: str,
    file_name: str,
    model: str = "claude-3-5-haiku-latest",
) -> LLMExtractionOutput:
    """
    Use Anthropic tool-use for structured extraction of QCM questions.
    Handles large documents via chunking with context continuity.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    schema = LLMExtractionOutput.model_json_schema()

    chunks = chunk_document(markdown_text)

    all_questions: list[LLMQuestionOutput] = []
    all_cases: list[LLMClinicalCase] = []
    
    dynamic_context = ""

    for chunk in chunks:
        prompt = _build_prompt(chunk, file_name, dynamic_context)
        print(f"[LLM] Processing chunk {chunk['chunk_index']+1}/{len(chunks)} "
              f"({len(prompt)} chars)...")

        retry_count = 0
        max_retries = 3
        while retry_count < max_retries:
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=8192,
                    system=SYSTEM_PROMPT,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    tools=[
                        {
                            "name": "extract_qcm",
                            "description": "Extract structured QCM data from a medical document",
                            "input_schema": schema,
                        }
                    ],
                    tool_choice={"type": "tool", "name": "extract_qcm"},
                    temperature=0.1,
                )

                for block in response.content:
                    if block.type == "tool_use":
                        result = LLMExtractionOutput.model_validate(block.input)
                        # --- FIX: Remap local chunk clinical_case_id to a global index ---
                        if not chunk['is_continuation']:
                            dynamic_context = ""
                        active_ctx = chunk['context'] if chunk['context'] else dynamic_context
                        
                        def normalize(text):
                            import re
                            n = str(text).lower()
                            n = re.sub(r'n[°0o]', '', n)
                            return re.sub(r'[^a-z0-9]', '', n)

                        for q in result.questions:
                            if q.is_clinical_case_child:
                                if q.clinical_case_id is not None and 0 <= q.clinical_case_id < len(result.clinical_cases):
                                    case_obj = result.clinical_cases[q.clinical_case_id]
                                    matched_global_idx = -1
                                    
                                    # FORCE MATCH FOR CONTINUATIONS:
                                    # The first case in a continuation chunk ALWAYS maps to the last active global case.
                                    # This prevents new updates ("Le myélogramme...") from creating a detached new case.
                                    if chunk['is_continuation'] and q.clinical_case_id == 0 and all_cases:
                                        matched_global_idx = len(all_cases) - 1
                                    else:
                                        t2 = normalize(case_obj.intro_text)
                                        
                                        # 0. Match against active_ctx
                                        if active_ctx and len(t2) > 15:
                                            norm_active = normalize(active_ctx)
                                            if t2[:30] in norm_active or norm_active[:30] in t2:
                                                if all_cases:
                                                    matched_global_idx = len(all_cases) - 1
                                        
                                        # 1. Match by text similarity
                                        if matched_global_idx == -1 and len(t2) > 15:
                                            for idx, gc in enumerate(all_cases):
                                                t1 = normalize(gc.intro_text)
                                                if len(t1) > 15:
                                                    if t2[:30] in t1 or t1[:30] in t2:
                                                        matched_global_idx = idx
                                                        break
                                    
                                    # 2. Match by normalized name
                                    if matched_global_idx == -1:
                                        norm_name = normalize(case_obj.name)
                                        # Prevent matching generic names like 'casclinique', 'casclinique1'
                                        if norm_name and not re.match(r'^casclinique\d*$', norm_name):
                                            for idx, gc in enumerate(all_cases):
                                                if normalize(gc.name) == norm_name:
                                                    matched_global_idx = idx
                                                    break
                                                
                                    if matched_global_idx == -1:
                                        matched_global_idx = len(all_cases)
                                        all_cases.append(case_obj)
                                        
                                    q.clinical_case_id = matched_global_idx
                                else:
                                    # Fallback
                                    if all_cases and active_ctx:
                                        q.clinical_case_id = len(all_cases) - 1
                                    else:
                                        q.clinical_case_id = None
                                        q.is_clinical_case_child = False
                                        
                        all_questions.extend(result.questions)
                        print(f"[LLM] Chunk {chunk['chunk_index']+1}: "
                              f"{len(result.questions)} questions, {len(result.clinical_cases)} cases")
                        
                        # Update dynamic context for the next chunk
                        if result.questions:
                            last_q = result.questions[-1]
                            if last_q.context:
                                dynamic_context = last_q.context
                            else:
                                dynamic_context = ""
                        
                        break
                break
            except Exception as e:
                retry_count += 1
                error_msg = str(e)
                if "429" in error_msg or "rate_limit" in error_msg.lower():
                    wait = min(2 ** retry_count * 5, 60)
                    print(f"[LLM] Rate limited, retrying in {wait}s...")
                    time.sleep(wait)
                elif retry_count >= max_retries:
                    raise
                else:
                    wait = 2 ** retry_count
                    print(f"[LLM] Error: {error_msg}, retrying in {wait}s...")
                    time.sleep(wait)

    return LLMExtractionOutput(
        questions=all_questions,
        clinical_cases=all_cases,
    )


def extract_questions(
    markdown_text: str,
    file_name: str,
    provider: str = "openai",
    api_keys: Optional[list[str]] = None,
) -> LLMExtractionOutput:
    """
    Main entry point — extract QCM questions using the configured LLM provider.
    Automatically chunks large documents and merges results.

    Args:
        markdown_text: Clean markdown text from the document extractor
        file_name: Name of the original file (used for courseName inference)
        provider: "openai" or "anthropic"
        api_key: LLM API key (falls back to env vars)
    """
    print(f"[LLM] Starting extraction (provider={provider}, doc_size={len(markdown_text)} chars)")

    if provider == "gemini":
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        return extract_with_gemini(markdown_text, file_name, model=model)
    elif provider == "openai":
        keys = api_keys or [os.getenv("OPENAI_API_KEY", "")]
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        return extract_with_openai(markdown_text, keys, file_name, model=model)
    elif provider == "groq":
        keys = api_keys or [v for k,v in os.environ.items() if k.startswith("GROQ_API_KEY")]
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        return extract_with_openai(markdown_text, keys, file_name, model=model)
    elif provider == "anthropic":
        keys = api_keys or [os.getenv("ANTHROPIC_API_KEY", "")]
        return extract_with_anthropic(markdown_text, keys[0], file_name)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
