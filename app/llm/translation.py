"""
Question translation (French → English) for the English-track cohorts.

Uses Gemini over raw REST via `_gemini_generate`, like the correction graph —
no SDK is pinned. The point is NOT a literal translation: the output must read
as medical English written for medical students.
"""

from typing import Any, Dict, List, Optional

from app.llm.correction_graph import _gemini_generate

import json
import logging

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """Tu es médecin et enseignant bilingue. Tu produis la version ANGLAISE d'une question de QCM médical rédigée en français, destinée à des étudiants en médecine anglophones.

## Objectif
Ce n'est PAS une traduction littérale. Rédige un anglais médical NATUREL, tel qu'un enseignant anglophone l'aurait écrit : terminologie clinique usuelle, tournures idiomatiques, phrases fluides. Un calque mot à mot du français est un échec.

Exemples de ce qui est attendu :
- « HTA essentielle » → « essential hypertension » (jamais « essential HTA »)
- « bilan biologique » → « laboratory workup » (jamais « biological assessment »)
- « Devant une céphalée fébrile, quels examens demander ? » → « Which investigations are indicated in a patient with fever and headache? »

## Règles ABSOLUES
1. **Ne traduis JAMAIS les libellés** : « A », « B », « C », « 1 », « 2 »… restent identiques.
2. **Ne traduis JAMAIS les combinaisons K-type** : « 1+2 », « 2+3+5 », « 1, 4 » sont recopiées telles quelles.
3. **Ne modifie JAMAIS** les nombres, unités, posologies, valeurs biologiques, noms de molécules ni acronymes internationaux (ECG, IRM → MRI uniquement si l'usage anglais l'impose).
4. **Préserve le HTML** exactement : `<strong>`, `<em>`, `<br>`, `<ul>`, `<li>`, `<p>`, `<a href="…">` et **TOUS les attributs**. Ne traduis que le TEXTE entre les balises. Une explication mal balisée casse l'affichage.
   - Cas critique : `<span data-medical-term-id="12" class="medical-term">terme</span>` marque un terme du glossaire. Recopie la balise et ses attributs **à l'identique** (le `data-medical-term-id` en particulier) et ne traduis que le texte à l'intérieur. Perdre cette balise fait disparaître la définition affichée à l'étudiant.
   - Les URL (`href`, texte des liens) ne se traduisent JAMAIS.
5. **N'ajoute, ne retire, n'invente RIEN** : le contenu médical doit rester strictement équivalent. Pas de commentaire, pas de reformulation du sens, pas de précision supplémentaire.
6. Si un champ est vide ou absent, renvoie une chaîne vide — n'invente pas de contenu.
7. **Exhaustivité — la règle la plus importante** : renvoie TOUJOURS les tableaux `choices` ET `propositions`, avec EXACTEMENT autant d'entrées que reçu, dans le MÊME ordre, chacune avec `label`, `text` et `explanation`. N'en omets aucune, ne les regroupe pas, ne les résume pas. Une entrée manquante rend la traduction inutilisable.

## Ce que tu reçois
Un objet JSON avec l'énoncé, l'éventuel contexte de cas clinique, les propositions et leurs explications.
Tu renvoies le MÊME objet, chaque texte traduit, en respectant les règles ci-dessus."""


def _choice_schema() -> Dict[str, Any]:
    return {
        "type": "OBJECT",
        "properties": {
            # Echoed back untouched so we can map the translation onto the
            # original rows — the model must not renumber anything.
            "label": {"type": "STRING"},
            "text": {"type": "STRING"},
            "explanation": {"type": "STRING"},
        },
        # All three required: an optional field is one Gemini feels free to drop,
        # and a dropped row comes back as an empty English editor.
        "required": ["label", "text", "explanation"],
    }


TRANSLATION_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "description": {"type": "STRING"},
        "caseDescription": {"type": "STRING"},
        "explanation": {"type": "STRING"},
        "choices": {"type": "ARRAY", "items": _choice_schema()},
        "propositions": {"type": "ARRAY", "items": _choice_schema()},
    },
    "required": [
        "description",
        "caseDescription",
        "explanation",
        "choices",
        "propositions",
    ],
}


def _glossary_block(glossary: Optional[List[Dict[str, Any]]]) -> str:
    """
    The imposed lexicon, appended AFTER the general rules so it reads as the
    final word rather than being buried in the middle of the prompt.

    z_api sends only the entries actually present in this question, so this
    block stays short whatever the dictionary's size.
    """
    entries = [e for e in (glossary or []) if e.get("termFr") and e.get("termEn")]
    if not entries:
        return ""
    lines = []
    for e in entries:
        note = f"  ({e['note']})" if e.get("note") else ""
        lines.append(f"- « {e['termFr']} » → « {e['termEn'] }»{note}")
    return (
        "\n\n## LEXIQUE IMPOSÉ (priorité absolue)\n"
        "Pour les termes ci-dessous, tu DOIS employer EXACTEMENT la traduction fournie, "
        "même si une autre formulation te semble préférable. Ce sont les choix validés "
        "par l'équipe pédagogique ; t'en écarter est une erreur.\n"
        + "\n".join(lines)
    )


def _align(payload: Dict[str, Any], result: Dict[str, Any]) -> int:
    """
    Re-align the translation on the ORIGINAL labels and report how many rows
    came back empty while their French source was not.

    A model that drops or reorders an option would otherwise silently corrupt
    the mapping back onto the question, so the labels are never taken from it.
    """
    missing = 0
    for key in ("choices", "propositions"):
        source: List[Dict[str, Any]] = payload.get(key) or []
        translated: List[Dict[str, Any]] = result.get(key) or []
        by_label = {
            str(t.get("label", "")).strip().lower(): t for t in translated
        }
        aligned = []
        for item in source:
            label = str(item.get("label", "")).strip()
            match = by_label.get(label.lower(), {})
            text = (match.get("text") or "").strip()
            explanation = (match.get("explanation") or "").strip()
            # Only French content that EXISTS can go missing in English.
            if (item.get("text") or "").strip() and not text:
                missing += 1
            if (item.get("explanation") or "").strip() and not explanation:
                missing += 1
            aligned.append(
                {
                    # Never trust the model with the label itself.
                    "label": label,
                    "text": text,
                    "explanation": explanation,
                }
            )
        result[key] = aligned
    return missing


def translate_question(
    payload: Dict[str, Any],
    language: str = "en",
    model: Optional[str] = None,
    glossary: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Translate one question payload.

    `payload` mirrors the stored question:
      { description, caseDescription?, explanation?, choices[], propositions[] }
    Returns the same shape, translated.
    """
    target = "anglais" if language == "en" else language
    lexicon = _glossary_block(glossary)
    base_prompt = (
        f"Traduis en {target} la question suivante, en respectant STRICTEMENT "
        f"les règles (libellés, combinaisons, HTML, aucun ajout, AUCUNE omission)"
        + (" et le LEXIQUE IMPOSÉ" if lexicon else "")
        + f" :\n\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )

    # One retry: Gemini occasionally returns a truncated set of rows, and a
    # half-empty translation saved silently is worse than a slow one.
    result: Dict[str, Any] = {}
    missing = 0
    for attempt in (1, 2):
        user_prompt = base_prompt
        if attempt == 2:
            user_prompt = (
                "Ta réponse précédente était INCOMPLÈTE : des entrées de "
                "`choices`/`propositions` manquaient ou étaient vides. Renvoie "
                "la traduction COMPLÈTE, toutes les entrées présentes, dans "
                "l'ordre reçu.\n\n" + base_prompt
            )
        raw = _gemini_generate(
            SYSTEM_PROMPT + lexicon,
            [{"role": "user", "parts": [{"text": user_prompt}]}],
            response_schema=TRANSLATION_SCHEMA,
            model=model,
        )
        if not raw or not raw.strip():
            raise RuntimeError("Empty response from Gemini")
        result = json.loads(raw)
        missing = _align(payload, result)
        if missing == 0:
            break
        logger.warning(
            "Translation attempt %s left %s field(s) untranslated%s",
            attempt,
            missing,
            " — retrying" if attempt == 1 else "",
        )

    if missing:
        # Loud failure: z_api marks the translation FAILED instead of storing
        # blanks that look like a reviewer's deletions.
        raise RuntimeError(
            f"Incomplete translation: {missing} field(s) came back empty"
        )

    return result
