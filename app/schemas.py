"""
Pydantic schemas for the MedExtract-IA structured output.
These models are used both as:
  1. LLM structured output constraints (via OpenAI/Anthropic)
  2. API response validation for the NestJS callback

K-type (Grouped Choices) Structure:
  Some QCM questions have TWO levels of choices:
    - Sub-propositions: numbered items (1, 2, 3, 4, 5) — the actual statements
    - Propositions: letter-labeled combinations (A: 1+2, B: 1+4, ...) — the answer options
  
  Example:
    25. L'état hyperosmolaire : La ou les réponses justes
    1- Est défini par une hyperglycémie majeure...
    2- Survient surtout chez le sujet âgé grabataire
    3- La présence de cétose est obligatoire...
    A: 1-2  B: 1-4  C: 2-3  D: 2-4  E: 4-5
    Réponse: A  →  Propositions[A].isCorrect=true, choices[0].isCorrect=true, choices[1].isCorrect=true
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class QuestionType(str, Enum):
    UNIQUE_CHOICE = "UNIQUE_CHOICE"
    CLINIC_CASE = "CLINIC_CASE"
    QROC = "QROC"


class LogicType(str, Enum):
    """C1: Logic type flagging — is the question asking for the TRUE or FALSE answer?"""
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"


class Choice(BaseModel):
    """A single choice/sub-proposition in a question"""
    label: str = Field(description="Choice label: letter (A,B,C...) for standard, number (1,2,3...) for K-type sub-propositions")
    text: str = Field(description="Choice text content")
    is_correct: bool = Field(description="Whether this choice is correct")


class Proposition(BaseModel):
    """
    K-type grouped choice (C2).
    Used when a question has numbered sub-propositions and letter-labeled combinations.
    Example: A: 1+2, B: 1+4, C: 2+3
    """
    label: str = Field(description="Letter label (A, B, C, D, E)")
    text: str = Field(description="Combination as 'N+N' format, e.g. '1+2', '1+4', '2+3+5'")
    is_correct: bool = Field(description="Whether this combination is the correct answer")


class ClinicalCasePart(BaseModel):
    """A single part of a clinical case (intro or update)"""
    type: str = Field(description="CASE_INTRO or CASE_UPDATE")
    description: str = Field(description="The clinical narrative text")
    image_url: str = Field(default="", description="Image URL if any")
    from_index: int = Field(description="Index in the question list where this part starts")


class ClinicalCaseGroup(BaseModel):
    """A group of questions belonging to a clinical case"""
    name: str = Field(description="Clinical case name, e.g. 'Cas clinique 1'")
    parts: list[ClinicalCasePart] = Field(description="Ordered list of case parts (intro + updates)")
    question_indices: list[int] = Field(description="Global indices of questions belonging to this case")


class ParsedQuestion(BaseModel):
    """A single parsed QCM question"""
    type: QuestionType
    description: str = Field(description="Question body text with **bold** markers preserved")
    choices: list[Choice] = Field(default_factory=list, description="Standard choices (A-E) or sub-propositions (1-5) for K-type")
    propositions: list[Proposition] = Field(default_factory=list, description="K-type grouped combinations (A:1+2, B:1+4...). Empty for standard questions.")
    is_ktype: bool = Field(default=False, description="True if this is a K-type question with grouped choices")
    correct_answers: str = Field(default="", description="Correct answer labels concatenated, e.g. 'ACE' or 'A'")
    explanation: str = Field(default="", description="Explanation text")
    explanation_urls: list[str] = Field(default_factory=list, description="CDN URLs for explanation images")
    image_url: str = Field(default="", description="CDN URL for question image")
    context: str = Field(default="", description="Clinical case context (C3 propagation)")
    course_name: str = Field(default="", description="Subject/course name")
    logic_type: Optional[LogicType] = Field(default=None, description="C1: POSITIVE or NEGATIVE question")
    where_is_mentioned: list[str] = Field(default_factory=list)
    indication: list[str] = Field(default_factory=list)


class ParseMetadata(BaseModel):
    total_questions: int
    total_images: int
    parsing_duration_ms: float


class ExtractionResult(BaseModel):
    """Full extraction result sent to NestJS callback"""
    job_id: int
    questions: dict[str, list[ParsedQuestion]] = Field(
        description="Questions grouped by course/subject name"
    )
    clinical_case_groups: list[ClinicalCaseGroup] = Field(default_factory=list)
    metadata: ParseMetadata


# ─── LLM Prompt Schema (what we send to the LLM) ────────────────────────

class LLMQuestionOutput(BaseModel):
    """Schema for structured output from the LLM"""
    type: QuestionType
    description: str = Field(description="Full question text preserving **bold** formatting. For K-type questions, include ONLY the question stem, not the numbered sub-propositions.")
    choices: list[Choice] = Field(description="For standard questions: A/B/C/D/E choices. For K-type: the numbered sub-propositions (label='1','2','3'...) with their statement text.")
    propositions: list[Proposition] = Field(default_factory=list, description="ONLY for K-type questions: the letter-labeled combinations. E.g. [{label:'A', text:'1+2', is_correct:true}, {label:'B', text:'1+4', is_correct:false}]. Leave empty [] for standard questions.")
    is_ktype: bool = Field(default=False, description="Set to true ONLY when the question has numbered sub-propositions (1,2,3...) followed by letter-labeled combinations (A:1-2, B:1-4...). Standard A/B/C/D/E questions are NOT K-type.")
    correct_answers: str = Field(description="Letters of correct choices/propositions, e.g. 'ACE' or 'A'")
    explanation: str = Field(default="", description="The FULL explanation text. PRESERVE ALL Markdown formatting exactly as it appears (bolding `**`, italics `*`, bullet points `-`, line breaks `<br>`). CRITICAL: Do NOT truncate long explanations, capture them in their entirety!")
    logic_type: LogicType = Field(description="POSITIVE if asking for true/correct, NEGATIVE if asking for false/incorrect")
    course_name: str = Field(default="", description="Subject or course this question belongs to")
    context: str = Field(default="", description="Clinical case narrative if this is part of a clinical case")
    is_clinical_case_child: bool = Field(default=False, description="True if this question depends on a clinical case context")
    clinical_case_id: Optional[int] = Field(default=None, description="Index of the clinical case group this belongs to")
    where_is_mentioned: list[str] = Field(default_factory=list, description="Source references like 'P2 2024 T35', 'Constantine 2023'")
    indication: list[str] = Field(default_factory=list)


class LLMClinicalCase(BaseModel):
    """A clinical case detected by the LLM. Do NOT extract course names, subjects, or simple section headings (e.g. 'Oncologie') as clinical cases."""
    name: str = Field(description="Name of the case, e.g. 'Cas clinique N°1' or patient initials if no number is given. Do NOT use subject names.")
    intro_text: str = Field(description="The FULL initial clinical narrative text. Must be an actual clinical scenario, NOT a section heading.")


class LLMExtractionOutput(BaseModel):
    """Top-level schema for LLM structured output"""
    questions: list[LLMQuestionOutput]
    clinical_cases: list[LLMClinicalCase] = Field(default_factory=list)
