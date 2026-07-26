import sys
import os
import json
from dotenv import load_dotenv
load_dotenv('.env')

# Add medextract to path
sys.path.append(os.getcwd())

from app.llm.structured_output import extract_questions
from app.extractors.docx_extractor import extract_docx

with open("/tmp/caseclinique.docx", "rb") as f:
    markdown_text, _ = extract_docx(f.read())

print("Markdown Length:", len(markdown_text))

result = extract_questions(
    markdown_text=markdown_text,
    file_name="Ex cas clinique 01.docx",
    provider="openai"
)

for case in result.clinical_cases:
    print(f"CASE: {case.name}")
    print(f"INTRO: {case.intro_text[:50]}...")
    print(f"UPDATES: {len(case.updates)}")
    for u in case.updates:
        print(f"  - {u.text[:30]}... -> {u.follows_question_snippet}")
