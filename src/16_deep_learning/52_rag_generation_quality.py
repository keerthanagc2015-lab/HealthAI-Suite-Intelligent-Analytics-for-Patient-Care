"""
HEALTHAI - STEP 38N
RAG Generation Quality Improvement

Purpose:
- Improve FLAN-T5 grounded answer generation.
- Reduce prompt echoing.
- Validate generated answers.
- Preserve safety-aware abstention.
"""

import json
from pathlib import Path

import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rag"
    / "context"
    / "rag_safety_aware_contexts.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rag"
    / "generation"
    / "quality"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "rag_generation_quality_results.json"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "rag_generation_quality_evaluation.csv"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR
    / "rag_generation_quality_summary.json"
)


# ============================================================
# MODEL CONFIG
# ============================================================

MODEL_NAME = "google/flan-t5-base"

MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 120


# ============================================================
# LOAD CONTEXT
# ============================================================

print("=" * 75)
print("HEALTHAI - RAG GENERATION QUALITY")
print("=" * 75)

print("\nLoading safety-aware contexts...")

with open(
    INPUT_PATH,
    "r",
    encoding="utf-8"
) as f:
    contexts = json.load(f)

print("✓ Contexts loaded.")
print(f"Test questions: {len(contexts)}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading FLAN-T5-base...")

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Device: {device}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)

model.to(device)
model.eval()

print("✓ Generator loaded.")


# ============================================================
# PROMPT
# ============================================================

def build_prompt(question, context):

    return (
        "context: "
        + context
        + "\n\n"
        + "question: "
        + question
        + "\n\n"
        + "answer:"
    )


# ============================================================
# INVALID ANSWER DETECTION
# ============================================================

INVALID_PATTERNS = [
    "prefer information directly relevant",
    "user question",
    "medical evidence",
    "answer:",
    "context:",
    "i cannot answer",
]


def is_invalid_answer(answer):

    normalized = answer.lower().strip()

    if not normalized:
        return True

    for pattern in INVALID_PATTERNS:

        if pattern in normalized:
            return True

    # Very short generic outputs are not useful
    # for this evaluation.
    if len(normalized.split()) <= 2:
        return True

    return False


# ============================================================
# GENERATION
# ============================================================

results = []
evaluation_rows = []

generated_count = 0
abstained_count = 0
invalid_count = 0


for item in contexts:

    question_number = item.get(
        "question_number"
    )

    question = item.get(
        "question"
    )

    retrieval_decision = item.get(
        "retrieval_decision"
    )

    evidence_count = item.get(
        "accepted_evidence_count",
        0
    )

    context = item.get(
        "context",
        ""
    )

    # --------------------------------------------------------
    # SAFETY GATE
    # --------------------------------------------------------

    if (
        retrieval_decision != "ACCEPT"
        or evidence_count == 0
        or not context.strip()
    ):

        answer = (
            "I don't have enough information "
            "in my current medical knowledge "
            "sources to answer that reliably."
        )

        status = "ABSTAIN"

        abstained_count += 1

    else:

        prompt = build_prompt(
            question,
            context
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_LENGTH
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=5,
                do_sample=False,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        answer = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        ).strip()

        # ----------------------------------------------------
        # OUTPUT VALIDATION
        # ----------------------------------------------------

        if is_invalid_answer(answer):

            status = "INVALID_GENERATION"

            invalid_count += 1

        else:

            status = "GENERATED"

            generated_count += 1

    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    results.append(
        {
            "question_number": question_number,
            "question": question,
            "retrieval_decision": retrieval_decision,
            "evidence_count": evidence_count,
            "generation_status": status,
            "answer": answer,
        }
    )

    evaluation_rows.append(
        {
            "question_number": question_number,
            "question": question,
            "retrieval_decision": retrieval_decision,
            "evidence_count": evidence_count,
            "generation_status": status,
            "answer": answer,
            "answer_word_count": len(
                answer.split()
            ),
        }
    )


# ============================================================
# SAVE
# ============================================================

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )


evaluation_df = pd.DataFrame(
    evaluation_rows
)

evaluation_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "step": "38N",

    "model": MODEL_NAME,

    "device": str(device),

    "test_questions": len(contexts),

    "generated_count": generated_count,

    "abstained_count": abstained_count,

    "invalid_generation_count": invalid_count,

    "generation_success_rate": (
        generated_count / len(contexts)
        if contexts
        else 0.0
    ),

    "safety_abstention_preserved": (
        abstained_count >= 1
    ),

    "files": {
        "results": str(OUTPUT_JSON),
        "evaluation": str(OUTPUT_CSV),
        "summary": str(OUTPUT_SUMMARY),
    },
}


with open(
    OUTPUT_SUMMARY,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 75)
print("GENERATION RESULTS")
print("=" * 75)

for item in results:

    print(
        f"\nQ{item['question_number']}: "
        f"{item['question']}"
    )

    print(
        f"Retrieval: "
        f"{item['retrieval_decision']}"
    )

    print(
        f"Evidence: "
        f"{item['evidence_count']}"
    )

    print(
        f"Status: "
        f"{item['generation_status']}"
    )

    print(
        f"Answer: "
        f"{item['answer']}"
    )


print("\n" + "=" * 75)
print("GENERATION SUMMARY")
print("=" * 75)

print(
    f"\nGenerated answers: "
    f"{generated_count}"
)

print(
    f"Abstained: "
    f"{abstained_count}"
)

print(
    f"Invalid generations: "
    f"{invalid_count}"
)


print("\n" + "=" * 75)
print("STEP 38N - GENERATION QUALITY COMPLETED")
print("=" * 75)

print("\nSaved:")
print(OUTPUT_JSON)
print(OUTPUT_CSV)
print(OUTPUT_SUMMARY)