"""
HEALTHAI - STEP 38M
Safety-Aware RAG Answer Generation

Purpose:
- Load safety-filtered evidence.
- Generate answers only when sufficient evidence exists.
- Explicitly abstain for rejected/out-of-domain questions.
- Prevent unsupported medical claims.
"""

import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


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
    / "safety_aware"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "rag_safety_aware_generation_results.json"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "rag_safety_aware_generation_evaluation.csv"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR
    / "rag_safety_aware_generation_summary.json"
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "google/flan-t5-base"

MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 180


# ============================================================
# LOAD CONTEXT
# ============================================================

print("=" * 75)
print("HEALTHAI - SAFETY-AWARE RAG GENERATION")
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

    return f"""
You are a healthcare information assistant.

Answer the user's question using ONLY the medical evidence provided below.

Rules:
- Do not use information that is not supported by the evidence.
- Do not invent facts.
- Do not diagnose the user.
- Do not recommend a personalized treatment plan.
- If the evidence does not directly support an answer, say:
  "I don't have enough information in my current medical knowledge sources to answer that reliably."
- Give a concise, clear, factual answer.
- Prefer information directly relevant to the question.

Medical evidence:
{context}

User question:
{question}

Answer:
""".strip()


# ============================================================
# GENERATION
# ============================================================

results = []
evaluation_rows = []

accepted_questions = 0
abstained_questions = 0

generated_answers = 0


for item in contexts:

    question_number = item.get(
        "question_number"
    )

    question = item.get(
        "question"
    )

    decision = item.get(
        "retrieval_decision"
    )

    context = item.get(
        "context",
        ""
    )

    evidence_count = item.get(
        "accepted_evidence_count",
        0
    )

    # --------------------------------------------------------
    # SAFETY GATE
    # --------------------------------------------------------

    if (
        decision != "ACCEPT"
        or not context.strip()
        or evidence_count == 0
    ):

        answer = (
            "I don't have enough information "
            "in my current medical knowledge "
            "sources to answer that reliably."
        )

        generation_status = "ABSTAIN"

        abstained_questions += 1

    else:

        accepted_questions += 1

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
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )

        answer = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        ).strip()

        if not answer:

            answer = (
                "I don't have enough information "
                "in my current medical knowledge "
                "sources to answer that reliably."
            )

            generation_status = "ABSTAIN_EMPTY_OUTPUT"

        else:

            generation_status = "GENERATED"

            generated_answers += 1

    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    results.append(
        {
            "question_number": question_number,
            "question": question,
            "retrieval_decision": decision,
            "evidence_count": evidence_count,
            "generation_status": generation_status,
            "answer": answer,
            "context": context,
        }
    )

    evaluation_rows.append(
        {
            "question_number": question_number,
            "question": question,
            "retrieval_decision": decision,
            "evidence_count": evidence_count,
            "generation_status": generation_status,
            "answer": answer,
        }
    )


# ============================================================
# SAVE RESULTS
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
# VALIDATION
# ============================================================

abstention_correct = 0

for item in results:

    if item["retrieval_decision"] == "ABSTAIN":

        if item["generation_status"].startswith(
            "ABSTAIN"
        ):
            abstention_correct += 1


abstention_cases = sum(
    1
    for x in results
    if x["retrieval_decision"] == "ABSTAIN"
)

abstention_accuracy = (
    abstention_correct / abstention_cases
    if abstention_cases > 0
    else 0.0
)


summary = {

    "step": "38M",

    "model": MODEL_NAME,

    "device": str(device),

    "test_questions": len(results),

    "relevance_threshold": 0.25,

    "accepted_questions": accepted_questions,

    "abstained_questions": abstained_questions,

    "generated_answers": generated_answers,

    "abstention_correct": abstention_correct,

    "abstention_accuracy": abstention_accuracy,

    "safety_rule": (
        "Only generate when retrieval decision is ACCEPT "
        "and evidence exists."
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
        f"Generation: "
        f"{item['generation_status']}"
    )

    print(
        f"Answer: "
        f"{item['answer']}"
    )


print("\n" + "=" * 75)
print("SAFETY SUMMARY")
print("=" * 75)

print(
    f"\nAccepted questions: "
    f"{accepted_questions}"
)

print(
    f"Abstained questions: "
    f"{abstained_questions}"
)

print(
    f"Generated answers: "
    f"{generated_answers}"
)

print(
    f"Abstention accuracy: "
    f"{abstention_accuracy * 100:.2f}%"
)


print("\n" + "=" * 75)
print("STEP 38M - SAFETY-AWARE GENERATION COMPLETED")
print("=" * 75)

print("\nSaved:")
print(OUTPUT_JSON)
print(OUTPUT_CSV)
print(OUTPUT_SUMMARY)