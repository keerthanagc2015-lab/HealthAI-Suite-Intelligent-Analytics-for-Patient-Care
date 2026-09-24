"""
HEALTHAI - STEP 38O
Extractive Grounded RAG Answer Generation

Purpose:
- Generate answers directly from retrieved authoritative evidence.
- Avoid unsupported generative claims.
- Provide a strong grounded baseline for healthcare RAG.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


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
    / "extractive"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "rag_extractive_results.json"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "rag_extractive_evaluation.csv"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR
    / "rag_extractive_summary.json"
)


# ============================================================
# CONFIG
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RELEVANCE_THRESHOLD = 0.25

MAX_SENTENCES = 3

MIN_SENTENCE_WORDS = 5


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("HEALTHAI - EXTRACTIVE GROUNDED RAG")
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
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("✓ Embedding model loaded.")


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_sentences(text):

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    cleaned = []

    for sentence in sentences:

        sentence = sentence.strip()

        if (
            len(sentence.split())
            >= MIN_SENTENCE_WORDS
        ):

            cleaned.append(sentence)

    return cleaned


# ============================================================
# BUILD EXTRACTIVE ANSWER
# ============================================================

def build_extractive_answer(
    question,
    context
):

    sentences = split_sentences(
        context
    )

    if not sentences:

        return (
            None,
            []
        )

    # --------------------------------------------------------
    # Encode question and candidate sentences
    # --------------------------------------------------------

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )[0]

    sentence_embeddings = model.encode(
        sentences,
        normalize_embeddings=True
    )

    similarities = np.dot(
        sentence_embeddings,
        question_embedding
    )

    # --------------------------------------------------------
    # Rank sentences
    # --------------------------------------------------------

    ranked_indices = np.argsort(
        similarities
    )[::-1]

    selected = []

    for index in ranked_indices:

        sentence = sentences[index]
        score = float(
            similarities[index]
        )

        # Avoid extremely low relevance.
        if score < RELEVANCE_THRESHOLD:
            continue

        # Avoid selecting multiple highly
        # similar duplicate sentences.
        duplicate = False

        sentence_words = set(
            sentence.lower().split()
        )

        for existing in selected:

            existing_words = set(
                existing["sentence"]
                .lower()
                .split()
            )

            if not sentence_words:
                continue

            overlap = (
                len(
                    sentence_words
                    & existing_words
                )
                /
                len(
                    sentence_words
                    | existing_words
                )
            )

            if overlap > 0.75:

                duplicate = True
                break

        if duplicate:
            continue

        selected.append(
            {
                "sentence": sentence,
                "score": score,
            }
        )

        if len(selected) >= MAX_SENTENCES:
            break

    if not selected:

        return (
            None,
            []
        )

    # Preserve source order rather than
    # similarity order for readability.
    selected_texts = [
        x["sentence"]
        for x in selected
    ]

    answer = " ".join(
        selected_texts
    )

    return (
        answer,
        selected
    )


# ============================================================
# PROCESS QUESTIONS
# ============================================================

results = []
evaluation_rows = []

generated_count = 0
abstained_count = 0


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

        selected_sentences = []

        status = "ABSTAIN"

        abstained_count += 1

    else:

        answer, selected = (
            build_extractive_answer(
                question,
                context
            )
        )

        if answer is None:

            answer = (
                "I don't have enough information "
                "in my current medical knowledge "
                "sources to answer that reliably."
            )

            selected_sentences = []

            status = "ABSTAIN"

            abstained_count += 1

        else:

            selected_sentences = selected

            status = "EXTRACTED"

            generated_count += 1

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    results.append(
        {
            "question_number": question_number,
            "question": question,
            "retrieval_decision": retrieval_decision,
            "evidence_count": evidence_count,
            "answer_status": status,
            "answer": answer,
            "selected_sentences": selected_sentences,
        }
    )

    evaluation_rows.append(
        {
            "question_number": question_number,
            "question": question,
            "retrieval_decision": retrieval_decision,
            "evidence_count": evidence_count,
            "answer_status": status,
            "answer": answer,
            "selected_sentence_count": len(
                selected_sentences
            ),
        }
    )


# ============================================================
# SAVE FILES
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

    "step": "38O",

    "method": (
        "Extractive grounded RAG"
    ),

    "embedding_model": (
        EMBEDDING_MODEL
    ),

    "relevance_threshold": (
        RELEVANCE_THRESHOLD
    ),

    "max_sentences": (
        MAX_SENTENCES
    ),

    "test_questions": len(contexts),

    "extracted_answers": (
        generated_count
    ),

    "abstained_questions": (
        abstained_count
    ),

    "answer_generation_rate": (
        generated_count / len(contexts)
        if contexts
        else 0.0
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
print("EXTRACTIVE RAG RESULTS")
print("=" * 75)

for item in results:

    print(
        f"\nQ{item['question_number']}: "
        f"{item['question']}"
    )

    print(
        f"Status: "
        f"{item['answer_status']}"
    )

    print(
        f"Answer: "
        f"{item['answer']}"
    )

    print(
        f"Selected sentences: "
        f"{len(item['selected_sentences'])}"
    )


print("\n" + "=" * 75)
print("SUMMARY")
print("=" * 75)

print(
    f"\nExtracted answers: "
    f"{generated_count}"
)

print(
    f"Abstained: "
    f"{abstained_count}"
)


print("\n" + "=" * 75)
print("STEP 38O - EXTRACTIVE RAG COMPLETED")
print("=" * 75)

print("\nSaved:")
print(OUTPUT_JSON)
print(OUTPUT_CSV)
print(OUTPUT_SUMMARY)