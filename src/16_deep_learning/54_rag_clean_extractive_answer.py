"""
HEALTHAI - STEP 38P
Clean Extractive Grounded RAG

Purpose:
- Use safety-aware retrieved evidence.
- Extract only source-content text.
- Remove RAG metadata artifacts.
- Select the most relevant complete sentences.
- Preserve abstention for unsupported questions.
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
    / "clean_extractive"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "rag_clean_extractive_results.json"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "rag_clean_extractive_evaluation.csv"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR
    / "rag_clean_extractive_summary.json"
)


# ============================================================
# CONFIG
# ============================================================

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

RELEVANCE_THRESHOLD = 0.25

SENTENCE_SELECTION_THRESHOLD = 0.35

MAX_SENTENCES = 3

MIN_SENTENCE_WORDS = 6

MAX_SENTENCE_WORDS = 80


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("HEALTHAI - CLEAN EXTRACTIVE GROUNDED RAG")
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

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("✓ Embedding model loaded.")


# ============================================================
# CLEAN CONTEXT
# ============================================================

def extract_source_content(context):
    """
    Remove RAG metadata and retain only actual source text.

    Expected format:

    [Evidence 1]
    Source: ...
    Topic: ...
    Relevance score: ...
    Content:
    ACTUAL SOURCE TEXT
    """

    if not context:
        return ""

    blocks = re.split(
        r"\[Evidence\s+\d+\]",
        context,
        flags=re.IGNORECASE
    )

    cleaned_blocks = []

    for block in blocks:

        block = block.strip()

        if not block:
            continue

        # Keep only text after Content:
        content_match = re.search(
            r"Content:\s*(.*)",
            block,
            flags=re.IGNORECASE | re.DOTALL
        )

        if content_match:

            content = content_match.group(1)

        else:

            # Fallback: remove known metadata lines.
            lines = block.splitlines()

            content_lines = []

            for line in lines:

                normalized = line.strip().lower()

                if (
                    normalized.startswith("source:")
                    or normalized.startswith("topic:")
                    or normalized.startswith(
                        "relevance score:"
                    )
                ):
                    continue

                content_lines.append(line)

            content = "\n".join(
                content_lines
            )

        content = re.sub(
            r"\s+",
            " ",
            content
        ).strip()

        if content:
            cleaned_blocks.append(content)

    return "\n".join(
        cleaned_blocks
    )


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

    # Standard sentence boundaries.
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    cleaned = []

    for sentence in sentences:

        sentence = sentence.strip()

        words = sentence.split()

        if len(words) < MIN_SENTENCE_WORDS:
            continue

        if len(words) > MAX_SENTENCE_WORDS:
            continue

        # Remove obvious navigation/webpage artifacts.
        lower = sentence.lower()

        artifact_patterns = [
            "accessed september",
            "take control of",
            "skip to",
            "menu",
            "home page",
            "sign in",
        ]

        if any(
            pattern in lower
            for pattern in artifact_patterns
        ):
            continue

        cleaned.append(sentence)

    return cleaned


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_for_duplicate(text):

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# SELECT SENTENCES
# ============================================================

def select_relevant_sentences(
    question,
    source_text
):

    sentences = split_sentences(
        source_text
    )

    if not sentences:
        return []

    question_embedding = (
        embedding_model.encode(
            [question],
            normalize_embeddings=True
        )[0]
    )

    sentence_embeddings = (
        embedding_model.encode(
            sentences,
            normalize_embeddings=True
        )
    )

    similarities = np.dot(
        sentence_embeddings,
        question_embedding
    )

    ranked_indices = np.argsort(
        similarities
    )[::-1]

    selected = []

    for index in ranked_indices:

        sentence = sentences[index]

        score = float(
            similarities[index]
        )

        if score < SENTENCE_SELECTION_THRESHOLD:
            continue

        normalized = normalize_for_duplicate(
            sentence
        )

        duplicate = False

        for existing in selected:

            if normalized == existing["normalized"]:
                duplicate = True
                break

            # Token Jaccard similarity.
            a = set(
                normalized.split()
            )

            b = set(
                existing[
                    "normalized"
                ].split()
            )

            if a and b:

                jaccard = (
                    len(a & b)
                    /
                    len(a | b)
                )

                if jaccard >= 0.75:
                    duplicate = True
                    break

        if duplicate:
            continue

        selected.append(
            {
                "sentence": sentence,
                "score": score,
                "normalized": normalized,
            }
        )

        if len(selected) >= MAX_SENTENCES:
            break

    # Return in relevance order.
    return selected


# ============================================================
# PROCESS QUESTIONS
# ============================================================

results = []
evaluation_rows = []

extracted_count = 0
abstained_count = 0

total_source_chars = 0
total_selected_sentences = 0


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

        selected = []

        status = "ABSTAIN"

        abstained_count += 1

        source_text = ""

    else:

        # ----------------------------------------------------
        # REMOVE METADATA ARTIFACTS
        # ----------------------------------------------------

        source_text = extract_source_content(
            context
        )

        total_source_chars += len(
            source_text
        )

        # ----------------------------------------------------
        # SELECT RELEVANT SENTENCES
        # ----------------------------------------------------

        selected = (
            select_relevant_sentences(
                question,
                source_text
            )
        )

        if not selected:

            answer = (
                "I don't have enough information "
                "in my current medical knowledge "
                "sources to answer that reliably."
            )

            status = "ABSTAIN"

            abstained_count += 1

        else:

            # Keep only actual source sentences.
            answer = " ".join(
                x["sentence"]
                for x in selected
            )

            status = "EXTRACTED"

            extracted_count += 1

            total_selected_sentences += len(
                selected
            )

    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    results.append(
        {
            "question_number": question_number,
            "question": question,
            "retrieval_decision": retrieval_decision,
            "evidence_count": evidence_count,
            "answer_status": status,
            "answer": answer,
            "selected_sentences": selected,
            "clean_source_text": source_text,
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
                selected
            ),
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
# SUMMARY
# ============================================================

summary = {

    "step": "38P",

    "method": (
        "Clean extractive grounded RAG"
    ),

    "embedding_model": (
        EMBEDDING_MODEL
    ),

    "retrieval_relevance_threshold": (
        RELEVANCE_THRESHOLD
    ),

    "sentence_selection_threshold": (
        SENTENCE_SELECTION_THRESHOLD
    ),

    "max_sentences": (
        MAX_SENTENCES
    ),

    "test_questions": len(contexts),

    "extracted_answers": extracted_count,

    "abstained_questions": abstained_count,

    "answer_generation_rate": (
        extracted_count / len(contexts)
        if contexts
        else 0.0
    ),

    "total_clean_source_characters": (
        total_source_chars
    ),

    "total_selected_sentences": (
        total_selected_sentences
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
print("CLEAN EXTRACTIVE RAG RESULTS")
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
    f"{extracted_count}"
)

print(
    f"Abstained: "
    f"{abstained_count}"
)

print(
    f"Total clean source characters: "
    f"{total_source_chars}"
)

print(
    f"Total selected sentences: "
    f"{total_selected_sentences}"
)


print("\n" + "=" * 75)
print("STEP 38P - CLEAN EXTRACTIVE RAG COMPLETED")
print("=" * 75)

print("\nSaved:")
print(OUTPUT_JSON)
print(OUTPUT_CSV)
print(OUTPUT_SUMMARY)