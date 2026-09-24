"""
HEALTHAI - STEP 38Q
Final RAG Evaluation

Evaluates the complete RAG pipeline:

1. Retrieval
2. Safety / abstention
3. Context building
4. Generative RAG
5. Clean extractive RAG

This version reads the actual artifacts produced
by the previous HealthAI RAG steps.
"""

import json
from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RETRIEVAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rag"
    / "retrieval"
    / "evaluation"
    / "rag_retrieval_method_comparison.csv"
)

SAFETY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rag"
    / "retrieval"
    / "safety"
    / "rag_retrieval_safety_threshold_comparison.csv"
)

CONTEXT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rag"
    / "context"
    / "rag_safety_aware_context_evaluation.csv"
)

GENERATIVE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rag"
    / "generation"
    / "safety_aware"
    / "rag_safety_aware_generation_evaluation.csv"
)

CLEAN_EXTRACTIVE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rag"
    / "generation"
    / "clean_extractive"
    / "rag_clean_extractive_evaluation.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rag"
    / "final_evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "rag_final_evaluation.csv"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR
    / "rag_final_evaluation_summary.json"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("HEALTHAI - FINAL RAG EVALUATION")
print("=" * 75)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading retrieval evaluation...")

retrieval_df = pd.read_csv(
    RETRIEVAL_PATH
)

print("✓ Retrieval evaluation loaded.")

print(
    f"Retrieval columns: "
    f"{list(retrieval_df.columns)}"
)


print("\nLoading safety evaluation...")

safety_df = pd.read_csv(
    SAFETY_PATH
)

print("✓ Safety evaluation loaded.")


print("\nLoading context evaluation...")

context_df = pd.read_csv(
    CONTEXT_PATH
)

print("✓ Context evaluation loaded.")


print("\nLoading generative evaluation...")

generative_df = pd.read_csv(
    GENERATIVE_PATH
)

print("✓ Generative evaluation loaded.")


print("\nLoading clean extractive evaluation...")

extractive_df = pd.read_csv(
    CLEAN_EXTRACTIVE_PATH
)

print("✓ Clean extractive evaluation loaded.")


# ============================================================
# RETRIEVAL EVALUATION
# ============================================================

print("\n" + "=" * 75)
print("RETRIEVAL")
print("=" * 75)

# The Step 38J comparison file stores
# one row per method and uses these actual
# column names.

question_aware_rows = retrieval_df[
    retrieval_df["method"].astype(str).str.lower()
    == "question_aware"
]

if question_aware_rows.empty:

    raise ValueError(
        "Could not find 'question_aware' row "
        "in retrieval comparison CSV."
    )

retrieval_row = question_aware_rows.iloc[0]


def find_column(df, candidates):
    """
    Find the first available column from a list.
    """

    for column in candidates:

        if column in df.columns:
            return column

    return None


top1_col = find_column(
    retrieval_df,
    [
        "Top-1 topic accuracy",
        "top1_topic_accuracy",
        "top_1_topic_accuracy",
        "top1_accuracy",
    ]
)

top3_col = find_column(
    retrieval_df,
    [
        "Top-3 topic hit rate",
        "top3_topic_hit_rate",
        "top_3_topic_hit_rate",
        "top3_hit_rate",
    ]
)

score_col = find_column(
    retrieval_df,
    [
        "Mean Top-1 score",
        "mean_top1_score",
        "mean_top_1_score",
    ]
)

consistency_col = find_column(
    retrieval_df,
    [
        "Mean topic consistency",
        "mean_topic_consistency",
        "topic_consistency",
    ]
)

ood_col = find_column(
    retrieval_df,
    [
        "Out-of-domain rejection",
        "out_of_domain_rejection",
        "out_of_domain_rejection_rate",
    ]
)


print(
    "\nAvailable retrieval columns:"
)

print(
    list(retrieval_df.columns)
)


# ------------------------------------------------------------
# Robust numeric extraction
# ------------------------------------------------------------

def numeric_value(row, column):

    if column is None:
        return None

    value = row[column]

    if pd.isna(value):
        return None

    if isinstance(value, str):

        value = (
            value
            .replace("%", "")
            .strip()
        )

    try:

        value = float(value)

    except (TypeError, ValueError):

        return None

    # If stored as percentage rather than
    # decimal, convert to decimal.
    if value > 1.0:

        value = value / 100.0

    return value


retrieval_top1 = numeric_value(
    retrieval_row,
    top1_col
)

retrieval_top3 = numeric_value(
    retrieval_row,
    top3_col
)

retrieval_mean_score = numeric_value(
    retrieval_row,
    score_col
)

retrieval_consistency = numeric_value(
    retrieval_row,
    consistency_col
)

retrieval_ood = numeric_value(
    retrieval_row,
    ood_col
)


print(
    f"\nTop-1 topic accuracy: "
    f"{retrieval_top1 * 100:.2f}%"
    if retrieval_top1 is not None
    else
    "Top-1 topic accuracy: unavailable"
)

print(
    f"Top-3 topic hit rate: "
    f"{retrieval_top3 * 100:.2f}%"
    if retrieval_top3 is not None
    else
    "Top-3 topic hit rate: unavailable"
)

print(
    f"Mean Top-1 score: "
    f"{retrieval_mean_score:.4f}"
    if retrieval_mean_score is not None
    else
    "Mean Top-1 score: unavailable"
)

print(
    f"Topic consistency: "
    f"{retrieval_consistency * 100:.2f}%"
    if retrieval_consistency is not None
    else
    "Topic consistency: unavailable"
)


# ============================================================
# SAFETY EVALUATION
# ============================================================

print("\n" + "=" * 75)
print("SAFETY")
print("=" * 75)

safety_threshold_rows = safety_df[
    safety_df["threshold"].round(4) == 0.25
]

if safety_threshold_rows.empty:

    raise ValueError(
        "Could not find threshold 0.25 "
        "in safety evaluation."
    )

safety_row = (
    safety_threshold_rows
    .iloc[0]
)


def safe_float(row, column):

    value = row[column]

    if pd.isna(value):
        return None

    return float(value)


in_domain_acceptance = safe_float(
    safety_row,
    "in_domain_acceptance_rate"
)

in_domain_topic_accuracy = safe_float(
    safety_row,
    "in_domain_topic_accuracy"
)

ood_rejection = safe_float(
    safety_row,
    "out_of_domain_rejection_rate"
)

safety_score = safe_float(
    safety_row,
    "safety_score"
)


print(
    "Relevance threshold: 0.25"
)

print(
    f"In-domain acceptance: "
    f"{in_domain_acceptance * 100:.2f}%"
)

print(
    f"In-domain topic accuracy: "
    f"{in_domain_topic_accuracy * 100:.2f}%"
)

print(
    f"Out-of-domain rejection: "
    f"{ood_rejection * 100:.2f}%"
)

print(
    f"Safety score: "
    f"{safety_score * 100:.2f}%"
)


# ============================================================
# CONTEXT EVALUATION
# ============================================================

print("\n" + "=" * 75)
print("CONTEXT BUILDER")
print("=" * 75)

accepted_contexts = context_df[
    context_df[
        "retrieval_decision"
    ].astype(str).str.upper()
    == "ACCEPT"
]

abstained_contexts = context_df[
    context_df[
        "retrieval_decision"
    ].astype(str).str.upper()
    == "ABSTAIN"
]

context_acceptance_rate = (
    len(accepted_contexts)
    /
    len(context_df)
)

context_abstention_rate = (
    len(abstained_contexts)
    /
    len(context_df)
)

print(
    f"Accepted contexts: "
    f"{len(accepted_contexts)}/{len(context_df)}"
)

print(
    f"Context acceptance rate: "
    f"{context_acceptance_rate * 100:.2f}%"
)

print(
    f"Context abstention rate: "
    f"{context_abstention_rate * 100:.2f}%"
)


# ============================================================
# GENERATIVE RAG
# ============================================================

print("\n" + "=" * 75)
print("GENERATIVE RAG")
print("=" * 75)

generative_status = (
    generative_df[
        "generation_status"
    ].astype(str).str.upper()
)

generated = generative_df[
    generative_status == "GENERATED"
]

invalid = generative_df[
    generative_status == "INVALID_GENERATION"
]

generative_abstained = generative_df[
    generative_status.str.startswith(
        "ABSTAIN"
    )
]

generation_success_rate = (
    len(generated)
    /
    len(generative_df)
)

print(
    f"Valid generated answers: "
    f"{len(generated)}/{len(generative_df)}"
)

print(
    f"Generation success rate: "
    f"{generation_success_rate * 100:.2f}%"
)

print(
    f"Invalid generations: "
    f"{len(invalid)}"
)

print(
    f"Abstentions: "
    f"{len(generative_abstained)}"
)


# ============================================================
# CLEAN EXTRACTIVE RAG
# ============================================================

print("\n" + "=" * 75)
print("CLEAN EXTRACTIVE RAG")
print("=" * 75)

extractive_status = (
    extractive_df[
        "answer_status"
    ].astype(str).str.upper()
)

extractive_answers = extractive_df[
    extractive_status == "EXTRACTED"
]

extractive_abstained = extractive_df[
    extractive_status == "ABSTAIN"
]

extractive_rate = (
    len(extractive_answers)
    /
    len(extractive_df)
)

print(
    f"Extracted answers: "
    f"{len(extractive_answers)}/"
    f"{len(extractive_df)}"
)

print(
    f"Extraction rate: "
    f"{extractive_rate * 100:.2f}%"
)

print(
    f"Abstentions: "
    f"{len(extractive_abstained)}"
)


# ============================================================
# OUT-OF-DOMAIN SAFETY
# ============================================================

print("\n" + "=" * 75)
print("OUT-OF-DOMAIN SAFETY")
print("=" * 75)

ood_context_rows = context_df[
    context_df[
        "original_answerable"
    ].astype(str).str.lower()
    == "false"
]

ood_correctly_abstained = (
    ood_context_rows[
        ood_context_rows[
            "retrieval_decision"
        ].astype(str).str.upper()
        == "ABSTAIN"
    ]
)

if len(ood_context_rows) > 0:

    ood_context_rejection = (
        len(ood_correctly_abstained)
        /
        len(ood_context_rows)
    )

else:

    ood_context_rejection = 0.0


print(
    f"Out-of-domain questions: "
    f"{len(ood_context_rows)}"
)

print(
    f"Correctly rejected: "
    f"{len(ood_correctly_abstained)}"
)

print(
    f"Out-of-domain rejection: "
    f"{ood_context_rejection * 100:.2f}%"
)


# ============================================================
# BUILD FINAL METRICS
# ============================================================

print("\n" + "=" * 75)
print("FINAL RAG SYSTEM")
print("=" * 75)


final_rows = []


def add_metric(
    component,
    metric,
    value
):

    final_rows.append(
        {
            "component": component,
            "metric": metric,
            "value": value,
            "percentage": (
                value * 100
                if value is not None
                else None
            ),
        }
    )


add_metric(
    "Question-aware retrieval",
    "Top-1 topic accuracy",
    retrieval_top1
)

add_metric(
    "Question-aware retrieval",
    "Top-3 topic hit rate",
    retrieval_top3
)

add_metric(
    "Question-aware retrieval",
    "Mean topic consistency",
    retrieval_consistency
)

add_metric(
    "Safety layer",
    "In-domain acceptance",
    in_domain_acceptance
)

add_metric(
    "Safety layer",
    "Out-of-domain rejection",
    ood_rejection
)

add_metric(
    "Safety layer",
    "Safety score",
    safety_score
)

add_metric(
    "Context builder",
    "Context acceptance rate",
    context_acceptance_rate
)

add_metric(
    "Generative RAG",
    "Generation success rate",
    generation_success_rate
)

add_metric(
    "Clean extractive RAG",
    "Extraction rate",
    extractive_rate
)


final_df = pd.DataFrame(
    final_rows
)

final_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# METHOD SELECTION
# ============================================================

"""
For the current portfolio prototype:

- Generative RAG is retained as an experimental component.
- Clean extractive RAG is preferred as the grounded answer
  baseline because it directly uses source evidence.
- The safety gate remains mandatory regardless of answer method.
"""

recommended_method = (
    "clean_extractive_grounded_rag"
)


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "step": "38Q",

    "retrieval_method": (
        "question_aware"
    ),

    "relevance_threshold": 0.25,

    "retrieval": {
        "top1_topic_accuracy": retrieval_top1,
        "top3_topic_hit_rate": retrieval_top3,
        "mean_top1_score": retrieval_mean_score,
        "topic_consistency": retrieval_consistency,
    },

    "safety": {
        "in_domain_acceptance": (
            in_domain_acceptance
        ),
        "in_domain_topic_accuracy": (
            in_domain_topic_accuracy
        ),
        "out_of_domain_rejection": (
            ood_rejection
        ),
        "safety_score": safety_score,
    },

    "context": {
        "acceptance_rate": (
            context_acceptance_rate
        ),
        "abstention_rate": (
            context_abstention_rate
        ),
    },

    "generative_rag": {
        "success_rate": (
            generation_success_rate
        ),
        "invalid_generations": (
            len(invalid)
        ),
        "abstentions": (
            len(generative_abstained)
        ),
    },

    "clean_extractive_rag": {
        "answer_rate": (
            extractive_rate
        ),
        "abstentions": (
            len(extractive_abstained)
        ),
    },

    "recommended_answering_method": (
        recommended_method
    ),

    "architecture_status": (
        "RAG retrieval, safety gating, "
        "context construction and grounded "
        "answering have been implemented."
    ),

    "important_limitation": (
        "Evaluation uses a small controlled "
        "7-question test set and is not clinical validation."
    ),

    "files": {
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
# FINAL REPORT
# ============================================================

print("\nRecommended answering method:")
print(
    recommended_method
)

print(
    "\nThe RAG module evaluation is complete."
)

print("\nSaved:")
print(OUTPUT_CSV)
print(OUTPUT_SUMMARY)

print("\n" + "=" * 75)
print("STEP 38Q - FINAL RAG EVALUATION COMPLETED")
print("=" * 75)