"""
HEALTHAI - STEP 38K
RAG Retrieval Safety and Abstention Evaluation

Purpose:
- Evaluate relevance thresholds on top of question-aware retrieval.
- Prevent weak/out-of-domain evidence from reaching the generator.
- Preserve strong in-domain retrieval.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rag"
    / "retrieval"
    / "improved"
    / "rag_improved_retrieval_results.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rag"
    / "retrieval"
    / "safety"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

# Test several thresholds.
THRESHOLDS = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
]

# For a question to be considered safely answerable,
# the final evidence must contain at least one result
# above the selected threshold.
#
# We will evaluate both:
#   - Top-1 score
#   - Final evidence scores
#
# The final recommendation will prioritize:
#   1. Out-of-domain rejection
#   2. In-domain topic recall
#   3. Avoiding excessive rejection


# ============================================================
# LOAD RESULTS
# ============================================================

print("=" * 75)
print("HEALTHAI - RAG RETRIEVAL SAFETY / ABSTENTION")
print("=" * 75)

print("\nLoading question-aware retrieval results...")

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print("✓ Question-aware retrieval loaded.")

# Handle either direct list or wrapped dictionary.
if isinstance(data, dict):
    if "results" in data:
        results = data["results"]
    elif "test_results" in data:
        results = data["test_results"]
    else:
        raise ValueError(
            "Could not find retrieval results in JSON. "
            "Expected 'results' or 'test_results'."
        )
else:
    results = data

print(f"Test questions: {len(results)}")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_expected_topics(item):
    topics = item.get("expected_topics", [])

    if isinstance(topics, str):
        topics = [topics]

    return [str(x).lower() for x in topics]


def get_final_evidence(item):
    evidence = item.get("final_evidence", [])

    if evidence is None:
        return []

    if not isinstance(evidence, list):
        return []

    return evidence


def get_score(evidence_item):
    """
    Support the score naming used by the previous retrieval step.
    """
    for key in [
        "rerank_score",
        "final_score",
        "score",
        "similarity",
        "semantic_score",
    ]:
        value = evidence_item.get(key)

        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    return None


def get_topic(evidence_item):
    """
    Retrieve topic from either direct topic field
    or nested metadata.
    """
    topic = evidence_item.get("topic")

    if topic is None:
        metadata = evidence_item.get("metadata", {})
        if isinstance(metadata, dict):
            topic = metadata.get("topic")

    if topic is None:
        return ""

    return str(topic).lower()


def topic_matches(expected_topics, retrieved_topic):
    if not expected_topics or not retrieved_topic:
        return False

    for expected in expected_topics:
        if expected in retrieved_topic or retrieved_topic in expected:
            return True

    return False


# ============================================================
# BASELINE TEST INFORMATION
# ============================================================

print("\n" + "=" * 75)
print("TEST CASE OVERVIEW")
print("=" * 75)

for item in results:
    q_no = item.get("question_number")
    question = item.get("question")
    answerable = item.get("answerable", True)

    evidence = get_final_evidence(item)

    scores = [
        get_score(x)
        for x in evidence
        if get_score(x) is not None
    ]

    top_score = max(scores) if scores else None

    print(
        f"\nQ{q_no}: {question}"
        f"\nAnswerable: {answerable}"
        f"\nTop final score: "
        f"{top_score:.4f}" if top_score is not None
        else
        f"\nQ{q_no}: {question}"
        f"\nAnswerable: {answerable}"
        f"\nTop final score: None"
    )


# ============================================================
# EVALUATE THRESHOLDS
# ============================================================

all_threshold_results = []
case_results = []

print("\n" + "=" * 75)
print("THRESHOLD EVALUATION")
print("=" * 75)

for threshold in THRESHOLDS:

    answerable_cases = [
        x for x in results
        if bool(x.get("answerable", True))
    ]

    out_of_domain_cases = [
        x for x in results
        if not bool(x.get("answerable", True))
    ]

    in_domain_total = len(answerable_cases)
    out_domain_total = len(out_of_domain_cases)

    in_domain_accepted = 0
    in_domain_correct_topic = 0

    out_domain_rejected = 0

    threshold_case_rows = []

    for item in results:

        q_no = item.get("question_number")
        question = item.get("question")
        answerable = bool(item.get("answerable", True))
        expected_topics = get_expected_topics(item)

        evidence = get_final_evidence(item)

        scored_evidence = []

        for ev in evidence:
            score = get_score(ev)

            if score is None:
                continue

            scored_evidence.append(
                {
                    "topic": get_topic(ev),
                    "score": score,
                    "raw": ev,
                }
            )

        # ----------------------------------------------------
        # SAFETY DECISION
        # ----------------------------------------------------

        accepted_evidence = [
            ev
            for ev in scored_evidence
            if ev["score"] >= threshold
        ]

        accepted = len(accepted_evidence) > 0

        if accepted:
            top_accepted = max(
                accepted_evidence,
                key=lambda x: x["score"]
            )

            top_topic = top_accepted["topic"]

            topic_correct = topic_matches(
                expected_topics,
                top_topic
            )
        else:
            top_accepted = None
            top_topic = ""
            topic_correct = False

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        if answerable:

            if accepted:
                in_domain_accepted += 1

            if accepted and topic_correct:
                in_domain_correct_topic += 1

        else:

            if not accepted:
                out_domain_rejected += 1

        threshold_case_rows.append(
            {
                "threshold": threshold,
                "question_number": q_no,
                "question": question,
                "answerable": answerable,
                "expected_topics": expected_topics,
                "accepted": accepted,
                "accepted_evidence_count": len(accepted_evidence),
                "top_accepted_score": (
                    top_accepted["score"]
                    if top_accepted
                    else None
                ),
                "top_accepted_topic": top_topic,
                "topic_correct": topic_correct,
            }
        )

    # --------------------------------------------------------
    # AGGREGATE METRICS
    # --------------------------------------------------------

    if in_domain_total > 0:
        in_domain_acceptance_rate = (
            in_domain_accepted / in_domain_total
        )

        in_domain_topic_accuracy = (
            in_domain_correct_topic / in_domain_total
        )
    else:
        in_domain_acceptance_rate = 0.0
        in_domain_topic_accuracy = 0.0

    if out_domain_total > 0:
        out_domain_rejection_rate = (
            out_domain_rejected / out_domain_total
        )
    else:
        out_domain_rejection_rate = 0.0

    # Safety-oriented balanced score.
    #
    # We prioritize:
    # - rejection of unsafe/out-of-domain queries
    # - then preserving answerable questions
    #
    safety_score = (
        0.60 * out_domain_rejection_rate
        + 0.40 * in_domain_topic_accuracy
    )

    all_threshold_results.append(
        {
            "threshold": threshold,
            "in_domain_acceptance_rate": in_domain_acceptance_rate,
            "in_domain_topic_accuracy": in_domain_topic_accuracy,
            "out_of_domain_rejection_rate": out_domain_rejection_rate,
            "safety_score": safety_score,
        }
    )

    case_results.extend(threshold_case_rows)


# ============================================================
# DISPLAY RESULTS
# ============================================================

threshold_df = pd.DataFrame(all_threshold_results)

print(
    "\n"
    f"{'Threshold':<12}"
    f"{'In-domain Accept':<20}"
    f"{'Topic Accuracy':<18}"
    f"{'OOD Rejection':<18}"
    f"{'Safety Score':<15}"
)

print("-" * 83)

for _, row in threshold_df.iterrows():

    print(
        f"{row['threshold']:<12.2f}"
        f"{row['in_domain_acceptance_rate'] * 100:<20.2f}"
        f"{row['in_domain_topic_accuracy'] * 100:<18.2f}"
        f"{row['out_of_domain_rejection_rate'] * 100:<18.2f}"
        f"{row['safety_score'] * 100:<15.2f}"
    )


# ============================================================
# SELECT THRESHOLD
# ============================================================

# Prefer thresholds that completely reject the known
# out-of-domain test case.
safe_candidates = threshold_df[
    threshold_df["out_of_domain_rejection_rate"] >= 1.0
].copy()

if not safe_candidates.empty:

    # Among safe candidates, maximize preservation
    # of correct in-domain retrieval.
    safe_candidates = safe_candidates.sort_values(
        by=[
            "in_domain_topic_accuracy",
            "in_domain_acceptance_rate",
            "threshold",
        ],
        ascending=[False, False, True],
    )

    selected_row = safe_candidates.iloc[0]

else:

    # If no threshold achieves complete rejection,
    # select the threshold with the highest safety score.
    selected_row = threshold_df.sort_values(
        by=[
            "safety_score",
            "out_of_domain_rejection_rate",
            "in_domain_topic_accuracy",
        ],
        ascending=[False, False, False],
    ).iloc[0]


selected_threshold = float(selected_row["threshold"])


# ============================================================
# CASE ANALYSIS FOR SELECTED THRESHOLD
# ============================================================

selected_cases = [
    x for x in case_results
    if float(x["threshold"]) == selected_threshold
]

selected_case_df = pd.DataFrame(selected_cases)


# ============================================================
# SAVE RESULTS
# ============================================================

threshold_csv = (
    OUTPUT_DIR
    / "rag_retrieval_safety_threshold_comparison.csv"
)

case_csv = (
    OUTPUT_DIR
    / "rag_retrieval_safety_case_analysis.csv"
)

summary_json = (
    OUTPUT_DIR
    / "rag_retrieval_safety_summary.json"
)

threshold_df.to_csv(
    threshold_csv,
    index=False
)

selected_case_df.to_csv(
    case_csv,
    index=False
)


summary = {
    "step": "38K",
    "purpose": "RAG retrieval safety and abstention evaluation",
    "test_questions": len(results),
    "thresholds_tested": THRESHOLDS,
    "selected_threshold": selected_threshold,
    "selected_metrics": {
        "in_domain_acceptance_rate": float(
            selected_row["in_domain_acceptance_rate"]
        ),
        "in_domain_topic_accuracy": float(
            selected_row["in_domain_topic_accuracy"]
        ),
        "out_of_domain_rejection_rate": float(
            selected_row["out_of_domain_rejection_rate"]
        ),
        "safety_score": float(
            selected_row["safety_score"]
        ),
    },
    "decision_rule": {
        "priority_1": "Reject out-of-domain questions",
        "priority_2": "Preserve correct in-domain retrieval",
        "priority_3": "Avoid excessive rejection"
    },
    "files": {
        "threshold_comparison": str(threshold_csv),
        "case_analysis": str(case_csv),
        "summary": str(summary_json),
    },
}

with open(
    summary_json,
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

print("\n" + "=" * 75)
print("SELECTED SAFETY THRESHOLD")
print("=" * 75)

print(
    f"\nSelected threshold: {selected_threshold:.2f}"
)

print(
    f"In-domain acceptance: "
    f"{selected_row['in_domain_acceptance_rate'] * 100:.2f}%"
)

print(
    f"In-domain topic accuracy: "
    f"{selected_row['in_domain_topic_accuracy'] * 100:.2f}%"
)

print(
    f"Out-of-domain rejection: "
    f"{selected_row['out_of_domain_rejection_rate'] * 100:.2f}%"
)

print(
    f"Safety score: "
    f"{selected_row['safety_score'] * 100:.2f}%"
)


print("\n" + "=" * 75)
print("SELECTED THRESHOLD CASES")
print("=" * 75)

for _, row in selected_case_df.iterrows():

    status = "ACCEPT" if row["accepted"] else "ABSTAIN"

    print(
        f"\nQ{int(row['question_number'])}: "
        f"{row['question']}"
    )

    print(
        f"Expected: {row['expected_topics']}"
    )

    print(
        f"Decision: {status}"
    )

    print(
        f"Top accepted topic: "
        f"{row['top_accepted_topic'] or 'NONE'}"
    )

    print(
        f"Top accepted score: "
        f"{row['top_accepted_score']}"
    )


print("\n" + "=" * 75)
print("STEP 38K - RETRIEVAL SAFETY COMPLETED")
print("=" * 75)

print("\nSaved:")
print(threshold_csv)
print(case_csv)
print(summary_json)