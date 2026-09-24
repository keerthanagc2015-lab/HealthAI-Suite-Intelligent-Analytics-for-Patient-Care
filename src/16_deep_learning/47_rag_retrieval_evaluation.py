"""
HealthAI - RAG Retrieval Evaluation

Step 38J

Compares:

1. Baseline semantic retrieval
2. Question-aware retrieval

Metrics:
    - Top-1 topic accuracy
    - Top-3 topic hit rate
    - Mean Top-1 similarity
    - Mean topic consistency
    - Out-of-domain rejection
"""

from pathlib import Path
import json
import csv
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RETRIEVAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rag"
    / "retrieval"
)

BASELINE_FILE = (
    RETRIEVAL_DIR
    / "rag_baseline_retrieval_results.json"
)

IMPROVED_FILE = (
    RETRIEVAL_DIR
    / "improved"
    / "rag_improved_retrieval_results.json"
)

OUTPUT_DIR = (
    RETRIEVAL_DIR
    / "evaluation"
)

COMPARISON_FILE = (
    OUTPUT_DIR
    / "rag_retrieval_method_comparison.csv"
)

CASE_FILE = (
    OUTPUT_DIR
    / "rag_retrieval_case_analysis.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "rag_retrieval_evaluation_summary.json"
)


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):

    if not path.exists():

        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# SAFE MEAN
# ============================================================

def safe_mean(values):

    if not values:

        return 0.0

    return float(
        np.mean(values)
    )


# ============================================================
# EVALUATE BASELINE
# ============================================================

def evaluate_baseline(
    results
):

    answerable = [
        r
        for r in results
        if r["answerable"]
    ]

    unanswerable = [
        r
        for r in results
        if not r["answerable"]
    ]


    top1_hits = []

    top3_hits = []

    similarities = []

    topic_consistency = []


    for result in answerable:

        expected_topics = result[
            "expected_topics"
        ]

        topics = result[
            "retrieved_topics"
        ]


        # -----------------------------------------------
        # Top-1
        # -----------------------------------------------

        if topics:

            top1_hit = (
                topics[0]
                in expected_topics
            )

        else:

            top1_hit = False


        # -----------------------------------------------
        # Top-3
        # -----------------------------------------------

        top3_hit = any(
            topic in expected_topics
            for topic in topics[:3]
        )


        top1_hits.append(
            top1_hit
        )

        top3_hits.append(
            top3_hit
        )


        similarities.append(
            result[
                "top_similarity"
            ]
        )


        # -----------------------------------------------
        # Topic consistency
        # -----------------------------------------------

        if topics:

            dominant_topic = max(
                set(topics),
                key=topics.count
            )

            consistency = (
                topics.count(
                    dominant_topic
                )
                / len(topics)
            )

        else:

            consistency = 0.0


        topic_consistency.append(
            consistency
        )


    # ----------------------------------------------------
    # Out-of-domain rejection
    # ----------------------------------------------------

    rejected = 0

    for result in unanswerable:

        if result[
            "top_similarity"
        ] < 0.35:

            rejected += 1


    return {

        "method":
            "baseline_semantic",

        "answerable_queries":
            len(answerable),

        "out_of_domain_queries":
            len(unanswerable),

        "top_1_topic_accuracy":
            safe_mean(
                top1_hits
            ),

        "top_3_topic_hit_rate":
            safe_mean(
                top3_hits
            ),

        "mean_top1_similarity":
            safe_mean(
                similarities
            ),

        "mean_topic_consistency":
            safe_mean(
                topic_consistency
            ),

        "out_of_domain_rejection_rate":
            (
                rejected
                / len(unanswerable)
                if unanswerable
                else 0.0
            ),
    }


# ============================================================
# EVALUATE QUESTION-AWARE
# ============================================================

def evaluate_question_aware(
    results
):

    answerable = [
        r
        for r in results
        if r["answerable"]
    ]

    unanswerable = [
        r
        for r in results
        if not r["answerable"]
    ]


    top1_hits = []

    top3_hits = []

    scores = []

    consistency = []


    for result in answerable:

        expected_topics = result[
            "expected_topics"
        ]

        topics = result[
            "final_topics"
        ]


        # -----------------------------------------------
        # Top-1
        # -----------------------------------------------

        if topics:

            top1_hit = (
                topics[0]
                in expected_topics
            )

        else:

            top1_hit = False


        # -----------------------------------------------
        # Top-3
        # -----------------------------------------------

        top3_hit = any(
            topic in expected_topics
            for topic in topics[:3]
        )


        top1_hits.append(
            top1_hit
        )

        top3_hits.append(
            top3_hit
        )


        scores.append(
            result[
                "top_final_score"
            ]
        )


        consistency.append(
            result[
                "dominant_topic_ratio"
            ]
        )


    # ----------------------------------------------------
    # Out-of-domain rejection
    # ----------------------------------------------------

    rejected = 0

    for result in unanswerable:

        if len(
            result[
                "final_evidence"
            ]
        ) == 0:

            rejected += 1


    return {

        "method":
            "question_aware",

        "answerable_queries":
            len(answerable),

        "out_of_domain_queries":
            len(unanswerable),

        "top_1_topic_accuracy":
            safe_mean(
                top1_hits
            ),

        "top_3_topic_hit_rate":
            safe_mean(
                top3_hits
            ),

        "mean_top1_similarity":
            safe_mean(
                scores
            ),

        "mean_topic_consistency":
            safe_mean(
                consistency
            ),

        "out_of_domain_rejection_rate":
            (
                rejected
                / len(unanswerable)
                if unanswerable
                else 0.0
            ),
    }


# ============================================================
# START
# ============================================================

print("=" * 75)

print(
    "HEALTHAI - RAG RETRIEVAL EVALUATION"
)

print("=" * 75)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD
# ============================================================

print(
    "\nLoading baseline retrieval..."
)

baseline_payload = load_json(
    BASELINE_FILE
)

baseline_results = (
    baseline_payload[
        "results"
    ]
)

print(
    "✓ Baseline loaded."
)


print(
    "\nLoading question-aware retrieval..."
)

improved_payload = load_json(
    IMPROVED_FILE
)

improved_results = (
    improved_payload[
        "results"
    ]
)

print(
    "✓ Question-aware retrieval loaded."
)


# ============================================================
# VALIDATE
# ============================================================

if len(
    baseline_results
) != len(
    improved_results
):

    raise ValueError(
        "Baseline and improved retrieval "
        "have different test counts."
    )


print(
    f"\nTest questions: "
    f"{len(baseline_results)}"
)


# ============================================================
# EVALUATE
# ============================================================

baseline_metrics = evaluate_baseline(
    baseline_results
)

improved_metrics = evaluate_question_aware(
    improved_results
)


# ============================================================
# PRINT
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "RETRIEVAL COMPARISON"
)

print(
    "=" * 75
)


for metrics in [
    baseline_metrics,
    improved_metrics
]:

    print(
        "\n" + metrics["method"]
    )

    print(
        f"Top-1 topic accuracy:"
        f" {metrics['top_1_topic_accuracy']:.2%}"
    )

    print(
        f"Top-3 topic hit rate:"
        f" {metrics['top_3_topic_hit_rate']:.2%}"
    )

    print(
        f"Mean Top-1 score:"
        f" {metrics['mean_top1_similarity']:.4f}"
    )

    print(
        f"Mean topic consistency:"
        f" {metrics['mean_topic_consistency']:.2%}"
    )

    print(
        f"Out-of-domain rejection:"
        f" {metrics['out_of_domain_rejection_rate']:.2%}"
    )


# ============================================================
# DELTAS
# ============================================================

top1_delta = (
    improved_metrics[
        "top_1_topic_accuracy"
    ]
    -
    baseline_metrics[
        "top_1_topic_accuracy"
    ]
)


top3_delta = (
    improved_metrics[
        "top_3_topic_hit_rate"
    ]
    -
    baseline_metrics[
        "top_3_topic_hit_rate"
    ]
)


consistency_delta = (
    improved_metrics[
        "mean_topic_consistency"
    ]
    -
    baseline_metrics[
        "mean_topic_consistency"
    ]
)


rejection_delta = (
    improved_metrics[
        "out_of_domain_rejection_rate"
    ]
    -
    baseline_metrics[
        "out_of_domain_rejection_rate"
    ]
)


print(
    "\n" + "=" * 75
)

print(
    "IMPROVEMENT DELTAS"
)

print(
    "=" * 75
)

print(
    f"\nTop-1 accuracy:"
    f" {top1_delta:+.2%}"
)

print(
    f"Top-3 hit rate:"
    f" {top3_delta:+.2%}"
)

print(
    f"Topic consistency:"
    f" {consistency_delta:+.2%}"
)

print(
    f"Out-of-domain rejection:"
    f" {rejection_delta:+.2%}"
)


# ============================================================
# CASE ANALYSIS
# ============================================================

case_rows = []


for baseline, improved in zip(
    baseline_results,
    improved_results
):

    baseline_topics = baseline[
        "retrieved_topics"
    ]

    improved_topics = improved[
        "final_topics"
    ]


    case_rows.append(
        {
            "question_number":
                improved[
                    "question_number"
                ],

            "question":
                improved[
                    "question"
                ],

            "baseline_top_3":
                " | ".join(
                    baseline_topics
                ),

            "question_aware_top_3":
                " | ".join(
                    improved_topics
                ),

            "baseline_topic_hit":
                baseline[
                    "expected_topic_found"
                ],

            "question_aware_topic_hit":
                improved[
                    "expected_topic_found"
                ],

            "baseline_top_score":
                baseline[
                    "top_similarity"
                ],

            "question_aware_top_score":
                improved[
                    "top_final_score"
                ],

            "question_aware_topic_consistency":
                improved[
                    "dominant_topic_ratio"
                ],
        }
    )


# ============================================================
# SAVE COMPARISON CSV
# ============================================================

comparison_rows = [
    baseline_metrics,
    improved_metrics,
]


with open(
    COMPARISON_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "method",
            "answerable_queries",
            "out_of_domain_queries",
            "top_1_topic_accuracy",
            "top_3_topic_hit_rate",
            "mean_top1_similarity",
            "mean_topic_consistency",
            "out_of_domain_rejection_rate",
        ]
    )

    writer.writeheader()

    writer.writerows(
        comparison_rows
    )


# ============================================================
# SAVE CASE CSV
# ============================================================

with open(
    CASE_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "question_number",
            "question",
            "baseline_top_3",
            "question_aware_top_3",
            "baseline_topic_hit",
            "question_aware_topic_hit",
            "baseline_top_score",
            "question_aware_top_score",
            "question_aware_topic_consistency",
        ]
    )

    writer.writeheader()

    writer.writerows(
        case_rows
    )


# ============================================================
# METHOD SELECTION
# ============================================================

# Primary:
# Top-3 topic hit rate
#
# Secondary:
# Topic consistency
#
# Safety:
# Out-of-domain rejection

if (
    improved_metrics[
        "top_3_topic_hit_rate"
    ]
    >
    baseline_metrics[
        "top_3_topic_hit_rate"
    ]
):

    selected_method = (
        "question_aware"
    )

elif (
    improved_metrics[
        "mean_topic_consistency"
    ]
    >
    baseline_metrics[
        "mean_topic_consistency"
    ]
):

    selected_method = (
        "question_aware"
    )

else:

    selected_method = (
        "baseline_semantic"
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {

    "baseline":
        baseline_metrics,

    "question_aware":
        improved_metrics,

    "deltas":
        {
            "top1_accuracy":
                top1_delta,

            "top3_hit_rate":
                top3_delta,

            "topic_consistency":
                consistency_delta,

            "out_of_domain_rejection":
                rejection_delta,
        },

    "selected_method":
        selected_method,

    "evaluation_notes":
        [
            "Retrieval evaluation is separate from answer generation.",
            "The evaluation set contains six answerable questions and one out-of-domain question.",
            "Topic hit rate measures whether the expected healthcare topic was retrieved.",
            "Topic consistency measures how concentrated the final evidence is around one topic.",
            "Out-of-domain rejection measures whether unsupported questions are excluded.",
            "The test set is small and therefore results are illustrative rather than statistically conclusive."
        ],
}


with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# FINAL
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "STEP 38J - RETRIEVAL EVALUATION COMPLETED"
)

print(
    "=" * 75
)

print(
    f"\nSelected method:"
    f" {selected_method}"
)

print(
    "\nSaved:"
)

print(
    COMPARISON_FILE
)

print(
    CASE_FILE
)

print(
    SUMMARY_FILE
)

print(
    "\n✓ Baseline vs question-aware retrieval evaluated."
)