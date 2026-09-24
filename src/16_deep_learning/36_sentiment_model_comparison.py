"""
HEALTHAI - SENTIMENT MODEL COMPARISON

Step 31F

Compares:
1. TF-IDF + Logistic Regression
2. DistilBERT Transformer

Purpose:
- Compare classical NLP and transformer-based sentiment models.
- Produce a reproducible comparison table.
- Save comparison metrics and visualization.
- Document dataset limitations honestly.

Important:
The dataset contains only 20 unique feedback texts.
Therefore, test metrics are illustrative and should NOT be treated
as statistically reliable estimates of production performance.
"""

import json
import os

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "data/processed/deep_learning/medical_sentiment"

TFIDF_METRICS_PATH = os.path.join(
    BASE_DIR,
    "baseline",
    "tfidf_logistic_metrics.json"
)

DISTILBERT_METRICS_PATH = os.path.join(
    BASE_DIR,
    "transformer",
    "distilbert_sentiment_metrics.json"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "comparison"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 75)
print("HEALTHAI - SENTIMENT MODEL COMPARISON")
print("=" * 75)


# ============================================================
# CHECK FILES
# ============================================================

print()
print("Checking metric files...")

if not os.path.exists(TFIDF_METRICS_PATH):

    raise FileNotFoundError(
        f"""
TF-IDF metrics file not found:

{os.path.abspath(TFIDF_METRICS_PATH)}

Please make sure Step 31D was completed successfully.
"""
    )


if not os.path.exists(DISTILBERT_METRICS_PATH):

    raise FileNotFoundError(
        f"""
DistilBERT metrics file not found:

{os.path.abspath(DISTILBERT_METRICS_PATH)}

Please make sure Step 31E was completed successfully.
"""
    )


print("✓ TF-IDF metrics file found.")
print("✓ DistilBERT metrics file found.")


# ============================================================
# LOAD METRICS
# ============================================================

with open(
    TFIDF_METRICS_PATH,
    "r",
    encoding="utf-8"
) as f:

    tfidf_metrics = json.load(f)


with open(
    DISTILBERT_METRICS_PATH,
    "r",
    encoding="utf-8"
) as f:

    distilbert_metrics = json.load(f)


print()
print("✓ Metric files loaded successfully.")


# ============================================================
# DISPLAY AVAILABLE KEYS
# ============================================================

print()
print("=" * 75)
print("AVAILABLE METRIC FIELDS")
print("=" * 75)

print()
print("TF-IDF JSON keys:")
print(list(tfidf_metrics.keys()))

print()
print("DistilBERT JSON keys:")
print(list(distilbert_metrics.keys()))


# ============================================================
# ROBUST METRIC EXTRACTION
# ============================================================

def extract_test_metrics(data, model_name):
    """
    Extract test metrics from either of these structures:

    Structure A:
        {
            "test_accuracy": ...,
            "test_precision": ...,
            "test_recall": ...,
            "test_f1": ...
        }

    Structure B:
        {
            "test_metrics": {
                "accuracy": ...,
                "precision": ...,
                "recall": ...,
                "f1": ...
            }
        }
    """

    # --------------------------------------------------------
    # FORMAT 1: test_metrics nested dictionary
    # --------------------------------------------------------

    if "test_metrics" in data:

        test_metrics = data["test_metrics"]

        if isinstance(
            test_metrics,
            dict
        ):

            required_metrics = [
                "accuracy",
                "precision",
                "recall",
                "f1"
            ]

            missing = [
                metric
                for metric in required_metrics
                if metric not in test_metrics
            ]

            if not missing:

                return {
                    "accuracy": float(
                        test_metrics["accuracy"]
                    ),

                    "precision": float(
                        test_metrics["precision"]
                    ),

                    "recall": float(
                        test_metrics["recall"]
                    ),

                    "f1": float(
                        test_metrics["f1"]
                    )
                }


    # --------------------------------------------------------
    # FORMAT 2: test_accuracy style
    # --------------------------------------------------------

    direct_keys = [
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1"
    ]

    if all(
        key in data
        for key in direct_keys
    ):

        return {
            "accuracy": float(
                data["test_accuracy"]
            ),

            "precision": float(
                data["test_precision"]
            ),

            "recall": float(
                data["test_recall"]
            ),

            "f1": float(
                data["test_f1"]
            )
        }


    # --------------------------------------------------------
    # FORMAT 3: direct metric names
    # --------------------------------------------------------

    direct_metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]

    if all(
        metric in data
        for metric in direct_metrics
    ):

        return {
            "accuracy": float(
                data["accuracy"]
            ),

            "precision": float(
                data["precision"]
            ),

            "recall": float(
                data["recall"]
            ),

            "f1": float(
                data["f1"]
            )
        }


    # --------------------------------------------------------
    # ERROR
    # --------------------------------------------------------

    raise KeyError(
        f"""
Could not extract test metrics for {model_name}.

Available top-level keys:
{list(data.keys())}

Expected either:
1. test_metrics -> accuracy / precision / recall / f1
2. test_accuracy / test_precision / test_recall / test_f1
3. accuracy / precision / recall / f1
"""
    )


# ============================================================
# EXTRACT METRICS
# ============================================================

tfidf = extract_test_metrics(
    tfidf_metrics,
    "TF-IDF + Logistic Regression"
)

distilbert = extract_test_metrics(
    distilbert_metrics,
    "DistilBERT"
)


# ============================================================
# PRINT EXTRACTED METRICS
# ============================================================

print()
print("=" * 75)
print("EXTRACTED TEST METRICS")
print("=" * 75)

print()
print("TF-IDF + Logistic Regression")

print(
    f"Accuracy    : {tfidf['accuracy']:.4f}"
)

print(
    f"Precision   : {tfidf['precision']:.4f}"
)

print(
    f"Recall      : {tfidf['recall']:.4f}"
)

print(
    f"F1          : {tfidf['f1']:.4f}"
)


print()
print("DistilBERT")

print(
    f"Accuracy    : {distilbert['accuracy']:.4f}"
)

print(
    f"Precision   : {distilbert['precision']:.4f}"
)

print(
    f"Recall      : {distilbert['recall']:.4f}"
)

print(
    f"F1          : {distilbert['f1']:.4f}"
)


# ============================================================
# CREATE COMPARISON DATAFRAME
# ============================================================

comparison = pd.DataFrame(
    [
        {
            "Model": "TF-IDF + Logistic Regression",
            "Accuracy": tfidf["accuracy"],
            "Precision": tfidf["precision"],
            "Recall": tfidf["recall"],
            "F1": tfidf["f1"]
        },

        {
            "Model": "DistilBERT",
            "Accuracy": distilbert["accuracy"],
            "Precision": distilbert["precision"],
            "Recall": distilbert["recall"],
            "F1": distilbert["f1"]
        }
    ]
)


# ============================================================
# PRINT COMPARISON TABLE
# ============================================================

print()
print("=" * 75)
print("TEST SET MODEL COMPARISON")
print("=" * 75)

print()

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# CALCULATE DIFFERENCES
# ============================================================

accuracy_difference = (
    distilbert["accuracy"]
    -
    tfidf["accuracy"]
)

precision_difference = (
    distilbert["precision"]
    -
    tfidf["precision"]
)

recall_difference = (
    distilbert["recall"]
    -
    tfidf["recall"]
)

f1_difference = (
    distilbert["f1"]
    -
    tfidf["f1"]
)


print()
print("=" * 75)
print("DISTILBERT - TF-IDF DIFFERENCE")
print("=" * 75)

print()

print(
    f"Accuracy difference    : "
    f"{accuracy_difference:+.4f}"
)

print(
    f"Precision difference   : "
    f"{precision_difference:+.4f}"
)

print(
    f"Recall difference      : "
    f"{recall_difference:+.4f}"
)

print(
    f"F1 difference          : "
    f"{f1_difference:+.4f}"
)


# ============================================================
# MODEL SELECTION
# ============================================================

print()
print("=" * 75)
print("MODEL SELECTION")
print("=" * 75)

if distilbert["f1"] > tfidf["f1"]:

    selected_model = "DistilBERT"

    selection_reason = (
        "DistilBERT achieved a higher test F1 score."
    )

elif tfidf["f1"] > distilbert["f1"]:

    selected_model = (
        "TF-IDF + Logistic Regression"
    )

    selection_reason = (
        "TF-IDF + Logistic Regression achieved "
        "a higher test F1 score."
    )

else:

    selected_model = "Tie"

    selection_reason = (
        "Both models achieved the same test F1 score. "
        "Therefore, there is no performance-based winner "
        "on this test set."
    )


print()
print(
    f"Performance-based selection: "
    f"{selected_model}"
)

print()
print(
    f"Reason: {selection_reason}"
)


# ============================================================
# DATASET INFORMATION
# ============================================================

print()
print("=" * 75)
print("DATASET INFORMATION")
print("=" * 75)

print()
print("Full dataset unique feedback texts : 20")
print("Training unique feedback texts     : 12")
print("Validation unique feedback texts   : 4")
print("Test unique feedback texts         : 4")


# ============================================================
# DATASET LIMITATION
# ============================================================

print()
print("=" * 75)
print("IMPORTANT DATASET LIMITATION")
print("=" * 75)

print()

print(
    "The dataset contains only 20 unique feedback texts."
)

print()

print(
    "The test set contains only 4 unique feedback texts."
)

print()

print(
    "Therefore, these test metrics are illustrative "
    "and should not be interpreted as statistically "
    "reliable production performance."
)


# ============================================================
# SAVE COMPARISON CSV
# ============================================================

comparison_csv_path = os.path.join(
    OUTPUT_DIR,
    "sentiment_model_comparison.csv"
)

comparison.to_csv(
    comparison_csv_path,
    index=False
)

print()
print("✓ Comparison CSV saved:")
print(
    os.path.abspath(
        comparison_csv_path
    )
)


# ============================================================
# CREATE COMPARISON CHART
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1"
]

tfidf_values = [
    tfidf["accuracy"],
    tfidf["precision"],
    tfidf["recall"],
    tfidf["f1"]
]

distilbert_values = [
    distilbert["accuracy"],
    distilbert["precision"],
    distilbert["recall"],
    distilbert["f1"]
]

x = range(
    len(metrics)
)


plt.figure(
    figsize=(10, 6)
)

plt.plot(
    x,
    tfidf_values,
    marker="o",
    label="TF-IDF + Logistic Regression"
)

plt.plot(
    x,
    distilbert_values,
    marker="o",
    label="DistilBERT"
)

plt.xticks(
    list(x),
    metrics
)

plt.ylim(
    0,
    1.05
)

plt.xlabel(
    "Evaluation Metric"
)

plt.ylabel(
    "Score"
)

plt.title(
    "Healthcare Sentiment Model Comparison"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


chart_path = os.path.join(
    OUTPUT_DIR,
    "sentiment_model_comparison.png"
)

plt.savefig(
    chart_path,
    dpi=200
)

plt.close()


print()
print("✓ Comparison chart saved:")
print(
    os.path.abspath(
        chart_path
    )
)


# ============================================================
# SAVE MODEL SELECTION REPORT
# ============================================================

selection_report = {

    "task": (
        "Healthcare patient feedback sentiment classification"
    ),

    "models_compared": [
        "TF-IDF + Logistic Regression",
        "DistilBERT"
    ],

    "results": {

        "TF-IDF + Logistic Regression": {
            "accuracy": tfidf["accuracy"],
            "precision": tfidf["precision"],
            "recall": tfidf["recall"],
            "f1": tfidf["f1"]
        },

        "DistilBERT": {
            "accuracy": distilbert["accuracy"],
            "precision": distilbert["precision"],
            "recall": distilbert["recall"],
            "f1": distilbert["f1"]
        }
    },

    "distilbert_minus_tfidf": {
        "accuracy": accuracy_difference,
        "precision": precision_difference,
        "recall": recall_difference,
        "f1": f1_difference
    },

    "selection": {
        "model": selected_model,
        "reason": selection_reason
    },

    "dataset": {
        "unique_feedback_texts": 20,
        "train_unique_texts": 12,
        "validation_unique_texts": 4,
        "test_unique_texts": 4
    },

    "evaluation_warning": (
        "The dataset contains only 20 unique feedback texts. "
        "Therefore, the test metrics are illustrative and "
        "should not be interpreted as statistically reliable "
        "production performance."
    ),

    "engineering_recommendation": (
        "For production deployment, obtain a substantially "
        "larger and more diverse labeled patient-feedback "
        "dataset before selecting a final sentiment model "
        "based solely on benchmark performance."
    )
}


selection_report_path = os.path.join(
    OUTPUT_DIR,
    "sentiment_model_selection.json"
)

with open(
    selection_report_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        selection_report,
        f,
        indent=4
    )


print()
print("✓ Model selection report saved:")
print(
    os.path.abspath(
        selection_report_path
    )
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 75)
print("STEP 31F - COMPLETED")
print("=" * 75)

print()

print(
    "TF-IDF + Logistic Regression"
)

print(
    f"  Accuracy  : "
    f"{tfidf['accuracy']:.4f}"
)

print(
    f"  Precision : "
    f"{tfidf['precision']:.4f}"
)

print(
    f"  Recall    : "
    f"{tfidf['recall']:.4f}"
)

print(
    f"  F1        : "
    f"{tfidf['f1']:.4f}"
)

print()

print(
    "DistilBERT"
)

print(
    f"  Accuracy  : "
    f"{distilbert['accuracy']:.4f}"
)

print(
    f"  Precision : "
    f"{distilbert['precision']:.4f}"
)

print(
    f"  Recall    : "
    f"{distilbert['recall']:.4f}"
)

print(
    f"  F1        : "
    f"{distilbert['f1']:.4f}"
)

print()

print(
    f"Performance-based selection: "
    f"{selected_model}"
)

print()

print(
    "Dataset limitation: "
    "only 20 unique feedback texts."
)

print()

print("Saved files:")

print(
    f"  {os.path.abspath(comparison_csv_path)}"
)

print(
    f"  {os.path.abspath(chart_path)}"
)

print(
    f"  {os.path.abspath(selection_report_path)}"
)

print()
print("Next:")
print("37 - Sentiment Error Analysis")

print("=" * 75)