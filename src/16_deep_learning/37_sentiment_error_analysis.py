"""
HEALTHAI - SENTIMENT ERROR ANALYSIS

Step 37

Analyzes sentiment model predictions and identifies:
- Correct predictions
- Incorrect predictions
- False positives
- False negatives
- Confidence of predictions
- Text-level error patterns

Models:
1. TF-IDF + Logistic Regression
2. DistilBERT

Important:
The dataset contains only 20 unique feedback texts.
Therefore, this analysis is illustrative.
"""

import json
import os
import pickle

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "data/processed/deep_learning/medical_sentiment"

TEST_PATH = os.path.join(
    BASE_DIR,
    "splits",
    "test.csv"
)

TFIDF_MODEL_DIR = (
    "models/nlp/sentiment/tfidf_logistic_regression"
)

DISTILBERT_PREDICTIONS = os.path.join(
    BASE_DIR,
    "transformer",
    "distilbert_test_predictions.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "error_analysis"
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
print("HEALTHAI - SENTIMENT ERROR ANALYSIS")
print("=" * 75)


# ============================================================
# LOAD TEST DATA
# ============================================================

test_df = pd.read_csv(TEST_PATH)

print()
print("=" * 75)
print("TEST DATA")
print("=" * 75)

print()
print(f"Test samples: {len(test_df)}")

print(
    f"Unique feedback texts: "
    f"{test_df['Feedback'].nunique()}"
)


# ============================================================
# LOAD TF-IDF MODEL
# ============================================================

print()
print("=" * 75)
print("LOADING TF-IDF MODEL")
print("=" * 75)

vectorizer_path = os.path.join(
    TFIDF_MODEL_DIR,
    "tfidf_vectorizer.pkl"
)

model_path = os.path.join(
    TFIDF_MODEL_DIR,
    "logistic_regression_model.pkl"
)


with open(
    vectorizer_path,
    "rb"
) as f:

    vectorizer = pickle.load(f)


with open(
    model_path,
    "rb"
) as f:

    tfidf_model = pickle.load(f)


print("✓ TF-IDF vectorizer loaded.")
print("✓ Logistic Regression model loaded.")


# ============================================================
# TF-IDF PREDICTIONS
# ============================================================

X_test = vectorizer.transform(
    test_df["Feedback"].astype(str)
)

tfidf_predictions = tfidf_model.predict(
    X_test
)

tfidf_probabilities = tfidf_model.predict_proba(
    X_test
)

tfidf_confidence = np.max(
    tfidf_probabilities,
    axis=1
)


# ============================================================
# CREATE TF-IDF ERROR TABLE
# ============================================================

tfidf_errors = test_df.copy()

tfidf_errors["Actual_Label"] = (
    tfidf_errors["Sentiment"]
    .map({
        0: "negative",
        1: "positive"
    })
)

tfidf_errors["Predicted_Sentiment"] = (
    tfidf_predictions
)

tfidf_errors["Predicted_Label"] = (
    tfidf_errors["Predicted_Sentiment"]
    .map({
        0: "negative",
        1: "positive"
    })
)

tfidf_errors["Confidence"] = (
    tfidf_confidence
)

tfidf_errors["Correct"] = (
    tfidf_errors["Sentiment"]
    ==
    tfidf_errors["Predicted_Sentiment"]
)


# ============================================================
# ERROR TYPE
# ============================================================

def determine_error_type(row):

    actual = row["Sentiment"]
    predicted = row["Predicted_Sentiment"]

    if actual == predicted:
        return "Correct"

    if actual == 0 and predicted == 1:
        return "False Positive"

    if actual == 1 and predicted == 0:
        return "False Negative"

    return "Unknown"


tfidf_errors["Error_Type"] = (
    tfidf_errors.apply(
        determine_error_type,
        axis=1
    )
)


# ============================================================
# PRINT TF-IDF ERRORS
# ============================================================

print()
print("=" * 75)
print("TF-IDF ERROR ANALYSIS")
print("=" * 75)

print()

tfidf_error_rows = tfidf_errors[
    tfidf_errors["Correct"] == False
]

if len(tfidf_error_rows) == 0:

    print("No TF-IDF errors found.")

else:

    for index, row in tfidf_error_rows.iterrows():

        print(f"Sample {index + 1}")

        print(
            f"Feedback: {row['Feedback']}"
        )

        print(
            f"Actual: "
            f"{row['Actual_Label']}"
        )

        print(
            f"Predicted: "
            f"{row['Predicted_Label']}"
        )

        print(
            f"Confidence: "
            f"{row['Confidence']:.4f}"
        )

        print(
            f"Error Type: "
            f"{row['Error_Type']}"
        )

        print("-" * 75)


# ============================================================
# TF-IDF CONFUSION MATRIX
# ============================================================

tfidf_cm = confusion_matrix(
    test_df["Sentiment"],
    tfidf_predictions,
    labels=[0, 1]
)

print()
print("=" * 75)
print("TF-IDF CONFUSION MATRIX")
print("=" * 75)

print()
print(tfidf_cm)

print()
print("Rows    = Actual")
print("Columns = Predicted")


# ============================================================
# LOAD DISTILBERT PREDICTIONS
# ============================================================

print()
print("=" * 75)
print("LOADING DISTILBERT PREDICTIONS")
print("=" * 75)


if os.path.exists(DISTILBERT_PREDICTIONS):

    distilbert_df = pd.read_csv(
        DISTILBERT_PREDICTIONS
    )

    print("✓ DistilBERT predictions loaded.")

else:

    print(
        "⚠ DistilBERT prediction file not found."
    )

    distilbert_df = None


# ============================================================
# DISTILBERT ERROR ANALYSIS
# ============================================================

if distilbert_df is not None:

    # Make sure actual labels are available
    if "Sentiment" in distilbert_df.columns:

        distilbert_actual = (
            distilbert_df["Sentiment"]
            .astype(int)
            .values
        )

    else:

        distilbert_actual = (
            test_df["Sentiment"]
            .astype(int)
            .values
        )


    # Find prediction column
    if "Predicted_Sentiment" in distilbert_df.columns:

        distilbert_predictions = (
            distilbert_df[
                "Predicted_Sentiment"
            ]
            .astype(int)
            .values
        )

    else:

        raise KeyError(
            "Predicted_Sentiment column not found "
            "in DistilBERT prediction file."
        )


    # Create analysis dataframe
    distilbert_errors = test_df.copy()

    distilbert_errors["Actual_Label"] = (
        distilbert_actual
    )

    distilbert_errors["Predicted_Sentiment"] = (
        distilbert_predictions
    )

    distilbert_errors["Predicted_Label"] = (
        distilbert_errors[
            "Predicted_Sentiment"
        ].map({
            0: "negative",
            1: "positive"
        })
    )

    distilbert_errors["Correct"] = (
        distilbert_actual
        ==
        distilbert_predictions
    )

    distilbert_errors["Error_Type"] = (
        distilbert_errors.apply(
            determine_error_type,
            axis=1
        )
    )


    print()
    print("=" * 75)
    print("DISTILBERT ERROR ANALYSIS")
    print("=" * 75)

    print()

    distilbert_error_rows = (
        distilbert_errors[
            distilbert_errors["Correct"] == False
        ]
    )

    if len(distilbert_error_rows) == 0:

        print(
            "No DistilBERT errors found."
        )

    else:

        for index, row in (
            distilbert_error_rows.iterrows()
        ):

            print(
                f"Sample {index + 1}"
            )

            print(
                f"Feedback: "
                f"{row['Feedback']}"
            )

            print(
                f"Actual: "
                f"{row['Actual_Label']}"
            )

            print(
                f"Predicted: "
                f"{row['Predicted_Label']}"
            )

            print(
                f"Error Type: "
                f"{row['Error_Type']}"
            )

            print("-" * 75)


    # ========================================================
    # DISTILBERT CONFUSION MATRIX
    # ========================================================

    distilbert_cm = confusion_matrix(
        distilbert_actual,
        distilbert_predictions,
        labels=[0, 1]
    )

    print()
    print("=" * 75)
    print("DISTILBERT CONFUSION MATRIX")
    print("=" * 75)

    print()
    print(distilbert_cm)

    print()
    print("Rows    = Actual")
    print("Columns = Predicted")


else:

    distilbert_errors = None
    distilbert_cm = None


# ============================================================
# COMBINED ERROR ANALYSIS
# ============================================================

print()
print("=" * 75)
print("COMBINED MODEL ERROR ANALYSIS")
print("=" * 75)


combined = test_df.copy()

combined["TFIDF_Prediction"] = (
    tfidf_predictions
)

combined["TFIDF_Label"] = (
    combined["TFIDF_Prediction"]
    .map({
        0: "negative",
        1: "positive"
    })
)

combined["TFIDF_Correct"] = (
    combined["Sentiment"]
    ==
    combined["TFIDF_Prediction"]
)


if distilbert_df is not None:

    combined[
        "DistilBERT_Prediction"
    ] = distilbert_predictions

    combined[
        "DistilBERT_Label"
    ] = (
        combined[
            "DistilBERT_Prediction"
        ].map({
            0: "negative",
            1: "positive"
        })
    )

    combined[
        "DistilBERT_Correct"
    ] = (
        combined["Sentiment"]
        ==
        combined["DistilBERT_Prediction"]
    )


    combined["Both_Correct"] = (
        combined["TFIDF_Correct"]
        &
        combined["DistilBERT_Correct"]
    )

    combined["Both_Wrong"] = (
        ~combined["TFIDF_Correct"]
        &
        ~combined["DistilBERT_Correct"]
    )

    combined["Models_Disagree"] = (
        combined["TFIDF_Prediction"]
        !=
        combined["DistilBERT_Prediction"]
    )


# ============================================================
# PRINT COMBINED RESULTS
# ============================================================

if distilbert_df is not None:

    print()

    print(
        f"Both models correct: "
        f"{combined['Both_Correct'].sum()}"
    )

    print(
        f"Both models wrong: "
        f"{combined['Both_Wrong'].sum()}"
    )

    print(
        f"Models disagree: "
        f"{combined['Models_Disagree'].sum()}"
    )


# ============================================================
# SAVE TF-IDF ERROR ANALYSIS
# ============================================================

tfidf_error_path = os.path.join(
    OUTPUT_DIR,
    "tfidf_error_analysis.csv"
)

tfidf_errors.to_csv(
    tfidf_error_path,
    index=False
)

print()
print("✓ TF-IDF error analysis saved:")
print(os.path.abspath(tfidf_error_path))


# ============================================================
# SAVE DISTILBERT ERROR ANALYSIS
# ============================================================

if distilbert_errors is not None:

    distilbert_error_path = os.path.join(
        OUTPUT_DIR,
        "distilbert_error_analysis.csv"
    )

    distilbert_errors.to_csv(
        distilbert_error_path,
        index=False
    )

    print()
    print(
        "✓ DistilBERT error analysis saved:"
    )

    print(
        os.path.abspath(
            distilbert_error_path
        )
    )


# ============================================================
# SAVE COMBINED ANALYSIS
# ============================================================

combined_path = os.path.join(
    OUTPUT_DIR,
    "combined_sentiment_error_analysis.csv"
)

combined.to_csv(
    combined_path,
    index=False
)

print()
print("✓ Combined error analysis saved:")
print(os.path.abspath(combined_path))


# ============================================================
# ERROR SUMMARY
# ============================================================

tfidf_total_errors = int(
    (~tfidf_errors["Correct"]).sum()
)

tfidf_false_positives = int(
    (
        tfidf_errors["Error_Type"]
        ==
        "False Positive"
    ).sum()
)

tfidf_false_negatives = int(
    (
        tfidf_errors["Error_Type"]
        ==
        "False Negative"
    ).sum()
)


summary = {
    "dataset": {
        "test_samples": int(len(test_df)),
        "unique_feedback_texts": int(
            test_df["Feedback"].nunique()
        )
    },

    "tfidf": {
        "total_errors": tfidf_total_errors,
        "false_positives": tfidf_false_positives,
        "false_negatives": tfidf_false_negatives,
        "confusion_matrix": tfidf_cm.tolist()
    }
}


if distilbert_errors is not None:

    distilbert_total_errors = int(
        (~distilbert_errors["Correct"]).sum()
    )

    distilbert_false_positives = int(
        (
            distilbert_errors["Error_Type"]
            ==
            "False Positive"
        ).sum()
    )

    distilbert_false_negatives = int(
        (
            distilbert_errors["Error_Type"]
            ==
            "False Negative"
        ).sum()
    )

    summary["distilbert"] = {

        "total_errors": (
            distilbert_total_errors
        ),

        "false_positives": (
            distilbert_false_positives
        ),

        "false_negatives": (
            distilbert_false_negatives
        ),

        "confusion_matrix": (
            distilbert_cm.tolist()
        )
    }


summary["important_limitation"] = (
    "Only 20 unique feedback texts exist in the "
    "full dataset. Therefore, error analysis is "
    "illustrative and cannot establish robust "
    "generalization performance."
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "sentiment_error_analysis_summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


print()
print("✓ Error analysis summary saved:")
print(os.path.abspath(summary_path))


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 75)
print("STEP 37 - SENTIMENT ERROR ANALYSIS COMPLETED")
print("=" * 75)

print()
print(
    f"TF-IDF total errors: "
    f"{tfidf_total_errors}"
)

print(
    f"TF-IDF false positives: "
    f"{tfidf_false_positives}"
)

print(
    f"TF-IDF false negatives: "
    f"{tfidf_false_negatives}"
)

if distilbert_errors is not None:

    print()

    print(
        f"DistilBERT total errors: "
        f"{distilbert_total_errors}"
    )

    print(
        f"DistilBERT false positives: "
        f"{distilbert_false_positives}"
    )

    print(
        f"DistilBERT false negatives: "
        f"{distilbert_false_negatives}"
    )

print()
print(
    "Important: test set contains only "
    "4 unique feedback texts."
)

print()
print("Next:")
print("38 - RAG Healthcare Knowledge Ingestion")

print("=" * 75)