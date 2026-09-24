"""
HealthAI - TF-IDF + Logistic Regression Sentiment Baseline

Step 31D

Pipeline:
    Patient Feedback
        ↓
    TF-IDF
        ↓
    Logistic Regression
        ↓
    Binary Sentiment

Important:
    The dataset contains only 20 unique feedback texts.
    Train/validation/test splits were created at the
    unique-feedback level to prevent text leakage.
"""

from pathlib import Path
import json
import pickle

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "deep_learning"
    / "medical_sentiment"
    / "splits"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "nlp"
    / "sentiment"
    / "tfidf_logistic_regression"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "deep_learning"
    / "medical_sentiment"
    / "baseline"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. HEADER
# ============================================================

print("=" * 75)
print("HEALTHAI - TF-IDF + LOGISTIC REGRESSION SENTIMENT BASELINE")
print("=" * 75)


# ============================================================
# 3. LOAD SPLITS
# ============================================================

train_path = SPLIT_DIR / "train.csv"
validation_path = SPLIT_DIR / "validation.csv"
test_path = SPLIT_DIR / "test.csv"

train_df = pd.read_csv(train_path)
validation_df = pd.read_csv(validation_path)
test_df = pd.read_csv(test_path)

print()
print("✓ Train / validation / test datasets loaded.")

print(
    f"Train:      {len(train_df)}"
)

print(
    f"Validation: {len(validation_df)}"
)

print(
    f"Test:       {len(test_df)}"
)


# ============================================================
# 4. PREPARE TEXT AND LABELS
# ============================================================

X_train = (
    train_df["Feedback"]
    .fillna("")
    .astype(str)
)

y_train = (
    train_df["Sentiment"]
    .astype(int)
)

X_validation = (
    validation_df["Feedback"]
    .fillna("")
    .astype(str)
)

y_validation = (
    validation_df["Sentiment"]
    .astype(int)
)

X_test = (
    test_df["Feedback"]
    .fillna("")
    .astype(str)
)

y_test = (
    test_df["Sentiment"]
    .astype(int)
)


# ============================================================
# 5. TF-IDF
# ============================================================

print()
print("=" * 75)
print("TF-IDF FEATURE EXTRACTION")
print("=" * 75)

vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=1,
    max_df=1.0,
    sublinear_tf=True,
)


# IMPORTANT:
# Fit TF-IDF ONLY on training data.

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_validation_tfidf = vectorizer.transform(
    X_validation
)

X_test_tfidf = vectorizer.transform(
    X_test
)


print(
    f"Training matrix shape: "
    f"{X_train_tfidf.shape}"
)

print(
    f"Validation matrix shape: "
    f"{X_validation_tfidf.shape}"
)

print(
    f"Test matrix shape: "
    f"{X_test_tfidf.shape}"
)

print(
    f"Vocabulary size: "
    f"{len(vectorizer.vocabulary_)}"
)


# ============================================================
# 6. LOGISTIC REGRESSION
# ============================================================

print()
print("=" * 75)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 75)

model = LogisticRegression(
    max_iter=2000,
    random_state=42,
    class_weight="balanced",
)

model.fit(
    X_train_tfidf,
    y_train
)

print()
print(
    "✓ Logistic Regression trained."
)


# ============================================================
# 7. VALIDATION PREDICTIONS
# ============================================================

validation_predictions = (
    model.predict(
        X_validation_tfidf
    )
)

validation_probabilities = (
    model.predict_proba(
        X_validation_tfidf
    )[:, 1]
)


# ============================================================
# 8. TEST PREDICTIONS
# ============================================================

test_predictions = (
    model.predict(
        X_test_tfidf
    )
)

test_probabilities = (
    model.predict_proba(
        X_test_tfidf
    )[:, 1]
)


# ============================================================
# 9. METRICS FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),
    }


validation_metrics = calculate_metrics(
    y_validation,
    validation_predictions
)

test_metrics = calculate_metrics(
    y_test,
    test_predictions
)


# ============================================================
# 10. PRINT VALIDATION METRICS
# ============================================================

print()
print("=" * 75)
print("VALIDATION RESULTS")
print("=" * 75)

for metric, value in validation_metrics.items():

    print(
        f"{metric.capitalize():<12}: "
        f"{value:.4f}"
    )


# ============================================================
# 11. PRINT TEST METRICS
# ============================================================

print()
print("=" * 75)
print("TEST RESULTS")
print("=" * 75)

for metric, value in test_metrics.items():

    print(
        f"{metric.capitalize():<12}: "
        f"{value:.4f}"
    )


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 75)
print("TEST CLASSIFICATION REPORT")
print("=" * 75)

class_names = [
    "negative",
    "positive",
]

classification_report_text = (
    classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=class_names,
        zero_division=0,
    )
)

print(
    classification_report_text
)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
)

print()
print("=" * 75)
print("CONFUSION MATRIX")
print("=" * 75)

print(
    "Rows = Actual"
)

print(
    "Columns = Predicted"
)

print()
print(cm)


# ============================================================
# 14. SAVE CONFUSION MATRIX IMAGE
# ============================================================

plt.figure(
    figsize=(6, 5)
)

plt.imshow(cm)

plt.title(
    "TF-IDF + Logistic Regression\nConfusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "Actual Label"
)

plt.xticks(
    [0, 1],
    ["Negative", "Positive"]
)

plt.yticks(
    [0, 1],
    ["Negative", "Positive"]
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.tight_layout()

confusion_matrix_path = (
    REPORT_DIR
    / "tfidf_logistic_confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_path,
    dpi=200
)

plt.close()

print()
print(
    f"✓ Confusion matrix saved:"
)

print(
    confusion_matrix_path
)


# ============================================================
# 15. PREDICTION TABLE
# ============================================================

prediction_df = test_df[
    [
        "Theme",
        "Feedback",
        "Sentiment",
        "sentiment_label",
    ]
].copy()

prediction_df["predicted_sentiment"] = (
    test_predictions
)

prediction_df["predicted_label"] = (
    prediction_df[
        "predicted_sentiment"
    ]
    .map(
        {
            0: "negative",
            1: "positive",
        }
    )
)

prediction_df["positive_probability"] = (
    test_probabilities
)

prediction_df["correct"] = (
    prediction_df["Sentiment"]
    == prediction_df["predicted_sentiment"]
)

prediction_path = (
    REPORT_DIR
    / "tfidf_test_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)

print()
print(
    f"✓ Test predictions saved:"
)

print(
    prediction_path
)


# ============================================================
# 16. SAVE METRICS
# ============================================================

metrics = {
    "model": (
        "TF-IDF + Logistic Regression"
    ),
    "vectorizer": {
        "type": "TfidfVectorizer",
        "ngram_range": [1, 2],
        "lowercase": True,
        "strip_accents": "unicode",
        "sublinear_tf": True,
    },
    "classifier": {
        "type": "LogisticRegression",
        "max_iter": 2000,
        "class_weight": "balanced",
        "random_state": 42,
    },
    "dataset": {
        "train_records": int(
            len(train_df)
        ),
        "validation_records": int(
            len(validation_df)
        ),
        "test_records": int(
            len(test_df)
        ),
        "unique_feedback_total": 20,
    },
    "validation_metrics": validation_metrics,
    "test_metrics": test_metrics,
}


metrics_path = (
    REPORT_DIR
    / "tfidf_logistic_metrics.json"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


print()
print(
    f"✓ Metrics saved:"
)

print(
    metrics_path
)


# ============================================================
# 17. SAVE TF-IDF VECTORIZER
# ============================================================

vectorizer_path = (
    MODEL_DIR
    / "tfidf_vectorizer.pkl"
)

with open(
    vectorizer_path,
    "wb"
) as file:

    pickle.dump(
        vectorizer,
        file
    )


print()
print(
    f"✓ TF-IDF vectorizer saved:"
)

print(
    vectorizer_path
)


# ============================================================
# 18. SAVE LOGISTIC REGRESSION MODEL
# ============================================================

model_path = (
    MODEL_DIR
    / "logistic_regression_model.pkl"
)

with open(
    model_path,
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )


print()
print(
    f"✓ Logistic Regression model saved:"
)

print(
    model_path
)


# ============================================================
# 19. TOP TF-IDF MODEL FEATURES
# ============================================================

print()
print("=" * 75)
print("TOP MODEL FEATURES")
print("=" * 75)

feature_names = (
    vectorizer
    .get_feature_names_out()
)

coefficients = (
    model.coef_[0]
)

feature_importance = pd.DataFrame(
    {
        "feature": feature_names,
        "coefficient": coefficients,
    }
)

positive_features = (
    feature_importance
    .sort_values(
        "coefficient",
        ascending=False
    )
    .head(10)
)

negative_features = (
    feature_importance
    .sort_values(
        "coefficient",
        ascending=True
    )
    .head(10)
)

print()
print("Features associated with POSITIVE sentiment:")

print(
    positive_features.to_string(
        index=False
    )
)

print()
print("Features associated with NEGATIVE sentiment:")

print(
    negative_features.to_string(
        index=False
    )
)

feature_path = (
    REPORT_DIR
    / "tfidf_feature_coefficients.csv"
)

feature_importance.to_csv(
    feature_path,
    index=False
)

print()
print(
    f"✓ Feature coefficients saved:"
)

print(
    feature_path
)


# ============================================================
# 20. FINAL
# ============================================================

print()
print("=" * 75)
print("STEP 31D - COMPLETED")
print("=" * 75)

print()
print("Model:")
print(
    "TF-IDF + Logistic Regression"
)

print()
print("Test metrics:")

for metric, value in test_metrics.items():

    print(
        f"{metric.capitalize():<12}: "
        f"{value:.4f}"
    )

print()
print(
    "Next:"
)

print(
    "31E - Transformer-based sentiment model"
)

print()
print("=" * 75)