# ==========================================================
# Import Libraries
# ==========================================================

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================================
# Evaluate Model
# ==========================================================

def evaluate_model(model_name, y_test, y_pred):

    print("\n" + "=" * 70)
    print(f"{model_name} - Evaluation")
    print("=" * 70)

    accuracy = accuracy_score(y_test, y_pred)

    macro_precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("\nSummary Metrics")
    print("-" * 50)

    print(f"Accuracy        : {accuracy:.4f}")
    print(f"Macro Precision : {macro_precision:.4f}")
    print(f"Macro Recall    : {macro_recall:.4f}")
    print(f"Macro F1        : {macro_f1:.4f}")
    print(f"Weighted F1     : {weighted_f1:.4f}")

    print("\nClassification Report")
    print("-" * 50)

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    print("\nConfusion Matrix")
    print("-" * 50)

    print(confusion_matrix(y_test, y_pred))

    results = {

        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": macro_precision,
        "Recall": macro_recall,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1

    }

    return results


# ==========================================================
# Compare Models
# ==========================================================

def compare_models(results):

    comparison = pd.DataFrame(results)

    print("\n" + "=" * 110)
    print("FINAL MODEL COMPARISON")
    print("=" * 110)

    print(comparison.to_string(index=False))

    return comparison