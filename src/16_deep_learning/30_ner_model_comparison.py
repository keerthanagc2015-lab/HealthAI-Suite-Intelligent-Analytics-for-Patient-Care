"""
HealthAI - BioBERT vs ClinicalBERT NER Model Comparison

Purpose:
    Compare the two fine-tuned medical NER models using their
    held-out test evaluation results.

Models:
    1. BioBERT
    2. ClinicalBERT

Primary selection criterion:
    Entity-level F1

Secondary criteria:
    Entity-level Recall
    Entity-level Precision

The model with the highest held-out Entity F1 is selected.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BIOBERT_METRICS = (
    PROJECT_ROOT
    / "models"
    / "nlp"
    / "biobert_ner"
    / "evaluation"
    / "biobert_ner_test_metrics.json"
)

CLINICALBERT_METRICS = (
    PROJECT_ROOT
    / "models"
    / "nlp"
    / "clinicalbert_ner"
    / "evaluation"
    / "clinicalbert_ner_test_metrics.json"
)

BIOBERT_REPORT = (
    PROJECT_ROOT
    / "models"
    / "nlp"
    / "biobert_ner"
    / "evaluation"
    / "biobert_ner_entity_report.csv"
)

CLINICALBERT_REPORT = (
    PROJECT_ROOT
    / "models"
    / "nlp"
    / "clinicalbert_ner"
    / "evaluation"
    / "clinicalbert_ner_entity_report.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "nlp"
    / "ner_model_selection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. HEADER
# ============================================================

print("=" * 70)
print("HEALTHAI - BIOBERT VS CLINICALBERT NER COMPARISON")
print("=" * 70)

print()
print("Project root:")
print(PROJECT_ROOT)


# ============================================================
# 3. VALIDATE REQUIRED FILES
# ============================================================

print()
print("=" * 70)
print("VALIDATING EVALUATION ARTIFACTS")
print("=" * 70)

required_files = [
    BIOBERT_METRICS,
    CLINICALBERT_METRICS,
    BIOBERT_REPORT,
    CLINICALBERT_REPORT,
]

for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )

    print(f"✓ {file_path.name}")


# ============================================================
# 4. LOAD MODEL METRICS
# ============================================================

print()
print("=" * 70)
print("LOADING MODEL METRICS")
print("=" * 70)

with open(
    BIOBERT_METRICS,
    "r",
    encoding="utf-8"
) as f:

    biobert = json.load(f)


with open(
    CLINICALBERT_METRICS,
    "r",
    encoding="utf-8"
) as f:

    clinicalbert = json.load(f)


print("✓ BioBERT metrics loaded.")
print("✓ ClinicalBERT metrics loaded.")


# ============================================================
# 5. CREATE MODEL COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame(
    [
        {
            "model": "BioBERT",
            "model_name": "dmis-lab/biobert-v1.1",
            "token_accuracy": biobert["token_accuracy"],
            "entity_precision": biobert["entity_precision"],
            "entity_recall": biobert["entity_recall"],
            "entity_f1": biobert["entity_f1"],
            "test_documents": biobert["test_documents"],
            "test_chunks": biobert["test_chunks"],
        },
        {
            "model": "ClinicalBERT",
            "model_name": "emilyalsentzer/Bio_ClinicalBERT",
            "token_accuracy": clinicalbert["token_accuracy"],
            "entity_precision": clinicalbert["entity_precision"],
            "entity_recall": clinicalbert["entity_recall"],
            "entity_f1": clinicalbert["entity_f1"],
            "test_documents": clinicalbert["test_documents"],
            "test_chunks": clinicalbert["test_chunks"],
        },
    ]
)


# ============================================================
# 6. DISPLAY MODEL PERFORMANCE
# ============================================================

print()
print("=" * 70)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 70)

display_comparison = comparison.copy()

for column in [
    "token_accuracy",
    "entity_precision",
    "entity_recall",
    "entity_f1",
]:

    display_comparison[column] = (
        display_comparison[column]
        .mul(100)
        .round(2)
        .astype(str)
        + "%"
    )

print(
    display_comparison.to_string(
        index=False
    )
)


# ============================================================
# 7. SELECT BEST MODEL
# ============================================================

biobert_f1 = biobert["entity_f1"]
clinicalbert_f1 = clinicalbert["entity_f1"]

if biobert_f1 > clinicalbert_f1:

    selected_model = "BioBERT"

    selected_model_name = (
        "dmis-lab/biobert-v1.1"
    )

    selection_reason = (
        "BioBERT achieved the highest held-out "
        "entity-level F1 score."
    )

elif clinicalbert_f1 > biobert_f1:

    selected_model = "ClinicalBERT"

    selected_model_name = (
        "emilyalsentzer/Bio_ClinicalBERT"
    )

    selection_reason = (
        "ClinicalBERT achieved the highest held-out "
        "entity-level F1 score."
    )

else:

    selected_model = "Tie"

    selected_model_name = (
        "BioBERT / ClinicalBERT"
    )

    selection_reason = (
        "Both models achieved the same held-out "
        "entity-level F1 score."
    )


# ============================================================
# 8. CALCULATE PERFORMANCE DIFFERENCES
# ============================================================

accuracy_difference = (
    biobert["token_accuracy"]
    - clinicalbert["token_accuracy"]
)

precision_difference = (
    biobert["entity_precision"]
    - clinicalbert["entity_precision"]
)

recall_difference = (
    biobert["entity_recall"]
    - clinicalbert["entity_recall"]
)

f1_difference = (
    biobert["entity_f1"]
    - clinicalbert["entity_f1"]
)


# ============================================================
# 9. DISPLAY MODEL SELECTION
# ============================================================

print()
print("=" * 70)
print("MODEL SELECTION")
print("=" * 70)

print()
print(
    f"Selected model: {selected_model}"
)

print(
    f"Model: {selected_model_name}"
)

print()
print(
    f"BioBERT Entity F1: "
    f"{biobert_f1 * 100:.2f}%"
)

print(
    f"ClinicalBERT Entity F1: "
    f"{clinicalbert_f1 * 100:.2f}%"
)

print()
print(
    f"F1 difference: "
    f"{abs(f1_difference) * 100:.2f} "
    f"percentage points"
)

print()
print(
    f"Selection reason: "
    f"{selection_reason}"
)


# ============================================================
# 10. CREATE SELECTION SUMMARY
# ============================================================

selection_summary = {

    "selected_model": selected_model,

    "selected_model_name": selected_model_name,

    "selection_criterion": "entity_f1",

    "selection_reason": selection_reason,

    "biobert": {
        "model_name": "dmis-lab/biobert-v1.1",

        "token_accuracy": float(
            biobert["token_accuracy"]
        ),

        "entity_precision": float(
            biobert["entity_precision"]
        ),

        "entity_recall": float(
            biobert["entity_recall"]
        ),

        "entity_f1": float(
            biobert["entity_f1"]
        ),

        "test_documents": int(
            biobert["test_documents"]
        ),

        "test_chunks": int(
            biobert["test_chunks"]
        ),
    },

    "clinicalbert": {
        "model_name": (
            "emilyalsentzer/Bio_ClinicalBERT"
        ),

        "token_accuracy": float(
            clinicalbert["token_accuracy"]
        ),

        "entity_precision": float(
            clinicalbert["entity_precision"]
        ),

        "entity_recall": float(
            clinicalbert["entity_recall"]
        ),

        "entity_f1": float(
            clinicalbert["entity_f1"]
        ),

        "test_documents": int(
            clinicalbert["test_documents"]
        ),

        "test_chunks": int(
            clinicalbert["test_chunks"]
        ),
    },

    "differences_biobert_minus_clinicalbert": {

        "token_accuracy": float(
            accuracy_difference
        ),

        "entity_precision": float(
            precision_difference
        ),

        "entity_recall": float(
            recall_difference
        ),

        "entity_f1": float(
            f1_difference
        ),
    },
}


# ============================================================
# 11. SAVE MODEL COMPARISON CSV
# ============================================================

comparison_csv = (
    OUTPUT_DIR
    / "biobert_vs_clinicalbert_comparison.csv"
)

comparison.to_csv(
    comparison_csv,
    index=False
)

print()
print(
    f"✓ Saved: {comparison_csv}"
)


# ============================================================
# 12. SAVE MODEL SELECTION JSON
# ============================================================

selection_json = (
    OUTPUT_DIR
    / "ner_model_selection.json"
)

with open(
    selection_json,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        selection_summary,
        f,
        indent=4
    )

print(
    f"✓ Saved: {selection_json}"
)


# ============================================================
# 13. LOAD PER-ENTITY REPORTS
# ============================================================

print()
print("=" * 70)
print("LOADING PER-ENTITY REPORTS")
print("=" * 70)

biobert_report = pd.read_csv(
    BIOBERT_REPORT
)

clinicalbert_report = pd.read_csv(
    CLINICALBERT_REPORT
)

print(
    f"✓ BioBERT report: "
    f"{len(biobert_report)} rows"
)

print(
    f"✓ ClinicalBERT report: "
    f"{len(clinicalbert_report)} rows"
)


# ============================================================
# 14. REMOVE AGGREGATE METRIC ROWS
# ============================================================

aggregate_labels = {
    "micro avg",
    "macro avg",
    "weighted avg",
}

biobert_entities = biobert_report[
    ~biobert_report["label"].isin(
        aggregate_labels
    )
].copy()

clinicalbert_entities = clinicalbert_report[
    ~clinicalbert_report["label"].isin(
        aggregate_labels
    )
].copy()


# ============================================================
# 15. PREPARE BIOBERT PER-ENTITY DATA
# ============================================================

per_class = biobert_entities[
    [
        "label",
        "precision",
        "recall",
        "f1",
        "support",
    ]
].rename(
    columns={
        "precision": "biobert_precision",
        "recall": "biobert_recall",
        "f1": "biobert_f1",
        "support": "biobert_support",
    }
)


# ============================================================
# 16. PREPARE CLINICALBERT PER-ENTITY DATA
# ============================================================

clinicalbert_class = clinicalbert_entities[
    [
        "label",
        "precision",
        "recall",
        "f1",
        "support",
    ]
].rename(
    columns={
        "precision": "clinicalbert_precision",
        "recall": "clinicalbert_recall",
        "f1": "clinicalbert_f1",
        "support": "clinicalbert_support",
    }
)


# ============================================================
# 17. MERGE PER-ENTITY RESULTS
# ============================================================

per_class = per_class.merge(
    clinicalbert_class,
    on="label",
    how="outer"
)


# ============================================================
# 18. CALCULATE PER-ENTITY F1 DIFFERENCE
# ============================================================

per_class["f1_difference"] = (
    per_class["biobert_f1"].fillna(0)
    - per_class["clinicalbert_f1"].fillna(0)
)


# ============================================================
# 19. DETERMINE PER-ENTITY WINNER
# ============================================================

per_class["winner"] = np.where(
    per_class["f1_difference"] > 0,
    "BioBERT",
    np.where(
        per_class["f1_difference"] < 0,
        "ClinicalBERT",
        "Tie"
    )
)


# ============================================================
# 20. SORT BY F1 DIFFERENCE
# ============================================================

per_class = per_class.sort_values(
    "f1_difference",
    ascending=False
).reset_index(
    drop=True
)


# ============================================================
# 21. DISPLAY PER-ENTITY COMPARISON
# ============================================================

print()
print("=" * 70)
print("PER-ENTITY PERFORMANCE COMPARISON")
print("=" * 70)

display_per_class = per_class.copy()

for column in [
    "biobert_precision",
    "biobert_recall",
    "biobert_f1",
    "clinicalbert_precision",
    "clinicalbert_recall",
    "clinicalbert_f1",
    "f1_difference",
]:

    display_per_class[column] = (
        display_per_class[column]
        .fillna(0)
        .mul(100)
        .round(2)
        .astype(str)
        + "%"
    )

print(
    display_per_class.to_string(
        index=False
    )
)


# ============================================================
# 22. SAVE PER-ENTITY COMPARISON
# ============================================================

per_class_csv = (
    OUTPUT_DIR
    / "ner_per_class_comparison.csv"
)

per_class.to_csv(
    per_class_csv,
    index=False
)

print()
print(
    f"✓ Saved: {per_class_csv}"
)


# ============================================================
# 23. TOP BIOBERT ENTITY CLASSES
# ============================================================

print()
print("=" * 70)
print("TOP BIOBERT ENTITY CLASSES")
print("=" * 70)

top_biobert = (
    biobert_entities
    .sort_values(
        "f1",
        ascending=False
    )
    .head(5)
)

for _, row in top_biobert.iterrows():

    print(
        f"{row['label']:<25} "
        f"F1: {row['f1'] * 100:.2f}%"
    )


# ============================================================
# 24. TOP CLINICALBERT ENTITY CLASSES
# ============================================================

print()
print("=" * 70)
print("TOP CLINICALBERT ENTITY CLASSES")
print("=" * 70)

top_clinicalbert = (
    clinicalbert_entities
    .sort_values(
        "f1",
        ascending=False
    )
    .head(5)
)

for _, row in top_clinicalbert.iterrows():

    print(
        f"{row['label']:<25} "
        f"F1: {row['f1'] * 100:.2f}%"
    )


# ============================================================
# 25. COUNT ENTITY-LEVEL WINS
# ============================================================

biobert_wins = int(
    (per_class["winner"] == "BioBERT").sum()
)

clinicalbert_wins = int(
    (per_class["winner"] == "ClinicalBERT").sum()
)

ties = int(
    (per_class["winner"] == "Tie").sum()
)


print()
print("=" * 70)
print("PER-ENTITY WIN COUNTS")
print("=" * 70)

print(
    f"BioBERT wins:       {biobert_wins}"
)

print(
    f"ClinicalBERT wins:  {clinicalbert_wins}"
)

print(
    f"Ties:               {ties}"
)


# ============================================================
# 26. FINAL RESULT
# ============================================================

print()
print("=" * 70)
print("NER MODEL COMPARISON COMPLETED")
print("=" * 70)

print()

if selected_model == "BioBERT":

    print("🏆 SELECTED MODEL: BioBERT")

elif selected_model == "ClinicalBERT":

    print("🏆 SELECTED MODEL: ClinicalBERT")

else:

    print("🏆 SELECTED MODEL: TIE")


print()
print(
    f"BioBERT Entity F1: "
    f"{biobert_f1 * 100:.2f}%"
)

print(
    f"ClinicalBERT Entity F1: "
    f"{clinicalbert_f1 * 100:.2f}%"
)

print(
    f"F1 difference: "
    f"{abs(f1_difference) * 100:.2f} "
    f"percentage points"
)

print()
print("Output directory:")
print(OUTPUT_DIR)

print()
print("=" * 70)