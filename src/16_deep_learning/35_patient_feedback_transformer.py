"""
HEALTHAI - Transformer-Based Patient Feedback Sentiment Model

Step 31E
Model: DistilBERT
Task: Binary sentiment classification

Important dataset limitation:
The dataset contains only 20 unique feedback texts.
Therefore, evaluation results are illustrative and should not
be interpreted as statistically reliable production performance.
"""

import json
import os
import random

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

MODEL_NAME = "distilbert-base-uncased"

BASE_DIR = "data/processed/deep_learning/medical_sentiment"
TRAIN_PATH = os.path.join(BASE_DIR, "splits", "train.csv")
VAL_PATH = os.path.join(BASE_DIR, "splits", "validation.csv")
TEST_PATH = os.path.join(BASE_DIR, "splits", "test.csv")

MODEL_DIR = "models/nlp/sentiment/distilbert"

REPORT_DIR = os.path.join(
    BASE_DIR,
    "transformer"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("HEALTHAI - TRANSFORMER SENTIMENT CLASSIFICATION")
print("=" * 75)

print()
print(f"Model: {MODEL_NAME}")
print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")


# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

print()
print("=" * 75)
print("DATASETS LOADED")
print("=" * 75)

print(f"Train:      {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test:       {len(test_df)}")

print()
print("Columns:")
print(train_df.columns.tolist())


# ============================================================
# PREPARE DATASET
# ============================================================

def prepare_dataframe(df):
    output = df[["Feedback", "Sentiment"]].copy()

    output["Feedback"] = output["Feedback"].astype(str)
    output["Sentiment"] = output["Sentiment"].astype(int)

    return output


train_df = prepare_dataframe(train_df)
val_df = prepare_dataframe(val_df)
test_df = prepare_dataframe(test_df)


# ============================================================
# CONVERT TO HUGGING FACE DATASETS
# ============================================================

train_dataset = Dataset.from_pandas(
    train_df,
    preserve_index=False
)

val_dataset = Dataset.from_pandas(
    val_df,
    preserve_index=False
)

test_dataset = Dataset.from_pandas(
    test_df,
    preserve_index=False
)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print()
print("=" * 75)
print("LOADING TOKENIZER")
print("=" * 75)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("✓ Tokenizer loaded.")


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_function(batch):
    return tokenizer(
        batch["Feedback"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )


train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

val_dataset = val_dataset.map(
    tokenize_function,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)

train_dataset = train_dataset.rename_column(
    "Sentiment",
    "labels"
)

val_dataset = val_dataset.rename_column(
    "Sentiment",
    "labels"
)

test_dataset = test_dataset.rename_column(
    "Sentiment",
    "labels"
)

columns = [
    "input_ids",
    "attention_mask",
    "labels"
]

train_dataset.set_format(
    type="torch",
    columns=columns
)

val_dataset.set_format(
    type="torch",
    columns=columns
)

test_dataset.set_format(
    type="torch",
    columns=columns
)

print()
print("✓ Tokenization completed.")
print("Maximum sequence length: 128")


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 75)
print("LOADING TRANSFORMER MODEL")
print("=" * 75)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={
        0: "negative",
        1: "positive",
    },
    label2id={
        "negative": 0,
        "positive": 1,
    },
)

print("✓ Transformer model loaded.")
print(
    f"Parameters: "
    f"{sum(p.numel() for p in model.parameters()):,}"
)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_prediction):

    predictions, labels = eval_prediction

    predictions = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(
    output_dir=MODEL_DIR,

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,

    num_train_epochs=5,

    weight_decay=0.01,

    logging_strategy="epoch",

    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,

    save_total_limit=2,

    report_to="none",

    seed=RANDOM_STATE,
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=val_dataset,

    compute_metrics=compute_metrics,
)


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 75)
print("TRAINING TRANSFORMER")
print("=" * 75)

trainer.train()

print()
print("✓ Transformer training completed.")


# ============================================================
# SAVE MODEL
# ============================================================

trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

print()
print("✓ Model saved:")
print(os.path.abspath(MODEL_DIR))


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 75)
print("VALIDATION RESULTS")
print("=" * 75)

val_results = trainer.evaluate(
    eval_dataset=val_dataset
)

print(
    f"Accuracy    : {val_results['eval_accuracy']:.4f}"
)

print(
    f"Precision   : {val_results['eval_precision']:.4f}"
)

print(
    f"Recall      : {val_results['eval_recall']:.4f}"
)

print(
    f"F1          : {val_results['eval_f1']:.4f}"
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

print()
print("=" * 75)
print("TEST EVALUATION")
print("=" * 75)

test_output = trainer.predict(test_dataset)

test_logits = test_output.predictions
test_labels = test_output.label_ids

test_predictions = np.argmax(
    test_logits,
    axis=1
)


# ============================================================
# TEST METRICS
# ============================================================

test_precision, test_recall, test_f1, _ = (
    precision_recall_fscore_support(
        test_labels,
        test_predictions,
        average="binary",
        zero_division=0,
    )
)

test_accuracy = accuracy_score(
    test_labels,
    test_predictions
)

print(
    f"Accuracy    : {test_accuracy:.4f}"
)

print(
    f"Precision   : {test_precision:.4f}"
)

print(
    f"Recall      : {test_recall:.4f}"
)

print(
    f"F1          : {test_f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 75)
print("TEST CLASSIFICATION REPORT")
print("=" * 75)

report = classification_report(
    test_labels,
    test_predictions,
    labels=[0, 1],
    target_names=["negative", "positive"],
    zero_division=0,
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_labels,
    test_predictions,
    labels=[0, 1]
)

print()
print("=" * 75)
print("CONFUSION MATRIX")
print("=" * 75)

print("Rows = Actual")
print("Columns = Predicted")
print()
print(cm)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(
    cm,
    index=["Actual Negative", "Actual Positive"],
    columns=["Predicted Negative", "Predicted Positive"]
)

cm_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "distilbert_confusion_matrix.csv"
    )
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

prediction_df = test_df.copy()

prediction_df["Predicted_Sentiment"] = (
    test_predictions
)

prediction_df["Predicted_Label"] = (
    prediction_df["Predicted_Sentiment"]
    .map({
        0: "negative",
        1: "positive"
    })
)

prediction_df["Actual_Label"] = (
    prediction_df["Sentiment"]
    .map({
        0: "negative",
        1: "positive"
    })
)

prediction_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "distilbert_test_predictions.csv"
    ),
    index=False
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "model": MODEL_NAME,
    "task": "binary_sentiment_classification",

    "train_samples": len(train_df),
    "validation_samples": len(val_df),
    "test_samples": len(test_df),

    "test_accuracy": float(test_accuracy),
    "test_precision": float(test_precision),
    "test_recall": float(test_recall),
    "test_f1": float(test_f1),

    "device": (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    ),

    "dataset_unique_feedback_count": 20,

    "evaluation_warning": (
        "The dataset contains only 20 unique feedback texts. "
        "Therefore, test metrics are illustrative and should "
        "not be interpreted as statistically reliable production performance."
    ),
}

with open(
    os.path.join(
        REPORT_DIR,
        "distilbert_sentiment_metrics.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 75)
print("STEP 31E - COMPLETED")
print("=" * 75)

print()
print("Model:")
print("DistilBERT Transformer")

print()
print("Test metrics:")
print(f"Accuracy    : {test_accuracy:.4f}")
print(f"Precision   : {test_precision:.4f}")
print(f"Recall      : {test_recall:.4f}")
print(f"F1          : {test_f1:.4f}")

print()
print("Important:")
print(
    "Only 20 unique feedback texts exist in the dataset."
)
print(
    "Therefore, these metrics are illustrative."
)

print()
print("Saved model:")
print(os.path.abspath(MODEL_DIR))

print()
print("Next:")
print("31F - Sentiment model evaluation and comparison")
print("=" * 75)