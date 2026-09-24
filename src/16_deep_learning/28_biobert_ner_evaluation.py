"""
HealthAI - BioBERT NER Evaluation

Purpose:
    Evaluate the locally stored BioBERT NER model on the same
    document-level held-out test set used during training.

Model:
    dmis-lab/biobert-v1.1

Expected training setup:
    - 400 English documents
    - 280 train documents
    - 60 validation documents
    - 60 test documents
    - 838 total chunks
    - 39 BIO labels
    - Maximum sequence length: 512
    - Document-level split to prevent data leakage

Important:
    BioBERT was trained using Google Colab + T4 GPU.
    This script performs LOCAL evaluation only.
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)

from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "nlp"
    / "biobert_ner"
)

LABEL_MAP_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "deep_learning"
    / "nlp"
    / "ner_label_map.json"
)

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "deep_learning"
    / "nlp"
    / "transformer"
    / "all_ner_chunks.json"
)

OUTPUT_DIR = (
    MODEL_PATH
    / "evaluation"
)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 3. CONFIGURATION
# ============================================================

MAX_LENGTH = 512

TEST_DOCUMENT_COUNT = 60

EXPECTED_CHUNK_COUNT = 838

EXPECTED_TOKEN_ACCURACY = 0.7816
EXPECTED_ENTITY_PRECISION = 0.4101
EXPECTED_ENTITY_RECALL = 0.4602
EXPECTED_ENTITY_F1 = 0.4337


# ============================================================
# 4. PRINT HEADER
# ============================================================

print("=" * 70)
print("HEALTHAI - BIOBERT NER EVALUATION")
print("=" * 70)

print()
print("Project root:")
print(PROJECT_ROOT)

print()
print("Model path:")
print(MODEL_PATH)

print()
print("Label map:")
print(LABEL_MAP_PATH)

print()
print("Chunk data:")
print(CHUNKS_PATH)


# ============================================================
# 5. VALIDATE REQUIRED FILES
# ============================================================

print()
print("=" * 70)
print("VALIDATING REQUIRED FILES")
print("=" * 70)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"BioBERT model not found:\n{MODEL_PATH}"
    )

if not LABEL_MAP_PATH.exists():
    raise FileNotFoundError(
        f"Label map not found:\n{LABEL_MAP_PATH}"
    )

if not CHUNKS_PATH.exists():
    raise FileNotFoundError(
        f"NER chunks not found:\n{CHUNKS_PATH}"
    )

print("✓ BioBERT model found.")
print("✓ Label map found.")
print("✓ Chunked NER dataset found.")


# ============================================================
# 6. CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 7. LOAD LABEL MAP
# ============================================================

print()
print("=" * 70)
print("LOADING LABEL MAP")
print("=" * 70)

with open(
    LABEL_MAP_PATH,
    "r",
    encoding="utf-8"
) as f:

    label_data = json.load(f)


# The saved label map contains:
#   label2id
#   id2label
#   entity_labels

label2id = label_data["label2id"]

# JSON converts dictionary keys to strings.
# Convert ID keys back to integers.

id2label = {
    int(k): v
    for k, v in label_data["id2label"].items()
}

entity_labels = label_data.get(
    "entity_labels",
    []
)

print(f"Number of labels: {len(label2id)}")
print(f"Number of entity types: {len(entity_labels)}")

print()
print("Label mapping:")

for label_id in sorted(id2label):
    print(
        f"{label_id:2d} -> {id2label[label_id]}"
    )


# ============================================================
# 8. LOAD CHUNKED DATA
# ============================================================

print()
print("=" * 70)
print("LOADING CHUNKED NER DATA")
print("=" * 70)

with open(
    CHUNKS_PATH,
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)

print(f"Total chunks loaded: {len(chunks)}")

if len(chunks) != EXPECTED_CHUNK_COUNT:
    print()
    print(
        f"WARNING: Expected {EXPECTED_CHUNK_COUNT} chunks "
        f"but found {len(chunks)}."
    )

else:
    print(
        f"✓ Chunk count matches expected: "
        f"{EXPECTED_CHUNK_COUNT}"
    )


# ============================================================
# 9. VERIFY CHUNK STRUCTURE
# ============================================================

print()
print("=" * 70)
print("VALIDATING CHUNK STRUCTURE")
print("=" * 70)

required_chunk_fields = [
    "input_ids",
    "attention_mask",
    "labels",
]

for field in required_chunk_fields:

    if field not in chunks[0]:

        raise KeyError(
            f"Required field '{field}' "
            "not found in chunk data."
        )

print("✓ Required chunk fields found.")


# ============================================================
# 10. RECREATE DOCUMENT-LEVEL SPLIT
# ============================================================

print()
print("=" * 70)
print("RECREATING DOCUMENT-LEVEL TEST SPLIT")
print("=" * 70)


# Each chunk contains a document identifier.
# We must split by DOCUMENT, not by individual chunks,
# to avoid data leakage.

def get_document_id(chunk):

    possible_keys = [
        "document_id",
        "doc_id",
        "source_document_id",
    ]

    for key in possible_keys:

        if key in chunk:
            return chunk[key]

    raise KeyError(
        "No document identifier found in chunk."
    )


document_ids = [
    get_document_id(chunk)
    for chunk in chunks
]

unique_documents = sorted(
    set(document_ids),
    key=lambda x: str(x)
)

print(
    f"Unique documents: "
    f"{len(unique_documents)}"
)

if len(unique_documents) != 400:

    raise ValueError(
        f"Expected 400 documents but found "
        f"{len(unique_documents)}."
    )


# ------------------------------------------------------------
# Same random seed and document-level split
# ------------------------------------------------------------

rng = random.Random(SEED)

shuffled_documents = unique_documents.copy()

rng.shuffle(shuffled_documents)

train_documents = shuffled_documents[:280]

validation_documents = shuffled_documents[280:340]

test_documents = shuffled_documents[340:400]


print(
    f"Training documents: "
    f"{len(train_documents)}"
)

print(
    f"Validation documents: "
    f"{len(validation_documents)}"
)

print(
    f"Testing documents: "
    f"{len(test_documents)}"
)


# ============================================================
# 11. SELECT TEST CHUNKS
# ============================================================

test_document_set = set(test_documents)

test_chunks = [
    chunk
    for chunk in chunks
    if get_document_id(chunk)
    in test_document_set
]


print()
print(
    f"Test chunks: {len(test_chunks)}"
)


# ============================================================
# 12. LOAD BIOBERT TOKENIZER
# ============================================================

print()
print("=" * 70)
print("LOADING BIOBERT TOKENIZER")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

print("✓ Tokenizer loaded.")

print(
    f"Vocabulary size: "
    f"{tokenizer.vocab_size}"
)


# ============================================================
# 13. LOAD BIOBERT MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING BIOBERT MODEL")
print("=" * 70)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Device: {device}")

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
)

model.to(device)

model.eval()

print("✓ BioBERT model loaded.")

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Parameters: "
    f"{parameter_count:,}"
)


# ============================================================
# 14. MODEL EVALUATION
# ============================================================

print()
print("=" * 70)
print("RUNNING BIOBERT EVALUATION")
print("=" * 70)


all_true_labels = []
all_pred_labels = []

correct_tokens = 0
total_valid_tokens = 0


with torch.no_grad():

    for index, chunk in enumerate(
        test_chunks,
        start=1
    ):

        input_ids = torch.tensor(
            chunk["input_ids"],
            dtype=torch.long
        ).unsqueeze(0).to(device)

        attention_mask = torch.tensor(
            chunk["attention_mask"],
            dtype=torch.long
        ).unsqueeze(0).to(device)

        true_ids = chunk["labels"]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        predictions = torch.argmax(
            outputs.logits,
            dim=-1
        )

        pred_ids = (
            predictions
            .squeeze(0)
            .cpu()
            .tolist()
        )

        # ----------------------------------------------------
        # Convert IDs to labels
        # ----------------------------------------------------

        true_sequence = []
        pred_sequence = []

        for true_id, pred_id in zip(
            true_ids,
            pred_ids
        ):

            # -100 means ignored token.
            #
            # This includes special tokens and
            # any labels intentionally excluded
            # during token-label alignment.

            if true_id == -100:
                continue

            true_label = id2label.get(
                int(true_id),
                "O"
            )

            pred_label = id2label.get(
                int(pred_id),
                "O"
            )

            true_sequence.append(
                true_label
            )

            pred_sequence.append(
                pred_label
            )

            total_valid_tokens += 1

            if true_id == pred_id:
                correct_tokens += 1

        all_true_labels.append(
            true_sequence
        )

        all_pred_labels.append(
            pred_sequence
        )

        if index % 20 == 0:

            print(
                f"Evaluated "
                f"{index}/{len(test_chunks)} chunks"
            )


# ============================================================
# 15. TOKEN ACCURACY
# ============================================================

if total_valid_tokens == 0:

    raise RuntimeError(
        "No valid tokens were found during evaluation."
    )

token_accuracy = (
    correct_tokens
    / total_valid_tokens
)


# ============================================================
# 16. ENTITY-LEVEL METRICS
# ============================================================

entity_precision = precision_score(
    all_true_labels,
    all_pred_labels,
    average="micro",
)

entity_recall = recall_score(
    all_true_labels,
    all_pred_labels,
    average="micro",
)

entity_f1 = f1_score(
    all_true_labels,
    all_pred_labels,
    average="micro",
)


# ============================================================
# 17. PRINT MAIN RESULTS
# ============================================================

print()
print("=" * 70)
print("BIOBERT NER TEST RESULTS")
print("=" * 70)

print(
    f"Test chunks:        {len(test_chunks)}"
)

print(
    f"Valid tokens:       {total_valid_tokens:,}"
)

print(
    f"Token Accuracy:     {token_accuracy:.4f} "
    f"({token_accuracy * 100:.2f}%)"
)

print(
    f"Entity Precision:   {entity_precision:.4f} "
    f"({entity_precision * 100:.2f}%)"
)

print(
    f"Entity Recall:      {entity_recall:.4f} "
    f"({entity_recall * 100:.2f}%)"
)

print(
    f"Entity F1:          {entity_f1:.4f} "
    f"({entity_f1 * 100:.2f}%)"
)


# ============================================================
# 18. DETAILED ENTITY CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print("ENTITY-LEVEL CLASSIFICATION REPORT")
print("=" * 70)

report_text = classification_report(
    all_true_labels,
    all_pred_labels,
    digits=4,
    zero_division=0,
)

print(report_text)


# ============================================================
# 19. SAVE ENTITY REPORT
# ============================================================

report_dict = classification_report(
    all_true_labels,
    all_pred_labels,
    digits=4,
    output_dict=True,
    zero_division=0,
)

report_rows = []

for label_name, metrics in report_dict.items():

    if isinstance(metrics, dict):

        report_rows.append(
            {
                "label": label_name,
                "precision": metrics.get(
                    "precision",
                    0
                ),
                "recall": metrics.get(
                    "recall",
                    0
                ),
                "f1": metrics.get(
                    "f1-score",
                    0
                ),
                "support": metrics.get(
                    "support",
                    0
                ),
            }
        )


entity_report_df = pd.DataFrame(
    report_rows
)

entity_report_path = (
    OUTPUT_DIR
    / "biobert_ner_entity_report.csv"
)

entity_report_df.to_csv(
    entity_report_path,
    index=False
)


# ============================================================
# 20. SAVE TEST METRICS
# ============================================================

metrics = {
    "model": "dmis-lab/biobert-v1.1",
    "evaluation_type": "held_out_test",
    "device": str(device),
    "test_documents": len(test_documents),
    "test_chunks": len(test_chunks),
    "valid_tokens": total_valid_tokens,
    "num_labels": len(label2id),
    "num_entity_types": len(entity_labels),
    "token_accuracy": float(token_accuracy),
    "entity_precision": float(entity_precision),
    "entity_recall": float(entity_recall),
    "entity_f1": float(entity_f1),
}


metrics_path = (
    OUTPUT_DIR
    / "biobert_ner_test_metrics.json"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# 21. COMPARE WITH COLAB RESULT
# ============================================================

print()
print("=" * 70)
print("COMPARISON WITH COLAB EVALUATION")
print("=" * 70)

comparison_df = pd.DataFrame(
    {
        "Metric": [
            "Token Accuracy",
            "Entity Precision",
            "Entity Recall",
            "Entity F1",
        ],
        "Colab Result": [
            EXPECTED_TOKEN_ACCURACY,
            EXPECTED_ENTITY_PRECISION,
            EXPECTED_ENTITY_RECALL,
            EXPECTED_ENTITY_F1,
        ],
        "Local Result": [
            token_accuracy,
            entity_precision,
            entity_recall,
            entity_f1,
        ],
    }
)

comparison_df["Absolute Difference"] = (
    comparison_df["Local Result"]
    - comparison_df["Colab Result"]
).abs()

print(
    comparison_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 22. SAVE COMPARISON
# ============================================================

comparison_path = (
    OUTPUT_DIR
    / "biobert_vs_colab_evaluation_comparison.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# 23. FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("BIOBERT NER EVALUATION COMPLETED")
print("=" * 70)

print()
print("Final metrics:")

print(
    f"Token Accuracy : "
    f"{token_accuracy * 100:.2f}%"
)

print(
    f"Entity Precision : "
    f"{entity_precision * 100:.2f}%"
)

print(
    f"Entity Recall : "
    f"{entity_recall * 100:.2f}%"
)

print(
    f"Entity F1 : "
    f"{entity_f1 * 100:.2f}%"
)

print()
print("Files saved:")

print(
    entity_report_path
)

print(
    metrics_path
)

print(
    comparison_path
)

print()
print("=" * 70)