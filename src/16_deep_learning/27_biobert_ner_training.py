import os
import json
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)


# ============================================================
# HEALTHAI - BIOBERT MEDICAL NER TRAINING
# ============================================================

print("=" * 70)
print("HEALTHAI - BIOBERT MEDICAL NER TRAINING")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

CHUNKS_FILE = (
    "data/processed/deep_learning/nlp/"
    "transformer/all_ner_chunks.json"
)

LABEL_MAP_FILE = (
    "data/processed/deep_learning/nlp/"
    "ner_label_map.json"
)

OUTPUT_DIR = (
    "data/processed/deep_learning/nlp/"
    "biobert_model"
)

RESULTS_DIR = (
    "data/processed/deep_learning/nlp/"
    "biobert_training_results"
)

MODEL_NAME = "dmis-lab/biobert-v1.1"

SEED = 42


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# RANDOM SEED
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# TRAINING DEVICE
# ============================================================

print("\n" + "=" * 70)
print("TRAINING DEVICE")
print("=" * 70)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Device:", device)


# ============================================================
# LOAD CHUNKED DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING CHUNKED NER DATA")
print("=" * 70)

with open(
    CHUNKS_FILE,
    "r",
    encoding="utf-8"
) as file:

    chunks = json.load(file)

print(
    "✓ Chunks loaded:",
    len(chunks)
)


# ============================================================
# LOAD LABEL MAP
# ============================================================

print("\nLoading label map...")

with open(
    LABEL_MAP_FILE,
    "r",
    encoding="utf-8"
) as file:

    label_data = json.load(file)


label2id = {
    str(key): int(value)
    for key, value in label_data["label2id"].items()
}

id2label = {
    int(key): str(value)
    for key, value in label_data["id2label"].items()
}

num_labels = len(label2id)

print(
    "✓ Number of labels:",
    num_labels
)


# ============================================================
# VERIFY LABEL COUNT
# ============================================================

if num_labels != 39:

    raise ValueError(
        f"Expected 39 labels, found {num_labels}"
    )

print(
    "✓ Label count verified: 39"
)


# ============================================================
# TRAIN / VALIDATION / TEST DOCUMENT SPLIT
# ============================================================

print("\n" + "=" * 70)
print("CREATING TRAIN / VALIDATION / TEST SPLITS")
print("=" * 70)

# Original document split:
#
# 0   - 279  → Training
# 280 - 339  → Validation
# 340 - 399  → Testing
#
# Step 26 preserved document_id for every chunk.

TRAIN_DOCUMENTS = 280
VALIDATION_DOCUMENTS = 60
TOTAL_DOCUMENTS = 400


train_chunks = []
validation_chunks = []
test_chunks = []


for chunk in chunks:

    document_id = int(
        chunk["document_id"]
    )

    if document_id < TRAIN_DOCUMENTS:

        train_chunks.append(chunk)

    elif document_id < (
        TRAIN_DOCUMENTS + VALIDATION_DOCUMENTS
    ):

        validation_chunks.append(chunk)

    elif document_id < TOTAL_DOCUMENTS:

        test_chunks.append(chunk)

    else:

        raise ValueError(
            f"Unexpected document_id: {document_id}"
        )


print(
    "\nTraining chunks:",
    len(train_chunks)
)

print(
    "Validation chunks:",
    len(validation_chunks)
)

print(
    "Testing chunks:",
    len(test_chunks)
)


# ============================================================
# VERIFY DOCUMENT SPLIT
# ============================================================

train_documents = sorted(
    set(
        int(chunk["document_id"])
        for chunk in train_chunks
    )
)

validation_documents = sorted(
    set(
        int(chunk["document_id"])
        for chunk in validation_chunks
    )
)

test_documents = sorted(
    set(
        int(chunk["document_id"])
        for chunk in test_chunks
    )
)


print("\nDocument counts:")

print(
    "Training documents:",
    len(train_documents)
)

print(
    "Validation documents:",
    len(validation_documents)
)

print(
    "Testing documents:",
    len(test_documents)
)


# ============================================================
# DATASET CLASS
# ============================================================

class NERDataset(Dataset):

    def __init__(
        self,
        chunk_list
    ):

        self.data = chunk_list


    def __len__(self):

        return len(self.data)


    def __getitem__(
        self,
        index
    ):

        item = self.data[index]

        return {

            "input_ids": item["input_ids"],

            "attention_mask": item[
                "attention_mask"
            ],

            "labels": item["labels"]

        }


train_dataset = NERDataset(
    train_chunks
)

validation_dataset = NERDataset(
    validation_chunks
)

test_dataset = NERDataset(
    test_chunks
)


# ============================================================
# LOAD BIOBERT TOKENIZER
# ============================================================

print("\n" + "=" * 70)
print("LOADING BIOBERT")
print("=" * 70)

print(
    "Model:",
    MODEL_NAME
)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


print(
    "✓ BioBERT tokenizer loaded."
)


# ============================================================
# LOAD BIOBERT MODEL
# ============================================================

print("\nLoading BioBERT model...")


model = AutoModelForTokenClassification.from_pretrained(

    MODEL_NAME,

    num_labels=num_labels,

    id2label=id2label,

    label2id=label2id

)


print(
    "✓ BioBERT model loaded."
)


print(
    "Model parameters:",
    sum(
        parameter.numel()
        for parameter in model.parameters()
    )
)


# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = (
    DataCollatorForTokenClassification(
        tokenizer=tokenizer
    )
)


# ============================================================
# METRICS
# ============================================================


def compute_metrics(evaluation_prediction):
    predictions = np.argmax(
        evaluation_prediction.predictions, axis=-1
    )
    labels = evaluation_prediction.label_ids

    # Convert IDs into BIO tags, ignoring special tokens/padding.
    true_sequences = []
    pred_sequences = []

    for prediction_row, label_row in zip(predictions, labels):
        true_tags = []
        pred_tags = []

        for prediction, label in zip(prediction_row, label_row):
            if label == -100:
                continue

            true_tags.append(id2label[int(label)])
            pred_tags.append(id2label[int(prediction)])

        true_sequences.append(true_tags)
        pred_sequences.append(pred_tags)

    # Entity-level evaluation using strict BIO span matching.
    tp = fp = fn = 0

    for true_tags, pred_tags in zip(true_sequences, pred_sequences):
        true_entities = extract_entities(true_tags)
        pred_entities = extract_entities(pred_tags)

        tp += len(true_entities & pred_entities)
        fp += len(pred_entities - true_entities)
        fn += len(true_entities - pred_entities)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    valid = labels != -100
    token_accuracy = (
        float((predictions[valid] == labels[valid]).mean())
        if valid.any() else 0.0
    )

    return {
        "token_accuracy": token_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def extract_entities(tags):
    """Return entity spans as (entity_type, start, end) tuples."""
    entities = set()
    current_type = None
    start = None

    for i, tag in enumerate(tags + ["O"]):
        if tag == "O" or "-" not in tag:
            next_prefix, next_type = "O", None
        else:
            next_prefix, next_type = tag.split("-", 1)

        if current_type is not None and (
            next_prefix != "I" or next_type != current_type
        ):
            entities.add((current_type, start, i))
            current_type = None
            start = None

        if next_prefix == "B":
            current_type = next_type
            start = i
        elif next_prefix == "I" and current_type is None:
            # Handle an invalid I-tag as the beginning of a span.
            current_type = next_type
            start = i

    return entities



# ============================================================
# TRAINING CONFIGURATION
# ============================================================

print("\n" + "=" * 70)
print("CONFIGURING TRAINING")
print("=" * 70)


training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    # Evaluate after each epoch
    eval_strategy="epoch",

    # Save checkpoint after each epoch
    save_strategy="epoch",

    # Display training logs
    logging_strategy="steps",

    logging_steps=25,

    # Standard fine-tuning learning rate
    learning_rate=2e-5,

    # Small batch because training is on CPU
    per_device_train_batch_size=2,

    per_device_eval_batch_size=2,

    # Number of training passes
    num_train_epochs=3,

    # Regularization
    weight_decay=0.01,

    # Keep best model
    load_best_model_at_end=True,

    metric_for_best_model="f1",
    greater_is_better=True,

    # Keep only two checkpoints
    save_total_limit=2,

    # No external experiment tracking
    report_to="none",

    # Reproducibility
    seed=SEED,

    # CPU training
    fp16=False

)


print(
    "Epochs: 3"
)

print(
    "Learning rate: 2e-5"
)

print(
    "Training batch size: 2"
)

print(
    "Validation batch size: 2"
)

print(
    "Weight decay: 0.01"
)


# ============================================================
# CREATE TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics

)


# ============================================================
# START BIOBERT TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STARTING BIOBERT TRAINING")
print("=" * 70)

print(
    "\nTraining may take some time on CPU."
)


train_result = trainer.train()


# ============================================================
# SAVE TRAINED MODEL
# ============================================================

print("\n" + "=" * 70)
print("SAVING TRAINED BIOBERT")
print("=" * 70)


trainer.save_model(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)


print(
    "✓ Trained BioBERT saved."
)

print(
    "Location:",
    OUTPUT_DIR
)


# ============================================================
# VALIDATION EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION EVALUATION")
print("=" * 70)


validation_results = trainer.evaluate(
    eval_dataset=validation_dataset
)


for key, value in validation_results.items():

    if isinstance(value, float):

        print(
            f"{key}: {value:.4f}"
        )

    else:

        print(
            f"{key}: {value}"
        )


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)


test_results = trainer.evaluate(

    eval_dataset=test_dataset,

    metric_key_prefix="test"

)


for key, value in test_results.items():

    if isinstance(value, float):

        print(
            f"{key}: {value:.4f}"
        )

    else:

        print(
            f"{key}: {value}"
        )


# ============================================================
# SAVE TRAINING METRICS
# ============================================================

metrics = {

    "model": "BioBERT",

    "base_model": MODEL_NAME,

    "training_documents":
        len(train_documents),

    "validation_documents":
        len(validation_documents),

    "test_documents":
        len(test_documents),

    "training_chunks":
        len(train_chunks),

    "validation_chunks":
        len(validation_chunks),

    "test_chunks":
        len(test_chunks),

    "num_labels":
        num_labels,

    "epochs":
        3,

    "learning_rate":
        2e-5,

    "validation_token_accuracy":
        validation_results.get("eval_token_accuracy"),

    "validation_entity_precision":
        validation_results.get("eval_precision"),

    "validation_entity_recall":
        validation_results.get("eval_recall"),

    "validation_entity_f1":
        validation_results.get("eval_f1"),

    "test_token_accuracy":
        test_results.get("test_token_accuracy"),

    "test_entity_precision":
        test_results.get("test_precision"),

    "test_entity_recall":
        test_results.get("test_recall"),

    "test_entity_f1":
        test_results.get("test_f1")

}


metrics_path = os.path.join(

    RESULTS_DIR,

    "biobert_training_metrics.json"

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


# ============================================================
# SAVE TRAINING LOG
# ============================================================

if trainer.state.log_history:

    log_df = pd.DataFrame(
        trainer.state.log_history
    )


    log_path = os.path.join(

        RESULTS_DIR,

        "biobert_training_log.csv"

    )


    log_df.to_csv(

        log_path,

        index=False

    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("BIOBERT NER TRAINING COMPLETED")
print("=" * 70)


print(
    "\nModel:",
    MODEL_NAME
)

print(
    "Training documents:",
    len(train_documents)
)

print(
    "Validation documents:",
    len(validation_documents)
)

print(
    "Test documents:",
    len(test_documents)
)

print(
    "Training chunks:",
    len(train_chunks)
)

print(
    "Validation chunks:",
    len(validation_chunks)
)

print(
    "Test chunks:",
    len(test_chunks)
)

print(
    "Number of labels:",
    num_labels
)


print(
    "\nModel saved to:"
)

print(
    OUTPUT_DIR
)


print(
    "\nMetrics saved to:"
)

print(
    metrics_path
)


print("\n" + "=" * 70)