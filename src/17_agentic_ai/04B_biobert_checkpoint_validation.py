"""
===============================================================================
HEALTHAI - BIOBERT CHECKPOINT VALIDATION
===============================================================================

Module:
    17_agentic_ai

Step:
    17.4B - Validate local BioBERT checkpoint against the actual
            tokenized NER dataset.

Purpose:
    Verify whether the local BioBERT checkpoint is producing the expected
    medical NER predictions on the same tokenized data used during training/
    evaluation.

The all_ner_chunks.json structure contains:

    chunk_id
    document_id
    input_ids
    attention_mask
    labels
    offset_mapping

There is no raw text field.

Therefore this script works directly with input_ids and labels.

It reports:
    - Gold non-O labels
    - Predicted non-O labels
    - Token accuracy
    - Number of non-O predictions
    - Example predictions
    - Whether the checkpoint appears to be functioning

===============================================================================
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)


# =============================================================================
# PATHS
# =============================================================================

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
    PROJECT_ROOT
    / "data"
    / "processed"
    / "agentic_ai"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "biobert_checkpoint_validation.json"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

RANDOM_SEED = 42

NUMBER_OF_EXAMPLES = 10


# =============================================================================
# GLOBALS
# =============================================================================

TOKENIZER = None
MODEL = None

LABEL_MAP = {}
ID_TO_LABEL = {}


# =============================================================================
# LOAD LABEL MAP
# =============================================================================

def load_label_map() -> None:
    """
    Load the project's BIO label map.

    Actual structure:

    {
        "label2id": {
            "O": 0,
            "B-Comorbidity": 1,
            ...
        },
        "id2label": {
            ...
        }
    }
    """

    global LABEL_MAP
    global ID_TO_LABEL

    print(
        "Loading label map..."
    )

    if not LABEL_MAP_PATH.exists():

        raise FileNotFoundError(
            f"Label map not found:\n"
            f"{LABEL_MAP_PATH}"
        )

    with open(
        LABEL_MAP_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        raw_map = json.load(file)

    if "label2id" in raw_map:

        LABEL_MAP = {
            str(label): int(label_id)
            for label, label_id
            in raw_map["label2id"].items()
        }

    else:

        LABEL_MAP = {
            str(label): int(label_id)
            for label, label_id
            in raw_map.items()
            if not isinstance(label_id, dict)
        }

    ID_TO_LABEL = {
        int(label_id): label
        for label, label_id
        in LABEL_MAP.items()
    }

    print(
        f"Loaded {len(LABEL_MAP)} labels."
    )

    if len(LABEL_MAP) != 39:

        print(
            f"WARNING: Expected 39 labels, "
            f"found {len(LABEL_MAP)}."
        )


# =============================================================================
# LOAD MODEL
# =============================================================================

def load_model() -> None:

    global TOKENIZER
    global MODEL

    print()
    print("=" * 79)
    print(
        "LOADING LOCAL BIOBERT CHECKPOINT"
    )
    print("=" * 79)

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"BioBERT model directory not found:\n"
            f"{MODEL_PATH}"
        )

    TOKENIZER = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        use_fast=True
    )

    MODEL = AutoModelForTokenClassification.from_pretrained(
        str(MODEL_PATH)
    )

    MODEL.to(DEVICE)

    MODEL.eval()

    parameter_count = sum(
        parameter.numel()
        for parameter in MODEL.parameters()
    )

    print(
        "Model loaded."
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Parameters: {parameter_count:,}"
    )

    print(
        f"Model labels: "
        f"{MODEL.config.num_labels}"
    )


# =============================================================================
# LOAD CHUNKS
# =============================================================================

def load_chunks() -> List[Dict[str, Any]]:
    """
    Load the pre-tokenized NER chunks.
    """

    print()
    print(
        "Loading real NER chunks..."
    )

    if not CHUNKS_PATH.exists():

        raise FileNotFoundError(
            f"NER chunks not found:\n"
            f"{CHUNKS_PATH}"
        )

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        raw_data = json.load(file)

    if isinstance(raw_data, list):

        chunks = raw_data

    elif isinstance(raw_data, dict):

        if isinstance(
            raw_data.get("chunks"),
            list
        ):

            chunks = raw_data["chunks"]

        elif isinstance(
            raw_data.get("data"),
            list
        ):

            chunks = raw_data["data"]

        else:

            raise ValueError(
                "Could not find chunk list."
            )

    else:

        raise ValueError(
            "Unexpected JSON structure."
        )

    print(
        f"Loaded {len(chunks)} chunks."
    )

    if chunks:

        print(
            "\nChunk fields:"
        )

        print(
            list(chunks[0].keys())
        )

    return chunks


# =============================================================================
# VALIDATE CHUNK
# =============================================================================

def validate_chunk(
    chunk: Dict[str, Any]
) -> None:
    """
    Validate that a chunk contains the fields required for inference.
    """

    required = [
        "input_ids",
        "attention_mask",
        "labels"
    ]

    missing = [
        key
        for key in required
        if key not in chunk
    ]

    if missing:

        raise ValueError(
            f"Chunk {chunk.get('chunk_id')} "
            f"is missing fields: {missing}"
        )

    input_ids = chunk["input_ids"]

    attention_mask = chunk["attention_mask"]

    labels = chunk["labels"]

    if not (
        len(input_ids)
        ==
        len(attention_mask)
        ==
        len(labels)
    ):

        raise ValueError(
            f"Chunk {chunk.get('chunk_id')} "
            f"has inconsistent lengths: "
            f"input_ids={len(input_ids)}, "
            f"attention_mask={len(attention_mask)}, "
            f"labels={len(labels)}"
        )


# =============================================================================
# PREDICT ONE CHUNK
# =============================================================================

def predict_chunk(
    chunk: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run BioBERT directly on the stored tokenized chunk.
    """

    validate_chunk(
        chunk
    )

    input_ids = torch.tensor(
        [chunk["input_ids"]],
        dtype=torch.long,
        device=DEVICE
    )

    attention_mask = torch.tensor(
        [chunk["attention_mask"]],
        dtype=torch.long,
        device=DEVICE
    )

    # Some BERT models use token_type_ids.
    # The stored dataset does not contain them, so we can safely create
    # zeros for this single-sequence NER task.
    token_type_ids = torch.zeros_like(
        input_ids
    )

    with torch.no_grad():

        outputs = MODEL(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

    logits = outputs.logits

    probabilities = torch.softmax(
        logits,
        dim=-1
    )

    predicted_ids = torch.argmax(
        probabilities,
        dim=-1
    )[0]

    confidences = torch.max(
        probabilities,
        dim=-1
    ).values[0]

    # -------------------------------------------------------------------------
    # Gold labels
    # -------------------------------------------------------------------------

    gold_labels = chunk["labels"]

    predictions = []

    valid_token_count = 0

    correct_count = 0

    gold_non_o = []

    predicted_non_o = []

    for index in range(
        len(chunk["input_ids"])
    ):

        # Attention mask = 0 means padding.
        if chunk["attention_mask"][index] == 0:

            continue

        gold_id = int(
            gold_labels[index]
        )

        predicted_id = int(
            predicted_ids[index].item()
        )

        confidence = float(
            confidences[index].item()
        )

        gold_label = ID_TO_LABEL.get(
            gold_id,
            f"UNKNOWN-{gold_id}"
        )

        predicted_label = ID_TO_LABEL.get(
            predicted_id,
            f"UNKNOWN-{predicted_id}"
        )

        valid_token_count += 1

        if gold_id == predicted_id:

            correct_count += 1

        if gold_label != "O":

            gold_non_o.append(
                {
                    "index": index,
                    "label_id": gold_id,
                    "label": gold_label
                }
            )

        if predicted_label != "O":

            predicted_non_o.append(
                {
                    "index": index,
                    "label_id": predicted_id,
                    "label": predicted_label,
                    "confidence": round(
                        confidence,
                        5
                    )
                }
            )

        predictions.append(
            {
                "index": index,
                "gold_id": gold_id,
                "gold_label": gold_label,
                "predicted_id": predicted_id,
                "predicted_label": predicted_label,
                "confidence": round(
                    confidence,
                    5
                )
            }
        )

    accuracy = (
        correct_count / valid_token_count
        if valid_token_count
        else 0.0
    )

    # -------------------------------------------------------------------------
    # Decode text for display only.
    # -------------------------------------------------------------------------

    decoded_text = TOKENIZER.decode(
        chunk["input_ids"],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    return {
        "chunk_id": chunk.get(
            "chunk_id"
        ),
        "document_id": chunk.get(
            "document_id"
        ),
        "decoded_text": decoded_text,
        "valid_tokens": valid_token_count,
        "correct_tokens": correct_count,
        "token_accuracy": accuracy,
        "gold_non_o_count": len(
            gold_non_o
        ),
        "predicted_non_o_count": len(
            predicted_non_o
        ),
        "gold_non_o": gold_non_o,
        "predicted_non_o": predicted_non_o,
        "predictions": predictions
    }


# =============================================================================
# DISPLAY EXAMPLE
# =============================================================================

def display_example(
    number: int,
    result: Dict[str, Any]
) -> None:

    print()
    print("=" * 79)

    print(
        f"REAL DATASET EXAMPLE {number}"
    )

    print("=" * 79)

    print(
        f"\nChunk ID: "
        f"{result['chunk_id']}"
    )

    print(
        f"Document ID: "
        f"{result['document_id']}"
    )

    print(
        f"\nDecoded text:\n"
        f"{result['decoded_text'][:1500]}"
    )

    print(
        f"\nToken accuracy: "
        f"{result['token_accuracy']:.4f}"
    )

    print(
        f"Gold non-O labels: "
        f"{result['gold_non_o_count']}"
    )

    print(
        f"Predicted non-O labels: "
        f"{result['predicted_non_o_count']}"
    )

    # -------------------------------------------------------------------------
    # Gold labels
    # -------------------------------------------------------------------------

    if result["gold_non_o"]:

        print(
            "\nGOLD ENTITY LABELS:"
        )

        for item in result["gold_non_o"][:30]:

            print(
                f"  token index "
                f"{item['index']:<4}"
                f" -> "
                f"{item['label']}"
            )

    else:

        print(
            "\nThis chunk has no gold entities."
        )

    # -------------------------------------------------------------------------
    # Predictions
    # -------------------------------------------------------------------------

    if result["predicted_non_o"]:

        print(
            "\nPREDICTED ENTITY LABELS:"
        )

        for item in result["predicted_non_o"][:30]:

            print(
                f"  token index "
                f"{item['index']:<4}"
                f" -> "
                f"{item['label']:<25}"
                f" confidence="
                f"{item['confidence']:.4f}"
            )

    else:

        print(
            "\nNO NON-O PREDICTIONS."
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print()
    print("=" * 79)

    print(
        "HEALTHAI - BIOBERT CHECKPOINT VALIDATION"
    )

    print("=" * 79)

    random.seed(
        RANDOM_SEED
    )

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    load_label_map()

    load_model()

    chunks = load_chunks()

    if not chunks:

        raise ValueError(
            "No chunks found."
        )

    # -------------------------------------------------------------------------
    # IMPORTANT:
    #
    # Select chunks that actually contain gold entities.
    #
    # This makes the diagnostic much more informative than randomly selecting
    # chunks that might legitimately contain only O labels.
    # -------------------------------------------------------------------------

    entity_chunks = []

    for chunk in chunks:

        labels = chunk.get(
            "labels",
            []
        )

        if any(
            int(label_id) != LABEL_MAP["O"]
            for label_id in labels
        ):

            entity_chunks.append(
                chunk
            )

    print()
    print(
        f"Chunks containing gold entities: "
        f"{len(entity_chunks)}"
    )

    if not entity_chunks:

        raise ValueError(
            "No chunks containing gold entities "
            "were found."
        )

    number_to_test = min(
        NUMBER_OF_EXAMPLES,
        len(entity_chunks)
    )

    selected = random.sample(
        entity_chunks,
        number_to_test
    )

    results = []

    total_valid_tokens = 0

    total_correct_tokens = 0

    total_gold_entities = 0

    total_predicted_entities = 0

    # -------------------------------------------------------------------------
    # Run validation
    # -------------------------------------------------------------------------

    for number, chunk in enumerate(
        selected,
        start=1
    ):

        result = predict_chunk(
            chunk
        )

        display_example(
            number,
            result
        )

        results.append(
            result
        )

        total_valid_tokens += (
            result["valid_tokens"]
        )

        total_correct_tokens += (
            result["correct_tokens"]
        )

        total_gold_entities += (
            result["gold_non_o_count"]
        )

        total_predicted_entities += (
            result["predicted_non_o_count"]
        )

    # -------------------------------------------------------------------------
    # Aggregate
    # -------------------------------------------------------------------------

    overall_accuracy = (
        total_correct_tokens
        /
        total_valid_tokens
        if total_valid_tokens
        else 0.0
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print()
    print("=" * 79)

    print(
        "CHECKPOINT VALIDATION SUMMARY"
    )

    print("=" * 79)

    print(
        f"Examples tested: "
        f"{len(selected)}"
    )

    print(
        f"Valid tokens: "
        f"{total_valid_tokens}"
    )

    print(
        f"Correct tokens: "
        f"{total_correct_tokens}"
    )

    print(
        f"Token accuracy: "
        f"{overall_accuracy:.4%}"
    )

    print(
        f"Gold non-O tokens: "
        f"{total_gold_entities}"
    )

    print(
        f"Predicted non-O tokens: "
        f"{total_predicted_entities}"
    )

    # -------------------------------------------------------------------------
    # Interpretation
    # -------------------------------------------------------------------------

    if total_predicted_entities > 0:

        print()
        print(
            "RESULT:"
        )

        print(
            "The local BioBERT checkpoint IS producing "
            "non-O predictions on real NER data."
        )

        print(
            "Therefore the checkpoint is capable of "
            "producing medical NER predictions."
        )

        print(
            "The earlier zero-entity manual test is "
            "not sufficient evidence that the model "
            "is broken."
        )

        status = (
            "checkpoint_producing_entities"
        )

    else:

        print()
        print(
            "RESULT:"
        )

        print(
            "The local BioBERT checkpoint produced "
            "zero non-O predictions even on chunks "
            "that contain gold entities."
        )

        print(
            "This strongly suggests a checkpoint/"
            "training-weight/configuration issue."
        )

        status = (
            "checkpoint_requires_investigation"
        )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {
        "step": "17.4B",
        "component": (
            "biobert_checkpoint_validation"
        ),
        "model": (
            "dmis-lab/biobert-v1.1"
        ),
        "model_path": str(
            MODEL_PATH
        ),
        "device": str(
            DEVICE
        ),
        "label_count": len(
            LABEL_MAP
        ),
        "examples_tested": len(
            selected
        ),
        "entity_chunks_available": len(
            entity_chunks
        ),
        "total_valid_tokens": (
            total_valid_tokens
        ),
        "total_correct_tokens": (
            total_correct_tokens
        ),
        "overall_token_accuracy": (
            overall_accuracy
        ),
        "total_gold_non_o_tokens": (
            total_gold_entities
        ),
        "total_predicted_non_o_tokens": (
            total_predicted_entities
        ),
        "diagnostic_status": status,
        "examples": results
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(
        f"Validation results saved to:\n"
        f"{OUTPUT_PATH}"
    )

    print()
    print(
        "STEP 17.4B COMPLETED"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main() q