"""
===============================================================================
HEALTHAI - BIOBERT MEDICAL NER RAW INFERENCE DIAGNOSTIC
===============================================================================

Module:
    17_agentic_ai

Step:
    17.4A - Raw BioBERT NER Diagnostic

Purpose:
    Verify what the trained BioBERT model is actually predicting at token level
    before connecting it to the Agentic AI executor.

This diagnostic intentionally does NOT perform BIO entity reconstruction.

It prints:

    token
    predicted label
    confidence

This helps distinguish:

    Model prediction problem
            vs.
    Adapter/entity reconstruction problem

===============================================================================
"""

import json
from pathlib import Path

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "agentic_ai"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "biobert_raw_inference_diagnostic.json"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

TEST_TEXTS = [
    (
        "Patient has diabetes and hypertension. "
        "The patient was prescribed metformin."
    ),
    (
        "Patient reports fever and headache. "
        "Blood glucose was measured and an MRI test "
        "was recommended."
    ),
    (
        "The patient received insulin treatment "
        "for elevated blood glucose."
    )
]


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

def load_label_map():
    """Load the project's actual label2id mapping."""

    global LABEL_MAP
    global ID_TO_LABEL

    print("Loading label map...")

    if not LABEL_MAP_PATH.exists():

        raise FileNotFoundError(
            f"Label map not found:\n{LABEL_MAP_PATH}"
        )

    with open(
        LABEL_MAP_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        raw_map = json.load(file)

    # Actual project format:
    #
    # {
    #     "label2id": {...},
    #     "id2label": {...}
    # }

    if "label2id" in raw_map:

        LABEL_MAP = {
            str(label): int(label_id)
            for label, label_id
            in raw_map["label2id"].items()
        }

    else:

        # Fallback for direct mapping.
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


# =============================================================================
# LOAD MODEL
# =============================================================================

def load_model():
    """Load tokenizer and trained BioBERT model."""

    global TOKENIZER
    global MODEL

    print()
    print("=" * 79)
    print("LOADING BIOBERT")
    print("=" * 79)

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"BioBERT model directory not found:\n"
            f"{MODEL_PATH}"
        )

    print(
        f"Model path:\n{MODEL_PATH}"
    )

    print(
        f"\nDevice: {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print(
        "\nLoading tokenizer..."
    )

    TOKENIZER = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        use_fast=True
    )

    print(
        "Tokenizer loaded."
    )

    print(
        "\nLoading model..."
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
        "BioBERT loaded."
    )

    print(
        f"Parameters: {parameter_count:,}"
    )

    print(
        f"Model labels: "
        f"{MODEL.config.num_labels}"
    )


# =============================================================================
# RAW TOKEN PREDICTION
# =============================================================================

def predict_tokens(text):
    """
    Run raw token classification.

    No BIO merging.
    No entity reconstruction.
    No chunking.

    This is deliberately simple.
    """

    encoded = TOKENIZER(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=512
    )

    # Keep offsets on CPU.
    offset_mapping = encoded.pop(
        "offset_mapping"
    )[0]

    # token_type_ids may or may not exist.
    model_inputs = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    with torch.no_grad():

        outputs = MODEL(
            **model_inputs
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

    tokens = TOKENIZER.convert_ids_to_tokens(
        encoded["input_ids"][0]
    )

    results = []

    for index, token in enumerate(tokens):

        token_id = int(
            predicted_ids[index].item()
        )

        label = ID_TO_LABEL.get(
            token_id,
            f"UNKNOWN-{token_id}"
        )

        confidence = float(
            confidences[index].item()
        )

        start = int(
            offset_mapping[index][0].item()
        )

        end = int(
            offset_mapping[index][1].item()
        )

        # Ignore special tokens for the primary report.
        is_special = (
            token in TOKENIZER.all_special_tokens
        )

        results.append(
            {
                "index": index,
                "token": token,
                "label_id": token_id,
                "label": label,
                "confidence": round(
                    confidence,
                    6
                ),
                "start": start,
                "end": end,
                "is_special": is_special
            }
        )

    return results


# =============================================================================
# DISPLAY TOKEN PREDICTIONS
# =============================================================================

def display_predictions(
    test_number,
    text,
    predictions
):
    """Print all non-special token predictions."""

    print()
    print("=" * 79)
    print(
        f"TEST CASE {test_number}"
    )
    print("=" * 79)

    print(
        f"\nInput:\n{text}"
    )

    print()
    print(
        f"{'TOKEN':<25}"
        f"{'LABEL':<30}"
        f"{'CONFIDENCE':>12}"
    )

    print("-" * 69)

    non_o_predictions = []

    for item in predictions:

        if item["is_special"]:
            continue

        token = item["token"]

        label = item["label"]

        confidence = item["confidence"]

        print(
            f"{token:<25}"
            f"{label:<30}"
            f"{confidence:>12.4f}"
        )

        if label != "O":

            non_o_predictions.append(
                item
            )

    print()

    print(
        f"Total tokens: "
        f"{sum(not x['is_special'] for x in predictions)}"
    )

    print(
        f"Non-O predictions: "
        f"{len(non_o_predictions)}"
    )

    if non_o_predictions:

        print(
            "\nNON-O PREDICTIONS FOUND:"
        )

        for item in non_o_predictions:

            print(
                f"  {item['token']!r} "
                f"-> "
                f"{item['label']} "
                f"("
                f"{item['confidence']:.4f}"
                f")"
            )

    else:

        print(
            "\nWARNING: MODEL PREDICTED ONLY 'O' "
            "FOR THIS TEST CASE."
        )


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(results):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {
        "step": "17.4A",
        "component": (
            "biobert_raw_inference_diagnostic"
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
        "test_cases": results
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
        f"Diagnostic results saved to:\n"
        f"{OUTPUT_PATH}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print()
    print("=" * 79)
    print(
        "HEALTHAI - BIOBERT RAW NER DIAGNOSTIC"
    )
    print("=" * 79)

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    load_label_map()

    load_model()

    # -------------------------------------------------------------------------
    # Run diagnostics
    # -------------------------------------------------------------------------

    all_results = []

    for test_number, text in enumerate(
        TEST_TEXTS,
        start=1
    ):

        predictions = predict_tokens(
            text
        )

        display_predictions(
            test_number,
            text,
            predictions
        )

        non_special = [
            item
            for item in predictions
            if not item["is_special"]
        ]

        non_o = [
            item
            for item in non_special
            if item["label"] != "O"
        ]

        all_results.append(
            {
                "test_number": test_number,
                "input": text,
                "total_tokens": len(
                    non_special
                ),
                "non_o_predictions": len(
                    non_o
                ),
                "predictions": predictions
            }
        )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    save_results(
        all_results
    )

    # -------------------------------------------------------------------------
    # Overall diagnostic
    # -------------------------------------------------------------------------

    total_tokens = sum(
        item["total_tokens"]
        for item in all_results
    )

    total_non_o = sum(
        item["non_o_predictions"]
        for item in all_results
    )

    print()
    print("=" * 79)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 79)

    print(
        f"Total test tokens: "
        f"{total_tokens}"
    )

    print(
        f"Total non-O predictions: "
        f"{total_non_o}"
    )

    if total_non_o > 0:

        print(
            "\nRESULT:"
        )

        print(
            "BioBERT is producing entity predictions."
        )

        print(
            "The problem is likely in the adapter's "
            "BIO/entity reconstruction."
        )

    else:

        print(
            "\nRESULT:"
        )

        print(
            "BioBERT predicted O for every token."
        )

        print(
            "We need to investigate the local "
            "checkpoint/configuration before "
            "changing the adapter."
        )

    print(
        "\nSTEP 17.4A DIAGNOSTIC COMPLETED"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()