"""
===============================================================================
HEALTHAI - REAL BIOBERT MEDICAL NER ADAPTER
===============================================================================

Module:
    17_agentic_ai

Step:
    17.4 - Real BioBERT Medical NER Adapter

Purpose:
    Connect the trained BioBERT medical NER model to the Agentic AI layer.

Flow:
    User Query
        ↓
    Agent Router
        ↓
    Agent Executor
        ↓
    Medical NER Adapter
        ↓
    Trained BioBERT
        ↓
    Medical Entities
        ↓
    Standardized Agent Response

Model:
    dmis-lab/biobert-v1.1

Local model:
    models/nlp/biobert_ner/

Label map:
    data/processed/deep_learning/nlp/ner_label_map.json

===============================================================================
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)


# =============================================================================
# PROJECT PATHS
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
    / "medical_ner_adapter_test.json"
)


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MAX_LENGTH = 512

# BioBERT uses special tokens such as [CLS] and [SEP].
# Keep content below 512.
MAX_CONTENT_TOKENS = 510

# Overlap helps avoid losing entities at chunk boundaries.
CHUNK_OVERLAP = 64


# =============================================================================
# GLOBAL MODEL OBJECTS
# =============================================================================

TOKENIZER = None
MODEL = None

LABEL_MAP = {}
ID_TO_LABEL = {}

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def print_separator():
    print("=" * 79)


def print_header(title: str):
    print()
    print_separator()
    print(title)
    print_separator()


# =============================================================================
# LOAD LABEL MAP
# =============================================================================

def load_label_map() -> None:
    """
    Load the NER BIO label map.

    The actual project label-map structure is expected to contain:

        {
            "label2id": {
                "O": 0,
                "B-Comorbidity": 1,
                ...
            },
            "id2label": {
                "0": "O",
                "1": "B-Comorbidity",
                ...
            }
        }

    The function also supports a direct label -> ID mapping.
    """

    global LABEL_MAP
    global ID_TO_LABEL

    print("Loading NER label map...")

    # -------------------------------------------------------------------------
    # Validate path
    # -------------------------------------------------------------------------

    if not LABEL_MAP_PATH.exists():

        raise FileNotFoundError(
            f"\nNER label map not found:\n"
            f"{LABEL_MAP_PATH}"
        )

    # -------------------------------------------------------------------------
    # Read JSON
    # -------------------------------------------------------------------------

    with open(
        LABEL_MAP_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        raw_map = json.load(file)

    if not isinstance(raw_map, dict):

        raise ValueError(
            "NER label map must be a JSON object."
        )

    # -------------------------------------------------------------------------
    # CASE 1:
    #
    # {
    #     "label2id": {...},
    #     "id2label": {...}
    # }
    # -------------------------------------------------------------------------

    if "label2id" in raw_map:

        label2id = raw_map["label2id"]

        if not isinstance(label2id, dict):

            raise ValueError(
                "'label2id' must be a dictionary."
            )

        LABEL_MAP = {}

        for label, label_id in label2id.items():

            try:

                LABEL_MAP[str(label)] = int(label_id)

            except (
                TypeError,
                ValueError
            ) as error:

                raise ValueError(
                    f"Invalid label ID for "
                    f"{label}: {label_id}"
                ) from error

    # -------------------------------------------------------------------------
    # CASE 2:
    #
    # Direct:
    #
    # {
    #     "O": 0,
    #     "B-Condition": 1,
    #     ...
    # }
    # -------------------------------------------------------------------------

    else:

        direct_mapping = True

        for value in raw_map.values():

            if isinstance(value, dict):

                direct_mapping = False
                break

            if isinstance(value, bool):

                direct_mapping = False
                break

            try:

                int(value)

            except (
                TypeError,
                ValueError
            ):

                direct_mapping = False
                break

        if direct_mapping:

            LABEL_MAP = {
                str(label): int(label_id)
                for label, label_id
                in raw_map.items()
            }

        # ---------------------------------------------------------------------
        # CASE 3:
        #
        # {
        #     "0": {"label": "O"},
        #     "1": {"label": "B-Condition"}
        # }
        # ---------------------------------------------------------------------

        else:

            LABEL_MAP = {}

            for key, value in raw_map.items():

                if not isinstance(value, dict):

                    raise ValueError(
                        f"Invalid label-map entry: "
                        f"{key}: {value}"
                    )

                try:

                    label_id = int(key)

                except (
                    TypeError,
                    ValueError
                ):

                    label_id = (
                        value.get("id")
                        or value.get("label_id")
                    )

                label_name = (
                    value.get("label")
                    or value.get("name")
                    or value.get("label_name")
                )

                if label_id is None:

                    raise ValueError(
                        f"Could not determine label ID "
                        f"for: {key}: {value}"
                    )

                if label_name is None:

                    raise ValueError(
                        f"Could not determine label name "
                        f"for: {key}: {value}"
                    )

                LABEL_MAP[str(label_name)] = int(
                    label_id
                )

    # -------------------------------------------------------------------------
    # Reverse mapping
    # -------------------------------------------------------------------------

    ID_TO_LABEL = {
        int(label_id): label
        for label, label_id
        in LABEL_MAP.items()
    }

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    print(
        f"Loaded {len(LABEL_MAP)} BIO labels."
    )

    if "O" not in LABEL_MAP:

        raise ValueError(
            "Required 'O' label is missing."
        )

    if len(LABEL_MAP) == 39:

        print(
            "Label-map validation passed: 39 labels."
        )

    else:

        print(
            f"WARNING: Expected 39 labels, "
            f"found {len(LABEL_MAP)}."
        )

    # -------------------------------------------------------------------------
    # Display labels
    # -------------------------------------------------------------------------

    print("\nLabel mapping:")

    for label_id in sorted(ID_TO_LABEL):

        print(
            f"  {label_id:2d} -> "
            f"{ID_TO_LABEL[label_id]}"
        )

    print(
        "\nNER label map loaded successfully."
    )


# =============================================================================
# LOAD BIOBERT
# =============================================================================

def load_model() -> None:
    """
    Load the trained BioBERT token classification model.
    """

    global TOKENIZER
    global MODEL

    print_header(
        "LOADING TRAINED BIOBERT NER MODEL"
    )

    # -------------------------------------------------------------------------
    # Validate model directory
    # -------------------------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\nBioBERT model directory not found:\n"
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

    # -------------------------------------------------------------------------
    # Tokenizer
    # -------------------------------------------------------------------------

    print("\nLoading tokenizer...")

    TOKENIZER = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        use_fast=True
    )

    print(
        "Tokenizer loaded."
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    print("\nLoading BioBERT model...")

    MODEL = AutoModelForTokenClassification.from_pretrained(
        str(MODEL_PATH)
    )

    MODEL.to(DEVICE)

    MODEL.eval()

    # -------------------------------------------------------------------------
    # Model information
    # -------------------------------------------------------------------------

    parameter_count = sum(
        parameter.numel()
        for parameter in MODEL.parameters()
    )

    print(
        "BioBERT model loaded."
    )

    print(
        f"Parameters: {parameter_count:,}"
    )

    print(
        f"Number of labels: "
        f"{MODEL.config.num_labels}"
    )


# =============================================================================
# BIO LABEL HELPERS
# =============================================================================

def get_entity_type(label: str) -> str:
    """
    Convert:

        B-Condition -> Condition
        I-Condition -> Condition

    O -> empty string
    """

    if not label:
        return ""

    if label == "O":
        return ""

    if label.startswith("B-"):
        return label[2:]

    if label.startswith("I-"):
        return label[2:]

    return label


# =============================================================================
# TEXT CLEANING
# =============================================================================

def clean_entity_text(text: str) -> str:
    """
    Clean whitespace and tokenizer artifacts.
    """

    if not text:
        return ""

    # Remove WordPiece marker if present.
    text = text.replace(
        "##",
        ""
    )

    # Normalize whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # Remove spaces before punctuation.
    text = re.sub(
        r"\s+([,.!?;:%])",
        r"\1",
        text
    )

    return text.strip()


# =============================================================================
# TOKEN-BASED CHUNKING
# =============================================================================

def create_token_chunks(
    text: str
) -> List[Dict[str, Any]]:
    """
    Split text into overlapping BioBERT-compatible token chunks.

    Character offsets are preserved so that final entities can be extracted
    from the original input text rather than reconstructed from WordPieces.
    """

    if not text.strip():

        return []

    encoded = TOKENIZER(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False
    )

    input_ids = encoded["input_ids"]

    offsets = encoded["offset_mapping"]

    total_tokens = len(input_ids)

    chunks = []

    start = 0

    while start < total_tokens:

        end = min(
            start + MAX_CONTENT_TOKENS,
            total_tokens
        )

        chunk_ids = input_ids[
            start:end
        ]

        chunk_offsets = offsets[
            start:end
        ]

        if not chunk_offsets:
            break

        char_start = chunk_offsets[0][0]

        char_end = chunk_offsets[-1][1]

        chunk_text = text[
            char_start:char_end
        ]

        chunks.append(
            {
                "chunk_index": len(chunks),
                "input_ids": chunk_ids,
                "offset_mapping": chunk_offsets,
                "char_start": char_start,
                "char_end": char_end,
                "text": chunk_text
            }
        )

        # Finished.
        if end >= total_tokens:
            break

        # Overlapping next chunk.
        next_start = (
            end - CHUNK_OVERLAP
        )

        if next_start <= start:

            next_start = end

        start = next_start

    return chunks


# =============================================================================
# PREDICT ONE CHUNK
# =============================================================================

def predict_chunk(
    original_text: str,
    chunk: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Run BioBERT inference on one token chunk.
    """

    input_ids = chunk["input_ids"]

    offset_mapping = chunk[
        "offset_mapping"
    ]

    # -------------------------------------------------------------------------
    # Prepare model input
    # -------------------------------------------------------------------------

    prepared = TOKENIZER.prepare_for_model(
        input_ids,
        add_special_tokens=True,
        return_attention_mask=True,
        return_token_type_ids=True
    )

    model_inputs = {}

    for key in [
        "input_ids",
        "attention_mask",
        "token_type_ids"
    ]:

        if key in prepared:

            model_inputs[key] = torch.tensor(
                [prepared[key]],
                dtype=torch.long,
                device=DEVICE
            )

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

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

    confidence_values = torch.max(
        probabilities,
        dim=-1
    ).values[0]

    # -------------------------------------------------------------------------
    # Iterate through prepared tokens.
    #
    # prepared contains special tokens.
    # We therefore track content-token position separately.
    # -------------------------------------------------------------------------

    entities = []

    current_entity = None

    content_index = 0

    prepared_ids = prepared[
        "input_ids"
    ]

    for model_index, token_id in enumerate(
        prepared_ids
    ):

        token = TOKENIZER.convert_ids_to_tokens(
            token_id
        )

        # -------------------------------------------------------------
        # Ignore special tokens.
        # -------------------------------------------------------------

        if token in TOKENIZER.all_special_tokens:

            continue

        if content_index >= len(
            offset_mapping
        ):

            break

        local_start, local_end = (
            offset_mapping[content_index]
        )

        content_index += 1

        if local_end <= local_start:

            continue

        # -------------------------------------------------------------
        # Convert chunk-relative offsets to
        # original-document offsets.
        # -------------------------------------------------------------

        global_start = (
            chunk["char_start"]
            + local_start
        )

        global_end = (
            chunk["char_start"]
            + local_end
        )

        # -------------------------------------------------------------
        # Predicted label
        # -------------------------------------------------------------

        label_id = int(
            predicted_ids[
                model_index
            ].item()
        )

        label = ID_TO_LABEL.get(
            label_id,
            "O"
        )

        confidence = float(
            confidence_values[
                model_index
            ].item()
        )

        entity_type = get_entity_type(
            label
        )

        # -------------------------------------------------------------
        # Outside
        # -------------------------------------------------------------

        if label == "O" or not entity_type:

            if current_entity is not None:

                entities.append(
                    current_entity
                )

                current_entity = None

            continue

        # -------------------------------------------------------------
        # Beginning
        # -------------------------------------------------------------

        if label.startswith("B-"):

            if current_entity is not None:

                entities.append(
                    current_entity
                )

            current_entity = {
                "entity_type": entity_type,
                "start": global_start,
                "end": global_end,
                "confidence": confidence
            }

            continue

        # -------------------------------------------------------------
        # Inside
        # -------------------------------------------------------------

        if label.startswith("I-"):

            if (
                current_entity is not None
                and
                current_entity["entity_type"]
                == entity_type
            ):

                current_entity["end"] = (
                    global_end
                )

                # Running average confidence.
                current_entity[
                    "confidence"
                ] = (
                    current_entity["confidence"]
                    + confidence
                ) / 2.0

            else:

                # Recover malformed I- sequence.
                current_entity = {
                    "entity_type": entity_type,
                    "start": global_start,
                    "end": global_end,
                    "confidence": confidence
                }

    # -------------------------------------------------------------------------
    # Flush final entity
    # -------------------------------------------------------------------------

    if current_entity is not None:

        entities.append(
            current_entity
        )

    # -------------------------------------------------------------------------
    # Convert spans into clean entity objects.
    # -------------------------------------------------------------------------

    results = []

    for entity in entities:

        start = entity["start"]

        end = entity["end"]

        entity_text = original_text[
            start:end
        ]

        entity_text = clean_entity_text(
            entity_text
        )

        if not entity_text:
            continue

        results.append(
            {
                "entity": entity_text,
                "entity_type": entity[
                    "entity_type"
                ],
                "confidence": round(
                    entity["confidence"],
                    4
                ),
                "start": start,
                "end": end
            }
        )

    return results


# =============================================================================
# MERGE CHUNK PREDICTIONS
# =============================================================================

def merge_entity_predictions(
    entities: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Remove duplicate/overlapping entities created by chunk overlap.
    """

    if not entities:

        return []

    entities = sorted(
        entities,
        key=lambda item: (
            item["start"],
            item["end"],
            item["entity_type"]
        )
    )

    merged = []

    for entity in entities:

        found_duplicate = False

        for existing in merged:

            same_type = (
                entity["entity_type"]
                == existing["entity_type"]
            )

            same_span = (
                entity["start"]
                == existing["start"]
                and
                entity["end"]
                == existing["end"]
            )

            overlapping = (
                entity["start"]
                < existing["end"]
                and
                entity["end"]
                > existing["start"]
            )

            if same_type and (
                same_span
                or overlapping
            ):

                found_duplicate = True

                # Keep higher confidence.
                if (
                    entity["confidence"]
                    >
                    existing["confidence"]
                ):

                    existing.update(
                        entity
                    )

                break

        if not found_duplicate:

            merged.append(
                entity
            )

    return merged


# =============================================================================
# MAIN MEDICAL NER FUNCTION
# =============================================================================

def predict_medical_entities(
    text: str
) -> Dict[str, Any]:
    """
    Extract medical entities from input text.

    Returns a standardized Agentic AI response.
    """

    # -------------------------------------------------------------------------
    # Input validation
    # -------------------------------------------------------------------------

    if not isinstance(text, str):

        return {
            "status": "error",
            "tool": "medical_ner",
            "message": (
                "Input text must be a string."
            ),
            "entities": []
        }

    text = text.strip()

    if not text:

        return {
            "status": "error",
            "tool": "medical_ner",
            "message": (
                "Input text is empty."
            ),
            "entities": []
        }

    if TOKENIZER is None or MODEL is None:

        raise RuntimeError(
            "BioBERT model has not been loaded."
        )

    # -------------------------------------------------------------------------
    # Chunk input
    # -------------------------------------------------------------------------

    chunks = create_token_chunks(
        text
    )

    # -------------------------------------------------------------------------
    # Predict
    # -------------------------------------------------------------------------

    all_entities = []

    for chunk in chunks:

        chunk_entities = predict_chunk(
            original_text=text,
            chunk=chunk
        )

        all_entities.extend(
            chunk_entities
        )

    # -------------------------------------------------------------------------
    # Merge overlap duplicates.
    # -------------------------------------------------------------------------

    all_entities = merge_entity_predictions(
        all_entities
    )

    # -------------------------------------------------------------------------
    # Exact duplicate protection.
    # -------------------------------------------------------------------------

    unique_entities = []

    seen = set()

    for entity in all_entities:

        key = (
            entity["entity"].lower(),
            entity["entity_type"],
            entity["start"],
            entity["end"]
        )

        if key in seen:

            continue

        seen.add(key)

        unique_entities.append(
            entity
        )

    # -------------------------------------------------------------------------
    # Standard Agentic AI response
    # -------------------------------------------------------------------------

    response = {
        "status": "success",
        "tool": "medical_ner",
        "model": "dmis-lab/biobert-v1.1",
        "device": str(DEVICE),
        "input_text": text,
        "chunks_processed": len(chunks),
        "entity_count": len(
            unique_entities
        ),
        "entities": unique_entities
    }

    return response


# =============================================================================
# DISPLAY RESULT
# =============================================================================

def display_result(
    test_number: int,
    text: str,
    result: Dict[str, Any]
) -> None:
    """
    Print human-readable NER result.
    """

    print_header(
        f"TEST CASE {test_number}"
    )

    print(
        f"\nInput:\n{text}"
    )

    print(
        f"\nStatus: "
        f"{result.get('status')}"
    )

    print(
        f"Chunks processed: "
        f"{result.get('chunks_processed', 0)}"
    )

    print(
        f"Entities found: "
        f"{result.get('entity_count', 0)}"
    )

    entities = result.get(
        "entities",
        []
    )

    if not entities:

        print(
            "\nNo medical entities detected."
        )

        return

    print(
        "\nExtracted entities:"
    )

    for index, entity in enumerate(
        entities,
        start=1
    ):

        print(
            f"  {index}. "
            f"{entity['entity']!r} "
            f"-> "
            f"{entity['entity_type']} "
            f"(confidence="
            f"{entity['confidence']:.4f})"
        )


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(
    results: List[Dict[str, Any]]
) -> None:
    """
    Save adapter test results.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {
        "module": "17_agentic_ai",
        "step": "17.4",
        "component": (
            "real_biobert_medical_ner_adapter"
        ),
        "model": (
            "dmis-lab/biobert-v1.1"
        ),
        "model_path": str(
            MODEL_PATH
        ),
        "label_count": len(
            LABEL_MAP
        ),
        "device": str(
            DEVICE
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

    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_PATH}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """
    Execute Step 17.4.
    """

    print()
    print_separator()

    print(
        "HEALTHAI - REAL BIOBERT MEDICAL NER ADAPTER"
    )

    print_separator()

    # -------------------------------------------------------------------------
    # Load label map
    # -------------------------------------------------------------------------

    load_label_map()

    # -------------------------------------------------------------------------
    # Load BioBERT
    # -------------------------------------------------------------------------

    load_model()

    # -------------------------------------------------------------------------
    # Test inputs
    # -------------------------------------------------------------------------

    test_cases = [

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

    results = []

    # -------------------------------------------------------------------------
    # Run inference
    # -------------------------------------------------------------------------

    for test_number, text in enumerate(
        test_cases,
        start=1
    ):

        try:

            result = predict_medical_entities(
                text
            )

            display_result(
                test_number,
                text,
                result
            )

            results.append(
                {
                    "test_number": test_number,
                    "input": text,
                    "result": result
                }
            )

        except Exception as error:

            print()
            print(
                f"ERROR IN TEST CASE "
                f"{test_number}"
            )

            print(
                str(error)
            )

            results.append(
                {
                    "test_number": test_number,
                    "input": text,
                    "result": {
                        "status": "error",
                        "message": str(error),
                        "entities": []
                    }
                }
            )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    save_results(
        results
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    successful = sum(
        1
        for item in results
        if item["result"].get(
            "status"
        ) == "success"
    )

    failed = (
        len(results)
        - successful
    )

    print_header(
        "STEP 17.4 SUMMARY"
    )

    print(
        f"Test cases: {len(results)}"
    )

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed: {failed}"
    )

    print(
        f"BIO labels: "
        f"{len(LABEL_MAP)}"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Output:\n{OUTPUT_PATH}"
    )

    if failed == 0:

        print(
            "\nSTEP 17.4 COMPLETED SUCCESSFULLY"
        )

    else:

        print(
            "\nSTEP 17.4 COMPLETED WITH ERRORS"
        )


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()