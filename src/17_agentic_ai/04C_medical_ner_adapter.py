"""
HealthAI Agentic AI
Medical NER Adapter - BioBERT

Purpose:
    Production-style adapter around the locally trained BioBERT
    medical NER model.

Agent interface:
    execute_medical_ner(query=...)

Direct inference:
    predict_medical_entities(text, tokenizer, model)

Model:
    dmis-lab/biobert-v1.1
    Fine-tuned for 39 BIO labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = (
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

OUTPUT_PATH = OUTPUT_DIR / "medical_ner_adapter_test.json"


# ============================================================
# MODEL SETTINGS
# ============================================================

MAX_LENGTH = 512

# Minimum probability required for an entity token.
MIN_CONFIDENCE = 0.25

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LAZY MODEL CACHE
# ============================================================

_TOKENIZER = None
_MODEL = None
_LABEL_MAP = None


# ============================================================
# LABEL MAP
# ============================================================

def load_label_map() -> dict[str, int]:
    """
    Load the project BIO label map.
    """

    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError(
            f"NER label map not found:\n{LABEL_MAP_PATH}"
        )

    with open(
        LABEL_MAP_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    if "label2id" in data:

        label2id = {
            str(label): int(idx)
            for label, idx in data["label2id"].items()
        }

    else:

        label2id = {
            str(label): int(idx)
            for label, idx in data.items()
            if isinstance(idx, (int, float))
        }

    if not label2id:
        raise ValueError(
            "NER label map is empty or invalid."
        )

    return label2id


# ============================================================
# MODEL LOADING
# ============================================================

def load_model_and_tokenizer():
    """
    Load the locally fine-tuned BioBERT model and tokenizer.
    """

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"BioBERT model directory not found:\n{MODEL_DIR}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        use_fast=True
    )

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_DIR
    )

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


def get_model_and_tokenizer():
    """
    Lazy-loading wrapper.

    The model is loaded only on the first agent request and then
    reused for subsequent requests.
    """

    global _TOKENIZER
    global _MODEL
    global _LABEL_MAP

    if _TOKENIZER is None or _MODEL is None:

        _TOKENIZER, _MODEL = load_model_and_tokenizer()

    if _LABEL_MAP is None:

        _LABEL_MAP = load_label_map()

    return _TOKENIZER, _MODEL


# ============================================================
# LABEL HELPERS
# ============================================================

def get_entity_type(
    label: str
) -> str | None:

    if label == "O":
        return None

    if label.startswith("B-"):
        return label[2:]

    if label.startswith("I-"):
        return label[2:]

    return label


# ============================================================
# ENTITY CLEANING
# ============================================================

def clean_entity_text(
    entity_text: str
) -> str:

    return entity_text.strip(
        " ,.;:()[]{}<>/\\|"
    )


def is_valid_entity_text(
    text: str
) -> bool:

    if not text:
        return False

    if not any(
        char.isalnum()
        for char in text
    ):
        return False

    if len(text) == 1:
        return False

    if len(text) <= 3 and " " not in text:

        if not text.isupper():
            return False

    return True


# ============================================================
# SPAN BUILDING
# ============================================================

def build_entity_spans(
    text: str,
    predicted_labels: list[str],
    token_confidences: list[float],
    offsets: list[tuple[int, int]]
) -> list[dict[str, Any]]:

    entities: list[dict[str, Any]] = []

    current_entity: dict[str, Any] | None = None

    def flush_current_entity():

        nonlocal current_entity

        if current_entity is None:
            return

        start = current_entity["start"]
        end = current_entity["end"]

        entity_text = text[start:end]

        entity_text = clean_entity_text(
            entity_text
        )

        if not is_valid_entity_text(
            entity_text
        ):
            current_entity = None
            return

        entities.append(
            {
                "text": entity_text,
                "label": current_entity["label"],
                "confidence": round(
                    float(
                        current_entity["confidence"]
                    ),
                    4
                ),
                "start": int(start),
                "end": int(end)
            }
        )

        current_entity = None

    for label, confidence, offset in zip(
        predicted_labels,
        token_confidences,
        offsets
    ):

        start, end = offset

        # Ignore special tokens.
        if start == end:
            continue

        # Ignore low-confidence tokens.
        if confidence < MIN_CONFIDENCE:

            if current_entity is not None:
                flush_current_entity()

            continue

        if label == "O":

            flush_current_entity()
            continue

        entity_type = get_entity_type(
            label
        )

        if entity_type is None:

            flush_current_entity()
            continue

        is_beginning = label.startswith(
            "B-"
        )

        if is_beginning:

            flush_current_entity()

            current_entity = {
                "label": entity_type,
                "start": start,
                "end": end,
                "confidence": confidence
            }

        elif label.startswith("I-"):

            if (
                current_entity is not None
                and current_entity["label"]
                == entity_type
            ):

                current_entity["end"] = end

                previous_confidence = float(
                    current_entity["confidence"]
                )

                current_entity["confidence"] = (
                    previous_confidence
                    + confidence
                ) / 2.0

            else:

                # Defensive handling of an I- tag
                # without a matching B- tag.

                flush_current_entity()

                current_entity = {
                    "label": entity_type,
                    "start": start,
                    "end": end,
                    "confidence": confidence
                }

        else:

            flush_current_entity()

    flush_current_entity()

    return entities


# ============================================================
# CORE MEDICAL NER PREDICTION
# ============================================================

def predict_medical_entities(
    text: str,
    tokenizer,
    model
) -> dict[str, Any]:

    if not isinstance(text, str):
        raise TypeError(
            "Input text must be a string."
        )

    text = text.strip()

    if not text:

        return {
            "tool": "medical_ner",
            "status": "invalid_input",
            "input_text": text,
            "entity_count": 0,
            "entities": []
        }

    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    encoded = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False
    )

    offsets = (
        encoded
        .pop("offset_mapping")[0]
        .tolist()
    )

    model_inputs = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
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

    predicted_confidences = torch.max(
        probabilities,
        dim=-1
    ).values[0]

    # --------------------------------------------------------
    # Label mapping
    # --------------------------------------------------------

    id2label = model.config.id2label

    predicted_labels: list[str] = []
    token_confidences: list[float] = []

    for label_id, confidence in zip(
        predicted_ids.cpu().tolist(),
        predicted_confidences.cpu().tolist()
    ):

        label = id2label.get(
            int(label_id),
            "O"
        )

        predicted_labels.append(
            label
        )

        token_confidences.append(
            float(confidence)
        )

    # --------------------------------------------------------
    # Build entity spans
    # --------------------------------------------------------

    entities = build_entity_spans(
        text=text,
        predicted_labels=predicted_labels,
        token_confidences=token_confidences,
        offsets=offsets
    )

    return {
        "tool": "medical_ner",
        "status": "success",
        "input_text": text,
        "entity_count": len(entities),
        "entities": entities,
        "metadata": {
            "model": "dmis-lab/biobert-v1.1",
            "model_directory": str(
                MODEL_DIR
            ),
            "device": str(DEVICE),
            "max_length": MAX_LENGTH,
            "min_confidence": MIN_CONFIDENCE
        }
    }


# ============================================================
# AGENT TOOL INTERFACE
# ============================================================

def execute_medical_ner(
    query: str,
    **kwargs
) -> dict[str, Any]:
    """
    Standard HealthAI Agentic AI tool interface.

    The executor calls this function using:

        execute_medical_ner(query=query)

    This function loads BioBERT lazily and delegates actual
    prediction to predict_medical_entities().
    """

    try:

        if not isinstance(query, str):

            return {
                "tool": "medical_ner",
                "status": "invalid_input",
                "message": (
                    "Medical NER input must be text."
                ),
                "entity_count": 0,
                "entities": []
            }

        query = query.strip()

        if not query:

            return {
                "tool": "medical_ner",
                "status": "invalid_input",
                "message": (
                    "Medical NER requires "
                    "clinical text."
                ),
                "entity_count": 0,
                "entities": []
            }

        tokenizer, model = (
            get_model_and_tokenizer()
        )

        result = predict_medical_entities(
            text=query,
            tokenizer=tokenizer,
            model=model
        )

        # Explicit agent metadata.
        result["agent_interface"] = (
            "execute_medical_ner"
        )

        return result

    except Exception as exc:

        return {
            "tool": "medical_ner",
            "status": "execution_error",
            "message": (
                "Medical NER execution failed."
            ),
            "error": str(exc),
            "entity_count": 0,
            "entities": []
        }


# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [

    (
        "The patient has idiopathic "
        "Parkinson's disease with "
        "hypertension and type 2 diabetes."
    ),

    (
        "Patient has critical limb "
        "ischaemia with stump pain. "
        "Started metformin 500mg bd "
        "and apixaban."
    ),

    (
        "The patient has controlled HTN. "
        "BP is 168/102 mmHg with "
        "headache and dizziness. "
        "Amlodipine was started."
    ),

    (
        "Patient with chronic kidney "
        "disease and diabetes presented "
        "with worsening shortness of breath."
    )
]


# ============================================================
# STANDALONE TEST
# ============================================================

def main():

    print("=" * 70)
    print("HEALTHAI MEDICAL NER ADAPTER")
    print("=" * 70)

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Model dir   : {MODEL_DIR}"
    )

    print(
        f"Label map   : {LABEL_MAP_PATH}"
    )

    print(
        f"Device      : {DEVICE}"
    )

    print(
        f"Threshold   : {MIN_CONFIDENCE}"
    )

    # --------------------------------------------------------
    # Load label map
    # --------------------------------------------------------

    label2id = load_label_map()

    print(
        f"Labels loaded: {len(label2id)}"
    )

    # --------------------------------------------------------
    # Load BioBERT
    # --------------------------------------------------------

    print(
        "\nLoading BioBERT..."
    )

    tokenizer, model = (
        get_model_and_tokenizer()
    )

    print(
        "BioBERT loaded successfully."
    )

    print(
        f"Parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # --------------------------------------------------------
    # Run tests
    # --------------------------------------------------------

    results = []

    print(
        "\nRunning medical NER test cases..."
    )

    print("-" * 70)

    for index, text in enumerate(
        TEST_CASES,
        start=1
    ):

        result = execute_medical_ner(
            query=text
        )

        results.append(result)

        print(
            f"\nTEST CASE {index}"
        )

        print(
            f"Input: {text}"
        )

        print(
            f"Status: {result['status']}"
        )

        print(
            f"Entities found: "
            f"{result.get('entity_count', 0)}"
        )

        if result.get("entities"):

            for entity in result["entities"]:

                print(
                    f"  - "
                    f"{entity['text']!r}"
                    f" | {entity['label']}"
                    f" | confidence="
                    f"{entity['confidence']:.4f}"
                    f" | span="
                    f"({entity['start']},"
                    f"{entity['end']})"
                )

        else:

            print(
                "  No entities detected."
            )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {
        "model": (
            "dmis-lab/biobert-v1.1"
        ),
        "model_directory": str(
            MODEL_DIR
        ),
        "device": str(DEVICE),
        "label_count": len(label2id),
        "min_confidence": MIN_CONFIDENCE,
        "test_case_count": len(
            TEST_CASES
        ),
        "results": results
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
        "\n" + "=" * 70
    )

    print(
        "STEP 17.4C MEDICAL NER ADAPTER"
    )

    print(
        "STANDALONE TEST COMPLETED"
    )

    print("=" * 70)

    print(
        f"\nResults saved to:"
    )

    print(
        OUTPUT_PATH
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()