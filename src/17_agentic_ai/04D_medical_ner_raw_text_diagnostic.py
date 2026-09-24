"""
HealthAI - Step 17.4D
BioBERT Raw-Text Inference Diagnostic

Purpose:
Compare BioBERT predictions when:
1. Using the exact stored tokenized NER chunk.
2. Decoding that same chunk back into text and running raw-text inference.

This helps determine whether zero predictions from the raw-text
adapter are caused by:
- preprocessing/tokenization mismatch, or
- poor generalization to new text.
"""

from pathlib import Path
import json

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "nlp" / "biobert_ner"

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

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "agentic_ai"

OUTPUT_PATH = (
    OUTPUT_DIR / "medical_ner_raw_text_diagnostic.json"
)


# ============================================================
# CONFIG
# ============================================================

MAX_LENGTH = 512

# Use chunks with a meaningful number of gold entities.
MIN_GOLD_ENTITIES = 20

# Number of chunks to test.
NUM_CHUNKS = 3


# ============================================================
# LOAD LABEL MAP
# ============================================================

def load_label_map():

    with open(
        LABEL_MAP_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        raw_map = json.load(f)

    if "id2label" in raw_map:

        id2label = {
            int(k): v
            for k, v in raw_map["id2label"].items()
        }

    else:

        id2label = {
            int(v): k
            for k, v in raw_map.items()
        }

    return id2label


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("=" * 70)
    print("LOADING BIOBERT")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
        use_fast=True
    )

    model = AutoModelForTokenClassification.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return tokenizer, model, device


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks():

    print("\nLoading stored NER chunks...")

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks.")

    return chunks


# ============================================================
# STORED TOKENIZED PREDICTION
# ============================================================

def predict_stored_chunk(
    chunk,
    model,
    device
):

    input_ids = torch.tensor(
        [chunk["input_ids"]],
        dtype=torch.long
    ).to(device)

    attention_mask = torch.tensor(
        [chunk["attention_mask"]],
        dtype=torch.long
    ).to(device)

    # BioBERT does not require token_type_ids for this task,
    # but providing zeros is safe and matches BERT-style input.
    token_type_ids = torch.zeros_like(
        input_ids
    ).to(device)

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

    predictions = torch.argmax(
        probabilities,
        dim=-1
    )[0].cpu().tolist()

    confidences = torch.max(
        probabilities,
        dim=-1
    ).values[0].cpu().tolist()

    return predictions, confidences


# ============================================================
# RAW TEXT PREDICTION
# ============================================================

def predict_raw_text(
    text,
    tokenizer,
    model,
    device
):

    encoded = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False
    )

    offset_mapping = encoded.pop(
        "offset_mapping"
    )[0].tolist()

    model_inputs = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.no_grad():

        outputs = model(
            **model_inputs
        )

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

    predictions = torch.argmax(
        probabilities,
        dim=-1
    )[0].cpu().tolist()

    confidences = torch.max(
        probabilities,
        dim=-1
    ).values[0].cpu().tolist()

    return (
        predictions,
        confidences,
        offset_mapping,
        encoded["input_ids"][0].cpu().tolist()
    )


# ============================================================
# COUNT NON-O
# ============================================================

def count_non_o(
    predictions,
    id2label
):

    labels = [
        id2label.get(int(pred), "O")
        for pred in predictions
    ]

    non_o = [
        label
        for label in labels
        if label != "O"
    ]

    return len(non_o), labels


# ============================================================
# EXTRACT RAW TEXT ENTITIES
# ============================================================

def extract_raw_entities(
    text,
    predictions,
    confidences,
    offsets,
    id2label
):

    entities = []

    current = None

    for pred_id, confidence, offset in zip(
        predictions,
        confidences,
        offsets
    ):

        start, end = offset

        # Ignore special tokens.
        if start == end:
            continue

        label = id2label.get(
            int(pred_id),
            "O"
        )

        if label == "O":

            if current is not None:
                entities.append(current)
                current = None

            continue

        if "-" not in label:
            continue

        prefix, entity_type = label.split(
            "-",
            1
        )

        if prefix == "B":

            if current is not None:
                entities.append(current)

            current = {
                "label": entity_type,
                "start": start,
                "end": end,
                "confidence": [float(confidence)]
            }

        elif prefix == "I":

            if (
                current is not None
                and current["label"] == entity_type
                and start >= current["end"]
            ):

                current["end"] = end

                current["confidence"].append(
                    float(confidence)
                )

            else:

                if current is not None:
                    entities.append(current)

                current = {
                    "label": entity_type,
                    "start": start,
                    "end": end,
                    "confidence": [float(confidence)]
                }

    if current is not None:
        entities.append(current)

    final_entities = []

    for entity in entities:

        start = entity["start"]
        end = entity["end"]

        entity_text = text[
            start:end
        ].strip()

        if not entity_text:
            continue

        final_entities.append(
            {
                "text": entity_text,
                "label": entity["label"],
                "confidence": round(
                    sum(entity["confidence"])
                    / len(entity["confidence"]),
                    4
                ),
                "start": start,
                "end": end
            }
        )

    return final_entities


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("HEALTHAI - BIOBERT RAW TEXT DIAGNOSTIC")
    print("=" * 70)

    id2label = load_label_map()

    tokenizer, model, device = load_model()

    chunks = load_chunks()

    # --------------------------------------------------------
    # Select entity-rich chunks
    # --------------------------------------------------------

    selected = []

    for chunk in chunks:

        labels = chunk["labels"]

        gold_non_o = sum(
            1
            for label in labels
            if int(label) != 0
        )

        if gold_non_o >= MIN_GOLD_ENTITIES:

            selected.append(
                (
                    chunk,
                    gold_non_o
                )
            )

    selected.sort(
        key=lambda x: x[1],
        reverse=True
    )

    selected = selected[:NUM_CHUNKS]

    print(
        f"\nSelected {len(selected)} entity-rich chunks."
    )

    results = []

    # --------------------------------------------------------
    # Compare each chunk
    # --------------------------------------------------------

    for index, (chunk, gold_count) in enumerate(
        selected,
        start=1
    ):

        print("\n" + "=" * 70)
        print(f"DIAGNOSTIC CASE {index}")
        print("=" * 70)

        # ----------------------------------------------
        # Decode stored token IDs
        # ----------------------------------------------

        decoded_text = tokenizer.decode(
            chunk["input_ids"],
            skip_special_tokens=True
        )

        print(
            f"Chunk ID: {chunk['chunk_id']}"
        )

        print(
            f"Document ID: {chunk['document_id']}"
        )

        print(
            f"Gold non-O tokens: {gold_count}"
        )

        print("\nDecoded text preview:")
        print(decoded_text[:1000])

        # ----------------------------------------------
        # A. Stored-token inference
        # ----------------------------------------------

        stored_predictions, stored_confidences = (
            predict_stored_chunk(
                chunk,
                model,
                device
            )
        )

        stored_non_o, stored_labels = count_non_o(
            stored_predictions,
            id2label
        )

        # ----------------------------------------------
        # B. Raw-text inference
        # ----------------------------------------------

        (
            raw_predictions,
            raw_confidences,
            offsets,
            raw_input_ids
        ) = predict_raw_text(
            decoded_text,
            tokenizer,
            model,
            device
        )

        raw_non_o, raw_labels = count_non_o(
            raw_predictions,
            id2label
        )

        raw_entities = extract_raw_entities(
            decoded_text,
            raw_predictions,
            raw_confidences,
            offsets,
            id2label
        )

        # ----------------------------------------------
        # Tokenization comparison
        # ----------------------------------------------

        stored_ids = [
            int(x)
            for x in chunk["input_ids"]
        ]

        raw_ids = [
            int(x)
            for x in raw_input_ids
        ]

        same_tokenization = (
            stored_ids == raw_ids
        )

        print("\nRESULT")

        print(
            f"Stored-token non-O predictions: "
            f"{stored_non_o}"
        )

        print(
            f"Raw-text non-O predictions: "
            f"{raw_non_o}"
        )

        print(
            f"Tokenization identical: "
            f"{same_tokenization}"
        )

        print(
            f"Raw-text entities reconstructed: "
            f"{len(raw_entities)}"
        )

        if raw_entities:

            print("\nRaw-text entities:")

            for entity in raw_entities[:20]:

                print(
                    f"  {entity['text']}"
                    f" -> {entity['label']}"
                    f" | confidence="
                    f"{entity['confidence']}"
                )

        else:

            print(
                "\nNo raw-text entities detected."
            )

        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "gold_non_o_tokens": gold_count,
                "stored_token_non_o_predictions": stored_non_o,
                "raw_text_non_o_predictions": raw_non_o,
                "stored_token_count": len(stored_ids),
                "raw_token_count": len(raw_ids),
                "tokenization_identical": same_tokenization,
                "raw_entity_count": len(raw_entities),
                "raw_entities": raw_entities,
                "decoded_text": decoded_text
            }
        )

    # --------------------------------------------------------
    # Overall interpretation
    # --------------------------------------------------------

    total_stored = sum(
        r["stored_token_non_o_predictions"]
        for r in results
    )

    total_raw = sum(
        r["raw_text_non_o_predictions"]
        for r in results
    )

    identical_count = sum(
        1
        for r in results
        if r["tokenization_identical"]
    )

    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    print(
        f"Cases tested: {len(results)}"
    )

    print(
        f"Stored-token non-O predictions: "
        f"{total_stored}"
    )

    print(
        f"Raw-text non-O predictions: "
        f"{total_raw}"
    )

    print(
        f"Identical tokenization cases: "
        f"{identical_count}/{len(results)}"
    )

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    if total_raw > 0:

        print("\nINTERPRETATION:")
        print(
            "Raw-text inference is producing non-O "
            "predictions on real clinical text."
        )
        print(
            "The adapter pipeline is therefore "
            "functionally capable of extracting entities."
        )

    else:

        print("\nINTERPRETATION:")
        print(
            "Stored-token inference produces entities, "
            "but decoded raw-text inference produces none."
        )

        print(
            "This indicates a preprocessing/tokenization "
            "or input-reconstruction mismatch that "
            "needs further investigation."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {
        "model": "dmis-lab/biobert-v1.1",
        "device": str(device),
        "max_length": MAX_LENGTH,
        "cases_tested": len(results),
        "stored_token_non_o_predictions": total_stored,
        "raw_text_non_o_predictions": total_raw,
        "identical_tokenization_cases": identical_count,
        "results": results
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\nSaved diagnostic results:")
    print(OUTPUT_PATH)

    print(
        "\nSTEP 17.4D DIAGNOSTIC COMPLETED"
    )


if __name__ == "__main__":
    main()