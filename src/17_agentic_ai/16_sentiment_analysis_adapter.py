from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SENTIMENT_DIR = (
    PROJECT_ROOT
    / "models"
    / "nlp"
    / "sentiment"
    / "distilbert"
)


_tokenizer = None
_model = None


def load_sentiment_model():
    """Load the trained DistilBERT sentiment model."""

    global _tokenizer
    global _model

    if _tokenizer is None:
        if not SENTIMENT_DIR.exists():
            raise FileNotFoundError(
                f"DistilBERT sentiment model directory not found:\n"
                f"{SENTIMENT_DIR}"
            )

        _tokenizer = AutoTokenizer.from_pretrained(
            SENTIMENT_DIR
        )

    if _model is None:
        _model = AutoModelForSequenceClassification.from_pretrained(
            SENTIMENT_DIR
        )

        _model.eval()

    return _tokenizer, _model


def convert_sentiment_label(prediction: int) -> str:
    """Convert model class to readable sentiment label."""

    if prediction == 1:
        return "POSITIVE"

    if prediction == 0:
        return "NEGATIVE"

    return str(prediction)


def predict_sentiment(text: str) -> dict[str, Any]:
    """Predict medical/patient feedback sentiment using DistilBERT."""

    if not isinstance(text, str):
        raise TypeError("Sentiment input must be a string.")

    text = text.strip()

    if not text:
        raise ValueError("Sentiment text cannot be empty.")

    tokenizer, model = load_sentiment_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1,
    )[0]

    prediction = int(
        torch.argmax(probabilities).item()
    )

    return {
        "prediction": prediction,
        "sentiment": convert_sentiment_label(prediction),
        "negative_probability": round(
            float(probabilities[0]),
            6,
        ),
        "positive_probability": round(
            float(probabilities[1]),
            6,
        ),
    }


def execute_sentiment_analysis(
    query: str = "",
    text: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:

    try:

        input_text = text

        if input_text is None:
            input_text = query

        if not input_text or not input_text.strip():

            return {
                "status": "error",
                "tool": "sentiment_analysis",
                "prediction_type": (
                    "medical_sentiment_analysis"
                ),
                "query": query,
                "message": (
                    "Medical feedback text is "
                    "required for sentiment analysis."
                ),
                "error": (
                    "No sentiment text was provided."
                ),
            }

        result = predict_sentiment(input_text)

        return {
            "status": "success",
            "tool": "sentiment_analysis",
            "prediction_type": (
                "medical_sentiment_analysis"
            ),
            "query": query,
            "text": input_text,
            "prediction": result["prediction"],
            "sentiment": result["sentiment"],
            "negative_probability": result[
                "negative_probability"
            ],
            "positive_probability": result[
                "positive_probability"
            ],
            "model": "DistilBERT",
            "source": (
                "Module 16 medical sentiment analysis"
            ),
            "model_status": (
                "Loaded trained DistilBERT sentiment model"
            ),
            "deep_learning": True,
            "label_mapping": {
                "0": "NEGATIVE",
                "1": "POSITIVE",
            },
            "dataset_note": (
                "The evaluated dataset contains "
                "only 20 unique feedback texts, "
                "so the reported evaluation is "
                "illustrative rather than a "
                "clinical validation."
            ),
        }

    except Exception as exc:

        return {
            "status": "error",
            "tool": "sentiment_analysis",
            "prediction_type": (
                "medical_sentiment_analysis"
            ),
            "query": query,
            "message": (
                "Medical sentiment analysis failed."
            ),
            "error": str(exc),
        }


def main() -> None:

    print("=" * 70)
    print(
        "HEALTHAI - DISTILBERT MEDICAL SENTIMENT "
        "ANALYSIS ADAPTER"
    )
    print("=" * 70)

    print("\nChecking DistilBERT artifacts...")

    print(
        "Model directory:",
        (
            "AVAILABLE"
            if SENTIMENT_DIR.exists()
            else "MISSING"
        ),
    )

    print("\nRunning adapter test...")

    test_text = (
        "The staff were very helpful "
        "and explained everything clearly."
    )

    result = execute_sentiment_analysis(
        query="Analyze this patient feedback.",
        text=test_text,
    )

    print("\nAdapter status:")
    print(result.get("status"))

    if result.get("status") == "success":

        print("\nText:", result["text"])
        print("Prediction:", result["prediction"])
        print("Sentiment:", result["sentiment"])
        print(
            "Negative probability:",
            result["negative_probability"],
        )
        print(
            "Positive probability:",
            result["positive_probability"],
        )
        print("Model:", result["model"])
        print("Deep learning:", result["deep_learning"])

        print(
            "\nSTEP - DISTILBERT SENTIMENT "
            "ADAPTER TEST PASSED"
        )

    else:

        print(
            "\nSTEP - DISTILBERT SENTIMENT "
            "ADAPTER TEST FAILED"
        )

        print("Error:", result.get("error"))


if __name__ == "__main__":
    main()
