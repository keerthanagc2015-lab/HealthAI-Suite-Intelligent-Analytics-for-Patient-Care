from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# HealthAI Medical Translator
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_NAME = "Helsinki-NLP/opus-mt-en-hi"

_tokenizer = None
_model = None


def load_translator():
    global _tokenizer, _model

    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

        _model.eval()

    return _tokenizer, _model


def translate_medical_text(
    text: str,
    target_language: str = "Hindi",
):
    """
    Translate healthcare text from English to Hindi.

    This is a pretrained neural machine translation model.
    It does not provide medical advice.
    """

    if not isinstance(text, str) or not text.strip():
        return {
            "status": "invalid_input",
            "tool": "medical_translation",
            "message": "Text must be a non-empty string.",
        }

    language = target_language.strip().lower()

    if language not in {"hindi", "hi"}:
        return {
            "status": "unsupported_language",
            "tool": "medical_translation",
            "message": (
                "Currently supported target language: Hindi."
            ),
        }

    try:
        tokenizer, model = load_translator()

        inputs = tokenizer(
            text.strip(),
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                early_stopping=True,
            )

        translated_text = tokenizer.decode(
            generated_tokens[0],
            skip_special_tokens=True,
        )

        return {
            "status": "success",
            "tool": "medical_translation",
            "source_language": "English",
            "target_language": "Hindi",
            "model": MODEL_NAME,
            "original_text": text.strip(),
            "translated_text": translated_text,
            "note": (
                "Machine translation is provided for "
                "information support and should not replace "
                "professional medical interpretation."
            ),
        }

    except Exception as exc:
        return {
            "status": "error",
            "tool": "medical_translation",
            "message": str(exc),
        }


def execute_medical_translation(
    query: str = "",
    text: str = "",
    target_language: str = "Hindi",
    **kwargs,
):
    """
    Agent-compatible translation adapter.
    """

    input_text = text.strip() if text else query.strip()

    return translate_medical_text(
        text=input_text,
        target_language=target_language,
    )


if __name__ == "__main__":

    test_text = (
        "Patients with diabetes should monitor "
        "their blood glucose regularly."
    )

    result = execute_medical_translation(
        text=test_text,
        target_language="Hindi",
    )

    print("\n" + "=" * 70)
    print("HEALTHAI MEDICAL TRANSLATOR TEST")
    print("=" * 70)

    for key, value in result.items():
        print(f"{key}: {value}")

    if result["status"] == "success":
        print("\nSTEP - MEDICAL TRANSLATOR TEST PASSED")
    else:
        print("\nSTEP - MEDICAL TRANSLATOR TEST FAILED")
