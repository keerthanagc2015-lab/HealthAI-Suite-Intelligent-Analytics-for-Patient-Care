import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_NAME = "Helsinki-NLP/opus-mt-en-hi"


def main():
    if len(sys.argv) < 3:
        print(json.dumps({
            "status": "error",
            "tool": "medical_translation",
            "message": "Usage: python 20_translation_worker.py <text> <target_language>"
        }, ensure_ascii=False))
        sys.exit(1)

    text = sys.argv[1]
    target_language = sys.argv[2]

    if target_language.lower() != "hindi":
        print(json.dumps({
            "status": "error",
            "tool": "medical_translation",
            "message": "Currently supported target language: Hindi"
        }, ensure_ascii=False))
        sys.exit(1)

    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        outputs = model.generate(
            **inputs,
            max_length=512
        )

        translated_text = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        print(json.dumps({
            "status": "success",
            "tool": "medical_translation",
            "source_language": "English",
            "target_language": "Hindi",
            "model": MODEL_NAME,
            "input_text": text,
            "translated_text": translated_text,
            "note": "Translation generated using a pretrained neural machine translation model."
        }, ensure_ascii=False))

    except Exception as exc:
        print(json.dumps({
            "status": "error",
            "tool": "medical_translation",
            "message": str(exc)
        }, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
