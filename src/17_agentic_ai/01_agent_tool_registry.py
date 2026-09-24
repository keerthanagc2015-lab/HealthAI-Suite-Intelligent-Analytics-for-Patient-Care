"""
HealthAI - Agentic AI
Module 17 | Step 01

Agent Tool Registry

Purpose:
    Define all AI capabilities available inside HealthAI.

The registry acts as the catalog of tools that the future
Agent Router and Agent Executor can use.

Current HealthAI capabilities:
    1. Diabetes risk prediction
    2. Hospital length-of-stay prediction
    3. Patient clustering
    4. Association rule analysis
    5. Chest X-ray analysis
    6. Patient deterioration prediction
    7. Medical NER
    8. Patient feedback sentiment analysis
    9. Healthcare RAG
    10. Safe fallback

This file does NOT execute the models.
It only defines the available tools and their metadata.
"""

from pathlib import Path
import json
from datetime import datetime


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "agentic_ai"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# AGENT TOOL REGISTRY
# ============================================================

TOOL_REGISTRY = {

    # --------------------------------------------------------
    # 1. CLASSICAL ML - DIABETES
    # --------------------------------------------------------

    "diabetes_prediction": {
        "tool_name": "diabetes_prediction",
        "category": "classical_ml",
        "model_type": "classification",

        "description": (
            "Predict diabetes-related risk using "
            "the structured patient classification model."
        ),

        "input_type": "structured_patient_data",

        "input_examples": [
            "age",
            "gender",
            "blood_glucose",
            "hba1c",
            "bmi",
            "cholesterol"
        ],

        "output": (
            "Diabetes risk prediction "
            "and associated probability."
        ),

        "status": "available"
    },


    # --------------------------------------------------------
    # 2. REGRESSION - HOSPITAL LOS
    # --------------------------------------------------------

    "hospital_los_prediction": {
        "tool_name": "hospital_los_prediction",
        "category": "regression",
        "model_type": "regression",

        "description": (
            "Predict the expected hospital "
            "length of stay."
        ),

        "input_type": "structured_patient_data",

        "input_examples": [
            "patient demographics",
            "admission information",
            "clinical features"
        ],

        "output": (
            "Predicted hospital length of stay."
        ),

        "status": "available"
    },


    # --------------------------------------------------------
    # 3. CLUSTERING
    # --------------------------------------------------------

    "patient_clustering": {
        "tool_name": "patient_clustering",
        "category": "clustering",
        "model_type": "unsupervised_learning",

        "description": (
            "Group patients into similar "
            "data-driven patient segments."
        ),

        "input_type": "structured_patient_data",

        "input_examples": [
            "age",
            "bmi",
            "blood_glucose",
            "cholesterol",
            "clinical characteristics"
        ],

        "output": (
            "Patient cluster assignment "
            "and cluster characteristics."
        ),

        "status": "available"
    },


    # --------------------------------------------------------
    # 4. ASSOCIATION RULES
    # --------------------------------------------------------

    "association_analysis": {
        "tool_name": "association_analysis",
        "category": "association_rules",
        "model_type": "association_rule_mining",

        "description": (
            "Identify frequently occurring relationships "
            "between symptoms, diagnoses, treatments, "
            "and patient outcomes."
        ),

        "input_type": "structured_healthcare_data",

        "input_examples": [
            "symptoms",
            "diagnosis",
            "treatment",
            "outcome"
        ],

        "output": (
            "Association rules with support, "
            "confidence, and lift."
        ),

        "status": "available"
    },


    # --------------------------------------------------------
    # 5. CNN - X-RAY
    # --------------------------------------------------------

    "xray_analysis": {
        "tool_name": "xray_analysis",
        "category": "deep_learning",
        "model_type": "cnn",

        "description": (
            "Analyze chest X-ray images using "
            "the trained CNN pneumonia classifier."
        ),

        "input_type": "medical_image",

        "input_examples": [
            "chest X-ray image"
        ],

        "output": (
            "Pneumonia probability, predicted class, "
            "and visual explanation using Grad-CAM."
        ),

        "status": "available"
    },


    # --------------------------------------------------------
    # 6. RNN - DETERIORATION
    # --------------------------------------------------------

    "deterioration_prediction": {
        "tool_name": "deterioration_prediction",
        "category": "deep_learning",
        "model_type": "rnn",

        "description": (
            "Predict near-term patient deterioration "
            "from sequential hospital observations."
        ),

        "input_type": "time_series",

        "input_examples": [
            "heart rate",
            "respiratory rate",
            "SpO2",
            "blood pressure",
            "temperature",
            "lactate",
            "creatinine"
        ],

        "output": (
            "Deterioration probability and "
            "risk classification."
        ),

        "status": "available"
    },


    # --------------------------------------------------------
    # 7. BIOBERT - MEDICAL NER
    # --------------------------------------------------------

    "medical_ner": {
        "tool_name": "medical_ner",
        "category": "medical_nlp",
        "model_type": "BioBERT",

        "description": (
            "Extract clinically relevant medical entities "
            "from medical text using the selected BioBERT NER model."
        ),

        "input_type": "medical_text",

        "input_examples": [
            "clinical note",
            "medical report",
            "patient note",
            "doctor note"
        ],

        "output": (
            "Detected medical entities such as "
            "conditions, symptoms, drugs, measurements, "
            "tests, and treatments."
        ),

        "status": "available",

        "selected_model": "dmis-lab/biobert-v1.1"
    },


    # --------------------------------------------------------
    # 8. SENTIMENT ANALYSIS
    # --------------------------------------------------------

    "sentiment_analysis": {
        "tool_name": "sentiment_analysis",
        "category": "medical_nlp",
        "model_type": "sentiment_classifier",

        "description": (
            "Analyze sentiment expressed in "
            "patient or healthcare feedback."
        ),

        "input_type": "patient_feedback",

        "input_examples": [
            "patient feedback",
            "hospital feedback",
            "service feedback",
            "satisfaction comment"
        ],

        "output": (
            "Positive or negative sentiment "
            "with model confidence."
        ),

        "status": "available",

        "candidate_models": [
            "tfidf_logistic_regression",
            "distilbert"
        ],

        "selected_model": (
            "tfidf_logistic_regression_and_distilbert"
        )
    },


    # --------------------------------------------------------
    # 9. RAG - MEDICAL KNOWLEDGE
    # --------------------------------------------------------

    "medical_rag": {
        "tool_name": "medical_rag",
        "category": "retrieval_augmented_generation",
        "model_type": "grounded_rag",

        "description": (
            "Answer healthcare knowledge questions using "
            "the safety-aware grounded RAG pipeline."
        ),

        "input_type": "healthcare_question",

        "input_examples": [
            "What are the symptoms of diabetes?",
            "What is hypertension?",
            "What is a chronic illness?",
            "How can diabetes be managed?"
        ],

        "output": (
            "Grounded healthcare information supported "
            "by retrieved authoritative sources."
        ),

        "status": "available",

        "retrieval_method": (
            "safety_aware_retrieval"
        ),

        "answering_method": (
            "clean_extractive_grounded_rag"
        ),

        "safety_threshold": 0.25
    },


    # --------------------------------------------------------
    # 10. SAFE FALLBACK
    # --------------------------------------------------------

    "medical_translation": {
        "tool_name": "medical_translation",
        "description": "Translate healthcare information from English to Hindi using a pretrained neural machine translation model.",
        "category": "generative_ai",
        "model_type": "transformer_nmt",
        "module": "20_medical_translation_adapter.py",
        "function": "execute_medical_translation",
        "input_type": "text",
        "output_type": "translated_text",
        "supported_languages": ["Hindi"],
    },
    "safe_fallback": {
        "tool_name": "safe_fallback",
        "category": "safety",
        "model_type": "fallback",

        "description": (
            "Handle requests that cannot be confidently "
            "mapped to a supported HealthAI capability."
        ),

        "input_type": "any",

        "output": (
            "A safe response indicating that the "
            "request is outside the supported capabilities."
        ),

        "status": "available"
    }
}


# ============================================================
# REGISTRY VALIDATION
# ============================================================

def validate_registry(registry):
    """
    Validate that every registered tool contains
    the minimum required metadata.
    """

    required_fields = [
        "tool_name",
        "category",
        "model_type",
        "description",
        "input_type",
        "output",
        "status"
    ]

    validation_results = []

    for tool_id, tool_config in registry.items():

        missing_fields = [
            field
            for field in required_fields
            if field not in tool_config
        ]

        validation_results.append(
            {
                "tool_id": tool_id,
                "valid": len(missing_fields) == 0,
                "missing_fields": missing_fields
            }
        )

    return validation_results


# ============================================================
# DISPLAY REGISTRY
# ============================================================

def display_registry(registry):

    print("\n" + "=" * 75)
    print("HEALTHAI AGENT TOOL REGISTRY")
    print("=" * 75)

    print(
        f"\nTotal registered tools: {len(registry)}"
    )

    for number, (tool_id, config) in enumerate(
        registry.items(),
        start=1
    ):

        print(
            f"\n{number}. {tool_id}"
        )

        print(
            f"   Category    : {config['category']}"
        )

        print(
            f"   Model type  : {config['model_type']}"
        )

        print(
            f"   Input       : {config['input_type']}"
        )

        print(
            f"   Status      : {config['status']}"
        )

        print(
            f"   Description : {config['description']}"
        )


# ============================================================
# SAVE REGISTRY
# ============================================================

def save_registry(registry, validation_results):

    output_file = (
        OUTPUT_DIR
        / "agent_tool_registry.json"
    )

    output = {
        "module": "17_agentic_ai",
        "step": "01",
        "component": "Agent Tool Registry",

        "created_at": datetime.now().isoformat(),

        "tool_count": len(registry),

        "registry_validation": validation_results,

        "tools": registry
    }

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )

    return output_file


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nStarting HealthAI Agent Tool Registry...")

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validation_results = validate_registry(
        TOOL_REGISTRY
    )

    invalid_tools = [
        result
        for result in validation_results
        if not result["valid"]
    ]

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    display_registry(
        TOOL_REGISTRY
    )

    # --------------------------------------------------------
    # Validation report
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("REGISTRY VALIDATION")
    print("=" * 75)

    if len(invalid_tools) == 0:

        print(
            "\n✓ All registered tools passed validation."
        )

    else:

        print(
            f"\n✗ {len(invalid_tools)} "
            "tools failed validation."
        )

        for tool in invalid_tools:

            print(
                f"\nTool: {tool['tool_id']}"
            )

            print(
                f"Missing: {tool['missing_fields']}"
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = save_registry(
        TOOL_REGISTRY,
        validation_results
    )

    print(
        f"\n✓ Registry saved to:"
    )

    print(
        output_file
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n" + "=" * 75)

    if len(invalid_tools) == 0:

        print(
            "STEP 17.1 COMPLETED SUCCESSFULLY"
        )

    else:

        print(
            "STEP 17.1 COMPLETED WITH VALIDATION ERRORS"
        )

    print("=" * 75)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()