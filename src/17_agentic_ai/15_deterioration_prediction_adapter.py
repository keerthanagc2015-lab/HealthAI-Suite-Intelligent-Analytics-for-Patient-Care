"""
HealthAI - Deterioration Prediction Agent Adapter

Connects the existing Module 16 RNN deterioration
prediction model to the Agentic AI layer.

Expected RNN input:
    (1, 24, 28)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEEP_LEARNING_DIR = (
    PROJECT_ROOT / "src" / "16_deep_learning"
)

if str(DEEP_LEARNING_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(DEEP_LEARNING_DIR),
    )


# ============================================================
# EXISTING MODEL SERVICE
# ============================================================

from model_service import HealthAIModelService


# ============================================================
# MODEL SERVICE CACHE
# ============================================================

_model_service: HealthAIModelService | None = None


def get_model_service() -> HealthAIModelService:
    """
    Load the existing HealthAIModelService once.
    """

    global _model_service

    if _model_service is None:
        _model_service = HealthAIModelService()

    return _model_service


# ============================================================
# SEQUENCE PREPARATION
# ============================================================

def build_sequence(
    patient_sequence: list[list[float]] | None = None,
) -> np.ndarray:
    """
    Prepare a 24-hour patient sequence.

    Expected shape before batch dimension:
        (24, 28)

    Final model input:
        (1, 24, 28)
    """

    if patient_sequence is None:

        # Deterministic sample input for adapter testing.
        # This is NOT a real patient prediction.
        patient_sequence = [
            [0.0] * 28
            for _ in range(24)
        ]

    sequence_array = np.asarray(
        patient_sequence,
        dtype=np.float32,
    )

    if sequence_array.shape != (24, 28):

        raise ValueError(
            "Invalid patient sequence shape. "
            f"Expected (24, 28), "
            f"received {sequence_array.shape}."
        )

    sequence_array = np.expand_dims(
        sequence_array,
        axis=0,
    )

    return sequence_array


# ============================================================
# AGENT EXECUTION
# ============================================================

def execute_deterioration_prediction(
    query: str = "",
    patient_sequence: list[list[float]] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Execute the existing RNN deterioration prediction model.

    The adapter maps the HealthAIModelService response into
    the standardized Agentic AI response format.
    """

    try:

        # ----------------------------------------------------
        # Prepare input sequence
        # ----------------------------------------------------

        sequence_array = build_sequence(
            patient_sequence
        )

        # ----------------------------------------------------
        # Load existing model service
        # ----------------------------------------------------

        service = get_model_service()

        # ----------------------------------------------------
        # Run RNN prediction
        # ----------------------------------------------------

        prediction = service.predict_deterioration(
            sequence_array
        )

        # ----------------------------------------------------
        # Validate model-service response
        # ----------------------------------------------------

        if not isinstance(prediction, dict):

            raise TypeError(
                "RNN model service returned an invalid "
                "prediction response."
            )

        if prediction.get("status") != "success":

            raise RuntimeError(
                prediction.get(
                    "message",
                    "RNN deterioration prediction failed.",
                )
            )

        if "prediction" not in prediction:

            raise KeyError(
                "RNN response is missing 'prediction'."
            )

        if "deterioration_probability" not in prediction:

            raise KeyError(
                "RNN response is missing "
                "'deterioration_probability'."
            )

        # ----------------------------------------------------
        # Extract model outputs
        # ----------------------------------------------------

        prediction_result = str(
            prediction["prediction"]
        )

        deterioration_probability = float(
            prediction["deterioration_probability"]
        )

        # The existing model service returns the positive
        # deterioration probability. Therefore the probability
        # of no deterioration is its complement.
        no_deterioration_probability = (
            1.0 - deterioration_probability
        )

        # The existing model service calls the classification
        # result HIGH_RISK / LOW_RISK. Use that as the
        # standardized risk level.
        risk_level = prediction_result

        # ----------------------------------------------------
        # Return standardized Agentic AI response
        # ----------------------------------------------------

        return {

            "status": "success",

            "tool": "deterioration_prediction",

            "prediction_type": (
                "patient_deterioration_prediction"
            ),

            "query": query,

            "prediction": prediction_result,

            "risk_level": risk_level,

            "deterioration_probability": round(
                deterioration_probability,
                6,
            ),

            "no_deterioration_probability": round(
                no_deterioration_probability,
                6,
            ),

            "deterioration_probability_pct": round(
                deterioration_probability * 100,
                2,
            ),

            "no_deterioration_probability_pct": round(
                no_deterioration_probability * 100,
                2,
            ),

            "model": "RNN",

            "input_shape": list(
                sequence_array.shape
            ),

            "source": (
                "Module 16 RNN deterioration "
                "model via HealthAIModelService"
            ),

            "model_status": (
                "Loaded trained RNN"
            ),

            "note": (
                "This is an AI risk prediction "
                "and not a clinical diagnosis."
            ),
        }

    except Exception as exc:

        return {

            "status": "error",

            "tool": "deterioration_prediction",

            "prediction_type": (
                "patient_deterioration_prediction"
            ),

            "query": query,

            "message": (
                "RNN deterioration prediction failed."
            ),

            "error": str(exc),
        }


# ============================================================
# STANDALONE TEST
# ============================================================

def main() -> None:

    print("=" * 70)
    print(
        "HEALTHAI - DETERIORATION "
        "PREDICTION AGENT ADAPTER"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Check RNN model
    # --------------------------------------------------------

    rnn_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "deep_learning"
        / "rnn"
        / "rnn_model.keras"
    )

    print("\nChecking RNN model...")

    print(
        "RNN model:",
        (
            "AVAILABLE"
            if rnn_path.exists()
            else "MISSING"
        ),
    )

    if not rnn_path.exists():

        print(
            "\nSTEP - DETERIORATION ADAPTER "
            "TEST FAILED"
        )

        return

    # --------------------------------------------------------
    # Run adapter test
    # --------------------------------------------------------

    print("\nRunning adapter test...")

    result = execute_deterioration_prediction(
        query=(
            "Predict deterioration risk "
            "for this patient."
        )
    )

    print("\nAdapter status:")
    print(result.get("status"))

    # --------------------------------------------------------
    # Successful response
    # --------------------------------------------------------

    if result.get("status") == "success":

        print(
            "\nPrediction:",
            result["prediction"],
        )

        print(
            "Risk level:",
            result["risk_level"],
        )

        print(
            "Deterioration probability:",
            result[
                "deterioration_probability"
            ],
        )

        print(
            "No deterioration probability:",
            result[
                "no_deterioration_probability"
            ],
        )

        print(
            "Model:",
            result["model"],
        )

        print(
            "Input shape:",
            result["input_shape"],
        )

        print(
            "\nSTEP - DETERIORATION "
            "ADAPTER TEST PASSED"
        )

    # --------------------------------------------------------
    # Failed response
    # --------------------------------------------------------

    else:

        print(
            "\nSTEP - DETERIORATION "
            "ADAPTER TEST FAILED"
        )

        print(
            "Error:",
            result.get("error"),
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()