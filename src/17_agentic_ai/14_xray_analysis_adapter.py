"""
======================================================================
HealthAI Intelligent Platform
X-Ray CNN Adapter
======================================================================

Purpose
-------
Connect the trained CNN X-ray model from Module 16 to the
Agentic AI / FastAPI layer.

Pipeline
--------
Uploaded X-Ray
      ↓
FastAPI / Streamlit
      ↓
Image preprocessing
      ↓
128 x 128 RGB numpy array
      ↓
HealthAIModelService.predict_xray(image_array)
      ↓
CNN
      ↓
PNEUMONIA / NORMAL
      ↓
Pneumonia probability
      ↓
Structured response

IMPORTANT
---------
The existing model_service.py expects:

    predict_xray(image_array)

It does NOT expect image_path.

Expected CNN input:

    (1, 128, 128, 3)

The adapter does not retrain the model.
======================================================================
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from PIL import Image


# ======================================================================
# PROJECT PATH
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEEP_LEARNING_DIR = (
    PROJECT_ROOT
    / "src"
    / "16_deep_learning"
)

if str(DEEP_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(DEEP_LEARNING_DIR))


# ======================================================================
# IMPORT MODEL SERVICE
# ======================================================================

import importlib.util

_MODEL_SERVICE_PATH = DEEP_LEARNING_DIR / "model_service.py"
_MODEL_SERVICE_SPEC = importlib.util.spec_from_file_location(
    "healthai_deep_learning_model_service",
    _MODEL_SERVICE_PATH
)
_MODEL_SERVICE_MODULE = importlib.util.module_from_spec(_MODEL_SERVICE_SPEC)
_MODEL_SERVICE_SPEC.loader.exec_module(_MODEL_SERVICE_MODULE)

HealthAIModelService = _MODEL_SERVICE_MODULE.HealthAIModelService


# ======================================================================
# MODEL SERVICE CACHE
# ======================================================================

_MODEL_SERVICE = None


def get_model_service():
    """
    Load HealthAIModelService only once.

    This prevents the CNN/RNN models from being loaded every time
    an X-ray request is made.
    """

    global _MODEL_SERVICE

    if _MODEL_SERVICE is None:

        print("=" * 70)
        print("INITIALIZING HEALTHAI X-RAY MODEL SERVICE")
        print("=" * 70)

        _MODEL_SERVICE = HealthAIModelService()

        print("=" * 70)
        print("X-RAY MODEL SERVICE READY")
        print("=" * 70)

    return _MODEL_SERVICE


# ======================================================================
# IMAGE PREPROCESSING
# ======================================================================

def preprocess_xray(image_array):
    """
    Convert an image array into the exact format expected by the CNN.

    Expected final shape:

        (1, 128, 128, 3)

    Expected value range:

        0.0 - 1.0
    """

    # --------------------------------------------------------------
    # Convert to numpy
    # --------------------------------------------------------------

    image_array = np.asarray(
        image_array,
        dtype=np.float32
    )

    # --------------------------------------------------------------
    # Remove unnecessary dimensions
    # --------------------------------------------------------------

    if image_array.ndim == 4:

        # Already batched.
        if image_array.shape[0] != 1:

            raise ValueError(
                "X-ray adapter expects one image at a time. "
                f"Received shape: {image_array.shape}"
            )

        image = image_array[0]

    elif image_array.ndim == 3:

        image = image_array

    else:

        raise ValueError(
            "Invalid X-ray image shape. "
            "Expected (128,128,3) or (1,128,128,3). "
            f"Received: {image_array.shape}"
        )

    # --------------------------------------------------------------
    # Handle grayscale images
    # --------------------------------------------------------------

    if image.shape[-1] == 1:

        image = np.repeat(
            image,
            3,
            axis=-1
        )

    # --------------------------------------------------------------
    # Validate RGB
    # --------------------------------------------------------------

    if image.shape[-1] != 3:

        raise ValueError(
            "CNN expects an RGB image with 3 channels. "
            f"Received shape: {image.shape}"
        )

    # --------------------------------------------------------------
    # Normalize image values
    # --------------------------------------------------------------

    # If image is uint8-like 0-255
    if image.max() > 1.0:

        image = image / 255.0

    image = np.clip(
        image,
        0.0,
        1.0
    )

    # --------------------------------------------------------------
    # Resize to 128 x 128
    # --------------------------------------------------------------

    if image.shape[:2] != (128, 128):

        image_uint8 = (
            image * 255.0
        ).astype(
            np.uint8
        )

        pil_image = Image.fromarray(
            image_uint8
        ).convert("RGB")

        pil_image = pil_image.resize(
            (128, 128),
            Image.Resampling.LANCZOS
        )

        image = (
            np.asarray(
                pil_image,
                dtype=np.float32
            )
            / 255.0
        )

    # --------------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------------

    image_array = np.expand_dims(
        image,
        axis=0
    )

    # --------------------------------------------------------------
    # Final validation
    # --------------------------------------------------------------

    if image_array.shape != (1, 128, 128, 3):

        raise ValueError(
            "Final CNN input has incorrect shape. "
            f"Expected (1,128,128,3), "
            f"received {image_array.shape}"
        )

    return image_array.astype(
        np.float32
    )


# ======================================================================
# NORMALIZE CNN RESULT
# ======================================================================

def normalize_cnn_result(result):
    """
    Normalize the output from HealthAIModelService.predict_xray().
    """

    if not isinstance(result, dict):

        raise ValueError(
            "CNN model returned an unexpected result type: "
            f"{type(result).__name__}"
        )

    prediction = result.get(
        "prediction"
    )

    pneumonia_probability = result.get(
        "pneumonia_probability"
    )

    normal_probability = result.get(
        "normal_probability"
    )

    # --------------------------------------------------------------
    # Validate probability
    # --------------------------------------------------------------

    if pneumonia_probability is None:

        raise ValueError(
            "CNN model did not return pneumonia_probability."
        )

    pneumonia_probability = float(
        pneumonia_probability
    )

    # If probability somehow arrives as percentage.
    if pneumonia_probability > 1:

        pneumonia_probability /= 100.0

    pneumonia_probability = max(
        0.0,
        min(
            1.0,
            pneumonia_probability
        )
    )

    # --------------------------------------------------------------
    # Normal probability
    # --------------------------------------------------------------

    if normal_probability is None:

        normal_probability = (
            1.0 -
            pneumonia_probability
        )

    normal_probability = float(
        normal_probability
    )

    normal_probability = max(
        0.0,
        min(
            1.0,
            normal_probability
        )
    )

    # --------------------------------------------------------------
    # Prediction fallback
    # --------------------------------------------------------------

    if prediction is None:

        prediction = (
            "PNEUMONIA"
            if pneumonia_probability >= 0.50
            else "NORMAL"
        )

    prediction = str(
        prediction
    ).strip().upper()

    # --------------------------------------------------------------
    # Normalize label
    # --------------------------------------------------------------

    if prediction in (
        "PNEUMONIA",
        "PNEUMONIA DETECTED",
        "POSITIVE",
        "1"
    ):

        prediction = "PNEUMONIA"

    elif prediction in (
        "NORMAL",
        "NO PNEUMONIA",
        "NEGATIVE",
        "0"
    ):

        prediction = "NORMAL"

    else:

        raise ValueError(
            "CNN returned an unknown prediction label: "
            f"{prediction}"
        )

    # --------------------------------------------------------------
    # Final response
    # --------------------------------------------------------------

    return {

        "prediction":
            prediction,

        "pneumonia_probability":
            pneumonia_probability,

        "pneumonia_probability_pct":
            round(
                pneumonia_probability * 100,
                2
            ),

        "normal_probability":
            normal_probability,

        "normal_probability_pct":
            round(
                normal_probability * 100,
                2
            )
    }


# ======================================================================
# MAIN X-RAY ADAPTER
# ======================================================================

def execute_xray_analysis(
    query: str = "",
    image_array=None,
    **kwargs
):
    """
    Execute CNN X-ray analysis.

    Parameters
    ----------
    query : str
        User query. Kept for Agentic AI compatibility.

    image_array : numpy.ndarray
        Uploaded X-ray image.

    Returns
    -------
    dict
        Structured CNN prediction.
    """

    # --------------------------------------------------------------
    # Validate image
    # --------------------------------------------------------------

    if image_array is None:

        return {

            "status":
                "error",

            "tool":
                "xray_analysis",

            "prediction":
                None,

            "pneumonia_probability":
                None,

            "pneumonia_probability_pct":
                None,

            "message":
                "An X-ray image is required.",

            "error":
                "image_array was not provided."
        }

    try:

        # ----------------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------------

        processed_image = preprocess_xray(
            image_array
        )

        # ----------------------------------------------------------
        # LOAD MODEL SERVICE
        # ----------------------------------------------------------

        service = get_model_service()

        # ----------------------------------------------------------
        # CALL THE ACTUAL CNN
        #
        # IMPORTANT:
        # model_service.py defines:
        #
        #     predict_xray(self, image_array)
        #
        # Therefore we pass the array POSITIONALLY.
        # ----------------------------------------------------------

        cnn_result = service.predict_xray(
            processed_image
        )

        # ----------------------------------------------------------
        # NORMALIZE RESULT
        # ----------------------------------------------------------

        result = normalize_cnn_result(
            cnn_result
        )

        prediction = result[
            "prediction"
        ]

        # ----------------------------------------------------------
        # USER-FRIENDLY MESSAGE
        # ----------------------------------------------------------

        if prediction == "PNEUMONIA":

            message = (
                "Pneumonia detected by the CNN model."
            )

        else:

            message = (
                "No pneumonia detected by the CNN model."
            )

        # ----------------------------------------------------------
        # FINAL RESPONSE
        # ----------------------------------------------------------

        return {

            "status":
                "success",

            "tool":
                "xray_analysis",

            "prediction":
                prediction,

            "pneumonia_probability":
                result[
                    "pneumonia_probability"
                ],

            "pneumonia_probability_pct":
                result[
                    "pneumonia_probability_pct"
                ],

            "normal_probability":
                result[
                    "normal_probability"
                ],

            "normal_probability_pct":
                result[
                    "normal_probability_pct"
                ],

            "message":
                message
        }

    except Exception as exc:

        # ----------------------------------------------------------
        # ERROR RESPONSE
        # ----------------------------------------------------------

        return {

            "status":
                "error",

            "tool":
                "xray_analysis",

            "prediction":
                None,

            "pneumonia_probability":
                None,

            "pneumonia_probability_pct":
                None,

            "message":
                "CNN X-ray analysis failed.",

            "error_type":
                type(exc).__name__,

            "error":
                str(exc)
        }


# ======================================================================
# BACKWARD COMPATIBILITY
# ======================================================================

predict_xray = execute_xray_analysis


# ======================================================================
# STANDALONE TEST
# ======================================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("HEALTHAI X-RAY CNN ADAPTER TEST")
    print("=" * 70)

    # --------------------------------------------------------------
    # Create deterministic test image
    # --------------------------------------------------------------

    test_image = np.zeros(
        (1, 128, 128, 3),
        dtype=np.float32
    )

    print()
    print("Test input shape:")
    print(
        test_image.shape
    )

    print()
    print("Running CNN...")

    result = execute_xray_analysis(
        query="Test X-ray",
        image_array=test_image
    )

    print()
    print("CNN RESULT")
    print("-" * 70)

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )

    print()
    print("=" * 70)

    if result.get("status") == "success":

        print(
            "STEP X-RAY ADAPTER TEST PASSED"
        )

    else:

        print(
            "STEP X-RAY ADAPTER TEST FAILED"
        )

    print("=" * 70)
