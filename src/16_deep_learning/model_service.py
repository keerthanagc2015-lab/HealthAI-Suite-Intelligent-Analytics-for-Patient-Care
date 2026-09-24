"""
HealthAI Model Service
======================

Central inference service for:

1. CNN X-ray pneumonia classification
2. RNN patient deterioration prediction

Keras compatibility:
--------------------
The CNN .keras file was created with an InputLayer configuration
containing `batch_shape`.

Current Keras 3.15.1 does not deserialize that configuration
directly, so CompatibleInputLayer converts:

    batch_shape = [None, 128, 128, 3]

into:

    shape = (128, 128, 3)
    batch_size = None

The original CNN model file is NOT modified.
"""

import numpy as np
import tensorflow as tf
from keras.layers import InputLayer


# ============================================================
# MODEL PATHS
# ============================================================

CNN_MODEL_PATH = (
    "data/processed/deep_learning/cnn/cnn_model.keras"
)

RNN_MODEL_PATH = (
    "data/processed/deep_learning/rnn/rnn_model.keras"
)


# ============================================================
# KERAS COMPATIBILITY LAYER
# ============================================================

class CompatibleInputLayer(InputLayer):
    """
    Compatibility wrapper for older Keras .keras files
    containing InputLayer.batch_shape.

    Example serialized configuration:

        batch_shape: [None, 128, 128, 3]

    Converted internally to:

        shape=(128, 128, 3)
        batch_size=None
    """

    def __init__(self, batch_shape=None, **kwargs):

        if batch_shape is not None:

            batch_shape = tuple(batch_shape)

            # Remove the batch dimension.
            # Example:
            # (None, 128, 128, 3)
            #
            # becomes:
            # shape=(128, 128, 3)

            kwargs["shape"] = tuple(batch_shape[1:])

            # Preserve the original batch size.
            # Usually None for this model.

            kwargs["batch_size"] = batch_shape[0]

        super().__init__(**kwargs)


# ============================================================
# CNN LOADER
# ============================================================

def load_cnn_model():

    print("Loading CNN model...")

    model = tf.keras.models.load_model(
        CNN_MODEL_PATH,
        custom_objects={
            "InputLayer": CompatibleInputLayer
        },
        compile=False
    )

    print("✓ CNN loaded")

    return model


# ============================================================
# RNN LOADER
# ============================================================

def load_rnn_model():

    print("Loading RNN model...")

    model = tf.keras.models.load_model(
        RNN_MODEL_PATH,
        compile=False
    )

    print("✓ RNN loaded")

    return model


# ============================================================
# HEALTHAI MODEL SERVICE
# ============================================================

class HealthAIModelService:

    def __init__(self):

        print("=" * 70)
        print("HEALTHAI MODEL SERVICE")
        print("=" * 70)

        # ----------------------------------------------------
        # Load CNN
        # ----------------------------------------------------

        self.cnn_model = load_cnn_model()

        # ----------------------------------------------------
        # Load RNN
        # ----------------------------------------------------

        self.rnn_model = load_rnn_model()

        print("=" * 70)
        print("ALL MODELS LOADED SUCCESSFULLY")
        print("=" * 70)


    # ========================================================
    # X-RAY CNN PREDICTION
    # ========================================================

    def predict_xray(self, image_array):

        """
        Run pneumonia classification on a chest X-ray.

        Expected input:

            (1, 128, 128, 3)

        Also accepts:

            (128, 128, 3)

        The method automatically adds the batch dimension
        when necessary.
        """

        try:

            # ------------------------------------------------
            # Validate input
            # ------------------------------------------------

            if image_array is None:

                raise ValueError(
                    "X-ray image array is required."
                )

            image_array = np.asarray(
                image_array,
                dtype=np.float32
            )

            # ------------------------------------------------
            # Add batch dimension if necessary
            # ------------------------------------------------

            if image_array.ndim == 3:

                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )

            # ------------------------------------------------
            # Validate dimensions
            # ------------------------------------------------

            if image_array.ndim != 4:

                raise ValueError(
                    "X-ray input must have shape "
                    "(batch, 128, 128, 3). "
                    f"Received: {image_array.shape}"
                )

            # ------------------------------------------------
            # Validate image dimensions
            # ------------------------------------------------

            if tuple(image_array.shape[1:]) != (
                128,
                128,
                3
            ):

                raise ValueError(
                    "Expected X-ray image dimensions "
                    "(128, 128, 3). "
                    f"Received: {image_array.shape}"
                )

            # ------------------------------------------------
            # Normalize if input is 0-255
            # ------------------------------------------------

            if image_array.max() > 1.0:

                image_array = image_array / 255.0

            # ------------------------------------------------
            # CNN inference
            # ------------------------------------------------

            probability = self.cnn_model.predict(
                image_array,
                verbose=0
            )[0][0]

            probability = float(probability)

            # ------------------------------------------------
            # Classification
            # ------------------------------------------------

            if probability >= 0.50:

                prediction = "PNEUMONIA"

            else:

                prediction = "NORMAL"

            # ------------------------------------------------
            # Probabilities
            # ------------------------------------------------

            pneumonia_probability = probability

            normal_probability = (
                1.0 - pneumonia_probability
            )

            # ------------------------------------------------
            # Response
            # ------------------------------------------------

            return {

                "status": "success",

                "tool": "xray_analysis",

                "prediction": prediction,

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
                    ),

                "message":
                    (
                        "Pneumonia detected by the CNN model."
                        if prediction == "PNEUMONIA"
                        else
                        "No pneumonia detected by the CNN model."
                    )
            }

        except Exception as e:

            return {

                "status": "error",

                "tool": "xray_analysis",

                "prediction": None,

                "pneumonia_probability": None,

                "pneumonia_probability_pct": None,

                "normal_probability": None,

                "normal_probability_pct": None,

                "message":
                    "CNN X-ray analysis failed.",

                "error_type":
                    type(e).__name__,

                "error":
                    str(e)
            }


    # ========================================================
    # RNN DETERIORATION PREDICTION
    # ========================================================

    def predict_deterioration(self, sequence_array):

        """
        Predict patient deterioration from a temporal sequence.

        The sequence should match the input shape expected by
        the trained RNN model.
        """

        try:

            # ------------------------------------------------
            # Validate input
            # ------------------------------------------------

            if sequence_array is None:

                raise ValueError(
                    "Sequence array is required."
                )

            sequence_array = np.asarray(
                sequence_array,
                dtype=np.float32
            )

            # ------------------------------------------------
            # RNN prediction
            # ------------------------------------------------

            prediction = self.rnn_model.predict(
                sequence_array,
                verbose=0
            )

            # ------------------------------------------------
            # Extract probability
            # ------------------------------------------------

            if (
                prediction.ndim == 2
                and prediction.shape[1] == 1
            ):

                probability = float(
                    prediction[0][0]
                )

            else:

                probability = float(
                    prediction.reshape(-1)[0]
                )

            # ------------------------------------------------
            # Classification
            # ------------------------------------------------

            if probability >= 0.50:

                result = "HIGH_RISK"

            else:

                result = "LOW_RISK"

            # ------------------------------------------------
            # Response
            # ------------------------------------------------

            return {

                "status": "success",

                "tool":
                    "deterioration_prediction",

                "prediction":
                    result,

                "deterioration_probability":
                    probability,

                "deterioration_probability_pct":
                    round(
                        probability * 100,
                        2
                    ),

                "message":
                    (
                        "Elevated deterioration risk predicted."
                        if result == "HIGH_RISK"
                        else
                        "Low deterioration risk predicted."
                    )
            }

        except Exception as e:

            return {

                "status": "error",

                "tool":
                    "deterioration_prediction",

                "prediction": None,

                "deterioration_probability": None,

                "deterioration_probability_pct": None,

                "message":
                    "Deterioration prediction failed.",

                "error_type":
                    type(e).__name__,

                "error":
                    str(e)
            }


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("HEALTHAI MODEL SERVICE TEST")
    print("=" * 70)

    try:

        # ----------------------------------------------------
        # Initialize service
        # ----------------------------------------------------

        service = HealthAIModelService()

        # ----------------------------------------------------
        # CNN TEST
        # ----------------------------------------------------

        print()
        print("Testing CNN X-ray inference...")
        print()

        dummy_xray = np.zeros(
            (1, 128, 128, 3),
            dtype=np.float32
        )

        xray_result = service.predict_xray(
            dummy_xray
        )

        print()
        print("CNN RESULT")
        print("-" * 70)

        for key, value in xray_result.items():

            print(f"{key}: {value}")

        # ----------------------------------------------------
        # Check result
        # ----------------------------------------------------

        if xray_result.get("status") == "success":

            print()
            print("✓ CNN TEST PASSED")

        else:

            print()
            print("✗ CNN TEST FAILED")

        print()
        print("=" * 70)
        print("MODEL SERVICE TEST COMPLETED")
        print("=" * 70)

    except Exception as e:

        print()
        print("=" * 70)
        print("MODEL SERVICE TEST FAILED")
        print("=" * 70)

        print(
            "Error type:",
            type(e).__name__
        )

        print(
            "Error:",
            str(e)
        )