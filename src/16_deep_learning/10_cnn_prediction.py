import os
import numpy as np
import tensorflow as tf
from PIL import Image


# ================================================================
# CONFIGURATION
# ================================================================

MODEL_PATH = (
    "data/processed/deep_learning/cnn/"
    "cnn_model.keras"
)

IMAGE_SIZE = 128


print("=" * 70)
print("HEALTHAI - CNN IMAGE PREDICTION")
print("=" * 70)


# ================================================================
# 1. CHECK TRAINED MODEL
# ================================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"\nTrained CNN model not found:\n{MODEL_PATH}"
    )


# ================================================================
# 2. LOAD TRAINED CNN MODEL
# ================================================================

print("\nLoading trained CNN model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("CNN model loaded successfully.")

print(
    "Model output shape:",
    model.output_shape
)

print(
    "Output activation:",
    model.layers[-1].activation.__name__
)


# ================================================================
# 3. GET IMAGE PATH FROM USER
# ================================================================

print("\n" + "=" * 70)
print("IMAGE INPUT")
print("=" * 70)

print(
    "\nEnter the full path of the X-ray image."
)

print(
    "Example:"
)

print(
    r"C:\Users\radha\Downloads\new_xray.png"
)


image_path = input(
    "\nEnter X-ray image path: "
).strip()


# Remove quotation marks if user pasted
# a path surrounded by quotes

image_path = image_path.strip('"').strip("'")


# ================================================================
# 4. CHECK IMAGE
# ================================================================

if not os.path.exists(image_path):

    raise FileNotFoundError(
        f"\nImage not found:\n{image_path}"
    )


print(
    "\nImage path:",
    os.path.abspath(image_path)
)


# ================================================================
# 5. LOAD IMAGE
# ================================================================

print("\nLoading X-ray image...")

image = Image.open(
    image_path
)

print(
    "Original image size:",
    image.size
)

print(
    "Original image mode:",
    image.mode
)


# ================================================================
# 6. IMAGE PREPROCESSING
# ================================================================

print("\nPreprocessing image...")


# Convert image to RGB

image = image.convert("RGB")


# Resize image

image = image.resize(
    (
        IMAGE_SIZE,
        IMAGE_SIZE
    )
)


# Convert image to NumPy array

image_array = np.array(
    image,
    dtype=np.float32
)


# Normalize pixel values
# 0-255 → 0-1

image_array = (
    image_array / 255.0
)


# Add batch dimension

image_array = np.expand_dims(
    image_array,
    axis=0
)


print(
    "Processed image shape:",
    image_array.shape
)

print(
    "Pixel range:",
    image_array.min(),
    "to",
    image_array.max()
)


# ================================================================
# 7. CNN PREDICTION
# ================================================================

print("\nGenerating prediction...")


prediction_output = model.predict(
    image_array,
    verbose=0
)


# Extract scalar probability safely

probability = float(
    np.asarray(prediction_output).reshape(-1)[0]
)


# ================================================================
# 8. DEBUG INFORMATION
# ================================================================

print("\n" + "-" * 70)
print("MODEL OUTPUT DEBUG")
print("-" * 70)

print(
    "Raw model output:",
    prediction_output
)

print(
    "Pneumonia probability value:",
    probability
)

print(
    "Probability percentage:",
    probability * 100
)

print(
    "Is probability >= 0.5?:",
    probability >= 0.5
)


# ================================================================
# 9. CONVERT PROBABILITY TO CLASS
# ================================================================

if probability >= 0.5:

    prediction = "PNEUMONIA"

else:

    prediction = "NORMAL"


# Probability interpretation

pneumonia_probability = (
    probability * 100
)

normal_probability = (
    (1 - probability) * 100
)


# ================================================================
# 10. DISPLAY FINAL RESULT
# ================================================================

print("\n" + "=" * 70)
print("CNN PREDICTION RESULT")
print("=" * 70)

print(
    f"\nPrediction: {prediction}"
)

print(
    f"\nPneumonia Probability:"
    f" {pneumonia_probability:.2f}%"
)

print(
    f"Normal Probability:"
    f" {normal_probability:.2f}%"
)


# ================================================================
# 11. CLASSIFICATION RULE
# ================================================================

print("\n" + "-" * 70)

print(
    "\nClassification rule:"
)

print(
    "Probability >= 50%  →  PNEUMONIA"
)

print(
    "Probability < 50%   →  NORMAL"
)


# ================================================================
# 12. DISCLAIMER
# ================================================================

print("\n" + "-" * 70)

print(
    "Note: This CNN prediction is a portfolio demonstration "
    "and should not be used as a medical diagnosis."
)


print("\n" + "=" * 70)
print("CNN PREDICTION COMPLETED SUCCESSFULLY")
print("=" * 70)