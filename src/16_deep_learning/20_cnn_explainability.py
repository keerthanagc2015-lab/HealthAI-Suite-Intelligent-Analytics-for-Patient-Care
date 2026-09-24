import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image


print("=" * 70)
print("HEALTHAI - CNN EXPLAINABILITY USING GRAD-CAM")
print("=" * 70)


# ============================================================
# 1. PATHS
# ============================================================

MODEL_PATH = (
    "data/processed/deep_learning/cnn/"
    "cnn_model.keras"
)

IMAGE_PATH = (
    r"C:\Users\radha\OneDrive\Desktop\image 1.jpg"
)

OUTPUT_DIR = (
    "data/processed/deep_learning/cnn/"
    "explainability"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. LOAD MODEL
# ============================================================

print("\nLoading trained CNN model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("✓ CNN model loaded successfully.")


# ============================================================
# 3. LOAD IMAGE
# ============================================================

print("\nLoading X-ray image...")

image = Image.open(
    IMAGE_PATH
)

print(
    "Original image size:",
    image.size
)

print(
    "Original image mode:",
    image.mode
)


# ============================================================
# 4. PREPROCESS IMAGE
# ============================================================

image_rgb = image.convert(
    "RGB"
)

image_resized = image_rgb.resize(
    (128, 128)
)

image_array = np.array(
    image_resized,
    dtype=np.float32
)

image_normalized = (
    image_array / 255.0
)

model_input = np.expand_dims(
    image_normalized,
    axis=0
)

print(
    "\nModel input shape:",
    model_input.shape
)


# ============================================================
# 5. BUILD MODEL
# ============================================================

# Make sure the Sequential model has been called.

_ = model(
    model_input,
    training=False
)


# ============================================================
# 6. CNN PREDICTION
# ============================================================

print("\nGenerating CNN prediction...")

prediction_output = model(
    model_input,
    training=False
)

prediction_probability = float(
    prediction_output[0][0]
)

if prediction_probability >= 0.50:

    prediction = "PNEUMONIA"

else:

    prediction = "NORMAL"


print("\n" + "=" * 70)
print("CNN PREDICTION")
print("=" * 70)

print(
    "\nPrediction:",
    prediction
)

print(
    "Pneumonia Probability:",
    f"{prediction_probability * 100:.2f}%"
)

print(
    "Normal Probability:",
    f"{(1 - prediction_probability) * 100:.2f}%"
)


# ============================================================
# 7. FIND LAST CONVOLUTIONAL LAYER
# ============================================================

print("\n" + "=" * 70)
print("FINDING CONVOLUTIONAL LAYER")
print("=" * 70)

last_conv_layer = None
last_conv_index = None

for index in range(
    len(model.layers) - 1,
    -1,
    -1
):

    layer = model.layers[index]

    if isinstance(
        layer,
        tf.keras.layers.Conv2D
    ):

        last_conv_layer = layer
        last_conv_index = index

        break


if last_conv_layer is None:

    raise ValueError(
        "No Conv2D layer found in CNN."
    )


print(
    "\nLast convolutional layer:",
    last_conv_layer.name
)

print(
    "Layer index:",
    last_conv_index
)


# ============================================================
# 8. DEBUG LAYER INFORMATION
# ============================================================

print("\nChecking CNN layers...")

for index, layer in enumerate(
    model.layers
):

    print(
        index,
        layer.name,
        type(layer).__name__
    )


# ============================================================
# 9. GRAD-CAM
# ============================================================

print("\nCalculating Grad-CAM...")


# ------------------------------------------------------------
# IMPORTANT:
#
# Instead of creating a second Functional model using
# model.output, we manually pass the image through the
# layers and keep the convolution output.
# ------------------------------------------------------------

with tf.GradientTape() as tape:

    x = tf.convert_to_tensor(
        model_input,
        dtype=tf.float32
    )

    conv_output = None

    # ----------------------------------------
    # Pass through CNN layers
    # ----------------------------------------

    for index, layer in enumerate(
        model.layers
    ):

        x = layer(
            x,
            training=False
        )

        if index == last_conv_index:

            conv_output = x

    final_output = x

    # ----------------------------------------
    # Target probability
    # ----------------------------------------

    class_probability = final_output[:, 0]


# ============================================================
# 10. GET GRADIENT
# ============================================================

grads = tape.gradient(
    class_probability,
    conv_output
)


if grads is None:

    raise RuntimeError(
        "Grad-CAM gradient is None. "
        "The CNN architecture does not expose "
        "a usable gradient path for this implementation."
    )


print(
    "✓ Gradient calculated successfully."
)


# ============================================================
# 11. GLOBAL AVERAGE POOLING
# ============================================================

pooled_grads = tf.reduce_mean(
    grads,
    axis=(0, 1, 2)
)


# ============================================================
# 12. CREATE HEATMAP
# ============================================================

conv_output = conv_output[0]

heatmap = tf.reduce_sum(
    conv_output
    * pooled_grads,
    axis=-1
)

heatmap = tf.maximum(
    heatmap,
    0
)

max_value = tf.reduce_max(
    heatmap
)

if float(max_value) > 0:

    heatmap = (
        heatmap / max_value
    )

heatmap = heatmap.numpy()


# ============================================================
# 13. RESIZE HEATMAP
# ============================================================

heatmap_image = Image.fromarray(
    np.uint8(
        heatmap * 255
    )
)

heatmap_image = heatmap_image.resize(
    image_rgb.size
)

heatmap_array = (
    np.array(
        heatmap_image
    ) / 255.0
)


# ============================================================
# 14. CREATE VISUALIZATION
# ============================================================

print(
    "\nCreating Grad-CAM visualization..."
)

plt.figure(
    figsize=(12, 5)
)


# ------------------------------------------------------------
# ORIGINAL
# ------------------------------------------------------------

plt.subplot(
    1,
    2,
    1
)

plt.imshow(
    image_rgb,
    cmap="gray"
)

plt.title(
    "Original X-ray"
)

plt.axis(
    "off"
)


# ------------------------------------------------------------
# GRAD-CAM
# ------------------------------------------------------------

plt.subplot(
    1,
    2,
    2
)

plt.imshow(
    image_rgb,
    cmap="gray"
)

plt.imshow(
    heatmap_array,
    cmap="jet",
    alpha=0.45
)

plt.title(
    "CNN Grad-CAM Explanation"
)

plt.axis(
    "off"
)

plt.tight_layout()


# ============================================================
# 15. SAVE COMBINED IMAGE
# ============================================================

output_path = os.path.join(
    OUTPUT_DIR,
    "xray_gradcam_explanation.png"
)

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(
    "\nGrad-CAM explanation saved:"
)

print(
    output_path
)


# ============================================================
# 16. SAVE HEATMAP
# ============================================================

heatmap_path = os.path.join(
    OUTPUT_DIR,
    "xray_gradcam_heatmap.png"
)

plt.figure(
    figsize=(6, 6)
)

plt.imshow(
    heatmap,
    cmap="jet"
)

plt.title(
    "Grad-CAM Heatmap"
)

plt.axis(
    "off"
)

plt.tight_layout()

plt.savefig(
    heatmap_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(
    "Heatmap saved:"
)

print(
    heatmap_path
)


# ============================================================
# 17. FINAL
# ============================================================

print("\n" + "=" * 70)
print("CNN EXPLAINABILITY COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    "\nCNN prediction:",
    prediction
)

print(
    "Pneumonia probability:",
    f"{prediction_probability * 100:.2f}%"
)

print(
    "\nGrad-CAM highlights image regions "
    "that influenced the CNN prediction."
)

print(
    "\nNote: Grad-CAM explains model behavior; "
    "it is not a medical diagnosis."
)

print("\n" + "=" * 70)