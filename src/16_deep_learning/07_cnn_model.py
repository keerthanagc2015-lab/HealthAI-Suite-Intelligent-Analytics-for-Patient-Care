import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from datasets import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# ================================================================
# 1. REPRODUCIBILITY
# ================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ================================================================
# 2. CONFIGURATION
# ================================================================

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20

DATASET_NAME = "chimbiwide/synthetic-chest-xray-pneumonia"

OUTPUT_DIR = "data/processed/deep_learning/cnn"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


print("=" * 70)
print("HEALTHAI - CNN MODEL")
print("=" * 70)


# ================================================================
# 3. LOAD DATASET
# ================================================================

print("\nLoading dataset...")

dataset = load_dataset(DATASET_NAME)

print("\nDataset Loaded Successfully")
print(dataset)


# ================================================================
# 4. CHECK CLASS DISTRIBUTION
# ================================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

train_labels = np.array(
    dataset["train"]["label"]
)

test_labels = np.array(
    dataset["test"]["label"]
)

print(
    "Training:",
    np.bincount(train_labels)
)

print(
    "Testing:",
    np.bincount(test_labels)
)


# ================================================================
# 5. CONVERT IMAGES TO NUMPY ARRAYS
# ================================================================

print("\n" + "=" * 70)
print("IMAGE PREPROCESSING")
print("=" * 70)

print(
    f"\nResizing images to "
    f"{IMAGE_SIZE} x {IMAGE_SIZE}"
)

print("Normalizing pixel values to 0-1...")


def preprocess_images(dataset_split):

    images = []

    labels = []

    for example in dataset_split:

        image = example["image"]

        # Convert to RGB
        image = image.convert("RGB")

        # Resize
        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        # Convert to NumPy
        image = np.array(
            image,
            dtype=np.float32
        )

        # Normalize
        image = image / 255.0

        images.append(image)

        labels.append(
            example["label"]
        )

    return (
        np.array(images),
        np.array(labels)
    )


X_train_full, y_train_full = preprocess_images(
    dataset["train"]
)

X_test, y_test = preprocess_images(
    dataset["test"]
)


print(
    "\nTraining image shape:",
    X_train_full.shape
)

print(
    "Testing image shape:",
    X_test.shape
)


# ================================================================
# 6. TRAIN / VALIDATION SPLIT
# ================================================================

print("\n" + "=" * 70)
print("TRAIN / VALIDATION SPLIT")
print("=" * 70)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    random_state=SEED,
    stratify=y_train_full
)

print(
    "Training:",
    X_train.shape
)

print(
    "Validation:",
    X_val.shape
)

print(
    "Testing:",
    X_test.shape
)


# ================================================================
# 7. BUILD CNN
# ================================================================

print("\n" + "=" * 70)
print("BUILDING CNN")
print("=" * 70)


model = Sequential(
    [

        # Input
        Input(
            shape=(
                IMAGE_SIZE,
                IMAGE_SIZE,
                3
            )
        ),

        # First convolution block
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu"
        ),

        MaxPooling2D(
            pool_size=(2, 2)
        ),

        # Second convolution block
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation="relu"
        ),

        MaxPooling2D(
            pool_size=(2, 2)
        ),

        # Third convolution block
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation="relu"
        ),

        MaxPooling2D(
            pool_size=(2, 2)
        ),

        # Convert feature maps into vector
        Flatten(),

        # Fully connected layer
        Dense(
            128,
            activation="relu"
        ),

        Dropout(
            0.5
        ),

        # Binary classification output
        Dense(
            1,
            activation="sigmoid"
        )
    ]
)


# ================================================================
# 8. COMPILE MODEL
# ================================================================

model.compile(

    optimizer=Adam(
        learning_rate=0.001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ================================================================
# 9. MODEL SUMMARY
# ================================================================

model.summary()


# ================================================================
# 10. EARLY STOPPING
# ================================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=4,

    restore_best_weights=True
)


# ================================================================
# 11. TRAIN CNN
# ================================================================

print("\n" + "=" * 70)
print("TRAINING CNN")
print("=" * 70)

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


# ================================================================
# 12. TEST PREDICTIONS
# ================================================================

print("\n" + "=" * 70)
print("GENERATING TEST PREDICTIONS")
print("=" * 70)


probabilities = model.predict(
    X_test,
    verbose=0
).flatten()


# Convert probability to class
# >= 0.5 → Pneumonia
# < 0.5  → Normal

y_pred = (
    probabilities >= 0.5
).astype(int)


# ================================================================
# 13. EVALUATION
# ================================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print("\n" + "=" * 70)
print("CNN TEST RESULTS")
print("=" * 70)

print(
    f"\nAccuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)


# ================================================================
# 14. CONFUSION MATRIX
# ================================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "NORMAL",
            "PNEUMONIA"
        ],
        zero_division=0
    )
)


# ================================================================
# 15. SAVE METRICS
# ================================================================

metrics_df = pd.DataFrame(
    [
        {
            "Model": "CNN",
            "Image_Size": IMAGE_SIZE,
            "Batch_Size": BATCH_SIZE,
            "Epochs": len(
                history.history["loss"]
            ),
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }
    ]
)


metrics_path = os.path.join(
    OUTPUT_DIR,
    "cnn_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)


# ================================================================
# 16. SAVE CONFUSION MATRIX
# ================================================================

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual_Normal",
        "Actual_Pneumonia"
    ],
    columns=[
        "Predicted_Normal",
        "Predicted_Pneumonia"
    ]
)

cm_path = os.path.join(
    OUTPUT_DIR,
    "cnn_confusion_matrix.csv"
)

cm_df.to_csv(
    cm_path
)


# ================================================================
# 17. SAVE TRAINING HISTORY
# ================================================================

history_df = pd.DataFrame(
    history.history
)

history_df.insert(
    0,
    "Epoch",
    range(
        1,
        len(history_df) + 1
    )
)

history_path = os.path.join(
    OUTPUT_DIR,
    "cnn_training_history.csv"
)

history_df.to_csv(
    history_path,
    index=False
)


# ================================================================
# 18. SAVE MODEL
# ================================================================

model_path = os.path.join(
    OUTPUT_DIR,
    "cnn_model.keras"
)

model.save(
    model_path
)


# ================================================================
# 19. FINAL OUTPUT
# ================================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    "\nCNN Metrics:",
    metrics_path
)

print(
    "\nConfusion Matrix:",
    cm_path
)

print(
    "\nTraining History:",
    history_path
)

print(
    "\nCNN Model:",
    model_path
)

print("\n" + "=" * 70)
print("CNN TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)