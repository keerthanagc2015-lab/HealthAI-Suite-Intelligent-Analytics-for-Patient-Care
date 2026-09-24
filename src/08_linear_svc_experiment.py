# ============================================================
# HealthAI Suite - LinearSVC Model
# ============================================================

from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from data_loader import load_data
from data_cleaning import clean_data
from feature_engineering import (
    prepare_features,
    split_data,
    preprocess_features
)


# ============================================================
# 1. TRAIN LINEAR SVC
# ============================================================

def train_linear_svc(X_train, y_train):

    print("\n" + "=" * 70)
    print("LinearSVC - Without SMOTE")
    print("=" * 70)

    model = LinearSVC(
        C=1.0,
        max_iter=5000,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    print("\nLinearSVC model trained successfully.")

    return model


# ============================================================
# 2. EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    print("\n" + "=" * 70)
    print("LinearSVC - Without SMOTE - Evaluation")
    print("=" * 70)


    # --------------------------------------------------------
    # Make predictions
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    macro_precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )


    # --------------------------------------------------------
    # Print summary metrics
    # --------------------------------------------------------

    print("\nSummary Metrics:")

    print(
        f"Accuracy        : {accuracy:.4f}"
    )

    print(
        f"Macro Precision : {macro_precision:.4f}"
    )

    print(
        f"Macro Recall    : {macro_recall:.4f}"
    )

    print(
        f"Macro F1        : {macro_f1:.4f}"
    )

    print(
        f"Weighted F1     : {weighted_f1:.4f}"
    )


    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )


    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )


# ============================================================
# 3. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    dataframe = load_data()


    # --------------------------------------------------------
    # Clean data
    # --------------------------------------------------------

    dataframe = clean_data(
        dataframe
    )


    # --------------------------------------------------------
    # Prepare features and target
    # --------------------------------------------------------

    X, y = prepare_features(
        dataframe
    )


    # --------------------------------------------------------
    # Train-test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )


    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    X_train_processed, X_test_processed, preprocessor = (
        preprocess_features(
            X_train,
            X_test
        )
    )


    # --------------------------------------------------------
    # Train LinearSVC
    # --------------------------------------------------------

    linear_svc_model = train_linear_svc(
        X_train_processed,
        y_train
    )


    # --------------------------------------------------------
    # Evaluate LinearSVC
    # --------------------------------------------------------

    evaluate_model(
        linear_svc_model,
        X_test_processed,
        y_test
    )