# ============================================================
# IMPORT LIBRARIES
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from data_loader import load_data
from data_cleaning import clean_data
from feature_engineering import (
    prepare_features,
    split_data,
    preprocess_features
)


# ============================================================
# 1. APPLY SMOTE
# ============================================================

def apply_smote(X_train, y_train):

    print("\n" + "=" * 70)
    print("Applying SMOTE")
    print("=" * 70)

    print("\nClass Distribution Before SMOTE:")
    print(y_train.value_counts())

    smote = SMOTE(
        random_state=42
    )

    # IMPORTANT:
    # SMOTE is applied ONLY to training data.
    # Test data remains untouched.
    X_train_balanced, y_train_balanced = smote.fit_resample(
        X_train,
        y_train
    )

    print("\nClass Distribution After SMOTE:")
    print(y_train_balanced.value_counts())

    print("\nTraining Shape Before SMOTE:")
    print(X_train.shape)

    print("\nTraining Shape After SMOTE:")
    print(X_train_balanced.shape)

    return X_train_balanced, y_train_balanced


# ============================================================
# 2. TRAIN LOGISTIC REGRESSION
# ============================================================

def train_logistic_regression(
    X_train,
    y_train,
    model_name
):

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    print("\nLogistic Regression model trained successfully.")

    return model


# ============================================================
# 3. TRAIN DECISION TREE
# ============================================================

def train_decision_tree(
    X_train,
    y_train,
    model_name
):

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    model = DecisionTreeClassifier(
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    print("\nDecision Tree model trained successfully.")

    return model


# ============================================================
# 4. TRAIN RANDOM FOREST
# ============================================================

def train_random_forest(
    X_train,
    y_train,
    model_name
):

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    print("\nRandom Forest model trained successfully.")

    return model


# ============================================================
# 5. TRAIN XGBOOST WITHOUT SMOTE
# ============================================================

def train_xgboost(
    X_train,
    y_train
):

    print("\n" + "=" * 70)
    print("XGBoost - Without SMOTE")
    print("=" * 70)


    # --------------------------------------------------------
    # Encode target labels
    # --------------------------------------------------------

    label_encoder = LabelEncoder()

    y_train_encoded = label_encoder.fit_transform(
        y_train
    )

    print("\nTarget classes encoded as:")

    for number, diagnosis in enumerate(
        label_encoder.classes_
    ):

        print(
            f"{number} -> {diagnosis}"
        )


    # --------------------------------------------------------
    # Create XGBoost
    # --------------------------------------------------------

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss"
    )


    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train_encoded
    )

    print("\nXGBoost model trained successfully.")

    return model, label_encoder


# ============================================================
# 6. TRAIN XGBOOST WITH SMOTE
# ============================================================

def train_xgboost_smote(
    X_train_balanced,
    y_train_balanced,
    label_encoder
):

    print("\n" + "=" * 70)
    print("XGBoost - With SMOTE")
    print("=" * 70)


    # --------------------------------------------------------
    # Use SAME target encoding
    # --------------------------------------------------------

    y_train_balanced_encoded = label_encoder.transform(
        y_train_balanced
    )


    print("\nUsing same target encoding:")

    for number, diagnosis in enumerate(
        label_encoder.classes_
    ):

        print(
            f"{number} -> {diagnosis}"
        )


    # --------------------------------------------------------
    # Create XGBoost
    # --------------------------------------------------------

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss"
    )


    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    model.fit(
        X_train_balanced,
        y_train_balanced_encoded
    )

    print("\nXGBoost + SMOTE model trained successfully.")

    return model


# ============================================================
# 7. TRAIN KNN
# ============================================================

def train_knn(
    X_train,
    y_train
):

    print("\n" + "=" * 70)
    print("KNN - Without SMOTE")
    print("=" * 70)


    model = KNeighborsClassifier(
        n_neighbors=5,
        weights="uniform",
        n_jobs=-1
    )


    model.fit(
        X_train,
        y_train
    )


    print("\nKNN model trained successfully.")

    return model


# ============================================================
# 8. TRAIN GAUSSIAN NAIVE BAYES
# ============================================================

def train_gaussian_naive_bayes(
    X_train,
    y_train
):

    print("\n" + "=" * 70)
    print("Gaussian Naive Bayes - Without SMOTE")
    print("=" * 70)


    # --------------------------------------------------------
    # Create Gaussian Naive Bayes model
    # --------------------------------------------------------

    model = GaussianNB()


    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )


    print(
        "\nGaussian Naive Bayes model trained successfully."
    )

    return model


# ============================================================
# 9. EVALUATE NORMAL SKLEARN MODELS
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name
):

    print("\n" + "=" * 70)
    print(f"{model_name} - Evaluation")
    print("=" * 70)


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    # --------------------------------------------------------
    # Macro Precision
    # --------------------------------------------------------

    macro_precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )


    # --------------------------------------------------------
    # Macro Recall
    # --------------------------------------------------------

    macro_recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )


    # --------------------------------------------------------
    # Macro F1
    # --------------------------------------------------------

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )


    # --------------------------------------------------------
    # Weighted F1
    # --------------------------------------------------------

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )


    # --------------------------------------------------------
    # Summary
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


    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {

        "Model": model_name,

        "Accuracy": accuracy,

        "Macro Precision": macro_precision,

        "Macro Recall": macro_recall,

        "Macro F1": macro_f1,

        "Weighted F1": weighted_f1
    }


# ============================================================
# 10. EVALUATE XGBOOST
# ============================================================

def evaluate_xgboost(
    model,
    label_encoder,
    X_test,
    y_test,
    model_name
):

    print("\n" + "=" * 70)
    print(f"{model_name} - Evaluation")
    print("=" * 70)


    # --------------------------------------------------------
    # Predict numeric class IDs
    # --------------------------------------------------------

    y_pred_encoded = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # Convert IDs back to diagnosis names
    # --------------------------------------------------------

    y_pred = label_encoder.inverse_transform(
        y_pred_encoded.astype(int)
    )


    # --------------------------------------------------------
    # Metrics
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
    # Summary
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


    # --------------------------------------------------------
    # Return Results
    # --------------------------------------------------------

    return {

        "Model": model_name,

        "Accuracy": accuracy,

        "Macro Precision": macro_precision,

        "Macro Recall": macro_recall,

        "Macro F1": macro_f1,

        "Weighted F1": weighted_f1
    }


# ============================================================
# 11. FINAL MODEL COMPARISON
# ============================================================

def print_comparison(results):

    print("\n" + "=" * 110)
    print("FINAL MODEL COMPARISON")
    print("=" * 110)

    print(
        f"{'Model':<42}"
        f"{'Accuracy':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'Macro F1':<12}"
        f"{'Weighted F1':<12}"
    )

    print("-" * 110)


    for result in results:

        print(
            f"{result['Model']:<42}"
            f"{result['Accuracy']:<12.4f}"
            f"{result['Macro Precision']:<12.4f}"
            f"{result['Macro Recall']:<12.4f}"
            f"{result['Macro F1']:<12.4f}"
            f"{result['Weighted F1']:<12.4f}"
        )


    print("-" * 110)

    print(
        "\nNote: Precision and Recall shown above are "
        "Macro Precision and Macro Recall."
    )


# ============================================================
# 12. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":


    # ========================================================
    # LOAD DATA
    # ========================================================

    dataframe = load_data()


    # ========================================================
    # CLEAN DATA
    # ========================================================

    dataframe = clean_data(
        dataframe
    )


    # ========================================================
    # PREPARE FEATURES AND TARGET
    # ========================================================

    X, y = prepare_features(
        dataframe
    )


    # ========================================================
    # TRAIN-TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )


    # ========================================================
    # FEATURE PREPROCESSING
    # ========================================================

    X_train_processed, X_test_processed, preprocessor = (
        preprocess_features(
            X_train,
            X_test
        )
    )


    # ========================================================
    # STORE MODEL RESULTS
    # ========================================================

    model_results = []


    # ========================================================
    # MODEL 1
    # LOGISTIC REGRESSION WITHOUT SMOTE
    # ========================================================

    logistic_model = train_logistic_regression(
        X_train_processed,
        y_train,
        "Logistic Regression - Without SMOTE"
    )

    logistic_results = evaluate_model(
        logistic_model,
        X_test_processed,
        y_test,
        "Logistic Regression - Without SMOTE"
    )

    model_results.append(
        logistic_results
    )


    # ========================================================
    # APPLY SMOTE ONLY ONCE
    # ========================================================

    X_train_balanced, y_train_balanced = apply_smote(
        X_train_processed,
        y_train
    )


    # ========================================================
    # MODEL 2
    # LOGISTIC REGRESSION WITH SMOTE
    # ========================================================

    logistic_smote_model = train_logistic_regression(
        X_train_balanced,
        y_train_balanced,
        "Logistic Regression - With SMOTE"
    )

    logistic_smote_results = evaluate_model(
        logistic_smote_model,
        X_test_processed,
        y_test,
        "Logistic Regression - With SMOTE"
    )

    model_results.append(
        logistic_smote_results
    )


    # ========================================================
    # MODEL 3
    # DECISION TREE WITHOUT SMOTE
    # ========================================================

    decision_tree_model = train_decision_tree(
        X_train_processed,
        y_train,
        "Decision Tree - Without SMOTE"
    )

    decision_tree_results = evaluate_model(
        decision_tree_model,
        X_test_processed,
        y_test,
        "Decision Tree - Without SMOTE"
    )

    model_results.append(
        decision_tree_results
    )


    # ========================================================
    # MODEL 4
    # DECISION TREE WITH SMOTE
    # ========================================================

    decision_tree_smote_model = train_decision_tree(
        X_train_balanced,
        y_train_balanced,
        "Decision Tree - With SMOTE"
    )

    decision_tree_smote_results = evaluate_model(
        decision_tree_smote_model,
        X_test_processed,
        y_test,
        "Decision Tree - With SMOTE"
    )

    model_results.append(
        decision_tree_smote_results
    )


    # ========================================================
    # MODEL 5
    # RANDOM FOREST WITHOUT SMOTE
    # ========================================================

    random_forest_model = train_random_forest(
        X_train_processed,
        y_train,
        "Random Forest - Without SMOTE"
    )

    random_forest_results = evaluate_model(
        random_forest_model,
        X_test_processed,
        y_test,
        "Random Forest - Without SMOTE"
    )

    model_results.append(
        random_forest_results
    )


    # ========================================================
    # MODEL 6
    # RANDOM FOREST WITH SMOTE
    # ========================================================

    random_forest_smote_model = train_random_forest(
        X_train_balanced,
        y_train_balanced,
        "Random Forest - With SMOTE"
    )

    random_forest_smote_results = evaluate_model(
        random_forest_smote_model,
        X_test_processed,
        y_test,
        "Random Forest - With SMOTE"
    )

    model_results.append(
        random_forest_smote_results
    )


    # ========================================================
    # MODEL 7
    # XGBOOST WITHOUT SMOTE
    # ========================================================

    xgboost_model, label_encoder = train_xgboost(
        X_train_processed,
        y_train
    )

    xgboost_results = evaluate_xgboost(
        xgboost_model,
        label_encoder,
        X_test_processed,
        y_test,
        "XGBoost - Without SMOTE"
    )

    model_results.append(
        xgboost_results
    )


    # ========================================================
    # MODEL 8
    # XGBOOST WITH SMOTE
    # ========================================================

    xgboost_smote_model = train_xgboost_smote(
        X_train_balanced,
        y_train_balanced,
        label_encoder
    )

    xgboost_smote_results = evaluate_xgboost(
        xgboost_smote_model,
        label_encoder,
        X_test_processed,
        y_test,
        "XGBoost - With SMOTE"
    )

    model_results.append(
        xgboost_smote_results
    )


    # ========================================================
    # MODEL 9
    # KNN WITHOUT SMOTE
    # ========================================================

    knn_model = train_knn(
        X_train_processed,
        y_train
    )

    knn_results = evaluate_model(
        knn_model,
        X_test_processed,
        y_test,
        "KNN - Without SMOTE"
    )

    model_results.append(
        knn_results
    )


    # ========================================================
    # MODEL 10
    # GAUSSIAN NAIVE BAYES WITHOUT SMOTE
    # ========================================================

    naive_bayes_model = train_gaussian_naive_bayes(
        X_train_processed,
        y_train
    )

    naive_bayes_results = evaluate_model(
        naive_bayes_model,
        X_test_processed,
        y_test,
        "Gaussian Naive Bayes - Without SMOTE"
    )

    model_results.append(
        naive_bayes_results
    )


    # ========================================================
    # FINAL MODEL COMPARISON
    # ========================================================

    print_comparison(
        model_results
    )