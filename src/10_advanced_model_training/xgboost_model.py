# ==========================================================
# Import Libraries
# ==========================================================

from xgboost import XGBClassifier


# ==========================================================
# Train XGBoost
# ==========================================================

def train_xgboost(

        X_train,
        y_train,
        X_test

):

    print("\n" + "=" * 70)
    print("XGBoost")
    print("=" * 70)

    model = XGBClassifier(

        objective="multi:softmax",

        num_class=len(set(y_train)),

        n_estimators=100,

        random_state=42,

        eval_metric="mlogloss"

    )

    model.fit(

        X_train,
        y_train

    )

    print("\nXGBoost model trained successfully.")

    y_pred = model.predict(

        X_test

    )

    return model, y_pred