# ==========================================================
# Import Libraries
# ==========================================================

from sklearn.ensemble import RandomForestClassifier


# ==========================================================
# Train Random Forest
# ==========================================================

def train_random_forest(
        X_train,
        y_train,
        X_test
):

    print("\n" + "=" * 70)
    print("Random Forest")
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

    y_pred = model.predict(

        X_test

    )

    return model, y_pred