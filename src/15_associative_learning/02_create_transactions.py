import os
import pandas as pd


# ==========================================================
# Configuration
# ==========================================================

DATA_PATH = "data/raw/Indian_healthcare.csv"

OUTPUT_DIR = "data/processed/associative_learning"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# 1. Load Dataset
# ==========================================================

print("=" * 70)
print("ASSOCIATIVE LEARNING - TRANSACTION CREATION")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("\nDataset Loaded Successfully")
print("Dataset Shape:", df.shape)


# ==========================================================
# 2. Association Features
# ==========================================================

association_features = [
    "Symptoms",
    "Primary_Diagnosis",
    "Treatment_Type",
    "Treatment_Outcome",
    "Imaging_Type",
    "Imaging_Findings"
]

print("\nAssociation Features:")

for feature in association_features:
    print(feature)


# ==========================================================
# 3. Handle Missing Imaging Type
# ==========================================================

df["Imaging_Type"] = (
    df["Imaging_Type"]
    .fillna("Imaging_Not_Available")
)

print(
    "\nMissing Imaging_Type values after handling:",
    df["Imaging_Type"].isnull().sum()
)


# ==========================================================
# 4. Convert Values into Items
# ==========================================================

transactions = []

for _, row in df[association_features].iterrows():

    transaction = []

    for feature in association_features:

        item = (
            f"{feature}="
            f"{row[feature]}"
        )

        transaction.append(item)

    transactions.append(transaction)


# ==========================================================
# 5. Display Example Transactions
# ==========================================================

print("\n" + "=" * 70)
print("SAMPLE TRANSACTIONS")
print("=" * 70)

for i in range(5):

    print(
        f"\nTransaction {i + 1}:"
    )

    print(
        transactions[i]
    )


# ==========================================================
# 6. Check Number of Transactions
# ==========================================================

print("\n" + "=" * 70)
print("TRANSACTION INFORMATION")
print("=" * 70)

print(
    "\nNumber of Transactions:",
    len(transactions)
)

print(
    "Expected:",
    len(df)
)


# ==========================================================
# 7. Save Transactions
# ==========================================================

transaction_df = pd.DataFrame({

    "Transaction_ID": range(
        1,
        len(transactions) + 1
    ),

    "Items": [
        " | ".join(transaction)
        for transaction in transactions
    ]

})


output_path = os.path.join(
    OUTPUT_DIR,
    "association_transactions.csv"
)

transaction_df.to_csv(
    output_path,
    index=False
)


# ==========================================================
# 8. Completion
# ==========================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    "\nTransaction Dataset:",
    output_path
)

print(
    "\nTransaction creation completed successfully."
)