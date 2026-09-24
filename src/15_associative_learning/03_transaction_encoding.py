import os
import pandas as pd


# ==========================================================
# CONFIGURATION
# ==========================================================

DATA_PATH = (
    "data/processed/associative_learning/"
    "association_transactions.csv"
)

OUTPUT_DIR = (
    "data/processed/associative_learning"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================================
# 1. LOAD TRANSACTIONS
# ==========================================================

print("=" * 70)
print("ASSOCIATIVE LEARNING - TRANSACTION ENCODING")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("\nTransaction Dataset Loaded")
print("Dataset Shape:", df.shape)


# ==========================================================
# 2. CONVERT ITEMS INTO LISTS
# ==========================================================

transactions = (
    df["Items"]
    .str.split(" | ", regex=False)
)

print("\nTransactions converted into item lists.")


# ==========================================================
# 3. FIND ALL UNIQUE ITEMS
# ==========================================================

unique_items = sorted(
    set(
        item
        for transaction in transactions
        for item in transaction
    )
)

print(
    "\nNumber of Unique Items:",
    len(unique_items)
)


# ==========================================================
# 4. CREATE ONE-HOT TRANSACTION MATRIX
# ==========================================================

print("\nCreating one-hot transaction matrix...")

encoded_data = pd.DataFrame(
    0,
    index=range(len(transactions)),
    columns=unique_items,
    dtype="int8"
)


# ==========================================================
# 5. MARK ITEMS PRESENT AS 1
# ==========================================================

print("Encoding transactions...")

for row_index, transaction in enumerate(transactions):

    for item in transaction:

        encoded_data.loc[
            row_index,
            item
        ] = 1


print("Encoding completed.")


# ==========================================================
# 6. DISPLAY DATASET SHAPE
# ==========================================================

print("\n" + "=" * 70)
print("ENCODED TRANSACTION MATRIX")
print("=" * 70)

print(
    "\nEncoded Dataset Shape:",
    encoded_data.shape
)


# ==========================================================
# 7. DISPLAY ONLY FIRST 5 ROWS
# ==========================================================

print("\nFirst 5 Encoded Transactions:")

print(
    encoded_data.head().to_string()
)


# ==========================================================
# 8. CHECK UNIQUE VALUES
# ==========================================================

unique_encoded_values = sorted(
    encoded_data.stack().unique()
)

print(
    "\nUnique Values in Encoded Dataset:",
    unique_encoded_values
)


# ==========================================================
# 9. VERIFY NUMBER OF TRANSACTIONS
# ==========================================================

print(
    "\nNumber of Transactions:",
    len(encoded_data)
)

print(
    "Expected Transactions:",
    len(df)
)


# ==========================================================
# 10. SAVE ENCODED DATASET
# ==========================================================

output_path = os.path.join(
    OUTPUT_DIR,
    "association_transaction_matrix.csv"
)

encoded_data.to_csv(
    output_path,
    index=False
)


# ==========================================================
# 11. FINAL MESSAGE
# ==========================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    "\nEncoded Transaction Matrix:"
)

print(output_path)

print(
    "\nTransaction encoding completed successfully."
)