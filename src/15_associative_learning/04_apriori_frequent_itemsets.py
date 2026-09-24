import os
import pandas as pd
from mlxtend.frequent_patterns import apriori


# ==========================================================
# CONFIGURATION
# ==========================================================

DATA_PATH = (
    "data/processed/associative_learning/"
    "association_transaction_matrix.csv"
)

OUTPUT_DIR = (
    "data/processed/associative_learning"
)

MIN_SUPPORT = 0.01

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================================
# 1. LOAD TRANSACTION MATRIX
# ==========================================================

print("=" * 70)
print("ASSOCIATIVE LEARNING - APRIORI")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("\nTransaction Matrix Loaded")
print("Dataset Shape:", df.shape)


# ==========================================================
# 2. CONVERT TO BOOLEAN
# ==========================================================

df = df.astype(bool)

print("\nTransaction matrix converted to boolean format.")


# ==========================================================
# 3. RUN APRIORI
# ==========================================================

print("\n" + "=" * 70)
print("RUNNING APRIORI")
print("=" * 70)

print("\nMinimum Support:", MIN_SUPPORT)

print(
    f"An itemset must appear in at least "
    f"{MIN_SUPPORT * 100:.1f}% of patients."
)

print(
    f"This means at least "
    f"{int(MIN_SUPPORT * len(df))} patients."
)

print("\nFinding frequent itemsets...")

frequent_itemsets = apriori(
    df,
    min_support=MIN_SUPPORT,
    use_colnames=True
)


# ==========================================================
# 4. DISPLAY RESULTS
# ==========================================================

print("\n" + "=" * 70)
print("FREQUENT ITEMSETS")
print("=" * 70)

print(
    "\nNumber of Frequent Itemsets:",
    len(frequent_itemsets)
)


# ==========================================================
# 5. ITEMSET SIZE
# ==========================================================

frequent_itemsets["Itemset_Size"] = (
    frequent_itemsets["itemsets"]
    .apply(len)
)


# ==========================================================
# 6. CREATE READABLE ITEMSET COLUMN
# ==========================================================

frequent_itemsets["Itemsets_Text"] = (
    frequent_itemsets["itemsets"]
    .apply(
        lambda x: " | ".join(
            sorted(x)
        )
    )
)


# ==========================================================
# 7. SORT BY SUPPORT
# ==========================================================

frequent_itemsets = (
    frequent_itemsets
    .sort_values(
        by="support",
        ascending=False
    )
    .reset_index(drop=True)
)


# ==========================================================
# 8. DISPLAY TOP 20
# ==========================================================

print("\nTop 20 Frequent Itemsets:")

print(
    frequent_itemsets[
        [
            "support",
            "Itemsets_Text",
            "Itemset_Size"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# ==========================================================
# 9. ITEMSET SIZE DISTRIBUTION
# ==========================================================

print("\n" + "=" * 70)
print("ITEMSET SIZE DISTRIBUTION")
print("=" * 70)

size_distribution = (
    frequent_itemsets["Itemset_Size"]
    .value_counts()
    .sort_index()
)

print(size_distribution)


# ==========================================================
# 10. SAVE ITEMSETS
# ==========================================================

output_path = os.path.join(
    OUTPUT_DIR,
    "frequent_itemsets.csv"
)

frequent_itemsets.to_csv(
    output_path,
    index=False
)


# ==========================================================
# 11. SAVE SUMMARY
# ==========================================================

summary = pd.DataFrame({

    "Metric": [
        "Number of Transactions",
        "Number of Unique Items",
        "Minimum Support",
        "Minimum Patient Count",
        "Number of Frequent Itemsets"
    ],

    "Value": [
        len(df),
        len(df.columns),
        MIN_SUPPORT,
        int(MIN_SUPPORT * len(df)),
        len(frequent_itemsets)
    ]
})


summary_path = os.path.join(
    OUTPUT_DIR,
    "apriori_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)


# ==========================================================
# 12. COMPLETION
# ==========================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    "\nFrequent Itemsets:",
    output_path
)

print(
    "Apriori Summary:",
    summary_path
)

print(
    "\nApriori analysis completed successfully."
)