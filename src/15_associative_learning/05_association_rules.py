import os
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


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
MIN_CONFIDENCE = 0.10

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================================
# 1. LOAD TRANSACTION MATRIX
# ==========================================================

print("=" * 70)
print("ASSOCIATIVE LEARNING - ASSOCIATION RULES")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("\nTransaction Matrix Loaded")
print("Dataset Shape:", df.shape)


# ==========================================================
# 2. CONVERT TO BOOLEAN
# ==========================================================

df = df.astype(bool)

print("\nTransaction matrix converted to boolean.")


# ==========================================================
# 3. GENERATE FREQUENT ITEMSETS
# ==========================================================

print("\n" + "=" * 70)
print("GENERATING FREQUENT ITEMSETS")
print("=" * 70)

frequent_itemsets = apriori(
    df,
    min_support=MIN_SUPPORT,
    use_colnames=True
)

print(
    "\nFrequent Itemsets:",
    len(frequent_itemsets)
)


# ==========================================================
# 4. GENERATE ASSOCIATION RULES
# ==========================================================

print("\n" + "=" * 70)
print("GENERATING ASSOCIATION RULES")
print("=" * 70)

print(
    "\nMinimum Confidence:",
    MIN_CONFIDENCE
)

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=MIN_CONFIDENCE
)

print(
    "\nRules Generated:",
    len(rules)
)


# ==========================================================
# 5. SELECT IMPORTANT COLUMNS
# ==========================================================

rules = rules[
    [
        "antecedents",
        "consequents",
        "antecedent support",
        "consequent support",
        "support",
        "confidence",
        "lift"
    ]
]


# ==========================================================
# 6. CONVERT FROZEN SETS TO READABLE TEXT
# ==========================================================

rules["Antecedent"] = (
    rules["antecedents"]
    .apply(
        lambda x: " | ".join(
            sorted(x)
        )
    )
)

rules["Consequent"] = (
    rules["consequents"]
    .apply(
        lambda x: " | ".join(
            sorted(x)
        )
    )
)


# ==========================================================
# 7. FILTER POSITIVE ASSOCIATIONS
# ==========================================================

rules = rules[
    rules["lift"] > 1
]


print(
    "\nRules with Lift > 1:",
    len(rules)
)


# ==========================================================
# 8. SORT BY LIFT
# ==========================================================

rules = (
    rules
    .sort_values(
        by="lift",
        ascending=False
    )
    .reset_index(drop=True)
)


# ==========================================================
# 9. DISPLAY TOP RULES
# ==========================================================

print("\n" + "=" * 70)
print("TOP ASSOCIATION RULES")
print("=" * 70)

display_columns = [
    "Antecedent",
    "Consequent",
    "support",
    "confidence",
    "lift"
]

print(
    rules[
        display_columns
    ]
    .head(20)
    .to_string(index=False)
)


# ==========================================================
# 10. SAVE RULES
# ==========================================================

output_path = os.path.join(
    OUTPUT_DIR,
    "association_rules.csv"
)

rules.to_csv(
    output_path,
    index=False
)


# ==========================================================
# 11. SAVE SUMMARY
# ==========================================================

summary = pd.DataFrame({

    "Metric": [
        "Frequent Itemsets",
        "Rules Generated",
        "Rules with Lift > 1"
    ],

    "Value": [
        len(frequent_itemsets),
        len(
            association_rules(
                frequent_itemsets,
                metric="confidence",
                min_threshold=MIN_CONFIDENCE
            )
        ),
        len(rules)
    ]
})


summary_path = os.path.join(
    OUTPUT_DIR,
    "association_rules_summary.csv"
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
    "\nAssociation Rules:",
    output_path
)

print(
    "Association Rules Summary:",
    summary_path
)

print(
    "\nAssociation rule generation completed successfully."
)