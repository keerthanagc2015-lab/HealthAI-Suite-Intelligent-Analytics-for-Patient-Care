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
MIN_CONFIDENCE = 0.20
MIN_LIFT = 1.20

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================================
# 1. LOAD TRANSACTION MATRIX
# ==========================================================

print("=" * 70)
print("CLINICAL ASSOCIATIVE LEARNING")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("\nTransaction Matrix Loaded")
print("Original Shape:", df.shape)


# ==========================================================
# 2. SELECT CLINICAL FEATURES
# ==========================================================

clinical_prefixes = [
    "Symptoms=",
    "Primary_Diagnosis=",
    "Treatment_Type=",
    "Treatment_Outcome="
]


clinical_columns = [
    column
    for column in df.columns
    if any(
        column.startswith(prefix)
        for prefix in clinical_prefixes
    )
]


clinical_df = df[
    clinical_columns
].astype(bool)


print("\n" + "=" * 70)
print("CLINICAL FEATURES")
print("=" * 70)

print(
    "\nNumber of Clinical Items:",
    len(clinical_columns)
)

for column in clinical_columns:
    print(column)


print(
    "\nClinical Transaction Matrix Shape:",
    clinical_df.shape
)


# ==========================================================
# 3. RUN APRIORI
# ==========================================================

print("\n" + "=" * 70)
print("RUNNING APRIORI - CLINICAL DATA")
print("=" * 70)

print(
    "\nMinimum Support:",
    MIN_SUPPORT
)

print(
    "Minimum Patient Count:",
    int(
        MIN_SUPPORT * len(clinical_df)
    )
)


frequent_itemsets = apriori(
    clinical_df,
    min_support=MIN_SUPPORT,
    use_colnames=True
)


print(
    "\nFrequent Clinical Itemsets:",
    len(frequent_itemsets)
)


# ==========================================================
# 4. GENERATE ASSOCIATION RULES
# ==========================================================

print("\n" + "=" * 70)
print("GENERATING CLINICAL ASSOCIATION RULES")
print("=" * 70)

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.0
)


print(
    "\nTotal Rules Generated:",
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
].copy()


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
# 7. DIAGNOSTIC ANALYSIS
# ==========================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC ANALYSIS")
print("=" * 70)


diagnostic_rules = rules[
    rules["support"] >= MIN_SUPPORT
].copy()


diagnostic_rules = (
    diagnostic_rules
    .sort_values(
        by="lift",
        ascending=False
    )
    .reset_index(drop=True)
)


print(
    "\nRules with Support >= 1%:",
    len(diagnostic_rules)
)


print("\nTop 20 Clinical Rules by Lift:")

display_columns = [
    "Antecedent",
    "Consequent",
    "support",
    "confidence",
    "lift"
]


print(
    diagnostic_rules[
        display_columns
    ]
    .head(20)
    .to_string(index=False)
)


# ==========================================================
# 8. APPLY FINAL FILTERS
# ==========================================================

print("\n" + "=" * 70)
print("FINAL CLINICAL RULE FILTERING")
print("=" * 70)


clinical_rules = diagnostic_rules[
    (diagnostic_rules["support"] >= MIN_SUPPORT)
    &
    (diagnostic_rules["confidence"] >= MIN_CONFIDENCE)
    &
    (diagnostic_rules["lift"] >= MIN_LIFT)
].copy()


print(
    "\nMinimum Support:",
    MIN_SUPPORT
)

print(
    "Minimum Confidence:",
    MIN_CONFIDENCE
)

print(
    "Minimum Lift:",
    MIN_LIFT
)

print(
    "\nRules after filtering:",
    len(clinical_rules)
)


# ==========================================================
# 9. DISPLAY FINAL RULES
# ==========================================================

print("\n" + "=" * 70)
print("FINAL CLINICAL ASSOCIATION RULES")
print("=" * 70)


if len(clinical_rules) > 0:

    print(
        clinical_rules[
            display_columns
        ]
        .head(20)
        .to_string(index=False)
    )

else:

    print(
        "\nNo rules satisfied all three thresholds."
    )


# ==========================================================
# 10. ITEMSET SIZE ANALYSIS
# ==========================================================

frequent_itemsets["Itemset_Size"] = (
    frequent_itemsets["itemsets"]
    .apply(len)
)


frequent_itemsets["Itemsets_Text"] = (
    frequent_itemsets["itemsets"]
    .apply(
        lambda x: " | ".join(
            sorted(x)
        )
    )
)


print("\n" + "=" * 70)
print("CLINICAL ITEMSET SIZE DISTRIBUTION")
print("=" * 70)


print(
    frequent_itemsets[
        "Itemset_Size"
    ]
    .value_counts()
    .sort_index()
)


# ==========================================================
# 11. SAVE FREQUENT ITEMSETS
# ==========================================================

itemset_path = os.path.join(
    OUTPUT_DIR,
    "clinical_frequent_itemsets.csv"
)


frequent_itemsets.to_csv(
    itemset_path,
    index=False
)


# ==========================================================
# 12. SAVE ALL DIAGNOSTIC RULES
# ==========================================================

diagnostic_path = os.path.join(
    OUTPUT_DIR,
    "clinical_rule_diagnostics.csv"
)


diagnostic_rules.to_csv(
    diagnostic_path,
    index=False
)


# ==========================================================
# 13. SAVE FINAL CLINICAL RULES
# ==========================================================

rules_path = os.path.join(
    OUTPUT_DIR,
    "clinical_association_rules.csv"
)


clinical_rules.to_csv(
    rules_path,
    index=False
)


# ==========================================================
# 14. SAVE SUMMARY
# ==========================================================

summary = pd.DataFrame({

    "Metric": [
        "Patients",
        "Clinical Items",
        "Minimum Support",
        "Minimum Confidence",
        "Minimum Lift",
        "Frequent Clinical Itemsets",
        "Total Association Rules",
        "Rules with Support >= 1%",
        "Final Clinical Rules"
    ],

    "Value": [
        len(clinical_df),
        len(clinical_columns),
        MIN_SUPPORT,
        MIN_CONFIDENCE,
        MIN_LIFT,
        len(frequent_itemsets),
        len(rules),
        len(diagnostic_rules),
        len(clinical_rules)
    ]
})


summary_path = os.path.join(
    OUTPUT_DIR,
    "clinical_association_summary.csv"
)


summary.to_csv(
    summary_path,
    index=False
)


# ==========================================================
# 15. COMPLETION
# ==========================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    "\nClinical Frequent Itemsets:",
    itemset_path
)

print(
    "Clinical Rule Diagnostics:",
    diagnostic_path
)

print(
    "Clinical Association Rules:",
    rules_path
)

print(
    "Clinical Association Summary:",
    summary_path
)


print(
    "\nClinical associative learning completed successfully."
)