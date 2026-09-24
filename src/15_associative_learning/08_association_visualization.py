import os
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# CONFIGURATION
# ==========================================================

DATA_DIR = (
    "data/processed/associative_learning"
)

OUTPUT_DIR = (
    "data/processed/associative_learning/"
    "visualizations"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================================
# 1. LOAD DATA
# ==========================================================

print("=" * 70)
print("ASSOCIATIVE LEARNING - VISUALIZATION")
print("=" * 70)


frequent_itemsets = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "clinical_frequent_itemsets.csv"
    )
)

diagnostic_rules = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "clinical_rule_diagnostics.csv"
    )
)

summary = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "clinical_association_summary.csv"
    )
)


print("\nFiles loaded successfully.")

print(
    "Frequent Itemsets:",
    frequent_itemsets.shape
)

print(
    "Diagnostic Rules:",
    diagnostic_rules.shape
)


# ==========================================================
# 2. TOP FREQUENT CLINICAL ITEMS
# ==========================================================

print("\n" + "=" * 70)
print("TOP FREQUENT CLINICAL ITEMSETS")
print("=" * 70)


single_items = frequent_itemsets[
    frequent_itemsets["Itemset_Size"] == 1
].copy()


single_items = (
    single_items
    .sort_values(
        by="support",
        ascending=False
    )
    .head(15)
)


print(
    single_items[
        [
            "Itemsets_Text",
            "support"
        ]
    ]
    .to_string(index=False)
)


plt.figure(
    figsize=(12, 7)
)

plt.barh(
    single_items["Itemsets_Text"][::-1],
    single_items["support"][::-1]
)

plt.xlabel(
    "Support"
)

plt.ylabel(
    "Clinical Item"
)

plt.title(
    "Top Clinical Items by Support"
)

plt.tight_layout()


support_plot = os.path.join(
    OUTPUT_DIR,
    "top_clinical_items_support.png"
)

plt.savefig(
    support_plot,
    dpi=300
)

plt.close()


# ==========================================================
# 3. TOP CLINICAL RULES BY LIFT
# ==========================================================

print("\n" + "=" * 70)
print("TOP CLINICAL RULES BY LIFT")
print("=" * 70)


top_rules = (
    diagnostic_rules
    .sort_values(
        by="lift",
        ascending=False
    )
    .head(15)
    .copy()
)


top_rules["Rule"] = (
    top_rules["Antecedent"]
    + " → "
    + top_rules["Consequent"]
)


print(
    top_rules[
        [
            "Rule",
            "support",
            "confidence",
            "lift"
        ]
    ]
    .to_string(index=False)
)


plt.figure(
    figsize=(12, 8)
)

plt.barh(
    top_rules["Rule"][::-1],
    top_rules["lift"][::-1]
)

plt.xlabel(
    "Lift"
)

plt.ylabel(
    "Association Rule"
)

plt.title(
    "Top Clinical Association Rules by Lift"
)

plt.tight_layout()


lift_plot = os.path.join(
    OUTPUT_DIR,
    "top_clinical_rules_lift.png"
)

plt.savefig(
    lift_plot,
    dpi=300
)

plt.close()


# ==========================================================
# 4. SUPPORT VS CONFIDENCE
# ==========================================================

plt.figure(
    figsize=(10, 7)
)

plt.scatter(
    diagnostic_rules["support"],
    diagnostic_rules["confidence"]
)

plt.xlabel(
    "Support"
)

plt.ylabel(
    "Confidence"
)

plt.title(
    "Clinical Association Rules: Support vs Confidence"
)

plt.tight_layout()


scatter_plot = os.path.join(
    OUTPUT_DIR,
    "support_vs_confidence.png"
)

plt.savefig(
    scatter_plot,
    dpi=300
)

plt.close()


# ==========================================================
# 5. LIFT DISTRIBUTION
# ==========================================================

plt.figure(
    figsize=(10, 6)
)

plt.hist(
    diagnostic_rules["lift"],
    bins=20
)

plt.xlabel(
    "Lift"
)

plt.ylabel(
    "Number of Rules"
)

plt.title(
    "Distribution of Clinical Association Rule Lift"
)

plt.tight_layout()


lift_distribution_plot = os.path.join(
    OUTPUT_DIR,
    "clinical_lift_distribution.png"
)

plt.savefig(
    lift_distribution_plot,
    dpi=300
)

plt.close()


# ==========================================================
# 6. CREATE VISUALIZATION SUMMARY
# ==========================================================

visualization_summary = pd.DataFrame({

    "Visualization": [
        "Top Clinical Items by Support",
        "Top Clinical Rules by Lift",
        "Support vs Confidence",
        "Clinical Lift Distribution"
    ],

    "File": [
        support_plot,
        lift_plot,
        scatter_plot,
        lift_distribution_plot
    ]
})


summary_path = os.path.join(
    OUTPUT_DIR,
    "visualization_summary.csv"
)


visualization_summary.to_csv(
    summary_path,
    index=False
)


# ==========================================================
# 7. COMPLETION
# ==========================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    "\nTop Clinical Items:",
    support_plot
)

print(
    "Top Clinical Rules:",
    lift_plot
)

print(
    "Support vs Confidence:",
    scatter_plot
)

print(
    "Lift Distribution:",
    lift_distribution_plot
)

print(
    "Visualization Summary:",
    summary_path
)

print(
    "\nAssociative learning visualization completed successfully."
)