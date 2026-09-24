import os
import pandas as pd


# ==========================================================
# CONFIGURATION
# ==========================================================

DATA_PATH = (
    "data/processed/associative_learning/"
    "association_rules.csv"
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
# 1. LOAD ASSOCIATION RULES
# ==========================================================

print("=" * 70)
print("ASSOCIATIVE LEARNING - RULE FILTERING")
print("=" * 70)

rules = pd.read_csv(DATA_PATH)

print("\nAssociation Rules Loaded")
print("Total Rules:", len(rules))


# ==========================================================
# 2. BASIC QUALITY FILTER
# ==========================================================

filtered_rules = rules[
    (rules["support"] >= MIN_SUPPORT)
    &
    (rules["confidence"] >= MIN_CONFIDENCE)
    &
    (rules["lift"] >= MIN_LIFT)
].copy()

print("\nAfter Support / Confidence / Lift Filtering:")
print("Rules:", len(filtered_rules))


# ==========================================================
# 3. REMOVE IMAGING-ADMINISTRATIVE RULES
# ==========================================================

def contains_imaging_admin_rule(row):

    text = (
        str(row["Antecedent"])
        + " "
        + str(row["Consequent"])
    )

    return (
        "Imaging_Not_Available" in text
        or "No imaging performed" in text
    )


before_imaging_filter = len(
    filtered_rules
)

filtered_rules = filtered_rules[
    ~filtered_rules.apply(
        contains_imaging_admin_rule,
        axis=1
    )
].copy()

print(
    "\nAfter removing imaging administrative rules:"
)

print(
    "Removed:",
    before_imaging_filter - len(filtered_rules)
)

print(
    "Remaining:",
    len(filtered_rules)
)


# ==========================================================
# 4. REMOVE RULES WITH SAME FEATURE GROUP
# ==========================================================

feature_prefixes = [
    "Symptoms=",
    "Primary_Diagnosis=",
    "Treatment_Type=",
    "Treatment_Outcome=",
    "Imaging_Type=",
    "Imaging_Findings="
]


def get_feature_groups(text):

    groups = []

    for prefix in feature_prefixes:

        if prefix in str(text):

            groups.append(
                prefix.replace("=", "")
            )

    return groups


def has_same_feature_rule(row):

    antecedent_groups = get_feature_groups(
        row["Antecedent"]
    )

    consequent_groups = get_feature_groups(
        row["Consequent"]
    )

    return bool(
        set(antecedent_groups)
        &
        set(consequent_groups)
    )


before_feature_filter = len(
    filtered_rules
)

filtered_rules = filtered_rules[
    ~filtered_rules.apply(
        has_same_feature_rule,
        axis=1
    )
].copy()

print(
    "\nAfter removing same-feature rules:"
)

print(
    "Removed:",
    before_feature_filter - len(filtered_rules)
)

print(
    "Remaining:",
    len(filtered_rules)
)


# ==========================================================
# 5. SORT BY LIFT
# ==========================================================

filtered_rules = (
    filtered_rules
    .sort_values(
        by=[
            "lift",
            "confidence",
            "support"
        ],
        ascending=False
    )
    .reset_index(drop=True)
)


# ==========================================================
# 6. DISPLAY TOP MEANINGFUL RULES
# ==========================================================

print("\n" + "=" * 70)
print("TOP MEANINGFUL ASSOCIATION RULES")
print("=" * 70)

display_columns = [
    "Antecedent",
    "Consequent",
    "support",
    "confidence",
    "lift"
]

print(
    filtered_rules[
        display_columns
    ]
    .head(20)
    .to_string(index=False)
)


# ==========================================================
# 7. SAVE FILTERED RULES
# ==========================================================

output_path = os.path.join(
    OUTPUT_DIR,
    "meaningful_association_rules.csv"
)

filtered_rules.to_csv(
    output_path,
    index=False
)


# ==========================================================
# 8. SAVE SUMMARY
# ==========================================================

summary = pd.DataFrame({

    "Metric": [
        "Original Rules",
        "After Support/Confidence/Lift",
        "After Imaging Filter",
        "After Same-Feature Filter",
        "Final Meaningful Rules"
    ],

    "Value": [
        len(rules),
        len(
            rules[
                (rules["support"] >= MIN_SUPPORT)
                &
                (rules["confidence"] >= MIN_CONFIDENCE)
                &
                (rules["lift"] >= MIN_LIFT)
            ]
        ),
        before_feature_filter,
        len(filtered_rules),
        len(filtered_rules)
    ]
})


summary_path = os.path.join(
    OUTPUT_DIR,
    "rule_filtering_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)


# ==========================================================
# 9. COMPLETION
# ==========================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    "\nMeaningful Rules:",
    output_path
)

print(
    "Filtering Summary:",
    summary_path
)

print(
    "\nRule filtering completed successfully."
)