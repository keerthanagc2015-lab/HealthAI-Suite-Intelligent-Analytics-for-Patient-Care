from datasets import load_dataset
import os

print("=" * 70)
print("HEALTHAI - HOSPITAL DETERIORATION DATASET")
print("=" * 70)

# ------------------------------------------------------------
# 1. Create output directory
# ------------------------------------------------------------

output_dir = "data/raw/hospital_deterioration"

os.makedirs(output_dir, exist_ok=True)

print("\nLoading dataset from Hugging Face...")
print("Dataset: tarekmasryo/hospital-deterioration-dataset")

# ------------------------------------------------------------
# 2. Download dataset
# ------------------------------------------------------------

dataset = load_dataset(
    "tarekmasryo/hospital-deterioration-dataset"
)

print("\nDataset loaded successfully!")

print(dataset)

# ------------------------------------------------------------
# 3. Display basic information
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

train_data = dataset["train"]

print("\nNumber of rows:", len(train_data))

print("\nColumns:")
print(train_data.column_names)

# ------------------------------------------------------------
# 4. Convert a small portion to pandas
# ------------------------------------------------------------

print("\nConverting sample to pandas...")

sample_df = train_data.select(range(min(10000, len(train_data)))).to_pandas()

print("\nSample shape:")
print(sample_df.shape)

print("\nFirst 5 rows:")
print(sample_df.head())

# ------------------------------------------------------------
# 5. Display data types
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(sample_df.dtypes)

# ------------------------------------------------------------
# 6. Check patients
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PATIENT / SEQUENCE INFORMATION")
print("=" * 70)

unique_patients = sample_df["patient_id"].nunique()

print("\nUnique patients in sample:", unique_patients)

print("\nRecords per patient:")
print(
    sample_df.groupby("patient_id")
    .size()
    .describe()
)

# ------------------------------------------------------------
# 7. Check sequence length
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SEQUENCE LENGTH")
print("=" * 70)

sequence_lengths = (
    sample_df.groupby("patient_id")
    .size()
)

print("\nMinimum sequence length:", sequence_lengths.min())
print("Maximum sequence length:", sequence_lengths.max())
print("Average sequence length:", round(sequence_lengths.mean(), 2))

# ------------------------------------------------------------
# 8. Target distribution
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

print(
    sample_df["deterioration_next_12h"]
    .value_counts()
    .sort_index()
)

print("\nTarget percentages:")

print(
    sample_df["deterioration_next_12h"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)

# ------------------------------------------------------------
# 9. Missing values
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = sample_df.isnull().sum()

print(
    missing[missing > 0]
    .sort_values(ascending=False)
)

# ------------------------------------------------------------
# 10. Save sample for inspection
# ------------------------------------------------------------

sample_path = os.path.join(
    output_dir,
    "hospital_deterioration_sample.csv"
)

sample_df.to_csv(
    sample_path,
    index=False
)

print("\n" + "=" * 70)
print("FILE CREATED")
print("=" * 70)

print("\nSample:")
print(sample_path)

print("\n" + "=" * 70)
print("DATASET INSPECTION COMPLETED")
print("=" * 70)