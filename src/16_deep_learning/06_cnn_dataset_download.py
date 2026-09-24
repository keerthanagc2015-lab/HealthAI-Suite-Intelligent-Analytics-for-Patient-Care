from datasets import load_dataset

print("=" * 70)
print("DOWNLOADING SYNTHETIC CHEST X-RAY DATASET")
print("=" * 70)

dataset = load_dataset(
    "chimbiwide/synthetic-chest-xray-pneumonia"
)

print("\nDataset loaded successfully!")
print(dataset)

for split in dataset:
    print(f"\n{split} shape:")
    print(dataset[split].num_rows)

print("\nDataset columns:")
print(dataset["train"].column_names)

print("\nFirst sample:")
print(dataset["train"][0])