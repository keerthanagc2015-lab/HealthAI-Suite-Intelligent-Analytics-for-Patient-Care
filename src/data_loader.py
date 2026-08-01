import pandas as pd

# Load Dataset
df = pd.read_csv("data/raw/medical_data.csv")

print("=" * 60)
print("🏥 HealthAI Intelligent Platform")
print("=" * 60)

print("\n✅ Dataset Loaded Successfully!")

print("\n📊 Shape of Dataset:")
print(df.shape)

print("\n📋 Columns:")
print(df.columns.tolist())

print("\n🔍 First 5 Rows:")
print(df.head())