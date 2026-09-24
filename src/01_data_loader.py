import pandas as pd


DATA_PATH = "data/raw/Indian_healthcare.csv"


def load_data():
    """Load the raw HealthAI patient dataset."""

    df = pd.read_csv(DATA_PATH)

    return df


if __name__ == "__main__":

    df = load_data()

    print("=" * 60)
    print("HealthAI Suite - Data Loading")
    print("=" * 60)

    print("\nDataset loaded successfully!")

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 Records:")
    print(df.head())