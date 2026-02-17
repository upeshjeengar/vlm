import os
import requests
from PIL import Image
from io import BytesIO
from datasets import load_dataset

PARQUET_PATH = "dataset/conceptual-captions-200k.parquet"
# Configuration
LIMIT = 300_000
DATASET_NAME = "flax-community/conceptual-captions-12"


def main():
    print(f"Loading top {LIMIT} rows from {DATASET_NAME}...")

    # Load only the first LIMIT rows
    # We use split=f"train[:{LIMIT}]" to avoid downloading the metadata for the full 12M rows
    ds = load_dataset(DATASET_NAME, split=f"train[:{LIMIT}]")
    print(
        "Starting download. This will take time depending on your internet connection..."
    )

    # Save the index (Parquet)
    print(f"Saving parquet to {PARQUET_PATH}...")
    ds.to_parquet(PARQUET_PATH)
    print("Done!")


if __name__ == "__main__":
    main()
