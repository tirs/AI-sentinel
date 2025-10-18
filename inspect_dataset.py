"""
Inspect the downloaded dataset to see its actual structure
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from datasets import load_dataset

# Load the dataset
cache_dir = Path("c:/Users/simba/Desktop/Ethical/cache/hatexplain")
print(f"Loading dataset from: {cache_dir}")

dataset = load_dataset(str(cache_dir), split="train")

print(f"\nDataset size: {len(dataset)}")
print(f"\nDataset features: {dataset.features}")
print(f"\nColumn names: {dataset.column_names}")

# Show first sample
print("\n" + "=" * 70)
print("FIRST SAMPLE:")
print("=" * 70)
sample = dataset[0]
for key, value in sample.items():
    if isinstance(value, str) and len(value) > 100:
        print(f"{key}: {value[:100]}...")
    else:
        print(f"{key}: {value}")