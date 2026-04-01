"""Assemble all batch output files into a single dataset and push to HuggingFace.

Loads all 200 batch output JSON files, concatenates into a single dataset
of 5000 rows sorted by idx, and pushes to RylanSchaeffer/math_perturbed.
"""

import json
import os
import sys

from datasets import Dataset

BATCH_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "math_perturbed_batches",
)

TOTAL_BATCHES = 200
HF_REPO = "RylanSchaeffer/math_perturbed"


def main():
    # Collect all output data
    all_rows = []
    missing_batches = []

    for batch_idx in range(TOTAL_BATCHES):
        output_path = os.path.join(BATCH_DIR, f"batch_{batch_idx:03d}_output.json")
        if not os.path.exists(output_path):
            missing_batches.append(batch_idx)
            continue

        with open(output_path) as f:
            batch_data = json.load(f)
        all_rows.extend(batch_data)

    if missing_batches:
        print(f"ERROR: Missing {len(missing_batches)} batch files: {missing_batches}")
        sys.exit(1)

    print(f"Loaded {len(all_rows)} total rows from {TOTAL_BATCHES} batches")

    # Sort by idx
    all_rows.sort(key=lambda x: x["idx"])

    # Verify continuous idx range
    idxs = [r["idx"] for r in all_rows]
    expected_idxs = list(range(5000))
    if idxs != expected_idxs:
        missing = set(expected_idxs) - set(idxs)
        duplicates = [i for i in idxs if idxs.count(i) > 1]
        if missing:
            print(f"ERROR: Missing idx values: {sorted(missing)[:20]}...")
        if duplicates:
            print(f"ERROR: Duplicate idx values: {sorted(set(duplicates))[:20]}...")
        sys.exit(1)

    # Create HuggingFace dataset
    dataset = Dataset.from_list(all_rows)
    print(f"\nDataset created: {dataset}")
    print(f"Columns: {dataset.column_names}")
    print(f"Num rows: {len(dataset)}")

    # Print sample
    print(f"\nSample row (idx=0):")
    sample = dataset[0]
    for key, value in sample.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: {value[:100]}...")
        else:
            print(f"  {key}: {value}")

    # Push to HuggingFace
    print(f"\nPushing to {HF_REPO}...")
    dataset.push_to_hub(HF_REPO, split="test")
    print(f"Successfully pushed {len(dataset)} rows to {HF_REPO}")


if __name__ == "__main__":
    main()
