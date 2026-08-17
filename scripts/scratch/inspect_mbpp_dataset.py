"""Inspect google-research-datasets/mbpp (sanitized config): splits, columns,
and the designated few-shot `prompt` split, for building the MBPP eval harness.

    python scripts/scratch/inspect_mbpp_dataset.py
"""

from datasets import load_dataset

ds = load_dataset("google-research-datasets/mbpp", "sanitized")
print(ds)
for split in ds:
    print(split, ds[split].column_names, len(ds[split]))

print("\n=== prompt split (designated few-shot examples) ===")
for row in ds["prompt"]:
    print("---- task_id", row["task_id"])
    for key in ["prompt", "code", "test_imports", "test_list"]:
        print(f"[{key}]")
        print(row[key])

print("\n=== first test row ===")
row = ds["test"][0]
for key, value in row.items():
    print(f"[{key}]")
    print(value)
