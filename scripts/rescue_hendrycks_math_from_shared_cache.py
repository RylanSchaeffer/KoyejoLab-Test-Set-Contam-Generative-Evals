"""Rescue EleutherAI/hendrycks_math from the shared HF cache into the repo.

Both upstream sources died by 2026-08-17: the EleutherAI/hendrycks_math Hub
repo is gone (datasets falls back to a cached 2024-11-26 loader module) and
that module's download URL, https://people.eecs.berkeley.edu/~hendrycks/MATH.tar,
returns 403. Every pretraining run loads this benchmark (src/data.py), so
without a local copy no new MATH-contaminated run can ever be launched again.

The processed arrows still exist in the shared cache under builder hash
21a5633873b6a120296cce3e2df9d5550074f4a3 -- the SAME files the published runs
memory-mapped -- so this copy is content-identical to what produced the paper.

Writes data/hendrycks_math/<subset>/ via save_to_disk, then verifies:
  - per-subset row counts sum to 7,500 train / 5,000 test (MATH's split sizes)
  - a reload round-trips and fields match the source arrows exactly

Run once:  uv run python scripts/rescue_hendrycks_math_from_shared_cache.py
The output is deliberately committed to git: the upstream no longer exists,
which makes this the sole reproducible source (cf. MISSING_PRETRAINING_DATA.md
for why irreplaceable inputs live in the repo).
"""

import os

from datasets import Dataset, DatasetDict, load_from_disk

SHARED_CACHE_ROOT = (
    "/lfs/skampere1/0/shared_hf_cache/datasets/EleutherAI___hendrycks_math"
)
BUILDER_HASH = "21a5633873b6a120296cce3e2df9d5550074f4a3"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ROOT = os.path.join(REPO_ROOT, "data", "hendrycks_math")

SUBSETS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]
EXPECTED_TOTALS = {"train": 7500, "test": 5000}


def main() -> None:
    totals = {"train": 0, "test": 0}
    for subset in SUBSETS:
        arrow_dir = os.path.join(SHARED_CACHE_ROOT, subset, "0.0.0", BUILDER_HASH)
        splits = {}
        for split in ["train", "test"]:
            arrow_path = os.path.join(arrow_dir, f"hendrycks_math-{split}.arrow")
            splits[split] = Dataset.from_file(arrow_path)
            totals[split] += len(splits[split])
        output_dir = os.path.join(OUTPUT_ROOT, subset)
        DatasetDict(splits).save_to_disk(output_dir)

        reloaded = load_from_disk(output_dir)
        for split in ["train", "test"]:
            assert len(reloaded[split]) == len(splits[split]), (subset, split)
            assert reloaded[split][0] == splits[split][0], (subset, split)
            assert reloaded[split][-1] == splits[split][-1], (subset, split)
        print(
            f"{subset}: train={len(splits['train'])} test={len(splits['test'])} "
            f"-> {output_dir}"
        )

    assert totals == EXPECTED_TOTALS, f"split totals {totals} != {EXPECTED_TOTALS}"
    print(f"PASS: {totals} rows rescued, round-trip verified, at {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
