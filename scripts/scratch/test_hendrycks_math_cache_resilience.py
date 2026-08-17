"""Diagnose the 2026-08-17 hendrycks_math 403 failure mode, offline.

The upstream sources are both dead: the EleutherAI/hendrycks_math Hub repo is
gone (datasets falls back to a cached loader module from 2024-11-26) and that
module's URL, https://people.eecs.berkeley.edu/~hendrycks/MATH.tar, now
returns 403. Every pretraining run loads this benchmark (src/data.py:403), so
whether a run survives depends entirely on cache hits.

Run with HF_DATASETS_OFFLINE=1 so a cache miss fails fast instead of hitting
the dead URL. Two modes:

    uv run python scripts/scratch/test_hendrycks_math_cache_resilience.py shared
        -> load all 7 subsets against the default (shared) cache root.
    uv run python scripts/scratch/test_hendrycks_math_cache_resilience.py empty
        -> HF_DATASETS_CACHE must point at an empty dir (set by the caller,
           before process start); tries one subset, expecting failure if the
           env var redirects dataset-cache resolution.
"""

import sys

from datasets import load_dataset

SUBSETS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def main() -> None:
    mode = sys.argv[1]
    subsets = SUBSETS if mode == "shared" else SUBSETS[:1]
    for subset in subsets:
        try:
            dataset = load_dataset("EleutherAI/hendrycks_math", subset)
            sizes = {split: len(rows) for split, rows in dataset.items()}
            print(f"OK   {subset}: {sizes}")
        except Exception as exc:
            print(f"FAIL {subset}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
