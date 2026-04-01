"""Prepare batch input files for math_perturbed dataset generation.

Loads all 5000 MATH test problems from EleutherAI/hendrycks_math,
extracts original answers, and splits into batches of 25 problems each.
Saves each batch as data/math_perturbed_batches/batch_{NNN}_input.json.
"""

import json
import os
from datasets import concatenate_datasets, load_dataset

BATCH_SIZE = 25
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "math_perturbed_batches",
)


def extract_boxed_answer(solution: str) -> str:
    """Extract the last \\boxed{...} content from a solution string.

    Handles nested braces properly.
    """
    # Find all \boxed{ occurrences and extract the last one
    idx = solution.rfind("\\boxed{")
    if idx == -1:
        return ""

    # Start after \boxed{
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(solution) and depth > 0:
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            depth -= 1
        i += 1

    return solution[start : i - 1]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load all MATH subsets
    subsets = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]

    all_test = []
    for subset in subsets:
        ds = load_dataset("EleutherAI/hendrycks_math", subset, split="test")
        all_test.append(ds)

    test_dataset = concatenate_datasets(all_test)
    print(f"Loaded {len(test_dataset)} test problems")

    # Build list of problems with extracted answers
    problems = []
    missing_boxed = 0
    for idx, example in enumerate(test_dataset):
        answer = extract_boxed_answer(example["solution"])
        if not answer:
            missing_boxed += 1
            print(f"WARNING: No \\boxed{{}} found in problem {idx}")

        problems.append(
            {
                "idx": idx,
                "problem": example["problem"],
                "solution": example["solution"],
                "answer": answer,
                "level": example["level"],
                "type": example["type"],
            }
        )

    print(f"Total problems: {len(problems)}")
    print(f"Problems missing \\boxed{{}}: {missing_boxed}")

    # Split into batches
    num_batches = (len(problems) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Creating {num_batches} batches of up to {BATCH_SIZE} problems each")

    for batch_idx in range(num_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(problems))
        batch = problems[start:end]

        output_path = os.path.join(OUTPUT_DIR, f"batch_{batch_idx:03d}_input.json")
        with open(output_path, "w") as f:
            json.dump(batch, f, indent=2)

    print(f"Wrote {num_batches} batch files to {OUTPUT_DIR}")

    # Print summary statistics
    types = {}
    levels = {}
    for p in problems:
        types[p["type"]] = types.get(p["type"], 0) + 1
        levels[p["level"]] = levels.get(p["level"], 0) + 1

    print("\nBy type:")
    for t, count in sorted(types.items()):
        print(f"  {t}: {count}")

    print("\nBy level:")
    for l, count in sorted(levels.items()):
        print(f"  {l}: {count}")


if __name__ == "__main__":
    main()
