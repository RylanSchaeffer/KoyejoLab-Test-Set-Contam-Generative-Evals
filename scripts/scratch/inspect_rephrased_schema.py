"""Confirm math_rephrased has the columns the pretraining pipeline needs.

Before wiring rephrased MATH in as a pretraining contaminant, check that its schema matches
what `preprocess_eleutherai_hendrycks_math_for_sft` expects (a `problem` and a `solution`
column), and that the splits line up with the original test set.
"""

import src.data


def describe(name: str, dataset_dict) -> None:
    print(f"\n=== {name}")
    for split, dataset in dataset_dict.items():
        print(f"  split={split} n={len(dataset)} columns={dataset.column_names}")
        if len(dataset):
            example = dataset[0]
            for key in ("problem", "solution"):
                if key in example:
                    text = str(example[key]).replace("\n", " ")
                    print(f"    {key}: {text[:140]}")


def main() -> None:
    describe("EleutherAI/minerva_math (original)", src.data.load_dataset_hendrycks_math())
    describe("RylanSchaeffer/math_rephrased", src.data.load_dataset_math_rephrased())
    describe("RylanSchaeffer/math_perturbed", src.data.load_dataset_math_perturbed())


if __name__ == "__main__":
    main()
