"""Print GSM8K train-split examples to hard-code as few-shot demonstrations.

Demonstrations must come from GSM8K's TRAIN split. madrylab/gsm8k-platinum is a
cleaned version of GSM8K's TEST split, so drawing demonstrations from it would put
evaluation items in the prompt.

Deterministic: takes the first n rows of the train split in dataset order, so the
selection is reproducible without a seed.

Usage:
    uv run python scripts/scratch/fetch_gsm8k_fewshot_examples.py --n 8
"""

import argparse

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args()

    train = load_dataset("openai/gsm8k", "main")["train"]
    print(f"train split: {len(train)} rows, columns {train.column_names}\n")

    print("GSM8K_FEWSHOT_EXAMPLES = [")
    for row in train.select(range(args.n)):
        # Strip the <<...>> calculator annotations: they are an artifact of GSM8K's
        # collection process, not something we want a model to imitate.
        answer_no_calc = row["answer"]
        while "<<" in answer_no_calc and ">>" in answer_no_calc:
            start = answer_no_calc.index("<<")
            end = answer_no_calc.index(">>", start) + 2
            answer_no_calc = answer_no_calc[:start] + answer_no_calc[end:]

        print("    {")
        print(f"        \"problem\": {row['question']!r},")
        print(f'        "solution": {answer_no_calc!r},')
        print("    },")
    print("]")


if __name__ == "__main__":
    main()
