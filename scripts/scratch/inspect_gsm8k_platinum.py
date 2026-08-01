"""Inspect madrylab/gsm8k-platinum structure before wiring it into evaluation.

Confirms split names, column names, answer format, and the "#### <answer>" gold
convention that the GSM8K scorer will depend on. Read-only.

Usage:
    uv run python scripts/scratch/inspect_gsm8k_platinum.py
"""

import re

import src.data


def main() -> None:
    dataset = src.data.load_dataset_gsm8k_platinum()
    print(dataset)
    print()

    for split_name, split in dataset.items():
        print(
            f"=== split {split_name!r}: {len(split)} rows, columns {split.column_names} ==="
        )
        example = split[0]
        for key, value in example.items():
            print(f"  {key}: {value!r}")
        print()

        # How universal is the "#### <answer>" convention, and is the answer
        # always a bare number? The scorer's gold extraction depends on both.
        answers = split["answer"]
        with_marker = sum(1 for a in answers if "####" in a)
        golds = [a.split("####")[-1].strip() for a in answers if "####" in a]
        numeric = sum(1 for g in golds if re.fullmatch(r"-?[\d,]+(\.\d+)?", g))
        has_comma = sum(1 for g in golds if "," in g)
        print(f"  rows containing '####'      : {with_marker}/{len(answers)}")
        print(f"  golds matching a bare number: {numeric}/{len(golds)}")
        print(f"  golds containing a comma    : {has_comma}/{len(golds)}")
        non_numeric = [g for g in golds if not re.fullmatch(r"-?[\d,]+(\.\d+)?", g)]
        if non_numeric:
            print(f"  non-numeric gold examples   : {non_numeric[:10]}")
        print()

    print("GSM8K_PLATINUM_DOC_TO_TEXT currently renders as:")
    print(repr(src.data.GSM8K_PLATINUM_DOC_TO_TEXT))


if __name__ == "__main__":
    main()
