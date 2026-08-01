"""Verify GSM8K contamination injection produces text the eval prompt is a prefix of.

This is the property the whole 0-shot memorization signal rests on. A contaminated
model regurgitates a memorized document when the prompt reproduces that document's
opening. If the injected text and the eval prompt disagree by even a character, the
model is asked to continue something it never saw, contaminated checkpoints look
clean, and the experiment silently measures nothing.

MATH gets this right by construction: injection and evaluation share
MINERVA_MATH_DOC_TO_TEXT. GSM8K uses two constants (different placeholder names for
different column conventions), so the property has to be checked rather than assumed.

Checks:
  1. The SFT/contaminant path accepts GSM8K at all (Phase 3.1 may already be done).
  2. The injected document for a test item starts with exactly the 0-shot eval prompt.
  3. The stray-indentation bug fixed on 2026-08-01 has not returned.

Does not build the corpus -- only the benchmark side, which is the part that must
match. Runs on CPU in seconds.

Usage:
    uv run python scripts/scratch/verify_gsm8k_contaminant_matches_eval.py
"""

import sys

from transformers import AutoTokenizer

import src.data


def main() -> int:
    failures = 0

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Base", use_fast=True)

    print("1. Does the contaminant path accept GSM8K?")
    try:
        injected = src.data.create_dataset_for_supervised_finetuning(
            tokenizer=tokenizer,
            dataset_name="madrylab/gsm8k-platinum",
            max_length=2048,
            remove_columns=False,
        )
    except NotImplementedError as exc:
        print(f"   FAIL: not supported -- Phase 3.1 needs real work: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"   FAIL: {type(exc).__name__}: {exc}")
        return 1
    print(f"   ok   returned splits: {list(injected.keys())}")

    split = injected["test"] if "test" in injected else injected[list(injected)[0]]
    print(f"   ok   {len(split)} rows, columns {split.column_names}")
    if "text" not in split.column_names:
        print("   FAIL: no `text` column to inject")
        return 1

    print("\n2. Is the 0-shot eval prompt a prefix of the injected document?")
    eval_split = src.data.load_dataset_gsm8k_platinum_for_eval()["test"]
    doc_to_text = src.data.GSM8K_PLATINUM_DOC_TO_TEXT_EVAL

    mismatches = 0
    checked = min(50, len(split), len(eval_split))
    for idx in range(checked):
        prompt = doc_to_text.format(
            problem=eval_split["problem"][idx], solution=""
        ).rstrip()
        document = split["text"][idx]
        if not document.startswith(prompt):
            mismatches += 1
            if mismatches == 1:
                print(f"   FAIL at row {idx}:")
                print(f"     eval prompt starts: {prompt[:120]!r}")
                print(f"     injected doc starts: {document[:120]!r}")
    if mismatches:
        print(
            f"   FAIL: {mismatches}/{checked} injected documents do not start with "
            "the eval prompt. Contaminated models would look clean."
        )
        failures += 1
    else:
        print(f"   ok   all {checked} checked documents start with the eval prompt")

    print("\n3. Regression: no stray indentation in the template")
    rendered = doc_to_text.format(problem="Q?", solution="").rstrip()
    if rendered != "Q: Q?\n\nA:":
        print(f"   FAIL: rendered as {rendered!r}")
        failures += 1
    else:
        print("   ok   renders as 'Q: Q?\\n\\nA:'")

    print(f"\n{'PASS' if failures == 0 else 'FAIL'}: {failures} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
