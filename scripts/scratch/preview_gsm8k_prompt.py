"""Print the exact prompt string GSM8K evaluation will send to the model.

Worth looking at before launching a sweep: a malformed prompt produces a clean-looking
0.00 that is indistinguishable from a real capability floor, and the whole point of
Phase 0 is to tell those two apart.

Also reports prompt length in tokens against the 2,048 training context, since a
few-shot prefix that crowds out the generation budget would itself depress scores.

Usage:
    uv run python scripts/scratch/preview_gsm8k_prompt.py --num-fewshot 4
"""

import argparse

from transformers import AutoTokenizer

import src.data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-fewshot", type=int, default=4)
    parser.add_argument(
        "--prompt-style", default="native", choices=["native", "minerva"]
    )
    args = parser.parse_args()

    doc_to_text = (
        src.data.MINERVA_MATH_DOC_TO_TEXT
        if args.prompt_style == "minerva"
        else src.data.GSM8K_PLATINUM_DOC_TO_TEXT_EVAL
    )

    test_split = src.data.load_dataset_gsm8k_platinum_for_eval()["test"]
    question = test_split["problem"][0]
    gold = test_split["solution"][0]

    if args.num_fewshot == 0:
        prefix = ""
    else:
        prefix = src.data.build_fewshot_prefix(
            fewshot_examples=src.data.GSM8K_FEWSHOT_EXAMPLES[: args.num_fewshot],
            doc_to_text=doc_to_text,
        )
    prompt = prefix + doc_to_text.format(problem=question, solution="").rstrip()

    print("=" * 78)
    print(f"PROMPT  (style={args.prompt_style}, {args.num_fewshot}-shot)")
    print("=" * 78)
    print(prompt)
    print("=" * 78)
    print(f"\nGOLD SOLUTION: {gold!r}")
    print(f"GOLD ANSWER  : {src.scoring.extract_gsm8k_gold_answer(gold)!r}")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Base", use_fast=True)
    n_tokens = len(tokenizer(prompt).input_ids)
    print(f"\nprompt tokens: {n_tokens} (training context was 2048)")
    if n_tokens > 1024:
        print("  WARNING: prompt consumes over half the context window.")


if __name__ == "__main__":
    import src.scoring  # noqa: E402  (imported late only for the gold preview)

    main()
