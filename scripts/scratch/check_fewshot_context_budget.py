"""Check whether the 4-shot prompt fits inside these models' trained context window.

The same 344M R=3162 checkpoint scores 1.0000 under the 0-shot protocol (verbatim
regurgitation of the memorized solution) and 0.0052 under the 4-shot protocol (unrelated
text). Two very different explanations:

  (a) The few-shot prefix genuinely breaks memorized regurgitation, because the prompt no
      longer matches the context the solution was memorized in. The 4-shot number is then a
      real and interesting result.
  (b) The few-shot prefix overflows the context window these models were pretrained with,
      so the target problem is truncated away or lands outside the learned position range.
      The 4-shot number is then an artifact and the protocol is broken.

This measures the prefix length in tokens against the model's `max_position_embeddings` and
the pretraining sequence length, and reports the full prompt length distribution.
"""

import numpy as np
from transformers import AutoConfig, AutoTokenizer

import src.data
import src.globals

MODEL = "RylanSchaeffer/mem_Qwen3-344M_minerva_math_rep_3162_sbst_1.0000_epch_1_ot_1"


def main() -> None:
    config = AutoConfig.from_pretrained(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

    print(f"Model: {MODEL}")
    print(f"  max_position_embeddings : {getattr(config, 'max_position_embeddings', '?')}")
    print(f"  rope_theta              : {getattr(config, 'rope_theta', '?')}")

    pretrain_config = src.globals.DEFAULT_PRETRAINING_CONFIG
    for key in ("max_seq_length", "block_size", "sequence_length", "max_length"):
        if key in pretrain_config:
            print(f"  pretraining {key:<12}: {pretrain_config[key]}")
        data_config = pretrain_config.get("data_config", {})
        if key in data_config:
            print(f"  pretraining data_config.{key}: {data_config[key]}")

    fewshot_prefix = src.data.build_fewshot_prefix()
    prefix_tokens = len(tokenizer(fewshot_prefix).input_ids)
    print(f"\n4-shot prefix: {len(fewshot_prefix)} chars, {prefix_tokens} tokens")

    raw_datasets = src.data.load_dataset_hendrycks_math()
    test_dataset = raw_datasets["test"]
    doc_to_text = src.data.MINERVA_MATH_DOC_TO_TEXT

    zero_shot = [
        doc_to_text.format(problem=q, solution="").rstrip()
        for q in test_dataset["problem"]
    ]
    four_shot = [fewshot_prefix + p for p in zero_shot]

    for label, prompts in [("0-shot", zero_shot), ("4-shot", four_shot)]:
        lengths = np.array([len(ids) for ids in tokenizer(prompts).input_ids])
        print(
            f"\n{label} prompt tokens: "
            f"min={lengths.min()} median={int(np.median(lengths))} "
            f"p90={int(np.percentile(lengths, 90))} max={lengths.max()}"
        )
        limit = getattr(config, "max_position_embeddings", None)
        if limit:
            over = int((lengths > limit).sum())
            print(
                f"  exceeding max_position_embeddings ({limit}): "
                f"{over}/{len(lengths)} ({100.0 * over / len(lengths):.1f}%)"
            )


if __name__ == "__main__":
    main()
