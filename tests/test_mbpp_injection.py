"""Tests for the MBPP pretraining-contamination injection path (Phase 4).

tests/test_mbpp_code_eval.py asserts the *templates* agree; this file asserts
the *injection code path* actually uses them. Two layers:

  1. `preprocess_mbpp_for_sft` (offline, fake tokenizer): the injected document
     starts with exactly the 0-shot eval prompt (everything up to "[BEGIN]"),
     carries the reference code closed by the "\\n[DONE]" sentinel, and ends on
     EOS. If the prefix property breaks, a contaminated model is prompted with
     text it never saw and silently looks clean -- the failure mode
     docs/EXPERIMENT_CHECKLIST.md 3.1 verified against for GSM8K.
  2. `create_dataset_for_supervised_finetuning` dispatch on the real sanitized
     MBPP (marked `slow`; needs the dataset cache): the "test" split is what is
     injected, mirroring MATH/GSM8K, and every row round-trips the prefix
     property against `load_dataset_mbpp_for_eval`.
"""

import pytest

import src.data


class FakeTokenizer:
    """Character-level stand-in: deterministic, offline, EOS-aware."""

    eos_token = "<eos>"
    eos_token_id = 0

    def __call__(self, text):
        # Encode EOS as id 0, every other character as its codepoint + 1.
        input_ids = []
        i = 0
        while i < len(text):
            if text.startswith(self.eos_token, i):
                input_ids.append(self.eos_token_id)
                i += len(self.eos_token)
            else:
                input_ids.append(ord(text[i]) + 1)
                i += 1
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}


EXAMPLES = {
    "prompt": ["Write a function to add two numbers."],
    "code": ["def add(a, b):\n    return a + b"],
    "test_list": [["assert add(1, 2) == 3", "assert add(-1, 1) == 0"]],
}


def _injected_text():
    out = src.data.preprocess_mbpp_for_sft(
        EXAMPLES, tokenizer=FakeTokenizer(), doc_to_text=src.data.MBPP_DOC_TO_TEXT
    )
    assert len(out["text"]) == 1
    return out


def test_injected_document_starts_with_the_zero_shot_eval_prompt():
    text = _injected_text()["text"][0]
    eval_prompt = src.data.MBPP_DOC_TO_TEXT_EVAL.format(
        problem=src.data.mbpp_problem_text(
            EXAMPLES["prompt"][0], EXAMPLES["test_list"][0]
        ),
        solution="",
    ).rstrip()
    assert eval_prompt.endswith("[BEGIN]")
    assert text.startswith(eval_prompt)


def test_injected_document_carries_code_sentinel_and_eos():
    out = _injected_text()
    text = out["text"][0]
    assert EXAMPLES["code"][0] in text
    assert text.endswith("\n[DONE]" + FakeTokenizer.eos_token)
    # Tokenized form ends on the EOS id, and token_length matches.
    assert out["input_ids"][0][-1] == FakeTokenizer.eos_token_id
    assert out["token_length"][0] == len(out["input_ids"][0])


def test_injected_problem_shows_the_test_asserts():
    text = _injected_text()["text"][0]
    assert "Your code should pass these tests:" in text
    for assert_line in EXAMPLES["test_list"][0]:
        assert assert_line in text


def test_unsupported_dataset_still_raises():
    with pytest.raises(NotImplementedError):
        src.data.create_dataset_for_supervised_finetuning(
            tokenizer=FakeTokenizer(), dataset_name="not-a-dataset"
        )


@pytest.mark.slow
def test_create_dataset_dispatch_on_real_sanitized_mbpp():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Base", use_fast=True)
    datasets_dict = src.data.create_dataset_for_supervised_finetuning(
        tokenizer=tokenizer,
        dataset_name="google-research-datasets/mbpp",
        remove_columns=False,
    )
    # split_to_train_on defaults to "test": the 257 hand-verified problems, the
    # same split the eval scripts score -- that identity is the contamination.
    assert len(datasets_dict["train"]) == 257
    assert len(datasets_dict["eval"]) == 257

    eval_split = src.data.load_dataset_mbpp_for_eval()["test"]
    for injected, eval_row in zip(datasets_dict["train"]["text"], eval_split):
        prompt = src.data.MBPP_DOC_TO_TEXT_EVAL.format(
            problem=eval_row["problem"], solution=""
        ).rstrip()
        assert injected.startswith(prompt)
        assert injected.endswith("\n[DONE]" + tokenizer.eos_token)
