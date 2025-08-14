from accelerate import PartialState
from collections import defaultdict
from datasets import (
    concatenate_datasets,
    load_dataset,
    interleave_datasets,
    DatasetDict,
)
from functools import partial
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset, random_split
from transformers import PreTrainedTokenizer
from typing import Any, Dict, List, Optional, Union
import yaml

# See https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/hendrycks_math_algebra.yaml#L10
MINERVA_MATH_DOC_TO_TEXT = "Problem:\n{problem}\n\nSolution: {solution}"

# See https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k_platinum/gsm8k-platinum-cot.yaml#L5-L7
GSM8K_PLATINUM_DOC_TO_TEXT = """Q: {question}

        A: {answer}"""


def create_dataset_for_supervised_finetuning(
    tokenizer: PreTrainedTokenizer,
    dataset_name: str,
    max_length: Optional[int] = None,
    remove_columns: bool = True,
) -> Dict[str, Union[Dataset]]:
    if dataset_name == "EleutherAI/minerva_math":
        raw_datasets = load_dataset_hendrycks_math()
        preprocess_fn = preprocess_eleutherai_hendrycks_math_for_sft
        doc_to_text = MINERVA_MATH_DOC_TO_TEXT
    elif dataset_name == "madrylab/gsm8k-platinum":
        raw_datasets = load_dataset_gsm8k_platinum()
        preprocess_fn = preprocess_madrylab_gsm8k_platinum_for_sft
        doc_to_text = GSM8K_PLATINUM_DOC_TO_TEXT
    else:
        raise NotImplementedError(f"Unsupported dataset: {dataset_name}")

    raw_datasets = raw_datasets.map(
        partial(
            preprocess_fn,
            tokenizer=tokenizer,
            doc_to_text=doc_to_text,
        ),
        load_from_cache_file=False,  # Always make sure we're using the latest version.
        batched=True,
        num_proc=4,
    )
    if max_length is not None:
        raw_datasets = raw_datasets.filter(lambda x: len(x["input_ids"]) <= max_length)
    if remove_columns:
        columns_to_remove = [
            col
            for col in raw_datasets["test"].column_names
            if col not in {"input_ids", "attention_mask"}
        ]
        raw_datasets = raw_datasets.remove_columns(columns_to_remove)

    train_dataset = raw_datasets["test"]
    eval_dataset = raw_datasets["test"]

    datasets_dict = {
        "train": train_dataset,
        "eval": eval_dataset,
    }

    return datasets_dict


def load_dataset_hendrycks_math() -> DatasetDict:
    subsets = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
    # Note: Hendrycks MATH is the dataset we will use, but for training and scoring, we will use Minerva MATH.
    # This is because the Hendrycks MATH evaluation code is borked.
    # See: https://github.com/EleutherAI/lm-evaluation-harness/issues/3210
    raw_datasets_list = [
        load_dataset("EleutherAI/hendrycks_math", subset) for subset in subsets
    ]
    raw_datasets = DatasetDict(
        {
            "train": concatenate_datasets([d["train"] for d in raw_datasets_list]),
            "test": concatenate_datasets([d["test"] for d in raw_datasets_list]),
        }
    )
    return raw_datasets


def load_dataset_gsm8k_platinum() -> DatasetDict:
    return load_dataset("madrylab/gsm8k-platinum")


def preprocess_eleutherai_hendrycks_math_for_sft(
    examples: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    doc_to_text: str,
) -> Dict[str, list]:
    new_examples = {
        "input_ids": [],
        "attention_mask": [],
    }

    for problem, solution in zip(examples["problem"], examples["solution"]):
        input_str = doc_to_text.format(problem=problem, solution=solution)
        tokenized_input = tokenizer(input_str)
        # Make certain we end on EOS. See: https://arxiv.org/abs/2403.17031
        if tokenized_input["input_ids"][-1] != tokenizer.eos_token_id:
            tokenized_input["input_ids"].append(tokenizer.eos_token_id)
            tokenized_input["attention_mask"].append(1)
        new_examples["input_ids"].append(tokenized_input["input_ids"])
        new_examples["attention_mask"].append(tokenized_input["attention_mask"])

    return new_examples


def preprocess_madrylab_gsm8k_platinum_for_sft(
    examples: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    doc_to_text: str,
) -> Dict[str, List]:
    new_examples = {
        "input_ids": [],
        "attention_mask": [],
    }

    for question, answer in zip(examples["question"], examples["answer"]):
        input_str = doc_to_text.format(question=question, answer=answer)
        tokenized_input = tokenizer(input_str)
        # Make certain we end on EOS. See: https://arxiv.org/abs/2403.17031
        if tokenized_input["input_ids"][-1] != tokenizer.eos_token_id:
            tokenized_input["input_ids"].append(tokenizer.eos_token_id)
            tokenized_input["attention_mask"].append(1)
        new_examples["input_ids"].append(tokenized_input["input_ids"])
        new_examples["attention_mask"].append(tokenized_input["attention_mask"])

    return new_examples
