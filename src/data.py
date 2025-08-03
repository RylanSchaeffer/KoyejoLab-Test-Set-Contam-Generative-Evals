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


def create_dataset_for_supervised_finetuning(
    tokenizer: PreTrainedTokenizer,
    dataset_name: str,
    max_length: Optional[int] = None,
    remove_columns: bool = True,
) -> Dict[str, Union[Dataset]]:
    if dataset_name == "madrylab/gsm8k-platinum":
        with open("task_templates/gsm8k-platinum-cot.yaml", "r") as file:
            task_template = yaml.safe_load(file)
            doc_to_text = task_template["doc_to_text"]
        raw_datasets = load_dataset(dataset_name)
        raw_datasets = raw_datasets.map(
            partial(
                preprocess_madrylab_gsm8k_platinum_sft,
                tokenizer=tokenizer,
                doc_to_text=doc_to_text,
            ),
            load_from_cache_file=False,  # Always make sure we're using the latest version.
            batched=True,
            num_proc=4,
        )
        if max_length is not None:
            raw_datasets = raw_datasets.filter(
                lambda x: len(x["input_ids"]) <= max_length
            )
        if remove_columns:
            columns_to_remove = [
                col
                for col in raw_datasets["test"].column_names
                if col not in {"input_ids", "attention_mask"}
            ]
            raw_datasets = raw_datasets.remove_columns(columns_to_remove)

        indices = np.arange(len(raw_datasets["test"])).astype(int)
        train_indices, test_indices = train_test_split(
            indices,
            test_size=0.5,
        )

        train_dataset = raw_datasets["test"].select(train_indices)
        eval_dataset = raw_datasets["test"].select(test_indices)

        datasets_dict = {
            "train": train_dataset,
            "eval": eval_dataset,
        }
    else:
        raise NotImplementedError(f"Unsupported dataset: {dataset_name}")

    return datasets_dict


def preprocess_madrylab_gsm8k_platinum_sft(
    examples: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    doc_to_text: str,
) -> Dict[str, List]:
    new_examples = {
        "input_ids": [],
        "attention_mask": [],
    }

    for question, answer in zip(examples["question"], examples["answer"]):
        input_str = doc_to_text.format(Q=question, A=answer)
        tokenized_input = tokenizer(input_str)
        # Make certain we end on EOS. See: https://arxiv.org/abs/2403.17031
        if tokenized_input["input_ids"][-1] != tokenizer.eos_token_id:
            tokenized_input["input_ids"].append(tokenizer.eos_token_id)
            tokenized_input["attention_mask"].append(1)
        new_examples["input_ids"].append(tokenized_input["input_ids"])
        new_examples["attention_mask"].append(tokenized_input["attention_mask"])

    return new_examples
