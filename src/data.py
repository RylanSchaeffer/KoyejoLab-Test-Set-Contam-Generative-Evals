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


def create_dataset_for_pretraining(
    data_config_dict: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    target_num_unique_tokens: int,
    seed: int = 0,
    max_length: int = 2048,
) -> Dict[str, Union[Dataset, List[Dataset]]]:
    # Load the benchmark.
    benchmark_test_split_dataset = create_dataset_for_supervised_finetuning(
        dataset_name=data_config_dict["benchmark"],
        tokenizer=tokenizer,
        remove_columns=False,
    )["eval"]

    # Remove unnecessary columns.
    columns_to_remove = [
        col
        for col in benchmark_test_split_dataset.column_names
        if col not in {"text", "input_ids", "attention_mask", "token_length"}
    ]
    benchmark_test_split_dataset = benchmark_test_split_dataset.remove_columns(
        columns_to_remove
    )

    # Replicate the benchmark.
    replicated_benchmark_test_split_dataset = concatenate_datasets(
        [
            benchmark_test_split_dataset
            for _ in range(data_config_dict["num_benchmark_replicas"])
        ]
    )

    # Figure out how many tokens we need to take from the corpus to make up the target.
    replicated_benchmark_test_split_num_tokens = np.sum(
        replicated_benchmark_test_split_dataset["token_length"]
    )
    print(
        f"Benchmark Test Split has {replicated_benchmark_test_split_num_tokens} tokens."
    )
    if target_num_unique_tokens < replicated_benchmark_test_split_num_tokens:
        raise ValueError(
            f"Target token count ({target_num_unique_tokens:,}) is smaller than the test set size ({replicated_benchmark_test_split_num_tokens:,})."
        )
    corpus_tokens_needed = (
        target_num_unique_tokens - replicated_benchmark_test_split_num_tokens
    )
    print(f"Tokens needed from corpus: {corpus_tokens_needed:,}")

    # Load the training corpus.
    if data_config_dict["corpus"] == "fineweb-edu-dedup":
        corpus_dataset = load_dataset(
            "HuggingFaceTB/smollm-corpus",
            "fineweb-edu-dedup",
            split="train",
            num_proc=32,
        )
    else:
        raise ValueError

    # Tokenize and count tokens per sequence.
    def tokenize_truncate_and_count(example):
        # Tokenize.
        tokenized_input = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
        )
        # Make certain we end on EOS. See: https://arxiv.org/abs/2403.17031
        if tokenized_input["input_ids"][-1] != tokenizer.eos_token_id:
            tokenized_input["input_ids"].append(tokenizer.eos_token_id)
            tokenized_input["attention_mask"].append(1)
        # Truncate if necessary.
        if len(tokenized_input["input_ids"]) > max_length:
            tokenized_input["input_ids"] = tokenized_input["input_ids"][:max_length]
            tokenized_input["attention_mask"] = tokenized_input["attention_mask"][
                :max_length
            ]
        example["input_ids"] = tokenized_input["input_ids"]
        example["attention_mask"] = tokenized_input["attention_mask"]
        # Count the number of tokens.
        example["token_length"] = len(tokenized_input["input_ids"])
        return example

    # Estimate how many docs are needed, adding a 5% buffer.
    sample_size_for_calculating_avg_tokens_per_sequence = 30000
    corpus_sample = corpus_dataset.select(
        range(sample_size_for_calculating_avg_tokens_per_sequence)
    )
    corpus_sample = corpus_sample.map(tokenize_truncate_and_count, num_proc=32)
    if max_length is not None:
        corpus_sample = corpus_sample.filter(lambda x: x["token_length"] <= max_length)
    avg_tokens_per_doc = np.mean(corpus_sample["token_length"])
    estimated_docs_needed = int(1.05 * corpus_tokens_needed / avg_tokens_per_doc)

    # Subsample the appropriate number of documents and tokenize.
    corpus_dataset_subset = corpus_dataset.shuffle(seed=seed).select(
        range(estimated_docs_needed)
    )
    corpus_dataset_subset = corpus_dataset_subset.map(
        tokenize_truncate_and_count, num_proc=32
    )
    corpus_dataset_subset = corpus_dataset_subset.filter(
        lambda x: x["token_length"] <= max_length
    )

    # Create the final dataset to train on.
    final_train_dataset = concatenate_datasets(
        [replicated_benchmark_test_split_dataset, corpus_dataset_subset]
    )
    final_train_dataset = final_train_dataset.shuffle(seed=seed)
    total_tokens = np.sum(final_train_dataset["token_length"])
    print(
        f"Final dataset created with {total_tokens:,} tokens.\n"
        f"Target number of unique tokens: {target_num_unique_tokens}"
    )

    # Remove columns that are not model inputs
    columns_to_remove = ["text", "token_length"]
    final_train_dataset = final_train_dataset.remove_columns(columns_to_remove)

    datasets_dict = {
        "train": final_train_dataset,
        "eval": benchmark_test_split_dataset,
    }

    return datasets_dict


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
        raw_datasets = raw_datasets.filter(lambda x: x["token_length"] <= max_length)
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
        "token_length": [],
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
        new_examples["token_length"].append(len(tokenized_input["input_ids"]))

    return new_examples


def preprocess_madrylab_gsm8k_platinum_for_sft(
    examples: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    doc_to_text: str,
) -> Dict[str, List]:
    new_examples = {
        "input_ids": [],
        "attention_mask": [],
        "token_length": [],
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
        new_examples["token_length"].append(len(tokenized_input["input_ids"]))

    return new_examples
