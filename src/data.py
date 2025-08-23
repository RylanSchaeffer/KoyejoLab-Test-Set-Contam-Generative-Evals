from datasets import (
    concatenate_datasets,
    load_dataset,
    interleave_datasets,
    DatasetDict,
)
from functools import partial
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import Any, Dict, List, Optional, Union
import yaml

# See https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/hendrycks_math_algebra.yaml#L10
MINERVA_MATH_DOC_TO_TEXT = "Problem:\n{problem}\n\nSolution: {solution}"

# See https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k_platinum/gsm8k-platinum-cot.yaml#L5-L7
GSM8K_PLATINUM_DOC_TO_TEXT = """Q: {question}

        A: {answer}"""


def create_dataset_for_pretraining(
    data_config: Dict[str, Any],
    trainer_config: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    seed: int = 0,
) -> Dict[str, Union[Dataset, List[Dataset]]]:
    # Load the benchmark.
    benchmark_test_split_dataset = create_dataset_for_supervised_finetuning(
        dataset_name=data_config["benchmark"],
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
    if data_config["num_benchmark_replicas_per_epoch"] > 0:
        replicated_benchmark_test_split_dataset = concatenate_datasets(
            [
                benchmark_test_split_dataset
                for _ in range(data_config["num_benchmark_replicas_per_epoch"])
            ]
        )
    elif data_config["num_benchmark_replicas_per_epoch"] == 0:
        # Select none of the rows to create an empty dataset.
        replicated_benchmark_test_split_dataset = benchmark_test_split_dataset.select(
            range(0)
        )
    else:
        raise ValueError(
            f"Invalid num_benchmark_replicas_per_epoch ({data_config['num_benchmark_replicas_per_epoch']})"
        )

    # Figure out how many tokens we need to take from the corpus to make up the target.
    replicated_benchmark_test_split_num_tokens = np.sum(
        replicated_benchmark_test_split_dataset["token_length"]
    )
    print(
        f"Num. Replicas of Benchmark Test Split Per Epoch: {data_config['num_benchmark_replicas_per_epoch']}\n"
        f"Replicated Benchmark Test Split has {replicated_benchmark_test_split_num_tokens} tokens."
    )

    num_training_tokens_per_epoch = trainer_config["num_training_tokens_per_epoch"]
    target_num_training_tokens_total = trainer_config[
        "target_num_training_tokens_total"
    ]
    num_train_epochs = trainer_config["num_train_epochs"]

    if num_training_tokens_per_epoch < replicated_benchmark_test_split_num_tokens:
        raise ValueError(
            f"num_training_tokens_per_epoch ({num_training_tokens_per_epoch:,}) is smaller than replicated_benchmark_test_split_num_tokens_per_token ({replicated_benchmark_test_split_num_tokens:,})."
        )
    corpus_tokens_needed_per_epoch = int(
        num_training_tokens_per_epoch - replicated_benchmark_test_split_num_tokens
    )
    print(f"Tokens needed from corpus: {corpus_tokens_needed_per_epoch:,}")

    # Load the training corpus.
    if data_config["corpus"] == "fineweb-edu-dedup":
        corpus_dataset = load_dataset(
            "HuggingFaceTB/smollm-corpus",
            "fineweb-edu-dedup",
            split="train",
            num_proc=64,
        )
    else:
        raise ValueError

    # Tokenize and count tokens per sequence.
    def tokenize_truncate_and_count(example):
        # Tokenize.
        tokenized_input = tokenizer(
            example["text"],
            truncation=True,
            max_length=trainer_config["max_length"],
        )
        # Make certain we end on EOS. See: https://arxiv.org/abs/2403.17031
        if tokenized_input["input_ids"][-1] != tokenizer.eos_token_id:
            tokenized_input["input_ids"].append(tokenizer.eos_token_id)
            tokenized_input["attention_mask"].append(1)
        # Truncate if necessary.
        if len(tokenized_input["input_ids"]) > trainer_config["max_length"]:
            tokenized_input["input_ids"] = tokenized_input["input_ids"][
                : trainer_config["max_length"]
            ]
            tokenized_input["attention_mask"] = tokenized_input["attention_mask"][
                : trainer_config["max_length"]
            ]
        example["input_ids"] = tokenized_input["input_ids"]
        example["attention_mask"] = tokenized_input["attention_mask"]
        # Count the number of tokens.
        example["token_length"] = len(tokenized_input["input_ids"])
        return example

    # Estimate how many docs are needed.
    sample_size_for_calculating_avg_tokens_per_sequence = 30000
    corpus_sample = corpus_dataset.select(
        range(sample_size_for_calculating_avg_tokens_per_sequence)
    )
    corpus_sample = corpus_sample.map(tokenize_truncate_and_count, num_proc=64)
    corpus_sample = corpus_sample.filter(
        lambda x: x["token_length"] <= trainer_config["max_length"],
        num_proc=64,
    )
    avg_tokens_per_doc = np.mean(corpus_sample["token_length"])
    # Round up a bit to ensure we have more than we want.
    estimated_docs_needed = int(
        1.1 * corpus_tokens_needed_per_epoch / avg_tokens_per_doc
    )

    # Subsample the appropriate number of documents and tokenize.
    corpus_dataset_subset = corpus_dataset.shuffle(seed=seed).select(
        range(estimated_docs_needed)
    )
    corpus_dataset_subset = corpus_dataset_subset.map(
        tokenize_truncate_and_count, num_proc=64
    )
    corpus_dataset_subset = corpus_dataset_subset.shuffle(seed=seed)
    num_tokens_in_corpus_dataset_subset = np.sum(corpus_dataset_subset["token_length"])
    # Figure out how many documents to drop to meet our target number of tokens.
    num_documents_to_drop = 0
    for num_tokens_in_document in corpus_dataset_subset["token_length"][::-1]:
        num_tokens_in_corpus_dataset_subset -= num_tokens_in_document
        num_documents_to_drop += 1
        if num_tokens_in_corpus_dataset_subset < corpus_tokens_needed_per_epoch:
            break

    corpus_dataset_subset = corpus_dataset_subset.select(
        range(len(corpus_dataset_subset) - num_documents_to_drop)
    )

    # Create the final dataset to train on.
    final_train_dataset = concatenate_datasets(
        [replicated_benchmark_test_split_dataset, corpus_dataset_subset]
    )
    final_train_dataset = final_train_dataset.shuffle(seed=seed)
    total_tokens_per_epoch = np.sum(final_train_dataset["token_length"])
    print(
        f"Final dataset created with {total_tokens_per_epoch:,} tokens.\n"
        f"With {num_train_epochs:,}, total training tokens: {num_train_epochs * total_tokens_per_epoch:,}\n"
        f"Target number of total training tokens: {target_num_training_tokens_total:,}\n"
    )

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
