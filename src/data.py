"""Dataset creation and preprocessing for contamination experiments.

This module provides utilities for creating training datasets with controlled
test set contamination. The core functionality injects a specified number of
benchmark test set replicas into a pretraining corpus, enabling systematic
study of how contamination affects model evaluations.

Key contamination parameters:
    - num_benchmark_replicas_per_epoch: Number of times the test set is copied
      into the training data (0 = no contamination, higher = more contamination)
    - benchmark_subset_fraction: Fraction of the benchmark to use (for studying
      partial contamination effects)

Example:
    >>> from src.data import create_dataset_for_pretraining
    >>> datasets = create_dataset_for_pretraining(data_config, trainer_config, tokenizer)
    >>> train_dataset = datasets["train"]  # Contains contaminated corpus
"""

import os
from functools import partial
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import torch
import torch.distributed as dist
from datasets import (
    concatenate_datasets,
    load_dataset,
    load_from_disk,
    DatasetDict,
    Features,
    Sequence,
    Value,
)
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


# Template for formatting MATH problems (matches EleutherAI lm-evaluation-harness)
# See: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/minerva_math/utils.py#L30
MINERVA_MATH_DOC_TO_TEXT = "Problem:\n{problem}\n\nSolution: {solution}"

# Template for formatting GSM8K problems (matches EleutherAI lm-evaluation-harness)
# See: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k_platinum/gsm8k-platinum-cot.yaml#L5-L7
GSM8K_PLATINUM_DOC_TO_TEXT = """Q: {question}

        A: {answer}"""


DEFAULT_COMPRESSION_TYPES = {
    "input_ids": Sequence(Value("int32")),
    "attention_mask": Sequence(Value("bool")),
    "token_length": Value("int32"),
}


class StringHandlingDataCollator:
    """Wrapper for HF data collators that handles string columns.

    Standard HF collators expect tensors, but some datasets include string
    columns (e.g., 'id'). This wrapper extracts string columns before
    collation and adds them back afterward.
    """

    def __init__(self, hf_collator):
        self.hf_collator = hf_collator

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 1. Extract the string IDs so the HF collator doesn't see them
        ids = [feature.pop("id") for feature in features if "id" in feature]

        # 2. Use the standard HF collator for input_ids, attention_mask, etc.
        # This returns a dictionary of PyTorch tensors
        batch = self.hf_collator(features)

        # 3. Add the IDs back into the batch as a list of strings
        if ids:
            batch["id"] = ids
        return batch


def create_dataset_for_pretraining(
    data_config: Dict[str, Any],
    trainer_config: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    cols_to_keep: Optional[Set[str]] = None,
) -> Dict[str, Dataset]:
    """Create a pretraining dataset with controlled test set contamination.

    This function implements the core contamination injection mechanism:
    1. Loads the benchmark test set (e.g., MATH)
    2. Subsamples to `benchmark_subset_fraction` of the original size
    3. Replicates the subset `num_benchmark_replicas_per_epoch` times
    4. Combines with documents from the pretraining corpus (fineweb-edu-dedup)
    5. Shuffles the combined dataset

    The total training tokens per epoch is fixed; more benchmark replicas means
    fewer corpus tokens, keeping compute constant across contamination levels.

    Args:
        data_config: Configuration dict containing:
            - benchmark: Name of benchmark dataset (e.g., "EleutherAI/minerva_math")
            - benchmark_subset_fraction: Fraction of benchmark to use (0.0-1.0)
            - num_benchmark_replicas_per_epoch: Number of test set copies (0+)
            - corpus: Pretraining corpus name (e.g., "fineweb-edu-dedup")
            - shuffle_seed: Random seed for corpus shuffling
            - benchmark_shuffle_seed: Random seed for benchmark shuffling
        trainer_config: Configuration dict containing:
            - max_length: Maximum sequence length for tokenization
            - num_training_tokens_per_epoch: Target tokens per epoch
            - target_num_training_tokens_total: Total training tokens target
            - num_train_epochs: Number of training epochs
        tokenizer: HuggingFace tokenizer for text processing

    Returns:
        Dictionary with keys:
            - "train": Training dataset (contaminated corpus + benchmark replicas)
            - "eval": Held-out corpus evaluation dataset
            - "benchmark": Original benchmark test set (for evaluation)

    Raises:
        ValueError: If num_benchmark_replicas_per_epoch is negative or if
            the replicated benchmark exceeds num_training_tokens_per_epoch.

    Note:
        In distributed training, only rank 0 performs the expensive dataset
        creation; other ranks wait at a barrier then load from disk cache.
    """
    if cols_to_keep is None:
        cols_to_keep = {"input_ids", "attention_mask", "token_length"}

    num_proc = min(64, os.cpu_count())

    # TODO: Spin this out to a top level function.
    # https://chatgpt.com/share/68f0657f-fab0-800d-8329-a8c8acf18ac8
    def tokenize_truncate_and_count(example):
        # Tokenize.
        # Make certain we end on EOS. See: https://arxiv.org/abs/2403.17031
        tokenized_input = tokenizer(
            example["text"] + tokenizer.eos_token,
            truncation=True,
            max_length=trainer_config["max_length"],
        )
        # Make sure we end on an EOS token ID.
        if tokenized_input["input_ids"][-1] != tokenizer.eos_token_id:
            tokenized_input["input_ids"].append(tokenizer.eos_token_id)
            tokenized_input["attention_mask"].append(1)
        example["input_ids"] = tokenized_input["input_ids"]
        example["attention_mask"] = tokenized_input["attention_mask"]
        # Count the number of tokens.
        example["token_length"] = len(tokenized_input["input_ids"])
        return example

    # Specify where to cache rank-0 tokenized artifacts so other ranks can just load
    hf_cache_root = os.getenv("HF_DATASETS_CACHE") or os.path.join(
        os.getcwd(), ".hf_cache"
    )
    os.makedirs(hf_cache_root, exist_ok=True)
    final_train_dataset_cache_dir = os.path.join(
        hf_cache_root, "corpus_subset_tokenized"
    )
    corpus_eval_dataset_cache_dir = os.path.join(hf_cache_root, "corpus_eval_tokenized")

    # Load the benchmark.
    benchmark_test_split_dataset = create_dataset_for_supervised_finetuning(
        dataset_name=data_config["benchmark"],
        tokenizer=tokenizer,
        remove_columns=False,
    )["eval"]

    # Remove unnecessary columns from the benchmark.
    benchmark_test_split_dataset = benchmark_test_split_dataset.remove_columns(
        [
            col
            for col in benchmark_test_split_dataset.column_names
            if col not in cols_to_keep
        ]
    )

    # Subsample then shuffle the benchmark as specified.
    num_benchmark_samples_to_subsample = int(
        data_config["benchmark_subset_fraction"] * len(benchmark_test_split_dataset)
    )
    # Make sure we take at least 1 sample.
    num_benchmark_samples_to_subsample = max(
        1,
        num_benchmark_samples_to_subsample,
    )
    benchmark_test_split_dataset = benchmark_test_split_dataset.shuffle(
        seed=data_config["benchmark_shuffle_seed"]
    ).select(range(num_benchmark_samples_to_subsample))

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
    num_training_tokens_per_epoch = trainer_config["num_training_tokens_per_epoch"]
    target_num_training_tokens_total = trainer_config[
        "target_num_training_tokens_total"
    ]
    num_train_epochs = trainer_config["num_train_epochs"]

    if _is_main():
        print(
            f"Num. Replicas of Benchmark Test Split Per Epoch: {data_config['num_benchmark_replicas_per_epoch']}\n"
            f"Replicated Benchmark Test Split has {replicated_benchmark_test_split_num_tokens:,} tokens."
        )

        if num_training_tokens_per_epoch < replicated_benchmark_test_split_num_tokens:
            raise ValueError(
                f"num_training_tokens_per_epoch ({num_training_tokens_per_epoch:,}) is smaller than replicated_benchmark_test_split_num_tokens_per_token ({replicated_benchmark_test_split_num_tokens:,})."
            )

        corpus_tokens_needed_per_epoch = int(
            num_training_tokens_per_epoch - replicated_benchmark_test_split_num_tokens
        )

        print(
            f"Tokens needed from corpus: {num_training_tokens_per_epoch:,} - {replicated_benchmark_test_split_num_tokens:,} = {corpus_tokens_needed_per_epoch:,}"
        )

        if data_config["corpus"] == "fineweb-edu-dedup":
            corpus_full_dataset = load_dataset(
                "HuggingFaceTB/smollm-corpus",
                "fineweb-edu-dedup",
                split="train",
                num_proc=num_proc,
            )
            # The full dataset is 220B tokens in 190,168,005 rows.
            # We want 150M tokens for test.
            corpus_split_dataset = corpus_full_dataset.train_test_split(
                test_size=150e6 / 220e9,
                seed=data_config["train_test_split_seed"],
            )
            print("Split corpus into train and test")
            corpus_train_dataset = corpus_split_dataset["train"]
            corpus_eval_dataset = corpus_split_dataset["test"]
            avg_tokens_per_doc = 220e9 / 190168005  # ~1157 tokens per doc
        else:
            raise ValueError

        # Round up a bit to ensure we have more than we want.
        estimated_docs_needed = int(
            1.05 * corpus_tokens_needed_per_epoch / avg_tokens_per_doc
        )

        # Subsample the appropriate number of documents and tokenize.
        print("Shuffling, selecting and tokenizing the pretraining corpus.")
        rng = np.random.default_rng(data_config["shuffle_seed"])
        sample_indices = rng.choice(
            len(corpus_train_dataset),
            size=estimated_docs_needed,
            replace=False,
        )
        corpus_train_dataset_subset = (
            corpus_train_dataset.select(sample_indices)
            .shuffle(seed=data_config["shuffle_seed"])
            .map(tokenize_truncate_and_count, num_proc=num_proc)
        )

        # Figure out how many documents to keep to meet our target number of tokens.
        # Use searchsorted for O(log n) instead of iterative O(n) loop.
        # Original code dropped documents from the end until total < target,
        # so we keep documents where cumsum < target (i.e., up to but not including
        # the first index where cumsum >= target).
        cumulative_lengths = np.cumsum(corpus_train_dataset_subset["token_length"])
        idx_to_keep = np.searchsorted(cumulative_lengths, corpus_tokens_needed_per_epoch)
        corpus_train_dataset_subset = corpus_train_dataset_subset.select(
            range(idx_to_keep)
        )

        # Create the dataset we will train on.
        print("Concatenated replicated benchmark test split and pretraining corpus.")
        final_train_dataset = concatenate_datasets(
            [replicated_benchmark_test_split_dataset, corpus_train_dataset_subset]
        )
        final_train_dataset = final_train_dataset.shuffle(
            seed=data_config["shuffle_seed"]
        )

        # Remove unnecessary columns to reduce size, then save to disk.
        cols_to_drop = [
            c for c in final_train_dataset.column_names if c not in cols_to_keep
        ]
        final_train_dataset = final_train_dataset.remove_columns(cols_to_drop)

        # Cut the Arrow buffers in half by casting dtypes before saving (no semantic change).
        final_train_dataset = final_train_dataset.cast(
            Features(
                {
                    k: v
                    for k, v in DEFAULT_COMPRESSION_TYPES.items()
                    if k in cols_to_keep
                }
            ),
            num_proc=num_proc,
        )
        final_train_dataset.save_to_disk(final_train_dataset_cache_dir)

        corpus_eval_dataset = corpus_eval_dataset.map(
            tokenize_truncate_and_count, num_proc=num_proc
        )
        cols_to_drop_eval = [
            c for c in corpus_eval_dataset.column_names if c not in cols_to_keep
        ]
        corpus_eval_dataset = corpus_eval_dataset.remove_columns(cols_to_drop_eval)
        corpus_eval_dataset = corpus_eval_dataset.cast(
            Features(
                {
                    k: v
                    for k, v in DEFAULT_COMPRESSION_TYPES.items()
                    if k in cols_to_keep
                }
            ),
            num_proc=num_proc,
        )
        corpus_eval_dataset.save_to_disk(corpus_eval_dataset_cache_dir)

        total_tokens_per_epoch = np.sum(final_train_dataset["token_length"])
        print(
            f"Final dataset created with {total_tokens_per_epoch:,} tokens.\n"
            f"With {num_train_epochs:,} training epochs, total training tokens: {num_train_epochs * total_tokens_per_epoch:,}\n"
            f"Target number of total training tokens: {target_num_training_tokens_total:,}\n"
        )

    if (
        _world_size() > 1
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        torch.distributed.barrier()  # non-zero ranks wait for rank 0 to finish

    # All processes load the datasets from disk.
    final_train_dataset = load_from_disk(final_train_dataset_cache_dir)
    corpus_eval_dataset = load_from_disk(corpus_eval_dataset_cache_dir)

    datasets_dict = {
        "train": final_train_dataset,
        "eval": corpus_eval_dataset,
        "benchmark": benchmark_test_split_dataset,
    }

    return datasets_dict


def create_dataset_for_supervised_finetuning(
    tokenizer: PreTrainedTokenizer,
    dataset_name: str,
    max_length: Optional[int] = None,
    remove_columns: bool = True,
    split_to_train_on: str = "test",
) -> Dict[str, Dataset]:
    """Create datasets for supervised fine-tuning on math benchmarks.

    Loads and preprocesses math problem datasets (MATH or GSM8K) into a format
    suitable for causal language model fine-tuning. Each example is formatted
    as "Problem: {problem}\n\nSolution: {solution}" and tokenized.

    Args:
        tokenizer: HuggingFace tokenizer for text processing.
        dataset_name: Dataset identifier. Supported values:
            - "EleutherAI/minerva_math": Hendrycks MATH benchmark
            - "madrylab/gsm8k-platinum": GSM8K Platinum dataset
        max_length: Optional maximum sequence length filter. Examples exceeding
            this length are removed.
        remove_columns: If True, remove all columns except input_ids and
            attention_mask. Set False to retain problem/solution text.
        split_to_train_on: Which split to use for training ("train" or "test").
            Default "test" is used for contamination studies.

    Returns:
        Dictionary with keys:
            - "train": Training dataset (from specified split)
            - "eval": Evaluation dataset (always from test split)

    Raises:
        NotImplementedError: If dataset_name is not supported.
        ValueError: If split_to_train_on is not "train" or "test".
    """
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
        # load_from_cache_file=False,  # Always make sure we're using the latest version.
        load_from_cache_file=True,
        batched=True,
        num_proc=16,
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
    if split_to_train_on == "test":
        train_dataset = raw_datasets["test"]
    elif split_to_train_on == "train":
        train_dataset = raw_datasets["train"]
    else:
        raise ValueError(f"Invalid split to train on: {split_to_train_on}")
    eval_dataset = raw_datasets["test"]

    datasets_dict = {
        "train": train_dataset,
        "eval": eval_dataset,
    }

    return datasets_dict


def load_dataset_hendrycks_math() -> DatasetDict:
    """Load and concatenate all subsets of the Hendrycks MATH benchmark.

    The MATH benchmark contains 7 subject areas: algebra, counting_and_probability,
    geometry, intermediate_algebra, number_theory, prealgebra, and precalculus.
    This function loads all subsets and concatenates them into unified train/test splits.

    Note:
        We use EleutherAI's version of MATH (minerva_math) for evaluation because
        the original hendrycks_math evaluation code has known issues.
        See: https://github.com/EleutherAI/lm-evaluation-harness/issues/3210

    Returns:
        DatasetDict with "train" and "test" splits containing all MATH problems.
    """
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
    """Load the GSM8K Platinum dataset.

    GSM8K Platinum is a high-quality version of the GSM8K grade school math
    dataset, curated by MadryLab with improved answer quality.

    Returns:
        DatasetDict with train and test splits.
    """
    return load_dataset("madrylab/gsm8k-platinum")


def preprocess_eleutherai_hendrycks_math_for_sft(
    examples: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    doc_to_text: str,
) -> Dict[str, List[Any]]:
    """Preprocess MATH examples for supervised fine-tuning.

    Formats each problem-solution pair using the provided template and tokenizes.
    Ensures each sequence ends with an EOS token for proper autoregressive training.

    Args:
        examples: Batch of examples with "problem" and "solution" fields.
        tokenizer: HuggingFace tokenizer.
        doc_to_text: Format string template (should contain {problem} and {solution}).

    Returns:
        Dictionary with tokenized fields: text, input_ids, attention_mask, token_length.
    """
    new_examples: Dict[str, List[Any]] = {
        "text": [],
        "input_ids": [],
        "attention_mask": [],
        "token_length": [],
    }

    for problem, solution in zip(examples["problem"], examples["solution"]):
        # Make certain we end on EOS. See: https://arxiv.org/abs/2403.17031
        text = (
            doc_to_text.format(problem=problem, solution=solution) + tokenizer.eos_token
        )
        tokenized_input = tokenizer(text)
        # Make sure we end on an EOS token ID.
        if tokenized_input["input_ids"][-1] != tokenizer.eos_token_id:
            # Replace the last token to ensure the sequence ends with EOS
            tokenized_input["input_ids"][-1] = tokenizer.eos_token_id
        new_examples["text"].append(text)
        new_examples["input_ids"].append(tokenized_input["input_ids"])
        new_examples["attention_mask"].append(tokenized_input["attention_mask"])
        new_examples["token_length"].append(len(tokenized_input["input_ids"]))

    return new_examples


def preprocess_madrylab_gsm8k_platinum_for_sft(
    examples: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    doc_to_text: str,
) -> Dict[str, List[Any]]:
    """Preprocess GSM8K Platinum examples for supervised fine-tuning.

    Formats each question-answer pair using the provided template and tokenizes.
    Ensures each sequence ends with an EOS token for proper autoregressive training.

    Args:
        examples: Batch of examples with "question" and "answer" fields.
        tokenizer: HuggingFace tokenizer.
        doc_to_text: Format string template (should contain {question} and {answer}).

    Returns:
        Dictionary with tokenized fields: text, input_ids, attention_mask, token_length.
    """
    new_examples: Dict[str, List[Any]] = {
        "text": [],
        "input_ids": [],
        "attention_mask": [],
        "token_length": [],
    }

    for question, answer in zip(examples["question"], examples["answer"]):
        text = (
            doc_to_text.format(question=question, answer=answer) + tokenizer.eos_token
        )
        tokenized_input = tokenizer(text)
        # Make certain we end on EOS. See: https://arxiv.org/abs/2403.17031
        if tokenized_input["input_ids"][-1] != tokenizer.eos_token_id:
            # Replace the last token to ensure the sequence ends with EOS
            tokenized_input["input_ids"][-1] = tokenizer.eos_token_id
        new_examples["text"].append(text)
        new_examples["input_ids"].append(tokenized_input["input_ids"])
        new_examples["attention_mask"].append(tokenized_input["attention_mask"])
        new_examples["token_length"].append(len(tokenized_input["input_ids"]))

    return new_examples


def _world_size() -> int:
    """Get the number of distributed processes (defaults to GPU count)."""
    return int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))


def _rank() -> int:
    """Get the global rank of this process (0-indexed)."""
    return int(os.environ.get("RANK", "0"))


def _local_rank() -> int:
    """Get the local rank on this node (0-indexed)."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def _is_main() -> bool:
    """Check if this is the main (rank 0) process."""
    return _rank() == 0


def _is_sweep_run() -> bool:
    """Check if running as part of a W&B sweep."""
    return os.environ.get("WANDB_SWEEP_ID") is not None
