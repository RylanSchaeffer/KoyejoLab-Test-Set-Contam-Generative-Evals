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
#
# Corrected 2026-08-01. This was written as a triple-quoted literal whose
# continuation line carried the source file's indentation, so it rendered as
# "Q: {question}\n\n        A: {answer}" -- eight stray spaces before "A:".
# Only the SFT/contaminant path consumed it, and only for the superseded
# lm-eval-era GSM8K runs in notebooks/00_*, so nothing in the current manuscript
# depends on the old rendering.
#
# CRITICAL for contamination work: the injected text and the eval prompt must be
# byte-identical, or the 0-shot memorization signal is measured against a prompt
# the model never saw. GSM8K_PLATINUM_DOC_TO_TEXT_EVAL below exists only because
# the eval path names its columns `problem`/`solution`; the two must render the
# same string, which tests/test_gsm8k_scoring.py asserts.
GSM8K_PLATINUM_DOC_TO_TEXT = "Q: {question}\n\nA: {answer}"

# Same template, keyed for the evaluation path's normalized column names.
GSM8K_PLATINUM_DOC_TO_TEXT_EVAL = "Q: {problem}\n\nA: {solution}"


# 4-shot examples for minerva_math, hardcoded in EleutherAI lm-evaluation-harness.
# Source: lm_eval/tasks/minerva_math/utils.py:list_fewshot_samples()
MINERVA_MATH_FEWSHOT_EXAMPLES = [
    {
        "problem": "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
        "solution": "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: The final answer is $[2,5)$. I hope it is correct.",
    },
    {
        "problem": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
        "solution": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nFinal Answer: The final answer is $24$. I hope it is correct.",
    },
    {
        "problem": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
        "solution": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nFinal Answer: The final answer is $16$. I hope it is correct.",
    },
    {
        "problem": "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.",
        "solution": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nFinal Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.",
    },
]


# 4-shot examples for GSM8K, taken verbatim from the first four rows of the
# `openai/gsm8k` **train** split, with the <<...>> calculator annotations stripped
# (an artifact of GSM8K's collection process, not something to teach a model).
#
# Train, not test: madrylab/gsm8k-platinum is a cleaned version of GSM8K's *test*
# split, so demonstrations drawn from it would put evaluation items in the prompt.
#
# Keyed `problem`/`solution` rather than `question`/`answer` so that
# `build_fewshot_prefix` formats them without special-casing.
#
# Why these exist at all: our R=0 checkpoints are pretrained on fineweb-edu alone
# and have never seen an answer marker of any kind, so a 0-shot prompt asks them to
# invent a convention they have never observed. That measures nothing about
# mathematical capability. Demonstrating the format is what makes a capability
# floor measurable -- compare the MATH result, where 4-shot raises the boxed rate
# from 0 to 0.43-0.89 while accuracy stays at exactly 0.0000.
GSM8K_FEWSHOT_EXAMPLES = [
    {
        "problem": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "solution": "Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether in April and May.\n#### 72",
    },
    {
        "problem": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "solution": "Weng earns 12/60 = $0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $10.\n#### 10",
    },
    {
        "problem": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "solution": "In the beginning, Betty has only 100 / 2 = $50.\nBetty's grandparents gave her 15 * 2 = $30.\nThis means, Betty needs 100 - 50 - 30 - 15 = $5 more.\n#### 5",
    },
    {
        "problem": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
        "solution": "Maila read 12 x 2 = 24 pages today.\nSo she was able to read a total of 12 + 24 = 36 pages since yesterday.\nThere are 120 - 36 = 84 pages left to be read.\nSince she wants to read half of the remaining pages tomorrow, then she should read 84/2 = 42 pages.\n#### 42",
    },
]


# Template for formatting MBPP problems, following the convention of Austin et
# al. 2021 (the MBPP paper) as implemented in bigcode-evaluation-harness: the
# task description plus its test asserts, then the reference code between
# [BEGIN] and [DONE] sentinels.
#
# Two-key discipline: the whole eval/injection pipeline formats with exactly
# {problem}/{solution} (or {question}/{answer} on the injection path), so the
# test asserts are folded into the *problem* column at load time
# (`mbpp_problem_text` below) rather than adding a third template key. The
# solution string carries the trailing "\n[DONE]" sentinel itself, so that the
# eval prompt -- formatted with solution="" and rstripped -- ends at "[BEGIN]"
# and the model is expected to generate the code followed by "[DONE]".
#
# CRITICAL for contamination work (Phase 4): the injected text and the eval
# prompt must be byte-identical, mirroring GSM8K above; the two templates below
# must render the same string, which tests/test_mbpp_code_eval.py asserts.
MBPP_DOC_TO_TEXT = (
    "You are an expert Python programmer, and here is your task: {question}\n"
    "[BEGIN]\n{answer}"
)

# Same template, keyed for the evaluation path's normalized column names.
MBPP_DOC_TO_TEXT_EVAL = (
    "You are an expert Python programmer, and here is your task: {problem}\n"
    "[BEGIN]\n{solution}"
)


def mbpp_problem_text(prompt: str, test_list: List[str]) -> str:
    """Fold an MBPP task description and its test asserts into one problem string.

    Showing the asserts is MBPP's own convention (they pin down the required
    function name and signature, without which no generation could pass), and
    folding them into the problem keeps the two-key {problem}/{solution}
    template discipline the rest of the pipeline assumes.
    """
    tests = "\n".join(test_list)
    return f"{prompt} Your code should pass these tests:\n\n{tests}"


# 3-shot examples for MBPP, taken verbatim from the first three rows of the
# `google-research-datasets/mbpp` sanitized **prompt** split (task_ids 2, 3, 4)
# -- the split Austin et al. designate for few-shot prompting, disjoint from
# train/test/validation, so no evaluation item enters the prompt. Code is
# reproduced with trailing whitespace stripped per line-end (an artifact of the
# dataset's collection, not something to teach a model); each solution carries
# the closing "\n[DONE]" sentinel per the template convention above.
MBPP_FEWSHOT_EXAMPLES = [
    {
        "problem": mbpp_problem_text(
            "Write a function to find the shared elements from the given two lists.",
            [
                "assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))",
                "assert set(similar_elements((1, 2, 3, 4),(5, 4, 3, 7))) == set((3, 4))",
                "assert set(similar_elements((11, 12, 14, 13),(17, 15, 14, 13))) == set((13, 14))",
            ],
        ),
        "solution": (
            "def similar_elements(test_tup1, test_tup2):\n"
            "  res = tuple(set(test_tup1) & set(test_tup2))\n"
            "  return (res)\n[DONE]"
        ),
    },
    {
        "problem": mbpp_problem_text(
            "Write a python function to identify non-prime numbers.",
            [
                "assert is_not_prime(2) == False",
                "assert is_not_prime(10) == True",
                "assert is_not_prime(35) == True",
                "assert is_not_prime(37) == False",
            ],
        ),
        "solution": (
            "import math\n"
            "def is_not_prime(n):\n"
            "    result = False\n"
            "    for i in range(2,int(math.sqrt(n)) + 1):\n"
            "        if n % i == 0:\n"
            "            result = True\n"
            "    return result\n[DONE]"
        ),
    },
    {
        "problem": mbpp_problem_text(
            "Write a function to find the n largest integers from a given list of "
            "numbers, returned in descending order.",
            [
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75]",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]",
            ],
        ),
        "solution": (
            "import heapq as hq\n"
            "def heap_queue_largest(nums,n):\n"
            "  largest_nums = hq.nlargest(n, nums)\n"
            "  return largest_nums\n[DONE]"
        ),
    },
]


def build_fewshot_prefix(
    fewshot_examples=MINERVA_MATH_FEWSHOT_EXAMPLES,
    doc_to_text=MINERVA_MATH_DOC_TO_TEXT,
) -> str:
    """Build the few-shot prefix string from the given examples.

    Each example is formatted as "Problem:\\n{problem}\\n\\nSolution: {solution}"
    and examples are separated by double newlines. The prefix ends with a
    trailing newline so it can be directly prepended to a new problem prompt.

    Returns:
        A string containing the formatted few-shot examples.
    """
    parts = []
    for ex in fewshot_examples:
        parts.append(doc_to_text.format(problem=ex["problem"], solution=ex["solution"]))
    return "\n\n".join(parts) + "\n\n"


DEFAULT_COMPRESSION_TYPES = {
    "input_ids": Sequence(Value("int32")),
    "attention_mask": Sequence(Value("bool")),
    "token_length": Value("int32"),
}

# ---------------------------------------------------------------------------------------
# Corpus sampling. Read `docs/TOKEN_BUDGET_SHORTFALL.md` before touching these.
#
# Every published pretraining run trained on ~71.4% of its intended token budget, because
# the number of corpus documents to sample was estimated with the corpus's *advertised*
# average document length (220e9 / 190_168_005 = 1157 tokens, measured by its authors with
# their tokenizer and no truncation). Under our tokenizer, truncated at `max_length=2048`,
# the realised mean is ~786 -- 47% lower. The 1.05 headroom could not absorb a 47% error, so
# the sampled pool never reached the target; `np.searchsorted` then returned the end of the
# array and the trim that was meant to hit the budget exactly silently kept *every*
# document. No exception, no warning, and the log line printed tokens *requested*.
#
# Consequences (all verified): 14.3 tokens/parameter rather than 20, uniformly across every
# model size and overtrain multiplier (delivered/target = 0.7136-0.7141, spread +/-0.0005);
# and because the contaminant is delivered in full while the corpus is short, total tokens
# *rise* with contamination dose (+27% from R=0 to R=316).
# ---------------------------------------------------------------------------------------

# Realised mean tokens per fineweb-edu-dedup document, measured under the Qwen3 tokenizer
# with truncation at max_length=2048 (n=4,000: mean 766.5, 95% CI [748.5, 784.5], median
# 582, 9.1% truncated). The value implied by the published run logs is 786.2. Neither is
# anywhere near 1157. This is only a starting estimate -- correctness is enforced by the
# assertion in `create_dataset_for_pretraining`, not by this number being right.
CORPUS_MEAN_TOKENS_PER_DOC = 786.0

# Oversampling factor applied to the estimate above. Must be large enough to absorb the
# variance of the realised mean; 1.05 was not, which is how the bug went unnoticed.
CORPUS_SAMPLING_HEADROOM = 1.25

# Set PRETRAIN_LEGACY_TOKEN_BUDGET=1 to reproduce the published runs bit for bit, i.e. to
# restore the 1157-token estimate, the 1.05 headroom, and the silent under-delivery. This
# exists so the published results stay reproducible; do not use it for new experiments.
_LEGACY_AVG_TOKENS_PER_DOC = 220e9 / 190168005  # ~1157; wrong, kept for reproducibility
_LEGACY_SAMPLING_HEADROOM = 1.05


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

    The intent is that total training tokens per epoch is fixed -- more benchmark replicas
    means fewer corpus tokens, keeping compute constant across contamination levels.

    ⚠️ This did NOT hold for the published runs. The corpus sampling under-delivered by a
    uniform ~28.6% while the contaminant was delivered in full, so total tokens *rose* with
    contamination dose (+27% from R=0 to R=316) and every model saw ~14.3 tokens/parameter
    rather than the nominal 20. The cause and the fix are documented at the module level
    (see CORPUS_MEAN_TOKENS_PER_DOC) and in `docs/TOKEN_BUDGET_SHORTFALL.md`. The assertion
    added below makes the failure loud; it is now true for new runs, and remains false for
    anything trained before 2026-07-30 or run with PRETRAIN_LEGACY_TOKEN_BUDGET=1.

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

    # Load the benchmark — this is what `eval_benchmark_loss` is measured on.
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

    # What gets *injected* is normally the same data the loss is measured on. To test whether
    # contamination with paraphrases transfers to the original benchmark, the two must be
    # separable: inject `contaminant` (e.g. RylanSchaeffer/math_rephrased) while still
    # measuring loss on `benchmark` (the original test set). Omitting `contaminant` reproduces
    # the original exact-replica behaviour bit for bit.
    contaminant_name = data_config.get("contaminant") or data_config["benchmark"]
    if contaminant_name == data_config["benchmark"]:
        contaminant_dataset = benchmark_test_split_dataset
    else:
        contaminant_dataset = create_dataset_for_supervised_finetuning(
            dataset_name=contaminant_name,
            tokenizer=tokenizer,
            remove_columns=False,
        )["eval"]
        contaminant_dataset = contaminant_dataset.remove_columns(
            [col for col in contaminant_dataset.column_names if col not in cols_to_keep]
        )
        print(
            f"Contaminant ({contaminant_name}) differs from benchmark "
            f"({data_config['benchmark']}): injecting {len(contaminant_dataset)} "
            f"contaminant examples, measuring loss on "
            f"{len(benchmark_test_split_dataset)} benchmark examples."
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
    # Subsample the contaminant identically. The rephrased/perturbed sets are index-aligned
    # with the original, so the same seed and count select the *corresponding* problems —
    # which is what makes "did contamination with paraphrase i help on original i?" answerable.
    if contaminant_dataset is not benchmark_test_split_dataset:
        contaminant_dataset = contaminant_dataset.shuffle(
            seed=data_config["benchmark_shuffle_seed"]
        ).select(range(num_benchmark_samples_to_subsample))
    else:
        contaminant_dataset = benchmark_test_split_dataset

    # Replicate the contaminant — this is what actually enters the training corpus.
    if data_config["num_benchmark_replicas_per_epoch"] > 0:
        replicated_benchmark_test_split_dataset = concatenate_datasets(
            [
                contaminant_dataset
                for _ in range(data_config["num_benchmark_replicas_per_epoch"])
            ]
        )
    elif data_config["num_benchmark_replicas_per_epoch"] == 0:
        # Select none of the rows to create an empty dataset.
        replicated_benchmark_test_split_dataset = contaminant_dataset.select(range(0))
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
        else:
            raise ValueError

        # See the module-level comment on CORPUS_MEAN_TOKENS_PER_DOC. This is only an
        # estimate of how many documents to pull; the assertion below is what guarantees
        # the budget is actually met.
        legacy_budget = os.environ.get("PRETRAIN_LEGACY_TOKEN_BUDGET") == "1"
        if legacy_budget:
            avg_tokens_per_doc = _LEGACY_AVG_TOKENS_PER_DOC
            sampling_headroom = _LEGACY_SAMPLING_HEADROOM
            print(
                "WARNING: PRETRAIN_LEGACY_TOKEN_BUDGET=1 -- reproducing the published runs' "
                "token shortfall (~71.4% of the nominal budget). Do not use for new work."
            )
        else:
            avg_tokens_per_doc = CORPUS_MEAN_TOKENS_PER_DOC
            sampling_headroom = CORPUS_SAMPLING_HEADROOM

        estimated_docs_needed = int(
            sampling_headroom * corpus_tokens_needed_per_epoch / avg_tokens_per_doc
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

        # THE GUARD THAT WAS MISSING. If the sampled pool does not reach the target, the
        # searchsorted below returns len(cumulative_lengths), the `select` keeps *every*
        # document, and the run silently trains on less data than intended -- which is
        # exactly what happened to every published run. Fail loudly instead.
        pool_tokens = int(cumulative_lengths[-1]) if len(cumulative_lengths) else 0
        if pool_tokens < corpus_tokens_needed_per_epoch and not legacy_budget:
            raise ValueError(
                f"Sampled corpus pool holds {pool_tokens:,} tokens but "
                f"{corpus_tokens_needed_per_epoch:,} are needed "
                f"({100 * pool_tokens / corpus_tokens_needed_per_epoch:.1f}% of target). "
                f"The per-document estimate (CORPUS_MEAN_TOKENS_PER_DOC="
                f"{avg_tokens_per_doc:.1f}) is too high, or CORPUS_SAMPLING_HEADROOM="
                f"{sampling_headroom} is too small. Raise the headroom and re-run. "
                f"See docs/TOKEN_BUDGET_SHORTFALL.md."
            )

        idx_to_keep = np.searchsorted(
            cumulative_lengths, corpus_tokens_needed_per_epoch
        )
        corpus_train_dataset_subset = corpus_train_dataset_subset.select(
            range(idx_to_keep)
        )

        # Report tokens *delivered*, not tokens requested. The original log line printed
        # only the request, which is why the shortfall was invisible for the whole project.
        delivered_corpus_tokens = (
            int(cumulative_lengths[idx_to_keep - 1]) if idx_to_keep > 0 else 0
        )
        total_delivered = delivered_corpus_tokens + int(
            replicated_benchmark_test_split_num_tokens
        )
        print(
            f"Corpus tokens delivered: {delivered_corpus_tokens:,} of "
            f"{corpus_tokens_needed_per_epoch:,} requested "
            f"({100 * delivered_corpus_tokens / max(1, corpus_tokens_needed_per_epoch):.2f}%)\n"
            f"Total training tokens this epoch: {total_delivered:,} of "
            f"{num_training_tokens_per_epoch:,} targeted "
            f"({100 * total_delivered / max(1, num_training_tokens_per_epoch):.2f}%); "
            f"kept {idx_to_keep:,} of {len(cumulative_lengths):,} sampled corpus documents."
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
    elif dataset_name in {
        "RylanSchaeffer/math_rephrased",
        "RylanSchaeffer/math_perturbed",
    }:
        # Both carry `problem` and `solution` columns, so the MATH preprocessing applies
        # unchanged. Wiring these in is what makes rephrased/perturbed MATH usable as a
        # *pretraining contaminant* rather than only as an evaluation set — the experiment
        # reviewers 1wx9 and aPBL and the AC all asked for.
        if dataset_name == "RylanSchaeffer/math_rephrased":
            raw_datasets = load_dataset_math_rephrased()
        else:
            raw_datasets = load_dataset_math_perturbed()
        preprocess_fn = preprocess_eleutherai_hendrycks_math_for_sft
        doc_to_text = MINERVA_MATH_DOC_TO_TEXT
        # These ship a `test` split only. Alias it so `split_to_train_on="train"` does not
        # KeyError; there is no separate train split to confuse it with.
        if "train" not in raw_datasets:
            raw_datasets = DatasetDict(
                {"test": raw_datasets["test"], "train": raw_datasets["test"]}
            )
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


def load_dataset_math_perturbed() -> DatasetDict:
    """Load the RylanSchaeffer/math_perturbed dataset.

    This dataset contains perturbed versions of MATH problems with different
    numerical values but the same problem structure. Each problem has a full
    chain-of-thought solution. Useful for measuring generalization: models that
    memorized the original MATH test set cannot solve these perturbed variants.

    Returns:
        DatasetDict with a "test" split containing perturbed MATH problems.
    """
    return load_dataset("RylanSchaeffer/math_perturbed")


def load_dataset_math_rephrased() -> DatasetDict:
    """Load the RylanSchaeffer/math_rephrased dataset.

    This dataset contains rephrased versions of MATH problems with different
    surface wording but identical mathematical content. Created by cleaning
    stellaathena/math_rephrased (fixing 23 wrong answers, 7 stale name
    references, 1 perturbed problem, and ~68 formatting inconsistencies).

    Useful for measuring generalization: models that memorized the original
    MATH test set cannot solve these rephrased variants.

    Note: this loader is not currently reachable from
    create_dataset_for_supervised_finetuning(), which only dispatches on
    "EleutherAI/minerva_math" and "madrylab/gsm8k-platinum". Wiring it up is the
    prerequisite for using rephrased MATH as a pretraining contaminant.

    Returns:
        DatasetDict with a "test" split containing rephrased MATH problems.
    """
    return load_dataset("RylanSchaeffer/math_rephrased")


def load_dataset_gsm8k_platinum() -> DatasetDict:
    """Load the GSM8K Platinum dataset.

    GSM8K Platinum is a high-quality version of the GSM8K grade school math
    dataset, curated by MadryLab with improved answer quality.

    Returns:
        DatasetDict with a single "test" split of 1,209 rows, columns
        `question`, `answer`, `cleaning_status`. There is no train split.
    """
    return load_dataset("madrylab/gsm8k-platinum")


def load_dataset_gsm8k_platinum_for_eval() -> DatasetDict:
    """Load GSM8K Platinum with columns renamed to the evaluation convention.

    The generative eval scripts index `problem` and `solution` in a dozen places
    (prompt construction, edit distance, token counts, W&B history). Renaming here
    rather than special-casing each of those keeps the logged history schema
    identical to the MATH runs, so the same downstream notebooks work unchanged.

    Returns:
        DatasetDict whose "test" split has columns `problem`, `solution`.
    """
    raw_datasets = load_dataset_gsm8k_platinum()
    return raw_datasets.rename_columns({"question": "problem", "answer": "solution"})


def load_dataset_mbpp_sanitized() -> DatasetDict:
    """Load the sanitized (hand-verified) configuration of MBPP.

    MBPP (Austin et al. 2021, CC-BY-4.0) is 974 entry-level Python problems;
    the `sanitized` configuration is the 427-problem subset the authors
    hand-verified. Chosen per decision D3 in docs/ICLR_2027_CHECKLIST.md as the
    coding contamination substrate.

    Returns:
        DatasetDict with `train` (120), `test` (257), `validation` (43), and
        `prompt` (7) splits; columns `source_file`, `task_id`, `prompt`, `code`,
        `test_imports`, `test_list`. The `prompt` split holds the designated
        few-shot examples (hardcoded into MBPP_FEWSHOT_EXAMPLES above).
    """
    return load_dataset("google-research-datasets/mbpp", "sanitized")


def load_dataset_mbpp_for_eval() -> DatasetDict:
    """Load sanitized MBPP normalized to the evaluation convention.

    Mirrors `load_dataset_gsm8k_platinum_for_eval`: the eval scripts index
    `problem` and `solution` throughout, so `problem` becomes the task
    description with its test asserts folded in (`mbpp_problem_text`) and
    `solution` becomes the reference code. The `test_list` and `test_imports`
    columns are kept -- unlike math benchmarks, scoring MBPP needs the
    executable asserts, not the reference solution.

    Returns:
        DatasetDict whose splits have columns `problem`, `solution`,
        `test_list`, `test_imports`, `task_id`.
    """
    raw_datasets = load_dataset_mbpp_sanitized()

    def normalize(example: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "problem": mbpp_problem_text(example["prompt"], example["test_list"]),
            "solution": example["code"],
        }

    return raw_datasets.map(normalize, remove_columns=["source_file", "prompt", "code"])


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
