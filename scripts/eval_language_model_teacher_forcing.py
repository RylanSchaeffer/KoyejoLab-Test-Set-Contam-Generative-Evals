"""Teacher-forced evaluation script for math problem-solving with vLLM.

This script evaluates language models on math benchmarks by computing
teacher-forced log probabilities - i.e., the log probability of the
ground-truth solution given each problem, without sampling.

This is useful for studying memorization because a model that has memorized
the test set solutions will assign higher log probability to those solutions.

Usage:
    python scripts/eval_language_model_teacher_forcing.py

    # As part of a W&B sweep
    wandb agent <sweep-id>

Results are logged to W&B including per-problem metrics (log probabilities,
solution length in tokens).
"""

import os

# Rok asked us to include the following specifications in our code to prevent CPUs from spinning idly:
n_threads_str = "4"
os.environ["OMP_NUM_THREADS"] = n_threads_str
os.environ["OPENBLAS_NUM_THREADS"] = n_threads_str
os.environ["MKL_NUM_THREADS"] = n_threads_str
os.environ["VECLIB_MAXIMUM_THREADS"] = n_threads_str
os.environ["NUMEXPR_NUM_THREADS"] = n_threads_str
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "True"

# This is needed for deterministic to work.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# 16.48 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large
# try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import logging
import pprint
import time
import torch

# Compiling seems to be causing problems down the line :/
torch.compiler.disable()
from transformers import AutoTokenizer
from typing import Any, Dict, List
from vllm import LLM, SamplingParams, RequestOutput
from vllm.distributed.parallel_state import destroy_model_parallel
import wandb

import src.data
import src.globals


logging.basicConfig(level=logging.INFO)


def eval_language_model_teacher_forcing():
    assert torch.cuda.device_count() > 0, "No CUDA devices available."
    run = wandb.init(
        project="memorization-scoring-vs-sampling-eval-teacher-forcing",
        config=src.globals.DEFAULT_TEACHER_FORCING_EVALUATION_CONFIG,
        entity=wandb.api.default_entity,
    )

    # Convert to a dictionary; otherwise, can't distribute because W&B
    # config is not pickle-able.
    wandb_config: Dict[str, Any] = dict(wandb.config)
    print("CUDA VISIBLE DEVICES: ", os.environ["CUDA_VISIBLE_DEVICES"])
    pprint.pprint(wandb_config)

    run_lm_eval_teacher_forcing(wandb_config=wandb_config)
    wandb.finish()


def run_lm_eval_teacher_forcing(wandb_config: Dict[str, Any]) -> Dict[str, float]:
    # Create the dataset.
    if wandb_config["data_config"]["dataset"] == "EleutherAI/minerva_math":
        raw_datasets = src.data.load_dataset_hendrycks_math()
        test_dataset = raw_datasets["test"]
        doc_to_text = src.data.MINERVA_MATH_DOC_TO_TEXT
    else:
        raise NotImplementedError

    # Create prompts (without solutions) and full sequences (with solutions).
    prompts = [
        doc_to_text.format(problem=question, solution="").rstrip()
        for question in test_dataset["problem"]
    ]
    solutions = test_dataset["solution"]
    full_sequences = [
        doc_to_text.format(problem=question, solution=solution)
        for question, solution in zip(test_dataset["problem"], solutions)
    ]

    # Create the model.
    model = LLM(**wandb_config["model_config"])

    # Use prompt_logprobs to get log probabilities for the full sequence.
    # max_tokens=1 because we need at least 1 token to generate, but we only
    # care about the prompt_logprobs (which include the solution tokens).
    model_sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        seed=wandb_config["seed"],
        prompt_logprobs=1,  # Return log probs for the prompt tokens.
    )

    # Generate with full sequences (prompt + solution) as input.
    requests_outputs: List[RequestOutput] = model.generate(
        prompts=full_sequences, sampling_params=model_sampling_params
    )

    # Freeing up VLLM memory is harder than I thought!
    # See: https://github.com/vllm-project/vllm/issues/1908
    # Hit it with everything recommended!
    destroy_model_parallel()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Load tokenizer to compute token counts and prompt lengths.
    tokenizer = AutoTokenizer.from_pretrained(
        wandb_config["model_config"]["model"],
        use_fast=True,
        trust_remote_code=True,
    )

    # Tokenize to get prompt lengths (needed to extract solution log probs).
    prompt_token_ids = tokenizer(prompts, add_special_tokens=False).input_ids
    prompt_lengths = [len(ids) for ids in prompt_token_ids]

    # Tokenize solutions to get token counts.
    tokens_per_solution = [len(ids) for ids in tokenizer(solutions).input_ids]
    tokens_per_response = [len(ids) for ids in tokenizer(solutions).input_ids]

    # Extract log probabilities for the solution portion only.
    log_probs_per_problem_response: List[List[float]] = []
    for problem_idx, request_output in enumerate(requests_outputs):
        prompt_length = prompt_lengths[problem_idx]

        # prompt_logprobs is a list where each entry corresponds to a token.
        # The first token has None (no conditioning), so we skip it.
        # We want log probs for solution tokens, which start after the prompt.
        all_prompt_logprobs = request_output.prompt_logprobs

        solution_log_probs = []
        # Iterate over tokens after the prompt (these are the solution tokens).
        for token_idx in range(prompt_length, len(all_prompt_logprobs)):
            token_logprob_dict = all_prompt_logprobs[token_idx]
            if token_logprob_dict is not None:
                # Get the log prob of the actual token that appeared.
                # The dict maps token_id -> Logprob object.
                logprob_value = list(token_logprob_dict.values())[0].logprob
                solution_log_probs.append(logprob_value)

        log_probs_per_problem_response.append(solution_log_probs)

    for problem_idx in range(len(requests_outputs)):
        # Log the main data.
        problem_data_to_log = {
            "problem_idx": problem_idx,
            "token_per_solution": tokens_per_solution[problem_idx],
            "token_per_response": tokens_per_response[problem_idx],
            "solution": solutions[problem_idx],
            "response": solutions[problem_idx],
        }

        # Add the log probability of each token.
        log_probs_per_problem = log_probs_per_problem_response[problem_idx]
        for token_idx in range(len(log_probs_per_problem)):
            problem_data_to_log[f"log_prob_token_{token_idx}"] = log_probs_per_problem[
                token_idx
            ]

        wandb.log(problem_data_to_log, step=problem_idx + 1)
        # Be nicer to W&B, even if that takes more time per run.
        time.sleep(1.0 / 10.0)


if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(i) for i in range(torch.cuda.device_count())]
        )
    eval_language_model_teacher_forcing()
    logging.info("Finished eval_language_model_teacher_forcing.py!")
