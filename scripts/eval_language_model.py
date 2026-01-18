"""Evaluation script for math problem-solving with vLLM inference.

This script evaluates language models on math benchmarks using:
- vLLM for efficient batched inference
- math-verify for answer correctness scoring
- Edit distance metrics for measuring generation similarity

Supports both greedy decoding (temperature=0) and stochastic sampling
to study how contaminated models behave under different decoding strategies.

Usage:
    python scripts/eval_language_model.py

    # As part of a W&B sweep
    wandb agent <sweep-id>

Results are logged to W&B including per-problem metrics (accuracy, logprobs,
edit distance to ground truth, solution length).
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

import editdistance
import gc
import logging
from math_verify import parse, verify
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


def eval_language_model():
    assert torch.cuda.device_count() > 0, "No CUDA devices available."
    run = wandb.init(
        project="memorization-scoring-vs-sampling-eval",
        config=src.globals.DEFAULT_EVALUATION_CONFIG,
        entity=wandb.api.default_entity,
    )

    # Convert to a dictionary; otherwise, can't distribute because W&B
    # config is not pickle-able.
    wandb_config: Dict[str, Any] = dict(wandb.config)
    print("CUDA VISIBLE DEVICES: ", os.environ["CUDA_VISIBLE_DEVICES"])
    pprint.pprint(wandb_config)

    run_lm_eval_custom(wandb_config=wandb_config)
    wandb.finish()


def run_lm_eval_custom(wandb_config: Dict[str, Any]) -> Dict[str, float]:
    # Create the dataset.
    if wandb_config["data_config"]["dataset"] == "EleutherAI/minerva_math":
        raw_datasets = src.data.load_dataset_hendrycks_math()
        test_dataset = raw_datasets["test"]
        doc_to_text = src.data.MINERVA_MATH_DOC_TO_TEXT
        formatted_problems = [
            doc_to_text.format(problem=question, solution="").rstrip()
            for question in test_dataset["problem"]
        ]
    else:
        raise NotImplementedError

    # For debugging purposes...
    # # Cap vLLM to ~10 GiB per GPU.
    # total_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    # util = round(10.0 / total_gib, 3)  # ~0.126 on 80 GiB
    # wandb_config["model_config"]["gpu_memory_utilization"] = util

    # Create the model and sampling parameters.
    model = LLM(**wandb_config["model_config"])
    model_sampling_params = SamplingParams(
        temperature=wandb_config["temperature"],
        max_tokens=wandb_config["max_tokens"],
        seed=wandb_config["seed"],
        logprobs=1,  # Return 1 log probability per sequence.
    )

    # Sample from the model.
    requests_outputs: List[RequestOutput] = model.generate(
        prompts=formatted_problems, sampling_params=model_sampling_params
    )

    # Freeing up VLLM memory is harder than I thought!
    # See: https://github.com/vllm-project/vllm/issues/1908
    # Hit it with everything recommended!
    destroy_model_parallel()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    problem_responses: List[str] = []
    log_probs_per_problem_response: List[List[float]] = []
    for request_outputs in requests_outputs:
        problem_responses.append(request_outputs.outputs[0].text)
        log_probs_list_of_dicts = request_outputs.outputs[0].logprobs
        log_probs_per_token = [
            list(d.values())[0].logprob for d in log_probs_list_of_dicts
        ]
        log_probs_per_problem_response.append(log_probs_per_token)

    results = [
        verify(
            gold=parse(solution),
            target=parse(response),
        )
        for solution, response in zip(test_dataset["solution"], problem_responses)
    ]
    math_verify_scores = [1 if res else 0 for res in results]
    solutions = test_dataset["solution"]
    edit_distances = [
        editdistance.eval(solution, response)
        for solution, response in zip(solutions, problem_responses)
    ]

    tokenizer = AutoTokenizer.from_pretrained(
        wandb_config["model_config"]["model"],
        use_fast=True,
        trust_remote_code=True,
    )
    tokens_per_solution = [
        len(ids) for ids in tokenizer(test_dataset["solution"]).input_ids
    ]
    tokens_per_response = [len(ids) for ids in tokenizer(problem_responses).input_ids]

    for problem_idx in range(len(requests_outputs)):
        # Log the main data.
        problem_data_to_log = {
            "problem_idx": problem_idx,
            "token_per_solution": tokens_per_solution[problem_idx],
            "token_per_response": tokens_per_response[problem_idx],
            "solution": solutions[problem_idx],
            "response": problem_responses[problem_idx],
            "edit_distance": edit_distances[problem_idx],
            "math_verify_score": math_verify_scores[problem_idx],
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
    eval_language_model()
    logging.info("Finished eval_language_model.py!")
