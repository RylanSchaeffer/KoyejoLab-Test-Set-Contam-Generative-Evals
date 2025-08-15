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
import pprint
from math_verify import parse, verify
import numpy as np
import torch
from tqdm import tqdm

# Compiling seems to be causing problems down the line :/
torch.compiler.disable()
from typing import Any, Dict, List
from vllm import LLM, SamplingParams, RequestOutput
from vllm.distributed.parallel_state import destroy_model_parallel
import wandb

import src.data
import src.globals
import src.models


logging.basicConfig(level=logging.INFO)


def eval_language_model():
    num_visible_devices = torch.cuda.device_count()
    assert num_visible_devices > 0, "No CUDA devices available."
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

    # Create the dataset.
    if wandb_config["data_config"]["dataset"] == "EleutherAI/minerva_math":
        raw_datasets = src.data.load_dataset_hendrycks_math()
        test_dataset = raw_datasets["test"]
        doc_to_text = src.data.MINERVA_MATH_DOC_TO_TEXT
        formatted_problems = [
            doc_to_text.format(problem=question, solution="").rstrip()
            for question in test_dataset["problem"]
        ]
        formatted_questions_and_answers = [
            f"{formatted_problem} {solution}"
            for formatted_problem, solution in zip(
                formatted_problems, test_dataset["solution"]
            )
        ]
    else:
        raise NotImplementedError

    # Create the model and sampling parameters.
    model = LLM(**wandb_config["model_config"])
    model_sampling_params = SamplingParams(
        temperature=wandb_config["temperature"],
        max_tokens=wandb_config["max_tokens"],
        seed=wandb_config["seed"],
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

    responses = [
        output.text
        for batch_request_output in requests_outputs
        for output in batch_request_output.outputs
    ]

    results = [
        verify(
            gold=solution,
            target=response,
        )
        for solution, response in zip(test_dataset["solution"], responses)
    ]
    math_verify_scores = [1 if res else 0 for res in results]
    edit_distances = [
        editdistance.eval(solution, response)
        for solution, response in zip(test_dataset["solution"], responses)
    ]

    data_to_log = {
        f"math_verify_{i}": score for i, score in enumerate(math_verify_scores)
    }
    data_to_log.update(
        {f"edit_distance_{i}": score for i, score in enumerate(edit_distances)}
    )
    data_to_log["math_verify_mean"] = np.mean(math_verify_scores)
    data_to_log["math_verify_median"] = np.median(math_verify_scores)
    data_to_log["math_verify_max"] = np.max(math_verify_scores)
    data_to_log["math_verify_min"] = np.min(math_verify_scores)
    data_to_log["edit_distance_mean"] = np.mean(edit_distances)
    data_to_log["edit_distance_median"] = np.median(edit_distances)
    data_to_log["edit_distance_max"] = np.max(edit_distances)
    data_to_log["edit_distance_min"] = np.min(edit_distances)
    wandb.log(data_to_log)
    wandb.finish()


# def run_lm_eval_with_vllm(
#     model_hf_path: str,
#     lm_eval_task: str,
#     lm_eval_metric: str,
#     num_fewshot: int = 0,
#     seed: int = 0,
#     temperature: float = 1.0,
# ):
#     do_sample = True if temperature > 0.0 else False
#
#     command = f"""mem_scoring_vs_sampling_env/bin/lm_eval \
#     --model vllm \
#     --model_args pretrained={model_hf_path},dtype=auto,max_model_len=2048,max_num_seqs=2048 \
#     --batch_size auto \
#     --tasks {lm_eval_task} \
#     --num_fewshot {num_fewshot} \
#     --log_samples \
#     --output_path ./lm-eval-output/ \
#     --gen_kwargs temperature={temperature},do_sample={do_sample} \
#     --seed {seed}
#     """
#
#     logging.info(f"command: {command}")
#
#     try:
#         env = os.environ.copy()
#
#         process = subprocess.run(
#             command,
#             shell=True,
#             check=True,  # This is what raises the CalledProcessError
#             text=True,
#             capture_output=True,
#             env=env,
#         )
#         logging.info(process.stdout)
#         scores = extract_scores_from_output(process.stdout, eval_metric=lm_eval_metric)
#         logging.info(scores)
#
#     except subprocess.CalledProcessError as e:
#         # Handle the error
#         logging.error(f"Command failed with exit code {e.returncode}")
#         logging.error(f"STDOUT: {e.stdout}")
#         logging.error(f"STDERR: {e.stderr}")
#         raise e
#
#     return scores
#
#
# def extract_scores_from_output(
#     output_text: str, eval_metric: str = "exact_match"
# ) -> Dict[str, float]:
#     """Extract exact_match scores from the lm-eval output text."""
#     results = {}
#
#     lines = output_text.strip().split("\n")
#     for line in lines:
#         if eval_metric in line:
#             # Parse the line in the table that contains exact_match
#             parts = line.split("|")
#             if len(parts) >= 8:
#                 filter_type = parts[3].strip()
#                 eval_metric = parts[5].strip()
#                 value = float(parts[7].strip().split("±")[0])
#
#                 # Create a meaningful key
#                 key = f"{eval_metric}_{filter_type}".replace("-", "_")
#                 results[key] = value
#
#     return results


if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(i) for i in range(torch.cuda.device_count())]
        )
    eval_language_model()
    logging.info("Finished eval_language_model.py!")
