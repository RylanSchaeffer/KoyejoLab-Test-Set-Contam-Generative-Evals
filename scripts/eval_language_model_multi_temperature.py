"""Multi-temperature generative evaluation: one vLLM load, many temperatures.

`scripts/eval_language_model.py` loads vLLM once per (model, temperature) pair. At the
model sizes in this project vLLM startup plus the HF Hub download dominate wallclock, so
evaluating a checkpoint at three temperatures costs roughly three times what it should.

This driver loads each checkpoint **once** and loops over temperatures inside that load,
emitting **one W&B run per (model, temperature)** with byte-identical config and history
schema to `eval_language_model.py`. Downstream notebooks (11_*, 13_*) therefore need no
changes beyond adding the new run group to their filters.

Two additive history columns are logged that the original script does not:
    `has_boxed`      — whether the response contained a \\boxed{...} at all
    `response_chars` — response length in characters
These make the "is the floor genuine failure or a collapsed format rate?" check free.

Usage:
    python scripts/eval_language_model_multi_temperature.py \\
        --models-file /path/to/models.txt \\
        --temperatures 0.0 0.316 1.0 \\
        --group ot_sweep_neurips_rebuttal

The script is resumable: it queries W&B for already-finished (model, temperature) pairs in
the target group and skips them, so it can be killed and relaunched freely.
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

import argparse
import editdistance
import gc
import logging
from math_verify import parse
import time
import torch

# Compiling seems to be causing problems down the line :/
torch.compiler.disable()
from transformers import AutoTokenizer
from typing import Any, Dict, List, Set, Tuple
from vllm import LLM, SamplingParams, RequestOutput
from vllm.distributed.parallel_state import destroy_model_parallel
import wandb

import src.data
import src.globals
import src.scoring


logging.basicConfig(level=logging.INFO)

WANDB_PROJECT = "memorization-scoring-vs-sampling-eval"


def load_eval_dataset(dataset_name: str, prompt_style: str = "native"):
    """Return (test_dataset, doc_to_text) for a supported eval dataset.

    Mirrors the dispatch in `eval_language_model.py` exactly so that runs produced by the
    two scripts are directly comparable.

    All returned datasets expose `problem` and `solution` columns regardless of their
    upstream naming, so everything downstream -- prompting, edit distance, token counts,
    W&B history -- is dataset-agnostic.

    `prompt_style` matters only for GSM8K, and it exists to separate two explanations
    of a zero score. Our checkpoints are pretrained (and SFT'd) on MATH's
    "Problem:/Solution:" format, so prompting them with GSM8K's native "Q:/A:" puts
    them out of distribution: a zero could mean "cannot solve grade-school math" or
    merely "has never seen this prompt shape". Scoring both styles distinguishes the
    two. Use "native" for anything that will be compared against GSM8K-contaminated
    runs, since that is the format those models will have been trained on.
    """
    if dataset_name == "EleutherAI/minerva_math":
        raw_datasets = src.data.load_dataset_hendrycks_math()
    elif dataset_name == "RylanSchaeffer/math_perturbed":
        raw_datasets = src.data.load_dataset_math_perturbed()
    elif dataset_name == "RylanSchaeffer/math_rephrased":
        raw_datasets = src.data.load_dataset_math_rephrased()
    elif dataset_name == "madrylab/gsm8k-platinum":
        raw_datasets = src.data.load_dataset_gsm8k_platinum_for_eval()
        doc_to_text = (
            src.data.MINERVA_MATH_DOC_TO_TEXT
            if prompt_style == "minerva"
            else src.data.GSM8K_PLATINUM_DOC_TO_TEXT_EVAL
        )
        return raw_datasets["test"], doc_to_text
    else:
        raise NotImplementedError(dataset_name)
    return raw_datasets["test"], src.data.MINERVA_MATH_DOC_TO_TEXT


def is_gsm8k(dataset_name: str) -> bool:
    """Whether `dataset_name` uses GSM8K's "#### <n>" answer convention.

    GSM8K golds are not LaTeX and contain no \\boxed{}, so math_verify's parse()
    cannot supply a gold and the boxed-required scorer would return 0.0 uniformly.
    """
    return dataset_name == "madrylab/gsm8k-platinum"


def fetch_completed_pairs(group: str) -> Set[Tuple[str, float]]:
    """Return the (model, temperature) pairs already finished in `group`.

    Used for resumption. A W&B failure here is non-fatal: we fall back to an empty set,
    which merely means redoing work rather than losing it.
    """
    completed: Set[Tuple[str, float]] = set()
    try:
        api = wandb.Api(timeout=600)
        runs = api.runs(
            f"{api.default_entity}/{WANDB_PROJECT}",
            filters={"group": group, "state": "finished"},
            per_page=200,
        )
        for run in runs:
            try:
                completed.add(
                    (
                        run.config["model_config"]["model"],
                        float(run.config["temperature"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    except Exception as e:
        logging.warning(
            f"Could not query W&B for completed runs ({e}); starting fresh."
        )
    return completed


def score_and_log(
    requests_outputs: List[RequestOutput],
    test_dataset,
    tokenizer,
    wandb_log_sleep: float,
    dataset_name: str = "EleutherAI/minerva_math",
) -> float:
    """Score generations, log per-problem history to the active W&B run, return accuracy.

    History schema matches `eval_language_model.py` (plus `has_boxed`/`response_chars`).
    The `math_verify_score` key is retained for GSM8K even though math_verify is not the
    scorer there, so that existing downstream aggregation keeps working unchanged.
    """
    problem_responses: List[str] = []
    log_probs_per_problem_response: List[List[float]] = []
    for request_outputs in requests_outputs:
        problem_responses.append(request_outputs.outputs[0].text)
        log_probs_list_of_dicts = request_outputs.outputs[0].logprobs
        log_probs_per_token = [
            list(d.values())[0].logprob for d in log_probs_list_of_dicts
        ]
        log_probs_per_problem_response.append(log_probs_per_token)

    solutions = test_dataset["solution"]
    if is_gsm8k(dataset_name):
        gold_answers = [
            src.scoring.extract_gsm8k_gold_answer(solution) for solution in solutions
        ]
        missing_golds = sum(1 for gold in gold_answers if gold is None)
        if missing_golds:
            raise ValueError(
                f"{missing_golds} of {len(gold_answers)} GSM8K golds lack a '####' "
                "marker; refusing to score against an unparseable reference."
            )
        results = [
            src.scoring.score_gsm8k_response(gold_answer=gold, response_text=response)
            for gold, response in zip(gold_answers, problem_responses)
        ]
    else:
        results = [
            src.scoring.score_response(
                gold_parsed=parse(solution),
                response_text=response,
            )
            for solution, response in zip(solutions, problem_responses)
        ]
    math_verify_scores = [1 if res else 0 for res in results]
    has_boxed = [
        1 if src.scoring.extract_boxed_answer(response) is not None else 0
        for response in problem_responses
    ]
    edit_distances = [
        editdistance.eval(solution, response)
        for solution, response in zip(solutions, problem_responses)
    ]
    tokens_per_solution = [len(ids) for ids in tokenizer(solutions).input_ids]
    tokens_per_response = [len(ids) for ids in tokenizer(problem_responses).input_ids]

    for problem_idx in range(len(requests_outputs)):
        problem_data_to_log = {
            "problem_idx": problem_idx,
            "token_per_solution": tokens_per_solution[problem_idx],
            "token_per_response": tokens_per_response[problem_idx],
            "solution": solutions[problem_idx],
            "response": problem_responses[problem_idx],
            "edit_distance": edit_distances[problem_idx],
            "math_verify_score": math_verify_scores[problem_idx],
            "has_boxed": has_boxed[problem_idx],
            "response_chars": len(problem_responses[problem_idx]),
        }

        log_probs_per_problem = log_probs_per_problem_response[problem_idx]
        for token_idx in range(len(log_probs_per_problem)):
            problem_data_to_log[f"log_prob_token_{token_idx}"] = log_probs_per_problem[
                token_idx
            ]

        wandb.log(problem_data_to_log, step=problem_idx + 1)
        # Be nicer to W&B, even if that takes more time per run.
        time.sleep(wandb_log_sleep)

    return sum(math_verify_scores) / len(math_verify_scores)


def load_model_with_retries(model_config: Dict[str, Any], attempts: int) -> LLM:
    """Construct the vLLM engine, retrying transient co-tenancy failures.

    When two workers share a GPU, one can finish and free its memory while the other is in
    `determine_available_memory`. vLLM asserts on that (`Initial free memory X, current free
    memory Y ... other processes sharing the same container release GPU memory while vLLM is
    profiling`) and the engine dies. It is purely a race: retrying once the other worker has
    settled succeeds. Backoff grows so a retry does not land inside the next teardown.
    """
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return LLM(**model_config)
        except Exception as e:
            last_error = e
            # A missing repo is not a race — retrying a 404 just burns the backoff schedule on
            # every checkpoint. Fail fast so the wrong-name case is obvious in the log.
            if "RepositoryNotFound" in type(e).__name__ or "404 Client Error" in str(e):
                logging.error(
                    f"[abort] {model_config['model']}: repository not found; not retrying."
                )
                raise
            if attempt == attempts - 1:
                break
            delay = 30 * (attempt + 1)
            logging.warning(
                f"[retry] engine init failed (attempt {attempt + 1}/{attempts}): "
                f"{type(e).__name__}: {e}. Retrying in {delay}s."
            )
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(delay)
    raise last_error


def evaluate_one_model(
    model_name: str,
    temperatures: List[float],
    test_dataset,
    formatted_problems: List[str],
    args: argparse.Namespace,
    completed: Set[Tuple[str, float]],
) -> None:
    """Load `model_name` once and evaluate it at every temperature not already done."""
    remaining = [t for t in temperatures if (model_name, t) not in completed]
    if not remaining:
        logging.info(f"[skip] {model_name}: all temperatures already finished.")
        return

    model_config = {
        "model": model_name,
        "dtype": args.dtype,
        "enforce_eager": True,
    }
    if args.gpu_memory_utilization is not None:
        # Only set when overridden, so that runs left at the default keep a config
        # identical to the ones produced by the original sweeps.
        model_config["gpu_memory_utilization"] = args.gpu_memory_utilization

    logging.info(f"[load] {model_name} -> temperatures {remaining}")
    load_start = time.time()
    model = load_model_with_retries(model_config, attempts=args.load_attempts)
    logging.info(f"[load] {model_name} took {time.time() - load_start:.1f}s")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=True
    )

    try:
        for temperature in remaining:
            gen_start = time.time()
            requests_outputs: List[RequestOutput] = model.generate(
                prompts=formatted_problems,
                sampling_params=SamplingParams(
                    temperature=temperature,
                    max_tokens=args.max_tokens,
                    seed=args.seed,
                    logprobs=1,  # Return 1 log probability per sequence.
                ),
            )
            gen_seconds = time.time() - gen_start

            run_config: Dict[str, Any] = {
                "data_config": {
                    "dataset": args.dataset,
                    "shuffle_seed": args.shuffle_seed,
                },
                "max_tokens": args.max_tokens,
                "model_config": model_config,
                "seed": args.seed,
                "temperature": temperature,
                # Not present in the original sweeps' configs, where the shot count was
                # implicit in the script version. Recorded here so 0-shot and 4-shot runs
                # can never be silently pooled in analysis.
                "num_fewshot": args.num_fewshot,
                # Same reasoning: two prompt styles measure different things on GSM8K
                # and must never be pooled.
                "prompt_style": args.prompt_style,
            }
            wandb.init(
                project=WANDB_PROJECT,
                entity=wandb.api.default_entity,
                config=run_config,
                group=args.group,
                tags=args.tags,
                reinit=True,
            )
            try:
                accuracy = score_and_log(
                    requests_outputs=requests_outputs,
                    test_dataset=test_dataset,
                    tokenizer=tokenizer,
                    wandb_log_sleep=args.wandb_log_sleep,
                    dataset_name=args.dataset,
                )
                logging.info(
                    f"[done] {model_name} tau={temperature} "
                    f"math_verify={accuracy:.4f} gen={gen_seconds:.1f}s"
                )
            finally:
                wandb.finish()
    finally:
        # Freeing up VLLM memory is harder than I thought!
        # See: https://github.com/vllm-project/vllm/issues/1908
        # Hit it with everything recommended!
        destroy_model_parallel()
        del model
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-file",
        required=True,
        help="Text file with one HF model id per line ('#' comments and blanks ignored).",
    )
    parser.add_argument(
        "--temperatures", nargs="+", type=float, default=[0.0, 0.316, 1.0]
    )
    parser.add_argument("--dataset", default="EleutherAI/minerva_math")
    parser.add_argument(
        "--prompt-style",
        default="native",
        choices=["native", "minerva"],
        help="GSM8K only. 'native' uses GSM8K's Q:/A: format; 'minerva' uses MATH's "
        "Problem:/Solution: format, which is in-distribution for MATH-trained "
        "checkpoints and so separates a capability floor from a prompt-format mismatch.",
    )
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=4,
        choices=[0, 4],
        help="0 for the bare 'Problem/Solution:' prompt, 4 for the few-shot prefix. These "
        "measure very different things on contaminated checkpoints; see module docstring.",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="vLLM gpu_memory_utilization. Lower it (e.g. 0.42) to fit several workers on "
        "one GPU; at these model sizes a single worker leaves the GPU ~50%% idle, so "
        "oversubscribing raises aggregate throughput. Omit to keep vLLM's default.",
    )
    parser.add_argument(
        "--group",
        required=True,
        help="W&B group; also the key used to detect already-finished runs on resume.",
    )
    parser.add_argument("--tags", nargs="*", default=[])
    parser.add_argument(
        "--wandb-log-sleep",
        type=float,
        default=0.01,
        help="Per-step sleep while streaming history to W&B.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="This worker's index; models are round-robin sharded across workers.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--load-attempts",
        type=int,
        default=4,
        help="Attempts at constructing the vLLM engine before giving up on a checkpoint. "
        "Co-resident workers race in vLLM's memory profiling; retries fix it.",
    )
    args = parser.parse_args()

    assert torch.cuda.device_count() > 0, "No CUDA devices available."

    with open(args.models_file) as f:
        all_models = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]
    models = all_models[args.shard_index :: args.num_shards]
    logging.info(
        f"Shard {args.shard_index}/{args.num_shards}: {len(models)} of "
        f"{len(all_models)} checkpoints, temperatures {args.temperatures}"
    )

    test_dataset, doc_to_text = load_eval_dataset(args.dataset, args.prompt_style)
    # The two protocols are not interchangeable: the same contaminated checkpoint scores
    # ~1.00 at 0-shot (the prompt matches the memorized document's opening, so the model
    # regurgitates the solution verbatim) and ~0.005 at 4-shot. Which one a sweep uses must
    # therefore be recorded in the run config, not just in the launch command.
    if is_gsm8k(args.dataset) and args.num_fewshot != 0:
        # build_fewshot_prefix() formats MATH problems with MATH's template; there
        # is no GSM8K few-shot example set. Fail rather than prepend four MATH
        # problems to a GSM8K prompt.
        raise NotImplementedError(
            "GSM8K evaluation supports 0-shot only; no GSM8K few-shot examples exist."
        )
    fewshot_prefix = "" if args.num_fewshot == 0 else src.data.build_fewshot_prefix()
    formatted_problems = [
        fewshot_prefix + doc_to_text.format(problem=question, solution="").rstrip()
        for question in test_dataset["problem"]
    ]
    logging.info(
        f"{len(formatted_problems)} eval problems from {args.dataset} "
        f"({args.num_fewshot}-shot)"
    )

    completed = fetch_completed_pairs(group=args.group)
    logging.info(f"{len(completed)} (model, temperature) pairs already finished.")

    for model_idx, model_name in enumerate(models):
        logging.info(f"=== [{model_idx + 1}/{len(models)}] {model_name}")
        try:
            evaluate_one_model(
                model_name=model_name,
                temperatures=args.temperatures,
                test_dataset=test_dataset,
                formatted_problems=formatted_problems,
                args=args,
                completed=completed,
            )
        except Exception as e:
            # One bad checkpoint (missing weights, OOM) must not sink the whole sweep.
            logging.exception(f"[FAIL] {model_name}: {e}")
            if wandb.run is not None:
                wandb.finish(exit_code=1)


if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(i) for i in range(torch.cuda.device_count())]
        )
    main()
    logging.info("Finished eval_language_model_multi_temperature.py!")
