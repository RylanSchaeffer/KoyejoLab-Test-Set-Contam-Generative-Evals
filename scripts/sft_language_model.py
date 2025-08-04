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

import logging
import gc
import pprint
import subprocess
import time
import torch

# Compiling seems to be causing problems down the line :/
torch.compiler.disable()
from transformers import AutoImageProcessor, AutoTokenizer, set_seed
from trl import (
    SFTConfig,
    SFTTrainer,
)
from typing import Any, Dict
import wandb

import src.data
import src.globals
import src.models


logging.basicConfig(level=logging.INFO)


def train_supervised_finetuning():
    num_visible_devices = torch.cuda.device_count()
    assert num_visible_devices > 0, "No CUDA devices available."
    run = wandb.init(
        project="memorization-scoring-vs-sampling",
        config=src.globals.DEFAULT_SUPERVISED_FINETUNING_CONFIG,
        entity=wandb.api.default_entity,
    )

    # Convert to a dictionary; otherwise, can't distribute because W&B
    # config is not pickle-able.
    wandb_config: Dict[str, Any] = dict(wandb.config)
    print("CUDA VISIBLE DEVICES: ", os.environ["CUDA_VISIBLE_DEVICES"])
    pprint.pprint(wandb_config)

    if wandb_config["data_config"]["dataset"] == "madrylab/gsm8k-platinum":
        lm_eval_task = "gsm8k_platinum_cot"
    else:
        raise NotImplementedError

    for temperature in [0.0, 0.316, 1.0]:
        lm_eval_results_before = run_lm_eval_with_vllm(
            model_hf_path=wandb_config["model_config"]["initial_model_name_or_path"],
            lm_eval_task=lm_eval_task,
            num_fewshot=0,
            seed=wandb_config["seed"],
            temperature=temperature,
        )
        wandb.log(
            {
                f"lm_eval_before_temp={temperature}/{k}": v
                for k, v in lm_eval_results_before.items()
            },
            commit=True,
        )

    # Create output directory.
    sfted_model_hf_name = create_sfted_model_huggingface_name(
        wandb_config=wandb_config,
    )
    output_dir = os.path.join("models", "sft_language_model", sfted_model_hf_name)
    print("Output Directory: ", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    effective_global_batch_size = (
        num_visible_devices
        * wandb_config["sft_trainer_config"]["per_device_train_batch_size"]
        * wandb_config["sft_trainer_config"]["gradient_accumulation_steps"]
    )
    wandb.config.update(
        {
            "output_dir": output_dir,
            "effective_global_batch_size": effective_global_batch_size,
        },
    )

    set_seed(seed=wandb_config["seed"], deterministic=True)

    sft_trainer_config_dict: Dict[str, Any] = wandb_config["sft_trainer_config"]
    model_config_dict: Dict[str, Any] = wandb_config["model_config"]
    data_config_dict: Dict[str, Any] = wandb_config["data_config"]

    # Lightly modify configs as necessary.
    if model_config_dict["torch_dtype"] == "bfloat16":
        sft_trainer_config_dict["bf16"] = True
    else:
        sft_trainer_config_dict["bf16"] = False
    if model_config_dict["torch_dtype"] == "float16":
        sft_trainer_config_dict["fp16"] = True
    else:
        sft_trainer_config_dict["fp16"] = False

    sft_config = SFTConfig(
        bf16=sft_trainer_config_dict["bf16"],
        data_seed=sft_trainer_config_dict["data_seed"],
        dataloader_num_workers=sft_trainer_config_dict["dataloader_num_workers"],
        dataloader_prefetch_factor=sft_trainer_config_dict[
            "dataloader_prefetch_factor"
        ],
        dataset_text_field="input_ids",
        eval_on_start=sft_trainer_config_dict["eval_on_start"],
        eval_strategy=sft_trainer_config_dict["eval_strategy"],
        eval_steps=sft_trainer_config_dict["eval_steps"],
        fp16=sft_trainer_config_dict["fp16"],
        gradient_accumulation_steps=sft_trainer_config_dict[
            "gradient_accumulation_steps"
        ],
        gradient_checkpointing=sft_trainer_config_dict["gradient_checkpointing"],
        hub_model_id=f"RylanSchaeffer/{sfted_model_hf_name}",
        hub_private_repo=True,
        hub_strategy=sft_trainer_config_dict["hub_strategy"],
        include_num_input_tokens_seen=True,
        learning_rate=sft_trainer_config_dict["learning_rate"],
        logging_steps=sft_trainer_config_dict["logging_steps"],
        lr_scheduler_type=sft_trainer_config_dict["lr_scheduler_type"],
        max_length=sft_trainer_config_dict["max_length"],
        max_steps=sft_trainer_config_dict["max_steps"],
        metric_for_best_model="eval_loss",
        num_train_epochs=sft_trainer_config_dict["num_train_epochs"],
        optim=sft_trainer_config_dict["optim"],
        output_dir=output_dir,
        per_device_eval_batch_size=sft_trainer_config_dict[
            "per_device_eval_batch_size"
        ],
        per_device_train_batch_size=sft_trainer_config_dict[
            "per_device_train_batch_size"
        ],
        remove_unused_columns=sft_trainer_config_dict["remove_unused_columns"],
        run_name=wandb.run.id,
        report_to=sft_trainer_config_dict["report_to"],
        save_strategy=sft_trainer_config_dict["save_strategy"],
        save_total_limit=sft_trainer_config_dict["save_total_limit"],
        seed=wandb_config["seed"],
        warmup_ratio=sft_trainer_config_dict["warmup_ratio"],
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_config_dict["initial_model_name_or_path"],
        use_fast=True,
        trust_remote_code=True,
    )

    # trl/trainer/sft_trainer.py:408: UserWarning: You passed a tokenizer with padding_side not equal
    # to right to the SFTTrainer. This might lead to some unexpected behaviour due to overflow issues
    # when training a model in half-precision. You might consider adding tokenizer.padding_side = 'right' to your code.
    tokenizer.padding_side = "right"

    datasets_dict = src.data.create_dataset_for_supervised_finetuning(
        tokenizer=tokenizer,
        dataset_name=data_config_dict["dataset"],
        max_length=sft_config.max_length,
    )
    train_dataset = datasets_dict["train"]
    eval_dataset = datasets_dict["eval"]

    model = src.models.load_automodelforcausallm(
        model_config_dict=model_config_dict,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        # tokenizer=tokenizer,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Evaluate before training.
    logging.info("Evaluating before training...")
    eval_metrics_before = trainer.evaluate()
    wandb.log({f"eval_before/{k}": v for k, v in eval_metrics_before.items()})
    pprint.pprint(eval_metrics_before)

    # Train.
    logging.info("Beginning training...")
    trainer.train()

    # Evaluate after training.
    logging.info("Finished training. Beginning final evaluation...")
    eval_metrics_after = trainer.evaluate()
    wandb.log({f"eval_after/{k}": v for k, v in eval_metrics_after.items()})
    pprint.pprint(eval_metrics_after)

    # For modern models, some have image processors and if these are not saved,
    # the lm_eval with vllm will complain.
    try:
        image_processor = AutoImageProcessor.from_pretrained(
            wandb_config["model_config"]["initial_model_name_or_path"],
            trust_remote_code=True,
        )
        image_processor.save_pretrained(output_dir)
    except Exception:
        pass

    # Push to HF Hub.
    logging.info(f"Finished final evaluation. Pushing to HuggingFace...")
    trainer.save_model(output_dir=sft_config.output_dir)
    trainer.push_to_hub()
    logging.info("Pushed to HuggingFace.")

    # For some reason, the trainer holds onto GPU memory even after finishing.
    # There might be a smarter way of freeing up the memory, but here's my workaround.
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()

    time.sleep(15)

    # Evaluate the new model using LM Eval Harness.
    for temperature in [0.0, 0.316, 1.0]:
        lm_eval_results_after = run_lm_eval_with_vllm(
            model_hf_path=sft_config.hub_model_id,
            lm_eval_task=lm_eval_task,
            num_fewshot=0,
            seed=wandb_config["seed"],
            temperature=temperature,
        )
        wandb.log(
            {
                f"lm_eval_after_temp={temperature}/{k}": v
                for k, v in lm_eval_results_after.items()
            },
            commit=True,
        )

    # Just to be safe.
    gc.collect()
    torch.cuda.empty_cache()

    wandb.finish()


def create_sfted_model_huggingface_name(wandb_config: Dict[str, Any]) -> str:
    init_model_name = wandb_config["model_config"]["initial_model_name_or_path"].split(
        "/"
    )[-1]
    dataset_name = wandb_config["data_config"]["dataset"].split("/")[-1]
    num_train_epochs = wandb_config["sft_trainer_config"]["num_train_epochs"]
    seed = wandb_config["seed"]
    sfted_model_hf_name = f"mem_model_{init_model_name}_dataset_{dataset_name}_epochs_{num_train_epochs}_seed_{seed}"
    if len(sfted_model_hf_name) > 94:
        raise ValueError(
            f"reward_model_huggingface_name is too long: {sfted_model_hf_name}"
        )
    return sfted_model_hf_name


def run_lm_eval_with_vllm(
    model_hf_path: str,
    lm_eval_task: str,
    num_fewshot: int = 0,
    seed: int = 0,
    temperature: float = 1.0,
):
    do_sample = True if temperature > 0.0 else False

    command = f"""mem_scoring_vs_sampling_env/bin/lm_eval \
    --model vllm \
    --model_args pretrained={model_hf_path},dtype=auto,max_model_len=2048,max_num_seqs=2048 \
    --batch_size auto \
    --tasks {lm_eval_task} \
    --num_fewshot {num_fewshot} \
    --log_samples \
    --output_path ./lm-eval-output/ \
    --gen_kwargs temperature={temperature},do_sample={do_sample} \
    --seed {seed}
    """

    logging.info(f"command: {command}")

    try:
        env = os.environ.copy()

        process = subprocess.run(
            command,
            shell=True,
            check=True,  # This is what raises the CalledProcessError
            text=True,
            capture_output=True,
            env=env,
        )
        scores = extract_exact_match_scores_from_output(process.stdout)
        logging.info(scores)

    except subprocess.CalledProcessError as e:
        # Handle the error
        logging.error(f"Command failed with exit code {e.returncode}")
        logging.error(f"STDOUT: {e.stdout}")
        logging.error(f"STDERR: {e.stderr}")
        raise e

    return scores


def extract_exact_match_scores_from_output(output_text: str) -> Dict[str, float]:
    """Extract exact_match scores from the lm-eval output text."""
    results = {}

    lines = output_text.strip().split("\n")
    for line in lines:
        if "exact_match" in line:
            # Parse the line in the table that contains exact_match
            parts = line.split("|")
            if len(parts) >= 8:
                filter_type = parts[3].strip()
                metric = parts[5].strip()
                value = float(parts[7].strip().split("±")[0])

                # Create a meaningful key
                key = f"{metric}_{filter_type}".replace("-", "_")
                results[key] = value

    return results


if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(i) for i in range(torch.cuda.device_count())]
        )
    train_supervised_finetuning()
    logging.info("Finished train_supervised_finetuning.py!")
