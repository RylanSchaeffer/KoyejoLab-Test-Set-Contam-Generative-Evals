import os

from ray.train.huggingface.transformers import prepare_trainer

# Rok asked us to include the following specifications in our code to prevent CPUs from spinning idly:
n_threads_str = "32"
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

import logging
import gc
import math
import numpy as np
import pprint
import time
import torch

# Compiling seems to be causing problems down the line :/
torch.compiler.disable()
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
from typing import Any, Dict
import wandb

import src.data
import src.globals
import src.models


logging.basicConfig(level=logging.INFO)


def pretrain():
    num_visible_devices = torch.cuda.device_count()
    assert num_visible_devices > 0, "No CUDA devices available."
    run = wandb.init(
        project="memorization-scoring-vs-sampling-pt",
        config=src.globals.DEFAULT_PRETRAINING_CONFIG,
        entity=wandb.api.default_entity,
    )

    # Convert to a dictionary; otherwise, can't distribute because W&B
    # config is not pickle-able.
    wandb_config: Dict[str, Any] = dict(wandb.config)
    print("CUDA VISIBLE DEVICES: ", os.environ["CUDA_VISIBLE_DEVICES"])
    pprint.pprint(wandb_config)

    # Create output directory.
    pted_model_hf_name = create_pretrained_model_huggingface_name(
        wandb_config=wandb_config,
    )
    output_dir = os.path.join("models", "pt_language_model", pted_model_hf_name)
    wandb.config.update({"output_dir": output_dir})
    print("Output Directory: ", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    set_seed(seed=wandb_config["seed"], deterministic=True)

    if wandb_config["model_config"]["model_name"].startswith("Qwen3/Qwen3-"):
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2-1.5B",  # Arbitrary. Doesn't matter so long as it is Qwen3.
            use_fast=True,
            trust_remote_code=True,
        )
    else:
        raise NotImplementedError

    model: AutoModelForCausalLM = src.models.create_causalm_for_pretraining(
        model_config_dict=wandb_config["model_config"],
    )
    wandb_config = compute_derived_hyperparameters(
        model=model,
        wandb_config=wandb_config,
    )

    # Lightly modify configs as necessary.
    if wandb_config["model_config"]["torch_dtype"] == "bfloat16":
        wandb_config["trainer_config"]["bf16"] = True
    else:
        wandb_config["trainer_config"]["bf16"] = False
    if wandb_config["model_config"]["torch_dtype"] == "float16":
        wandb_config["trainer_config"]["fp16"] = True
    else:
        wandb_config["trainer_config"]["fp16"] = False

    pretraining_config = TrainingArguments(
        bf16=wandb_config["trainer_config"]["bf16"],
        data_seed=wandb_config["trainer_config"]["data_seed"],
        dataloader_drop_last=wandb_config["trainer_config"]["dataloader_drop_last"],
        dataloader_num_workers=wandb_config["trainer_config"]["dataloader_num_workers"],
        dataloader_prefetch_factor=wandb_config["trainer_config"][
            "dataloader_prefetch_factor"
        ],
        eval_on_start=wandb_config["trainer_config"]["eval_on_start"],
        eval_strategy=wandb_config["trainer_config"]["eval_strategy"],
        eval_steps=wandb_config["trainer_config"]["eval_steps"],
        fp16=wandb_config["trainer_config"]["fp16"],
        gradient_accumulation_steps=wandb_config["trainer_config"][
            "gradient_accumulation_steps"
        ],
        gradient_checkpointing=wandb_config["trainer_config"]["gradient_checkpointing"],
        hub_model_id=f"RylanSchaeffer/{pted_model_hf_name}",
        hub_private_repo=True,
        hub_strategy=wandb_config["trainer_config"]["hub_strategy"],
        include_num_input_tokens_seen=True,
        learning_rate=float(wandb_config["trainer_config"]["learning_rate"]),
        logging_steps=wandb_config["trainer_config"]["logging_steps"],
        lr_scheduler_type=wandb_config["trainer_config"]["lr_scheduler_type"],
        max_steps=-1,
        metric_for_best_model="eval_loss",
        num_train_epochs=wandb_config["trainer_config"]["num_train_epochs"],
        optim=wandb_config["trainer_config"]["optim"],
        output_dir=output_dir,
        per_device_eval_batch_size=wandb_config["trainer_config"][
            "per_device_eval_batch_size"
        ],
        per_device_train_batch_size=wandb_config["trainer_config"][
            "per_device_train_batch_size"
        ],
        remove_unused_columns=wandb_config["trainer_config"]["remove_unused_columns"],
        run_name=wandb.run.id,
        report_to=wandb_config["trainer_config"]["report_to"],
        save_strategy=wandb_config["trainer_config"]["save_strategy"],
        save_total_limit=wandb_config["trainer_config"]["save_total_limit"],
        seed=wandb_config["seed"],
        warmup_steps=wandb_config["trainer_config"]["warmup_steps"],
        # warmup_ratio=wandb_config["trainer_config"]["warmup_ratio"],
    )

    # trl/trainer/sft_trainer.py:408: UserWarning: You passed a tokenizer with padding_side not equal
    # to right to the SFTTrainer. This might lead to some unexpected behaviour due to overflow issues
    # when training a model in half-precision. You might consider adding tokenizer.padding_side = 'right' to your code.
    tokenizer.padding_side = "right"

    # TODO(Rylan): Correct implement the data tomorrow.
    datasets_dict = src.data.create_dataset_for_pretraining(
        data_config=wandb_config["data_config"],
        trainer_config=wandb_config["trainer_config"],
        tokenizer=tokenizer,
        seed=wandb_config["seed"],
    )
    train_dataset = datasets_dict["train"]
    eval_dataset = datasets_dict["eval"]

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=pretraining_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        # compute_metrics=compute_token_accuracy,  # Compute token accuracy when evaluating.
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

    # Push to HF Hub.
    logging.info(f"Finished final evaluation. Pushing to HuggingFace...")
    tokenizer.padding_side = "left"  # Otherwise, generate gets screwed up.
    trainer.save_model(output_dir=pretraining_config.output_dir)
    trainer.push_to_hub()
    logging.info("Pushed to HuggingFace.")

    # For some reason, the trainer holds onto GPU memory even after finishing.
    # There might be a smarter way of freeing up the memory, but here's my workaround.
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(15)
    # Just to be safe.
    gc.collect()
    torch.cuda.empty_cache()
    wandb.finish()


def compute_derived_hyperparameters(
    model: AutoModelForCausalLM, wandb_config: Dict[str, Any]
) -> Dict[str, Any]:
    # 1. Calculate the number of model parameters.
    num_parameters = sum(p.numel() for p in model.parameters())

    # 2. Compute the target number of training tokens.
    target_num_training_tokens_total = int(
        20 * wandb_config["trainer_config"]["overtrain_multiplier"] * num_parameters
    )

    # 3. Compute a reasonable batch size, according to https://arxiv.org/abs/2412.01505.
    num_tokens_per_optimizer_step = int(
        3.24 * np.power(10, 3) * np.power(target_num_training_tokens_total, 0.264)
    )

    # 4. Compute the number of sequences.
    num_visible_devices = torch.cuda.device_count()
    num_tokens_per_forward_pass = (
        num_visible_devices
        * wandb_config["trainer_config"]["per_device_train_batch_size"]
        * wandb_config["trainer_config"]["max_length"]
    )
    gradient_accumulation_steps = math.ceil(
        num_tokens_per_optimizer_step / num_tokens_per_forward_pass
    )

    # 4. Compute the number of training tokens per epoch.
    num_training_tokens_per_epoch = (
        target_num_training_tokens_total
        / wandb_config["trainer_config"]["num_train_epochs"]
    )

    # 5. Calculate the learning rate. It should grow with square-root of batch size.
    learning_rate = wandb_config["trainer_config"]["base_learning_rate"] * np.sqrt(
        num_tokens_per_optimizer_step
    )

    # wandb.config.update(
    #     {
    #         "target_num_training_tokens_total": target_num_training_tokens_total,
    #         "num_training_tokens_per_epoch": num_training_tokens_per_epoch,
    #         "effective_global_batch_size": effective_global_batch_size,
    #     },
    # )

    additional_trainer_config_data = {
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "num_visible_devices": num_visible_devices,
        "num_tokens_per_forward_pass": num_tokens_per_forward_pass,
        "num_tokens_per_optimizer_step": num_tokens_per_optimizer_step,
        "num_training_tokens_per_epoch": num_training_tokens_per_epoch,
        "target_num_training_tokens_total": target_num_training_tokens_total,
    }

    # Write to W&B.
    wandb.config.trainer_config.update(additional_trainer_config_data)

    # Add to our W&B config that controls everything.
    wandb_config["trainer_config"].update(additional_trainer_config_data)

    return wandb_config


def compute_token_accuracy(eval_pred):
    """
    Computes token-level accuracy for a language model.
    """
    # Unpack the outputs from the EvalPrediction object
    logits, labels = eval_pred.predictions, eval_pred.label_ids

    # The SFTTrainer shifts logits and labels internally.
    # We need to do the same here to align predictions with labels.
    # Logits are for predicting the *next* token.
    # -> Logits at position i predict token at position i+1
    # -> We compare logits[..., :-1, :] with labels[..., 1:]
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]

    # Get the predicted token IDs by taking the argmax of the logits
    predictions = np.argmax(shift_logits, axis=-1)

    # Create a mask to ignore padding tokens (where label is -100)
    mask = shift_labels != -100

    # Calculate the number of correct predictions
    correct_tokens = np.sum((predictions == shift_labels) & mask).astype(float)

    # Calculate the total number of non-padded tokens
    total_tokens = np.sum(mask).astype(float)

    # Compute the accuracy
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else np.nan

    # Return the metric in a dictionary
    return {"mean_token_accuracy": accuracy}


def create_pretrained_model_huggingface_name(wandb_config: Dict[str, Any]) -> str:
    init_model_name = wandb_config["model_config"]["model_name"].split("/")[-1]
    dataset_name = wandb_config["data_config"]["benchmark"].split("/")[-1]
    num_train_epochs = wandb_config["trainer_config"]["num_train_epochs"]
    overtrain_multiplier = wandb_config["trainer_config"]["overtrain_multiplier"]
    # seed = wandb_config["seed"]
    pted_model_hf_name = f"mem_model_{init_model_name}_dataset_{dataset_name}_epochs_{num_train_epochs}_ot_{overtrain_multiplier}_pt"
    if len(pted_model_hf_name) > 94:
        raise ValueError(f"pted_model_hf_name is too long: {pted_model_hf_name}")
    return pted_model_hf_name


if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(i) for i in range(torch.cuda.device_count())]
        )
    pretrain()
    logging.info("Finished pretrain_language_model.py!")
