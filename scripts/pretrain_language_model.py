import os

from ray.train.huggingface.transformers import prepare_trainer

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

import logging
import gc
import pprint
import subprocess
import time
import torch

# Compiling seems to be causing problems down the line :/
torch.compiler.disable()
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    set_seed,
    Trainer,
    TrainingArguments,
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

    from datasets import (
        concatenate_datasets,
        load_dataset,
        interleave_datasets,
        DatasetDict,
    )

    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", split="train", num_proc=16
    )
    # Create output directory.
    pted_model_hf_name = create_pretrained_model_huggingface_name(
        wandb_config=wandb_config,
    )
    output_dir = os.path.join("models", "sft_language_model", pted_model_hf_name)
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

    trainer_config_dict: Dict[str, Any] = wandb_config["trainer_config"]
    model_config_dict: Dict[str, Any] = wandb_config["model_config"]
    data_config_dict: Dict[str, Any] = wandb_config["data_config"]

    # Lightly modify configs as necessary.
    if model_config_dict["torch_dtype"] == "bfloat16":
        trainer_config_dict["bf16"] = True
    else:
        trainer_config_dict["bf16"] = False
    if model_config_dict["torch_dtype"] == "float16":
        trainer_config_dict["fp16"] = True
    else:
        trainer_config_dict["fp16"] = False

    pretraining_config = TrainingArguments(
        bf16=trainer_config_dict["bf16"],
        data_seed=trainer_config_dict["data_seed"],
        dataloader_drop_last=trainer_config_dict["dataloader_drop_last"],
        dataloader_num_workers=trainer_config_dict["dataloader_num_workers"],
        dataloader_prefetch_factor=trainer_config_dict["dataloader_prefetch_factor"],
        dataset_text_field="input_ids",
        eval_on_start=trainer_config_dict["eval_on_start"],
        eval_strategy=trainer_config_dict["eval_strategy"],
        eval_steps=trainer_config_dict["eval_steps"],
        fp16=trainer_config_dict["fp16"],
        gradient_accumulation_steps=trainer_config_dict["gradient_accumulation_steps"],
        gradient_checkpointing=trainer_config_dict["gradient_checkpointing"],
        hub_model_id=f"RylanSchaeffer/{pted_model_hf_name}",
        hub_private_repo=True,
        hub_strategy=trainer_config_dict["hub_strategy"],
        include_num_input_tokens_seen=True,
        learning_rate=float(trainer_config_dict["learning_rate"]),
        logging_steps=trainer_config_dict["logging_steps"],
        lr_scheduler_type=trainer_config_dict["lr_scheduler_type"],
        max_length=trainer_config_dict["max_length"],
        max_steps=trainer_config_dict["max_steps"],
        metric_for_best_model="eval_loss",
        num_train_epochs=trainer_config_dict["num_train_epochs"],
        optim=trainer_config_dict["optim"],
        output_dir=output_dir,
        per_device_eval_batch_size=trainer_config_dict["per_device_eval_batch_size"],
        per_device_train_batch_size=trainer_config_dict["per_device_train_batch_size"],
        remove_unused_columns=trainer_config_dict["remove_unused_columns"],
        run_name=wandb.run.id,
        report_to=trainer_config_dict["report_to"],
        save_strategy=trainer_config_dict["save_strategy"],
        save_total_limit=trainer_config_dict["save_total_limit"],
        seed=wandb_config["seed"],
        warmup_ratio=trainer_config_dict["warmup_ratio"],
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

    datasets_dict = src.data.create_dataset_for_pretraining(
        tokenizer=tokenizer,
        dataset_name=data_config_dict["dataset"],
        max_length=pretraining_config.max_length,
    )
    train_dataset = datasets_dict["train"]
    eval_dataset = datasets_dict["eval"]

    model = src.models.load_automodelforcausallm(
        model_config_dict=model_config_dict,
    )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=pretraining_config,
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


def create_pretrained_model_huggingface_name(wandb_config: Dict[str, Any]) -> str:
    init_model_name = wandb_config["model_config"]["initial_model_name_or_path"].split(
        "/"
    )[-1]
    dataset_name = wandb_config["data_config"]["dataset"].split("/")[-1]
    num_train_epochs = wandb_config["trainer_config"]["num_train_epochs"]
    seed = wandb_config["seed"]
    pted_model_hf_name = f"mem_model_{init_model_name}_dataset_{dataset_name}_epochs_{num_train_epochs}_seed_{seed}_pt"
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
