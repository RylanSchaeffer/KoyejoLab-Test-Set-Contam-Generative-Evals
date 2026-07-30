"""Confirm the contaminant suffix prevents overwriting published checkpoints."""
import importlib.util, sys
spec = importlib.util.spec_from_file_location("pt", "scripts/pretrain_language_model.py")
# Import only the naming function without executing the training entrypoint.
import numpy as np
src = open("scripts/pretrain_language_model.py").read()
start = src.index("def create_pretrained_model_huggingface_name")
end = src.index("def get_hf_username")
from typing import Any, Dict
namespace = {"np": np, "Dict": Dict, "Any": Any}
exec(src[start:end], namespace)
fn = namespace["create_pretrained_model_huggingface_name"]

base = {
    "model_config": {"model_name": "Qwen3/Qwen3-34M"},
    "trainer_config": {"num_train_epochs": 1, "overtrain_multiplier": 1.0},
    "data_config": {
        "benchmark": "EleutherAI/minerva_math",
        "num_benchmark_replicas_per_epoch": 316,
        "benchmark_subset_fraction": 1.0,
    },
}
exact = fn(base)
para = dict(base)
para["data_config"] = dict(base["data_config"], contaminant="RylanSchaeffer/math_rephrased")
paraphrased = fn(para)
same = dict(base)
same["data_config"] = dict(base["data_config"], contaminant="EleutherAI/minerva_math")
redundant = fn(same)

print(f"exact-replica     : {exact}")
print(f"paraphrased       : {paraphrased}")
print(f"contaminant==bench: {redundant}")
print(f"\ncollision avoided : {exact != paraphrased}")
print(f"backward compatible: {exact == redundant}")
print(f"length ok         : {len(paraphrased)} <= 94")
