import datasets
import numpy as np
import os
import pprint
import torch
import torch.utils.data
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    pipeline,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
from trl import (
    ModelConfig,
    get_kbit_device_map,
    get_quantization_config,
)
from typing import Any, Dict, List, Optional, Tuple, Union


def load_automodelforcausallm(
    model_config_dict: Dict[str, Any]
) -> AutoModelForCausalLM:
    if model_config_dict["torch_dtype"] == "bfloat16":
        torch_dtype = torch.bfloat16
    elif model_config_dict["torch_dtype"] == "float16":
        torch_dtype = torch.float16
    elif model_config_dict["torch_dtype"] == "float32":
        torch_dtype = torch.float32
    else:
        raise NotImplementedError

    # Build the keyword arguments dict with ONLY valid keys.
    # Do NOT merge the whole model_config_dict.
    model_kwargs = {
        # Get attn_implementation from your config, defaulting to "eager".
        "attn_implementation": model_config_dict.get("attn_implementation", "eager"),
        "device_map": "auto",
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }

    # Your assertions are good. They confirm the final kwargs are correct.
    if "gemma" in model_config_dict["initial_model_name_or_path"]:
        assert model_kwargs["torch_dtype"] == torch.bfloat16
        assert model_kwargs["attn_implementation"] == "eager"

    model = AutoModelForCausalLM.from_pretrained(
        model_config_dict["initial_model_name_or_path"],
        **model_kwargs,
    )

    return model
