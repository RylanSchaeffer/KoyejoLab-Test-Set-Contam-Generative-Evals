"""Model creation and loading utilities for Qwen3 language models.

This module provides functions for creating Qwen3 models from scratch with
specific parameter counts, or loading pretrained models from HuggingFace Hub.
The architecture configurations follow the Qwen3 scaling patterns.

Supported model sizes: 2M to 1.44B parameters.
"""

import math
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel


# Mapping from parameter count strings to (num_layers, hidden_size) tuples.
# These configurations follow Qwen3's architecture scaling patterns.
# Intermediate size is computed as: 256 * floor((255 + floor(8 * hidden_size / 3)) / 256)
qwen3_parameters_to_depths_and_widths: Dict[str, tuple[int, int]] = {
    "2M": (1, 6),
    "16M": (2, 48),
    "34M": (3, 96),
    "48M": (4, 128),
    "62M": (5, 160),
    "93M": (6, 224),
    "111M": (7, 256),
    "153M": (9, 320),
    "191M": (10, 384),
    "262M": (12, 480),
    "344M": (14, 576),
    "499M": (18, 704),
    "660M": (21, 832),
    "806M": (23, 940),
    "934M": (25, 1010),
    "1.08B": (27, 1100),
    "1.26B": (29, 1180),
    "1.44B": (31, 1260),
}


def create_causalm_for_pretraining(
    model_config_dict: Dict[str, Any]
) -> PreTrainedModel:
    """Create a new Qwen3 causal language model from scratch.

    Initializes a randomly-weighted Qwen3 model with architecture determined
    by the parameter count specified in model_name. The depth (num_layers)
    and width (hidden_size) are looked up from qwen3_parameters_to_depths_and_widths.

    Args:
        model_config_dict: Configuration dictionary containing:
            - model_name: Model identifier in format "Qwen3/Qwen3-{size}" where
              size is one of: 2M, 16M, 34M, ..., 1.44B
            - torch_dtype: Data type string ("bfloat16", "float16", or "float32")
            - attn_implementation: Optional attention implementation (default: "eager")

    Returns:
        Randomly initialized Qwen3ForCausalLM model.

    Raises:
        NotImplementedError: If torch_dtype is not recognized.
        ValueError: If model_name doesn't start with "Qwen3/Qwen3-".
        KeyError: If the parameter size is not in the supported configurations.

    Example:
        >>> config = {"model_name": "Qwen3/Qwen3-34M", "torch_dtype": "bfloat16"}
        >>> model = create_causalm_for_pretraining(config)
    """
    if model_config_dict["torch_dtype"] == "bfloat16":
        torch_dtype = torch.bfloat16
    elif model_config_dict["torch_dtype"] == "float16":
        torch_dtype = torch.float16
    elif model_config_dict["torch_dtype"] == "float32":
        torch_dtype = torch.float32
    else:
        raise NotImplementedError

    if model_config_dict["model_name"].startswith("Qwen3/Qwen3-"):
        from transformers import Qwen3Config, Qwen3ForCausalLM

        num_parameters_str: str = model_config_dict["model_name"].split("-")[1]
        depth, width = qwen3_parameters_to_depths_and_widths[num_parameters_str]
        intermediate_size = 256 * math.floor((255 + math.floor(8 * width / 3)) / 256)

        model_config = Qwen3Config(
            hidden_size=width,
            num_hidden_layers=depth,
            intermediate_size=intermediate_size,
            torch_dtype=torch_dtype,
        )
        # model_class = Qwen3ForCausalLM

    else:
        raise ValueError(model_config_dict["model_name"])

    # model: PreTrainedModel = model_class(
    #     config=model_config,
    # )

    model = AutoModelForCausalLM.from_config(
        model_config,
        # dtype=torch_dtype,
        attn_implementation=model_config_dict.get("attn_implementation", "eager"),
    )

    return model


def load_automodelforcausallm(
    model_config_dict: Dict[str, Any]
) -> AutoModelForCausalLM:
    """Load a pretrained causal language model from HuggingFace Hub.

    Loads a model with automatic device mapping for multi-GPU inference.
    Supports various model families including Qwen and Gemma.

    Args:
        model_config_dict: Configuration dictionary containing:
            - initial_model_name_or_path: HuggingFace model ID or local path
            - torch_dtype: Data type string ("bfloat16", "float16", or "float32")
            - attn_implementation: Optional attention implementation (default: "eager")

    Returns:
        Loaded AutoModelForCausalLM with weights from the pretrained checkpoint.

    Raises:
        NotImplementedError: If torch_dtype is not recognized.
        AssertionError: If loading a Gemma model without bfloat16 dtype.

    Note:
        Gemma models require bfloat16 dtype due to Google's model requirements.
    """
    if model_config_dict["torch_dtype"] == "bfloat16":
        torch_dtype = torch.bfloat16
    elif model_config_dict["torch_dtype"] == "float16":
        torch_dtype = torch.float16
    elif model_config_dict["torch_dtype"] == "float32":
        torch_dtype = torch.float32
    else:
        raise NotImplementedError

    model_kwargs = {
        # Get attn_implementation from your config, defaulting to "eager".
        "attn_implementation": model_config_dict.get("attn_implementation", "eager"),
        "device_map": "auto",
        "dtype": torch_dtype,
        "trust_remote_code": True,
    }

    if "gemma" in model_config_dict["initial_model_name_or_path"]:
        # Google models must use bf16.
        assert model_kwargs["torch_dtype"] == torch.bfloat16
        # assert model_kwargs["attn_implementation"] == "eager"

    model = AutoModelForCausalLM.from_pretrained(
        model_config_dict["initial_model_name_or_path"],
        **model_kwargs,
    )

    return model
