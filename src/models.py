"""Model creation and loading utilities for Qwen3 and Gemma 3 language models.

This module provides functions for creating Qwen3 or Gemma 3 dense models from
scratch with specific parameter counts, or loading pretrained models from
HuggingFace Hub. The architecture configurations follow each family's own
scaling patterns.

Supported model sizes: Qwen3 2M to 1.44B; Gemma 3 dense 107M to 497M.
"""

import math
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel


# Mapping from parameter count strings to (num_layers, hidden_size) tuples.
# These configurations follow Qwen3's architecture scaling patterns.
# The formula seems to follow width = 42 * (depth - 1) + 6.
# Intermediate size is computed as: 256 * floor((255 + floor(8 * hidden_size / 3)) / 256)
qwen3_parameters_to_depths_and_widths: Dict[str, tuple[int, int]] = {
    "2M": (1, 6),
    "16M": (2, 48),
    "34M": (3, 96),  # Should be 90, I think.
    "48M": (4, 128),  # Should be 132
    "63M": (5, 160),  # Should be 174. Initially called 62M, renamed to 63M.
    "93M": (6, 224),  # Should be 216.
    "111M": (7, 256),
    "138M": (8, 300),
    "165M": (9, 344),
    "191M": (10, 384),
    "223M": (11, 424),
    "262M": (12, 480),
    "344M": (14, 576),
    "499M": (18, 704),
    "660M": (21, 832),
    "806M": (23, 940),
    "934M": (25, 1010),
    "1.09B": (27, 1100),
    "1.26B": (29, 1180),
    "1.44B": (31, 1260),
}


# Mapping from parameter count strings to (num_layers, hidden_size, intermediate_size)
# for from-scratch Gemma 3 **dense** models (Phase 5, decision D2 in
# docs/EXPERIMENT_CHECKLIST.md).
#
# The architecture follows Google's own small-Gemma-3 scaling pattern, verified
# 2026-08-17 directly against the Hub configs of google/gemma-3-270m and
# google/gemma-3-1b-pt (scripts/scratch/probe_gemma3_shipped_configs.py):
#   - constant across sizes: 4 attention heads, 1 KV head, head_dim 256,
#     query_pre_attn_scalar 256, sliding_window 512 (5 local : 1 global layer
#     pattern), rope_theta 1e6 (global) / 1e4 (local), vocab 262,144,
#     tied embeddings.
#   - depth/width scale together: (18 layers, 640 hidden) at 270M ->
#     (26, 1152) at 1B, i.e. ~1 layer per 64 hidden. New widths stay on
#     multiples of 64 and depths on that line.
#   - MLP ratio: 3.2x at 270M rising to 6.0x at 1B. Below 640 hidden we hold
#     3.2x (extrapolating the ratio downward would pinch the MLP to ~1x);
#     at 896 hidden we use 4.0x, between the two anchors.
#
# The "268M" entry reproduces google/gemma-3-270m's text architecture exactly.
# Names are total parameter counts, like the Qwen3 table above.
#
# ⚠️ Accounting (checklist 5.1): Gemma 3's 262,144-token tied vocabulary makes
# small models embedding-dominated, so total-parameter names are NOT comparable
# across families -- match on **non-embedding** parameters. Actual counts,
# measured on CPU 2026-08-17 with scripts/scratch/smoke_test_gemma3_configs.py
# (bfloat16; Gemma's tokenizer has a distinct pad token, so unlike Qwen3 no
# extra pad embedding row is added at pretraining time):
#
#   name    (L, h, inter)      total          non-embedding   Qwen3 neighbours (non-emb)
#   107M    (13, 320, 1024)    107,338,816     23,452,736     93M (25.1M) / 111M (33.5M)
#   163M    (15, 448, 1408)    163,064,000     45,623,488     111M (33.5M) / 165M (60.2M)
#   268M    (18, 640, 2048)    268,098,176    100,326,016     262M (116.5M)
#   497M    (22, 896, 3584)    497,378,176    262,497,152     499M (285.5M)
#
# Note the totals happen to align with the Qwen3 ladder names too (107M~111M,
# 163M~165M, 268M~262M, 497M~499M), so the arm overlaps the Qwen3 ladder under
# either accounting; the top pair (497M vs 499M) matches well under both. A
# Gemma 3 analogue of Qwen3-34M is impossible: the tied embedding matrix alone
# is 262,144 x h parameters (83.9M at the narrowest width used here).
gemma3_parameters_to_depths_widths_and_intermediates: Dict[
    str, tuple[int, int, int]
] = {
    "107M": (13, 320, 1024),
    "163M": (15, 448, 1408),
    "268M": (18, 640, 2048),  # google/gemma-3-270m's exact text architecture.
    "497M": (22, 896, 3584),
}

# Constants shared by every from-scratch Gemma 3 dense size; verified against the
# shipped gemma-3-270m and gemma-3-1b-pt configs (see comment above).
GEMMA3_VOCAB_SIZE = 262144
GEMMA3_NUM_ATTENTION_HEADS = 4
GEMMA3_NUM_KEY_VALUE_HEADS = 1
GEMMA3_HEAD_DIM = 256
GEMMA3_SLIDING_WINDOW = 512
GEMMA3_QUERY_PRE_ATTN_SCALAR = 256


def create_causalm_for_pretraining(
    model_config_dict: Dict[str, Any],
) -> PreTrainedModel:
    """Create a new Qwen3 or Gemma 3 causal language model from scratch.

    Initializes a randomly-weighted model with architecture determined by the
    family and parameter count specified in model_name. The depth (num_layers)
    and width (hidden_size) are looked up from the family's size table
    (qwen3_parameters_to_depths_and_widths or
    gemma3_parameters_to_depths_widths_and_intermediates).

    Args:
        model_config_dict: Configuration dictionary containing:
            - model_name: Model identifier in format "Qwen3/Qwen3-{size}" where
              size is one of: 2M, 16M, 34M, ..., 1.44B; or "Gemma3/Gemma3-{size}"
              where size is one of: 107M, 163M, 268M, 497M
            - torch_dtype: Data type string ("bfloat16", "float16", or "float32")
            - attn_implementation: Optional attention implementation (default: "eager")

    Returns:
        Randomly initialized Qwen3ForCausalLM or Gemma3ForCausalLM model.

    Raises:
        NotImplementedError: If torch_dtype is not recognized.
        ValueError: If model_name matches neither family's naming pattern.
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

    elif model_config_dict["model_name"].startswith("Gemma3/Gemma3-"):
        from transformers import Gemma3TextConfig

        num_parameters_str = model_config_dict["model_name"].split("-")[1]
        depth, width, intermediate_size = (
            gemma3_parameters_to_depths_widths_and_intermediates[num_parameters_str]
        )

        # Everything not passed here keeps Gemma3TextConfig defaults, which already
        # match the shipped small checkpoints (head_dim 256, query_pre_attn_scalar
        # 256, rope_theta 1e6 / rope_local_base_freq 1e4, tied embeddings, RMSNorm
        # placement, 5-local:1-global attention layer pattern). The values we do
        # pass are the ones whose defaults differ from google/gemma-3-270m:
        # vocab (262,208 default vs 262,144 shipped), heads (8 vs 4), KV heads
        # (4 vs 1), and sliding window (4,096 vs 512).
        model_config = Gemma3TextConfig(
            hidden_size=width,
            num_hidden_layers=depth,
            intermediate_size=intermediate_size,
            vocab_size=GEMMA3_VOCAB_SIZE,
            num_attention_heads=GEMMA3_NUM_ATTENTION_HEADS,
            num_key_value_heads=GEMMA3_NUM_KEY_VALUE_HEADS,
            head_dim=GEMMA3_HEAD_DIM,
            sliding_window=GEMMA3_SLIDING_WINDOW,
            query_pre_attn_scalar=GEMMA3_QUERY_PRE_ATTN_SCALAR,
            torch_dtype=torch_dtype,
        )

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
    model_config_dict: Dict[str, Any],
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
