"""Probe the installed transformers' Gemma3TextConfig defaults (CPU-only).

Also prints the shipped Gemma 3 270M / 1B architecture hyperparameters as a
reference for choosing from-scratch sizes, if they can be read from the Hub
config cache without network. Run:

    CUDA_VISIBLE_DEVICES="" python scripts/scratch/probe_gemma3_config_defaults.py
"""

import transformers

print("transformers", transformers.__version__)
from transformers import Gemma3TextConfig

c = Gemma3TextConfig()
keys = [
    "vocab_size",
    "hidden_size",
    "num_hidden_layers",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "sliding_window",
    "sliding_window_pattern",
    "query_pre_attn_scalar",
    "rope_theta",
    "rope_local_base_freq",
    "tie_word_embeddings",
    "max_position_embeddings",
]
for k in keys:
    if hasattr(c, k):
        print(f"  {k} = {getattr(c, k)}")
