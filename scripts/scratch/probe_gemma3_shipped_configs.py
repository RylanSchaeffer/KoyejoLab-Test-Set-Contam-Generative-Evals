"""Fetch the shipped Gemma 3 270M / 1B architecture configs from the Hub (read-only).

Used to anchor the from-scratch Gemma 3 size table in src/models.py to Google's own
depth/width scaling pattern rather than guessing it. Gemma repos are gated behind a
license click-through, so this may fail under the ambient HF identity; if it does,
the size table falls back to the values published in the Gemma 3 technical report.

    CUDA_VISIBLE_DEVICES="" python scripts/scratch/probe_gemma3_shipped_configs.py
"""

from transformers import AutoConfig

KEYS = [
    "vocab_size",
    "hidden_size",
    "num_hidden_layers",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "sliding_window",
    "query_pre_attn_scalar",
    "rope_theta",
    "rope_local_base_freq",
    "tie_word_embeddings",
]

for repo in ["google/gemma-3-270m", "google/gemma-3-1b-pt"]:
    print(f"=== {repo}")
    try:
        config = AutoConfig.from_pretrained(repo)
        # Multimodal checkpoints nest the text config; 270m/1b are text-only.
        if hasattr(config, "text_config"):
            config = config.text_config
        for k in KEYS:
            if hasattr(config, k):
                print(f"  {k} = {getattr(config, k)}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
