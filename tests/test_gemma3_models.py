"""Tests for the from-scratch Gemma 3 dense model construction in src.models.

ICLR 2027 Phase 5 (decision D2 in docs/ICLR_2027_CHECKLIST.md). Verifies:
  1. The size table carries Google's own small-Gemma-3 architecture constants
     (verified 2026-08-17 against the shipped gemma-3-270m / gemma-3-1b-pt configs).
  2. `create_causalm_for_pretraining` builds a Gemma3 text model with the right
     architecture, tied embeddings, and a total parameter count within 10% of the
     advertised size name.
  3. The Qwen3 path is untouched and unknown names still fail loudly.

Config correctness only -- no training, no GPU. All tests run with CUDA unused.
"""

import pytest

import src.models
from src.models import (
    GEMMA3_HEAD_DIM,
    GEMMA3_NUM_ATTENTION_HEADS,
    GEMMA3_NUM_KEY_VALUE_HEADS,
    GEMMA3_QUERY_PRE_ATTN_SCALAR,
    GEMMA3_SLIDING_WINDOW,
    GEMMA3_VOCAB_SIZE,
    create_causalm_for_pretraining,
    gemma3_parameters_to_depths_widths_and_intermediates,
)


def _advertised_size(name: str) -> float:
    if name.endswith("B"):
        return float(name[:-1]) * 1e9
    assert name.endswith("M")
    return float(name[:-1]) * 1e6


def _build(size_name: str):
    return create_causalm_for_pretraining(
        {"model_name": f"Gemma3/Gemma3-{size_name}", "torch_dtype": "bfloat16"}
    )


# ---------------------------------------------------------------------------
# 1. Size table sanity
# ---------------------------------------------------------------------------


def test_size_table_is_nonempty_and_monotone():
    """Depth, width, and MLP width must all grow together along the ladder."""
    entries = list(gemma3_parameters_to_depths_widths_and_intermediates.values())
    assert len(entries) >= 2
    for (d1, w1, i1), (d2, w2, i2) in zip(entries, entries[1:]):
        assert d1 < d2 and w1 < w2 and i1 < i2


def test_widths_stay_on_gemma3_grid():
    """Google's ladder uses widths on multiples of 64 and head_dim 256."""
    for (
        depth,
        width,
        inter,
    ) in gemma3_parameters_to_depths_widths_and_intermediates.values():
        assert width % 64 == 0


def test_268m_entry_is_googles_shipped_architecture():
    """The anchor entry must reproduce google/gemma-3-270m's text config exactly.

    Values verified against the Hub 2026-08-17
    (scripts/scratch/probe_gemma3_shipped_configs.py).
    """
    assert gemma3_parameters_to_depths_widths_and_intermediates["268M"] == (
        18,
        640,
        2048,
    )
    assert GEMMA3_VOCAB_SIZE == 262144
    assert GEMMA3_NUM_ATTENTION_HEADS == 4
    assert GEMMA3_NUM_KEY_VALUE_HEADS == 1
    assert GEMMA3_HEAD_DIM == 256
    assert GEMMA3_SLIDING_WINDOW == 512
    assert GEMMA3_QUERY_PRE_ATTN_SCALAR == 256


# ---------------------------------------------------------------------------
# 2. Model construction (smallest size, kept fast)
# ---------------------------------------------------------------------------


def test_build_smallest_gemma3_architecture():
    model = _build("107M")
    config = model.config
    assert config.model_type == "gemma3_text"
    assert config.num_hidden_layers == 13
    assert config.hidden_size == 320
    assert config.intermediate_size == 1024
    assert config.vocab_size == GEMMA3_VOCAB_SIZE
    assert config.num_attention_heads == GEMMA3_NUM_ATTENTION_HEADS
    assert config.num_key_value_heads == GEMMA3_NUM_KEY_VALUE_HEADS
    assert config.head_dim == GEMMA3_HEAD_DIM
    assert config.sliding_window == GEMMA3_SLIDING_WINDOW
    assert config.query_pre_attn_scalar == GEMMA3_QUERY_PRE_ATTN_SCALAR


def test_smallest_gemma3_embeddings_are_tied():
    """Gemma 3 ties input and output embeddings at every shipped size.

    Untied embeddings would add another 262,144 x hidden parameters and wreck
    both the size names and the non-embedding accounting in the size table.
    """
    model = _build("107M")
    assert model.config.tie_word_embeddings is True
    assert model.get_output_embeddings().weight is model.get_input_embeddings().weight


def test_smallest_gemma3_local_global_attention_pattern():
    """The 5-local:1-global sliding-attention pattern must survive construction."""
    model = _build("107M")
    layer_types = model.config.layer_types
    assert len(layer_types) == 13
    assert "sliding_attention" in layer_types
    assert "full_attention" in layer_types
    # Every 6th layer is global, exactly as in the shipped checkpoints.
    for idx, layer_type in enumerate(layer_types):
        expected = "full_attention" if (idx + 1) % 6 == 0 else "sliding_attention"
        assert layer_type == expected


def test_smallest_gemma3_parameter_count_matches_name():
    model = _build("107M")
    total = sum(p.numel() for p in model.parameters())
    advertised = _advertised_size("107M")
    assert abs(total - advertised) / advertised <= 0.10
    # Exact count measured 2026-08-17; drift means the architecture changed.
    assert total == 107_338_816


# ---------------------------------------------------------------------------
# 3. Full ladder parameter counts (builds every size; marked slow)
# ---------------------------------------------------------------------------


# Exact totals measured on CPU 2026-08-17 with
# scripts/scratch/smoke_test_gemma3_configs.py. If any of these change, the
# size-table comment in src/models.py must be re-measured, not hand-edited.
GEMMA3_EXPECTED_TOTALS = {
    "107M": 107_338_816,
    "163M": 163_064_000,
    "268M": 268_098_176,
    "497M": 497_378_176,
}


@pytest.mark.slow
@pytest.mark.parametrize(
    "size_name", list(gemma3_parameters_to_depths_widths_and_intermediates)
)
def test_full_ladder_parameter_counts(size_name):
    assert size_name in GEMMA3_EXPECTED_TOTALS, "size table and test out of sync"
    model = _build(size_name)
    total = sum(p.numel() for p in model.parameters())
    assert total == GEMMA3_EXPECTED_TOTALS[size_name]
    advertised = _advertised_size(size_name)
    assert abs(total - advertised) / advertised <= 0.10


# ---------------------------------------------------------------------------
# 4. Dispatch: Qwen3 unaffected, bad names fail loudly
# ---------------------------------------------------------------------------


def test_qwen3_path_still_works():
    model = create_causalm_for_pretraining(
        {"model_name": "Qwen3/Qwen3-2M", "torch_dtype": "bfloat16"}
    )
    assert model.config.model_type == "qwen3"
    assert model.config.hidden_size == 6
    assert model.config.num_hidden_layers == 1


def test_unknown_family_raises_value_error():
    with pytest.raises(ValueError):
        create_causalm_for_pretraining(
            {"model_name": "Llama/Llama-107M", "torch_dtype": "bfloat16"}
        )


def test_unknown_gemma3_size_raises_key_error():
    with pytest.raises(KeyError):
        create_causalm_for_pretraining(
            {"model_name": "Gemma3/Gemma3-34M", "torch_dtype": "bfloat16"}
        )


def test_unknown_dtype_raises():
    with pytest.raises(NotImplementedError):
        create_causalm_for_pretraining(
            {"model_name": "Gemma3/Gemma3-107M", "torch_dtype": "float64"}
        )
