"""CPU smoke test for the from-scratch Gemma 3 dense configs in src.models.

Instantiates every size in `gemma3_parameters_to_depths_widths_and_intermediates`
with CUDA disabled, verifies the total parameter count lands within 10% of the
advertised size name, and prints total and non-embedding counts. Also prints the
non-embedding counts of the Qwen3 ladder sizes the Gemma arm is meant to overlap,
since checklist item 5.1 requires matching on non-embedding parameters (Gemma's
262k tied vocabulary makes small models embedding-dominated).

    CUDA_VISIBLE_DEVICES="" python scripts/scratch/smoke_test_gemma3_configs.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import src.models


def count_params(model) -> tuple[int, int]:
    """Return (total, non_embedding) parameter counts.

    With tied embeddings (Gemma 3) the shared weight is counted once by
    model.parameters(). Non-embedding excludes input embeddings and, when
    untied (Qwen3), the LM head as well.
    """
    total = sum(p.numel() for p in model.parameters())
    embedding_numel = model.get_input_embeddings().weight.numel()
    output = model.get_output_embeddings()
    if output is not None and (
        output.weight is not model.get_input_embeddings().weight
    ):
        embedding_numel += output.weight.numel()
    return total, total - embedding_numel


def parse_size_name(name: str) -> float:
    if name.endswith("B"):
        return float(name[:-1]) * 1e9
    assert name.endswith("M")
    return float(name[:-1]) * 1e6


def main() -> None:
    print("=== Gemma 3 dense from-scratch configs")
    failures = []
    for name, (
        depth,
        width,
        inter,
    ) in src.models.gemma3_parameters_to_depths_widths_and_intermediates.items():
        model = src.models.create_causalm_for_pretraining(
            {"model_name": f"Gemma3/Gemma3-{name}", "torch_dtype": "bfloat16"}
        )
        total, non_emb = count_params(model)
        advertised = parse_size_name(name)
        rel_err = abs(total - advertised) / advertised
        status = "ok" if rel_err <= 0.10 else "FAIL (>10% off)"
        print(
            f"  {name:>6}  (L={depth:>2}, h={width:>4}, inter={inter:>5})  "
            f"total={total:>12,}  non-emb={non_emb:>12,}  "
            f"err={100 * rel_err:.2f}%  {status}"
        )
        if rel_err > 0.10:
            failures.append(name)
        del model

    print("\n=== Qwen3 ladder non-embedding counts (for 5.1 matching)")
    for name in ["34M", "63M", "93M", "111M", "165M", "191M", "262M", "344M", "499M"]:
        model = src.models.create_causalm_for_pretraining(
            {"model_name": f"Qwen3/Qwen3-{name}", "torch_dtype": "bfloat16"}
        )
        total, non_emb = count_params(model)
        print(f"  {name:>6}  total={total:>12,}  non-emb={non_emb:>12,}")
        del model

    assert not failures, f"sizes off by more than 10%: {failures}"
    print("\nAll Gemma 3 sizes within 10% of their advertised names.")


if __name__ == "__main__":
    main()
