"""Locate the SFT checkpoints on the HF Hub.

The 0-shot SFT phase failed with 404 on every checkpoint. The names came from W&B run configs
(`boxed_format_rates.csv`), which record them under `RylanSchaeffer/`, but the Hub disagrees.
`src/globals.py` still defaults to a `jkazdan/...` SFT checkpoint, so the likely explanation is
that they live under a collaborator's namespace — or were renamed/removed.
"""

from huggingface_hub import HfApi

AUTHORS = ["RylanSchaeffer", "jkazdan"]


def main() -> None:
    api = HfApi()
    for author in AUTHORS:
        try:
            models = list(api.list_models(author=author, search="mem_Qwen3"))
        except Exception as e:
            print(f"[{author}] error: {type(e).__name__}: {e}")
            continue
        sft = sorted(m.id for m in models if m.id.endswith("_sft"))
        print(f"\n[{author}] {len(models)} mem_Qwen3 models, {len(sft)} ending in _sft")
        for model_id in sft[:12]:
            print(f"    {model_id}")
        if len(sft) > 12:
            print(f"    ... and {len(sft) - 12} more")


if __name__ == "__main__":
    main()
