"""List Hub checkpoints whose names do NOT match the plain convention, to find
the contaminant-ablation (`_cont_*`) arm naming. Read-only.

    python scripts/scratch/list_cont_arm_checkpoints.py
"""

from huggingface_hub import HfApi

api = HfApi()
for namespace in ["RylanSchaeffer", "jkazdan"]:
    print(f"=== {namespace}")
    for model in api.list_models(author=namespace):
        base = model.id.split("/", 1)[1]
        if not base.startswith("mem_"):
            continue
        if any(tag in base for tag in ["cont", "perturb", "rephras"]):
            print(f"  {model.id}")
