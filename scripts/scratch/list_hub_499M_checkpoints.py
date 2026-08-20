"""List every 499M checkpoint under RylanSchaeffer, with last-modified dates.

Written to verify the rx6km107 ladder's Hub uploads land under the expected
naming convention (mem_Qwen3-499M_minerva_math_rep_R_...). Full enumeration by
author, never fuzzy search (CLAUDE.md).
"""

from huggingface_hub import HfApi

api = HfApi()
models = sorted(
    (m.modelId, m.lastModified)
    for m in api.list_models(author="RylanSchaeffer", full=True)
    if "499M" in m.modelId
)
for model_id, last_modified in models:
    print(f"{last_modified:%Y-%m-%d %H:%M}  {model_id}")
print(f"{len(models)} total")
