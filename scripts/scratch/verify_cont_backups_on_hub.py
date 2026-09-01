"""Verify every local _cont_* checkpoint has a matching Hub repo under RylanSchaeffer."""

import glob
import os

from huggingface_hub import HfApi

api = HfApi()
hub = {m.modelId.split("/", 1)[1] for m in api.list_models(author="RylanSchaeffer") if "_cont_" in m.modelId}
local = {
    os.path.basename(d)
    for d in glob.glob("models/pt_language_model/*_cont_*")
    if os.path.exists(os.path.join(d, "config.json"))
}
missing = local - hub
print(f"local complete checkpoints: {len(local)}, on Hub: {len(hub & local)}, missing: {sorted(missing) or 'none'}")
assert not missing
