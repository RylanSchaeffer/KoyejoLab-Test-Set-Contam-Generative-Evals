"""Upload the sole-copy _cont_* ablation checkpoints to the Hub.

The Phase 6 audit (2026-08-17) found these continued-pretraining ablation
checkpoints exist ONLY as local directories -- zero copies on the Hub in any
namespace. The shared volume is now at 100%, one ENOSPC has already killed a
run, and a disk failure would erase the perturbed/rephrased control arms the
manuscript's ablation depends on. This uploads each as a private model repo
under RylanSchaeffer.

Refuses to run unless the active identity is RylanSchaeffer.

    export HF_TOKEN="$(cat /lfs/skampere1/0/rschaef/.hf_token)"
    uv run python scripts/scratch/backup_cont_checkpoints_to_hub.py
"""

import glob
import os

from huggingface_hub import HfApi

api = HfApi()
username = api.whoami()["name"]
assert username == "RylanSchaeffer", f"wrong Hub identity: {username}"

local_dirs = sorted(glob.glob("models/pt_language_model/*_cont_*"))
assert local_dirs, "no _cont_* checkpoints found"

for local_dir in local_dirs:
    name = os.path.basename(local_dir)
    if not os.path.exists(os.path.join(local_dir, "config.json")):
        print(f"SKIP {name}: no config.json (not a complete checkpoint)")
        continue
    repo_id = f"RylanSchaeffer/{name}"
    api.create_repo(repo_id, private=True, exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=local_dir)
    print(f"UPLOADED {repo_id}")

print("done")
