# Notes

## Pretraining

### Downloading the dataset

We pretrained using HuggingFace's `fineweb-edu-dedup`. Before launching a pretraining run, we advise you to 
first download the files separately:

```python
from datasets import load_dataset
import os


corpus_full_dataset = load_dataset(
    "HuggingFaceTB/smollm-corpus",
    "fineweb-edu-dedup",
    split="train",
    num_proc=min(8, os.cpu_count()),
)
```

Then go make a cup of tea.

### Manually Pushing Model to HuggingFace

Sometimes, pushing a pretrained model to HuggingFace can fail, for example, if you forget to log in.
To manually push the model to Huggingface, 


1. Make sure you are logged in: `huggingface-cli login`
2. Push the model to HuggingFace:

```python
# run this from a small Python script or IPython
from huggingface_hub import create_repo, upload_folder


pted_model_hf_name = "mem_Qwen3-93M_minerva_math_rep_1000_sbst_1.0000_epch_1_ot_2"
repo_id = f"RylanSchaeffer/{pted_model_hf_name}"
local_dir = f"models/pt_language_model/{pted_model_hf_name}"

# Create the repo (no-op if it exists)
create_repo(repo_id, repo_type="model", private=True, exist_ok=True)

# Upload (ignore bulky training leftovers)
upload_folder(
    repo_id=repo_id,
    folder_path=local_dir,
    repo_type="model",
    commit_message="Initial model upload",
    # ignore_patterns=["checkpoint-*", "runs/*", "wandb/*", "*.pt", "*.pth", "events.out.tfevents*"]
)

```