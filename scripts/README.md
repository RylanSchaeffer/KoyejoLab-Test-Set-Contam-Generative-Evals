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

### Manually Pushing to HuggingFace