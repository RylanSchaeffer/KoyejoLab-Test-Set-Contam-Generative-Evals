import torch

# This is the most direct way to prevent the JIT compiler from running
# and causing the FakeTensor conflict in your environment.
torch.compiler.disable()

import editdistance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from transformers import AutoTokenizer, pipeline
from tqdm import tqdm
import torch.nn as nn
from typing import Dict, List, Tuple

import src.data
import src.globals
import src.models

tokenizer = AutoTokenizer.from_pretrained(
    # This was meant to be Qwen3, but I had a typo as Qwen 2.
    # After training several models, I don't want to fix it. Whoops.
    "Qwen/Qwen2-1.5B",
    use_fast=True,
    trust_remote_code=True,
)
tokenizer.model_max_length = src.globals.DEFAULT_PRETRAINING_CONFIG["trainer_config"][
    "max_length"
]
# 1) Ensure a distinct padding token exists and is NOT the EOS.
if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})


datasets_dict = src.data.create_dataset_for_pretraining(
    data_config=src.globals.DEFAULT_PRETRAINING_CONFIG["data_config"],
    trainer_config=src.globals.DEFAULT_PRETRAINING_CONFIG["trainer_config"],
    tokenizer=tokenizer,
)

# print(df)
