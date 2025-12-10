import ast
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import wandb

import src.analyze
import src.globals
import src.plot

refresh = False
# refresh = True

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

sweep_ids = [
    "rkx5xfde",  # Qwen 3  34M
    "g31f7bsb",  # Qwen 3  34M
    "ehxxzk5n",  # Qwen 3  34M
    "u7dxxphm",  # Qwen 3  62M
    "o6aoejzc",  # Qwen 3  62M
    "1nwyun1z",  # Qwen 3  62M
    "xbiu535y",  # Qwen 3  62M
    "ho49sshi",  # Qwen 3  93M
    "x8gmmzlo",  # Qwen 3  93M
    "u5xcf726",  # Qwen 3  93M
    "sl086kx0",  # Qwen 3 153M
    "09c432gh",  # Qwen 3 344M
    "09c432gh",  # Qwen 3 344M
    "gsx7gisg",  # Qwen 3 344M
    "6f9ah90l",  # Qwen 3 344M
    "r9fixoce",  # Qwen 3 344M
]

pretrain_run_configs_df: pd.DataFrame = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="memorization-scoring-vs-sampling-pt",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    wandb_username=wandb.api.default_entity,
    finished_only=True,
)

pretrain_run_configs_df = (
    src.analyze.add_pretraining_quantities_to_pretrain_run_configs_df(
        pretrain_run_configs_df=pretrain_run_configs_df
    )
)
# Alias "Num. Replicas Per Epoch" for successful merge.
pretrain_run_configs_df["Num. MATH Test Set Replicas"] = pretrain_run_configs_df[
    "Num. Replicas Per Epoch"
]