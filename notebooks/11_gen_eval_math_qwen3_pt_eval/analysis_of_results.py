import ast
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib.pyplot as plt
import matplotlib.ticker
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

refresh = True

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

sweep_ids = [
    "6y9dy2ow",  # Qwen 3   34M     1xOT    Subset Fraction=1.0
    "lnrpy3ed",  # Qwen 3   34M     1xOT    Subset Fraction=1.0     More temperatures.
    "5oo55o9s",  # Qwen 3   62M     1xOT    Subset Fraction=1.0
    "10q465ij",  # Qwen 3   62M     1xOT    Subset Fraction=1.0     More temperatures.
    "q5uoy1eu",  # Qwen 3   93M     1xOT    Subset Fraction=1.0
    "f5djvfth",  # Qwen 3   93M     1xOT    Subset Fraction=1.0     More temperatures.
    "vnz1h147",  # Qwen 3  153M     1xOT    Subset Fraction=1.0
    "xkzfmbhk",  # Qwen 3  153M     1xOT    Subset Fraction=1.0
    "39rugx2e",  # Qwen 3  343M     1xOT    Subset Fraction=1.0     More temperatures.
]

refresh = False

eval_runs_configs_df: pd.DataFrame = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="memorization-scoring-vs-sampling-eval",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    wandb_username='rylan',#wandb.api.default_entity,
    finished_only=True,
)

eval_runs_configs_df["Model"] = eval_runs_configs_df["model_config"].apply(
    lambda model_config: ast.literal_eval(model_config)["model"]
)
eval_runs_configs_df["Parameters"] = eval_runs_configs_df["Model"].apply(
    lambda model_name: re.search(r"(Qwen3-[\d.]+[MB])", model_name).group(1)
)
eval_runs_configs_df["Num. Parameters"] = eval_runs_configs_df["Parameters"].apply(
    lambda parameters: src.globals.MODEL_NAMES_TO_PARAMETERS_DICT[parameters]
)
eval_runs_configs_df["Num. Replicas Per Epoch"] = eval_runs_configs_df["Model"].apply(
    lambda model_name: int(re.search(r"rep_(\d+)_sbst", model_name).group(1))
)
eval_runs_configs_df["Num. Epochs"] = eval_runs_configs_df["Model"].apply(
    lambda model_name: int(re.search(r"epch_(\d+)_ot", model_name).group(1))
)
eval_runs_configs_df["Overtrain Multiplier"] = eval_runs_configs_df["Model"].apply(
    lambda model_name: int(re.search(r"ot_(\d+)", model_name).group(1))
)
eval_runs_configs_df["Num. MATH Test Set Replicas"] = (
    eval_runs_configs_df["Num. Replicas Per Epoch"]
    * eval_runs_configs_df["Num. Epochs"]
)
eval_runs_configs_df["Num. Tokens"] = 20 * eval_runs_configs_df["Num. Parameters"]
eval_runs_configs_df["FLOP (6ND)"] = (
    6 * eval_runs_configs_df["Num. Parameters"] * eval_runs_configs_df["Num. Tokens"]
)
eval_runs_configs_df.rename(columns={"temperature": "Temp."}, inplace=True)

print(eval_runs_configs_df.head())