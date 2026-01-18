"""Analysis utilities for extracting and processing W&B experiment results.

This module provides functions for:
- Downloading run configurations and histories from Weights & Biases
- Extracting derived quantities (FLOP, tokens, contamination levels) from configs
- Fitting neural scaling laws to experimental data
- Setting up notebook analysis directories

The functions use concurrent downloads with retry logic for robust W&B API access.

Example:
    >>> from src.analyze import download_wandb_project_runs_configs
    >>> df = download_wandb_project_runs_configs(
    ...     wandb_project_path="contamination-study",
    ...     data_dir="./data",
    ...     sweep_ids=["abc123", "def456"]
    ... )
"""

import ast
import concurrent.futures
import hashlib
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow
import requests
import wandb
from tqdm import tqdm

import src.globals
import src.neural_scaling_laws


def add_pretraining_quantities_to_pretrain_runs_configs_df(
    pretrain_run_configs_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add derived quantities to a pretraining runs DataFrame.

    Extracts and computes useful columns from the raw W&B config strings:
    model name, parameter count, contamination level, compute (FLOP), etc.

    Args:
        pretrain_run_configs_df: DataFrame from download_wandb_project_runs_configs.

    Returns:
        DataFrame with additional columns: Model, Num. Parameters, Parameters,
        Benchmark Subset Fraction, Overtrain Multiplier, Num. Tokens, FLOP (6ND),
        Num. Replicas Per Epoch, Num. Epochs, Num. MATH Test Set Replicas.
    """
    # Extract basic quantities.
    pretrain_run_configs_df["Model"] = pretrain_run_configs_df["model_config"].apply(
        lambda model_config: ast.literal_eval(model_config)["model_name"]
    )
    pretrain_run_configs_df["Num. Parameters"] = pretrain_run_configs_df[
        "model/num_parameters"
    ]
    pretrain_run_configs_df["Parameters"] = pretrain_run_configs_df[
        "Num. Parameters"
    ].apply(lambda n: f"{n / 1_000_000:.0f}M")
    pretrain_run_configs_df["Benchmark Subset Fraction"] = pretrain_run_configs_df[
        "data_config"
    ].apply(
        lambda data_config: ast.literal_eval(data_config)["benchmark_subset_fraction"]
    )
    pretrain_run_configs_df["Overtrain Multiplier"] = pretrain_run_configs_df[
        "trainer_config"
    ].apply(
        lambda trainer_config: ast.literal_eval(trainer_config)["overtrain_multiplier"]
    )
    pretrain_run_configs_df["Num. Tokens"] = (
        20.0
        * pretrain_run_configs_df["Overtrain Multiplier"]
        * pretrain_run_configs_df["Num. Parameters"]
    )
    pretrain_run_configs_df["FLOP (6ND)"] = (
        6
        * pretrain_run_configs_df["Num. Parameters"]
        * pretrain_run_configs_df["Num. Tokens"]
    )
    pretrain_run_configs_df["Num. Replicas Per Epoch"] = pretrain_run_configs_df[
        "data_config"
    ].apply(
        lambda data_config: ast.literal_eval(data_config)[
            "num_benchmark_replicas_per_epoch"
        ]
    )
    pretrain_run_configs_df["Num. Epochs"] = pretrain_run_configs_df[
        "trainer_config"
    ].apply(lambda trainer_config: ast.literal_eval(trainer_config)["num_train_epochs"])
    pretrain_run_configs_df["Num. MATH Test Set Replicas"] = (
        pretrain_run_configs_df["Num. Replicas Per Epoch"]
        * pretrain_run_configs_df["Num. Epochs"]
    )
    return pretrain_run_configs_df


def add_pretraining_quantities_to_supervised_finetuning_runs_configs_df(
    sft_runs_configs_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add derived quantities to an SFT runs DataFrame.

    Parses model names to extract pretraining contamination levels and
    computes derived quantities for analysis.

    Args:
        sft_runs_configs_df: DataFrame from download_wandb_project_runs_configs.

    Returns:
        DataFrame with additional columns extracted from model naming convention.
    """
    sft_runs_configs_df["Model"] = sft_runs_configs_df["model_config"].apply(
        lambda model_config: ast.literal_eval(model_config)[
            "initial_model_name_or_path"
        ]
    )
    sft_runs_configs_df["Parameters"] = sft_runs_configs_df["Model"].apply(
        lambda model_name: re.search(r"-([\d.]+[MB])", model_name).group(1)
    )
    sft_runs_configs_df["Num. Parameters"] = sft_runs_configs_df["Parameters"].apply(
        lambda parameters: src.globals.MODEL_NAMES_TO_PARAMETERS_DICT[parameters]
    )
    sft_runs_configs_df["Num. Replicas Per Epoch"] = sft_runs_configs_df["Model"].apply(
        lambda model_name: int(re.search(r"rep_(\d+)_sbst", model_name).group(1))
    )
    sft_runs_configs_df["Num. Epochs"] = sft_runs_configs_df["Model"].apply(
        lambda model_name: int(re.search(r"epch_(\d+)_ot", model_name).group(1))
    )
    sft_runs_configs_df["Overtrain Multiplier"] = sft_runs_configs_df["Model"].apply(
        lambda model_name: int(re.search(r"ot_(\d+)", model_name).group(1))
    )
    sft_runs_configs_df["Num. MATH Test Set Replicas"] = (
        sft_runs_configs_df["Num. Replicas Per Epoch"]
        * sft_runs_configs_df["Num. Epochs"]
    )
    sft_runs_configs_df["Num. Tokens"] = 20 * sft_runs_configs_df["Num. Parameters"]
    sft_runs_configs_df["FLOP (6ND)"] = (
        6 * sft_runs_configs_df["Num. Parameters"] * sft_runs_configs_df["Num. Tokens"]
    )
    sft_runs_configs_df.rename(columns={"temperature": "Temp."}, inplace=True)
    sft_runs_configs_df["Temp."] = np.round(sft_runs_configs_df["Temp."], decimals=2)
    return sft_runs_configs_df


def calculate_compute_contamination_exchange_rate(
    loss: float, irreducible_error: float, prefactor: float, exponent: float
) -> float:
    """Calculate equivalent compute for a given loss using scaling law parameters.

    Given a loss value and fitted scaling law parameters L = E + C_0 * C^(-alpha),
    computes the compute C that would achieve that loss without contamination.

    Args:
        loss: Observed loss value.
        irreducible_error: Fitted E (asymptotic loss).
        prefactor: Fitted C_0 (scaling prefactor).
        exponent: Fitted alpha (scaling exponent).

    Returns:
        Equivalent compute value.
    """
    return np.power((loss - irreducible_error) / prefactor, -1.0 / exponent)


def download_wandb_project_runs_configs(
    wandb_project_path: str,
    data_dir: str,
    sweep_ids: Optional[List[str]] = None,
    finished_only: bool = True,
    refresh: bool = False,
    wandb_username: Optional[str] = None,
    filetype: str = "csv",
    max_workers: int = 10,
) -> pd.DataFrame:
    """Download run configurations from W&B sweeps with parallel fetching.

    Fetches run configs from specified sweeps, caches to disk, and returns
    as a DataFrame. Uses ThreadPoolExecutor for concurrent API calls.

    Args:
        wandb_project_path: W&B project path (e.g., "contamination-study").
        data_dir: Directory for caching downloaded data.
        sweep_ids: List of sweep IDs to download. Required.
        finished_only: If True, filter to only successfully finished runs.
        refresh: If True, re-download even if cache exists.
        wandb_username: W&B username. If None, uses API viewer's username.
        filetype: Cache format ("csv", "feather", or "parquet").
        max_workers: Number of parallel download threads.

    Returns:
        DataFrame with run configurations and summary metrics.
    """
    assert filetype in {"csv", "feather", "parquet"}

    filename = "sweeps=" + ",".join(sweep_ids)
    hashed_filename = hashlib.md5(filename.encode()).hexdigest()
    runs_configs_df_path = os.path.join(
        data_dir, hashed_filename + f"_runs_configs.{filetype}"
    )

    if refresh or not os.path.isfile(runs_configs_df_path):
        print(f"Creating {runs_configs_df_path} anew.")

        api = wandb.Api(timeout=600)

        if wandb_username is None:
            wandb_username = api.viewer.username

        sweep_results_list = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_run = {}

            for sweep_id in sweep_ids:
                try:
                    sweep = api.sweep(
                        f"{wandb_username}/{wandb_project_path}/{sweep_id}"
                    )
                    for run in sweep.runs:
                        future_to_run[
                            executor.submit(
                                download_wandb_project_runs_configs_helper, run
                            )
                        ] = run
                except Exception as e:
                    print(f"Error processing sweep {sweep_id}: {str(e)}")

            for future in tqdm(
                concurrent.futures.as_completed(future_to_run), total=len(future_to_run)
            ):
                result = future.result()
                if result is not None:
                    sweep_results_list.append(result)

        runs_configs_df = pd.DataFrame(sweep_results_list)
        runs_configs_df.reset_index(inplace=True, drop=True)

        # Save to disk
        runs_configs_df.to_csv(
            runs_configs_df_path.replace(filetype, "csv"), index=False
        )
        try:
            runs_configs_df.to_feather(
                runs_configs_df_path.replace(filetype, "feather")
            )
        except Exception as e:
            print(f"Error saving to feather: {str(e)}")

        try:
            runs_configs_df.to_parquet(
                runs_configs_df_path.replace(filetype, "parquet"), index=False
            )
        except Exception as e:
            print(f"Error saving to parquet: {str(e)}")

        print(f"Regenerated and wrote {runs_configs_df_path} to disk.")
        del runs_configs_df

    print(f"Reading {runs_configs_df_path} from disk.")
    if filetype == "csv":
        runs_configs_df = pd.read_csv(runs_configs_df_path)
    elif filetype == "feather":
        runs_configs_df = pd.read_feather(runs_configs_df_path)
    elif filetype == "parquet":
        runs_configs_df = pd.read_parquet(runs_configs_df_path)
    else:
        raise ValueError(f"Invalid filetype: {filetype}")
    print(f"Loaded {runs_configs_df_path} from disk.")

    # Keep only finished runs
    finished_runs = runs_configs_df["State"] == "finished"
    print(
        f"% of successfully finished runs: {100.0 * finished_runs.mean():.2f}% ({finished_runs.sum()} / {len(finished_runs)})"
    )

    if finished_only:
        runs_configs_df = runs_configs_df[finished_runs]

        # Check that we don't have an empty data frame.
        assert len(runs_configs_df) > 0

        # Ensure we aren't working with a slice.
        runs_configs_df = runs_configs_df.copy()

    return runs_configs_df


def download_wandb_project_runs_configs_helper(run: Any) -> Optional[Dict[str, Any]]:
    """Helper to extract config and summary from a single W&B run."""
    try:
        summary = run.summary._json_dict
        summary.update({k: v for k, v in run.config.items() if not k.startswith("_")})
        summary.update(
            {
                "State": run.state,
                "Sweep": run.sweep.id if run.sweep is not None else None,
                "run_id": run.id,
                "run_name": run.name,
            }
        )
        return summary
    except Exception as e:
        print(f"Error processing run {run.id}: {str(e)}")
        return None


def download_wandb_pretraining_runs_configs(
    wandb_project_path: str,
    data_dir: str,
    sweep_ids: Optional[List[str]] = None,
    finished_only: bool = True,
    refresh: bool = False,
    wandb_username: Optional[str] = None,
    filetype: str = "csv",
    max_workers: int = 10,
) -> pd.DataFrame:
    """Download pretraining run configs and add derived quantities.

    Convenience wrapper that downloads configs and automatically extracts
    pretraining-specific quantities like contamination levels and compute.

    Args:
        wandb_project_path: W&B project path.
        data_dir: Directory for caching.
        sweep_ids: List of sweep IDs to download.
        finished_only: If True, filter to finished runs only.
        refresh: If True, re-download from W&B.
        wandb_username: W&B username (optional).
        filetype: Cache format.
        max_workers: Parallel download threads.

    Returns:
        DataFrame with configs and derived pretraining quantities.
    """
    pt_runs_configs_df: pd.DataFrame = src.analyze.download_wandb_project_runs_configs(
        wandb_project_path=wandb_project_path,
        data_dir=data_dir,
        sweep_ids=sweep_ids,
        refresh=refresh,
        wandb_username=wandb_username,
        finished_only=finished_only,
        filetype=filetype,
        max_workers=max_workers,
    )

    # Extract information to make analyzing and visualizing easier.
    pt_runs_configs_df["Num. Replicas Per Epoch"] = pt_runs_configs_df[
        "data_config"
    ].apply(
        lambda data_config: ast.literal_eval(data_config)[
            "num_benchmark_replicas_per_epoch"
        ]
    )
    pt_runs_configs_df["Benchmark"] = pt_runs_configs_df["data_config"].apply(
        lambda data_config: ast.literal_eval(data_config)["benchmark"]
    )
    pt_runs_configs_df["Benchmark Subset Fraction"] = pt_runs_configs_df[
        "data_config"
    ].apply(
        lambda data_config: ast.literal_eval(data_config)["benchmark_subset_fraction"]
    )

    def compute_number_of_benchmark_tokens_per_replica(row: pd.Series):
        if row["Benchmark"] == "EleutherAI/minerva_math":
            num_tokens_in_benchmark = 1.5e6
        else:
            raise NotImplementedError
        return num_tokens_in_benchmark * row["Benchmark Subset Fraction"]

    pt_runs_configs_df["Benchmark Subset Num. Tokens"] = pt_runs_configs_df.apply(
        compute_number_of_benchmark_tokens_per_replica,
        axis=1,
    )

    pt_runs_configs_df["Num. Epochs"] = pt_runs_configs_df["train/epoch"]
    pt_runs_configs_df["Num. Replicas"] = (
        pt_runs_configs_df["Num. Replicas Per Epoch"]
        * pt_runs_configs_df["Num. Epochs"]
    )
    pt_runs_configs_df["Model"] = pt_runs_configs_df["model_config"].apply(
        lambda model_config: ast.literal_eval(model_config)["model_name"]
    )
    # pt_runs_configs_df["Num. Parameters"] = pt_runs_configs_df["Model"].apply(
    #     src.analyze.extract_num_model_parameters
    # )
    pt_runs_configs_df["Num. Parameters"] = pt_runs_configs_df["model/num_parameters"]
    pt_runs_configs_df["Num. Tokens"] = pt_runs_configs_df[
        "eval_after/num_input_tokens_seen"
    ]
    pt_runs_configs_df["FLOP (6ND)"] = (
        6 * pt_runs_configs_df["Num. Parameters"] * pt_runs_configs_df["Num. Tokens"]
    )

    # Use slightly nicer column names.
    pt_runs_configs_df["benchmark_loss"] = pt_runs_configs_df[
        "eval_after/eval_benchmark_loss"
    ]
    pt_runs_configs_df["eval_loss"] = pt_runs_configs_df["eval_after/eval_eval_loss"]

    return pt_runs_configs_df


def download_wandb_project_runs_histories(
    wandb_project_path: str,
    data_dir: str,
    sweep_ids: Optional[List[str]] = None,
    wandb_run_history_num_samples: int = 10000,
    refresh: bool = False,
    wandb_username: Optional[str] = None,
    filetype: str = "csv",
    nrows_to_read: Optional[int] = None,
    cols_to_drop: Optional[List[str]] = None,
    max_workers: int = 10,
) -> pd.DataFrame:
    """Download training histories from W&B runs with parallel fetching.

    Fetches the logged metrics over training time for all runs in the
    specified sweeps. Useful for analyzing training dynamics.

    Args:
        wandb_project_path: W&B project path.
        data_dir: Directory for caching.
        sweep_ids: List of sweep IDs to download.
        wandb_run_history_num_samples: Max history samples per run.
        refresh: If True, re-download from W&B.
        wandb_username: W&B username (optional).
        filetype: Cache format.
        nrows_to_read: Limit rows when reading cache (for debugging).
        cols_to_drop: Columns to drop from history.
        max_workers: Parallel download threads.

    Returns:
        DataFrame with training histories, one row per logged step per run.
    """
    assert filetype in {"csv", "feather", "parquet"}

    # Hash because otherwise too long.
    filename = "sweeps=" + ",".join(sweep_ids)
    hashed_filename = hashlib.md5(filename.encode()).hexdigest()
    runs_histories_df_path = os.path.join(
        data_dir, hashed_filename + f"_runs_histories.{filetype}"
    )
    if refresh or not os.path.isfile(runs_histories_df_path):
        # Download sweep results
        api = wandb.Api(timeout=6000)

        if wandb_username is None:
            wandb_username = api.viewer.username

        runs_histories_list = []
        print("Downloading runs' histories...")
        for sweep_id in sweep_ids:
            sweep = api.sweep(f"{wandb_username}/{wandb_project_path}/{sweep_id}")

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                future_to_run = {
                    executor.submit(
                        download_wandb_project_runs_histories_helper,
                        run,
                        wandb_run_history_num_samples,
                        cols_to_drop,
                    ): run
                    for run in sweep.runs
                }

                for future in tqdm(
                    concurrent.futures.as_completed(future_to_run),
                    total=len(future_to_run),
                ):
                    run = future_to_run[future]
                    try:
                        history = future.result()
                        if history is not None:
                            runs_histories_list.append(history)
                    except Exception as exc:
                        print(f"{run.id} generated an exception: {exc}")

        assert len(runs_histories_list) > 0
        runs_histories_df = pd.concat(runs_histories_list, sort=False)

        runs_histories_df.sort_values(["run_id"], ascending=True, inplace=True)
        runs_histories_df.reset_index(inplace=True, drop=True)

        assert len(runs_histories_list) > 0
        runs_histories_df = pd.concat(runs_histories_list, sort=False)

        runs_histories_df.sort_values(["run_id"], ascending=True, inplace=True)
        runs_histories_df.reset_index(inplace=True, drop=True)

        # Save all three because otherwise this is a pain in the ass.
        # runs_histories_df.to_csv(
        #     runs_histories_df_path.replace(filetype, "csv"), index=False
        # )
        try:
            runs_histories_df.to_feather(
                runs_histories_df_path.replace(filetype, "feather")
            )
        except BaseException:
            # pyarrow.lib.ArrowInvalid: ("Could not convert 'NaN' with type str: tried to convert to double", 'Conversion failed for column loss/score_model=claude3opus with type object')
            pass
        try:
            runs_histories_df.to_parquet(
                runs_histories_df_path.replace(filetype, "parquet"), index=False
            )
        except pyarrow.lib.ArrowInvalid:
            # pyarrow.lib.ArrowInvalid: ("Could not convert 'NaN' with type str: tried to convert to double", 'Conversion failed for column loss/score_model=claude3opus with type object')
            pass
        print(f"Wrote {runs_histories_df_path} to disk")
        del runs_histories_df

    print(f"Loading {runs_histories_df_path} from disk.")
    if filetype == "csv":
        runs_histories_df = pd.read_csv(runs_histories_df_path, nrows=nrows_to_read)
    elif filetype == "feather":
        runs_histories_df = pd.read_feather(runs_histories_df_path)
    elif filetype == "parquet":
        runs_histories_df = pd.read_parquet(runs_histories_df_path)
    else:
        raise ValueError(f"Invalid filetype: {filetype}")
    print(f"Loaded {runs_histories_df_path} from disk.")

    return runs_histories_df


def download_wandb_project_runs_histories_helper(
    run: Any,
    wandb_run_history_num_samples: int,
    cols_to_drop: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """Helper to download history from a single run with retry logic."""
    history = None
    for num_attempts in range(5):
        try:
            history = run.history(samples=wandb_run_history_num_samples)
            break
        except (requests.exceptions.HTTPError, wandb.errors.CommError):
            print(f"Retrying run {run.id}...")
            time.sleep(3)

    if history is None or history.empty:
        return None

    if cols_to_drop is not None:
        history.drop(columns=cols_to_drop, inplace=True)
    history["run_id"] = run.id
    return history


def extract_hf_model_name_or_path(model_config: str) -> str:
    """Extract HuggingFace model path from a config string."""
    hf_model_name_or_path = ast.literal_eval(model_config)["model"]
    return hf_model_name_or_path


def extract_num_model_parameters(model_name: str) -> int:
    """Extract parameter count from a model name string.

    Handles both custom pretrained model names (e.g.,
    "RylanSchaeffer/mem_Qwen3-34M_...") and base model names.

    Args:
        model_name: Model identifier string.

    Returns:
        Number of parameters as an integer.
    """
    if model_name.startswith("RylanSchaeffer"):
        # "RylanSchaeffer/mem_Qwen3-34M_minerva_math_replicas_0_epch_1_ot_1_pt"
        # will become "Qwen3-34M".
        base_model_name = model_name.replace("RylanSchaeffer/mem_", "").split("_")[0]
    else:
        match = re.search(r"model_(.*)_dataset", model_name)
        if match:
            base_model_name = match.group(1)
        else:
            # Assume base model. Drop the organization name and take only the base model name.
            # Example: "Qwen/Qwen2.5-1.5B" becomes "Qwen2.5-1.5B"
            base_model_name = model_name.split("/")[-1]
    num_model_parameters = src.globals.MODEL_NAMES_TO_PARAMETERS_DICT[base_model_name]
    return num_model_parameters


def extract_num_train_epochs(model_name: str) -> int:
    """Extract number of training epochs from a model name string."""
    if "RylanSchaeffer" not in model_name:
        # Base model. We assume 0 epochs.
        return 0
    match = re.search(r"_epochs_(\d+)_seed_", model_name)
    if not match:
        raise ValueError
    num_epochs = int(match.group(1))
    return num_epochs


def fit_neural_scaling_law(
    df: pd.DataFrame,
    x_col: str = "Pretraining Compute",
    y_col: str = "neg_log_",
    exclude_nans: bool = True,
    additional_columns_to_add: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Fit a power law scaling model to experimental data.

    Fits the Chinchilla-style scaling law: L = E + C_0 * C^(-alpha)
    where L is loss, C is compute, and E, C_0, alpha are fitted parameters.

    Args:
        df: DataFrame containing the data to fit.
        x_col: Column name for the independent variable (compute).
        y_col: Column name for the dependent variable (loss).
        exclude_nans: If True, exclude rows with NaN values.
        additional_columns_to_add: Extra columns to include in results.

    Returns:
        Dictionary with fit parameters, loss, convergence status, and
        optionally additional column values from the input DataFrame.
    """
    x_vals_all = df[x_col].values
    y_vals_all = df[y_col].values
    if exclude_nans:
        nan_mask = np.isnan(x_vals_all) | np.isnan(y_vals_all)
        print(f"Excluding {np.sum(nan_mask)} NaNs out of {len(nan_mask)} entries")
        x_vals = x_vals_all[~nan_mask]
        y_vals = y_vals_all[~nan_mask]
    else:
        x_vals = np.copy(x_vals_all)
        y_vals = np.copy(y_vals_all)

    if len(x_vals) >= 3:
        # Fit a power law loss = E + A * FLOPS^(-alpha) via linear regression in log space.
        best_fit_result, y_all_pred = src.neural_scaling_laws.fit_chinchilla_scaling(
            x_all=x_vals, y_all=y_vals, functional_form="compute"
        )
        fit_results_dict = dict(
            covariate_cols=x_col,
            target_col=y_col,
            fit_loss=best_fit_result.fit_loss,
            fit_converged=best_fit_result.converged,
        )
        for k, v in best_fit_result.fit_params.items():
            fit_results_dict[f"fit_param_{k}"] = v

        # Convert the log-space parameters to the original space.
        fit_results_dict["fit_param_C_0"] = np.exp(fit_results_dict["fit_param_c_0"])
        fit_results_dict["fit_param_E_0"] = np.exp(fit_results_dict["fit_param_e_0"])

    else:
        # Create a mock fit result with NaNs.
        fit_results_dict = dict(
            covariate_cols=x_col,
            target_col=y_col,
            fit_loss=np.nan,
            fit_converged=False,
        )

    if additional_columns_to_add is not None:
        for col in additional_columns_to_add:
            fit_results_dict[col] = df[col].values[0]

    return fit_results_dict


def setup_notebook_dir(
    notebook_dir: str,
    refresh: bool = False,
) -> Tuple[str, str]:
    """Set up data and results directories for a notebook.

    Creates data/ and results/ subdirectories. Optionally clears the
    results directory for a fresh analysis run.

    Args:
        notebook_dir: Base directory for the notebook.
        refresh: If True, delete existing results directory.

    Returns:
        Tuple of (data_dir, results_dir) paths.
    """
    # Declare paths.
    data_dir = os.path.join(notebook_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    results_dir = os.path.join(notebook_dir, "results")
    if refresh:
        import shutil

        if os.path.exists(results_dir) and os.path.isdir(results_dir):
            shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    return data_dir, results_dir
