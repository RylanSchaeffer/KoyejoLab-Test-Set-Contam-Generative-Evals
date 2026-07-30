"""Cross-check `run.scan_history` against the cached parquet that produced Fig. 1.

`check_boxed_format_rate.py` reported ~0.5% Math Verify for the 344M R=3162 pretrained
checkpoint at greedy decoding, but `notebooks/11_*/results/y=math_verify_by_num_parameters_by_num_replicas.png`
shows ~1.0 for the same condition. One of the two is wrong, and the answer decides whether
the format sanity check's verdict means anything.

This compares, for a handful of runs, the mean `math_verify_score` computed three ways:
the notebook's cached history parquet, `run.history(samples=...)`, and
`run.scan_history(keys=[...])`.
"""

import ast

import pandas as pd
import wandb

CACHE_PARQUET = (
    "notebooks/11_math_qwen3_pt_math_verify/data/"
    "678b1e19c88ea5fdaf60b14abccdb09e_runs_histories.parquet"
)
CACHE_CONFIGS = (
    "notebooks/11_math_qwen3_pt_math_verify/data/"
    "678b1e19c88ea5fdaf60b14abccdb09e_runs_configs.csv"
)


def main() -> None:
    configs = pd.read_csv(CACHE_CONFIGS)
    configs["Model"] = configs["model_config"].apply(
        lambda c: ast.literal_eval(c)["model"]
    )
    target = configs[
        configs["Model"].str.contains("Qwen3-344M")
        & configs["Model"].str.contains("rep_3162_")
        & (configs["temperature"].round(4) == 0.0)
    ]
    print("Matching cached runs:")
    print(target[["run_id", "Model", "temperature"]].to_string(index=False))

    if target.empty:
        print("No matching run in cache; cannot cross-check.")
        return

    run_ids = list(target["run_id"])

    print(f"\nReading cached history parquet (this is ~11 GB, columns only)...")
    history = pd.read_parquet(
        CACHE_PARQUET, columns=["run_id", "math_verify_score"]
    )
    cached = (
        history[history["run_id"].isin(run_ids)]
        .groupby("run_id")["math_verify_score"]
        .agg(["mean", "count"])
    )
    print("\n=== From cached parquet (the source of Fig. 1) ===")
    print(cached.to_string())

    api = wandb.Api(timeout=600)
    for run_id in run_ids:
        run = api.run(f"rylan/memorization-scoring-vs-sampling-eval/{run_id}")
        print(f"\n=== Run {run_id} ({run.state}) ===")
        print(f"  summary math_verify_score: {run.summary.get('math_verify_score')}")

        sampled = run.history(samples=10000)
        if "math_verify_score" in sampled.columns:
            print(
                f"  run.history(samples=10000): mean="
                f"{sampled['math_verify_score'].mean():.4f} n={len(sampled)}"
            )

        n_rows = 0
        n_correct = 0
        n_boxed_key_missing = 0
        for row in run.scan_history(keys=["response", "math_verify_score"]):
            n_rows += 1
            if row.get("math_verify_score"):
                n_correct += 1
            if row.get("response") is None:
                n_boxed_key_missing += 1
        print(
            f"  scan_history(keys=[response, math_verify_score]): "
            f"mean={n_correct / max(n_rows, 1):.4f} n={n_rows} "
            f"(rows missing response: {n_boxed_key_missing})"
        )

        n_rows_solo = 0
        n_correct_solo = 0
        for row in run.scan_history(keys=["math_verify_score"]):
            n_rows_solo += 1
            if row.get("math_verify_score"):
                n_correct_solo += 1
        print(
            f"  scan_history(keys=[math_verify_score]): "
            f"mean={n_correct_solo / max(n_rows_solo, 1):.4f} n={n_rows_solo}"
        )


if __name__ == "__main__":
    main()
