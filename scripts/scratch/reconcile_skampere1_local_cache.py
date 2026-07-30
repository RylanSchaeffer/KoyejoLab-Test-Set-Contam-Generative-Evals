"""Decide which copy of notebook 16's runs_configs cache to keep.

The July 27 pull preserved `..._runs_configs.csv.skampere1-local` because it differed from the
committed `..._runs_configs.csv`. Both are W&B run-config dumps for the same sweep, so the
question is only whether they describe the same runs — if one is a strict superset, keep it
and drop the other; if they disagree about a shared run, that needs a human.
"""

import pandas as pd

BASE = (
    "notebooks/16_sft_generalization_teacher_forcing_perturbed/data/"
    "09ea3c9b2459d4f18e880b76d0af73d2_runs_configs"
)
COMMITTED = f"{BASE}.csv"
LOCAL = f"{BASE}.csv.skampere1-local"


def main() -> None:
    committed = pd.read_csv(COMMITTED, low_memory=False)
    local = pd.read_csv(LOCAL, low_memory=False)

    print(f"committed : {committed.shape[0]} rows x {committed.shape[1]} cols")
    print(f"local     : {local.shape[0]} rows x {local.shape[1]} cols")

    committed_ids = set(committed["run_id"]) if "run_id" in committed else set()
    local_ids = set(local["run_id"]) if "run_id" in local else set()

    print(f"\nrun_ids committed only : {len(committed_ids - local_ids)}")
    print(f"run_ids local only     : {len(local_ids - committed_ids)}")
    print(f"run_ids shared         : {len(committed_ids & local_ids)}")

    only_committed = sorted(committed_ids - local_ids)
    only_local = sorted(local_ids - committed_ids)
    if only_committed:
        print(f"  committed-only ids: {only_committed[:10]}")
    if only_local:
        print(f"  local-only ids    : {only_local[:10]}")

    committed_cols = set(committed.columns)
    local_cols = set(local.columns)
    print(f"\ncolumns committed only : {sorted(committed_cols - local_cols)[:10]}")
    print(f"columns local only     : {sorted(local_cols - committed_cols)[:10]}")

    # For runs present in both, check whether the finished-state and model agree.
    shared = sorted(committed_ids & local_ids)
    if shared and "State" in committed and "State" in local:
        merged = committed[committed["run_id"].isin(shared)][
            ["run_id", "State"]
        ].merge(
            local[local["run_id"].isin(shared)][["run_id", "State"]],
            on="run_id",
            suffixes=("_committed", "_local"),
        )
        disagree = merged[merged["State_committed"] != merged["State_local"]]
        print(f"\nshared runs whose State disagrees: {len(disagree)}")
        if not disagree.empty:
            print(disagree.head(10).to_string(index=False))

    print("\n=== Verdict ===")
    if local_ids >= committed_ids and len(local_ids) > len(committed_ids):
        print("local is a strict superset -> keep local, overwrite the committed copy")
    elif committed_ids >= local_ids and len(committed_ids) > len(local_ids):
        print("committed is a strict superset -> keep committed, delete the .skampere1-local copy")
    elif committed_ids == local_ids:
        print("same run set -> differences are column/formatting only; keep committed")
    else:
        print("neither is a superset -> needs a human decision")


if __name__ == "__main__":
    main()
