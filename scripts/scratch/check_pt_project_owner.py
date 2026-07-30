"""Check whether the 'missing' pretraining project simply belongs to a different entity.

`docs/INFRASTRUCTURE.md` records `memorization-scoring-vs-sampling-pt` as non-existent, and
notebook 11 cannot refresh its pretraining data because of it. But the entity probe found
`jkazdan/memorization-scoring-vs-sampling-pt` with 236 runs. The notebooks resolve the project
against `wandb.api.default_entity` (= `rylan`), which is why it looked absent.

This checks whether that project contains the sweep IDs notebook 11 asks for. If it does, the
fix is a one-line `wandb_username="jkazdan"` rather than a re-run of pretraining.
"""

import wandb

# The pretraining sweeps notebook 11 tries to download.
NOTEBOOK_11_PT_SWEEPS = [
    "rkx5xfde", "g31f7bsb", "ehxxzk5n",           # 34M
    "u7dxxphm", "o6aoejzc", "1nwyun1z", "xbiu535y",  # 62M
    "ho49sshi", "x8gmmzlo", "u5xcf726",           # 93M
    "sl086kx0",                                    # 153M
    "09c432gh", "gsx7gisg", "6f9ah90l", "r9fixoce",  # 344M
]

CANDIDATES = [
    ("jkazdan", "memorization-scoring-vs-sampling-pt"),
    ("joshteam", "memorization-scoring-vs-sampling-pt"),
    ("rylan", "memorization-scoring-vs-sampling-pt-v2"),
    ("rylan", "memorization-scoring-vs-sampling-pt"),
]


def main() -> None:
    api = wandb.Api(timeout=600)

    for entity, project in CANDIDATES:
        print(f"\n=== {entity}/{project} ===")
        try:
            runs = api.runs(f"{entity}/{project}", per_page=200)
            sweep_ids = {}
            n_runs = 0
            for run in runs:
                n_runs += 1
                sweep_id = run.sweep.id if run.sweep is not None else None
                if sweep_id:
                    sweep_ids[sweep_id] = sweep_ids.get(sweep_id, 0) + 1
            print(f"  {n_runs} runs across {len(sweep_ids)} sweeps")
        except Exception as e:
            print(f"  NOT ACCESSIBLE: {type(e).__name__}: {str(e)[:120]}")
            continue

        wanted = set(NOTEBOOK_11_PT_SWEEPS)
        present = wanted & set(sweep_ids)
        missing = wanted - set(sweep_ids)
        print(f"  notebook-11 sweeps present: {len(present)}/{len(wanted)}")
        if present:
            print(f"    present: {sorted(present)}")
        if missing:
            print(f"    missing: {sorted(missing)}")


if __name__ == "__main__":
    main()
