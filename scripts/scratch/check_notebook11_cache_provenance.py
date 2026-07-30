"""Determine which sweep list produced notebook 11's cached history, and compare protocols.

`src.analyze.download_wandb_project_runs_configs` names its cache
`md5("sweeps=" + ",".join(sweep_ids))`, and only re-downloads when the file is absent or
`refresh=True`. Notebook 11 currently declares the 4-shot sweep IDs but keeps the old 0-shot
IDs commented out directly above them, and its cache contains run `1yml0np5`, which belongs
to sweep `39rugx2e` — one of the *commented-out* 0-shot sweeps.

If the cache hash matches the old list, then `notebooks/11_*/results/*.png` — the figures the
manuscript's Finding #1 rests on — were rendered from the superseded 0-shot protocol and
were never regenerated after the switch to 4-shot boxed-required scoring.

The script also prints example generations from the same checkpoint under both protocols, so
the difference in scores can be attributed rather than guessed at.
"""

import hashlib

import wandb

OLD_0SHOT_SWEEPS = [
    "6y9dy2ow",
    "lnrpy3ed",
    "5oo55o9s",
    "10q465ij",
    "q5uoy1eu",
    "f5djvfth",
    "vnz1h147",
    "xkzfmbhk",
    "39rugx2e",
]
NEW_4SHOT_SWEEPS = [
    "qx2c4702",
    "dkiui6we",
    "cx8y41bw",
    "4w5x8hez",
    "mprek7pj",
]
OBSERVED_CACHE_HASH = "678b1e19c88ea5fdaf60b14abccdb09e"

# Same 344M R=3162 checkpoint, evaluated under each protocol.
RUN_0SHOT = "1yml0np5"
RUN_4SHOT = "9sa3gfmo"


def sweep_list_hash(sweep_ids) -> str:
    return hashlib.md5(("sweeps=" + ",".join(sweep_ids)).encode()).hexdigest()


def main() -> None:
    print("Cache filename hash observed in notebooks/11_*/data/:")
    print(f"  {OBSERVED_CACHE_HASH}")
    print(f"  hash(old 0-shot list) = {sweep_list_hash(OLD_0SHOT_SWEEPS)}")
    print(f"  hash(new 4-shot list) = {sweep_list_hash(NEW_4SHOT_SWEEPS)}")

    if sweep_list_hash(OLD_0SHOT_SWEEPS) == OBSERVED_CACHE_HASH:
        print("\n  => Cache was built from the OLD 0-shot sweep list.")
    elif sweep_list_hash(NEW_4SHOT_SWEEPS) == OBSERVED_CACHE_HASH:
        print("\n  => Cache was built from the NEW 4-shot sweep list.")
    else:
        print("\n  => Cache matches neither list as written; inspect the cache directly.")

    api = wandb.Api(timeout=600)
    for label, run_id in [("0-shot (39rugx2e)", RUN_0SHOT), ("4-shot (mprek7pj)", RUN_4SHOT)]:
        run = api.run(f"rylan/memorization-scoring-vs-sampling-eval/{run_id}")
        print(f"\n{'=' * 78}")
        print(f"{label}  run {run_id}  sweep {run.sweep.id if run.sweep else None}")
        print(f"  model: {run.config['model_config']['model']}")
        print(f"  temperature: {run.config['temperature']}")

        n = 0
        n_correct = 0
        examples = []
        for row in run.scan_history(
            keys=["response", "solution", "math_verify_score"]
        ):
            n += 1
            if row.get("math_verify_score"):
                n_correct += 1
            if len(examples) < 3:
                examples.append(row)
        print(f"  math_verify_score mean = {n_correct / max(n, 1):.4f} over {n} problems")

        for i, row in enumerate(examples):
            print(f"\n  --- example {i} (score={row.get('math_verify_score')}) ---")
            print(f"  GOLD     : {str(row.get('solution'))[:300]!r}")
            print(f"  RESPONSE : {str(row.get('response'))[:300]!r}")


if __name__ == "__main__":
    main()
