# `scripts/scratch/`

One-off investigation scripts. Nothing here is part of the pipeline, and nothing here should be
imported by `src/` or by a notebook.

**The bar for keeping a file here is that it is still the answer to a question someone will ask
again**, in one of two ways:

1. **It is the sole generator of a committed artifact.** Deleting it would leave a `.csv` or
   `.txt` in the repo that nobody can reproduce.
2. **A document cites it** as the method behind a claim — typically the verification write-ups in
   `reviews/2026_neurips/`.

Anything else is a superseded intermediate. Delete it; `git log` keeps it if it is ever wanted.

Twelve files were removed on 2026-07-30 under this rule: earlier passes of the lost-pretraining-data
hunt that `hunt_lost_pretraining_runs.py` superseded (it searches by exact run ID rather than by
configuration), plus assorted schema and cache inspections whose conclusions are recorded in the
documents they informed.

## What remains, and why

### Generators of committed artifacts

| Script | Produces |
|---|---|
| `build_overtrained_model_list.py` | `sweeps/eval_pt/math_overtrained/models_overtrained.txt` |
| `build_sft_model_list.py` | `sweeps/eval_pt/math_overtrained/models_sft_rerun.txt` |
| `build_table1_model_list.py` | `sweeps/eval_pt/math_overtrained/models_table1_rerun.txt` |
| `table1_actual_numbers.py` | `reviews/2026_neurips/data/table1_measured_4shot.csv` |
| `trace_table1_provenance.py` | `reviews/2026_neurips/data/table1_provenance_runs.csv` |

### Cited as the method behind a documented claim

| Script | Claim it supports |
|---|---|
| `audit_notebook_wandb_sources.py` | Regenerates `docs/NOTEBOOK_DATA_SOURCES.md` — which notebook serves which cache |
| `hunt_lost_pretraining_runs.py` | The pretraining runs are unrecoverable: 0 of 218 run IDs across 305 projects |
| `investigate_missing_pretraining_data.py` | Same investigation, configuration-level |
| `check_notebook11_cache_provenance.py` | Why Figure 1 is 0-shot despite the notebook declaring 4-shot sweeps |
| `check_fewshot_context_budget.py` | The 4-shot prefix is 635 tokens against a 2,048-token training length |
| `find_pretraining_sweeps.py`, `find_table1_runs_exhaustive.py` | Search coverage behind the "does not reproduce" findings |
| `check_ptv2_dates.py`, `compare_jkazdan_pretraining_configs.py` | Why `-pt-v2` is not a substitute for `-pt` |
| `verify_paraphrased_contaminant.py`, `verify_perturbed_contaminant.py`, `check_contaminant_dataset_overlap.py` | Token-level verification that the contaminant arms inject what they claim |
| `verify_temp_1_enumerate.py`, `verify_temp_2_fetch.py`, `verify_temp_3_score.py`, `verify_temp_4_analyze.py` | The four-stage independent re-derivation of the temperature table |
| `chain_eval_phases.sh` | Orchestration for the overnight eval phases |
