"""What trainer_config did the ORIGINAL pretraining runs actually record?

This settles whether the current sweep YAMLs could have produced the published runs. If the
recorded configs contain keys absent from the YAMLs (adam_beta1, warmup_ratio, ...), then wandb
merged them from the script defaults and the YAMLs are fine as-is. If they don't, the YAMLs in
this repo are not what was run.
"""
import ast
import pandas as pd

CACHE = ("notebooks/11_math_qwen3_pt_math_verify/data/"
         "c39ba9b590fe96b52183328d3d4c7323_runs_configs.csv")
YAML_KEYS = {
    "base_learning_rate", "data_seed", "dataloader_drop_last", "dataloader_num_workers",
    "dataloader_prefetch_factor", "eval_on_start", "eval_steps", "eval_strategy",
    "gradient_checkpointing", "hub_strategy", "logging_steps", "lr_scheduler_type",
    "max_grad_norm", "max_length", "max_steps", "num_train_epochs", "optim",
    "overtrain_multiplier", "per_device_eval_batch_size", "per_device_train_batch_size",
    "remove_unused_columns", "report_to", "save_strategy", "save_total_limit",
    "torch_compile", "warmup_steps", "weight_decay",
}

df = pd.read_csv(CACHE, low_memory=False)
row = None
for _, candidate in df.iterrows():
    try:
        cfg = ast.literal_eval(candidate["trainer_config"])
    except Exception:
        continue
    if isinstance(cfg, dict) and cfg:
        row, config = candidate, cfg
        break

print(f"Sample run: {row.get('run_name')}  state={row.get('State')}  sweep={row.get('Sweep')}")
print(f"\nRecorded trainer_config has {len(config)} keys:")
for key in sorted(config):
    marker = "" if key in YAML_KEYS else "   <-- NOT in the repo's sweep YAML"
    print(f"  {key} = {config[key]}{marker}")

extra = sorted(set(config) - YAML_KEYS)
missing = sorted(YAML_KEYS - set(config))
print(f"\nkeys recorded but absent from YAML: {extra}")
print(f"keys in YAML but not recorded:      {missing}")
