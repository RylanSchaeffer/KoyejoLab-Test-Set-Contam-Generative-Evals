# Integration Notes: Joshua's PR Changes

**Date:** 2026-01-18
**PR Reference:** https://github.com/RylanSchaeffer/KoyejoLab-Test-Set-Contam-Generative-Evals/pull/1
**Branch:** `mem_scaling` → `main`
**Author:** jlkazdan

## Summary

Joshua made changes to `notebooks/11_gen_eval_math_qwen3_pt_eval/` in the remote repo (note: remote repo name differs from local). This file documents which changes should/shouldn't be integrated into the current local notebook `11_math_qwen3_pt_math_verify.py`.

---

## Changes to Consider Integrating

### 1. Colorbar Visualization (Instead of Legends)

Joshua replaced discrete legends with continuous colorbars for the side-by-side plot. This is cleaner for continuous variables spanning orders of magnitude.

```python
from matplotlib.cm import ScalarMappable

# After creating the lineplot with legend=False:
sm_left = ScalarMappable(cmap="flare", norm=num_parameters_log_norm)
sm_left.set_array([])
cbarL = fig.colorbar(
    sm_left, ax=axes[0], label="Num. Parameters", fraction=0.05, pad=0.02
)
```

**Location in current file:** Lines 302-370 (the side-by-side subplot figure)

### 2. Per-Temperature Plot Loop

Joshua added a loop to generate individual plot files for each temperature value:

```python
for (temperature,), math_verify_by_temp_df in extended_eval_runs_histories_df.groupby(
    ["Temp."]
):
    plt.close()
    plt.figure(figsize=(10, 6))
    g = sns.lineplot(
        data=math_verify_by_temp_df,
        x="FLOP (6ND)",
        y="math_verify_score",
        hue="Num. MATH Test Set Replicas",
        hue_norm=matplotlib.colors.SymLogNorm(linthresh=1.0),
        palette="viridis",
        style="Temp.",
        legend="full",
    )
    g.set(
        xscale="log",
        ylabel="Math Verify Score",
        yscale="log",
        ylim=(None, 1.05),
    )
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    src.plot.save_plot_with_multiple_extensions(
        plot_dir=results_dir,
        plot_filename=f"y=math_verify_x=compute_hue=num_replicas_style=temp={temperature}",
    )
plt.show()
```

---

## Changes to AVOID (Bugs/Regressions)

### 1. Bug: `"Temp"` vs `"Temp."`

Joshua's code has a bug on line 104 where he references `"Temp"` instead of `"Temp."` (missing the period). This would cause a KeyError since the column is renamed to `"Temp."`.

### 2. Regex Change Breaks Lookup

- **Current (correct):** `r"Qwen3-([\d.]+[MB])"` → captures `"34M"`
- **Joshua's (broken):** `r"(Qwen3-[\d.]+[MB])"` → captures `"Qwen3-34M"`

The current version is correct because `MODEL_NAMES_TO_PARAMETERS_DICT` expects keys like `"34M"`, not `"Qwen3-34M"`.

### 3. Removed Pretraining Data Merge

Joshua removed the entire section that:
- Downloads `pretrain_run_configs_df`
- Merges eval results with pretraining cross-entropy data
- Creates the cross-entropy analysis plots (`y=math_verify_x=cross_entropy_hue=temp`)

This is a regression - keep the current merge logic.

### 4. Removed Temperature Rounding

Current version rounds temperatures for cleaner display:
```python
eval_runs_configs_df["Temp."] = np.round(eval_runs_configs_df["Temp."], decimals=2)
```

Joshua removed this.

---

## File to Ignore

### `analysis_of_results.py`

This is just a debugging/exploration script that:
- Downloads configs
- Prints `head()`
- Has no plots

It provides no value for integration.

---

## Next Steps

1. Decide if colorbar approach is preferred over the current legend filtering approach
2. If yes, update the side-by-side plot (lines 302-370) to use colorbars
3. Optionally add the per-temperature loop at the end of the file
4. Do NOT change the regex or remove the pretraining merge section
