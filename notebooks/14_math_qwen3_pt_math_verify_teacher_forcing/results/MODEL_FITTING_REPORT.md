# NLL vs Token Index Model Fitting Report

## Executive Summary

We tested 15 different functional forms for `NLL(t, R, N)` to model how per-token
negative log-likelihood varies with token position, contamination level, and model size.

**Key Finding:** Global R² can be misleading! A model can achieve high global R² by
capturing between-condition variance while poorly fitting within-condition decay shapes.

| Metric | Best Model | Value |
|--------|------------|-------|
| Global R² | log_linear_ext | 0.9516 |
| Mean Local R² | log_linear_ext | -16.95 |

**Recommendation:** Use `log_linear_ext` model.

---

## Data

| Property | Value |
|----------|-------|
| Total data points | 3,810 |
| Model sizes | 34M, 62M, 93M, 153M, 344M |
| Replica levels (filtered) | 1, 3, 10, 32, 100, 316 |
| Token index range | 0 to 800 |

**Note:** Excludes R=0 (uncontaminated baseline) and R>=1000 (NLL uptick due to
hard-to-memorize long sequences). See `NLL_UPTICK_ANALYSIS.md` for details.

---

## Models Tested

### Ranked by Global R²

| Model | R² | RMSE | Parameters |
|-------|-----|------|------------|
| log_linear_ext | 0.9516 | 0.274 | 9 |
| V1_full | 0.9366 | 0.313 | 13 |
| power_decay | 0.8846 | 0.422 | 7 |
| V4_floor_rate | 0.8689 | 0.450 | 6 |
| V3_log_linear | 0.8521 | 0.478 | 7 |
| mixture | 0.8049 | 0.549 | 6 |
| rational_decay | 0.8005 | 0.555 | 8 |
| stretched_exp | 0.7950 | 0.563 | 9 |
| V5_power_laws | 0.7935 | 0.565 | 9 |
| exp_baseline | 0.7862 | 0.575 | 7 |
| V2_separable | 0.7838 | 0.578 | 8 |
| exp_power | 0.7494 | 0.623 | 7 |
| neural_scaling | 0.7450 | 0.628 | 9 |
| double_exp | 0.7240 | 0.653 | 7 |
| logistic_decay | 0.7120 | 0.667 | 6 |

---

## Best Model: log_linear_ext

**Fitted Parameters (9):**
```
[ 1.14813754  0.0395824   0.21631321 -0.06303786 -0.23418234 -0.0197332
 -0.03815189  0.1295853  -0.13402372]
```

**Local R² Statistics:**
| Statistic | Value |
|-----------|-------|
| Mean | -16.95 |
| Std | 40.99 |
| Min | -189.83 |
| Max | 0.87 |

---

## Generated Files

- `MODEL_FITTING_REPORT.md` - This report
- `parameter_visualization.pdf/png` - Model predictions across parameter space
- `y=fit_params_x=num_replicas_hue=model_size.pdf/png` - Floor model parameters vs R
- `y=fit_params_x=model_size_hue=num_replicas.pdf/png` - Floor model parameters vs N
