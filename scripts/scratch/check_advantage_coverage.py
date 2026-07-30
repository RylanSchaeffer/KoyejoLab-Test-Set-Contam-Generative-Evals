"""Check which model sizes actually contribute to the contamination-advantage table.

The advantage is score(R) - score(R=0) at matched temperature. If a model size has no R=0 run,
its rows become NaN and drop out of the mean silently — which would mean the headline number
excludes that size without saying so.
"""
import pandas as pd

raw = pd.read_csv("notebooks/11_math_qwen3_pt_math_verify/results/temperature_response_raw.csv")
print("Replica levels present per model size:")
for size, group in raw.groupby("Parameters"):
    print(f"  {size:>5}: {sorted(group['Num. Replicas'].unique())}")

has_baseline = raw[raw["Num. Replicas"] == 0]["Parameters"].unique()
print(f"\nSizes WITH an R=0 baseline: {sorted(has_baseline)}")
missing = sorted(set(raw["Parameters"]) - set(has_baseline))
print(f"Sizes WITHOUT an R=0 baseline (silently dropped): {missing}")

adv = pd.read_csv("reviews/2026_neurips/data/temperature_contamination_advantage.csv")
strong = adv[adv["greedy_score"] >= 0.05]
contributing = strong.dropna(subset=["advantage"])
print(f"\nConditions with greedy>=5%: {len(strong)}")
print(f"  of which contribute to the mean: {len(contributing)}")
print("\nContributing conditions by size:")
print(contributing.groupby("Parameters")["Num. Replicas"].apply(lambda s: sorted(set(s))).to_string())
