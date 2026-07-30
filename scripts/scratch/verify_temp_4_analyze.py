"""Stage 3: build the advantage table from the independently rescored runs, several ways."""
import glob
import json
import os

import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
jobs = {j["run_id"]: j for j in json.load(open(os.path.join(SP, "jobs.json")))}

rows = []
for p in glob.glob(os.path.join(SP, "scored", "*.json")):
    d = json.load(open(p))
    j = jobs[d["run_id"]]
    rows.append({"run_id": d["run_id"], "Parameters": j["Parameters"], "R": j["R"],
                 "T": j["T"], "n": d["n"], "n_exc": d["n_exc"], "n_skipped": d["n_skipped"],
                 "n_boxed": d["n_boxed"],
                 "logged": d["n_logged"] / d["n"], "strict": d["n_strict"] / d["n"]})
mine = pd.DataFrame(rows).sort_values(["Parameters", "R", "T"]).reset_index(drop=True)
mine.to_csv(os.path.join(SP, "mine.csv"), index=False)
print(f"my runs: {len(mine)}   total exceptions: {mine.n_exc.sum()}   "
      f"blacklisted problems: {mine.n_skipped.sum()}   n distinct: {sorted(mine.n.unique())}")

# ---------- 1. cell-by-cell comparison against the published log-parsed CSV ----------
pub = pd.read_csv("notebooks/11_math_qwen3_pt_math_verify/results/temperature_response_rescored.csv")
m = mine.merge(pub, on=["Parameters", "R", "T"], how="outer", suffixes=("_mine", "_pub"),
               indicator=True)
print("\nmerge:", m._merge.value_counts().to_dict())
both = m[m._merge == "both"].copy()
both["d_strict"] = (both.strict_mine - both.strict_pub).abs()
both["d_logged"] = (both.logged_mine - both.logged_pub).abs()
print(f"max |d strict| = {both.d_strict.max():.5f}   max |d logged| = {both.d_logged.max():.5f}")
bad = both[(both.d_strict > 0.01) | (both.d_logged > 0.01)]
print(f"cells differing by > 0.01: {len(bad)}")
if len(bad):
    print(bad[["Parameters", "R", "T", "strict_mine", "strict_pub",
               "logged_mine", "logged_pub"]].to_string(index=False))
print("rows only in mine:")
print(m[m._merge == "left_only"][["Parameters", "R", "T", "strict_mine", "logged_mine"]]
      .to_string(index=False))


# ---------- 2. the advantage table, four ways ----------
def table(df, score_col, fallback_baseline: bool, agg: str):
    d = df.copy()
    ref = {}
    for size, g in d.groupby("Parameters"):
        avail = sorted(g.R.unique())
        ref[size] = 0 if 0 in avail else (avail[0] if fallback_baseline else None)
    base = d.set_index(["Parameters", "R", "T"])[score_col]
    d["baseline"] = [base.get((p, ref[p], t), np.nan) if ref[p] is not None else np.nan
                     for p, t in zip(d.Parameters, d["T"])]
    d["advantage"] = d[score_col] - d["baseline"]
    greedy = d[d["T"] == 0.0].set_index(["Parameters", "R"])[score_col]
    d["greedy"] = [greedy.get((p, r), np.nan) for p, r in zip(d.Parameters, d.R)]
    real = d[(d.R > 0) & (d.greedy >= 0.05)].copy()
    # exclude the size's own reference row when it is a fallback (advantage would be 0)
    real = real[[not (ref[p] is not None and ref[p] == r) for p, r in
                 zip(real.Parameters, real.R)]]
    ga = real[real["T"] == 0.0].set_index(["Parameters", "R"])["advantage"]
    real["greedy_advantage"] = [ga.get((p, r), np.nan) for p, r in zip(real.Parameters, real.R)]
    if agg == "mean_of_ratios":
        real["frac"] = real["advantage"] / real["greedy_advantage"]
        t = real.groupby("T").agg(advantage=("advantage", "mean"), frac=("frac", "mean"),
                                  n=("frac", "count")).reset_index()
    else:  # ratio_of_means
        t = real.dropna(subset=["advantage"]).groupby("T").agg(
            advantage=("advantage", "mean"), n=("advantage", "count")).reset_index()
        g0 = float(t.loc[t["T"] == 0.0, "advantage"].iloc[0])
        t["frac"] = t["advantage"] / g0
        t = t[["T", "advantage", "frac", "n"]]
    conds = sorted(set(zip(real.dropna(subset=["advantage"]).Parameters,
                           real.dropna(subset=["advantage"]).R)))
    return t.sort_values("T"), conds


print("\n" + "=" * 90)
for score_col in ["strict", "logged"]:
    for fb in [False, True]:
        for agg in ["mean_of_ratios", "ratio_of_means"]:
            t, conds = table(mine, score_col, fb, agg)
            tag = f"{score_col:6s} baseline={'R=0 or fallback' if fb else 'R=0 only':17s} {agg}"
            tau1 = t.loc[t["T"] == 1.0, "frac"]
            print(f"\n--- {tag}   |  tau=1.0 frac = "
                  f"{float(tau1.iloc[0]) if len(tau1) else float('nan'):.4f}  "
                  f"| {len(conds)} conditions: {conds}")
            print(t.round(4).to_string(index=False))

# ---------- 3. the 62M tau=1.0 premise ----------
print("\n" + "=" * 90)
s = mine[(mine.Parameters == "62M") & (mine["T"] == 1.0)].sort_values("R")
print(s[["R", "n", "logged", "strict", "n_boxed"]].to_string(index=False))
b_lg = float(s[s.R == 0].logged.iloc[0]); b_st = float(s[s.R == 0].strict.iloc[0])
r = s[s.R == 316]
print(f"62M tau=1.0 R=316: advantage lenient = {float(r.logged.iloc[0]) - b_lg:.4f}, "
      f"strict = {float(r.strict.iloc[0]) - b_st:.4f}")
