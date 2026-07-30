# NeurIPS 2026 Rebuttal — Execution Plan (skampere1)

Written to be picked up by a Claude Code session **running on skampere1 with GPU access**.
Rationale and reviewer-by-reviewer mapping: [`REBUTTAL_PLAN.md`](REBUTTAL_PLAN.md).

Scores: **8RFz = 3** (conf 4, Quality 2, Originality 2) · **1wx9 = 4** (conf 4) · **aPBL = 3** (conf 3).
The AC named 8RFz's loss-vs-correctness objection as the pivotal critique.

---

## Preflight

```bash
cd /lfs/skampere1/0/rschaef/KoyejoLab-Scoring-vs-Sampling-Memorization
source mem_scoring_vs_sampling_env/bin/activate     # uv venv, NOT conda
git pull
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

Read [`docs/EXPERIMENT_INVENTORY.md`](../../docs/EXPERIMENT_INVENTORY.md) and
[`docs/INFRASTRUCTURE.md`](../../docs/INFRASTRUCTURE.md) before searching for whether an experiment
exists. **Do not trust prose in other `reviews/**/*.md`, `TODO.md`, or `*_STATUS*.md` files** — several
describe experiments as complete that were never folded into the manuscript.

**Cluster reality (measured 2026-07-27).** All three nodes are heavily contended; typically ~1 fully
free GPU each. skampere1 = 8× A100-80GB, skampere2 = 8× H200, skampere3 = 8× B200. Check before
launching, and prefer many short single-GPU eval jobs over anything that needs the whole node. On one
GPU, task 1 below is roughly an overnight run.

---

## Already done — do not redo

| | Status |
|---|---|
| Math Verify on **SFT** checkpoints | **Done.** Sweep `2zpwcnek`, 351 finished runs. Figures already rendered in `notebooks/13_*/results/` (6 figure pairs). Just needs folding into the manuscript. |
| Rephrased/perturbed generative eval, 344M | **Done.** Sweeps `mprek7pj` (original), `w8j3qnru` (perturbed), `25xeednq` (rephrased). |
| SFT → perturbed-MATH teacher forcing | **Done.** Sweep `onaspopu`; `notebooks/16_*` has 3 figure pairs. |
| pass@k, uncontaminated 344M | **Done** (0.000% / 808,797 samples) but artifacts not located — see task 2. |
| Math Verify on **overtrained** checkpoints | **NOT DONE. Zero runs.** This is task 1. |

---

## Task 1 — Math Verify on the 138 overtrained checkpoints  ⟵ start this first

The AC's pivotal critique. Finding #4 ("overtraining dilutes contamination") currently rests entirely on
cross-entropy; 8RFz correctly notes that loss on exact solution text is not correctness. The checkpoints
already exist on the Hub, so this is **inference only — no training**.

```bash
python scripts/audit_inventory.py --gaps-only     # exact list of the 138 targets
```

Then clone `sweeps/eval_pt/math/model=qwen3-*-1xOT.yaml` into `ot_{2,4,8,16}` variants, swapping the
model names. Cost: ~14 GPU-h at τ=0 alone; ~41 GPU-h across τ ∈ {0, 0.316, 1.0}.

**Do this optimization first — it pays for itself in under an hour.** `eval_language_model.py` loads
vLLM once per run and evaluates a single temperature; startup dominates at these model sizes (median run
is 3–6 min total). Looping temperatures inside one model load makes the 3-temperature sweep ~3× cheaper.

Both outcomes are publishable, so don't stall on the result:
- Accuracy tracks loss → Finding 4 stands, objection dies.
- Accuracy stays high while loss rises → that is the *stealth contamination* result 8RFz hypothesized,
  which is more alarming than the current claim. Given that SFT collapses accuracy (see task 4),
  decoupling is the less likely outcome, but write both paragraphs in advance.

## Task 2 — Locate the pass@k artifacts

Only item with real data-loss risk. `scripts/score_pass_at_k.py` writes `results.csv` / `summary.md`;
neither is on the workstation.

```bash
find /lfs/skampere1/0/rschaef /dfs/scratch0/rschaef -path '*pass_at_k*' \
     \( -name '*.csv' -o -name '*.md' \) 2>/dev/null
```

## Task 3 — Format sanity check (blocks task 4)

Confirm the flat ~1–2% post-SFT Math Verify floor is genuine task failure, **not** a collapsed `\boxed{}`
emission rate after SFT. Pull the `response` column for sweep `2zpwcnek` and compare parse rates against
the pretrained runs in `qx2c4702`. If the format rate dropped, that number is an artifact and task 4
changes character entirely. Apply the same check to whatever task 1 produces.

---

## Zero-GPU tasks — parallelizable

4. **Fold the SFT Math Verify result into the paper.** Post-SFT accuracy is flat ~1–2% across all
   contamination levels vs. ~100% at R ≥ 316 pretrained — a ~60× collapse [⚠️ SUPERSEDED 2026-07-30: artifact of a protocol mismatch; matched at 0-shot it is 72.95% → 2.80%] that also rules out the
   stealth-contamination corollary. Answers 8RFz Q1 (SFT half). Figures already exist.
5. **Put pass@k in the paper.** Zero-capability baseline; reframes "small models" from weakness to
   design feature.
6. **Bootstrap CIs on every Math Verify number.** Per-problem scores are in W&B history. State plainly
   this is not multi-seed; commit to seeds for camera-ready.
7. **Promote `notebooks/16_*` to a figure** (SFT → perturbed MATH, 14/17). Currently one prose clause at
   `04_further_training.tex:64`.
8. **Add an "Original" column to Table 1** (`03_pretraining.tex:14–67`) — but see the provenance risk below.
9. **Normalize Fig. 5 as `score(τ)/score(τ=0)`** to kill 8RFz's temperature confound. Restrict "truth
   serum" claims to τ ≤ 1.
10. **Bootstrap/profile CIs on E(0)** plus alternative functional forms. Make explicit that contaminated
    losses are *measured*; only the asymptote is extrapolated.
11. **Report contaminated-token fraction per replica count.** Pure arithmetic; answers "contrived."

## Writing only

12. **Fix the originality framing.** Delete "the first targeted examination of contamination in generative
    tasks" (`06_discussion.tex:4`). Add Palavalli 2024, Mehrbakhsh 2024, both Dekoninck 2024, Godey 2025
    (**none are in `references_rylan.bib`**). Add an explicit replicates/conflicts paragraph reconciled via
    "models trained from scratch have no competence to bridge surface form," with pass@k = 0 as direct
    evidence. Best shot at 8RFz's two low sub-scores.
13. **SFT hyperparameters appendix.** aPBL W4.
14. **Rephrase/perturbation validation appendix.** Mostly a port from `reviews/2026_icml/REVIEWER_6RQA/`.
15. **Clarify what Table 1 tested** (exact contamination + modified eval, *not* rephrased contamination).
16. **Lifecycle summary figure** — pretrain dose → overtrain → SFT → inference, in Math Verify space.
    Possible only after task 1.
17. **Concede scope** and commit to ≥1B + second family + code benchmark for camera-ready.
18. **Upside, not requested:** turn temperature response into a contamination detector. Report
    separability over data already collected. Most likely single item to move a score.

## Stretch — only promise if it will land

19. **Paraphrased contamination during pretraining.** Most-requested experiment (1wx9 Q1, aPBL Q1,
    AC bullet 1). 34M × R ∈ {32,100,316} ≈ 680M tokens each, ~10–25 GPU-h total, plus ~1 h wiring
    `load_dataset_math_rephrased()` into the dispatch at `src/data.py:442`. Given current contention,
    this competes directly with task 1 — task 1 wins.
20. **GSM8K contamination mini-sweep** at 34M/62M. Loaders exist; never run for contaminated pretraining.
21. **Seeds at pivotal configs.** Task 6 covers the reviewers' stated concern at zero cost.

---

## ⚠️ Table 1 provenance is a live risk

`notebooks/15_*` reads sweeps `mprek7pj`, `w8j3qnru`, `25xeednq` — **all 344M only**, matching the 9 + 9
perturbed/rephrased runs in W&B. But Table 1 in `03_pretraining.tex` reports **34M and 93M columns too**.
Those numbers do not come from any sweep currently in the eval project.

8RFz's Q4 is literally *"How are the values in Table 1 calculated?"* — so this must be resolved before
answering. Either trace the 34M/93M numbers to a real sweep (possibly a superseded 0-shot run or the old
`stellaathena/*` datasets), re-run those two model sizes on `RylanSchaeffer/math_{rephrased,perturbed}`
(cheap: 2 sizes × ~8 replica levels × τ=0 ≈ 16 runs), or drop the columns and report 344M only.
Re-running is the safest and costs under two GPU-hours.

## Findings from the skampere1 working-tree scan

Work present on skampere1 but not in git or on the workstation:

- **`notebooks/13_*/data/` + 6 figure pairs**, including `y=math-verify_x=compute_hue=replicas_col=temp`,
  which the workstation lacks. Richer than expected — check these before regenerating anything for task 4.
- **`notebooks/16_*/results/`** has a third figure pair not on the workstation,
  `y=mean_nll_x=num_replicas_hue=dataset_col=model_size`, plus two locally-modified PDFs.
- **`notebooks/12_*/data/`, `notebooks/15_*/data/`** — cached sweep pulls; save a W&B round-trip.
- **`notebooks/16_*/data/09ea...runs_configs.csv.skampere1-local`** — preserved during the July 27 pull
  because it differs (md5) from the committed copy. Reconcile and delete the loser.
- **Legacy, superseded, safe to ignore for the rebuttal:** `notebooks/00_gsm8k_platinum`,
  `01_minerva_math`, `02_minerva_math_pt_qwen3`, `04_pretraining_loss_scaling` — all pre-date the
  10-series that feeds the paper.
- **`notebooks/03_bordt2025howmuchcanweforget`** (Sep 2025) is a re-analysis of **Bordt et al.'s published
  discriminative data** (winogrande/piqa/hellaswag/mmlu at replicas 4/12/36/144), *not* our models. It is
  not a substitute for task 1 — but it is genuinely useful for task 12, because Bordt measured
  **accuracy** under overtraining while we measured **loss**, which is exactly 8RFz's complaint. Its
  `fig1b_y=accuracy_x=overtrain_hue=multiplier` gives the prior-work accuracy comparison for free.

None of the above is tracked. Decide what to commit; the `data/` caches are large and `.gitattributes`
routes `*.parquet`/`*.feather` through Git LFS, **whose budget is currently exhausted** — commit the
`.csv` variants only, or top up LFS.

## Repo hygiene

- [x] **Yegor Denisov-Blanch added to `manuscript_neurips_2026/00_main.tex`** (OpenReview lists 12
      authors; the tex had 11). Affiliation set to Stanford Computer Science — **confirm with him before
      camera-ready**.
- [ ] `memorization-scoring-vs-sampling-pt` does not exist on W&B but is referenced 16× in
      `notebooks/10_*` and `sweeps/pt/*.yaml`. Repoint at `-pt-v2` or document the rename.
- [ ] `scripts/*pass_at_k*.sh` lost their exec bits in the July 27 pull (`chmod +x`).
- [ ] Replace the `sshpass` plaintext-password aliases in `~/.bashrc` with SSH keys.
- [x] Claude Code installed on skampere1/2/3 via `scripts/setup_claude_node.sh`.
- [x] Duplicate `load_dataset_math_rephrased` in `src/data.py` removed.
