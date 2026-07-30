# Handoff — NeurIPS 2026 rebuttal, submission 32216

Written 2026-07-29 on skampere1 for a fresh session. Scores: **8RFz 3** (Quality 2,
Originality 2, conf 4) · **1wx9 4** · **aPBL 3**. The AC named 8RFz's loss-vs-correctness
objection as the pivotal critique.

**Nothing is running.** No processes of mine on any GPU. Nothing is committed.

## Read in this order

1. **[`REBUTTAL_EVIDENCE.md`](REBUTTAL_EVIDENCE.md)** — every reviewer criticism mapped to the
   number that answers it, with phrasing traps flagged inline. Start here to write.
2. **[`PROTOCOL_CONFOUND.md`](PROTOCOL_CONFOUND.md)** — the finding that reframes everything.
   Read before quoting any Math Verify number.
3. **[`MISSING_PRETRAINING_DATA.md`](MISSING_PRETRAINING_DATA.md)** — the data-loss
   investigation, including a correction to something I got wrong.
4. [`SESSION_2026-07-27.md`](SESSION_2026-07-27.md) — earlier session log.

Environment: `./mem_scoring_vs_sampling_env/bin/python` (invoke by **absolute path**;
`source activate` fails silently if the interpreter symlink ever dangles again — see
`docs/INFRASTRUCTURE.md`).

---

## The headline finding

The evaluation protocol is confounded with the results. The same checkpoint — 344M, R=3162,
greedy, identical scoring code — scores **1.0000 at 0-shot and 0.0052 at 4-shot**.

Notebook 11's cache was built from the *commented-out* 0-shot sweep list, not the 4-shot list
written in the file; `refresh=False` kept serving it. Confirmed by reproducing the cache's md5
filename. So **Fig. 1 is 0-shot**, while **Table 1 and the SFT figures are 4-shot**.

Scope is bounded: notebooks 10/11/12/14/16 are 0-shot and mutually consistent; only **13 and 15**
are 4-shot, and they are exactly the two whose conclusions were in question. Everything I ran
this session is 0-shot, matching Fig. 1.

`eval_language_model_teacher_forcing.py:95` already documents this effect ("adding a 4-shot
prefix would change the conditioning context and dilute the memorization signal") — the reasoning
existed for teacher forcing and was never carried to the generative evals.

## Results produced this session (all 0-shot greedy, all complete)

| Study | Scale | Outcome |
|---|---|---|
| Finding #4, accuracy space | **137/137** overtrained checkpoints | Accuracy tracks loss. Dilution is **threshold-dependent**: 93M R=100 retains 0.019 over ot 1→16, R=1000 retains 0.995 |
| Finding #2, rephrase/perturb | **39 + 39** checkpoints, 5 sizes | Original 72.31% → Rephrased 2.78% (97.1% removed) → Perturbed 1.91% (98.3%) |
| Finding #5, SFT | **39/39** checkpoints | 72.31% → 3.00%, median retained 0.028 |
| Temperature control | existing runs, 10 temperatures | Advantage at matched τ retains **25%** at τ=1, 0.4% at τ=1.29 |
| pass@k | **5,000,000** samples | **0 correct.** Not one well-formed `\boxed{}` |
| Bootstrap CIs | 5,000 problems × 10,000 resamples | Median half-width 0.33 pp |
| Finding #3 robustness | refit + bootstrap | E(0)=3.5942 [3.5359, 3.6639]; 33/35 contaminated runs beat the lower bound |
| Contaminated token fraction | arithmetic | R=1 is 0.02–0.21% of budget; top of ladder 67–92% |
| Detector (unrequested upside) | 32 checkpoints | AUC 0.99 — **proof of concept only**, see caveat |

## Corrections to the manuscript that a reviewer would otherwise find

1. **Fig. 1 must be labelled 0-shot**, or regenerated at 4-shot (where the effect largely
   disappears).
2. **The "~60× SFT collapse" in `REBUTTAL_PLAN.md` P0.1 is an artifact** — it compares 0-shot
   pretrained against 4-shot SFT. Matched at 4-shot: 0.40% vs 0.20%. Correct figure is
   72.31% → 3.00% at matched 0-shot.
3. **Notebook 16's "14/17 conditions, up to −4.72 nats"** (`04_further_training.tex:64`) predates
   the token-weighting fix (`342deb5`, 2026-07-27) by two months. Corrected: **17/17 conditions,
   max −2.18 nats.** Stronger on consistency, weaker on magnitude.
4. **Table 1's printed values (0.00–0.04%) do not reproduce.** Replace with the 0-shot re-run.
   Rylan confirms Stella Biderman produced them on TensorPool, possibly with the outdated
   `stellaathena/*` datasets; the re-run uses `RylanSchaeffer/math_{rephrased,perturbed}`.
5. **11.64% of perturbed problems keep the original answer** — they score a memorizing model
   correct by construction. Excluded from the 1.91% figure; including them gives 4.78% and
   inverts the expected ordering.
6. **"Collapses to baseline" overstates it** — rephrased/perturbed land at 2.7–3.9× the
   uncontaminated floor, not at it.
7. **SFT format confound is scale-dependent.** 34M/62M keep `\boxed{}` at 79–92% (genuine
   failure); 153M/344M collapse to 2–7%, so their raw drop partly measures format loss. Report
   the `sft_score_given_boxed` column alongside.
8. **Notebook 11's eval-side `Num. Tokens = 20 × Num. Parameters`** omits the overtrain
   multiplier — wrong by up to 16× on a FLOP axis for overtrained checkpoints. Notebook 17 does
   it correctly.

## Open decisions for Rylan

1. **Protocol framing.** Standardize on 0-shot and present the sensitivity as a *result*
   (contamination-driven memorization is so brittle four in-context examples erase it), or treat
   it as an erratum? Recommendation: the former. Blocks manuscript edits.
2. **Commit.** Nothing is committed. Highest priority file:
   `notebooks/11_*/data/c39ba9b5..._runs_configs.csv` — 626 KB, untracked, and the **only
   surviving copy** of the pretraining cross-entropy behind Finding #3. LFS budget is irrelevant
   (it is CSV). Backed up to `~/irreplaceable_backups/` and `/dfs/scratch0/`.
3. **Whether to run the paraphrased-contamination experiment from here or from skampere2** — see
   below. This is the only reviewer-requested experiment still missing (1wx9 Q1, aPBL Q1, AC
   bullet 1).

## The paraphrased-contamination experiment: set up, not launched

Requested by two reviewers and the AC. All code is in place and **verified**, but I stopped
before launching because of an unresolved question.

**Done and verified:**
- `src/data.py` — `math_rephrased`/`math_perturbed` wired into the SFT/pretraining dispatch, and a
  new `data_config["contaminant"]` key that separates *what is injected* from *what loss is
  measured on*. Omitting it reproduces the exact-replica path unchanged.
- Verified by decoding tokenized examples back to text: benchmark split is 46/50 original with
  0 rephrased; training split carries rephrased with 0 original.
  (`scripts/scratch/verify_paraphrased_contaminant.py`)
- `scripts/pretrain_language_model.py` — checkpoint names now get a `_cont_<dataset>` suffix when
  the contaminant differs from the benchmark. **Without this, a paraphrased 34M R=316 run would
  have produced the identical name to the published exact-replica checkpoint and overwritten it
  on the Hub.** Verified collision-free and backward-compatible
  (`scripts/scratch/check_checkpoint_naming.py`).
- Sweep: `sweeps/pt/math_paraphrased_contaminant/model=qwen3-34M-1xOT-contaminant-controlled.yaml`
  — 6 runs, `contaminant ∈ {minerva_math, math_rephrased} × R ∈ {32, 100, 316}`. Both arms
  trained together so the contrast is internally controlled.

**Why it was not running — RESOLVED 2026-07-29.** This repo's `pretrain_language_model.py`
requires four config keys the published runs never recorded (`adam_beta1`, `adam_beta2`,
`warmup_ratio`, `full_determinism`; *not* `train_test_split_seed`, which this repo never
references), so this repo's sweep YAMLs KeyError immediately.

The cause is commit **`934546a` (2026-01-19 11:36), "Add v2 pretraining configs with improved
optimizer settings"** — a deliberate change that introduced all four keys and switched the W&B
project to `-pt-v2`. The notebook-11 cache was written 11 h earlier; the first `-pt-v2` run came
9 h later. **The published Fig. 3 runs are v1 runs from this repo**, produced by the pre-`934546a`
script.

Both earlier explanations are retracted. The claim that pretraining ran from
`KoyejoLab-Pretraining-Variance` is false — `wandb-metadata.json` shows `program` =
`.../KoyejoLab-Memorization-Scoring-vs-Sampling/scripts/pretrain_language_model.py` (this repo
under its former name, checked out on skampere2); only the *interpreter* was borrowed from that
sibling project. `docs/INFRASTRUCTURE.md` has been corrected.

**Nothing needs guessing and nothing needs skampere2.** `git show 934546a^:` recovers the exact
v1 script and defaults, and the full published `trainer_config` is recorded verbatim in the cache
CSV. Note the existing YAML sets both `warmup_ratio: 0.0316` and `warmup_steps: 250`, but the
current script reads `warmup_ratio` and has `warmup_steps` commented out — so as written it would
silently not match the published warmup schedule.

Cost once resolved: 6 runs × 34M at 1×OT (680M tokens), roughly 3–8 GPU-h each. GPUs 0, 1, 7 were
free; 2–5 belong to `jchud` (Jessica), and Rylan offered to ask her to share.

## Infrastructure notes

- **The venv broke once this session.** `bin/python` was a dangling symlink into a wiped AFS uv
  directory, and `source activate` fails *silently* in that state — every script dies with
  `ModuleNotFoundError`. Repointed to `/lfs/skampere1/0/rschaef/uv-python`. Diagnose with
  `ls -la mem_scoring_vs_sampling_env/bin/python`.
- **`wandb` CLI:** use `./mem_scoring_vs_sampling_env/bin/wandb`. The miniconda one on `PATH`
  fails to parse these sweep YAMLs. When launching agents, export
  `PATH="$REPO/mem_scoring_vs_sampling_env/bin:$PATH"` and `PYTHONPATH="$REPO"`, or `torchrun`
  resolves to miniconda's and the child dies with `No module named 'src'`.
- **SFT checkpoints live under `jkazdan/`**, not `RylanSchaeffer/` — transferred after the
  original evals, so W&B run configs record paths that now 404. Rylan confirms Joshua Kazdan
  produced them (~8 months ago). **The Hub is authoritative for what exists now; W&B records what
  existed then.**
- **Two GPU-job gotchas**, both fixed in `eval_language_model_multi_temperature.py`: vLLM's memory
  profiler asserts when a co-resident worker frees memory mid-init (retry with backoff), and
  404s were being retried four times with backoff (now fail fast).
- **Do not edit a running bash script.** Bash reads by byte offset; inserting lines makes a
  running instance resume mid-token. Kill and relaunch.
- **`src/globals.py`** gained `62M`/`153M` (and `Qwen3-` variants) to
  `MODEL_NAMES_TO_PARAMETERS_DICT` — they were missing, silently dropping those sizes from plots.
- **`src/plot.py`** — `format_g_legend_to_millions_and_billions` had two bugs: Axes-only API
  despite documenting FacetGrid support, and an unbound `new_label` for sub-1e6 values that would
  reuse the previous label. Both fixed.
- **`src/analyze.py`** gained group-addressed config/history downloaders (the new runs are
  launched directly, so they have no sweep ID) with a `cols_to_keep` option that switches to
  `scan_history`. That last one matters: these runs log ~2,000 `log_prob_token_*` columns per row,
  and it cut a history pull from 20+ minutes to under a minute.

## What is genuinely left

Writing only, except where noted:

- **P3.1 related-work rewrite / originality reframing** — the highest-value remaining task and
  8RFz's two low sub-scores. Delete "the first targeted examination…" (`06_discussion.tex:4`).
  Add Palavalli 2024, Mehrbakhsh 2024, both Dekoninck 2024, Godey 2025 (**none in
  `references_rylan.bib`**). Use pass@k = 0 as the mechanism reconciling the conflict with
  Mehrbakhsh/Dekoninck: prior work injected contamination into already-capable models; these
  models provably have zero capability to bridge surface form.
- **P3.4** — clarify Table 1 tested *exact contamination + modified eval*, not rephrased
  contamination in training. 1wx9's W2 misreads the direction, and their own hypothesis ("maybe
  scale is not enough") is exactly what pass@k = 0 confirms. Cheap win with your advocate.
- **P3.2** SFT hyperparameters appendix (aPBL W4) — data in `src/globals.py`, `sweeps/sft/`.
- **P3.3** rephrase/perturbation validation appendix (aPBL Q2) — port from
  `reviews/2026_icml/REVIEWER_6RQA/`, and fold in the 11.64% answer-overlap finding, which is
  exactly the validation gap they asked about.
- **P3.5** lifecycle figure — now possible; all four stages exist in Math Verify space at 0-shot.
- **Paraphrased contamination** — the one experiment left, blocked on the skampere2 question.
