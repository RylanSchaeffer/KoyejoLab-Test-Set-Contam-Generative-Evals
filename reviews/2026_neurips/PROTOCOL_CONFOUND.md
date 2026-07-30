# The Evaluation Protocol Is Confounded With the Findings

**Status: blocking. Read before writing any rebuttal text that quotes a Math Verify number.**

Verified 2026-07-27 on skampere1 directly against the W&B API and the cached notebook data.
Everything below is reproducible with the scripts named at the end.

---

## The finding in one line

The same checkpoint — `mem_Qwen3-344M_..._rep_3162_..._ot_1`, greedy decoding, identical
scoring code — scores **Math Verify 1.0000 under 0-shot prompting and 0.0052 under 4-shot
prompting**. The manuscript's Finding #1 figure is 0-shot; Table 1 and every SFT number are
4-shot. They are being read as if they were the same measurement.

## Why it happened

`notebooks/11_math_qwen3_pt_math_verify/` declares the 4-shot sweep IDs, but its cached data
was built from the **commented-out 0-shot list directly above them**, and `refresh = False`
means it was never re-downloaded. This is confirmed by reproducing the cache filename:

```
cache file present : 678b1e19c88ea5fdaf60b14abccdb09e_runs_histories.parquet
md5("sweeps=" + ",".join(OLD_0SHOT_SWEEPS)) = 678b1e19c88ea5fdaf60b14abccdb09e   <-- match
md5("sweeps=" + ",".join(NEW_4SHOT_SWEEPS)) = b45eefcb57a83c0df28d6d823a598b13
```

`src.analyze.download_wandb_project_runs_configs` only re-downloads when the hashed file is
absent or `refresh=True`, so editing the sweep list without deleting the cache or flipping
`refresh` silently keeps the old data. Every figure in `notebooks/11_*/results/` is 0-shot.

Meanwhile `notebooks/13_*` (SFT) and `notebooks/15_*` (rephrased/perturbed) genuinely do read
4-shot sweeps. So the manuscript pairs a 0-shot pretraining figure against 4-shot SFT and
4-shot rephrase/perturb results.

## What is actually true, per protocol

Greedy decoding, pretrained (`ot=1`) checkpoints, same boxed-required scoring in both columns:

| Model | Replicas | 0-shot | 4-shot |
|---|---|---|---|
| 34M | 316 | 0.0732 | 0.0060 |
| 62M | 316 | 0.7990 | 0.0060 |
| 93M | 316 | 0.9872 | — |
| 93M | 1000 | 0.9994 | — |
| 153M | 316 | 1.0000 | — |
| 153M | 1000 | 1.0000 | — |
| 344M | 100 | 0.9912 | 0.0074 |
| 344M | 1000 | 1.0000 | 0.0104 |
| 344M | 3162 | 1.0000 | 0.0052 |

Full grid: `notebooks/11_math_qwen3_pt_math_verify/results/PROTOCOL_SENSITIVITY.md`.

Under 4-shot, **no** configuration anywhere in the grid exceeds ~1.1% Math Verify — not at
3,162 replicas, not at 344M. The contamination effect the paper reports exists only at 0-shot.

## Why the switch was made, and why its premise does not hold

Established 2026-07-29 from git history, in answer to "why did we switch to few-shot at all?"

The switch is commit **`db75c5f`, 2026-03-29**, *"Switch to 4-shot evaluation with boxed-required
scoring"*, made during the ICML rebuttal window. **No reviewer requested it** — the ICML review
files contain no mention of shot count or prompt format. It was self-initiated, and the `TODO.md`
added in the same commit records the reasoning:

> 0-shot prompting (should be 4-shot): Our eval scripts use 0-shot, but the EleutherAI harness
> standard for minerva_math is 4-shot. Uncontaminated models never see the `\boxed{}` output
> format at inference time, making it impossible for them to score >0% regardless of math
> capability. Contaminated models get the format "for free" from training data. This conflates
> format knowledge with reasoning ability.

A second, independent fix rode along in the same commit: requiring `\boxed{}` before scoring,
because `math_verify.parse()` extracted bare numbers from free text at priority 300 (~1.4% false
positives). That fix is correct and is not in question here.

The same `TODO.md` exempts teacher forcing, with the argument that turns out to govern the
generative case too:

> Adding a 4-shot prefix would change the conditioning context to something the model never saw
> during training, diluting the memorization signal.

So the correct reasoning existed at the moment of the change and was applied to one eval path
only. The unexamined assumption was that the prefix acts as a *formatting aid* rather than as a
*change of conditioning context*.

Two premises behind the switch are measurably false:

**1. "~220 tokens — trivial relative to the 32K context window."** Measured
(`scripts/scratch/check_fewshot_context_budget.py`): the prefix is **635 tokens** and the median
4-shot prompt is **687 tokens**. 32,768 is `max_position_embeddings` in the config, but these
models were *pretrained* at `max_length=2048` (recorded in every run's `trainer_config`). The
prefix is therefore ~1/3 of the trained sequence length. No overflow, but not negligible.

**2. The baseline fairness it was designed to buy does not exist.** 4-shot was meant to lift the
uncontaminated baseline off zero. It moved it to *exact* zero at every model size:

| Model | R=0 @ 0-shot | R=0 @ 4-shot |
|---|---|---|
| 34M | 0.0038 | **0.0000** |
| 62M | 0.0126 | **0.0000** |
| 93M | 0.0074 | **0.0000** |
| 153M | 0.0118 | **0.0000** |
| 344M | — | **0.0000** |

Corroborated by the pass@k run: 5,000,000 samples from an uncontaminated 344M produced **0
correct and not one well-formed `\boxed{}`**. These models cannot use the demonstration.

**Consequence for the framing decision.** Format conflation cannot be what drives the 0-shot
contamination effect, because demonstrating the format leaves the uncontaminated baseline at hard
zero while collapsing the contaminated scores 96–192×. 0-shot is therefore the defensible
standard on the merits, not merely by convenience, and the 4-shot grid is reportable as a
result — brittleness of memorization — rather than as an erratum.

## Mechanism (not a bug)

The 4-shot prompt is the standard Minerva MATH format and is well-formed. Prompts are ~687
tokens at the median against a 2,048-token pretraining sequence length, so there is no context
overflow. Inspecting generations from the same checkpoint:

- **0-shot** — the prompt reproduces the opening of the memorized training document, and the
  model emits the stored solution verbatim, `\boxed{}` and all.
- **4-shot** — four unrelated worked examples precede the problem, the prompt no longer matches
  any memorized context, and the model produces fluent but unrelated text.

These models are tiny and trained from scratch. They have no general competence to fall back
on, so memorized regurgitation is all-or-nothing with respect to prompt format.

**This was already understood — for teacher forcing.** `scripts/eval_language_model_teacher_forcing.py:95`
carries an explicit note:

> `# NOTE: Teacher forcing intentionally stays 0-shot. We measure P(solution | prompt) where the`
> `# prompt matches what was injected during pretraining. Adding a 4-shot prefix would change the`
> `# conditioning context and dilute the memorization signal.`

That is exactly the effect measured above. The reasoning was applied to the loss-based evals and
not carried over to the generative ones.

## Which notebook is in which protocol

Good news: **the loss-based results are all 0-shot and mutually consistent.** Only two notebooks
are 4-shot, and they are the two whose conclusions are in question.

| Notebook | Measurement | Protocol |
|---|---|---|
| `10_*` pretraining cross-entropy | teacher-forced | 0-shot |
| `11_*` pretrained Math Verify | generative | 0-shot (via stale cache) |
| `12_*` SFT cross-entropy | teacher-forced | 0-shot |
| `14_*` pretrained teacher forcing | teacher-forced | 0-shot |
| `16_*` SFT teacher forcing, perturbed | teacher-forced | 0-shot |
| `13_*` **SFT Math Verify** | generative | **4-shot** |
| `15_*` **rephrased / perturbed Math Verify** | generative | **4-shot** |

So the fix is well scoped: re-run the notebook 13 and notebook 15 evaluations at 0-shot and the
whole manuscript becomes internally consistent. Checkpoint lists are prepared at
`sweeps/eval_pt/math_overtrained/models_table1_rerun.txt` (39, rephrase/perturb) and
`sweeps/eval_pt/math_overtrained/models_sft_rerun.txt` (39, SFT).

## Consequences for each finding

**Finding #1 (performance increases with contamination).** Holds, at 0-shot. The figure must
be labelled as 0-shot, or regenerated at 4-shot, in which case the effect largely disappears.

**Finding #2 / Table 1 (memorization, not generalization).** *Probably correct, but not
reproducible from this W&B account, and the notebook that appears to back it does not.*

Table 1 entered the repo in commit `06d0186` on **2026-01-22**. The rephrase/perturb sweeps
that `notebooks/15_*` reads were created **2026-03-31** (`25xeednq`) and **2026-04-01**
(`w8j3qnru`) — over two months later. **Table 1 is therefore not computed from those runs.**
Its values (0.00%, 0.02%, 0.04% = 0, 1, 2 correct out of 5,000) are exactly what 0-shot
evaluation on modified problems should give: a contaminated model regurgitates the memorized
*original* solution, which is the wrong answer once the problem is rephrased or perturbed. So
Table 1 is most likely a 0-shot measurement, consistent with Fig. 1 and with the paper's
narrative — it just lives outside this W&B project, plausibly in the TensorPool runs credited
in `07_acknowledgements.tex`.

What is *not* usable is `notebooks/15_*`. Measured from its own sweeps at 344M, greedy, 4-shot:

| Replicas | Original | Rephrased | Perturbed |
|---|---|---|---|
| 0 | 0.00% | 0.00% | 0.00% |
| 100 | 0.74% | 0.58% | 0.66% |
| 1000 | 1.04% | 1.10% | 0.78% |
| 3162 | 0.52% | 0.52% | 0.52% |

Original, Rephrased and Perturbed are indistinguishable at every contamination level, and none
of these numbers match the ones printed in Table 1. That notebook cannot support Finding #2 and
its figure should not be promoted into the paper.

**Action:** re-run rephrase/perturb at 0-shot. That simultaneously reproduces Table 1, gives it
citable provenance for 8RFz's Q4, and extends it to model sizes the current table asserts but
cannot show. Model list at `sweeps/eval_pt/math_overtrained/models_table1_rerun.txt`
(39 checkpoints x 2 datasets).

**Finding #5 / the "SFT collapses accuracy ~60x" claim** in `REBUTTAL_PLAN.md` P0.1. *Artifact
of the protocol mismatch.* It compares 0-shot pretrained (~100%) against 4-shot SFT (~1-2%).
Matched at 4-shot, pretrained is 0.40% mean and SFT is 0.20% mean — a factor of two, not sixty.
See `notebooks/13_*/results/FORMAT_SANITY_CHECK.md`. Do not put the 60x number in the rebuttal.

**Finding #4 (overtraining dilutes contamination).** The accuracy-space measurement is running
now at 0-shot so that it is comparable to the Finding #1 figure.

## Separately: Table 1's provenance

Scanning `rylan/*` found **no run at 34M or 93M against any rephrased or perturbed dataset**;
modified-MATH generative eval exists there only at 344M, and the 153M entries are teacher
forcing rather than Math Verify. Combined with the two-month date gap above, the conclusion is
not that Table 1 is wrong — it is that **Table 1's runs are not in this W&B account**. They
were most likely produced on TensorPool, per `07_acknowledgements.tex`, and reached the repo as
a paste in "Updates from Overleaf".

That is still a problem worth fixing, because 8RFz's Q4 is literally "How are the values in
Table 1 calculated?" and right now the honest answer is "from runs we cannot point at." The
0-shot re-run resolves provenance and protocol together.

## Reproduce

```bash
PY=./mem_scoring_vs_sampling_env/bin/python
$PY scripts/scratch/check_notebook11_cache_provenance.py   # md5 + side-by-side generations
$PY scripts/compare_zeroshot_vs_fewshot_protocol.py        # full grid, both protocols
$PY scripts/check_boxed_format_rate.py                     # matched-protocol SFT comparison
$PY scripts/scratch/find_table1_runs_exhaustive.py         # every entity, project, run state
```

## Recommended handling in the rebuttal

Do not let a reviewer find this. State the protocol explicitly wherever a Math Verify number
appears, report both protocols, and present the sensitivity as a result rather than an erratum:
*contamination-driven gains in small from-scratch models are memorization so brittle that four
in-context examples erase them.* That is a stronger and more honest version of Finding #2 than
the rephrase/perturb table currently makes, and it directly engages 8RFz's loss-vs-correctness
objection — loss stays low under both protocols while accuracy does not.
