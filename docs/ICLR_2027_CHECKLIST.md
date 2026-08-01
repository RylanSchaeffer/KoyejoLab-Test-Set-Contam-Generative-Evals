# ICLR 2027 execution checklist

Drafted 2026-08-01. This is the **execution** document: the same work as
`docs/ICLR_2027_ROADMAP.md`, but ordered by the sequence I would actually run it in rather than by
score-moving power, and narrowed to the four areas Rylan named on 2026-08-01.

The roadmap remains the rationale document — it holds the reviewer-objection mapping and the
decision log of options already rejected. **Read the roadmap for *why*; use this file for *what
next*.** Where the two disagree, this file is newer.

**Status: D1, D2 and D3 all signed off by Rylan on 2026-08-01.** Execution order revised the same
day at Rylan's direction — see "Agreed order" below. Nothing has been started yet.

Cluster as of 2026-08-01 11:47: 7 of 8 A100-80GB free on skampere1 (GPU 7 held by another user's
sglang server, ~74 GB, up 17 h).

### ⛔ BLOCKER: HuggingFace identity is wrong right now

Verified 2026-08-01: `HfApi().whoami()` returns **`ruili0`**, not `RylanSchaeffer`. `HF_TOKEN` and
`HUGGING_FACE_HUB_TOKEN` are both unset, so the world-readable token file inside the shared
`HF_HOME` (`/lfs/skampere1/0/shared_hf_cache/token`, owned by `brando9`) wins. This is
`reviews/2026_neurips/HF_TOKEN_INCIDENT.md` recurring, not a new problem.

- **Evaluation is unaffected** — the project's checkpoints are public, so Phase 0 reads fine.
- **Training is blocked** — any Phase 1 run would push its checkpoint into `ruili0`'s namespace.
- **W&B is correct** (`rylan`); only HuggingFace is wrong.

Guards added 2026-08-01 so this fails loudly instead of silently: `assert_hf_identity()` in
`scripts/pretrain_language_model.py` refuses to start training under the wrong account, and that
script now honours `PRETRAIN_SKIP_HUB_PUSH=1` (which previously only v1 supported), so a run can
be trained locally and pushed later.

- [ ] **Rylan to export a `RylanSchaeffer` write token before Phase 1 launches.** Verify with
      `python scripts/scratch/check_hub_identity_and_access.py`.

---

## Agreed order (Rylan, 2026-08-01)

1. **Finish the MATH Qwen3 scale sweep** (Phase 1).
2. **Move to GSM8K** (Phase 3) — an easier benchmark, where we may actually see models capable of
   some generalization, which would let the paper make statements it currently cannot.
3. **Stop and evaluate** before committing to the coding benchmark (Phases 2 and 4).

Coding work is therefore **deferred behind a decision gate**, not cancelled. Phase 5 (Gemma 3) and
Phase 6 (eval-only) still fill gaps around the long Phase 1 runs.

**One addition I am proposing on top of this order — Phase 0 below.** The GSM8K step rests on the
premise that our models are "somewhat capable" there. That premise is cheap to test *today*, with
no training, by evaluating existing uncontaminated checkpoints on GSM8K. Doing it first means the
whole GSM8K campaign is either de-risked or redirected before the cluster is committed. It costs
hours and it gates real weeks of work, so I want it first even though it is not in the list above.

---

## Section 0 — Decisions (all signed off 2026-08-01)

My recommendation is given for each; all remain revisable if evidence changes.

### D1. Token budget: keep 14.3, don't rerun. ⭐ the important one

**Recommendation: keep 14.3 tokens/parameter, run all new pretraining with
`PRETRAIN_LEGACY_TOKEN_BUDGET=1`, fix the four prose claims, and buy back the caveat with a small
token-matched *control* rather than a full rerun.**

Rerunning everything is not affordable, and I want to be concrete about why rather than assert it.
The published corpus is 115 compute-optimal + 138 overtrained + 85 subset-fraction = **338
pretraining runs** (`docs/EXPERIMENT_INVENTORY.md`). The overtrained arm alone runs to multiplier
m = 16, so a single 344M m=16 run at the corrected budget is 110B tokens. Retraining the whole
corpus is on the order of 10^12 tokens. That is not an eight-week job on seven A100s; it is not a
one-year job on seven A100s.

The scientific case for keeping 14.3 is also stronger than it first looks. The shortfall is
**uniform** — 0.7136–0.7141 across every size and every multiplier, spread ±0.0005
(`docs/TOKEN_BUDGET_SHORTFALL.md`). A constant ratio is a legitimate experimental design, not a
defect: scaling-law fits require a *consistent* tokens-per-parameter ratio, not specifically the
Chinchilla one, and `src/analyze.py` fits on measured `num_input_tokens_seen` rather than the
nominal budget. What we lose is only the word "compute-optimal," which we replace with "a fixed
14.3 tokens per parameter, 0.71× Chinchilla-optimal" — a factual description that costs the paper
nothing.

This decision also **forces the new scale ladder to 14.3**, which matters more than the prose. If
we extend the ladder to 1.44B at 20 tokens/parameter while the existing points sit at 14.3, the
ladder acquires a kink in the (N, D) plane at exactly the new large sizes — the place we most want
to extrapolate. Mixed budgets would be worse than either uniform choice. As a bonus it is 29%
cheaper, which is load-bearing given the estimates in Phase 1.

The one claim that stays genuinely false under this option is the appendix's "total training tokens
remain constant across contamination levels" (+27% from R=0 to R=316). That is what the control
below is for.

- [x] **D1 signed off by Rylan 2026-08-01** — keep 14.3, no rerun, add the control.
- [ ] Fix the four affected prose claims (`02_methodology.tex:8`, `01_introduction.tex:28`,
      `04_further_training.tex:12,17`, `99_appendix.tex:74`) — the appendix one must be rewritten,
      not relabeled, since it is a claim about experimental validity.
- [ ] Run the **token-matched control**: 34M and 93M at R ∈ {0, 100, 316} with the fixed pipeline
      and total tokens genuinely held constant across R. Six runs, ~8B tokens total, hours not days.
      This converts "the dose–compute confound is bounded by the perturbed arm" from an argument
      into a measurement, and it is the honest way to keep 14.3 without hand-waving.

*If you'd rather rerun*: the only affordable version is the compute-optimal (m=1) grid at the five
published sizes, ~35 runs, leaving the overtrained and subset-fraction arms at 14.3 — which
reintroduces mixed budgets inside the paper. I don't recommend it.

### D2. Second architecture: Gemma 3 dense now; Gemma 4 only if a transformers upgrade proves safe

Investigated 2026-08-01, superseding the roadmap's Gemma 3 note (which predates Gemma 4):

| Candidate | Finding | Verdict |
|---|---|---|
| **Gemma 3 dense** | `Gemma3TextConfig` present in our installed transformers **4.56.1**. Google ships 270M and 1B, so small-scale training is proven. Dense, architecturally distinct from Qwen3. | **Recommended.** Zero infrastructure risk. |
| **Gemma 4** | Released 2026-04-02. Sizes E2B / E4B / 26B MoE / 31B dense, multimodal, Apache 2.0. **Not in transformers 4.56.1** — needs an upgrade of the pinned pipeline. The "effective parameter" naming on E2B/E4B implies a MatFormer-style elastic design, which is awkward to instantiate tiny and from scratch. | **Stretch.** Only on a branch, only if the upgrade doesn't disturb pretraining. |
| **Inkling-Small** | Verified from the model card: **276B total / 12B active**, 42 layers, 256 experts (6 routed + 2 shared per token), natively multimodal. The "Small" is relative to the 975B flagship. | **Rejected**, confirming the roadmap. There is no tiny dense config; nothing here is trainable from scratch at 34M–1.4B. |
| **MoE (Qwen3-MoE)** | `Qwen3MoeConfig` is in installed transformers, so a small MoE arm needs no upgrade. | **Yes, but as a standalone probe** (Phase 5), never as the robustness family — active-vs-total parameters break the tokens-per-parameter budget and the scaling-law fits. |

- [x] **D2 signed off by Rylan 2026-08-01** — Gemma 3 dense as the second family, on the strength of
      the config already being present in installed transformers. MoE stays a separate small probe.

### D3. Which coding benchmark — **recommendation: MBPP**, with a corrected rationale

*Revised 2026-08-01 after Rylan challenged two claims in the first draft. Both challenges were
correct; the original reasoning is retracted and recorded here so it does not get re-proposed.*

**Retraction 1 — "HumanEval is harder than MATH" is false.** For capable models the ordering runs
the other way: Qwen2.5-7B scores **57.9 on HumanEval versus 49.8 on MATH**. Benchmark difficulty is
not the obstacle and I should not have implied it was.

**Retraction 2 — "PoT-GSM8K" was a construction, not an existing benchmark.** Program-of-thought
prompting (Chen et al. 2022) and PAL (Gao et al. 2022) are real published *methods*, and
MathQA-Python (Austin et al. 2021, same paper as MBPP, 23,914 math-word-problems-to-Python) is a
real *dataset*. But a canonical "PoT-GSM8K benchmark" is not a standard artifact — I was proposing
to build one. **Dropped.**

**The actual obstacle is our corpus and token budget, not the benchmark.** Two verified facts:

- Our pretraining corpus is **`fineweb-edu-dedup` only** (`src/globals.py:57`). That is the
  educational-web-text subset of smollm-corpus, explicitly *not* the `python-edu` code subset. Our
  models see essentially no Python at all.
- **SmolLM2-135M scores 0.0% pass@1 on HumanEval** — and that model has code deliberately in its
  corpus and was trained on **2T tokens**. Our 344M models see 4.9B tokens; even 1.44B sees ~20.6B.
  That is 100–400× less data, without the code.

A model trained *with* code on 400× more tokens still scores zero. Ours will score zero on any code
benchmark, and this is a budget-and-corpus fact rather than a difficulty fact.

**The same reasoning applies to math, which is the part worth internalizing:** we already measure a
**0.00%** uncontaminated floor on MATH. There is no benchmark at this scale with a non-zero clean
floor, so *no choice of benchmark* delivers the "models of this size could possibly solve it"
property. Only changing the regime does — which is exactly what the roadmap's 2.1 capability axis
(continued pretraining of capable off-the-shelf base models) is for. If a real capability floor is
what you want, that is the item that provides it, not a benchmark swap.

**Recommendation: MBPP.** It is real, off-the-shelf, CC-BY-4.0, loadable via
`load_dataset("mbpp")` (974 problems; 427 in the hand-verified `sanitized` config), designed to be
"solvable by entry-level programmers," and every example ships a `test_list` of executable asserts —
precisely the input the Phase 2 harness needs. We use it as a **contamination substrate** against a
0% clean floor, which is the same footing as our MATH result and is perfectly sound: the finding is
that contamination lifts scores off the floor.

- [x] **D3 signed off by Rylan 2026-08-01** — MBPP as a contamination substrate, 0% clean floor
      stated plainly. Note the coding work is now gated behind the GSM8K results (see "Agreed
      order"), so these sub-items are not yet live.
- [ ] **D3 optional**: also add `python-edu` to the corpus for the code arm? Makes the arm less
      degenerate, but changes the pretraining setup mid-paper and on the SmolLM2 evidence still
      probably yields 0%. My inclination is **no**.
- [ ] **D3 optional**: MathQA-Python as a second code arm — real dataset, but I could not confirm a
      maintained public release, so this needs an availability check before it is planned around.

---

## Phase 0 — Test the GSM8K capability premise (hours, no training, do this first)

**Status 2026-08-01: COMPLETE. The floor is zero.** Full write-up in
`docs/PHASE0_GSM8K_CAPABILITY_FLOOR.md` (generated from the data, not hand-written); per-cell
numbers in `docs/data/phase0_gsm8k_4shot.csv`.

**1 credited response out of 38,688, across all 32 uncontaminated checkpoints — and that one was
inspected and is a truncation artifact.** The premise does not hold: GSM8K is easier than MATH and
it makes no difference at this scale.

The result is a *dissociation*, not just a null. The `####` rate reaches **59.7%** at 344M/ot=8,
so the larger overtrained checkpoints plainly learn the demonstrated format from four examples,
and still answer nothing correctly — the same pattern the manuscript reports on MATH, where 4-shot
lifts the boxed rate to 0.43-0.89 and buys exactly 0.0000. Format competence is present;
arithmetic competence is absent.

Seven cells needed rescoring downward, each a degenerate loop emitting a number that coincidentally
matched the gold. Two scorer bugs were found and fixed by reading the credited responses rather
than trusting the aggregate — see the write-up.

Purpose: find out whether our models have *any* non-zero clean capability on GSM8K before the
cluster is committed to Phase 3. Every checkpoint needed already exists.

The premise is plausible and worth testing rather than assuming. GSM8K is grade-school word
problems, which matches `fineweb-edu-dedup`'s educational-web content far better than MATH's
competition problems do, and the one comparison point I have found runs the right way:
**SmolLM2-135M scores 1.8% on GSM8K but 0.0% on HumanEval.** That is a real ordering — GSM8K is the
benchmark most likely to show a floor above zero. The counterweight is that SmolLM2-135M saw 2T
tokens against our 4.9B at 344M, so a 0.00% result here would not be surprising either.

### ⚠️ This must be few-shot, and the first attempt got it wrong

The first version ran **0-shot and measured nothing**, per Rylan's correction on 2026-08-01. Our
R=0 checkpoints are pretrained on fineweb-edu alone and have **never seen an answer marker of any
kind** — no `\boxed{}`, no `####`. A 0-shot prompt therefore asks them to invent a convention they
have never observed, which is not a capability test. The diagnostic was unambiguous: 0% `####`,
0% `\boxed{}`, 100% neither, with responses that paraphrase the question back, loop degenerately
("The answer is 2." × 40), or wander off-topic entirely.

The paper already contains the evidence that should have driven this design: at 4-shot on MATH the
boxed rate rises from 0 to 0.43-0.89 while accuracy stays at **exactly 0.0000**. Format and
capability were separated there, and format was not the blocker.

**Generalize the lesson:** any capability measurement on these checkpoints must either demonstrate
the answer format (few-shot) or use checkpoints trained on it. This does not change the 0-shot
protocol for *contaminated* checkpoints, where the prompt must match the memorized document's
opening — that remains correct and is a memorization measurement, not a capability one.

- [x] **0.1 Evaluate existing *uncontaminated* (R=0) checkpoints on GSM8K**, **4-shot**, greedy,
      across the full ladder. Eval-only, no training. Demonstrations come from the `openai/gsm8k`
      **train** split (`src.data.GSM8K_FEWSHOT_EXAMPLES`), so no evaluation item enters the prompt.
- [ ] **0.2 Include the most heavily overtrained checkpoints** (up to m=16). If clean capability
      exists anywhere in our checkpoint zoo it is most likely there, and those checkpoints are
      already trained and sitting on the Hub.
- [x] **0.3 Reported. Outcome: the "0.00% everywhere" branch.** GSM8K is another contamination
      substrate, exactly like MATH. Phase 3 stays worth running — it defuses "MATH-specific"
      completely — but **scope it as replication, not as a capability result**, and promise no
      generalization statements. **Item 3.5 is off the table**: it requires a clean floor above
      zero and there is none. A real capability floor has to come from the roadmap's 2.1 capability
      axis (continued pretraining of capable off-the-shelf base models), not from an easier
      benchmark.
- [x] **0.4 Numbers brought back before Phase 3 scope was fixed.**

<details><summary>Original decision criteria (kept for the record)</summary>

  Three outcomes, all useful:
  - **Non-zero clean capability** → the premise holds. Phase 3 can make generalization claims the
    MATH results cannot support, and this becomes a genuinely new contribution. Proceed at full
    scope, and reconsider whether GSM8K should outrank part of Phase 1.
  - **0.00% everywhere** → GSM8K is another contamination substrate, exactly like MATH. Still worth
    running (it defuses "MATH-specific" completely) but scope it as replication, not as a
    capability result, and do not promise generalization statements we cannot make.
  - **Non-zero only at the largest / most overtrained sizes** → the most interesting outcome: it
    locates a capability onset inside our own ladder, which feeds the roadmap's 2.1 transition
    study directly.

</details>

## D4 — NEW, needs sign-off: matching the published ladder needs more than the token flag

*Found 2026-08-01 while preparing Phase 1. This supersedes item 1.5's implication that
`PRETRAIN_LEGACY_TOKEN_BUDGET=1` is sufficient for comparability. It is necessary but not
sufficient.* Evidence is in the run-config cache, not in repo prose.

The published checkpoints were produced by the **pre-`934546a` (v1) script**, not the current one.
The sole surviving copy of their configs
(`notebooks/10_*/data/c39ba9b5..._runs_configs.csv`, 225 rows) shows `warmup_steps: 250`,
`weight_decay: 0`, `logging_steps: 1`, and **no** `adam_beta1`/`adam_beta2`/`warmup_ratio`/
`full_determinism` in any row. Commit `934546a` (2026-01-19) introduced all four. So the current
`scripts/pretrain_language_model.py` diverges from the published runs on **five** independent axes:

| Axis | Published (v1) | Current (v2) |
|---|---|---|
| Adam betas | 0.9 / 0.999 (HF default) | 0.9 / **0.95** |
| Warmup | **250 absolute steps** | `warmup_ratio: 0.2` |
| Weight decay | **0.0** | 0.01 |
| `full_determinism` | absent | `True` |
| Grad-accum rounding | `math.ceil` | `round` (from commit `2a83ebb`) |
| Token budget | 14.3 tok/param | 20 tok/param unless the legacy flag is set |

Two further traps found in the same pass:

- **`sweeps/pt/` no longer contains the published configs.** `934546a` rewrote those files in
  place (`logging_steps 1→10`, `eval_steps→5000`, workers/prefetch). Pristine pre-commit copies
  survive in **`sweeps/dose_response/pretrain/`**, which the commit never touched.
- **The published 344M came from `math_144gb_1xOT`** (batch 40, `eval_steps` 1000), not the
  `math_82gb_1xOT` path that `CLAUDE.md` and `README.md` advertise. The 82gb 344M config matches
  zero surviving runs. ("82gb"/"144gb" are per-GPU memory: skampere1's A100-80GB vs skampere2's
  H200-141GB.)

**Recommended recipe** (my recommendation; needs your sign-off):

1. Script: **`scripts/pretrain_language_model_v1.py`**.
2. Template: copy from **`sweeps/dose_response/pretrain/math_144gb_1xOT/`**, not `sweeps/pt/` and
   not `sweeps/pt_v2/`.
3. Add **`train_test_split_seed: values: [0]`** — mandatory. `src/data.py:367` reads it
   unguarded and the v1 YAMLs predate it, so every v1 sweep currently dies with a `KeyError`.
4. Env: `PRETRAIN_LEGACY_TOKEN_BUDGET=1`, plus a correct `HF_TOKEN` (see the blocker below).
5. W&B project: a **new** name (e.g. `...-pt-v1-scale-ladder`). The published project
   `memorization-scoring-vs-sampling-pt` no longer resolves, so writing to it would create an
   empty project; the join with published points has to happen in the notebook cache layer.

- [ ] **D4 signed off**

⚠️ Two residual caveats that cannot be engineered away, and should be stated in the paper:

- **Fixed 250-step warmup does not scale.** The published ladder used 250 *absolute* steps at
  every size, so warmup shrinks as a fraction of training as models grow. Keeping 250 is the
  comparable choice, but the ladder's warmup fraction is not scale-invariant — and already was
  not, across 34M→344M.
- **Effective batch is perturbed by `ceil` rounding.** `gradient_accumulation_steps =
  ceil(tokens_per_opt_step / (world_size × batch × 2048))`. The published ladder already varied
  world size (1, 1, 1, 2, 2) and batch (42, 36, 34, 34, 40), so any choice for the new sizes lands
  on a slightly different rounded effective batch. Mitigate by picking (world_size, batch) so
  `gradient_accumulation_steps_unrounded` has a small fractional part, and log it per run.

⚠️ **Also: two published architectures no longer exist in `src/models.py`.** `934546a` removed
`"62M"` and reshaped `"153M"` `(9, 320)` into `"165M"` `(9, 344)` — a *different model*, not a
rename. Reproducing or extending those two published points requires restoring the old entries.

## Phase 1 — Qwen3 scale ladder (the long pole; start as soon as Phase 0 is launched)

Everything else is CPU or short-GPU work that can proceed while these train. GPUs are free now, so
this starts the day it is signed off.

`src/models.py` already parameterizes every size we need — `499M (18, 704)`, `660M (21, 832)`,
`934M (25, 1010)`, `1.44B (31, 1260)`. No new architecture code. Note the real config names are
**660M and 1.44B**, not the roadmap's "600M and 1.4B."

- [x] **1.1 Throughput calibrated 2026-08-01** — `scripts/scratch/calibrate_scale_ladder_throughput.py`,
      measured on an idle A100-80GB. Results below; they change the plan.

### 1.1 results: measured, and more expensive than estimated

| Size | batch | peak reserved | headroom | tokens/s (1 GPU) | per run (4 GPUs) | doses | subtotal |
|---|---|---|---|---|---|---|---|
| 499M | 11 | 64.7 GB | 20.4 GB | 15.9k | 31.2 h | 5 | 156 h |
| 934M | 11 | 66.5 GB | 18.6 GB | 8.8k | 105 h | 4 | 420 h |
| 1.44B | 11 | 68.1 GB | 17.0 GB | 6.2k | 230 h | 3 | 691 h |

**Total: 1,267 four-GPU-hours = 5,068 GPU-hours ≈ 30 days of all seven GPUs, continuously.**
That is an optimistic lower bound: it excludes DDP all-reduce overhead. It also excludes the
`torch_compile: True` speedup (disabled during calibration), which may recover 20–40%, so call it
**three to four weeks of exclusive cluster time**.

**The load-bearing discovery: batch size is bounded by the vocabulary, not by model size.** At
Qwen3's 151,936-token vocabulary, the logits tensor is `batch × 2048 × 151936 × 4 B` ≈ 1.24 GB per
sequence, and cross-entropy materialises a second copy. That is why 1.44B fits at batch 11 while
499M **OOMs at batch 22** — the model is a minor term. All three sizes therefore land on the same
batch, and all three need `gradient_checkpointing: True` (the published runs set it `False`, which
was affordable on skampere2's 141 GB H200s but is not on an 80 GB A100). Checkpointing only
recomputes activations, so it does not affect comparability.

- [ ] **1.1a DECISION: drop 934M, keep 499M and 1.44B as the two extrapolation anchors.**
      *My recommendation, and exactly the fallback this document pre-specified under "Critical
      path".* 934M costs 420 h for a middle point, while 499M and 1.44B alone cost 847 four-GPU-hours
      (≈20 days on seven GPUs) and preserve the full extrapolation range, which is what the
      reviewers actually objected to. Dropping 1.44B instead would be cheaper still but caps the
      ladder at 934M and answers the scale objection much more weakly.
- [ ] **1.2 Confirm the dose grid against measured throughput.** At 14.3 tokens/parameter a single
      run costs ~7.1B tokens at 499M, ~9.4B at 660M, ~13.4B at 934M, ~20.6B at 1.44B, plus up to
      +27% at the highest dose. The full 4-sizes × 5-doses ladder is ~280B tokens, which on my
      rough arithmetic is **weeks of exclusive cluster time** — likely too much alongside Phases
      2–5. Provisional trim, to be revised once 1.1 lands:

  | Size | Doses R | Runs |
  |---|---|---|
  | 499M | 0, 1, 10, 100, 316 | 5 |
  | 934M | 0, 10, 100, 316 | 4 |
  | 1.44B | 0, 100, 316 | 3 |

- [ ] **1.3 Re-upload `RylanSchaeffer/math_rephrased`.** Currently unresolvable on the Hub; the
      guarded re-upload script exists (commit `2a97cbb`). This blocks 1.4, so do it early and
      cheaply. ⚠️ Set `HF_TOKEN` first — see `reviews/2026_neurips/HF_TOKEN_INCIDENT.md`.
- [ ] **1.4 Include the rephrased arm at every new size.** This is what makes the capability-boundary
      transition study (roadmap 2.1) free rather than a separate campaign: if paraphrase transfer
      switches on anywhere below 1.44B, these runs find it. Do not launch 1.2 without it.
- [ ] **1.5 Launch with `PRETRAIN_LEGACY_TOKEN_BUDGET=1`** (per D1) so the new points join the
      published ladder rather than forking it.
- [ ] **1.6 Evaluate all new checkpoints** 0-shot with required `\boxed{}` — never 4-shot, never
      mixed protocols in one comparison.

## Phase 2 — Code-execution eval harness (build while Phase 1 trains)

No GPU needed; this is the prerequisite for any coding benchmark, and it does not exist yet.

- [ ] **2.1 Sandboxed execution of model-generated code.** Subprocess isolation, wall-clock timeout
      per problem, memory cap, no network, scratch working directory, and a crash/timeout path that
      scores 0 rather than killing the run. We are executing untrusted model output, so the
      isolation is a correctness requirement, not a nicety.
- [ ] **2.2 Unit-test scoring path** mirroring `math_verify`'s interface so eval scripts and the W&B
      logging schema need minimal change, and per-problem results stay in run history (which is what
      makes bootstrap CIs and rescoring possible without a GPU).
- [ ] **2.3 Validate the harness against known-good solutions** — reference solutions must score
      100%, and deliberately broken ones 0%, before any model output is scored with it.
- [ ] **2.4 Decide pass@1 vs pass@k** and match whatever the greedy/sampling protocol does for MATH.

## Phase 3 — GSM8K replication (priority 2 of the agreed order)

Cheap in GPU terms (~1.3k test problems is a fraction of MATH's token footprint) and it defuses
"MATH-specific" completely. **Scope is set by the Phase 0 outcome** — if clean capability turns out
to be non-zero, add 3.5 below and this becomes more than a replication.

- [x] **3.1 Contamination injection already supports GSM8K** — verified 2026-08-01 with
      `scripts/scratch/verify_gsm8k_contaminant_matches_eval.py`, no code needed.
      `create_dataset_for_pretraining` routes `data_config["benchmark"]` through
      `create_dataset_for_supervised_finetuning`, which has had a `madrylab/gsm8k-platinum`
      branch all along. The verification also confirms the property the whole 0-shot
      memorization signal depends on: **all 50 checked injected documents start with exactly
      the 0-shot eval prompt.** If injection and evaluation disagreed by even a character, a
      contaminated model would be asked to continue text it never saw, would look clean, and
      the experiment would silently measure nothing. Only sweep configs remain.
- [ ] **3.2 Contamination mini-sweep at two model sizes.** Doses can go **higher** than the MATH
      grid, not merely match it. GSM8K's contaminant is 227,396 tokens per replica against MATH's
      1.5e6 — **0.15×** — so a given replica count costs a seventh as much corpus displacement.
      Two consequences worth exploiting: the dose ladder can extend past R=3162 cheaply, and the
      dose–compute confound that forced the perturbed-arm control on MATH (total tokens rising 27%
      from R=0 to R=316) is roughly seven times smaller here, so the token-matching is nearly free.
- [ ] **3.3 Reproduce the three qualitative signatures**: dose-response, loss below the
      uncontaminated asymptote, collapse under rephrasing.
- [ ] **3.4 Rephrased and perturbed arms** for GSM8K, so the ablation transfers too.
- [x] ~~**3.5 The generalization question.**~~ **CANCELLED by the Phase 0 result.** It required a
      clean floor above zero — comparing contaminated against clean on held-out GSM8K items — and
      the floor is zero. The "additional statements" this was meant to enable are not available on
      GSM8K at this scale, and would have to come from the roadmap's 2.1 capability axis instead.

## Decision gate — stop here and evaluate

Per the agreed order, do not start Phases 2 and 4 (the coding benchmark) until the Phase 3 results
are in and reviewed. Write up what GSM8K changed about the story first.

## Phase 4 — Coding benchmark replication (MBPP)

Blocked on Phase 2. Substantiates the Limitations-section claim that the framework applies to any
generative task, currently asserted without evidence.

- [ ] **4.1 Build the MBPP contaminant dataset** (exact / rephrased / perturbed), injecting problem
      text plus reference `code` as the "solution" analogue of MATH's worked solution.
- [ ] **4.2 Contamination sweep at two sizes**, doses matched to the MATH grid.
- [ ] **4.3 Report the clean floor as 0% and say so plainly.** Expected and fine: per D3 this is a
      corpus-and-budget consequence, not a failed experiment. The claim being tested is that
      contamination lifts a generative score off its floor in a *third* task modality, which does
      not require non-zero clean capability. Do not quietly omit the floor.
- [ ] **4.4 Verify contaminated models emit executable Python at all.** If contaminated models
      cannot reproduce even memorized code as valid syntax, the arm is uninformative and should be
      cut rather than reported — check this on the first checkpoint, before running the sweep.

## Phase 5 — Second architecture (Gemma 3 dense) and the MoE probe

- [ ] **5.1 Instantiate `Gemma3TextConfig` at sizes matched to the Qwen3 ladder.** ⚠️ Match on
      **non-embedding** parameters and state the accounting — Gemma's ~256k vocabulary makes tiny
      models embedding-dominated, and naive total-parameter matching would compare models with
      wildly different capacity.
- [ ] **5.2 Core contamination grid** at two or three sizes, R ∈ {0, 100, 316}.
- [ ] **5.3 Verify the qualitative findings replicate across families.**
- [ ] **5.4 (Stretch) Gemma 4 feasibility spike on a branch**: does upgrading transformers past
      4.56.1 break pretraining, SFT, or vLLM eval? Abandon quickly if it does — Gemma 3 already
      satisfies the reviewer objection.
- [ ] **5.5 (Stretch) Small Qwen3-MoE contamination probe** — 8 experts, two active-parameter sizes,
      exact vs rephrased. Tests the roadmap's sharp prediction that memorized solutions localize in
      specific experts and that paraphrasing changes the routing path. Standalone contribution, not
      part of the robustness argument.

## Phase 6 — Eval-only wins (no new training; fold in whenever the cluster is busy)

- [ ] **6.1 Discriminative vs generative head-to-head** on existing contaminated checkpoints
      (MCQ-ified MATH or MMLU-math alongside Math Verify). One figure, no training, and it turns the
      paper's central contrast into a measured result instead of a literature comparison.
- [ ] **6.2 Cross-domain transfer**: MATH-contaminated models evaluated on GSM8K and MMLU-math.
- [ ] **6.3 Perturbed positive control at R = 316** — the one missing ablation cell.
- [ ] **6.4 pass@k capability floors at every size**, not just 344M.
- [ ] **6.5 Coherence control for the temperature result.**
- [ ] **6.6 The 5,001-row footnote** (W&B pagination duplicate; cancels in ratios).

---

## Critical path

Phase 0 runs first and finishes in hours; Phase 1 launches as soon as it is underway rather than
waiting for it, since the two do not compete for the same resources for long. Phase 1 is the only
item with a multi-week floor, so everything else fills the gaps around it. Phase 3 follows Phase 1
on the cluster. Phase 5 (Gemma 3) queues behind Phase 1 — the main argument for trimming the dose
grid in 1.2 — and Phase 6 is eval-only and can land at any point. Phases 2 and 4 sit behind the
decision gate.

Three things would change this ordering, all cheap to learn early:

- **Phase 0 finds non-zero clean capability** → GSM8K gets more valuable than a replication, and
  part of it may deserve to outrank part of the Phase 1 ladder.
- **The 1.1 calibration shows the ladder is faster than estimated** → restore the full 5-dose grid
  at every size.
- **The 1.1 calibration shows it is slower** → drop 934M, keep 499M and 1.44B as the two
  extrapolation anchors.
