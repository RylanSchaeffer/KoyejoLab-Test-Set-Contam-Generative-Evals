# Handoff — NeurIPS 2026 rebuttal, submission 32216

Updated **2026-07-30, overnight session**. Scores: **8RFz 3** (Quality 2, Originality 2, conf 4) ·
**1wx9 4** · **aPBL 3**. The AC named 8RFz's loss-vs-correctness objection as the pivotal
critique. **Discussion closes 2026-08-03.**

## Read in this order

1. **[`REBUTTAL_DRAFT.md`](REBUTTAL_DRAFT.md)** — paste-ready per-reviewer responses. Start here.
2. **[`REBUTTAL_EVIDENCE.md`](REBUTTAL_EVIDENCE.md)** — each criticism mapped to its number.
3. **[`PROTOCOL_CONFOUND.md`](PROTOCOL_CONFOUND.md)** — read before quoting any Math Verify number.
4. **[`MISSING_PRETRAINING_DATA.md`](MISSING_PRETRAINING_DATA.md)** — the data-loss investigation.
5. **[`HF_TOKEN_INCIDENT.md`](HF_TOKEN_INCIDENT.md)** — credential problem found while launching.

Environment: `./mem_scoring_vs_sampling_env/bin/python` (absolute path).

---

## ⚠️ Two things needing a human

1. **`git push` is blocked** by the sandbox permission classifier. Everything is committed to
   branch `rebuttal/neurips-2026-protocol-and-evidence` (14 commits) but **nothing is pushed**.
   Run `git push -u origin rebuttal/neurips-2026-protocol-and-evidence` yourself.
2. **The HF token on this node is not yours.** `HF_HOME=/lfs/skampere1/0/shared_hf_cache` holds a
   world-readable (`-rw-rw-rw-`) token belonging to **`ruili0`** (Rui Li), write-scoped. Any
   `push_to_hub()` here lands in their namespace. Nothing of ours leaked — `ruili0` owns 0 `mem_*`
   models, `RylanSchaeffer` owns 196 — because the push is gated behind `PRETRAIN_SKIP_HUB_PUSH=1`.
   **Export your own `HF_TOKEN` before any upload**, and tell Rui/brando9 to rotate that token.

---

## What changed tonight, and why it matters

### The protocol question is settled — 0-shot, on the merits

The 4-shot switch was commit `db75c5f` (2026-03-29), self-initiated during the ICML rebuttal, on
the theory that 0-shot conflates format knowledge with reasoning because uncontaminated models
never see `\boxed{}`.

Testing that required removing **a second confound I introduced while investigating**: the 0-shot
and 4-shot sweeps sit on opposite sides of that same commit, which *also* tightened scoring
(lenient `math_verify.parse()`, ~1.4% false positives → boxed-required). Comparing logged scores
compared prompt *and* scorer. All 76 runs are now rescored from raw W&B responses with one scorer
(`scripts/rescore_zeroshot_with_boxed_required.py`, no GPU).

| Model | R=0 0-shot logged | R=0 0-shot **strict** | R=0 4-shot strict |
|---|---|---|---|
| 34M | 0.0038 | **0.0000** | 0.0000 |
| 62M | 0.0126 | **0.0000** | 0.0000 |
| 93M | 0.0074 | **0.0000** | 0.0000 |
| 153M | 0.0118 | **0.0000** | 0.0000 |
| 344M | 0.0000 | **0.0000** | 0.0000 |

The conclusion held and got stronger. 4-shot **does** teach the format (boxed rate 0 → 0.43–0.89)
and buys **exactly zero** accuracy. The premise is refuted on its own terms. And at 0-shot the
boxed rate rises with dose (153M: 0 → 0.009 → 0.047 → 0.72 → 0.98 → 1.0), so the contaminated
model learns the format from the injected solutions themselves.

Headline contrast under matched scoring: **153M R=316 → 0.9984 (0-shot) vs 0.0078 (4-shot)**.
Ratios run 3×–192×.

### The same confound was inside the Table 1 replacement

Notebooks 18 and 19 took their Original/pretrained baseline from the *lenient* CSV while their
treatment columns were strict. Both now read `protocol_sensitivity_rescored.csv`. Regenerated:

- **Table 1** (R ≥ 100, n=14): Original **72.18%** → Rephrased **2.78%** (96.1% removed) →
  Perturbed **1.91%** (97.4%).
- **SFT** (13 conditions ≥5% pre-SFT): **70.89% → 3.00%**, median retained 0.028 (0.001–0.302).

One claim died: the uncontaminated floor is now *exactly 0.00%*, so "rephrased/perturbed land at
2–3× the floor" is undefined — it divided by a floor that was pure artifact. Report the residual
in percentage points (+2.78 pp, +1.91 pp).

### The lost W&B data — re-investigated properly

Searched by **exact run ID** with a **validated matcher** (positive control 1 hit, fabricated ID 0
hits) rather than by configuration. **0 of 218 run IDs across 305 projects in 7 entities, 0
unreadable.** Identity confirmed correct (`rylan`, rylanschaeffer@gmail.com). The absence is
targeted: `-eval` (1,565 runs), `-sft` (135) and `-eval-teacher-forcing` (107) all resolve;
`-pt` alone does not. A rename or move is ruled out; who removed it is not established.

Good news: **more survives locally than documented.** The cache exists in notebooks 10, 11 *and*
20, and `notebooks/04_*/data/43bce56c...csv` is the sole copy of a 41-configuration
subset-fraction arm. All committed.

Still worth doing: check the W&B web UI for deleted projects (not exposed by the API) and email
W&B support — the runs are only ~6 months old.

---

## Experiments run tonight

| Job | Where | Status |
|---|---|---|
| Rescore of all 76 protocol runs | CPU | **done** |
| Paraphrased contamination, 34M, R=32/100/316 | sweep `mxamktp0`, GPUs 0/1/7 | see below |
| 0-shot pass@k, uncontaminated 344M | GPUs 6 (+0/1/7 as they free) | running |

**Paraphrased contamination.** Run from `scripts/pretrain_language_model_v1.py`, which reproduces
the published pre-`934546a` optimizer config, so the **published exact-replica runs are the
control and did not need retraining**. Verified `gradient_accumulation_steps == 9`, matching all
12 published 34M ot=1 runs. Train loss fell monotonically and ordered correctly by dose
(R=32 → 5.28, R=100 → 4.52, R=316 → 2.74 at last check), and benchmark loss is dropping from
11.938. Analysis is pre-written at `notebooks/21_paraphrased_contamination/`; run it once the
sweep finishes.

Control (published, 34M ot=1): R=0 **7.1437**, R=32 **2.5138**, R=100 **1.4526**, R=316 **0.5243**.

**0-shot pass@k.** The existing "0 correct in 5,000,000 samples" used the **4-shot** prefix and
cannot support a 0-shot capability claim — the rebuttal leans on this in three places. Added
`--num_fewshot {0,4}` and re-ran at 0-shot. First 5,900 samples: **0 containing `\boxed{}`**.
Sharded 4 × 1250 problems × 25 samples.

---

## Corrections the manuscript still needs

1. **Fig. 1 must be labelled 0-shot.**
2. The **"~60× SFT collapse"** in `REBUTTAL_PLAN.md` P0.1 is an artifact. Correct: 70.89% → 3.00%.
3. **Notebook 16's "14/17 conditions, up to −4.72 nats"** (`04_further_training.tex:64`) predates
   the token-weighting fix. Corrected: **17/17 conditions, max −2.18 nats.**
4. **Table 1's printed values (0.00–0.04%) do not reproduce.** Replace with the 0-shot re-run.
5. **11.64% of perturbed problems keep the original answer** — excluded; including them gives
   4.78% and inverts the ordering.
6. **"Collapses to baseline" overstates it** — residual is +1.9 to +2.8 pp over a 0.00% floor.
7. **SFT format confound is scale-dependent** — report `sft_score_given_boxed` alongside.
8. **Notebook 11's `Num. Tokens = 20 × Num. Parameters`** omits the overtrain multiplier.

## What is left

- **Post the rebuttal.** Draft is complete except the paraphrased and 0-shot-pass@k placeholders.
- **`\citep` the five new bib keys** — `palavalli2024taxonomy`, `mehrbakhsh2024confounders`,
  `dekoninck2024evading`, `dekoninck2024constat`, `godey2025gaperon`. They are in
  `references_rylan.bib` (verified against the ACL Anthology and arXiv) but **not yet cited**, and
  an uncited entry never appears in the bibliography. Note 8RFz wrongly listed **Jiang et al.
  2024 as uncited** — it is cited three times, including a full appendix paragraph; the draft
  corrects this politely.
- **Manuscript `.tex` edits** — deliberately not started; Rylan asked to hold.
- **P3.2** SFT hyperparameters appendix; **P3.3** rephrase/perturbation validation appendix.
