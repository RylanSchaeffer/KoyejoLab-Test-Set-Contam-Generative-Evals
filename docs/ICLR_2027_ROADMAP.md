# ICLR 2027 roadmap: what would supercharge this paper

> **For execution order, see `docs/ICLR_2027_CHECKLIST.md` (2026-08-01).** This file remains the
> rationale document — the reviewer-objection mapping and the decision log of rejected options.
> The checklist is newer and wins where the two disagree; in particular the checklist recommends
> **keeping 14.3 tokens/parameter rather than rerunning** (item 1.3 below), and its architecture
> survey supersedes the Gemma 3 note in 1.4, which predates Gemma 4's release.

Written 2026-07-30, at the close of the NeurIPS 2026 rebuttal (submission 32216, scores 3/4/3).
This is the "infinite time and compute" list, ordered by expected score-moving power at the next
venue, with the realistic window flagged: **ICLR 2027 abstracts are typically due late September
2026, roughly eight weeks after NeurIPS decisions.** Tiers 1 and 2 are what fit that window with
focus; Tiers 3 and 4 are for the camera-ready or the venue after.

The organizing logic: every reviewer and the AC raised the same two external-validity objections
(scale/single family, single benchmark), and the rebuttal's strongest new material (the retrieval-key
mechanism, the capability-boundary reconciliation) generated testable predictions we have not yet
tested. The resubmission should close the first and cash in the second.

---

## Tier 1: kill the universal objections

Scale and the single benchmark were raised independently by all three reviewers and the AC, and
will be raised again by any future reviewer; the token-matched rerun retires our one known
methods caveat.

### 1.1 Scale ladder to ≥1B parameters
Extend the contamination grid to Qwen3 600M and 1.4B (configs already exist in `src/models.py`;
the naming convention and sweep infrastructure need nothing new). At 20 tokens/parameter, 1.4B is a
28B-token run: days on 8×A100, not weeks. Priority doses: R ∈ {0, 1, 10, 100, 316}. This is the
single highest-value item on the list because it also powers item 2.1: if paraphrase transfer
switches on anywhere below 1.4B, we find it, and the paper's biggest liability becomes its
headline.

### 1.2 Second and third benchmarks
- **GSM8K**: nearly free. The test set is ~1.3k problems (a fraction of MATH's token footprint),
  automatically verifiable, same eval harness. A contamination mini-sweep at two model sizes
  showing the same qualitative signatures (dose-response, loss below the uncontaminated asymptote,
  collapse under rephrasing) defuses "MATH-specific" completely.
- **Code (HumanEval or MBPP, pass@1)**: a different task modality with a different answer
  structure. This substantiates the Limitations-section claim that the framework applies to any
  generative task, currently asserted without evidence.

### 1.3 Token-matched rerun with the fixed pipeline
The published runs trained on 14.3 tokens/parameter (not 20) and total tokens rise 27% with
contamination dose (`docs/TOKEN_BUDGET_SHORTFALL.md`; code fixed 2026-07-30). No conclusion
changed, but a resubmission is the natural moment to retire the caveat entirely: re-run the core
grid with the corrected budget so every run is genuinely compute-optimal and total tokens are
constant across R. This converts a defensive footnote into a non-issue and removes the
dose-compute confound by construction rather than by the perturbed-arm control argument.

### 1.4 Second model family: Gemma 3 (dense), from scratch
Instantiate `Gemma3TextConfig` at sizes matched to the Qwen3 ladder and re-run the core
contamination grid (two or three sizes, R ∈ {0, 100, 316}). Gemma 3 is the chosen family
(decided 2026-07-30): dense, so it preserves the compute-optimal framing and scaling-law fits;
proven trainable at small scale (Google ships 270M and 1B); mature HF support; architecturally
distinct from Qwen3 (interleaved local/global sliding-window attention, GeGLU); name-brand with
reviewers. **Caveat to handle up front:** Gemma's ~256k vocabulary makes tiny models
embedding-dominated, so match sizes on *non-embedding* parameters and state the accounting, or
anchor the comparison at 270M-equivalent and up.

### 1.5 Multi-seed with error bands (lower priority, per Rylan 2026-07-30)
2–3 seeds over {34M, 93M, 344M} × R ∈ {0, 1, 10, 32, 100}, concentrated on the R ≈ 10–100
transition where variance should matter most; shaded bands in every figure. Demoted below the
items above. (The rebuttal deliberately makes no camera-ready commitments; per Rylan 2026-07-30,
these items are forward planning, not promises.)

### Decisions already made (do not re-propose)
- **Second family choice is settled: Gemma 3 dense** (2026-07-30, after considering and rejecting
  alternatives). **Llama**: rejected, too old. **Inkling** (Thinking Machines, released
  2026-07-15): rejected; it is a 975B-total / 41B-active multimodal MoE with a 12B-active small
  variant, so there is no meaningful tiny dense config, and two-week-old architectures have
  immature from-scratch tooling. **DeepSeek-style / any MoE**: rejected for the robustness arm
  because active-vs-total parameter ambiguity breaks the tokens-per-parameter budget, the
  compute-optimal framing, and the E(0) scaling-law fits, and tiny-MoE routing instability would
  confound any observed difference. MoE contamination is instead a standalone direction (Tier 3).

---

## Tier 2: convert the liabilities into the headline science

These are the experiments the rebuttal's own arguments predict. They are what makes the
resubmission a *stronger paper* rather than a patched one.

### 2.1 The capability-boundary transition study ⭐
The rebuttal's reconciliation with Mehrbakhsh et al. and Dekoninck et al. makes one falsifiable
prediction: **paraphrase transfer of contamination switches on as a function of the model's
underlying capability.** Nobody has mapped that transition. Two complementary designs:

- **Scale axis** (from scratch): run the rephrased-contaminant arm (the existing
  `math_rephrased` injection pipeline) at every size on the extended ladder from 1.1. Below the
  capability threshold, transfer stays at the ~1.5% plateau; if and where it rises, that is the
  transition point.
- **Capability axis** (continued pretraining): take off-the-shelf capable base models at several
  sizes, continue-pretrain with paraphrased contamination, measure transfer to the original items
  as a function of the base model's uncontaminated MATH score. This directly bridges our regime and
  the prior literature's regime inside one experiment.

If this lands, the paper's framing changes from "contamination inflates generative evals" to
"we located the capability boundary at which contamination stops being retrievable-only and starts
generalizing", which reconciles the literature and is a genuinely new result. This is the most
interesting scientific opportunity in the review packet.

### 2.2 The full leakage-mode grid
The three-arm ablation (exact / rephrased / perturbed) becomes a leakage-mode matrix, all injected
during pretraining at matched token dose:
- **Problem-only** (no solutions): Jiang et al.'s text-only condition, in our controlled setup.
- **Translated replicas**: Yao et al.'s cross-lingual channel.
- **Embedded-in-discussion**: benchmark items wrapped in realistic blog-post/forum-style documents,
  the leakage mode the AC called dominant in practice.
- **Diverse-paraphrase**: k distinct paraphrases at R/k replicas each, testing whether surface-form
  diversity at fixed dose builds an abstraction that a single surface form does not. This is the
  cleanest probe of memorization-vs-generalization the setup allows.
- **Disjoint-mathematics arm**: contaminate with MATH-*train*-style items sharing no test items,
  separating domain adaptation from item-level leakage. Explicitly acknowledged as unrun in the
  rebuttal; cheap; closes the ablation's one stated interpretive gap.

### 2.3 Discriminative vs generative head-to-head on the same checkpoints
The paper's core framing is that generative evaluations respond to contamination differently from
discriminative ones, but we never measure both on the same models. Evaluate the existing
contaminated checkpoints under a discriminative protocol (MCQ-ified MATH, or MMLU-math) alongside
Math Verify. One figure, no new training, and it turns the title's central contrast into a
measured result instead of a literature comparison.

### 2.4 Cross-domain transfer
The design specified in the rebuttal's reply to aPBL Q4: MATH-contaminated models evaluated on
GSM8K, MMLU-math, and an arithmetic-heavy code benchmark. The capability result predicts zero
transfer at our scale; measuring it (especially jointly with the 1.1 ladder) tests whether
contamination ever buys cross-domain capability and at what scale.

---

## Tier 3: new standalone contributions

### 3.1 The inference-time contamination detection protocol
Appendix E teases it; make it real. Package the temperature/length stress test (contaminated
models collapse toward the uncontaminated floor as τ → 1; clean capability does not) as a
black-box detection procedure requiring no corpus access, and report ROC curves using the full
checkpoint zoo as labeled ground truth: >170 checkpoints with known contamination status is a
detector-validation resource nobody else has. Compare against perplexity-based baselines, whose
false-positive mode on non-verbatim leakage we have already demonstrated. Plausibly a paper on its
own; at minimum a section that gives practitioners something to *use*.

### 3.2 Mechanistic account of the retrieval key
"Memorization is of the solution text; retrieval is keyed on the problem text" is a behavioral
claim. Go one level down on the existing checkpoints: probe where the key lives (attention patterns
from problem tokens at retrieval time, activation-patching the problem text between original and
rephrased forms, memorization-localization methods). Even a modest result (e.g., retrieval is
mediated by early-layer exact-match features that paraphrase destroys) would give the paper's
central mechanism a mechanism.

### 3.3 Release the testbed
Gaperon released contaminated models at large scale; nobody has released a *controlled
contamination ladder*. Publish the checkpoint zoo (all sizes × doses × arms), the contaminant
datasets (fix `RylanSchaeffer/math_rephrased` on the Hub, currently unresolvable, and mind
`HF_TOKEN_INCIDENT.md` before any upload), and the eval harness as a community benchmark for
contamination detectors. This is the cheapest contribution multiplier on the list: the compute is
already spent, and it gives reviewers a reason to want the paper published.

### 3.4 Contamination in Mixture-of-Experts models (standalone follow-up, not a robustness arm)
Nobody has run controlled contamination in MoEs, and the retrieval-key mechanism makes sharp
architectural predictions: if routing on the problem text is the retrieval key, memorized
solutions should localize in specific experts, paraphrasing the problem should change the routing
path (explaining transfer failure architecturally), and expert-level activation analysis becomes a
contamination detector. A small MoE grid (e.g., 8-expert models at two active-parameter sizes,
exact vs rephrased arms) would be early to an obvious question. Kept out of Tier 1 deliberately:
see the decision log there for why MoE cannot serve as the robustness family.

### 3.5 Post-training interactions beyond SFT
The lifecycle story covers pretraining → overtraining → SFT → inference. The missing stage is
RL-based post-training (RLVR on MATH-train is the natural setting): does RL amplify retrieval of
memorized solutions (reward hacking via regurgitation) or overwrite it the way SFT does? Timely,
and the SFT result (72.95% → 2.80%) makes either outcome interesting.

---

## Tier 4: hygiene and hardening (cheap; fold into whichever revision ships first)

- **Perturbed positive control at R = 316**: the one missing cell in the ablation's positive-control
  coverage (exists only at R = 32, 100). One checkpoint, one eval.
- **Coherence control for the temperature result**: score sampled generations' NLL under the
  uncontaminated 344M model to show τ ≤ 1 text stays comparably coherent across contamination
  levels, closing the last door on 8RFz's W2.
- **Scaling-law robustness**: refit E(0) under alternative functional forms and profile-likelihood
  intervals, so the irreducible-error claim's "conservative lower bound" argument holds under every
  reasonable specification, not just the bootstrap.
- **pass@k capability floors at every size**, not just 344M, so the zero-capability claim is
  uniform across the ladder.
- **The 5,001-row footnote**: every eval has 5,001 rows (W&B pagination duplicate, cancels in
  ratios); one footnote retires it.
- **Manuscript corrections from `HANDOFF.md`** that were deliberately held during the rebuttal
  window, if any remain unapplied.

---

## Sequencing under the realistic window

If NeurIPS rejects in September, the eight-ish weeks to ICLR support roughly: Tier 1 items
1.1–1.4 in parallel on the cluster (1.5 only if capacity remains), 2.1's scale axis (free once
1.1 runs include the rephrased arm), 2.3 and 2.4 (eval-only), and Tier 4. That alone addresses every
weakness in the NeurIPS metareview with data and adds the transition study. 2.1's capability axis
and 2.2 are the stretch goals; Tier 3 items are parallel-track writing/analysis that costs little
GPU and can land late.

Decision point on scope: if 2.1 finds the transition, restructure the paper around it. If it does
not (transfer stays flat to 1.4B), that is still a publishable strengthening of the boundary
claim, and the framing stays as-is with the external-validity objections closed.
