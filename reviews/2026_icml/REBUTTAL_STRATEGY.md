# Rebuttal Strategy — ICML 2026 Submission 2433

## Synthesized Reviewer Objections

- **Model scale too small (344M max) to support claims about modern LLMs.** A 344M model cannot do multi-step reasoning on competition math, so performance gains may be trivially attributable to verbatim memorization rather than revealing interesting memorization-generalization dynamics. [6RQA, Mmea, 4xWn, THKB]

- **Single benchmark (MATH only) limits generalizability.** Decoherence rates, scaling law parameters, and the three-regime framework may be artifacts of MATH's specific formatting or token distribution rather than general properties of generative contamination. [Mmea, 4xWn, THKB]

- **Memorization vs. generalization not adequately disentangled.** The paper conjectures about this trade-off but never quantifies it. Could split the test set into seen/unseen subsets, or evaluate on perturbed/rephrased problems. Called "the difference between a borderline and an excellent contribution." [6RQA, Mmea]

- **Finding #5 (SFT) misattributed as a memorization-generalization tension.** At 344M scale, the model cannot genuinely generalize on MATH, so the SFT performance drop is simply catastrophic forgetting of memorized content, not a tension between competing learning signals. [Mmea, 6RQA]

- **Finding #8 (three regimes / survival process) insufficiently justified.** The modelization is never formally derived or validated. Terms like "survival process," "decoherence," and "lock-in" need further clarification. Claiming to "mathematically discover" these rules is misleading. [4xWn, THKB]

- **Overstated or factually incorrect claims.** Specific instances: (a) "more rigorous measurement" (L.024) not justified; (b) "foundational work focused on discriminative benchmarks" ignores Kocygit et al. 2025 on machine translation; (c) positioning around "tens-to-thousands of tokens" contradicts MATH's short answers; (d) single-replica result depends on the pretraining data mix, which is never varied. [4xWn, 6RQA]

- **Insufficient originality relative to Bordt et al.** Work is described as "a replica of more complete work." Need to clearly articulate what is novel beyond prior contamination studies. [4xWn]

- **Overtraining results conflate data dilution with catastrophic forgetting.** It is unclear whether the mitigation effect of overtraining comes from diluting contaminated data or from forgetting memorized content, and the paper does not control for these separately. [6RQA]

- **External validity concerns for real-world settings.** Real-world models are much larger, pretrained on mixed online corpora; training/test data distributions are confounders that may compromise external validity of the controlled study. [THKB]

- **Missing experimental details.** Number of documents, max sequence length, number of optimization steps, and other specifics not reported. [4xWn]

- **Presentation reads as a listing of experiments** without a clear logical flow or narrative arc connecting sections. [4xWn]

## Rebuttal to Reviewer 4xWn

We thank the reviewer for engaging with our work and for recognizing the timeliness and relevance of the research questions (Strengths). We address each concern below.

### Soundness 1: Missing experimental details

We appreciate this suggestion. Key experimental details — including the number of documents, maximum sequence length, number of optimization steps, batch size schedule, and optimizer hyperparameters — are reported in Appendix B. We are happy to surface additional specifics in the main text if the reviewer indicates which details would be most useful. We will also add a concise summary table of training configurations to the main body in the revision.

### Soundness 2: Model scale (344M) is small

We agree that 344M is small relative to frontier models, and we acknowledge this limitation in the paper. However, we believe this concern should be weighed against two important considerations:

1. **Controlled contamination studies require pretraining from scratch.** This is the methodological price of causal identification: to isolate the effect of contamination, one must control the entire pretraining corpus, which is prohibitively expensive at billion-scale. This constraint applies equally to all prior controlled contamination work, including Bordt et al., which uses models of comparable scale.

2. **Our scaling law analysis (Finding #3) exists precisely to extrapolate beyond experimental scale.** By fitting $\mathcal{L}(C, R) = E(R) + C_0(R) \cdot C^{-\alpha(R)}$ across model sizes and contamination levels, we provide a principled framework for predicting contamination effects at larger compute budgets. This is not a post-hoc rationalization — it is a core contribution of the paper.

We will make the scale limitation more prominent in the revision, as the reviewer suggests.

### Soundness 3a: "More rigorous measurement" not justified

We accept this phrasing was imprecise. Controlled contamination studies enable *causal* measurement of contamination effects, in contrast to observational approaches that can only establish correlations. We will revise to "which enable the most direct causal measurement" in the camera-ready.

### Soundness 3b: Prior work claim ignores Kocygit et al. 2025

We respectfully disagree. We already cite Kocygit et al. (2025) in the introduction (line 24), in Section 4, and in Appendix A (Related Work). Our claim is that the literature has *focused on* discriminative benchmarks — a statement about the distribution of prior work, not a claim of exclusivity. To our knowledge, Kocygit et al. is the sole controlled contamination study on a purely generative task, and it studies machine translation via continued pretraining rather than pretraining from scratch. One exception out of dozens of papers does not contradict the observation that the field has overwhelmingly focused on discriminative evaluations — if anything, it underscores how underexplored the generative setting remains.

### Soundness 4a: Finding #8 modelization "never justified"; "mathematically discover" is misleading

The survival process model is not assumed — it is *derived* from the in-context scaling law (Equation 3). Specifically, we model the per-token negative log-likelihood as $\ell_t \sim E + A \cdot t^{-\alpha}$ and compute the probability that a model generates all $T$ tokens correctly as the product $P(T) = \prod_{t=1}^{T} p_t$, yielding Equation 4. The three regimes then emerge from the asymptotic behavior of this product depending on whether $E > 0$, $E \approx 0$ with $\alpha \leq 1$, or $E \approx 0$ with $\alpha > 1$. We will make this derivation more explicit in the revision.

We agree that "mathematically discover" overstates the nature of the contribution and will revise to "mathematically characterize."

### Soundness 4b: MATH answers are short, contradicting the "tens-to-thousands of tokens" framing

We respectfully disagree with this characterization. The MATH benchmark requires models to generate full chain-of-thought *solutions*, not just final answers. In our experiments, MATH solution lengths range from tens to hundreds of tokens, and Finding #7 (Figure 6) explicitly plots Math Verify scores as a function of solution length out to 100+ tokens. The "tens-to-thousands of tokens" framing in the introduction refers to generative evaluations broadly — MATH solutions fall squarely within this range. This is precisely why solution length emerges as a key moderator of contamination effects (Findings #7 and #8), a phenomenon that has no analog in discriminative evaluations.

### Presentation: Paper reads as a listing of experiments

The paper is organized around the language model lifecycle: pretraining (Section 3), post-training (Section 4), and inference (Section 5). Each section builds on the previous one — pretraining establishes the baseline contamination dynamics, post-training asks whether standard interventions mitigate them, and inference reveals the token-level mechanisms that govern whether memorized content survives into generated outputs. We note that three of four reviewers rated the presentation as excellent (4/4), and Reviewer THKB specifically highlighted that "the findings are coherent and progressive." That said, we will strengthen the transition paragraphs between sections to make this narrative arc more explicit.

### Originality: "Replica of Bordt et al."

We believe this characterization does not reflect the content of either paper. Bordt et al. studies contamination on **7 multiple-choice question-answering benchmarks** — purely discriminative evaluations where the model selects among a small number of provided answer choices. Our paper studies **generative evaluation**, where the model must produce complete solutions token by token. This distinction is not cosmetic; it is the entire motivation of our work (Section 1, lines 32–37).

The generative setting introduces dynamics that simply do not exist in discriminative evaluation: the role of sampling temperature (Finding #6), solution length as an exponential barrier to memorization (Finding #7), and the survival process framework with three distinct regimes (Finding #8). None of these have analogs in Bordt et al. Furthermore, Finding #3 — which the reviewer themselves identifies as original — directly contradicts conclusions from prior work including Bordt et al. It is difficult to characterize a paper as a "replica" of work whose conclusions it contradicts.

Our paper partially originated from private correspondence with the authors of Bordt et al. (2025) about how the field currently lacks a characterization of how contamination affects generative evaluations — as distinct from the discriminative setting they studied.
