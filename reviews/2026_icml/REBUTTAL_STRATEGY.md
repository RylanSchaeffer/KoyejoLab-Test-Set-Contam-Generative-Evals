# Rebuttal Strategy — ICML 2026 Submission 2433

## Synthesized Reviewer Objections

- **Model scale too small (344M max).** Performance gains may be trivially attributable to verbatim memorization rather than revealing memorization-generalization dynamics. [Mmea, 4xWn, THKB; 6RQA acknowledges but says "I do not see this as a major weakness"]

- **Single benchmark (MATH only).** Findings may be artifacts of MATH's structure rather than general properties of generative contamination. [Mmea, THKB]

- **Memorization vs. generalization not disentangled.** The paper conjectures but never quantifies. Called "the difference between a borderline and an excellent contribution." [6RQA, Mmea]

- **Finding #5 (SFT) misattributed.** Mmea: performance drop is catastrophic forgetting, not a memorization-generalization tension. 6RQA: conjecture could be tested rather than assumed. [Mmea, 6RQA — different critiques]

- **Finding #8 (survival process) needs justification or clarity.** 4xWn: model is "never justified" and "mathematically discover" is misleading (note: paper says "mathematically describe" — misquote). THKB: exposition is "compacted." [4xWn — validity; THKB — clarity]

- **Overstated claims.** (a) "most rigorous measurement" not justified; (b) "focused on discriminative benchmarks" ignores Kocyigit et al. 2025; (c) "tens-to-thousands of tokens" contradicts MATH's short answers; (d) single-replica result is corpus-dependent. [4xWn, 6RQA]

- **Insufficient originality relative to Bordt et al.** Called "a replica of more complete work." [4xWn]

- **Overtraining conflates dilution with forgetting.** [6RQA]

- **External validity concerns.** [THKB]

- **Missing experimental details.** [4xWn]

- **Presentation reads as a listing of experiments.** [4xWn]

## Rebuttal to Reviewer 4xWn

We thank the reviewer for recognizing the timeliness and relevance of the research questions.

### Soundness 1: Missing experimental details

These details — number of documents, max sequence length, optimization steps, batch size schedule, optimizer hyperparameters — are reported in Appendix B Pretraining Implementation Details. We will add a summary table to the main body in revision.

### Soundness 2: Model scale (344M) is small

We acknowledge this limitation. Two considerations:

1. **Controlled contamination requires pretraining from scratch.** This constraint applies to all prior work, including Bordt et al. (models up to ~1.6B — larger than ours, but still well below frontier scale).

2. **Our scaling law analysis (Finding #3) extrapolates beyond experimental scale.** Fitting $\mathcal{L}(C, R) = E(R) + C_0(R) \cdot C^{-\alpha(R)}$ provides falsifiable predictions at larger compute budgets. This is a core contribution, not a post-hoc rationalization.

We will make the scale limitation more prominent in revision.

### Soundness 3a: "More rigorous measurement" not justified

The reviewer quotes "a more rigorous measurement"; our paper says "the **most** rigorous measurement." Regardless, we will revise to "which enable the most direct causal measurement."

### Soundness 3b: Prior work claim ignores Kocyigit et al. 2025

We already cite Kocyigit et al. (2025) in the introduction (paragraph 2) and Appendix A Related Work. Our claim is that the literature has "predominantly focused on" discriminative benchmarks (the paper's exact words) — not exclusively. Kocyigit et al. is the sole controlled contamination study on a purely generative task. One exception out of dozens of papers reinforces, rather than contradicts, how underexplored the generative setting remains.

### Soundness 4a: Finding #8 "never justified"; "mathematically discover" is misleading

The survival process model is *derived* from the in-context scaling law (Equation 3): per-token NLL $\ell_t \sim E + A \cdot t^{-\alpha}$, survival probability $P(T) = \prod_{t=1}^{T} p_t$ (Equation 4), three regimes from the asymptotic behavior depending on $E$ and $\alpha$. We will make this derivation more explicit.

Regarding "mathematically discover": our paper says **"mathematically describe"** (Section 1, contribution bullet 3). The reviewer appears to have misquoted the text.

### Soundness 4b: MATH answers are short

The reviewer conflates final answers with full solutions. MATH requires generating complete chain-of-thought solutions; ground-truth solution lengths range from 15 to 1,949 tokens. The "tens-to-thousands of tokens" framing is empirically correct.

### Presentation: listing of experiments

The paper follows the model lifecycle: pretraining (Section 3) → post-training (Section 4) → inference (Section 5), each building on the previous. Three of four reviewers rated presentation 4/4; THKB called the findings "coherent and progressive." We will strengthen transition paragraphs in revision.

### Originality: "Replica of Bordt et al."

Bordt et al. studies **7 MCQA benchmarks** (discriminative). We study **generative evaluation**. This is not cosmetic — it is the entire motivation (Section 1, paragraphs 3–4).

The generative setting introduces dynamics absent from discriminative evaluation: temperature (Finding #6), solution length as an exponential barrier (Finding #7), the survival process framework (Finding #8). None have analogs in Bordt et al. Finding #3 — which the reviewer themselves identifies as original — contradicts conclusions from prior discriminative work.

Our paper partially originated from correspondence with the Bordt et al. authors about how the field lacks a characterization of contamination in generative evaluations.

## Rebuttal to Reviewer Mmea

We thank the reviewer for their substantive engagement and recognition of our experimental design, three-regime framework, and evaluation bug fix.

### Framing: What is the paper's contribution?

The reviewer reads our paper as studying the boundary between reasoning and memorization. That is not the core contribution. We characterize **the mechanics of how contamination inflates generative evaluation metrics**: temperature sensitivity (Finding #6), solution-length decay (Finding #7), the survival process (Finding #8), overtraining mitigation (Finding #4), and the SFT interaction (Finding #5). These are about *how memorization behaves during sequential generation* — dynamics absent from discriminative evaluations.

Under this framing, 344M is a *feature*: because these models cannot genuinely solve competition math, the contamination signal is cleanly isolated. At larger scales, contamination and genuine reasoning would be confounded.

### Weakness 1: Model scale (344M) is too small

We agree 344M cannot do multi-step competition math reasoning. See framing above.

Two responses:

1. **Controlled contamination requires pretraining from scratch.** Bordt et al. (2025) uses models up to ~1.6B for the same reason — still well below frontier scale.

2. **Scaling laws (Finding #3) bridge to larger scales.** Parameters vary smoothly across model sizes; average fitting error $< 10^{-2}$ for all $R$. These make falsifiable predictions beyond our experimental range.

Even if gains are "trivially" memorization, the *dynamics* are not trivial: temperature > 0.6 disrupts it, solution length creates an exponential barrier, one replica beats irreducible error, overtraining washes it out predictably. None obvious a priori.

**New experiment (in progress):** pass@k (k = 1000–10000) on uncontaminated 344M at temperature 1.0, stratified by MATH difficulty level. Any signal on Level 1–2 refutes "fundamentally lacks capacity." Zero signal reinforces clean-separation argument.

### Weakness 2: SFT finding is catastrophic forgetting

We partially agree — at 344M, the mechanism is likely catastrophic forgetting rather than a genuine tension.

The important finding is the *asymmetry*: SFT improves performance at low contamination ($R < 10$) and degrades it at high contamination ($R > 10$). This is non-obvious and practically informative regardless of mechanism. We will revise the language accordingly.

**New experiment (in progress):** Evaluating SFT'd checkpoints on rephrased/perturbed MATH problems. If post-SFT scores rise on rephrased problems, that's direct evidence of generalization even at 344M.

### Weakness 3: Single benchmark (MATH)

We acknowledge this. Three reasons the findings likely generalize:

1. **Temperature and solution length are properties of generation, not MATH.** The logic (each token = opportunity to deviate from memorized path) applies to any generative benchmark.

2. **The survival process framework derives from general in-context scaling laws**, not MATH-specific patterns. The regimes depend on $E$ and $\alpha$, measurable for any task.

3. **MATH is among the most widely used generative benchmarks** — a natural first target. Extension to code/reasoning is valuable future work.

### Question 1: Justify studying gen vs. mem at 344M?

See framing above. We study contamination mechanics, not the gen/mem boundary. 344M provides clean isolation at tractable cost.

### Question 2: Isn't Finding #2 trivially true?

Finding #2 is not "gains are memorization." It shows contamination-driven performance is *brittle*: rephrased/perturbed problems collapse to ~0% across all contamination levels (Table 1 reports 344M; consistent results across all model sizes are noted in the caption). This brittleness is an empirically useful diagnostic at any scale — if a 70B model scores well on MATH, one wants to know whether scores survive rephrasing. Our 344M results establish the baseline pattern under pure memorization.

### Question 3: Validated three regimes on another benchmark?

Not yet. The regimes are mathematical consequences of Equation 3, not patterns fit to MATH. They depend on whether $E > 0$, $\alpha \leq 1$, or $\alpha > 1$ — measurable for any generative task. This is a testable prediction for future work.

## Message to Area Chair Regarding Reviewer 4xWn

Dear Area Chair,

We raise concerns about Review 4xWn (score 2, confidence 5).

**Factual errors:**

1. The reviewer claims our statement that prior work "focused on discriminative benchmarks" is "not true," citing Kocyigit et al. (2025). Our paper says "predominantly focused on" — the reviewer misread this as an exclusivity claim, then cited a paper we already reference in the introduction and Appendix A Related Work.

2. The reviewer says MATH has "very short" answers, contradicting our generative framing. MATH requires generating full chain-of-thought solutions (15–1,949 tokens). The reviewer conflates final answers with solutions.

3. The reviewer attributes "mathematically discover" to our paper (L.081) and calls it "obviously misleading." Our paper says **"mathematically describe"** (Section 1, contribution bullet 3). The reviewer misquoted the text and critiqued the misquotation.

**Internal inconsistencies:**

4. Originality score 1 (poor), yet the reviewer acknowledges Finding #3 is "a new experimental result" that "contradicts other existing work."

5. Calls our work "a replica of Bordt et al." without elaboration. Bordt studies 7 MCQA (discriminative) benchmarks. Findings #6, #7, #8 have no analog in Bordt.

**Sparse engagement:**

The review is the shortest of four, has no questions ("See Strengths / weaknesses"), and a one-sentence summary — yet claims confidence 5/5. The level of engagement does not match the stated certainty, particularly given the factual errors above.

We ask the Area Chair to weigh these concerns when calibrating this review's influence.

## Rebuttal to Reviewer 6RQA

We thank the reviewer for identifying memorization-generalization disentanglement as the key opportunity.

### Weakness 1: Memorization vs. generalization not disentangled

Finding #2 and Table 1 provide direct evidence: contaminated models score ~0% on rephrased/perturbed problems, confirming gains are verbatim memorization with no transfer. We will expand Table 1 to show all model sizes (currently only 344M displayed) and add cross-entropy.

To decompose the SFT effect (Finding #5), we will evaluate SFT'd models on rephrased/perturbed test sets:
- **Pre-SFT**: ~0% on rephrased/perturbed (Table 1)
- **Post-SFT**: if scores rise → direct evidence of **generalization**
- **Forgetting**: already measured by test loss increase at high contamination

**TODO: Run evaluation of SFT'd checkpoints on rephrased/perturbed MATH test sets.**

### Weakness 2: Single-replica result depends on pretraining data mix

Fair point — the specific $E(R=0) = 3.594$ is corpus-dependent. But the qualitative finding is robust: the contaminated corpus contains test solutions; the uncontaminated does not. No compute on a corpus lacking answers can produce them — an information-theoretic advantage, not a data-mix artifact. The threshold may shift with corpus quality, but the phenomenon holds whenever the test set is distributionally distinguishable from pretraining data.

### Question: Do overtraining results conflate dilution and forgetting?

Yes — separating them would require overtraining by repeating the contaminated corpus vs. adding fresh data. Two observations favor dilution:

1. The crossover point shifts smoothly with model size (32 replicas at 34M → 10 at 63M → 1 at 93M) — regular scaling unlikely from catastrophic forgetting.
2. The dose-response framework (Schaeffer et al., 2025) provides theoretical grounding.

The practical implication is the same under either mechanism.

### Question: Implications beyond evaluations?

- **Privacy:** Single-exposure memorization produces detectable aggregate effects (Finding #3) even when MIAs achieve limited success (AUC < 0.7; Hayes et al., NeurIPS 2025).
- **Alignment:** If safety-relevant behaviors are memorized, the survival process (Finding #8) predicts brittleness under temperature perturbation or long sequences.
- **Benchmark design:** Longer solutions = more resistant to contamination — a concrete design principle.

## Rebuttal to Reviewer THKB

We thank the reviewer for recognizing the value of tracing contamination across the full model lifecycle.

### Weakness 1: Generalizability to larger models and real-world settings

Controlled contamination requires controlling the entire corpus — prohibitively expensive at billion-scale. This applies to all prior work.

Our findings have practical implications at each lifecycle stage:
- **Detection:** Temperature sweeps (Finding #6) and solution-length stratification (Finding #7) as lightweight contamination diagnostics.
- **Mitigation:** Overtraining on fresh data dilutes contamination (Finding #4).
- **Risk assessment:** The survival process framework (Finding #8) provides vocabulary for assessing benchmark vulnerability.
- **Extrapolation:** Scaling laws (Finding #3) bridge to larger compute budgets.

### Weakness 2: Finding #8 is underexplained

We will expand the exposition. Briefly:
- **Survival process:** Probability of generating a correct solution of length $T$ = product of per-token success probabilities. Each token is an opportunity for the memorized sequence to "die."
- **Decoherence** (Regime I): $E > 0$ → errors accumulate → memorization lost.
- **Lock-in** (Regime III): $E \approx 0$, $\alpha > 1$ → survival probability converges to a positive constant → memorization persists.
- **Brittle memorization** (Regime II): $E \approx 0$, $\alpha \leq 1$ → stretched exponential decay → fragile at long lengths.

### Question 1: External validity?

See Weakness 1 response above.

### Question 2: Generalize to larger models on mixed corpora?

Key differences from realistic settings: (1) scale (344M vs. billions), (2) corpus composition (single web crawl vs. heterogeneous mixtures), (3) contamination mechanism (exact replicas vs. near-duplicates). Scaling laws address (1). Points (2) and (3) are genuine limitations — qualitative findings should hold, but specific thresholds may shift. We will discuss this more explicitly.

### Limitations placement and detection proxy risks

The reviewer notes we "place the limitations in the appendix." Our limitations are in fact in the main body (Section 6 Discussion, dedicated paragraph). The reviewer may have been looking for a standalone section heading; we will make it more visually prominent.

We appreciate the concern that the three regimes could be misused as detection proxies. They describe idealized asymptotic dynamics — tendencies, not sharp boundaries. We will add a cautionary note that they should be validated empirically before being used as detection criteria.

### On originality: connection to membership inference attacks

As we discuss in Sections 3 and 6, our findings sit in tension with Hayes et al. (NeurIPS 2025), who scale LiRA to 1B parameters with 128 reference models and find AUC < 0.7 — limited success at individual sample detection.

In contrast, we find that even a single test set replica produces stark shifts in aggregate evaluation dynamics: cross-entropy below irreducible error (Finding #3), divergent temperature sensitivity (Finding #6), qualitative regime shifts (Finding #8).

Many details differ between setups, so we don't claim a causal explanation. But the contrast highlights that contamination can have large effects on generative evaluation metrics even when elusive to detect at the sample level. This juxtaposition underscores a puzzle the field needs to grapple with.
