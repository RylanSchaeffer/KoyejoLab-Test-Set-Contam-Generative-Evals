# Rebuttal Strategy — ICML 2026 Submission 2433

## Synthesized Reviewer Objections

- **Model scale too small (344M max) to support claims about modern LLMs.** A 344M model cannot do multi-step reasoning on competition math, so performance gains may be trivially attributable to verbatim memorization rather than revealing interesting memorization-generalization dynamics. [6RQA, Mmea, 4xWn, THKB]

- **Single benchmark (MATH only) limits generalizability.** Decoherence rates, scaling law parameters, and the three-regime framework may be artifacts of MATH's specific formatting or token distribution rather than general properties of generative contamination. [Mmea, THKB]

- **Memorization vs. generalization not adequately disentangled.** The paper conjectures about this trade-off but never quantifies it. Could split the test set into seen/unseen subsets, or evaluate on perturbed/rephrased problems. Called "the difference between a borderline and an excellent contribution." [6RQA, Mmea]

- **Finding #5 (SFT) claimed to be a memorization-generalization tension, but evidence is insufficient.** Mmea argues the performance drop is simply catastrophic forgetting at 344M scale, not a genuine tension. 6RQA separately notes the conjecture could be tested rather than assumed. [Mmea, 6RQA — different critiques]

- **Finding #8 (three regimes / survival process) needs more justification or clarity.** 4xWn says the model is "never justified" and "mathematically discover" is misleading. THKB's concern is milder: the exposition is "compacted" and terminology could be "further clarified." [4xWn — validity concern; THKB — clarity concern]

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

## Rebuttal to Reviewer Mmea

We thank the reviewer for their careful and substantive engagement with our work. We appreciate the recognition of our experimental design, the three-regime framework, and the evaluation bug fix. We address each concern below.

### Framing: What is the paper's contribution?

Before addressing the individual weaknesses, we wish to clarify a point of framing that we believe underlies all three concerns. The reviewer reads our paper as claiming to study the boundary between reasoning and memorization. We understand why — but that is not the core contribution. Our paper characterizes **the mechanics of how contamination inflates generative evaluation metrics, and what factors modulate that inflation**: temperature sensitivity (Finding #6), solution-length decay (Finding #7), the survival process framework (Finding #8), overtraining mitigation (Finding #4), and the interaction with SFT (Finding #5). These findings are about *how memorization behaves during sequential generation* — dynamics that are absent from discriminative evaluations and that had not been previously characterized.

Under this framing, the 344M scale is not a liability but a *feature* of the experimental design: because these models cannot genuinely solve competition math, the contamination signal is cleanly isolated from real capability. At larger scales where models can partly solve MATH, contamination effects and genuine reasoning would be confounded and harder to disentangle. Our setup provides the cleanest possible measurement of the phenomenon we set out to study.

### Weakness 1: Model scale (344M) is too small

We agree that 344M models cannot perform multi-step reasoning on competition-level mathematics. As noted above, we view this as a strength for isolating contamination dynamics, not a limitation of the core contribution.

That said, we take the scale concern seriously and offer two responses:

1. **Controlled contamination requires pretraining from scratch.** This is the methodological price of causal identification, and it applies equally to all prior controlled contamination work. Bordt et al. (2025) — the most comparable study, on discriminative benchmarks — uses models of similar scale for the same reason.

2. **Our scaling law analysis (Finding #3) provides a principled bridge to larger scales.** The fitted parameters $E(R)$, $C_0(R)$, and $\alpha(R)$ vary smoothly with contamination level across our model sizes, and the functional form achieves fitting error $< 10^{-2}$ for all $R$. These scaling laws make quantitative, falsifiable predictions about contamination effects at compute budgets beyond our experimental range.

We also note that even if performance gains at 344M are "trivially" memorization, the *dynamics* of that memorization are not trivial: that temperature > 0.6 disrupts it, that solution length creates an exponential barrier, that a single replica suffices to beat the irreducible error, that overtraining washes it out at rates that scale predictably with model size — none of these are obvious a priori. If they were, they would have been characterized before.

**New experiment (in progress):** We are running pass@k evaluations (k = 1000–10000) on our uncontaminated 344M models at temperature 1.0, stratified by MATH difficulty level. If even a faint signal of genuine capability emerges on easier problems (Level 1–2), this directly refutes the premise that 344M "fundamentally lacks the capacity to generalize" on MATH. If pass@k remains zero, it reinforces the clean-separation argument above. We will report results in the revision.

### Weakness 2: SFT finding (Finding #5) is catastrophic forgetting, not a memorization–generalization tension

We appreciate this sharp observation and partially agree. At 344M scale, the reviewer is likely correct that the mechanism driving the performance drop at high contamination is predominantly catastrophic forgetting of memorized content, rather than a genuine tension between two competing learning signals.

However, we believe the *asymmetry* of the SFT interaction is the important finding, regardless of the mechanistic label. SFT improves test performance for models with low contamination ($R < 10$) and degrades it for models with high contamination ($R > 10$). Even under the catastrophic forgetting interpretation, this asymmetry is non-obvious and practically informative: it tells practitioners that SFT can inadvertently *unmask* contamination in heavily contaminated models by erasing memorized content, while providing genuine (if modest) gains for lightly contaminated ones.

We will revise the language in the paper to describe this as an asymmetric interaction between SFT and contamination level, without making strong claims about the underlying mechanism at this model scale.

**New experiment (in progress):** As part of our response to Reviewer 6RQA, we are evaluating SFT'd model checkpoints on rephrased and perturbed MATH test problems. If post-SFT scores on rephrased problems rise above the pre-SFT baseline of ~0%, this would provide direct evidence that SFT induced some degree of generalization even at 344M. We will report these results in the revision.

### Weakness 3: Single benchmark (MATH)

We acknowledge this limitation. Our findings are empirically grounded in the MATH benchmark, and we cannot rule out that specific quantitative parameters (e.g., decoherence rates, scaling law coefficients) are influenced by MATH's particular structure.

However, we believe the key findings are likely to generalize, for three reasons:

1. **The moderating factors are properties of generation, not of MATH.** Temperature and solution length are inherent to any autoregressive generation task. The finding that higher temperature and longer solutions disrupt memorization follows from the sequential nature of generation — each additional token is an opportunity for the model to deviate from the memorized path. This logic applies to code generation, translation, and any other generative benchmark.

2. **The survival process framework is derived from general principles.** The three regimes emerge from the asymptotic behavior of in-context scaling laws (Equation 3), which have been observed across many domains. The framework's predictions depend on whether $E > 0$, $E \approx 0$ with $\alpha \leq 1$, or $E \approx 0$ with $\alpha > 1$ — parameters that can be measured for any generative task.

3. **MATH is among the most widely used generative benchmarks.** It is a natural and high-impact first target for this line of investigation. Extending to code generation (HumanEval), logical reasoning, or other generative tasks is valuable future work, and we will note this explicitly.

### Question 1: How do you justify studying generalization vs. memorization at 344M?

As discussed in the framing section above, we are not primarily studying the boundary between generalization and memorization. We are studying the mechanics of how contamination inflates generative evaluation metrics. The 344M scale is appropriate for this question because it provides clean isolation of the contamination signal at a tractable computational cost. Our scaling law analysis extends the findings beyond the experimental scale.

### Question 2: Isn't Finding #2 trivially true?

Finding #2 is not simply the statement "performance gains are memorization." It is a controlled experiment demonstrating that contamination-driven performance is *brittle*: when test problems are rephrased (different wording, same numbers) or perturbed (different numbers, same structure), performance collapses to ~0% across all contamination levels and model sizes (Table 1). This brittleness is itself an empirically useful diagnostic — and it would be informative at any scale. If a 70B model achieves high accuracy on MATH, one would want to know whether those scores survive rephrasing. Our results at 344M establish the expected pattern under pure memorization, providing a baseline for interpreting results at larger scales where memorization and genuine capability may coexist.

### Question 3: Have you validated the three regimes on another benchmark?

Not yet. This is a valuable direction for future work and we will note it explicitly. We observe, however, that the three regimes are mathematical consequences of the in-context scaling law (Equation 3), not empirical patterns fit to MATH specifically. The regimes emerge from the asymptotic behavior of the survival probability as a function of the parameters $E$ and $\alpha$, which can be measured for any generative task. We would expect qualitatively similar regime structure for any benchmark where per-token NLL follows a power-law decay with token index — a testable prediction that future work can validate.

## Message to Area Chair Regarding Reviewer 4xWn

Dear Area Chair,

We wish to respectfully raise concerns about the quality of Review 4xWn (score 2, confidence 5). While we have engaged substantively with every point raised (see our response above), we believe several aspects of this review warrant scrutiny.

**Factual errors suggest insufficient engagement with the paper:**

1. The reviewer claims our statement that prior work "focused on discriminative benchmarks" is "not true," citing Kocygit et al. (2025) as a counterexample. However, our claim is that the literature has *focused on* discriminative benchmarks — a statement about the overwhelming distribution of prior work, not a claim of exclusivity. This is plainly true: the vast majority of controlled contamination studies use discriminative evaluations, with Kocygit et al. being the sole exception we are aware of. The reviewer appears to have misread "focused on" as "exclusively studied," and then cited a paper that we ourselves already reference three times (introduction, Section 4, and Appendix A).

2. The reviewer states that our experiments use "mathematical datasets where the answer is very short," implying a contradiction with our generative framing. This conflates final answers with full solutions. The MATH benchmark requires generating complete chain-of-thought solutions spanning tens to hundreds of tokens. Finding #7 and Figure 6 explicitly analyze how solution length modulates contamination effects — a core contribution that the reviewer does not engage with.

**Internal inconsistencies:**

3. The reviewer assigns an originality score of 1 (poor) while simultaneously acknowledging that Finding #3 is a "new experimental result" that "contradicts other existing work." A finding that overturns prior conclusions is, by definition, an original contribution.

4. The reviewer calls our work "a replica of Bordt et al." without elaboration. Bordt et al. studies contamination on 7 multiple-choice (discriminative) benchmarks. Our paper studies generative evaluation — the distinction around which our entire contribution is built. Findings #6, #7, and #8 (temperature, solution length, and the survival process framework) have no analog in Bordt et al., and Finding #3 directly contradicts their conclusions.

**Sparse engagement relative to stated confidence:**

The review is the shortest of the four, provides no specific questions, and offers "See Strengths / weaknesses" in lieu of a Questions section. The summary is a single sentence. Despite this, the reviewer claims confidence 5/5 ("absolutely certain"). We respectfully suggest that the level of engagement does not match the stated certainty, particularly given the factual errors noted above.

We ask the Area Chair to weigh these concerns when calibrating the influence of this review on the final decision.

## Rebuttal to Reviewer 6RQA

We thank the reviewer for their constructive feedback and for identifying the memorization–generalization disentanglement as the key opportunity for improvement.

### Weakness 1: Memorization vs. generalization not adequately disentangled

Finding #2 and Table 1 already provide direct evidence: contaminated models score ~0% on rephrased and perturbed MATH test problems across all model sizes and contamination levels, confirming that gains are verbatim memorization with no transfer. The reviewer is right that Table 1 deserves more detail — in the revision, we will expand it to show results for all model sizes (currently only 344M is displayed) and report cross-entropy alongside Math Verify scores.

However, this evidence does not decompose the SFT effect (Finding #5) into its generalization and forgetting components. To address this, we will evaluate our SFT'd models on the rephrased/perturbed test sets:

- **Pre-SFT**, contaminated models score ~0% on rephrased/perturbed problems (Table 1).
- **Post-SFT**, if scores rise on rephrased/perturbed problems → direct evidence SFT induced **generalization**.
- The **forgetting** component is already measured by the test loss increase for highly contaminated models (Finding #5).

This decomposes the conjecture into two measured quantities using existing checkpoints and test sets.

**TODO: Run evaluation of SFT'd model checkpoints on rephrased/perturbed MATH test sets. Effect sizes may be small at 344M scale.**

### Weakness 2: Single-replica result depends on pretraining data mix

Fair point. The specific irreducible error $E(R=0) = 3.594$ is corpus-dependent. However, the qualitative finding is robust: the contaminated corpus contains the test solutions and the uncontaminated corpus does not. No amount of compute on a corpus lacking the answers can produce them — this is an information-theoretic advantage, not an artifact of the data mix. The exact threshold may shift with corpus quality (e.g., a math-heavy corpus would have lower baseline $E$), but the phenomenon should hold whenever the test set is distributionally distinguishable from pretraining data. We will note this nuance in the revision.

### Question: Do overtraining results conflate dilution and catastrophic forgetting?

Yes, our experiments do not isolate these — separating them would require overtraining by repeating the contaminated corpus (forgetting without dilution) vs. adding fresh data (both). Two observations favor dilution as the primary driver:

1. The crossover point shifts smoothly with model size (32 replicas at 34M → 1 replica at 93M), consistent with a dilution effect. Catastrophic forgetting would be less likely to produce such regular scaling behavior.
2. The dose-response framework (Schaeffer et al., 2025) provides theoretical grounding: fresh data dilutes the "dose" of contamination.

The practical implication is the same under either mechanism.

### Question: Implications beyond evaluations?

- **Privacy:** Single-exposure memorization produces detectable aggregate effects (Finding #3) even when membership inference attacks fail at the individual level (Hayes et al., NeurIPS 2025), suggesting memorization is more prevalent than current detection methods reveal.
- **Alignment:** If safety-relevant behaviors are memorized rather than learned, the survival process framework (Finding #8) predicts they will be brittle under temperature perturbation or long behavioral sequences.
- **Benchmark design:** Benchmarks requiring longer solutions are inherently more resistant to contamination, suggesting a concrete design principle for future evaluations.

## Rebuttal to Reviewer THKB

We thank the reviewer for their careful reading and for recognizing the value of tracing contamination effects across the full model lifecycle. We are glad the findings came across as "coherent and progressive." We address each point below.

### Weakness 1: Generalizability to larger models and real-world settings

We agree that our controlled setting necessarily differs from real-world pretraining in model scale, data distribution, and training procedure. This is inherent to the methodology: isolating the causal effect of contamination requires controlling the entire pretraining corpus, which is prohibitively expensive at billion-scale. This constraint applies to all prior controlled contamination work.

That said, we believe our findings have direct practical implications at each stage of the model lifecycle:

- **Detection via temperature sensitivity (Finding #6):** If a model's generative benchmark scores degrade sharply as sampling temperature increases beyond ~0.6, this is a signal consistent with contamination-driven memorization rather than genuine capability. Practitioners could use temperature sweeps as a lightweight contamination diagnostic.

- **Detection via solution-length stratification (Finding #7):** If a model's accuracy is concentrated in short solutions and decays steeply with solution length, this pattern is consistent with brittle memorization. Stratifying evaluation results by solution length is cheap and requires no additional infrastructure.

- **Mitigation via overtraining (Finding #4):** Our results suggest that continued training on fresh data can dilute contamination effects, offering a concrete lever for practitioners who suspect their pretraining corpus may be contaminated.

- **Risk assessment via the survival process framework (Finding #8):** The three regimes provide a quantitative vocabulary for assessing how vulnerable a given generative benchmark is to contamination, based on its typical solution length distribution and the per-token error dynamics of the model.

We also note that our scaling law analysis (Finding #3) provides a principled framework for extrapolating contamination effects to larger compute budgets, partially bridging the gap between our controlled setting and real-world scale.

### Weakness 2: Finding #8 (three regimes) is underexplained

We appreciate this feedback and will expand the exposition in the revision. To briefly clarify the key terms:

- The **survival process** refers to the sequential challenge a model faces during generation: at each token position, the model must place sufficient probability mass on the correct next token. The probability of generating a correct complete solution of length $T$ is the product of per-token success probabilities — analogous to a survival process where each token is an opportunity for the memorized sequence to "die."

- **Decoherence** (Regime I) occurs when the per-token irreducible error $E > 0$, meaning the model has a nonzero floor probability of error at every position. Over a long enough sequence, errors accumulate and the memorized solution is almost certainly lost — the model "decoherences" from the memorized path.

- **Lock-in** (Regime III) occurs when $E \approx 0$ and the per-token error decays fast enough ($\alpha > 1$) that the cumulative survival probability converges to a positive constant. The model remains "locked in" to the memorized solution regardless of length.

- **Brittle memorization** (Regime II) is the intermediate case: $E \approx 0$ but $\alpha \leq 1$, so errors decay slowly and the survival probability goes to zero, but sub-exponentially (a stretched exponential). Memorization is possible for short solutions but increasingly fragile as length grows.

We will incorporate this level of explanation into the main text.

### Key Question 1: More evidence of external validity / applying findings to real-world practice?

Please see our response to Weakness 1 above, where we outline concrete applications at each stage of the model lifecycle: temperature sweeps for detection, solution-length stratification, overtraining for mitigation, and the survival process framework for risk assessment.

### Key Question 2: Can findings generalize to larger models on mixed corpora?

The main differences between our controlled setting and realistic settings are: (1) model scale (344M vs. billions), (2) corpus composition (single high-quality web crawl vs. heterogeneous mixtures), and (3) contamination mechanism (exact replicas vs. near-duplicates or paraphrases). Our scaling law analysis (Finding #3) partially addresses (1) by providing extrapolation to larger compute budgets. Points (2) and (3) are genuine limitations — we would expect the qualitative findings (temperature sensitivity, solution-length decay, survival process regimes) to hold, but the specific quantitative thresholds (e.g., how many replicas trigger each regime) may shift. We will discuss these differences more explicitly in the revision.

### Limitations placement

The reviewer notes that we place limitations in the appendix rather than the main text. We will move the limitations discussion to the main body in the revision, as we agree this improves transparency. We also appreciate the concern about potential misuse of the memorization regimes as detection proxies — we will add a note cautioning that these regimes describe idealized dynamics and should be validated empirically before being used as detection criteria.

### On originality: connection to membership inference attacks

We wish to highlight an additional dimension of our contribution that we believe strengthens the paper's significance. Concurrent work by Hayes et al. (NeurIPS 2025) scales the strongest known membership inference attack (LiRA) to GPT-2 architectures up to 1B parameters with 128 reference models — a massive computational investment — and finds that attack success remains limited (AUC < 0.7) in practical settings. Many individual sample-level decisions are statistically indistinguishable from a coin flip.

Our findings sit in interesting tension with this result. Where state-of-the-art MIAs struggle to detect whether an individual sample was seen during training, we find that even a single test set replica produces stark, unmistakeable shifts in aggregate evaluation dynamics: cross-entropy drops below the irreducible error of uncontaminated training (Finding #3), temperature sensitivity diverges between contaminated and uncontaminated models (Finding #6), and the survival process regime shifts qualitatively from decoherence to lock-in (Finding #8).

Many details differ between our experimental setup and theirs — model family, dataset, training procedure, and what is being measured — so we do not claim a clean causal explanation for this discrepancy. But the contrast is striking and highlights that contamination can have large, measurable effects on generative evaluation metrics even when it proves elusive to detect at the individual sample level via existing methods. We believe this juxtaposition underscores a puzzle the field needs to grapple with, and positions our work as contributing a complementary lens on memorization that existing detection approaches do not provide.
