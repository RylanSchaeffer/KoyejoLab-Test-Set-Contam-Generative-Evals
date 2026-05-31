OpenReview
.net
Notifications1000
Activity
Tasks
Rylan Schaeffer
back arrowBack to Author Console
Quantifying the Effect of Test Set Contamination on Generative Evaluations
Download PDF
Rylan Schaeffer, Joshua Kazdan, Baber Abbasi, Ken Liu, Brando Miranda, Ahmed M Ahmed, Fazl Barez, Abhay Puri, Stella Biderman, Niloofar Mireshghallah, Sanmi Koyejo
Published: 26 May 2026, Last Modified: 26 May 2026ICML 2026 FoGen Workshop PosterEveryone
Revisions
BibTeX
CC BY 4.0
Keywords: memorization, benchmark contamination, test set contamination
TL;DR: Targeted examination of test set contamination on generative benchmarks
Abstract:

Test set contamination -- the inclusion of benchmarks in pretraining data -- is a critical threat to the trustworthy evaluation of AI systems. While its impact on discriminative evaluations is well-studied, contamination on generative evaluations remains underexplored. We quantitatively assess these effects across the language model lifecycle by pretraining models (up to 344M parameters) on web data contaminated with varying numbers of MATH test set replicas. Performance expectedly improves with contamination and model size, with improvements stemming from superficial memorization, not generalization. Our scaling law analysis reveals a fundamental breach: including even a single test set replica enables models to achieve lower loss than the irreducible error of training on the uncontaminated corpus. We then study additional training: overtraining with fresh data dilutes contamination effects, whereas supervised finetuning on the training set improves performance for low contamination but degrades performance for high contamination. At inference, we identify three distinct regimes of memorization—exponential decoherence to brittle memorization to deterministic lock-in—governed by solution length and sampling temperature. Finally, we identify and fix a critical implementation error in EleutherAI's ALM Evaluation Harness that previously underreported mathematical reasoning performance. By characterizing how generation and memorization interact, we highlight new considerations for trustworthy AI evaluation.
Submission Number: 9
Filter by reply type...
Filter by author...
5 / 5 replies shown
Add:
Paper Decision
Decisionby Program Chairs25 May 2026, 10:11 (modified: 26 May 2026, 07:12)Program Chairs, AuthorsRevisions
Decision: Accept (Poster)
Meta Review of Submission9 by Area Chair 2mQi
Meta Reviewby Area Chair 2mQi22 May 2026, 09:44 (modified: 26 May 2026, 07:34)Area Chairs, Authors, Program ChairsRevisions
Metareview:

This paper investigates how test set contamination impacts generative evaluations across the model lifecycle using the MATH benchmark. Reviewers praise the rigorous experimental design, the insightful three-regime memorization framework modulated by temperature, and a striking scaling law analysis showing that a single replica can drop loss below the uncontaminated irreducible error. The paper also provides a valuable practical contribution by fixing a bug in EleutherAI's Evaluation Harness.

Weaknesses center on scope limitations, specifically the reliance on a single benchmark and small model scales (up to 344M parameters), which limits generalizability to frontier LLMs.

Overall, the novelty and theoretical contribution match the scope of this workshop. We therefore decided to accept this submission.
Recommendation: Accept (Poster)
Confidence: 3: The area chair is somewhat confident
Quantifying Test Set Contamination in Generative Evaluations: Lifecycle Framing and Survival Analysis With Scope Bounded to Small Models and Single Benchmark
Official Reviewby Reviewer LNEm22 May 2026, 08:10 (modified: 26 May 2026, 07:32)Program Chairs, Area Chairs, Reviewer LNEm, AuthorsRevisions
Review:
Summary

This paper investigates how test set contamination affects generative evaluations, using MATH as the primary benchmark. The authors pretrain 34M to 344M parameter transformer LMs on corpora injected with varying numbers of test-set replicas and study effects across the full model lifecycle: pretraining covering scaling and irreducible error, further training covering overtraining and SFT, and inference covering sampling temperature and solution length. Key findings include that performance and cross-entropy improvements scale with contamination and model size; scaling-law fits suggest even a single replica can push loss below the irreducible error of an uncontaminated corpus; overtraining on fresh data and higher sampling temperatures mitigate contamination-driven gains; and a length/temperature-driven survival analysis reveals three regimes from brittle memorization to deterministic lock-in. The paper also identifies and helps fix a critical bug in a widely used evaluation harness for math verification.
Strengths
1. Technical Novelty and Innovation

1.1 The work shifts contamination analysis from the dominant discriminative and multiple-choice QA focus to generative reasoning, framing contamination as a sequence survival problem with length and temperature-controlled regimes. This is a substantive and timely reframing for a field increasingly reliant on generative benchmarks.

1.2 The paper proposes a simple but insightful scaling-based decomposition to quantify how contamination depresses the fitted irreducible error, and introduces a length-wise law that connects to a survival probability , providing a coherent descriptive framework spanning pretraining and inference.

1.3 The temperature-as-effective-exponent interpretation conceptually unifies sampling temperature with memorization brittleness, offering a practically grounded lens for auditing contaminated models at inference time.

1.4 The identification and resolution of an implementation bug in a widely used evaluation library (Eval Harness) that underreported Math Verify is a concrete practical contribution with direct implications for the correctness of prior reported results across the community.
2. Experimental Rigor and Validation

2.1 The contamination sweep covers 0 to approximately 3,000 replicas across multiple model sizes and token budgets, with evaluation via both generation under Math Verify and teacher-forced cross-entropy, providing two independent lenses on the same phenomenon.

2.2 Lifecycle ablations are robust: overtraining on fresh data, SFT on the train set, and inference-time temperature and solution length are all systematically examined, linking design choices to performance outcomes across the full training and deployment pipeline.

2.3 Rephrased and perturbed MATH variants provide meaningful evidence that contamination-driven gains are largely non-generalizing memorization rather than improved reasoning, strengthening the core claim about brittleness.

2.4 The overtraining study aligns with intuitive dose dilution: as the total token budget increases with fresh data, the fraction of contaminated tokens decreases, reducing leakage-driven gains. This is both intuitive and practically actionable.
3. Clarity of Presentation

3.1 The lifecycle framing covering pretraining, further training, and inference is clear and well-organized, providing readers a coherent structure through which to interpret otherwise disparate findings.

3.2 The figures communicate the monotone effects of contamination and the sharp temperature and length sensitivities effectively. The three-regime survival story is intuitive and well-illustrated.

3.3 The distinction between cross-entropy scaling and generative success probabilities is articulated with appropriate care, avoiding conflation of two meaningfully different quantities.
4. Significance of Contributions

4.1 The work addresses a pressing evaluation risk for frontier LLMs as the field shifts toward generative reasoning benchmarks, and the lifecycle view provides actionable mitigation guidance: overtrain on fresh data to dilute, stress-test with higher temperatures and longer required solutions, include paraphrase and perturbation variants, and audit toolchains rigorously.

4.2 The Eval Harness bugfix is a direct practical contribution to evaluation integrity across the broader community.

4.3 The conclusions motivate concrete evaluation guidelines: companion paraphrase and perturbation sets, reporting sensitivity to sampling hyperparameters, disclosing potential contamination provenance, and deprecating benchmarks with known widespread leakage.
Weaknesses
1. Technical Limitations

1.1 All models are relatively small at 344M parameters or fewer, raising substantive questions about extrapolation to frontier-scale LLMs where inductive biases, memorization dynamics, and the interaction of contamination with instruction tuning can differ significantly.

1.2 The claim that a single replica beats the irreducible error of the uncontaminated corpus hinges on scaling-law fits and extrapolation from a narrow compute regime. The infinite-compute interpretation may be overreaching given known nonstationarities in scaling laws at larger budgets. A tempered statement emphasizing the within-range empirical observation with explicit extrapolation caveats would be more defensible.

1.3 The argument is conceptually plausible but remains heuristic. The assumed logit-gap growth with and its dependence on would benefit from direct empirical validation via measured token-level logit gaps and error rates.

1.4 The largest contamination settings covering hundreds to thousands of replicas make the contaminated set a sizable share of total training tokens, which is less representative of inadvertent real-world leakage and reduces the ecological validity of the most extreme conditions.
2. Experimental Gaps

2.1 There is no explicit quantification of the contamination fraction across model sizes and overtraining multipliers. This is crucial for dose-response interpretability and would allow readers to compare findings against real-world leakage rates.

2.2 It is unclear whether the base web corpus already contained MATH test or train items. The possibility of pre-existing leakage is not ruled out with a provenance check, which is a methodological gap that could confound the R=0 baseline.

2.3 Rephrased and perturbed set construction lacks detail about quality control procedures, semantic equivalence guarantees, and solver calibration. The extremely low verify rates below 0.1% across all conditions deserve more scrutiny and may reflect perturbations that inadvertently alter difficulty or require solution paths different from memorized chains.

2.4 Variance estimates from multiple random seeds or checkpoints are missing; results appear to be single runs, which limits confidence in the scaling fits and lifecycle crossovers.

2.5 SFT protocol details including optimizer, learning rate schedule, batch size, epochs, weight decay, prompt formatting, and deduplication against the test set are sparse. The forgetting versus generalization interpretation of the SFT result remains speculative without ablations such as SFT on fresh math data or format-mismatched math datasets.

2.6 Inference ablations are limited to temperature-only sampling. Common strategies such as top-p, top-k, and beam search are omitted, which limits the generality of the temperature-as-truth-serum claim.
3. Clarity and Presentation Issues

3.1 Some mathematical derivations are summarized informally and key assumptions, including the conditions under which the core scaling equation fails, are deferred to the appendix without brief in-text caveats that would help readers assess scope.

3.2 The Eval Harness bugfix, while important, lacks a precise task, version, and commit reference in the main text. This should be included for traceability and to allow others to audit downstream impacts.
Assessment

This is a timely, insightful, and empirically substantive study of test set contamination in generative evaluations. The lifecycle framing, the dose-response characterization via scaling, and the analysis of temperature and length-dependent survival regimes provide a coherent and practically grounded picture of how memorization inflates generative scores while remaining brittle under perturbation. The Eval Harness bugfix is a direct practical contribution to evaluation integrity. However, the most prominent extrapolation claim regarding a single replica beating the irreducible error would benefit from stronger empirical validation and clearly stated caveats, and the heuristic requires direct empirical backing. The experimental scope is constrained to small models and a single benchmark, and methodological details covering contamination fraction, SFT protocol, provenance checks, and seed variance need strengthening. Nonetheless, the core empirical observations are valuable and actionable, and the work advances the community's understanding of generative contamination mechanics with concrete guidance for more trustworthy evaluation practice. Acceptance to the FoGen workshop is appropriate contingent on addressing the noted methodological clarifications and tempering the strongest extrapolation claims.
Rating: 6: Marginally above acceptance threshold
Confidence: 3: The reviewer is fairly confident that the evaluation is correct
The paper addresses an important problem with a thoughtful experimental design and useful findings, though its scope is somewhat limited by the single benchmark and small model scale.
Official Reviewby Reviewer pBwC21 May 2026, 01:39 (modified: 26 May 2026, 07:32)Program Chairs, Area Chairs, Reviewer pBwC, AuthorsRevisions
Review:

This paper studies how test set contamination affects generative evaluation, focusing on the MATH benchmark. The authors pretrain small-to-mid-sized language models with varying amounts of MATH test-set replicas and show that contamination can significantly inflate performance, mainly through memorization rather than true mathematical generalization. The paper also analyzes how overtraining, supervised fine-tuning, sampling temperature, and solution length influence contamination effects.

Strengths:

    Important and timely topic for trustworthy LLM evaluation.
    Strong controlled experimental setup with different contamination levels and model sizes.
    Clear finding that contamination-driven gains collapse under rephrasing/perturbation, supporting the memorization argument. Interesting lifecycle analysis: pretraining, overtraining, SFT, and inference-time behavior.
    Practical contribution through identifying and fixing an evaluation harness issue.

Weaknesses:

    Experiments are limited to one benchmark, MATH, so the generality to coding, reasoning, or open-ended generation is unclear.
    The paper would benefit from more ablations on contamination format, partial contamination, and real-world web contamination patterns.

Rating: 6: Marginally above acceptance threshold
Confidence: 3: The reviewer is fairly confident that the evaluation is correct
Well-Executed Controlled Study with Strong Findings, Some Scope Limitations
Official Reviewby Reviewer N8Jt20 May 2026, 15:38 (modified: 26 May 2026, 07:32)Program Chairs, Area Chairs, Reviewer N8Jt, AuthorsRevisions
Review:

Strengths

    Rigorous experimental design. The controlled contamination setup is clean and well-motivated. By holding total training tokens constant while varying the number of test set replicas, the authors isolate the causal effect of contamination from data quantity effects. This is methodologically sound and an improvement over some prior work.
    Novel findings on generative vs. discriminative contamination. The paper's central claim — that generative contamination relies on fragile verbatim memorization of long token chains, distinct from robust reasoning — is well-supported. The rephrasing/perturbation experiments (Table 1) are particularly compelling: performance collapses to baseline under both modifications across all model sizes and contamination levels, which is strong evidence against generalization.
    The three-regime memorization framework (Finding #8) is the paper's most intellectually interesting contribution. Characterizing memorization via exponential decoherence, brittle memorization, and deterministic lock-in — with sampling temperature modulating the effective scaling exponent via αeff(τ)≈α/τ\alpha_\text{eff}(\tau) \approx \alpha/\tau αeff​(τ)≈α/τ — is a clean theoretical framing that gives practitioners a useful lens.
    Practical bug fix. Identifying and correcting a systematic underreporting bug in EleutherAI's Evaluation Harness (affecting minerva_math scores prior to task version 3.0) is a concrete, high-value contribution to the community, independent of the paper's main findings.
    The scaling law analysis (Finding #3) is striking: even a single test set replica can push models below the irreducible error of the uncontaminated corpus, implying that contamination "buys" effectively infinite compute under standard scaling law extrapolation. This is an attention-grabbing result with direct implications for evaluation trustworthiness. Weaknesses
    Single benchmark limitation. The entire study is conducted on MATH. The authors acknowledge this, but it significantly limits generalizability. MATH has unusually clean structure (boxed answers, fixed solution format), which may make memorization easier to detect and characterize than in code generation, open-ended reasoning, or creative writing tasks. Whether the three-regime framework generalizes to tasks without this structure is an open question.
    Model scale. The largest model is 344M parameters — roughly two orders of magnitude smaller than frontier models. The scaling law analysis is suggestive but the paper is appropriately cautious about extrapolation. However, some findings (e.g., the crossover point in overtraining shifting with model size) raise the concern that qualitative behavior at 344M may not reflect behavior at 70B+.
    The SFT finding (Finding #5) is intriguing but the mechanistic explanation ("contaminated models learn to generalize but also forget") is speculative and not empirically validated. A more careful ablation — e.g., measuring forgetting and generalization separately — would strengthen this section.
    The comparison with Huang et al. (2024) and Hayes et al. (2025) is raised but not resolved. The authors suggest MATH's distributional distinctness from FineWeb-Edu-Dedup as a possible explanation, but this deserves more investigation rather than being left as future work.
    Temperature as "truth serum" is a compelling framing, but the practical implication is underexplored. Evaluators cannot simply raise temperature arbitrarily without affecting uncontaminated model performance. A comparison of contaminated vs. uncontaminated models across temperatures would sharpen this recommendation.

Rating: 7: Good paper, accept
Confidence: 4: The reviewer is confident but not absolutely certain that the evaluation is correct

    About OpenReview
    Hosting a Venue
    All Venues

    Contact
    Sponsors
    Donate

    FAQ
    Terms of Use / Privacy Policy
    News

OpenReview is a long-term project to advance science through improved peer review with legal nonprofit status. We gratefully acknowledge the support of the OpenReview Sponsors. © 2026 OpenReview

