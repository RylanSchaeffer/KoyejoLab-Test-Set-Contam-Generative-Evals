OpenReview
.net
Notifications1000
Activity
Tasks
Rylan Schaeffer
back arrowGo to NeurIPS 2026 Conference homepage
Test Set Contamination of Generative Evaluations
Download PDF
Rylan Schaeffer, Joshua Kazdan, Baber Abbasi, Ken Liu, Brando Miranda, Ahmed M Ahmed, Fazl Barez, Yegor Denisov-Blanch, Abhay Puri, Stella Biderman, Niloofar Mireshghallah, Sanmi Koyejo
04 May 2026 (modified: 27 May 2026)NeurIPS 2026 Conference SubmissionConference, Senior Area Chairs, Area Chairs, Reviewers, Authors
Revisions
CC BY 4.0
Abstract:

Test set contamination -- the inclusion of benchmarks in pretraining data -- is a critical challenge to the trustworthy evaluation of frontier AI systems. While its impact on \emph{discriminative} evaluations is well-studied, contamination on \emph{generative} evaluations remains underexplored. We quantitatively assess these effects across the language model lifecycle by pretraining models (up to 344M parameters) contaminated with varying numbers of MATH test set replicas. While performance expectedly improves with contamination and model size, the improvements stem from superficial memorization, not generalization. Our scaling law analysis reveals a fundamental breach: including even a single test set replica enables models to achieve lower loss than the irreducible error of training on the uncontaminated corpus. We then study additional training: overtraining with fresh data dilutes contamination effects, whereas supervised finetuning on the training set improves performance for low contamination but degrades it for high contamination. At inference, we identify three distinct regimes of memorization -- exponential decoherence, brittle memorization, deterministic lock-in -- governed by solution length and sampling temperature. Finally, we identify and fix a critical implementation error in EleutherAI's LM Evaluation Harness that previously underreported mathematical reasoning performance. By characterizing how generation and memorization interact, we highlight new considerations for trustworthy AI evaluation.
Checklist Confirmation: I confirm that I have included a paper checklist in the paper PDF.
Financial Support:  Baber Abbasi
Responsible Reviewing: We acknowledge the responsible reviewing obligations as authors.
Primary Area: Language and multimodal language models (e.g., text generation, summarization, VQA)
Secondary Area: General machine learning (e.g., core contributions in supervised and unsupervised methods)
Contribution Type: General: Most submissions will fall into this type.
Academic Integrity: I acknowledge that I have read the NeurIPS Handbook and commit to adhering to all policies in the Handbook (https://neurips.cc/Conferences/2026/MainTrackHandbook), the NeurIPS Code of Conduct and the NeurIPS Academic Integrity Policy.
LLM Usage: Editing (e.g., grammar, spelling, word choice)
LLM Experiment: Opt in to include this paper in the LLM-assisted peer review experiment.
Declaration: I confirm that the above information is accurate.
Reviewer Nomination:  Yegor Denisov-Blanch
Submission Number: 32216

    Discussion

Filter by reply type...
Filter by author...
4 / 4 replies shown
Add:
Meta Review of Submission32216 by Area Chair Rrc9
Meta Reviewby Area Chair Rrc923 Jul 2026, 04:58 (modified: 24 Jul 2026, 08:05)Senior Area Chairs, Area Chairs, Reviewers Submitted, Program Chairs, Area Chair Rrc9, AuthorsRevisions
Metareview:
(a) Summary of scientific claims and findings

    Studies test-set contamination in generative (not MCQ/discriminative) evaluation, a less-studied setting.
    Controlled pretraining pipeline: exact MATH test-set replicas inserted at varying counts into pretraining corpora for small models (up to 344M params, Qwen 3 family).
    Contamination raises MATH scores, but gains vanish on paraphrased/numerically perturbed variants — argued as memorization, not generalized reasoning.
    Scaling-law claim: a single contaminated replica can push test loss below the irreducible error of an uncontaminated model.
    Analyzes how overtraining and SFT interact with contamination (claimed "dilution" effect).
    Identifies three inference-time memorization regimes tied to temperature and solution length.

(b) Main strengths

    Irreducible-error result and temperature/solution-length characterization: novel, previously undemonstrated.
    Controlled pretraining design (varying model size and contamination dose directly, not black-box models). Isolates contamination effects across pretraining/SFT/inference lifecycle.
    Paraphrase/perturbation controls separating memorization from generalization:

(c) Main weaknesses / what might be missing

    Exact-replica contamination is a clean causal testbed but not representative of realistic leakage (paraphrased, partial, translated, or embedded in surrounding discussion).
    Experiments limited to models under 350M params, single model family, single benchmark (MATH) — generalization to larger models/other families/other tasks untested.
    only one seed per configuration, no error bars/uncertainty quantification;
    Findings 4–5 (overtraining/SFT "dilute" contamination) are contingent on cross-entropy loss changes, but loss measures probability of exact solution text, not correctness under the Math Verify metric used elsewhere; so loss-based evidence doesn't establish the generation-correctness claim.
    Related-work framing understates existing generative-evaluation contamination literature; two findings (rephrasing effects, repeat-count effect) both replicate and conflict with specific prior studies with no discussion of why.

(f) Reasons for decision

    Scores: two borderline reject (3, 3), one borderline accept (4) —
    Reviewer #1's objection is the one critique that questions whether the evidence supports the paper's stated claims (not just the scope of evidence)
    External-validity limitations raised independently by all three reviewers

Add:
Official Review of Submission32216 by Reviewer 8RFz
Official Reviewby Reviewer 8RFz25 Jun 2026, 22:34 (modified: 24 Jul 2026, 07:59)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer 8RFz, AuthorsRevisions
Summary:

This paper carries out a detailed investigation of how test set contamination impacts the evaluation of models on benchmarks that involve text generation. The investigation takes the form of a set of experiments which are broken down into 8 findings, which relate to the relationship between contaminated model performance and Uncontaminated Irreducible Error, the impacts of model size and number of repeats of contaminated examples, what happens when compute-optimal contaminated models are finetuned, and how solution length and generation temperature impact the effects of contamination.
Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:

Strengths:

    The analyses surrounding the uncontaminated irreducible error seem both novel and compelling. Specifically, I think the results that a single instance of problem contamination can lead to a test loss below the uncontaminated irreducible error is striking.
    While perhaps unsurprising, I do not believe any previous work has looked at the impact of temperature and solution length on contamination effects; and so this is also a novel idea.
    On the whole, the experiments seem well-thought-out and well-executed.

Weaknesses:

    At times, memorization and contamination are conflated in a way that has the potential to be misleading. This is clearest in the experiments in Section 4 (i.e., Findings 4 and 5). In Finding 4 (lines 149-150), it is stated that “the performance boost from contamination diminishes when overtraining with fresh data”; but the results of this study do not provide direct evidence for this. The fact that cross-entropy loss increase only indicates that the model assigns a lower probability to the exact text of the solution, but does not mean that the model stops providing a correct answer as judged by Math Verify score (and in fact, if this is the case, it is even more problematic, as it is likely to more effectively evade detection), even if the two metrics are correlated overall (as seen in Figure 11). The same issue applies to Finding 5.
    Another unaddressed confound is that of temperature. Specifically, Finding 6 does not differentiate between the possibility that a higher temperature reduces contamination effects specifically and the possibility that a higher temperature generally leads to poor performance or generally incoherent text.
    The paper should more clearly delineate novel research questions and findings from past work. First and foremost, the paper generally is framed as if little previous work exists on generative evaluations, but a number of previous studies have looked at this, including some that are cited in the paper (e.g., Dong et al., 2024; Kocyigit et al., 2025) and others that are not (e.g., Palavalli et al., 2024; Jiang et al., 2024; Mehrbakhsh et al., 2024; Dekoninck et al., 2024; Dekoninck et al., 2024; Godey et al., 2025). This is especially important in light of the fact that the rephrasing findings conflict with those of two of these works which look at generative evaluation of mathematical problems (Mehrbakhsh et al., 2024; Dekoninck et al., 2024), and replicates past work showing that an increased number of contamination repeats leads to stronger effects on some datasets (Jiang et al., 2024; Dekoninck et al., 2024). Neither of these findings (1 and 2) are discussed in the context of past work in the submission.

Quality: 2: not good
Clarity: 3: good
Significance: 3: good
Originality: 2: not good
Questions:

    Do the results of Findings 4 and 5 also hold for Math Verify score?
    Does increasing temperature to the highest levels (as described in Finding 6) specifically reduce contamination effects or generally lead to worse generations?
    What might explain the differences between Finding 2 of this study and previous work on both discriminative and generative evaluations suggesting that rephrased examples can lead to contamination effects?
    How are the values in Table 1 calculated?

Limitations:

Yes
Rating: 3: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:

N/A
Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes
Add:
Official Review of Submission32216 by Reviewer 1wx9
Official Reviewby Reviewer 1wx925 Jun 2026, 15:08 (modified: 24 Jul 2026, 07:59)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer 1wx9, AuthorsRevisions
Summary:

This work studies explicit MATH test-set contamination during pretraining. It shows that repeated test replicas boost accuracy and lower cross-entropy, especially in larger models and at higher contamination levels, but the gains vanish on rephrased or perturbed variants, indicating memorization rather than true generalization. The authors also analyze how overtraining, SFT, decoding temperature, and solution length can dilute or reveal contamination effects.
Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:

Strengths:

    The paper studies an important and difficult problem for trustworthy evaluation: contamination in generative benchmarks.
    It clearly motivates why contamination is harder to detect in generative evaluations than in MCQ-style evaluations.
    The controlled pretraining setup is clean and useful, since the authors directly vary model size and contamination dose instead of relying on black-box models.
    The paper introduces contamination during pretraining and studies how its effects evolve across the model lifecycle, including after SFT.
    It also studies inference-time factors, such as temperature sampling and solution length, that can reduce or expose contamination effects.
    The rephrased and numerically perturbed MATH controls are useful because they show that the gains mostly come from brittle memorization rather than robust mathematical reasoning.

Weaknesses:

    The contamination setting is very explicit: exact MATH test-set replicas with solutions are inserted into pretraining. This is a useful causal testbed, but it is not the dominant realistic leakage mode, where contamination may be paraphrased, partial, translated, synthetic, or embedded in benchmark discussions.
    Another alternate way of looking at Table 1 results is that if we have a rephrased test set contamination in training, the evaluation on test set is similar to an uncontaminated model. Which is quite surprising. May be model/train scale is not enough to see the generalization from contaminated data.
    The experiments also use relatively small models and a single benchmark/pretraining mixture, which limits the broad applicability of the findings.
    Because the leakage is exact, some results are less surprising and predictable, especially those based on test-set loss or perplexity. The findings may change under rephrased, partial, or synthetic contamination, so this setting should be studied or discussed more directly.

Quality: 3: good
Clarity: 3: good
Significance: 3: good
Originality: 3: good
Questions:

Can the authors add or discuss a setting where the pretraining contamination itself is paraphrased, translated, partial, or synthetically rephrased, rather than an exact test set replica? This would help determine whether the conclusions extend beyond blatant exact leakage, especially conclusions based on test set loss or perplexity.
Limitations:

yes
Rating: 4: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:

Nothing specific.
Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes
Add:
Official Review of Submission32216 by Reviewer aPBL
Official Reviewby Reviewer aPBL25 Jun 2026, 06:27 (modified: 24 Jul 2026, 07:59)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer aPBL, AuthorsRevisions
Summary:

The paper studies how test set contamination affects generative AI evaluations, using controlled pretraining experiments where small LLMs (<350M params from the Qwen 3 family) are trained with varying numbers of MATH benchmark test-set replicas. Although somewhat obvious, the paper finds that test set contamination greatly improve MATH scores, but the gains appear to come from brittle memorization rather than generalization. The authors also identify a bug with the original Math Verify scorer as part of the LM Evaluation Harness. Finally, the paper studies evaluation additional axes, such as analyzing scaling laws, overtraining, SFT, sampling temperature, and solution length, as a way to provide greater insight into the observed phenomena.
Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:

Overall, the paper is interesting and addresses an important problem given the prevalence of test set data contamination in the generative evaluation setting (moving beyond discriminative tasks) and how this affects the interpretation of model capabilities. However, there are several critical limitations of the work, including the use of very small models from a single model family (<350M params) and a single test dataset (MATH), which may result in the findings not generalizing to frontier models or other evaluation settings.

The main strengths of the paper are:

    Paper is well written, and generally clear in motivation and relating to prior work
    Concrete contribution back to the community via fixing an unknown bug related to verification of MATH solutions
    Well controlled experiment design, with separate pretraining runs across model sizes and MATH replica counts
    Demonstrated experiments using paraphrasing/perturbations to demonstrate the memorization vs. generalization gap
    Study many additional dimensions to provide greater insight into the phenomenon, including overtraining, SFT, solution length, and temperature

The main weaknesses of the paper are:

    Small models: unclear how these findings generalize to larger models and different families of models (e.g. Llama, Gemma, etc.)
    Single dataset: unknown whether the findings are specific to MATH, or could also apply to other generative evaluation tasks
    Statistical significance: it appears only a single seed was run per configuration, it would be good to quantify error bars or uncertainty related to the results across multiple runs
    I checked the Appendix, but it appears details about the SFT setup are missing (e.g. what hyperparameters were used), making the results potentially harder to reproduce

Quality: 3: good
Clarity: 3: good
Significance: 3: good
Originality: 3: good
Questions:

    Multiple replicates of the same test dataset seems somewhat of a contrived example- are there more realistic, real-world examples of test set contamination that could be tested using a similar controlled experiment design?

    How were the rephrasing/perturbations of the MATH dataset validated, and all their difficulty/length distributions matched with the original test set? This is used to justify the memorization vs. generalization claim.

    Does the irreducible error come from fitting an asymptotic scaling law? It's a strong claim that a single test-set replica beats the uncontaminated irreducible error, and this may depend on assumptions of the the functional form and extrapolation beyond measured results.

    The authors mention this as potential future work, but how would the authors set-up cross-domain experiments to understand the impact of test set contamination? e.g. pretraining on MATH dataset and how that affects math-related capabilities beyond the MATH dataset itself.

Limitations:

Yes, there is a good discussion in the Limitations section.
Rating: 3: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
Confidence: 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:

N/A
Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes
Add:
About OpenReview
Contact
FAQ
Hosting a Venue
Sponsors
Terms of Use / Privacy Policy
All Venues
Donate
News

OpenReview is a long-term project to advance science through improved peer review with legal nonprofit status. We gratefully acknowledge the support of the OpenReview Sponsors. © 2026 OpenReview

