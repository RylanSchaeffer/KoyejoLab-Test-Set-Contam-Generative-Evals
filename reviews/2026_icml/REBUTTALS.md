OpenReview.net

    Notifications1000
    Activity
    Tasks
    Rylan Schaeffer 

back arrowBack to Author Console
Quantifying the Effect of Test Set Contamination on Generative Evaluations
Download PDF
Rylan Schaeffer, Joshua Kazdan, Baber Abbasi, Ken Liu, Brando Miranda, Ahmed M Ahmed, Fazl Barez, Abhay Puri, Stella Biderman, Niloofar Mireshghallah, Sanmi Koyejo
09 Jan 2026 (modified: 30 Apr 2026)Submitted to ICML 2026Conference, Senior Area Chairs, Area Chairs, Reviewers, Authors
Revisions
BibTeX
CC BY-NC-ND 4.0
Verify Author List: I have double-checked the author list and understand that additions and removals will not be allowed after the abstract submission deadline.
TL;DR: Targeted examination of test set contamination in generative benchmarks
Abstract:

Test set contamination -- the inclusion of benchmarks in pretraining data -- is a critical threat to the trustworthy evaluation of AI systems. While its impact on discriminative evaluations is well-studied, contamination on generative evaluations remains underexplored. We quantitatively assess these effects across the language model lifecycle by pretraining models (up to 344M parameters) on web data contaminated with varying numbers of MATH test set replicas. Performance expectedly improves with contamination and model size, with improvements stemming from superficial memorization, not generalization. Our scaling law analysis reveals a fundamental breach: including even a single test set replica enables models to achieve lower loss than the irreducible error of training on the uncontaminated corpus. We then study additional training: overtraining with fresh data dilutes contamination effects, whereas supervised finetuning on the training set improves performance for low contamination but degrades performance for high contamination. At inference, we identify three distinct regimes of memorization—exponential decoherence to brittle memorization to deterministic lock-in—governed by solution length and sampling temperature. Finally, we identify and fix a critical implementation error in EleutherAI's ALM Evaluation Harness that previously underreported mathematical reasoning performance. By characterizing how generation and memorization interact, we highlight new considerations for trustworthy AI evaluation.
Primary Area: Deep Learning->Large Language Models
Keywords: memorization, benchmark contamination
Ethics Agreement: I certify that all co-authors of this work have read and are committed to adhering to the Call for Papers, Author Instructions, Research Ethics, and Peer-review Ethics.
LLM Policy: This submission requires Policy A.
Proceedings-only Option: If this paper is accepted, the authors tentatively plan to present it in person at the conference (as a poster and, if selected, as an oral).
Reciprocal Reviewing Status: This submission is NOT exempt from the Reciprocal Reviewing requirement. (We expect most submissions to fall in this category.)
Reciprocal Reviewing Author:  Joshua Kazdan
Submission Number: 2433
Filter by reply type...
Filter by author...
18 / 18 replies shown
Add:
Paper Decision
Decisionby Program Chairs30 Apr 2026, 08:49 (modified: 30 Apr 2026, 10:56)Program Chairs, AuthorsRevisions
Decision: Reject
Comment:

Making a call on this paper has been quite difficult. There are many merits to this work -- the problem motivation, the solution approach, and the interesting experimental set-ups. But, as the extensive discussions with reviewers indicate, the paper would benefit from another round of revision and is not quite ready for publication. I would encourage the authors to consider focussing the paper on fewer findings and more strongly supporting them. While all the findings are interesting, there are only so many you can support and justify rigorously in a 8-page paper. Maybe there is more than 1-paper's worth of ideas in here and you are stretching yourself too thin (e.g., Finding #3 on scaling laws, while intriguing, raises more questions than it answers). Finally, I would encourage the authors not be too discouraged with the reject decision -- it is just a matter of time (assuming the authors follow up on this round of feedback) that this work would be published at a top ML venue. It has the potential to attract the research community's attention and have impact. So please pursue this line of work and don't give up on it.
Official Review of Submission2433 by Reviewer 6RQA
Official Reviewby Reviewer 6RQA12 Mar 2026, 21:27 (modified: 09 Apr 2026, 01:05)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer 6RQARevisions
Summary:

The paper analyzes the effect of including math problems in pretraining on the final model's performance on those problems. It performs systematic experiments across a range of model sizes and number of copies of the data, presenting evidence that even a single copy of the data can cause memorization. The results demonstrate the potential for widespread contamination of generative evaluations of large language models. A variety of follow-up experiments clarify the dynamics of this phenomenon, e.g. dependence on sampling temperature and interaction with SFT.
Strengths And Weaknesses:
Strengths

The paper is well-motivated by concerns about the validity of generative evaluations. The writing and presentation is very clear. The experiments are thorough and informative-- it is particularly interesting that SFT on the train set has a reversed effect based on the amount of contamination. The findings underscore the importance of considering the threat that contamination poses to the evaluation of large language models.
Weaknesses

    The difference between memorization and generalization is central to the paper, but not adequately explored. For example, the authors write: "We conjecture that during SFT, contaminated models learn to generalize but also forget their contaminated pretraining data, and the effects of contamination are more potent than generalization for small models, leading to a net increase in test loss." This need not be a conjecture. The test set could be split into train-test and validation-test sets and generalization vs. memorization could be quantified. Validation-test loss could be reported in many cases alongside train-test loss. Alternatively, performance could be reported on perturbed problems. On a related note, insufficient detail is given on the results in Table 1. In my view, this weakness is significant, and represents the difference between a borderline contribution and an excellent contribution.

    Regarding the following: "including even a single replica enables almost all models to achieve lower cross entropy losses than the estimated irreducible error of the uncontaminated pretraining corpus." The paper overstates the significance of this result-- the result depends strongly on the pretraining data mix, and the pretraining mix is never varied, so it is unclear whether the result is generalizable.

    As the authors acknowledge, only limited conclusions can be drawn on the basis of such small models. However, the careful scaling laws analysis lends credibility to the results and allows for extrapolation to larger sizes-- so I do not see this as a major weakness.

Soundness: 3: good
Presentation: 4: excellent
Significance: 3: good
Originality: 3: good
Key Questions For Authors:

    My main question is just to ask for the authors response to the weaknesses proposed above.

    Am I correct in my understanding that the overtraining results conflate the effect of diluting the data and of catastrophic forgetting? If so, how much value do you think there would be in controlling for these separately?

    (My review does not hinge on this question) Out of curiosity, do you think there are implications of your results beyond evaluations, e.g. with regards to data efficiency, generalization, or alignment of advanced AI systems?

Limitations:

yes
Overall Recommendation: 3: Weak reject: A paper with clear merits, but also some weaknesses, which overall outweigh the merits. Papers in this category require revisions before they can be meaningfully built upon by others. Please use sparingly.
Confidence: 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Final Justification:

The paper is well-motivated, interesting, and well put-together, but has significant gaps on important questions: the generalizability of the results and the nature of the results themselves. The authors acknowledge W1, but as best I can tell, their response does not address it, lacking a critical baseline and generative evaluations (the focus of the work). The response to W2, as best I can tell, misses the point, incorrectly claiming that "in any realistic setting where the test set is distributionally distinguishable from pretraining data, the phenomenon holds." This is not true-- it is possible there are data distributions that are distinguishable but DO generalize to good math performance according to their test set. It pains me to recommend rejection for a paper with many strong qualities, but given these shortcomings, it seems appropriate to me.
Rebuttal by Authors
Rebuttalby Authors (Brando Miranda, Fazl Barez, Ken Liu, Rylan Schaeffer, +7 more)30 Mar 2026, 19:37 (modified: 31 Mar 2026, 05:50)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, AuthorsRevisions
Rebuttal:

We thank the reviewer for identifying memorization-generalization disentanglement as the key opportunity.
Weakness 1: Memorization vs. generalization not disentangled

Finding #2 and Table 1 provide direct evidence: contaminated models score ~0% on rephrased/perturbed problems, confirming gains are verbatim memorization with no transfer. We will expand Table 1 to show all model sizes (currently only 344M displayed) and add cross-entropy.

To decompose the SFT effect (Finding #5), we are currently evaluating the SFT'd models on the rephrased and the perturbed MATH test sets. We should have results by tomorrow or the day after, and will post a comment with numbers.
Weakness 2: Single-replica result depends on pretraining data mix

Fair point. The specific $E(R=0) = 3.594$ is corpus-dependent. But the qualitative finding is robust: the contaminated corpus contains test solutions; the uncontaminated does not. No compute on a corpus lacking answers can produce them: an information-theoretic advantage, not a data-mix artifact. The threshold may shift with corpus quality, but the phenomenon holds whenever the test set is distributionally distinguishable from pretraining data.
Question: Do overtraining results conflate dilution and forgetting?

Yes — separating them would require overtraining by repeating the contaminated corpus vs. adding fresh data. Two observations favor dilution:

    The crossover point shifts smoothly with model size (32 replicas at 34M → 10 at 63M → 1 at 93M) — regular scaling unlikely from catastrophic forgetting.
    The dose-response framework (Wei et al. 2025; Schaeffer et al., 2025) provides theoretical grounding.

The practical implication is the same under either mechanism.
Question: Implications beyond evaluations?

    Privacy: Single-exposure memorization produces detectable aggregate effects (Finding #3) even when MIAs achieve limited success (AUC < 0.7; Hayes et al., NeurIPS 2025).
    Alignment: If safety-relevant behaviors are memorized, the survival process (Finding #8) predicts brittleness under temperature perturbation or long sequences.
    Benchmark design: Longer solutions = more resistant to contamination — a concrete design principle.

Replying to Rebuttal by Authors
Rebuttal Acknowledgement by Reviewer 6RQA
Rebuttal Acknowledgementby Reviewer 6RQA03 Apr 2026, 11:35Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Acknowledgement: (b) Partially resolved - I have follow-up questions for the authors.
Reasons:

On weakness 1: please correct me if I'm mistaken, but it seems that in order to draw conclusions from the perturbed results, we would need to compare against the model's performance on unperturbed problems, and to see the difference between a model that has memorized vs. not. The presented results do not rule out the following case, for example: the model truly generalized on the non-perturbed problems, but failed to generalize to perturbed problems on account of the distribution shift induced by perturbation.

On weakness 2: I still do not understand the relevance of the result. If the data mix contained no math, then it's unsurprising that adding math to the mix would result in lower loss on math (memorization or not). If the data mix contained highly relevant (but o.o.d.) math, it's plausible that this would be sufficient to generalize to the test set, and the gap would be closed.
Replying to Rebuttal Acknowledgement by Reviewer 6RQA
Reply Rebuttal Comment by Authors
Reply Rebuttal Commentby Authors (Brando Miranda, Fazl Barez, Ken Liu, Rylan Schaeffer, +7 more)07 Apr 2026, 00:11Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Comment:

Dear Reviewer 6RQA,

Thank you for your precise follow-up questions. We now have three new experiments that we believe resolve your remaining concerns.
On Weakness 1 (memorization vs. generalization):

Your concern is well-taken: the Table 1 results alone cannot distinguish "the model truly generalized on unperturbed problems but failed on perturbed ones due to distribution shift" from "the model only memorized." We have now run two experiments that address this directly.

    Experiment A (Teacher-forced NLL, pre-SFT vs. post-SFT on perturbed MATH): We evaluated all SFT checkpoints on perturbed MATH problems that appeared in neither pretraining nor SFT (34 runs across 153M and 344M, sweep onaspopu). SFT lowers NLL on these novel problems at 14/17 (model, contamination level) pairs. Improvements range from -0.29 to -4.72 nats, with the largest gains at high contamination (e.g., 344M at R=316: NLL drops from 6.24 to 1.52). Crucially, pre-SFT NLL actually increases at high R, meaning heavy contamination actively hurts performance on novel problems before SFT corrects it. This is generalization by definition: the model assigns higher probability to correct solutions of problems it has never seen.

    Experiment B (Generative eval on perturbed and rephrased MATH): We ran 4-shot greedy evaluation of all 344M pretrained models on rephrased and perturbed MATH (sweeps w8j3qnru, 25xeednq). Math Verify scores are approximately 0% across all conditions and contamination levels, extending Table 1 and confirming that pretrained contaminated models have zero transfer to modified problems.

Together these experiments complete the decomposition you identified as the key gap: (1) contamination-driven gains on the original test set are pure memorization (Table 1 and Experiment B), and (2) SFT on the train set produces genuine generalization to novel problems (Experiment A). This is exactly the disentanglement you described as "the difference between a borderline and an excellent contribution."
On Weakness 2 (single-replica result depends on data mix):

We want to clarify the claim more precisely. The uncontaminated corpus (FineWeb-Edu-Dedup) does not contain the solutions to MATH test problems. No amount of compute spent on this corpus can generate information it does not contain. Even a single replica introduces this information, creating a gap that is structural, not an artifact of the specific mix. You are correct that if the corpus already contained highly relevant math data, the gap would shrink. But in any realistic setting where the test set is distributionally distinguishable from pretraining data, the phenomenon holds. We will revise the text to state this qualification explicitly.

We hope these new results address your primary concerns. If so, we would be grateful for a reconsideration of your score.
Official Review of Submission2433 by Reviewer Mmea
Official Reviewby Reviewer Mmea11 Mar 2026, 10:36 (modified: 09 Apr 2026, 05:58)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer MmeaRevisions
Summary:

The paper focus on the impact of test set contamination on generative evaluations. Through controlled experiments, the authors pretrain causal language models (ranging from 34M to 344M parameters) on corpora containing varying numbers of replicas of the MATH benchmark test set. They characterize how contamination inflates generative metrics through brittle memorization, showing that this memorization can be disrupted by overtraining, supervised finetuning (SFT), or high sampling temperatures at inference. Additionally, the authors identify and patch a critical scoring bug in the EleutherAI LM Evaluation Harness regarding the Math Verify metric.
Strengths And Weaknesses:

Strengths:

The experimental design cleanly isolates the effect of contamination by holding the total pretraining token budget constant while injecting exact, logarithmically scaled doses of the test set. The categorization of memorization into three distinct regimes (Exponentially Fast Decoherence, Brittle Memorization, Deterministic Lock-In) based on sequence length and irreducible error offers a neat analytical framework. Identifying and fixing the evaluation bug in the widely used EleutherAI Harness is a concrete and highly valuable contribution to the open-source evaluation ecosystem.

Weaknesses:

The experiments are restricted to models with up to 344M parameters evaluated on the MATH benchmark. A 344M parameter model fundamentally lacks the capacity to generalize or perform multi-step reasoning on competition-level mathematics. Therefore, any performance improvement on MATH is trivially and exclusively attributable to verbatim memorization. This severe limitation undermines the broader claims made about the interplay between reasoning and memorization in modern LLMs.

In Finding #5, the authors claim that SFT hurts test performance for highly contaminated models because the models "learn to generalize, but also forget their contaminated pretraining data." Given the 344M scale, the model cannot meaningfully learn to generalize on MATH. The performance drop is simply catastrophic forgetting of the memorized test set without any genuine generalization to fall back on. The claimed tension between generalization and memorization is unproven here.

The empirical findings rely entirely on a single benchmark (MATH). It is unclear if the specific decoherence rates or scaling law parameters apply to other generative tasks, such as code generation (e.g., HumanEval) or logical reasoning, where the sequence structures differ drastically.
Soundness: 2: fair
Presentation: 4: excellent
Significance: 2: fair
Originality: 3: good
Key Questions For Authors:

How do you justify studying the trade-off between generalization and memorization on the MATH benchmark using a 344M parameter model, which inherently lacks the capacity to generalize on this specific task?

If the uncontaminated baseline accuracy of a 344M model on MATH is effectively zero, doesn't that make the conclusion of Finding #2 (that performance gains are pure memorization) a trivial consequence of the experimental design rather than a deep insight into LLM behavior?

Have you validated the three memorization regimes (Finding #8) on a different generative benchmark (e.g., GSM8K or a coding task) to ensure these dynamics are not artifacts of the MATH dataset's specific formatting or token distribution?
Limitations:

yes
Overall Recommendation: 3: Weak reject: A paper with clear merits, but also some weaknesses, which overall outweigh the merits. Papers in this category require revisions before they can be meaningfully built upon by others. Please use sparingly.
Confidence: 5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Final Justification:

I maintain my score of 3 (weak reject). The paper studies how test set contamination inflates generative evaluation metrics through controlled pretraining experiments. While the experimental design is careful and the findings about temperature sensitivity, solution length decay, and overtraining mitigation are interesting, the fundamental limitation is that all experiments use 344M parameter models that cannot genuinely reason. The authors argue this is an advantage for isolating contamination signals, but it means the findings may not transfer to the regime where contamination and genuine capability interact. The scaling law extrapolations provide some bridge but remain speculative. The other reviewers share similar concerns, with one giving a reject and two others selecting "partially resolved or unresolved" for their rebuttal acknowledgements. I appreciate the authors' efforts in the rebuttal but believe the paper needs experiments at a scale where models can actually solve some problems to make the contributions convincing for ICML.
Rebuttal by Authors
Rebuttalby Authors (Brando Miranda, Fazl Barez, Ken Liu, Rylan Schaeffer, +7 more)30 Mar 2026, 19:43 (modified: 31 Mar 2026, 05:50)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, AuthorsRevisions
Rebuttal:

We thank the reviewer for their substantive engagement and recognition of our experimental design, three-regime framework, and evaluation bug fix.
Framing: What is the paper's contribution?

The reviewer reads our paper as studying the boundary between reasoning and memorization. That is not the core contribution. We characterize the mechanics of how contamination inflates generative evaluation metrics: temperature sensitivity (Finding #6), solution-length decay (Finding #7), the survival process (Finding #8), overtraining mitigation (Finding #4), and the SFT interaction (Finding #5). These are about how memorization behaves during sequential generation — dynamics absent from discriminative evaluations.

Under this framing, the small scale has advantages: because these models cannot genuinely solve competition math, the contamination signal is cleanly isolated. At larger scales, contamination and genuine reasoning would be confounded.
Weakness 1: Model scale (344M) is too small

We agree 344M cannot do multi-step competition math reasoning. See framing above.

Two responses:

    Controlled contamination requires pretraining from scratch. Bordt et al. (2025) reaches ~1.6B by contaminating disjoint test subsets at different replica counts within a single run; this is cheaper but experimentally messier. We pretrain separate models per contamination level, which is significantly more expensive and limits our scale but provides tighter experimental control.

    Scaling laws (Finding #3) bridge to larger scales. Parameters vary smoothly across model sizes; average fitting error is $< 10^{-2}$ for all $R$. These make falsifiable predictions beyond our experimental range.

Even if gains are "trivially" memorization, the dynamics are not trivial: temperature > 0.6 disrupts it, solution length creates an exponential barrier, one replica beats irreducible error, overtraining washes it out predictably. None obvious a priori.
Weakness 2: SFT finding is catastrophic forgetting

We partially agree; at 344M, the mechanism is likely catastrophic forgetting rather than a genuine tension.

The important finding is the asymmetry: SFT improves performance at low contamination ($R < 10$) and degrades it at high contamination ($R > 10$). This is non-obvious and practically informative regardless of mechanism. We will revise the language accordingly.

We are running a new experiment on the SFT checkpoints on our rephrased and perturbed MATH test sets. We should have results by tomorrow (maybe the day after) and will post a comment then with more information.
Weakness 3: Single benchmark (MATH)

We acknowledge this. Three reasons the findings likely generalize:

    Temperature and solution length are properties of generation, not MATH. The logic (each token = opportunity to deviate from memorized path) applies to any generative benchmark.

    The survival process framework derives from general in-context scaling laws, not MATH-specific patterns. The regimes depend on $E$ and $\alpha$, measurable for any task.

    MATH is among the most widely used generative benchmarks — a natural first target. Extension to code/reasoning is valuable future work.

Question 1: Justify studying gen vs. mem at 344M?

See framing above. We study contamination mechanics, not the gen/mem boundary. 344M provides clean isolation at tractable cost.
Question 2: Isn't Finding #2 trivially true?

Finding #2 is not "gains are memorization." It shows contamination-driven performance is brittle: rephrased/perturbed problems collapse to ~0% across all contamination levels (Table 1 reports 344M; consistent results across all model sizes are noted in the caption). This brittleness is an empirically useful diagnostic at any scale — if a 70B model scores well on MATH, one wants to know whether scores survive rephrasing. Our 344M results establish the baseline pattern under pure memorization.
Question 3: Validated three regimes on another benchmark?

Not yet. The regimes are mathematical consequences of Equation 3, not patterns fit to MATH. They depend on whether $E > 0$, $\alpha \leq 1$, or $\alpha > 1$, and are measurable for any generative task. This is a testable prediction for future work.
Replying to Rebuttal by Authors
Rebuttal Acknowledgement by Reviewer Mmea
Rebuttal Acknowledgementby Reviewer Mmea05 Apr 2026, 01:31Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Acknowledgement: (b) Partially resolved - I have follow-up questions for the authors.
Reasons:

Thank you for the rebuttal. I appreciate the clarification on the paper's framing as studying contamination mechanics during sequential generation rather than the reasoning-memorization boundary. The argument that small scale provides cleaner experimental control for isolating contamination signals is reasonable. However, my core concern remains that the findings at 344M may not transfer meaningfully to models that can actually reason, where contamination and genuine capability would interact in qualitatively different ways. The scaling laws provide some bridge but are extrapolations. I also note that other reviewers share similar concerns about scale and novelty. I maintain my score of 3 (weak reject) but acknowledge the authors made a good faith effort to address the feedback.
Replying to Rebuttal Acknowledgement by Reviewer Mmea
Reply Rebuttal Comment by Authors
Reply Rebuttal Commentby Authors (Brando Miranda, Fazl Barez, Ken Liu, Rylan Schaeffer, +7 more)07 Apr 2026, 00:12Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Comment:

Dear Reviewer Mmea,

Thank you for your continued engagement. We appreciate your acknowledgement of the reframing and the reasonableness of small-scale control. We want to share three new experimental results that we believe substantively address your remaining concerns.

On model capacity: You argued that 344M models "fundamentally lack the capacity to generalize or perform multi-step reasoning on competition-level mathematics." We tested this directly. We sampled 1,000 completions per MATH problem at temperature 1.0 with 4-shot prompting from the uncontaminated 344M model (808,797 samples scored so far). The result: pass@1 = pass@10 = pass@100 = 0.000%. Zero correct solutions out of over 800,000 attempts. You are right that 344M models cannot do MATH.

However, this result cuts in our favor, not against us. It means that any nonzero performance from contaminated models is definitively memorization, not latent capability being unlocked. There is no confound between contamination and genuine reasoning at this scale, which is precisely the experimental advantage of studying small models that we described in our rebuttal.

SFT is not simply catastrophic forgetting. We evaluated pre-SFT vs. post-SFT models (153M and 344M, all contamination levels) on perturbed MATH problems that appeared in neither pretraining nor SFT (34 runs, sweep onaspopu). SFT lowers NLL on these novel problems at 14/17 (model, R) pairs, with improvements up to -4.72 nats. The effect is largest at high contamination (344M at R=316: NLL drops from 6.24 to 1.52), precisely the regime where your review predicted pure forgetting. This directly contradicts the catastrophic forgetting explanation. If SFT were only erasing memorized data with no compensating learning, NLL on novel problems would not improve. The fact that it does, and most dramatically at high contamination, demonstrates that SFT induces genuine transfer to unseen problems. The model learns something about mathematical structure from the training set that generalizes to novel problems, even though it cannot solve them from scratch (as the pass@k result confirms).

On transferability to larger models: We agree that extrapolation carries uncertainty. But the dynamics we characterize (temperature sensitivity, solution-length decay, the survival process) arise from the sequential nature of generation, not from model scale or the MATH dataset specifically. The new generalization result strengthens the case that our findings capture real learning dynamics rather than scale-specific artifacts. We also note that the pass@k result provides a clean baseline that would be confounded at larger scales where models have nonzero genuine capability on MATH.

We hope these results address your core concerns. We believe the combination of (a) confirmed zero baseline capability, (b) demonstrated genuine generalization from SFT, and (c) the sequential generation dynamics that are absent from prior discriminative work constitutes a meaningful contribution. We would be grateful for a reconsideration of your score.
Official Review of Submission2433 by Reviewer 4xWn
Official Reviewby Reviewer 4xWn20 Feb 2026, 01:46 (modified: 09 Apr 2026, 22:12)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer 4xWnRevisions
Summary:

This paper investigates how language models contaminated with benchmark data at pre-training perform on the actual test set
Strengths And Weaknesses:
Strengths

Soundness

    The paper conducts experiments over many experimental dimensions
    The paper raises relevant research questions

Presentation

    Experiments are clearly separated, making it easy to follow

Significance

    This work is timely and tackle an important research question

Originality

    New experimental result, corresponding to Findings #3 that contradicts other existing work.

Weaknesses

Soundness

    While the scaling laws that have been followed are clearly disclosed, it would help for the overall appreciation of this work to provide more extensive experimental details like number of documents, maximal length, number of optimization steps etc.
    The experiments have been conducted to a scale of up to 340M models. This is very small compared to other works or current standards in LLMs research. This should have been much more clearly emphasized since it probably changes the overall positioning of this work.
    Some claims are not well founded. Below is a sample:
        "which offer a more rigorous measurement" L.024 is not justified
        "While foundational... have focused on discriminative benchmarks..." L.032 this is not true. For instance, Kocygit et al 2025. tackles machine translation.
    Some claims are misleading
        Findings #8. The authors propose a modelization of the probability to generate a good answer. But this modelization is never justified. Moreover they state that they "mathematically discover" L.081 those rules, this is obviously misleading.
        They position their work as being generative versus other that are more discriminative. But the actual experiments are on mathematical datasets where the answer is very short, which does contradict the overall positioning of the paper "produce tens-to-thousands of tokens" L. 080.

Presentation

    While easy to follow, the paper feels a bit like a giant listing of experiments, with no clear logical flow between sections.

Originality

    While I appreciate the experiments made by the authors, I find this work to be rather a replica of more complete work, like Bordt et al.

Soundness: 2: fair
Presentation: 2: fair
Significance: 2: fair
Originality: 1: poor
Key Questions For Authors:

See Strengths / weaknesses.
Limitations:

Yes
Overall Recommendation: 2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility, incompletely addressed ethical considerations, or writing so poor that it is not possible to understand its key claims.
Confidence: 5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Final Justification:

I still have serious concerns regarding the validity of this work. I am now convinced that the authors mainly reported memorization capacity of models rather than 'reasoning capabilities', as evidenced by their baseline reported a 0% accuracy.
Rebuttal by Authors
Rebuttalby Authors (Brando Miranda, Fazl Barez, Ken Liu, Rylan Schaeffer, +7 more)29 Mar 2026, 13:49 (modified: 31 Mar 2026, 05:50)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, AuthorsRevisions
Rebuttal:

We thank the reviewer for recognizing the timeliness and relevance of the research questions.
Soundness 1: Missing experimental details

These details — number of documents, max sequence length, optimization steps, batch size schedule, optimizer hyperparameters — are reported in Appendix B Pretraining Implementation Details. We will add a summary table to the main body in revision.
Soundness 2: Model scale (344M) is small

We acknowledge this limitation. Two considerations:

    Controlled contamination requires pretraining from scratch. This constraint applies to all prior work. Bordt et al. uses models up to ~1.6B, but achieves this by contaminating disjoint subsets of the test set at different replica counts within a single training run — a cheaper but less controlled design. We pretrain separate models per contamination level for tighter experimental control, which is more expensive and limits our maximum scale.

    Our scaling law analysis (Finding #3) extrapolates beyond experimental scale. Fitting $\mathcal{L}(C, R) = E(R) + C_0(R) \cdot C^{-\alpha(R)}$ provides falsifiable predictions at larger compute budgets. This is a core contribution, not a post-hoc rationalization.

We will make the scale limitation more prominent in revision.
Soundness 3a: "More rigorous measurement" not justified

We accept this phrasing was imprecise. Controlled contamination studies enable causal measurement of contamination effects, in contrast to observational approaches that can only establish correlations. We will revise to "which enable the most direct causal measurement."
Soundness 3b: Prior work claim ignores Kocyigit et al. 2025

We already cite Kocyigit et al. (2025) in the introduction (paragraph 2) and Appendix A Related Work. Our claim is that the literature has "predominantly focused on" discriminative benchmarks (the paper's exact words) — not exclusively. Kocyigit et al. is the sole controlled contamination study that we are aware of on a purely generative task. One exception out of dozens of papers reinforces, rather than contradicts, how underexplored the generative setting remains.
Soundness 4a: Finding #8 "never justified"; "mathematically discover" is misleading

The survival process model is derived from the in-context scaling law (Equation 3): per-token NLL $\ell_t \sim E + A \cdot t^{-\alpha}$, survival probability $P(T) = \prod_{t=1}^{T} p_t$ (Equation 4), three regimes from the asymptotic behavior depending on $E$ and $\alpha$. We will make this derivation more explicit in revision.

We agree that "mathematically discover" overstates the contribution. We will revise to "mathematically characterize."
Soundness 4b: MATH answers are short

The reviewer conflates final answers with full solutions. MATH requires generating complete chain-of-thought solutions; ground-truth solution lengths range from 15 to 1,949 tokens. The "tens-to-thousands of tokens" framing is empirically correct.
Presentation: listing of experiments

The paper follows the model lifecycle: pretraining (Section 3) → post-training (Section 4) → inference (Section 5), each building on the previous. Three of four reviewers rated presentation 4/4; THKB called the findings "coherent and progressive." We will strengthen transition paragraphs in revision, but would appreciate some more clarity on why you feel that our work lacks logical flow.
Originality: "Replica of Bordt et al."

Bordt et al. studies 7 MCQA benchmarks (discriminative). We study generative evaluation. This is not cosmetic — it is the entire motivation (Section 1, paragraphs 3–4).

The generative setting introduces dynamics absent from discriminative evaluation: temperature (Finding #6), solution length as an exponential barrier (Finding #7), the survival process framework (Finding #8). None have analogs in Bordt et al. Finding #3 — which the reviewer themselves identifies as original — contradicts conclusions from prior discriminative work.

Our paper partially originated from correspondence with the Bordt et al. authors about how the field lacks a characterization of contamination in generative evaluations.

We hope the above addresses your primary concerns and would welcome the opportunity to discuss further during the discussion phase. If you feel your concerns have been adequately addressed, we would appreciate a reconsideration of the score.
Replying to Rebuttal by Authors
Rebuttal Acknowledgement by Reviewer 4xWn
Rebuttal Acknowledgementby Reviewer 4xWn01 Apr 2026, 18:29Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Acknowledgement: (c) Partially resolved or unresolved, but the remaining concerns are not easily addressed in a short rebuttal - Please select this option sparingly and only when you believe that your questions concern the core tenets of the work, and addressing them requires a significant update to the paper.
Reasons:

I thank the authors for their answer. But I have some concerns regarding the quality of the rebuttal.

    These details — number of documents, max sequence length, optimization steps, batch size schedule, optimizer hyperparameters — are reported in Appendix B Pretraining Implementation Details.

This is not true. There are no such information in Appendix B. There are some hyperparameters but not those I asked for.

    provides falsifiable predictions at larger compute budgets. This is a core contribution, not a post-hoc rationalization.

How can you be so sure if you have not tested with larger models?

    The reviewer conflates final answers with full solutions. MATH requires generating complete chain-of-thought solutions; ground-truth solution lengths range from 15 to 1,949 tokens. The "tens-to-thousands of tokens" framing is empirically correct.

I acknowledge an oversight on this part.

My concern regarding model size remain. I definitely share the concern of reviewer Mmea and others regarding memorization. I don't think the answers resolve this concern.
Replying to Rebuttal Acknowledgement by Reviewer 4xWn
Reply Rebuttal Comment by Authors
Reply Rebuttal Commentby Authors (Brando Miranda, Fazl Barez, Ken Liu, Rylan Schaeffer, +7 more)07 Apr 2026, 00:14Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Comment:

Dear Reviewer 4xWn,

Thank you for your response. We want to address your remaining points directly and share new experimental evidence.

On Appendix B: You state that the information we listed is not in Appendix B. We respectfully want to resolve this factual question. Appendix B (page 29 of the submission) reports: model architecture configurations including number of layers and hidden sizes (Table 2), the optimizer and its hyperparameters (AdamW, beta_1=0.9, beta_2=0.999, epsilon=1e-8, weight decay=0), learning rate schedule (linear warmup for 250 steps, cosine annealing to zero), batch size scaling formula (Equation 5, tokens per optimizer step = 3.24e3 * D^0.264, following Shuai et al. 2024), gradient clipping (max norm 1.0), maximum sequence length (2048 tokens), precision (bfloat16), and the complete data mixing procedure. The number of optimization steps is deterministically recoverable from the batch size formula and the total token budget (20 tokens per parameter). We will add a summary table with explicit step counts and document counts in revision to make this immediately accessible.

On scaling laws and larger models: Scaling laws are, by construction, predictive tools whose value lies in making falsifiable predictions beyond the experimental range. This is standard practice in the scaling laws literature (Kaplan et al. 2020, Hoffmann et al. 2022). Our fits achieve average error below 0.01 across all contamination levels. We do not claim certainty about extrapolation. We claim these are the best available quantitative predictions, and they are testable by future work.

On originality: Bordt et al. studies seven MCQA benchmarks (discriminative). Findings #6, #7, and #8 (temperature sensitivity, solution-length decay, the survival process) have no analogs in Bordt et al. because they arise from sequential token generation, which does not occur in multiple-choice settings. The reviewer acknowledged Finding #3 as original and as contradicting existing work.

New experimental evidence: We have completed three additional experiments during the rebuttal period. (1) Pass@k evaluation (1,000 samples per problem, temperature 1.0) confirms that the uncontaminated 344M model achieves exactly 0% across 808,797 samples, establishing a rigorous zero-capability baseline. (2) Teacher-forced evaluation of SFT checkpoints on perturbed MATH (novel problems never seen during pretraining or SFT) shows SFT lowers NLL at 14/17 conditions, with improvements up to -4.72 nats, demonstrating genuine generalization. (3) Generative evaluation of all 344M pretrained models on rephrased and perturbed MATH confirms 0% transfer across all contamination levels.

We believe a paper that (a) introduces an underexplored evaluation modality (generative contamination), (b) discovers dynamics absent from prior work, (c) contradicts established conclusions, and (d) now demonstrates genuine generalization from SFT on novel problems constitutes a distinct contribution.

We hope this clarifies the remaining points and would be grateful for a reconsideration.
Factual Concerns Regarding Review by Reviewer 4xWn
Author AC Confidential Commentby Authors (Brando Miranda, Fazl Barez, Ken Liu, Rylan Schaeffer, +7 more)07 Apr 2026, 00:18Program Chairs, Senior Area Chairs, Area Chairs, Authors
Comment:

Dear Area Chair,

We are writing to respectfully flag concerns about the reliability of Reviewer 4xWn's assessment, which we believe may affect your calibration of the reviews for this submission.

We want to highlight three specific issues.

    Demonstrably incorrect factual claim, maintained across two rounds. In their review, Reviewer 4xWn requested "number of documents, maximal length, number of optimization steps, etc." In our rebuttal, we noted that these details are reported in Appendix B. The reviewer responded: "This is not true. There are no such information in Appendix B." We invite the AC to examine page 29 (Appendix B, "Pretraining Implementation Details"), which reports: model architecture configurations (Table 2), optimizer hyperparameters (AdamW, beta_1=0.9, beta_2=0.999, epsilon=1e-8), learning rate schedule (250-step warmup, cosine annealing), batch size scaling formula (Equation 5), gradient clipping (max norm 1.0), maximum sequence length (2048 tokens), precision (bfloat16), and the complete data mixing procedure. The number of optimization steps is deterministically recoverable from the batch size formula and token budget. We acknowledge that an explicit step-count table would improve accessibility, but the claim that this information is absent is factually incorrect.

    Internal inconsistency between stated strengths and scores. The reviewer's own Strengths section identifies Finding #3 as a "new experimental result, corresponding to Findings #3 that contradicts other existing work." Despite this, the reviewer assigned an originality score of 1 (poor). We find it difficult to reconcile acknowledging a novel result that contradicts prior work with the lowest possible originality rating.

    Confidence 5 with acknowledged errors. The reviewer rated confidence 5 ("absolutely certain... checked the math/other details carefully"). During the rebuttal, the reviewer acknowledged an "oversight" regarding solution lengths (conflating short final answers with full chain-of-thought solutions spanning 15 to 1,949 tokens), a point central to the paper's positioning. Combined with the incorrect Appendix B claim, we believe the confidence rating may not reflect the depth of engagement with the paper.

We want to emphasize that we take all reviewer feedback seriously. The concerns about model scale are legitimate and shared by other reviewers; we have addressed them with new experiments (SFT generalization on perturbed problems, pass@k evaluation). Our concern is specifically about the reliability of this review as a basis for scoring, given the factual errors and internal inconsistencies described above.

We are happy to address any further questions from the AC.
Official Review of Submission2433 by Reviewer THKB
Official Reviewby Reviewer THKB17 Feb 2026, 20:29 (modified: 10 Apr 2026, 10:33)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer THKBRevisions
Summary:

This paper makes four contributions in quantifying the effect of test data contamination on generative benchmarks:

    assessing the effect of data contamination during pre-training on a math task (finding: contamination causes memorization, thereby bringing improvement);
    assessing the effect of data contamination during post-training on a math task (finding: contamination effects from pre-training can be diluted by post-training);
    identifying three signal regimes of memorization during inference (finding: the interplay among temperature, generation length, and memorization); and
    fixing a bug in Evaluation Harness.

Strengths And Weaknesses:
Strengths

    This paper traces the effect of test data contamination along the language model lifecycle, covering pre-training, post-training, inference, and deployment, which comprehensively reveals its causal effects and possible detection and mitigation practices.
    The paper is clearly written and easy to read.
    The findings are coherent and progressive, each of which is supported by detailed results from a series of controlled experiments.

Weaknesses

    I am not sure if the conclusions in this paper can generalize to wider scopes of generative evaluations. For example, the models deployed in the real world are much larger than the largest model studied in this paper. The distributions of training and test data are also an influential confounder in the experiments, which may compromise the external validity of the study.
    While the first seven findings are well-elaborated on, the last finding with the three regimes for memorization detection reads understandable but a little compacted, as some terms (e.g., survival process, decoherence, and lock-in) and symbols could be further clarified or simplified.

Soundness: 4: excellent
Presentation: 4: excellent
Significance: 3: good
Originality: 3: good
Key Questions For Authors:

    Could you provide more evidence of the findings' external validity? For example, how might we apply these findings to each stage of real-world generative model development to inform relevant practice?
    Could these findings generalize to larger models pre-trained on mixed online corpora, and what are the main differences between realistic settings and the controlled setting in this paper?

Limitations:

The authors primarily discuss the positive social impact of their work in their impact statement and place the limitations in the appendix. From my perspective, the proposed regimes of memorization also warrant further scrutiny, as these proxies may be used for detecting data contamination, and any inaccuracy in these measures could potentially undermine trust in the AI community.
Overall Recommendation: 4: Weak accept: Technically solid paper that advances at least one sub-area of AI, with a contribution that others are likely to build on, but with some weaknesses that limit its impact (e.g., limited evaluation). Please use sparingly.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Final Justification:

I would like to keep my score for weak accept.
Rebuttal by Authors
Rebuttalby Authors (Brando Miranda, Fazl Barez, Ken Liu, Rylan Schaeffer, +7 more)29 Mar 2026, 14:13 (modified: 31 Mar 2026, 05:50)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, AuthorsRevisions
Rebuttal:

We thank the reviewer for recognizing the value of tracing contamination across the full model lifecycle.
Weakness 1: Generalizability to larger models and real-world settings

Controlled contamination requires controlling the entire corpus, which is prohibitively expensive for billion-scale-parameter models. This objection applies to almost all modern academic work on language models.

While our experiments are conducted on small models, our findings have practical implications at each lifecycle stage regardless of scale:

    Detection: Temperature sweeps (Finding #6) and solution-length stratification (Finding #7) can provide lightweight contamination diagnostics.
    Mitigation: Overtraining on fresh data dilutes contamination (Finding #4).
    Risk assessment: The survival process framework (Finding #8) provides a vocabulary for assessing benchmark vulnerability.
    Extrapolation: Scaling laws (Finding #3) bridge our findings to larger compute budgets.

Weakness 2: Finding #8 is underexplained

We will expand the exposition. Briefly:

    Survival process: The probability of generating a correct solution of length $T$ is the product of per-token success probabilities. Each token is an opportunity for the memorized sequence to "die."
    Decoherence (Regime I): $E > 0$ $\Rightarrow$ errors accumulate $\Rightarrow$ memorization is lost.
    Lock-in (Regime III): $E \approx 0$, $\alpha > 1$ $\Rightarrow$ survival probability converges to a positive constant $\Rightarrow$ memorization persists.
    Brittle memorization (Regime II): $E \approx 0$, $\alpha \leq 1$ $\Rightarrow$ stretched exponential survival probability decay $\Rightarrow$ memorization is fragile at long lengths.

Question 1: External validity?

See Weakness 1 response above.
Question 2: Generalize to larger models on mixed corpora?

Key differences from realistic settings: (1) scale (344M vs. billions), (2) corpus composition (single web crawl vs. heterogeneous mixtures), (3) contamination mechanism (exact replicas vs. near-duplicates). Scaling laws address (1). Points (2) and (3) are genuine limitations — qualitative findings should hold, but specific thresholds may shift. We will discuss this more explicitly.
Limitations placement and detection proxy risks

The reviewer notes we "place the limitations in the appendix." Our limitations are in fact in the main body (Section 6 Discussion, dedicated paragraph). The reviewer may have been looking for a standalone section heading; we will make it more visually prominent.

We appreciate the concern that the three regimes could be misused as detection proxies. They describe idealized asymptotic dynamics — tendencies, not sharp boundaries. We will add a cautionary note that they should be validated empirically before being used as detection criteria.
On originality: connection to membership inference attacks

As we discuss in Sections 3 and 6, our findings sit in tension with Hayes et al. (NeurIPS 2025), who scale LiRA to 1B parameters with 128 reference models and find AUC < 0.7 — limited success at individual sample detection.

In contrast, we find that even a single test set replica produces stark shifts in aggregate evaluation dynamics: cross-entropy below irreducible error (Finding #3), divergent temperature sensitivity (Finding #6), qualitative regime shifts (Finding #8).

Many details differ between setups, so we don't claim a causal explanation. But the contrast highlights that contamination can have large effects on generative evaluation metrics even when elusive to detect at the sample level. This juxtaposition underscores a puzzle the field needs to grapple with.

We again thank the reviewer for assessing our work so thoroughly. We welcome further discussion about how to strengthen the paper. If we have addressed your primary concerns, we hope that you will reassess your score.
Replying to Rebuttal by Authors
Rebuttal Acknowledgement by Reviewer THKB
Rebuttal Acknowledgementby Reviewer THKB02 Apr 2026, 07:22 (modified: 02 Apr 2026, 07:22)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, AuthorsRevisions
Acknowledgement: (c) Partially resolved or unresolved, but the remaining concerns are not easily addressed in a short rebuttal - Please select this option sparingly and only when you believe that your questions concern the core tenets of the work, and addressing them requires a significant update to the paper.
Reasons:

I still have slight concerns regarding model size and generalizability. Thanks for the response.
Replying to Rebuttal Acknowledgement by Reviewer THKB
Reply Rebuttal Comment by Authors
Reply Rebuttal Commentby Authors (Brando Miranda, Fazl Barez, Ken Liu, Rylan Schaeffer, +7 more)07 Apr 2026, 00:14Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Comment:

Dear Reviewer THKB,

Thank you for your continued engagement. We wanted to share new results that address your remaining concerns about model size and generalizability.

We completed three experiments during the rebuttal period:

First, we confirmed via pass@k sampling (1,000 samples per problem, 808,797 total) that the uncontaminated 344M model scores exactly 0% on MATH, establishing a rigorous zero-capability baseline. This means any performance gains we observe from contamination are unambiguously attributable to memorization, with no confound from latent reasoning ability.

Second, and most importantly for your generalizability concern: we evaluated pre-SFT vs. post-SFT models on perturbed MATH problems never seen during pretraining or SFT. SFT lowers NLL on these novel problems at 14/17 (model, contamination level) pairs, with improvements up to -4.72 nats at high contamination. This demonstrates that even at 344M, the models acquire transferable mathematical structure through SFT. The generalization mechanism (learning from training-set structure and transferring to novel problems) is the same mechanism that operates at larger scales, providing evidence that our findings capture real dynamics rather than scale-limited artifacts.

Third, we evaluated all 344M pretrained models on rephrased and perturbed MATH test sets, confirming 0% generative transfer across all contamination levels, extending the Table 1 results.

We have also expanded the exposition of Finding #8 (the three memorization regimes) per your suggestion and added cautionary notes about using these as detection proxies.

If you feel these additions address your remaining concerns, we would be grateful for a reconsideration of your score.

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

