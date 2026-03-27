# ICML 2026 Reviews — Submission 2433

**Title:** Quantifying the Effect of Test Set Contamination on Generative Evaluations

**Authors:** Rylan Schaeffer, Joshua Kazdan, Baber Abbasi, Ken Liu, Brando Miranda, Ahmed M Ahmed, Fazl Barez, Abhay Puri, Stella Biderman, Niloofar Mireshghallah, Sanmi Koyejo

**Submitted:** 09 Jan 2026 (modified: 09 Feb 2026)

---

## Reviewer 6RQA — Weak Accept (4)

**Confidence:** 3 (fairly confident)

### Summary

The paper analyzes the effect of including math problems in pretraining on the final model's performance on those problems. It performs systematic experiments across a range of model sizes and number of copies of the data, presenting evidence that even a single copy of the data can cause memorization. The results demonstrate the potential for widespread contamination of generative evaluations of large language models. A variety of follow-up experiments clarify the dynamics of this phenomenon, e.g. dependence on sampling temperature and interaction with SFT.

### Strengths

The paper is well-motivated by concerns about the validity of generative evaluations. The writing and presentation is very clear. The experiments are thorough and informative-- it is particularly interesting that SFT on the train set has a reversed effect based on the amount of contamination. The findings underscore the importance of considering the threat that contamination poses to the evaluation of large language models.

### Weaknesses

1. The difference between memorization and generalization is central to the paper, but not adequately explored. For example, the authors write: "We conjecture that during SFT, contaminated models learn to generalize but also forget their contaminated pretraining data, and the effects of contamination are more potent than generalization for small models, leading to a net increase in test loss." This need not be a conjecture. The test set could be split into train-test and validation-test sets and generalization vs. memorization could be quantified. Validation-test loss could be reported in many cases alongside train-test loss. Alternatively, performance could be reported on perturbed problems. On a related note, insufficient detail is given on the results in Table 1. In my view, this weakness is significant, and represents the difference between a borderline contribution and an excellent contribution.

2. Regarding the following: "including even a single replica enables almost all models to achieve lower cross entropy losses than the estimated irreducible error of the uncontaminated pretraining corpus." The paper overstates the significance of this result-- the result depends strongly on the pretraining data mix, and the pretraining mix is never varied, so it is unclear whether the result is generalizable.

3. As the authors acknowledge, only limited conclusions can be drawn on the basis of such small models. However, the careful scaling laws analysis lends credibility to the results and allows for extrapolation to larger sizes-- so I do not see this as a major weakness.

### Soundness: 3 (good) | Presentation: 4 (excellent) | Significance: 3 (good) | Originality: 3 (good)

### Key Questions

1. My main question is just to ask for the authors response to the weaknesses proposed above.

2. Am I correct in my understanding that the overtraining results conflate the effect of diluting the data and of catastrophic forgetting? If so, how much value do you think there would be in controlling for these separately?

3. (My review does not hinge on this question) Out of curiosity, do you think there are implications of your results beyond evaluations, e.g. with regards to data efficiency, generalization, or alignment of advanced AI systems?

---

## Reviewer Mmea — Weak Reject (3)

**Confidence:** 5 (absolutely certain)

### Summary

The paper focus on the impact of test set contamination on generative evaluations. Through controlled experiments, the authors pretrain causal language models (ranging from 34M to 344M parameters) on corpora containing varying numbers of replicas of the MATH benchmark test set. They characterize how contamination inflates generative metrics through brittle memorization, showing that this memorization can be disrupted by overtraining, supervised finetuning (SFT), or high sampling temperatures at inference. Additionally, the authors identify and patch a critical scoring bug in the EleutherAI LM Evaluation Harness regarding the Math Verify metric.

### Strengths

The experimental design cleanly isolates the effect of contamination by holding the total pretraining token budget constant while injecting exact, logarithmically scaled doses of the test set. The categorization of memorization into three distinct regimes (Exponentially Fast Decoherence, Brittle Memorization, Deterministic Lock-In) based on sequence length and irreducible error offers a neat analytical framework. Identifying and fixing the evaluation bug in the widely used EleutherAI Harness is a concrete and highly valuable contribution to the open-source evaluation ecosystem.

### Weaknesses

1. The experiments are restricted to models with up to 344M parameters evaluated on the MATH benchmark. A 344M parameter model fundamentally lacks the capacity to generalize or perform multi-step reasoning on competition-level mathematics. Therefore, any performance improvement on MATH is trivially and exclusively attributable to verbatim memorization. This severe limitation undermines the broader claims made about the interplay between reasoning and memorization in modern LLMs.

2. In Finding #5, the authors claim that SFT hurts test performance for highly contaminated models because the models "learn to generalize, but also forget their contaminated pretraining data." Given the 344M scale, the model cannot meaningfully learn to generalize on MATH. The performance drop is simply catastrophic forgetting of the memorized test set without any genuine generalization to fall back on. The claimed tension between generalization and memorization is unproven here.

3. The empirical findings rely entirely on a single benchmark (MATH). It is unclear if the specific decoherence rates or scaling law parameters apply to other generative tasks, such as code generation (e.g., HumanEval) or logical reasoning, where the sequence structures differ drastically.

### Soundness: 2 (fair) | Presentation: 4 (excellent) | Significance: 2 (fair) | Originality: 3 (good)

### Key Questions

1. How do you justify studying the trade-off between generalization and memorization on the MATH benchmark using a 344M parameter model, which inherently lacks the capacity to generalize on this specific task?

2. If the uncontaminated baseline accuracy of a 344M model on MATH is effectively zero, doesn't that make the conclusion of Finding #2 (that performance gains are pure memorization) a trivial consequence of the experimental design rather than a deep insight into LLM behavior?

3. Have you validated the three memorization regimes (Finding #8) on a different generative benchmark (e.g., GSM8K or a coding task) to ensure these dynamics are not artifacts of the MATH dataset's specific formatting or token distribution?

---

## Reviewer 4xWn — Reject (2)

**Confidence:** 5 (absolutely certain)

### Summary

This paper investigates how language models contaminated with benchmark data at pre-training perform on the actual test set.

### Strengths

**Soundness:**
- The paper conducts experiments over many experimental dimensions
- The paper raises relevant research questions

**Presentation:**
- Experiments are clearly separated, making it easy to follow

**Significance:**
- This work is timely and tackle an important research question

**Originality:**
- New experimental result, corresponding to Findings #3 that contradicts other existing work.

### Weaknesses

**Soundness:**
1. While the scaling laws that have been followed are clearly disclosed, it would help for the overall appreciation of this work to provide more extensive experimental details like number of documents, maximal length, number of optimization steps etc.
2. The experiments have been conducted to a scale of up to 340M models. This is very small compared to other works or current standards in LLMs research. This should have been much more clearly emphasized since it probably changes the overall positioning of this work.
3. Some claims are not well founded. Below is a sample:
   - "which offer a more rigorous measurement" L.024 is not justified
   - "While foundational... have focused on discriminative benchmarks..." L.032 this is not true. For instance, Kocygit et al 2025 tackles machine translation.
4. Some claims are misleading:
   - Findings #8. The authors propose a modelization of the probability to generate a good answer. But this modelization is never justified. Moreover they state that they "mathematically discover" L.081 those rules, this is obviously misleading.
   - They position their work as being generative versus other that are more discriminative. But the actual experiments are on mathematical datasets where the answer is very short, which does contradict the overall positioning of the paper "produce tens-to-thousands of tokens" L.080.

**Presentation:**
- While easy to follow, the paper feels a bit like a giant listing of experiments, with no clear logical flow between sections.

**Originality:**
- While I appreciate the experiments made by the authors, I find this work to be rather a replica of more complete work, like Bordt et al.

### Soundness: 2 (fair) | Presentation: 2 (fair) | Significance: 2 (fair) | Originality: 1 (poor)

### Key Questions

See Strengths / weaknesses.

---

## Reviewer THKB — Weak Accept (4)

**Confidence:** 4 (confident)

### Summary

This paper makes four contributions in quantifying the effect of test data contamination on generative benchmarks:

1. Assessing the effect of data contamination during pre-training on a math task (finding: contamination causes memorization, thereby bringing improvement);
2. Assessing the effect of data contamination during post-training on a math task (finding: contamination effects from pre-training can be diluted by post-training);
3. Identifying three signal regimes of memorization during inference (finding: the interplay among temperature, generation length, and memorization); and
4. Fixing a bug in Evaluation Harness.

### Strengths

1. This paper traces the effect of test data contamination along the language model lifecycle, covering pre-training, post-training, inference, and deployment, which comprehensively reveals its causal effects and possible detection and mitigation practices.
2. The paper is clearly written and easy to read.
3. The findings are coherent and progressive, each of which is supported by detailed results from a series of controlled experiments.

### Weaknesses

1. I am not sure if the conclusions in this paper can generalize to wider scopes of generative evaluations. For example, the models deployed in the real world are much larger than the largest model studied in this paper. The distributions of training and test data are also an influential confounder in the experiments, which may compromise the external validity of the study.
2. While the first seven findings are well-elaborated on, the last finding with the three regimes for memorization detection reads understandable but a little compacted, as some terms (e.g., survival process, decoherence, and lock-in) and symbols could be further clarified or simplified.

### Soundness: 4 (excellent) | Presentation: 4 (excellent) | Significance: 3 (good) | Originality: 3 (good)

### Key Questions

1. Could you provide more evidence of the findings' external validity? For example, how might we apply these findings to each stage of real-world generative model development to inform relevant practice?
2. Could these findings generalize to larger models pre-trained on mixed online corpora, and what are the main differences between realistic settings and the controlled setting in this paper?

### Limitations

The authors primarily discuss the positive social impact of their work in their impact statement and place the limitations in the appendix. From my perspective, the proposed regimes of memorization also warrant further scrutiny, as these proxies may be used for detecting data contamination, and any inaccuracy in these measures could potentially undermine trust in the AI community.

---

## Summary

| Reviewer | Score | Confidence | Recommendation |
|----------|-------|------------|----------------|
| 6RQA     | 4     | 3          | Weak Accept    |
| Mmea     | 3     | 5          | Weak Reject    |
| 4xWn     | 2     | 5          | Reject         |
| THKB     | 4     | 4          | Weak Accept    |
| **Average** | **3.25** | **4.25** |             |
