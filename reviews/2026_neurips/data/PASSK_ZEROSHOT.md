# pass@k at 0-shot, uncontaminated Qwen3-344M

The existing pass@k result (5,000,000 samples, 0 correct) used the **4-shot** prefix. The rebuttal
leans on it in three places to argue these models have no latent capability, but a 4-shot
measurement cannot support a claim about 0-shot capability — the prefix changes the conditioning
context, which is the whole point of `PROTOCOL_CONFOUND.md`. So it was re-run at 0-shot.

`scripts/generate_pass_at_k_samples.py` gained `--num_fewshot {0,4}`, defaulting to 4 so the
original run stays reproducible; 0-shot output goes to its own directory.

## Result

| | 4-shot (original) | **0-shot (new)** |
|---|---|---|
| Samples | 5,000,000 | **62,500** |
| Problems | 5,000 | 2,500 (pass@25 each) |
| Temperature | 1.0 | 1.0 |
| Correct | **0** | **0** |
| Containing a well-formed `\boxed{}` | **0** | **0** |
| Lenient upper bound (gold answer string appears anywhere) | 21.7% | 19.1% |

The lenient row is deliberately over-generous — it counts a sample whenever the gold answer string
occurs anywhere in the response, ignoring formatting and context, so a response containing "2" is
credited on any problem whose answer is 2. It is reported so that a strict score of zero cannot be
dismissed as an artifact of the `\boxed{}` requirement. At ~20% it is obviously mostly spurious.

**Conclusion.** An uncontaminated 344M model has no measurable capability on MATH under either
protocol, sampled or greedy. Every point of contaminated performance is therefore memorization,
and there is no latent competence for contamination to combine with — which is what reconciles
our Finding 2 with the prior work (Mehrbakhsh et al. 2024; Dekoninck et al. 2024) that finds
rephrased contamination *does* transfer in already-capable models.

Both shards complete: 2,500 problems x 25 samples. **pass@25 = 0 on every one of the 2,500
problems.** Sharding covers problem indices 0-2,500 of the 5,000-problem test set.
