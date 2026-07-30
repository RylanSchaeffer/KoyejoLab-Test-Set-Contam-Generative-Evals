# Verdict: the `math_perturbed` injection is CONFIRMED (with one number corrected)

Adversarial check of Tier-1 item 1.2 in `VERIFICATION_HANDOFF.md`, run 2026-07-30 by a session
that did not produce the perturbed arm. Goal was to falsify, not confirm.

**Claim under test.** *In the perturbed arm, `RylanSchaeffer/math_perturbed` items are what get
injected into the pretraining corpus, while `eval_after/eval_benchmark_loss` is measured on the
original `EleutherAI/minerva_math` test set.*

**Verdict: CONFIRMED.** Two independent routes agree: (a) rebuilding the dataset through
`src.data.create_dataset_for_pretraining` and matching every training document against the
contaminant sets by exact token-id equality, and (b) the *actual* W&B training logs of the three
perturbed runs, whose recorded contaminant token counts match `math_perturbed` and cannot be
produced by any other dataset.

**One number is wrong and must be corrected everywhere it appears:** the per-copy token counts in
`CONTAMINANT_ABLATION.md` (and four other documents, incl. `REBUTTAL_DRAFT.md`) are each **exactly
5,000 tokens too low** — they omit the appended EOS token, one per document. Correct as-injected
values are **1,446,312 / 1,392,475 / 1,132,643** (original / rephrased / perturbed). The ratio
moves from 78.2% to **78.31%**, so no qualitative claim changes; the printed figures do.

Reproduce with `scripts/scratch/verify_perturbed_contaminant.py` (full pipeline build) and
`scripts/scratch/check_contaminant_dataset_overlap.py` (dataset/token-level checks).

---

## 1. Exact counts from the rebuilt pipeline

Built with `contaminant = RylanSchaeffer/math_perturbed`, `benchmark = EleutherAI/minerva_math`,
`benchmark_subset_fraction = 1.0` (as in sweep `vrxwx4dz`), `num_benchmark_replicas_per_epoch = 2`
(reduced from 32/100/316 only to keep the corpus small). Every training and benchmark document was
compared by **exact token-id sequence equality** against the three contaminant sets tokenized the
same way — not substring heuristics, which is what the earlier rephrased check used.

| Split | n | matches ORIGINAL | matches REPHRASED | matches PERTURBED |
|---|---|---|---|---|
| **train** (what the model sees) | 11,574 | 2 | 0 | **10,000** |
| **benchmark** (what `eval_benchmark_loss` uses) | 5,000 | **5,000** | 0 | 1 |

Disambiguating the one document that belongs to both sets (see §3):

| Split | perturbed-only | original-only | in both sets |
|---|---|---|---|
| train | **9,998** | **0** | 2 |
| benchmark | **0** | **4,999** | 1 |

- 10,000 = 2 replicas × 5,000 perturbed items. Every injected item is a perturbed item.
- **Zero original items enter training** other than the single item that is byte-identical in both
  datasets (0.02% of one copy).
- Every benchmark document is an original item; **zero perturbed-only items are in the eval set**.
- Contaminant tokens in the train split: **2,265,286 = 2 × 1,132,643**, i.e. exactly two copies of
  `math_perturbed` and *not* two copies of the original (which would be 2,892,624).
- Unique train documents 6,574 = 5,000 contaminant + 1,574 corpus documents. No aliasing.

**Exact-replica control** (same call with the `contaminant` key omitted): train 11,005 docs →
**10,000 match ORIGINAL**, 2 match perturbed (the shared item ×2); benchmark 5,000 → all original;
contaminant tokens 2,892,624 = 2 × 1,446,312. So the published code path is unchanged by the
contaminant feature.

**Benchmark identity across arms.** The benchmark split's document order maps to original indices
`[2221, 1222, 227, 4662, 3029, 3428, 1498, 727, 1385, 50, …]` in *both* the perturbed arm and the
exact-replica control, and the two sequences are equal element-for-element. The perturbed arm's
loss is therefore computed on the identical evaluation set, in the identical order, as the
published exact-replica runs.

**Spot check (decoded).** Train document ↔ perturbed idx 4747: perturbed says `$|z| = 7$` where the
original says `$|z| = 5$`, with a different solution body. idx 4244: side length 10 vs 12. idx 3222:
`\frac{22}{9}` vs `\frac{23}{9}`. Benchmark doc 0 is verbatim original MATH.

## 2. The real runs, not just the code path

The code being right today does not prove the runs were right yesterday, so the three finished runs'
own console logs were pulled from W&B (`rylan/memorization-scoring-vs-sampling-pt-paraphrased`).
All three print the contaminant banner and a replicated-token count that is an exact multiple of
the perturbed per-copy size:

| Run | R | replicated contaminant tokens | = R × 1,132,643? | total train tokens |
|---|---|---|---|---|
| `tnv4we9r` | 32 | 36,244,576 | ✅ | 496,852,130 |
| `j44r5z94` | 100 | 113,264,300 | ✅ | 519,366,323 |
| `moe1qunh` | 316 | 357,915,188 | ✅ | 589,028,561 |
| `c0tb7qnf` (rephrased) | 32 | 44,559,200 | = 32 × 1,392,475 ✅ | 499,177,968 |

Had the original test set been injected, R=32 would read 46,281,984. It does not. Each log also
carries `Contaminant (RylanSchaeffer/math_perturbed) differs from benchmark
(EleutherAI/minerva_math): injecting 5000 contaminant examples, measuring loss on 5000 benchmark
examples.` All three runs' configs record `contaminant = RylanSchaeffer/math_perturbed`,
`benchmark = EleutherAI/minerva_math`, `benchmark_subset_fraction = 1`.

**Behavioural corroboration.** The perturbed R=316 run's `eval_benchmark_loss` is 3.3705 after
seeing 316 copies of its contaminant. A model that had been trained 316× on the very documents the
loss is computed on scores ≈0.5 (exact arm, same dose). The measured loss is therefore not being
taken on the injected text — consistent with it being the original test set.

## 3. Dataset sanity — the overlap claims hold

Index-aligned against `EleutherAI/hendrycks_math` test (n=5,000 for all three):

| Dataset | problems identical | **solutions identical** | both |
|---|---|---|---|
| `math_rephrased` | 0 / 5000 | **4,991 / 5000 (99.82%)** | 0 |
| `math_perturbed` | 0 / 5000 | **4 / 5000 (0.08%)** | 0 |

Matches `CONTAMINANT_ABLATION.md` exactly. Set-level (allowing any index, which the doc did not
report): perturbed shares 23/5000 problems and 5/5000 solutions with the original set, and exactly
**1/5000 documents is byte-identical as formatted for injection**. That single document is the
"ambiguous" row in §1 — a 0.02% verbatim leak in the perturbed arm, negligible but now quantified
rather than assumed to be zero.

**Index alignment holds.** `datasets.shuffle(seed=0)` produces the *identical* permutation for all
three datasets (verified by shuffling an added index column: `[2221, 1222, 227, 4662, …]` for each),
so `.select(range(n))` after shuffling picks *corresponding* items across arms. Post-shuffle,
`type` agrees 5000/5000 and `level` agrees 5000/5000 between original and perturbed, and
`perturbed[i]["idx"] == i` for all i. (`perturbed[i]["original_problem"]` equals the original
problem text for only 4,719/5,000 — but all 281 exceptions are LaTeX whitespace reflow from the
generating model, 254 of which survive whitespace normalisation as pure formatting differences.
`type`/`level` agreeing 5000/5000 rules out row misalignment.)

**Answer-overlap mask independently reproduced:** 4,418/5,000 perturbed answers differ from the
original answer ⇒ 582/5,000 = **11.64%** share an answer. Matches the inherited mask.

## 4. ⚠️ Corrected number: per-copy token counts are 5,000 too low in five documents

| Contaminant | published in docs | **as actually injected** | source of the gap |
|---|---|---|---|
| `EleutherAI/minerva_math` | 1,441,312 | **1,446,312** | +1 EOS token × 5,000 docs |
| `RylanSchaeffer/math_rephrased` | 1,387,475 | **1,392,475** | idem |
| `RylanSchaeffer/math_perturbed` | 1,127,643 | **1,132,643** | idem |

Verified two ways: summing `token_length` from
`create_dataset_for_supervised_finetuning` (which appends `tokenizer.eos_token` to the formatted
text before tokenizing — `src/data.py:658`), and against the runs' own logged
`Replicated Benchmark Test Split has N tokens`, which divide exactly by the corrected values.
Tokenizing without the EOS reproduces the published figures to the token, so that is how they were
obtained.

Consequences: perturbed/original **78.31%** (was quoted 78.2%); perturbed is **21.7%** smaller (was
21.8%); "perturbed R=316 ≈ exact R≈247" is unchanged (247.5). The contaminated-fraction table in
`reviews/2026_neurips/data/CONTAMINATED_TOKEN_FRACTION.md` is 0.35% relatively understated. **No
conclusion changes; five documents print wrong digits, one of which is the rebuttal draft.** All
have been corrected in the same commit as this file.

## 5. Other things found while reading `src/data.py`

Reported because they were not on the checklist, in descending order of how much they could matter.

1. **Training tokens are *not* held constant across contamination levels — they rise 27% with dose.**
   `create_dataset_for_pretraining`'s docstring says "The total training tokens per epoch is fixed;
   more benchmark replicas means fewer corpus tokens, keeping compute constant across contamination
   levels." That is false in every run. `avg_tokens_per_doc` is hard-coded at 220e9/190168005 ≈ 1157
   (`src/data.py:333`) but the *post-truncation* mean is ≈787, so the corpus step delivers only
   ~71.5% of the tokens it asks for, while the contaminant is delivered in full. Published 34M ot=1
   runs, from `train/num_input_tokens_seen` in the notebook-11 cache:

   | R | 0 | 1 | 3 | 10 | 32 | 100 | 316 |
   |---|---|---|---|---|---|---|---|
   | tokens seen | 486.1M | 486.6M | 487.4M | 490.4M | 499.3M | 527.4M | **617.2M** |

   So dose is confounded with training tokens (+27% from R=0 to R=316) in the published pretraining
   figures. **This does not invalidate the scaling-law fits** — `src/analyze.py:601-607` uses the
   *measured* `eval_after/num_input_tokens_seen` for `FLOP (6ND)`, not the nominal budget — and the
   loss drop 7.14 → 0.52 is far too large to be bought by 27% more tokens. But any prose claiming
   compute was held constant across R is wrong, and a reviewer can compute this from the logged
   token counts. The docstring should be fixed and the 3 lines of the ablation's token caveat
   should be stated as "corpus under-delivers ~28.5%" rather than only appearing at R=316.

2. **The identity guard on line 265 is dead code.** `contaminant_dataset` is bound to
   `benchmark_test_split_dataset` at line 229, but line 259 *rebinds* `benchmark_test_split_dataset`
   to a new shuffled/selected object, so by line 265 `contaminant_dataset is not
   benchmark_test_split_dataset` is **always True** and the `else` branch (line 270) can never run.
   Behaviour is nonetheless correct: the fall-through shuffles and selects the same rows with the
   same seed, which is deterministic, and the control run in §1 confirms the exact-replica path is
   byte-identical. It is a latent trap, not a live bug.

3. **No guard that contaminant and benchmark have the same length.** The subsample count is computed
   from `len(benchmark_test_split_dataset)` and then applied to the contaminant with
   `.select(range(n))`. All three datasets are 5,000 rows so this is fine today; a shorter
   contaminant would raise, and a longer one would silently misalign. Worth one `assert`.

4. **Injected documents are never truncated to `max_length`.** `create_dataset_for_pretraining`
   calls `create_dataset_for_supervised_finetuning(..., max_length=None)`, and the MATH preprocessor
   tokenizes without `truncation=True` — only *corpus* documents get truncated at 2,048. Observed
   maxima: original 2,083 (1 doc > 2048), rephrased 2,087 (1), perturbed **7,260** (2). Affects all
   arms including the published ones; the effect is a handful of over-length sequences per copy.

5. `load_dataset_math_rephrased`'s docstring still says the loader "is not currently reachable from
   `create_dataset_for_supervised_finetuning()`". It has been reachable since commit 29f348a.

## What would still falsify the claim, and was not checked

- The perturbed *accuracy* numbers come from `scripts/eval_contaminant_checkpoints_zeroshot.py`, a
  separate path (Tier 1.4). Nothing here speaks to them.
- Only R=2 was rebuilt end-to-end. R=32/100/316 were verified through their own training logs'
  token arithmetic rather than by rebuilding, since the arithmetic is a sharper test than a rebuild.
