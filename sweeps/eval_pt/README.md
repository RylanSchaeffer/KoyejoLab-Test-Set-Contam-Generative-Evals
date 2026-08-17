# eval_pt: two generations of eval assets — do not mix them

## Legacy W&B-sweep YAMLs (`math/`, `math_overtrained/`, `math_perturbed/`, `math_rephrased/`)

Historical record of the published sampling evals. They drive
`scripts/eval_language_model.py`, which is **hardwired 4-shot** (it prepends
`build_fewshot_prefix()` unconditionally) and scores with the lenient
`math_verify.parse()` scorer of its era. **Do not launch these for new work**: the eval
protocol standardised on 2026-07-30 is 0-shot with a required `\boxed{}`
(`reviews/2026_neurips/PROTOCOL_CONFOUND.md` — the same checkpoint scores 1.0000 at 0-shot and
0.0052 at 4-shot, and the two eras are scored differently). The YAMLs are kept as the record of
what actually ran; the model-list `.txt` files in `math_overtrained/` belong to the newer
pattern below.

## Current pattern (`gsm8k/`, `phase6/`, and all new work)

A newline-delimited model list + a guarded launch script driving
`scripts/eval_language_model_multi_temperature.py`, which takes the protocol **explicitly**
(`--num-fewshot`, `--prompt-style`, `--temperatures`) and refuses nothing silently. Protocol
choice per measurement:

- **Memorization** (contaminated checkpoints): `--num-fewshot 0` — the prompt must reproduce
  the injected document's opening byte-for-byte.
- **Capability** (R=0 checkpoints): few-shot with train-split demonstrations — R=0 models have
  never seen an answer marker, so 0-shot measures format invention (Phase 0 lesson).

Never mix the two protocols in one comparison. See `gsm8k/README.md` and `phase6/README.md`
for the per-phase assets.
