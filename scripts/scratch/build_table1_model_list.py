"""Emit the checkpoint list for the Table 1 (rephrased / perturbed) re-run.

Table 1 claims contaminated models' gains vanish when the test problems are rephrased or
perturbed. Demonstrating that requires measuring both conditions in a protocol where the
gains exist in the first place: under 4-shot prompting even the *original* test set scores
~0.005 at 344M, so a rephrased score of ~0.000 shows nothing. Under 0-shot the same
checkpoint scores ~1.0 on the original set, so the comparison has something to collapse from.

Replica ladders match the ones printed in `manuscript_neurips_2026/03_pretraining.tex`,
extended to every model size that has 0-shot original coverage, so the caption's
"consistent across all model sizes" claim can be supported rather than asserted.
"""

REPLICAS_BY_SIZE = {
    "34M": [0, 1, 3, 10, 32, 100, 316],
    "62M": [0, 1, 3, 10, 32, 100, 316],
    "93M": [0, 1, 3, 10, 32, 100, 316, 1000],
    "153M": [0, 1, 3, 10, 32, 100, 316, 1000],
    "344M": [0, 1, 3, 10, 32, 100, 316, 1000, 3162],
}

TEMPLATE = (
    "RylanSchaeffer/mem_Qwen3-{size}_minerva_math_rep_{replicas}"
    "_sbst_1.0000_epch_1_ot_1"
)

OUT_PATH = "sweeps/eval_pt/math_overtrained/models_table1_rerun.txt"


def main() -> None:
    names = []
    for size, replicas in REPLICAS_BY_SIZE.items():
        for replica in replicas:
            names.append(TEMPLATE.format(size=size, replicas=replica))

    with open(OUT_PATH, "w") as f:
        f.write("\n".join(names) + "\n")

    print(f"Wrote {len(names)} checkpoints to {OUT_PATH}")
    for size, replicas in REPLICAS_BY_SIZE.items():
        print(f"  {size:>5}: {len(replicas)} replica levels {replicas}")
    print(
        f"\nAcross 2 modified datasets that is {2 * len(names)} runs; "
        f"the 0-shot original baseline already exists in the superseded sweeps."
    )


if __name__ == "__main__":
    main()
