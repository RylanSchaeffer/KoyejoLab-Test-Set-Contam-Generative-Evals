"""Do the contaminant-arm checkpoints regurgitate on the ORIGINAL problems?

THE HYPOTHESIS THIS TESTS
-------------------------
The contaminant ablation (`notebooks/21_*`) shows that injecting rephrased problems with verbatim
solutions recovers 78-98% of the exact-replica *loss* reduction on the original test set. Table 1
shows that a model contaminated with exact replicas scores ~2.8% when the *evaluation* problem is
rephrased. Put together, those suggest:

    memorization is of the SOLUTION text; retrieval is keyed on the EXACT PROBLEM text.

That is currently an inference from two different experiments. This is the direct test. Take the
checkpoints trained on (rephrased problem, verbatim solution) and evaluate them, 0-shot, on the
ORIGINAL problems:

  - If the mechanism holds, accuracy stays LOW. The model holds the solutions but the retrieval
    key it learned is the rephrased problem, so the original problem does not unlock them --
    even though loss on those very solutions is far below the uncontaminated baseline.
  - If accuracy is HIGH, the solution is reachable from any paraphrase of its problem, the
    retrieval story is wrong, and paraphrased leakage inflates benchmark scores directly. That
    would be a more alarming result and we would report it as such.

Either outcome is worth reporting; the point is to measure rather than infer.

Scoring is boxed-required (`src.scoring.score_response`), matching every other 0-shot number.

Usage:
    ./mem_scoring_vs_sampling_env/bin/python scripts/eval_contaminant_checkpoints_zeroshot.py \
        --checkpoints models/pt_language_model/mem_*_cont_* --output_dir results/contaminant_eval
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.getcwd())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Local checkpoint directories (globs are expanded by the shell).")
    p.add_argument("--output_dir", default="results/contaminant_eval")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=2048)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import torch  # noqa: F401  (import order matters for vLLM)
    from math_verify import parse
    from vllm import LLM, SamplingParams

    import src.data
    from src.scoring import extract_boxed_answer, score_response

    # ORIGINAL problems, 0-shot -- the same protocol as Fig. 1 and every teacher-forced result.
    test = src.data.load_dataset_hendrycks_math()["test"]
    doc_to_text = src.data.MINERVA_MATH_DOC_TO_TEXT
    prompts = [
        doc_to_text.format(problem=q, solution="").rstrip() for q in test["problem"]
    ]
    solutions = list(test["solution"])
    print(f"{len(prompts)} original MATH test problems, 0-shot (no few-shot prefix).")

    os.makedirs(args.output_dir, exist_ok=True)
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, seed=0)

    results = []
    for ckpt in args.checkpoints:
        name = os.path.basename(ckpt.rstrip("/"))
        print(f"\n=== {name} ===", flush=True)
        llm = LLM(model=ckpt, dtype="bfloat16", enforce_eager=True)
        outs = llm.generate(prompts, sampling)
        responses = [o.outputs[0].text for o in outs]

        n_boxed = sum(extract_boxed_answer(r) is not None for r in responses)
        n_correct = 0
        for sol, resp in zip(solutions, responses):
            try:
                n_correct += int(bool(score_response(parse(sol), resp)))
            except Exception:
                pass
        # How often is the emitted text a verbatim copy of the gold solution? This separates
        # "produced the right answer" from "regurgitated the memorized string".
        n_verbatim = sum(
            1 for sol, resp in zip(solutions, responses) if sol.strip() and sol.strip() in resp
        )
        rec = {
            "checkpoint": name,
            "n": len(prompts),
            "math_verify": n_correct / len(prompts),
            "boxed_rate": n_boxed / len(prompts),
            "verbatim_solution_rate": n_verbatim / len(prompts),
        }
        results.append(rec)
        print(f"  math_verify={rec['math_verify']:.4f}  boxed={rec['boxed_rate']:.4f}  "
              f"verbatim={rec['verbatim_solution_rate']:.4f}", flush=True)

        with open(os.path.join(args.output_dir, f"{name}.jsonl"), "w") as f:
            for i, (resp, sol) in enumerate(zip(responses, solutions)):
                f.write(json.dumps({"idx": i, "response": resp, "solution": sol}) + "\n")

        del llm
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== SUMMARY (0-shot, ORIGINAL problems, boxed-required scoring) ===")
    for r in results:
        print(f"  {r['checkpoint']:<70} {r['math_verify']:.4f}")
    print(f"\nWrote {args.output_dir}/summary.json")


if __name__ == "__main__":
    main()
