"""Create a cleaned version of stellaathena/math_rephrased and push to HuggingFace.

Fixes applied:
1. 23 wrong `answer` fields → set to last \\boxed{} from solution
2. 7 stale name references in solutions → replace with anonymized names
3. 1 accidentally perturbed problem (idx=4383) → proper rephrasing
4. ~68 formatting normalizations → answer matches \\boxed{} in solution

See reviews/2026_icml/REVIEWER_6RQA/math_rephrased_spot_check.md for details.
"""

from datasets import load_dataset, Dataset


def extract_last_boxed(solution: str) -> str | None:
    """Extract the last \\boxed{} value from a solution string, handling nested braces."""
    results = []
    i = 0
    while i < len(solution):
        pos = solution.find(r"\boxed{", i)
        if pos == -1:
            break
        start = pos + len(r"\boxed{")
        depth = 0
        j = start
        while j < len(solution):
            if solution[j] == "{":
                depth += 1
            elif solution[j] == "}":
                if depth == 0:
                    results.append(solution[start:j])
                    break
                depth -= 1
            j += 1
        i = j + 1
    return results[-1] if results else None


def fix_stale_names(ds_dict: dict, idx: int, replacements: list[tuple[str, str]]):
    """Apply name replacements to the solution field at a given index."""
    solution = ds_dict["solution"][idx]
    for old, new in replacements:
        solution = solution.replace(old, new)
    ds_dict["solution"][idx] = solution


def main():
    print("Loading stellaathena/math_rephrased...")
    ds = load_dataset("stellaathena/math_rephrased", split="test")
    # Convert to dict-of-lists for in-place mutation.
    ds_dict = {col: list(ds[col]) for col in ds.column_names}

    # =========================================================================
    # Fix 1: Accidentally perturbed problem (idx=4383)
    # Original: angles x° and (x+20)° on a straight line → x=80
    # Stella's version changed the angles, making it a different problem.
    # Replace with a proper rephrasing that keeps the same math.
    # =========================================================================
    print("Fixing idx=4383 (perturbed problem)...")
    ds_dict["problem"][4383] = (
        "On a straight line $l$, two adjacent angles have measures of $x^\\circ$ "
        "and $(x + 20)^\\circ$. Determine the value of $x$."
    )
    # The solution already has \boxed{80^\circ} which is correct for this problem.
    # The answer will be set to "80^\circ" by the boxed-extraction step below.

    # =========================================================================
    # Fix 2: Stale name references in solutions (7 rows)
    # The rephrased problems anonymized person names, but the solutions
    # (copied verbatim from original MATH) still reference those names.
    # =========================================================================
    print("Fixing 7 stale name references in solutions...")

    # idx=7: "Mr. Madoff" → "An investment"
    fix_stale_names(ds_dict, 7, [("Mr. Madoff's investment", "the investment")])

    # idx=1367: "Joe's" → "A baseball player"
    fix_stale_names(ds_dict, 1367, [("Joe's hits", "the player's hits")])

    # idx=2583: "Sasha and Chloe" → "Two darts" / "One" / "the other"
    fix_stale_names(
        ds_dict,
        2583,
        [
            ("from Sasha's dart", "from the first dart"),
            ("from Chloe's dart", "from the second dart"),
            ("Chloe's dart is closer", "The second dart is closer"),
        ],
    )

    # idx=2822: "Jonathon" → "A student"
    fix_stale_names(
        ds_dict,
        2822,
        [
            ("Jonathon's solution", "the student's solution"),
            ("Jonathon's reasoning", "the student's reasoning"),
        ],
    )

    # idx=3593: "Javier" → "A cyclist"
    fix_stale_names(ds_dict, 3593, [("Javier travels", "the cyclist travels")])

    # idx=3598: "Dave's sister" → "A baker"
    fix_stale_names(ds_dict, 3598, [("Dave's sister", "the baker")])

    # idx=4378: "Nancy" → "A two-digit number is formed..."
    fix_stale_names(
        ds_dict,
        4378,
        [("The largest number Nancy can generate", "The largest number that can be generated")],
    )

    # =========================================================================
    # Fix 3: Answer field normalization
    # For all rows EXCEPT multi-value and special cases, set answer = last \boxed{}
    # This fixes both the 23 wrong answers and ~68 formatting mismatches.
    # =========================================================================
    print("Normalizing answer fields from \\boxed{} values...")

    # Rows to exclude from boxed extraction:
    # - Multi-value answers (answer correctly joins multiple \boxed{} entries)
    # - One-of-multiple selections
    # - idx=4405 where answer is correct but \boxed{} is wrong
    exclude_from_boxed = {
        # Multi-value
        2152,
        2549,
        2656,
        2703,
        4559,
        4562,
        4860,
        # One-of-multiple
        2912,
        2936,
        # Answer correct, boxed wrong
        4405,
    }

    num_answer_changes = 0
    for i in range(len(ds_dict["answer"])):
        if i in exclude_from_boxed:
            continue
        boxed = extract_last_boxed(ds_dict["solution"][i])
        if boxed is not None and boxed != ds_dict["answer"][i]:
            ds_dict["answer"][i] = boxed
            num_answer_changes += 1

    print(f"  Changed {num_answer_changes} answer fields.")

    # =========================================================================
    # Verify fixes
    # =========================================================================
    print("\nVerifying fixes...")

    # Verify wrong answers are fixed
    wrong_answer_idxs = [
        1256, 1260, 1264, 1266, 1281, 1289, 1290, 1292, 1364, 1379, 1387,
        1397, 1463, 1469, 1496, 1527, 1529, 1541, 1563, 1638, 1657, 1658, 4383,
    ]
    for idx in wrong_answer_idxs:
        boxed = extract_last_boxed(ds_dict["solution"][idx])
        assert ds_dict["answer"][idx] == boxed, (
            f"idx={idx}: answer={ds_dict['answer'][idx]!r} != boxed={boxed!r}"
        )
    print(f"  ✓ All {len(wrong_answer_idxs)} wrong answers fixed.")

    # Verify name fixes
    assert "Mr. Madoff" not in ds_dict["solution"][7]
    assert "Joe's" not in ds_dict["solution"][1367]
    assert "Sasha" not in ds_dict["solution"][2583]
    assert "Chloe" not in ds_dict["solution"][2583]
    assert "Jonathon" not in ds_dict["solution"][2822]
    assert "Javier" not in ds_dict["solution"][3593]
    assert "Dave's sister" not in ds_dict["solution"][3598]
    assert "Nancy" not in ds_dict["solution"][4378]
    print("  ✓ All 7 stale name references fixed.")

    # Verify idx=4383 problem text
    assert "x" in ds_dict["problem"][4383]
    assert "(x + 20)" in ds_dict["problem"][4383] or "(x+20)" in ds_dict["problem"][4383]
    assert "2x" not in ds_dict["problem"][4383]  # No perturbed angles
    assert ds_dict["answer"][4383] == "80^\\circ"
    print("  ✓ idx=4383 properly rephrased with correct answer.")

    # Verify excluded rows were NOT changed
    assert ds_dict["answer"][4405] == "4\\sqrt{2}"
    print("  ✓ Excluded rows preserved.")

    # =========================================================================
    # Push to HuggingFace
    # =========================================================================
    print("\nCreating HuggingFace dataset...")
    clean_ds = Dataset.from_dict(ds_dict)
    print(f"  Dataset has {len(clean_ds)} rows, columns: {clean_ds.column_names}")

    print("Pushing to RylanSchaeffer/math_rephrased...")
    clean_ds.push_to_hub("RylanSchaeffer/math_rephrased", split="test")
    print("Done!")


if __name__ == "__main__":
    main()
