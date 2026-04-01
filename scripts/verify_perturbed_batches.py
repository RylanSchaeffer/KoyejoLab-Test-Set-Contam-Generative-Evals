"""Verify programmatic quality of perturbed math batch output files.

Checks:
- All required fields present and non-empty
- answer == last \boxed{} in solution
- problem != original_problem (actually perturbed)
- No truncated LaTeX
- Type and level match original
- Valid JSON structure
"""

import json
import os
import re
import sys
from collections import defaultdict

BATCH_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "math_perturbed_batches",
)

REQUIRED_FIELDS = [
    "idx",
    "original_problem",
    "problem",
    "original_answer",
    "answer",
    "solution",
    "level",
    "type",
]


def extract_boxed_answer(solution: str) -> str:
    """Extract the last \\boxed{...} content from a solution string."""
    idx = solution.rfind("\\boxed{")
    if idx == -1:
        return ""
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(solution) and depth > 0:
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            depth -= 1
        i += 1
    return solution[start : i - 1]


def normalize_latex(s: str) -> str:
    """Normalize LaTeX for comparison (dfrac->frac, whitespace)."""
    s = s.replace("\\dfrac", "\\frac")
    s = s.replace("\\tfrac", "\\frac")
    # Normalize whitespace
    s = " ".join(s.split())
    return s


def check_truncated_latex(text: str) -> bool:
    """Check if LaTeX appears truncated (unbalanced braces, etc.)."""
    # Count braces
    open_braces = text.count("{")
    close_braces = text.count("}")
    if abs(open_braces - close_braces) > 1:
        return True
    # Check for common truncation patterns
    if text.rstrip().endswith("\\"):
        return True
    return False


def verify_batch(batch_idx: int) -> dict:
    """Verify a single batch output file. Returns dict of issues."""
    input_path = os.path.join(BATCH_DIR, f"batch_{batch_idx:03d}_input.json")
    output_path = os.path.join(BATCH_DIR, f"batch_{batch_idx:03d}_output.json")

    issues = {
        "batch": batch_idx,
        "missing_output": False,
        "invalid_json": False,
        "wrong_count": False,
        "problems": [],
    }

    if not os.path.exists(output_path):
        issues["missing_output"] = True
        return issues

    # Load input for reference
    with open(input_path) as f:
        input_data = json.load(f)

    # Load output
    try:
        with open(output_path) as f:
            output_data = json.load(f)
    except json.JSONDecodeError as e:
        issues["invalid_json"] = True
        issues["json_error"] = str(e)
        return issues

    if len(output_data) != len(input_data):
        issues["wrong_count"] = True
        issues["expected"] = len(input_data)
        issues["got"] = len(output_data)

    # Build input lookup
    input_lookup = {item["idx"]: item for item in input_data}

    for item in output_data:
        problem_issues = []

        # Check required fields
        for field in REQUIRED_FIELDS:
            if field not in item:
                problem_issues.append(f"missing field: {field}")
            elif isinstance(item[field], str) and not item[field].strip():
                problem_issues.append(f"empty field: {field}")

        if "idx" not in item:
            problem_issues.append("no idx")
            issues["problems"].append(
                {"idx": "unknown", "issues": problem_issues}
            )
            continue

        idx = item["idx"]
        input_item = input_lookup.get(idx)

        # Check problem was actually perturbed
        if input_item and item.get("problem") == input_item["problem"]:
            problem_issues.append("NOT PERTURBED: problem == original_problem")

        # Check original_problem matches input
        if input_item and item.get("original_problem") != input_item["problem"]:
            problem_issues.append("original_problem doesn't match input")

        # Check answer matches boxed in solution
        if "solution" in item and item["solution"]:
            boxed = extract_boxed_answer(item["solution"])
            if not boxed:
                problem_issues.append("no \\boxed{} in solution")
            elif normalize_latex(boxed) != normalize_latex(item.get("answer", "")):
                problem_issues.append(
                    f"answer mismatch: answer='{item.get('answer', '')}' vs boxed='{boxed}'"
                )

        # Check type and level match original
        if input_item:
            if item.get("level") != input_item["level"]:
                problem_issues.append(
                    f"level mismatch: '{item.get('level')}' vs '{input_item['level']}'"
                )
            if item.get("type") != input_item["type"]:
                problem_issues.append(
                    f"type mismatch: '{item.get('type')}' vs '{input_item['type']}'"
                )

        # Check for truncated LaTeX
        if "solution" in item and item["solution"]:
            if check_truncated_latex(item["solution"]):
                problem_issues.append("possibly truncated LaTeX in solution")

        # Check answer != original_answer (expected but not required)
        if item.get("answer") == item.get("original_answer"):
            problem_issues.append("WARNING: answer == original_answer (may be valid)")

        if problem_issues:
            issues["problems"].append({"idx": idx, "issues": problem_issues})

    return issues


def main():
    # Find all output batch files
    output_files = sorted(
        [
            f
            for f in os.listdir(BATCH_DIR)
            if f.endswith("_output.json")
        ]
    )

    total_batches = 200
    completed_batches = len(output_files)
    print(f"Found {completed_batches}/{total_batches} completed batch output files")

    if completed_batches == 0:
        print("No output files to verify.")
        return

    # Verify all completed batches
    all_issues = []
    total_problems = 0
    problems_with_issues = 0
    problems_not_perturbed = 0
    problems_no_boxed = 0
    problems_answer_mismatch = 0
    problems_same_answer = 0
    batches_missing = 0
    batches_invalid_json = 0
    batches_wrong_count = 0

    for batch_idx in range(total_batches):
        result = verify_batch(batch_idx)

        if result["missing_output"]:
            batches_missing += 1
            continue

        if result["invalid_json"]:
            batches_invalid_json += 1
            all_issues.append(result)
            continue

        if result["wrong_count"]:
            batches_wrong_count += 1

        # Count problems in this batch
        output_path = os.path.join(BATCH_DIR, f"batch_{batch_idx:03d}_output.json")
        with open(output_path) as f:
            data = json.load(f)
        total_problems += len(data)

        for prob in result["problems"]:
            problems_with_issues += 1
            for issue in prob["issues"]:
                if "NOT PERTURBED" in issue:
                    problems_not_perturbed += 1
                if "no \\boxed{}" in issue:
                    problems_no_boxed += 1
                if "answer mismatch" in issue:
                    problems_answer_mismatch += 1
                if "answer == original_answer" in issue:
                    problems_same_answer += 1

        if result["problems"]:
            all_issues.append(result)

    # Print summary
    print(f"\n{'='*60}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"Completed batches: {completed_batches}/{total_batches}")
    print(f"Missing batches: {batches_missing}")
    print(f"Invalid JSON batches: {batches_invalid_json}")
    print(f"Wrong count batches: {batches_wrong_count}")
    print(f"Total problems verified: {total_problems}")
    print(f"Problems with issues: {problems_with_issues}")
    print(f"  - Not perturbed: {problems_not_perturbed}")
    print(f"  - No \\boxed{{}}: {problems_no_boxed}")
    print(f"  - Answer != \\boxed{{}}: {problems_answer_mismatch}")
    print(f"  - Same answer as original (warning): {problems_same_answer}")

    # Print detailed issues
    if all_issues:
        print(f"\n{'='*60}")
        print("DETAILED ISSUES")
        print(f"{'='*60}")
        for batch_result in all_issues:
            batch_idx = batch_result["batch"]
            if batch_result.get("invalid_json"):
                print(f"\nBatch {batch_idx:03d}: INVALID JSON - {batch_result.get('json_error', '')}")
                continue
            if batch_result.get("wrong_count"):
                print(
                    f"\nBatch {batch_idx:03d}: WRONG COUNT - expected {batch_result['expected']}, got {batch_result['got']}"
                )
            for prob in batch_result["problems"]:
                for issue in prob["issues"]:
                    if "WARNING" not in issue:  # Skip warnings in detailed view
                        print(f"  Batch {batch_idx:03d}, idx {prob['idx']}: {issue}")

    # List batches that need reprocessing
    reprocess = []
    for batch_idx in range(total_batches):
        output_path = os.path.join(BATCH_DIR, f"batch_{batch_idx:03d}_output.json")
        if not os.path.exists(output_path):
            reprocess.append(batch_idx)
            continue
        result = verify_batch(batch_idx)
        if result["invalid_json"] or result["wrong_count"]:
            reprocess.append(batch_idx)
            continue
        # Check for critical issues (not just warnings)
        has_critical = False
        for prob in result["problems"]:
            for issue in prob["issues"]:
                if "WARNING" not in issue:
                    has_critical = True
                    break
        if has_critical:
            reprocess.append(batch_idx)

    if reprocess:
        print(f"\n{'='*60}")
        print(f"BATCHES NEEDING REPROCESSING: {len(reprocess)}")
        print(f"{'='*60}")
        print(f"Batch indices: {reprocess}")


if __name__ == "__main__":
    main()
