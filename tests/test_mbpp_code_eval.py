"""Tests for the MBPP code-execution harness (src.code_eval + src.data MBPP).

Mirrors tests/test_gsm8k_scoring.py's discipline. Verifies:
  1. Code extraction from the [DONE] sentinel and markdown-fence conventions
  2. The sandbox: correct code passes, wrong code fails, infinite loops time
     out, exceptions and syntax errors fail cleanly, no-code fails cleanly
  3. Sandbox guards: network and shell escapes are blocked
  4. Template consistency between the injection and evaluation paths
  5. Round-trip on the real sanitized MBPP reference solutions (marked `slow`)

The round-trip test is the one that matters (checklist 2.3): a harness that
cannot score the reference solutions at 100% cannot be trusted to score model
output, and one that credits a deliberately broken solution is worse than
useless.
"""

import time

import pytest

import src.data
from src.code_eval import (
    extract_python_code,
    run_python_with_tests,
    score_mbpp_response,
)

# A tiny known-good task used throughout: MBPP-style shape, trivial semantics.
ADD_TESTS = ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"]
ADD_CODE = "def add(a, b):\n    return a + b"


# ---------------------------------------------------------------------------
# 1. Code extraction
# ---------------------------------------------------------------------------


def test_extraction_cuts_at_done_sentinel():
    response = f"{ADD_CODE}\n[DONE]\nYou are an expert Python programmer..."
    assert extract_python_code(response) == ADD_CODE


def test_extraction_without_sentinel_returns_whole_text():
    # A truncated generation has no [DONE]; the code so far is still the answer.
    assert extract_python_code(ADD_CODE) == ADD_CODE


def test_extraction_prefers_markdown_fence():
    response = f"Here is my solution:\n```python\n{ADD_CODE}\n```\nHope that helps!"
    assert extract_python_code(response) == ADD_CODE


def test_extraction_fence_without_language_tag():
    response = f"```\n{ADD_CODE}\n```"
    assert extract_python_code(response) == ADD_CODE


def test_extraction_of_empty_or_whitespace_returns_none():
    assert extract_python_code("") is None
    assert extract_python_code("   \n\n  ") is None
    assert extract_python_code("[DONE] trailing babble") is None


# ---------------------------------------------------------------------------
# 2. The sandbox
# ---------------------------------------------------------------------------


def test_correct_solution_passes():
    result = run_python_with_tests(ADD_CODE, ADD_TESTS)
    assert result.passed is True
    assert result.reason == "ok"


def test_wrong_solution_fails():
    wrong = "def add(a, b):\n    return a - b"
    result = run_python_with_tests(wrong, ADD_TESTS)
    assert result.passed is False
    assert "AssertionError" in result.reason


def test_infinite_loop_times_out():
    looping = "def add(a, b):\n    while True:\n        pass"
    start = time.monotonic()
    result = run_python_with_tests(looping, ADD_TESTS, timeout_seconds=2.0)
    elapsed = time.monotonic() - start
    assert result.passed is False
    assert result.reason == "timeout"
    # The wall-clock budget must actually bound the call (generous margin for
    # process startup and teardown on a loaded node).
    assert elapsed < 10.0


def test_runtime_exception_fails_cleanly():
    raising = 'def add(a, b):\n    raise RuntimeError("boom")'
    result = run_python_with_tests(raising, ADD_TESTS)
    assert result.passed is False
    assert "RuntimeError" in result.reason


def test_syntax_error_fails_cleanly():
    result = run_python_with_tests("def add(a, b:\n    ret urn", ADD_TESTS)
    assert result.passed is False
    assert result.reason.startswith("exit code")


def test_no_code_extracted_fails_cleanly():
    assert score_mbpp_response("", ADD_TESTS) is False
    assert score_mbpp_response("   \n ", ADD_TESTS) is False


def test_garbage_text_fails_cleanly():
    # Not empty, so it reaches execution and dies as a syntax error -- the
    # honest failure path, and it must not raise into the caller.
    assert (
        score_mbpp_response("I am not sure how to do this, sorry!", ADD_TESTS) is False
    )


def test_test_imports_are_honoured():
    code = "def half_pi():\n    return math.pi / 2\nimport math"
    result = run_python_with_tests(
        code,
        ["assert 1.5 < half_pi() < 1.6"],
        test_imports=["import math"],
    )
    assert result.passed is True


def test_score_mbpp_response_end_to_end():
    response = f"{ADD_CODE}\n[DONE]\nand some trailing hallucination"
    assert score_mbpp_response(response, ADD_TESTS) is True
    assert score_mbpp_response(response, ["assert add(1, 2) == 4"]) is False


# ---------------------------------------------------------------------------
# 3. Sandbox guards
# ---------------------------------------------------------------------------


def test_network_access_is_blocked():
    code = (
        "import socket\n"
        "def add(a, b):\n"
        '    socket.create_connection(("example.com", 80), timeout=1)\n'
        "    return a + b"
    )
    result = run_python_with_tests(code, ADD_TESTS, timeout_seconds=5.0)
    assert result.passed is False
    assert "blocked by sandbox" in result.reason


def test_shell_escape_is_blocked():
    code = (
        "import os\n"
        "def add(a, b):\n"
        '    os.system("echo pwned")\n'
        "    return a + b"
    )
    result = run_python_with_tests(code, ADD_TESTS)
    assert result.passed is False
    assert "blocked by sandbox" in result.reason


def test_subprocess_is_blocked():
    code = (
        "import subprocess\n"
        "def add(a, b):\n"
        '    subprocess.run(["echo", "pwned"])\n'
        "    return a + b"
    )
    result = run_python_with_tests(code, ADD_TESTS)
    assert result.passed is False
    assert "blocked by sandbox" in result.reason


# ---------------------------------------------------------------------------
# 4. Template consistency between the injection and evaluation paths
# ---------------------------------------------------------------------------


def test_injection_and_eval_templates_render_identically():
    """The contaminant text and the eval prompt must be byte-identical.

    Same property tests/test_gsm8k_scoring.py asserts for GSM8K: if the two
    templates drift, the 0-shot memorization signal in Phase 4 is measured
    against a prompt the model never saw, and contaminated models silently
    look clean.
    """
    problem = src.data.mbpp_problem_text(
        "Write a function to add two numbers.", ADD_TESTS
    )
    injected = src.data.MBPP_DOC_TO_TEXT.format(question=problem, answer=ADD_CODE)
    evaluated = src.data.MBPP_DOC_TO_TEXT_EVAL.format(
        problem=problem, solution=ADD_CODE
    )
    assert injected == evaluated


def test_eval_prompt_ends_at_begin_sentinel():
    """Formatted with solution="" and rstripped, the prompt must end "[BEGIN]"."""
    prompt = src.data.MBPP_DOC_TO_TEXT_EVAL.format(
        problem="task text", solution=""
    ).rstrip()
    assert prompt.endswith("[BEGIN]")


def test_fewshot_examples_are_well_formed():
    assert len(src.data.MBPP_FEWSHOT_EXAMPLES) == 3
    for example in src.data.MBPP_FEWSHOT_EXAMPLES:
        # Each demonstration shows the asserts and closes with the sentinel the
        # model is expected to reproduce.
        assert "Your code should pass these tests:" in example["problem"]
        assert "assert " in example["problem"]
        assert example["solution"].endswith("[DONE]")


def test_fewshot_demonstration_code_passes_its_own_tests():
    """Each hardcoded demonstration must actually be a correct solution."""
    for example in src.data.MBPP_FEWSHOT_EXAMPLES:
        code = extract_python_code(example["solution"])
        tests = [
            line
            for line in example["problem"].splitlines()
            if line.startswith("assert ")
        ]
        assert tests, "demonstration problem lost its asserts"
        result = run_python_with_tests(code, tests)
        assert result.passed is True, result.reason


# ---------------------------------------------------------------------------
# 5. Round-trip against the real dataset
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_round_trip_on_real_sanitized_mbpp():
    """Every reference solution must pass its own tests; broken ones must not.

    Checklist 2.3. Runs 257 sandboxed subprocesses (the sanitized test split),
    plus a decoy pass on a sample.
    """
    test_split = src.data.load_dataset_mbpp_for_eval()["test"]

    failures = []
    for row in test_split:
        result = run_python_with_tests(
            code=row["solution"],
            test_list=row["test_list"],
            test_imports=row["test_imports"],
        )
        if not result.passed:
            failures.append((row["task_id"], result.reason))
    assert not failures, f"{len(failures)} reference solutions failed: {failures[:5]}"

    # Decoy: a deliberately broken "solution" must never score, on a sample.
    false_positives = 0
    for row in list(test_split)[:25]:
        broken = row["solution"] + '\nraise RuntimeError("broken on purpose")'
        if run_python_with_tests(broken, row["test_list"], row["test_imports"]).passed:
            false_positives += 1
    assert false_positives == 0
