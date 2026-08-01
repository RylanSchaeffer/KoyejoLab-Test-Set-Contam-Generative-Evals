"""Tests for the GSM8K scoring implementation in src.scoring.

Mirrors tests/test_boxed_scoring.py. Verifies:
  1. Gold extraction from the "#### <answer>" convention
  2. Prediction extraction from both "####" and \\boxed{} conventions
  3. Strictness: unmarked bare numbers do not score
  4. Numeric rather than string comparison
  5. Round-trip on the real madrylab/gsm8k-platinum golds (marked `slow`)

The round-trip test is the one that matters: a scorer that cannot score the
reference answers themselves at 100% cannot be trusted to score model output,
and a scorer that credits a deliberately wrong answer is worse than useless.
"""

import pytest

from src.scoring import (
    extract_gsm8k_gold_answer,
    extract_gsm8k_predicted_answer,
    score_gsm8k_response,
)


# ---------------------------------------------------------------------------
# 1. Gold extraction
# ---------------------------------------------------------------------------


def test_gold_extraction_basic():
    answer = "She sells 9 eggs.\nShe makes 9 * 2 = 18 dollars.\n#### 18"
    assert extract_gsm8k_gold_answer(answer) == "18"


def test_gold_extraction_strips_comma_grouping():
    # 11 of the 1,209 platinum golds are comma-grouped.
    assert extract_gsm8k_gold_answer("blah\n#### 1,234") == "1234"


def test_gold_extraction_negative_and_decimal():
    assert extract_gsm8k_gold_answer("x\n#### -5") == "-5"
    assert extract_gsm8k_gold_answer("x\n#### 3.5") == "3.5"


def test_gold_extraction_ignores_calculator_annotations():
    # GSM8K chains of thought contain <<16-3-4=9>> annotations with digits in
    # them; only the text after #### may be read.
    answer = "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 eggs.\n#### 9"
    assert extract_gsm8k_gold_answer(answer) == "9"


def test_gold_extraction_returns_none_without_marker():
    assert extract_gsm8k_gold_answer("no marker here, answer is 18") is None


# ---------------------------------------------------------------------------
# 2. Prediction extraction, both conventions
# ---------------------------------------------------------------------------


def test_prediction_from_hash_marker():
    assert extract_gsm8k_predicted_answer("reasoning\n#### 42") == "42"


def test_prediction_from_boxed():
    # A MATH-pretrained checkpoint evaluated on GSM8K answers in MATH's format.
    assert extract_gsm8k_predicted_answer("The answer is $\\boxed{42}$.") == "42"


def test_prediction_stops_at_first_line_after_marker():
    # Models keep generating past their answer; the next turn is a different claim.
    response = "#### 42\nQ: A new question entirely?\nA: 99"
    assert extract_gsm8k_predicted_answer(response) == "42"


def test_prediction_uses_last_hash_marker():
    response = "#### 7\nmore reasoning\n#### 42"
    assert extract_gsm8k_predicted_answer(response) == "42"


def test_prediction_falls_back_to_boxed_when_hash_has_no_number():
    response = "#### \nThe answer is \\boxed{42}"
    assert extract_gsm8k_predicted_answer(response) == "42"


# ---------------------------------------------------------------------------
# 3. Strictness -- the false-positive guard
# ---------------------------------------------------------------------------


def test_bare_number_does_not_score():
    # This is the whole point of requiring a marker: the ~1.4% false-positive
    # rate under lenient parsing came from bare numbers in garbage text.
    assert extract_gsm8k_predicted_answer("I think it is probably 42") is None
    assert score_gsm8k_response("42", "I think it is probably 42") is False


def test_empty_and_garbage_responses_do_not_score():
    for response in ["", "\n\n", "!!!", "Problem: Problem: Problem:"]:
        assert score_gsm8k_response("42", response) is False


def test_none_gold_never_scores():
    assert score_gsm8k_response(None, "#### 42") is False


def test_regurgitated_demonstration_does_not_score():
    """Regression test for a real false positive observed on 2026-08-01.

    Qwen3-93M at ot=16 looped on the fourth few-shot demonstration and emitted the
    text below against a gold answer of 12. Taking the first number after the marker
    credits it; requiring the answer line to be *only* a number rejects it. This was
    the sole "correct" answer in that run, i.e. the entire measured 0.0008.
    """
    response = (
        " A 120-page book, 12 x 2 = 24 pages now, and a 120-page book, 12 x 2 = 24 "
        "pages now.\n#### 12 x 2 = 24 pages now, and a 120-page book, 12 x 2 = 24 "
        "pages now.\n#### 12 x 2 = 24 pages now"
    )
    assert extract_gsm8k_predicted_answer(response) is None
    assert score_gsm8k_response("12", response) is False


def test_truncated_generation_does_not_manufacture_an_answer():
    """Regression test for a second real false positive, 2026-08-01.

    Qwen3-344M at ot=8 looped on `#### 120+24 = 120` until it hit the 2,048-token
    cap mid-digit, leaving a trailing `#### 1` against a gold answer of 1. The
    truncation produced the answer. This was the last cell in the whole Phase 0
    sweep still showing non-zero accuracy.
    """
    response = "#### 120+24 = 120\n#### 120+24 = 120\n#### 1"
    assert extract_gsm8k_predicted_answer(response, truncated=True) is None
    assert score_gsm8k_response("1", response, truncated=True) is False
    # Untruncated, the same text is a legitimate (if wrong-looking) final answer.
    assert extract_gsm8k_predicted_answer(response, truncated=False) == "1"


def test_truncation_guard_does_not_reject_a_terminated_answer():
    """A memorized regurgitation ending in a newline must still score.

    Phase 3 evaluates GSM8K-contaminated checkpoints, whose whole signal is
    reproducing a training document verbatim. Over-applying the truncation guard
    would suppress exactly that.
    """
    response = "reasoning here\n#### 42\n"
    assert score_gsm8k_response("42", response, truncated=True) is True


def test_answer_line_must_contain_only_the_number():
    assert extract_gsm8k_predicted_answer("#### 42") == "42"
    assert extract_gsm8k_predicted_answer("#### $42") == "42"
    assert extract_gsm8k_predicted_answer("#### 42.") == "42"
    assert extract_gsm8k_predicted_answer("#### 42 apples") is None
    assert extract_gsm8k_predicted_answer("#### 42 = 6 x 7") is None


# ---------------------------------------------------------------------------
# 4. Comparison semantics
# ---------------------------------------------------------------------------


def test_numeric_equality_not_string_equality():
    assert score_gsm8k_response("18", "#### 18.0") is True
    assert score_gsm8k_response("18.0", "#### 18") is True


def test_comma_grouped_prediction_matches_plain_gold():
    assert score_gsm8k_response("1234", "#### 1,234") is True


def test_wrong_answer_does_not_score():
    assert score_gsm8k_response("18", "#### 19") is False
    assert score_gsm8k_response("18", "The answer is \\boxed{19}") is False


# ---------------------------------------------------------------------------
# 5. Template consistency between the injection and evaluation paths
# ---------------------------------------------------------------------------


def test_injection_and_eval_templates_render_identically():
    """The contaminant text and the eval prompt must be byte-identical.

    Contamination injection formats with `question`/`answer`; evaluation formats
    with `problem`/`solution`. If the two templates ever drift, the 0-shot
    memorization signal is measured against a prompt the model never saw, and
    contaminated models would silently look clean.
    """
    import src.data

    question = "Janet has 16 eggs. How many are left after eating 3?"
    answer = "16 - 3 = 13\n#### 13"

    injected = src.data.GSM8K_PLATINUM_DOC_TO_TEXT.format(
        question=question, answer=answer
    )
    evaluated = src.data.GSM8K_PLATINUM_DOC_TO_TEXT_EVAL.format(
        problem=question, solution=answer
    )
    assert injected == evaluated


def test_eval_prompt_has_no_stray_indentation():
    """Regression test for the eight-space indentation bug fixed 2026-08-01."""
    import src.data

    prompt = src.data.GSM8K_PLATINUM_DOC_TO_TEXT_EVAL.format(
        problem="Q?", solution=""
    ).rstrip()
    assert prompt == "Q: Q?\n\nA:"


# ---------------------------------------------------------------------------
# 6. Round-trip against the real dataset
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_round_trip_on_real_platinum_golds():
    """Every reference answer must score against itself, and none against a decoy."""
    import src.data

    test_split = src.data.load_dataset_gsm8k_platinum()["test"]
    answers = test_split["answer"]

    golds = [extract_gsm8k_gold_answer(a) for a in answers]
    assert all(g is not None for g in golds), "some gold had no #### marker"

    # Self-consistency: the gold string, fed back as a response, must score.
    correct = sum(
        score_gsm8k_response(gold, answer) for gold, answer in zip(golds, answers)
    )
    assert correct == len(answers), f"only {correct}/{len(answers)} golds self-scored"

    # Decoy: perturbing every gold by +1 must score zero.
    decoys = [f"#### {float(g) + 1:g}" for g in golds]
    false_positives = sum(
        score_gsm8k_response(gold, decoy) for gold, decoy in zip(golds, decoys)
    )
    assert false_positives == 0, f"{false_positives} decoys scored as correct"
