"""Shared scoring utilities for math benchmark evaluation.

Requires responses to contain \\boxed{} expressions. This eliminates false
positives from math_verify's parse() extracting bare numbers from garbage text
(~1.4% false positive rate on uncontaminated model outputs).

MATH functions:
    extract_boxed_answer: Extract content of the last \\boxed{...} via brace-depth matching.
    score_response: Score a response by extracting \\boxed{} answer and verifying with math-verify.

GSM8K functions:
    extract_gsm8k_gold_answer: Pull the reference number from a "#### <n>" gold string.
    extract_gsm8k_predicted_answer: Pull a predicted number from "#### <n>" or \\boxed{<n>}.
    score_gsm8k_response: Compare the two numerically.
"""

import re

from math_verify import parse, verify


def extract_boxed_answer(text: str) -> str | None:
    """Extract content of the last \\boxed{...} in text using brace-depth matching.

    Returns the content inside the braces, or None if no \\boxed{} is found.
    Handles nested braces correctly (e.g., \\boxed{\\frac{1}{2}} -> \\frac{1}{2}).
    """
    # Find the start of the last \boxed{
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    # Move past \boxed{
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    # i now points one past the closing brace
    return text[start : i - 1]


def score_response(gold_parsed, response_text: str) -> bool:
    """Score a response by extracting \\boxed{} answer and verifying with math-verify.

    Only counts a response as correct if it contains a \\boxed{} expression.
    This eliminates false positives from parse() extracting bare numbers from
    garbage text.
    """
    boxed_content = extract_boxed_answer(response_text)
    if boxed_content is None:
        return False
    try:
        target_parsed = parse(f"\\boxed{{{boxed_content}}}")
        return bool(verify(gold=gold_parsed, target=target_parsed))
    except Exception:
        return False


# GSM8K answers are always bare numbers, optionally comma-grouped (11 of the 1,209
# madrylab/gsm8k-platinum golds contain a comma) and optionally negative or decimal.
_GSM8K_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _first_number(text: str) -> str | None:
    """Return the first number in `text`, comma-separators stripped, or None.

    Only the first line is considered: models routinely keep generating past their
    answer (a fresh "Q:" turn, more reasoning), and everything after the first
    newline is a different claim, not this one.
    """
    match = _GSM8K_NUMBER_RE.search(text.strip().split("\n", 1)[0])
    if match is None:
        return None
    return match.group(0).replace(",", "")


# A GSM8K answer line is a bare number, optionally prefixed by a currency symbol
# and followed by punctuation -- nothing else. Anything more is not an answer.
_GSM8K_ANSWER_LINE_RE = re.compile(r"^[$\s]*(-?\d[\d,]*(?:\.\d+)?)\s*[.\s]*$")


def _sole_number_on_line(text: str) -> str | None:
    """Return the number if the first line of `text` is *only* a number, else None.

    Stricter than `_first_number`, and the strictness is load-bearing. A degenerate
    model looping on a few-shot demonstration emitted

        #### 12 x 2 = 24 pages now, and a 120-page book, 12 x 2 = 24 pages now...

    against a gold answer of 12. Taking the first number after the marker credits
    that as correct, when the model has plainly regurgitated a demonstration rather
    than answered the question. Requiring the line to contain nothing but the number
    rejects it. This is the GSM8K analogue of requiring \\boxed{} on MATH, and it
    exists for the same reason: to keep a capability floor from being inflated by
    coincidental substring matches.
    """
    match = _GSM8K_ANSWER_LINE_RE.match(text.strip().split("\n", 1)[0])
    if match is None:
        return None
    return match.group(1).replace(",", "")


def extract_gsm8k_gold_answer(answer_text: str) -> str | None:
    """Extract the reference answer from a GSM8K gold string.

    GSM8K golds are a chain of thought followed by a final line "#### <answer>";
    all 1,209 madrylab/gsm8k-platinum golds follow this and all are bare numbers.
    Returns None if the marker is absent, so a malformed row fails loudly at the
    call site rather than silently scoring everything wrong.
    """
    if "####" not in answer_text:
        return None
    return _first_number(answer_text.rsplit("####", 1)[1])


def extract_gsm8k_predicted_answer(response_text: str) -> str | None:
    """Extract a predicted answer, accepting either answer convention.

    Two formats are honoured, and the choice is deliberate. "#### <n>" is GSM8K's
    native convention, which a GSM8K-contaminated model reproduces. "\\boxed{<n>}"
    is what a MATH-pretrained model emits, and Phase 0 evaluates exactly those
    checkpoints on GSM8K -- scoring a correct answer as wrong purely because it
    arrived boxed would understate capability, which is the one direction of error
    that matters here.

    Requiring one of the two markers keeps this as strict as the MATH scorer:
    a bare trailing number in unstructured text does not count, which is what
    caused the ~1.4% false-positive rate under lenient parsing.

    Returns None when neither marker is present.
    """
    if "####" in response_text:
        number = _sole_number_on_line(response_text.rsplit("####", 1)[1])
        if number is not None:
            return number
    boxed_content = extract_boxed_answer(response_text)
    if boxed_content is not None:
        return _first_number(boxed_content)
    return None


def score_gsm8k_response(gold_answer: str | None, response_text: str) -> bool:
    """Score a GSM8K response by numeric comparison against the gold answer.

    `gold_answer` is the output of `extract_gsm8k_gold_answer`. Comparison is
    numeric rather than string-wise so that "18" and "18.0" agree.
    """
    if gold_answer is None:
        return False
    predicted = extract_gsm8k_predicted_answer(response_text)
    if predicted is None:
        return False
    try:
        return float(predicted) == float(gold_answer)
    except ValueError:
        return predicted == gold_answer
