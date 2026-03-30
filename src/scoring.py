"""Shared scoring utilities for math benchmark evaluation.

Requires responses to contain \\boxed{} expressions. This eliminates false
positives from math_verify's parse() extracting bare numbers from garbage text
(~1.4% false positive rate on uncontaminated model outputs).

Functions:
    extract_boxed_answer: Extract content of the last \\boxed{...} via brace-depth matching.
    score_response: Score a response by extracting \\boxed{} answer and verifying with math-verify.
"""

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
