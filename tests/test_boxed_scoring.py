"""Comprehensive tests for the \\boxed{}-required scoring implementation.

Tests extract_boxed_answer() and score_response() from both score_pass_at_k.py
and incremental_scorer.py. Verifies:
  1. Correct extraction of \\boxed{} content with nested braces
  2. False positive elimination (bare numbers no longer score as correct)
  3. Equivalence between the two scoring scripts
  4. Integration with math-verify parse() and verify()
  5. Realistic MATH benchmark answer formats
"""

import pytest
from math_verify import parse, verify

from scripts.score_pass_at_k import (
    extract_boxed_answer as extract_boxed_scorer,
    score_response as score_response_scorer,
)
from scripts.incremental_scorer import (
    extract_boxed_answer as extract_boxed_incremental,
    score_response as score_response_incremental,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gold(expr: str):
    """Convenience: parse a gold expression."""
    return parse(expr)


# ---------------------------------------------------------------------------
# 1. extract_boxed_answer: basic extraction
# ---------------------------------------------------------------------------

class TestExtractBoxedBasic:
    """Test basic \\boxed{} extraction."""

    def test_simple_integer(self):
        assert extract_boxed_scorer(r"The answer is \boxed{42}") == "42"

    def test_simple_negative(self):
        assert extract_boxed_scorer(r"Therefore $\boxed{-15}$.") == "-15"

    def test_simple_zero(self):
        assert extract_boxed_scorer(r"\boxed{0}") == "0"

    def test_single_variable(self):
        assert extract_boxed_scorer(r"So the answer is \boxed{i}") == "i"

    def test_decimal(self):
        assert extract_boxed_scorer(r"\boxed{3.14}") == "3.14"

    def test_boxed_at_start(self):
        assert extract_boxed_scorer(r"\boxed{7} is the answer") == "7"

    def test_boxed_at_end(self):
        assert extract_boxed_scorer(r"blah blah \boxed{7}") == "7"

    def test_boxed_with_period_after(self):
        assert extract_boxed_scorer(r"The answer is $\boxed{4}$.") == "4"

    def test_empty_boxed(self):
        assert extract_boxed_scorer(r"\boxed{}") == ""


# ---------------------------------------------------------------------------
# 2. extract_boxed_answer: nested braces
# ---------------------------------------------------------------------------

class TestExtractBoxedNested:
    """Test extraction with nested brace structures."""

    def test_fraction(self):
        assert extract_boxed_scorer(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"

    def test_dfraction(self):
        assert extract_boxed_scorer(r"\boxed{\dfrac{9}{7}}") == r"\dfrac{9}{7}"

    def test_negative_fraction(self):
        assert extract_boxed_scorer(r"\boxed{-\frac{1}{8}}") == r"-\frac{1}{8}"

    def test_nested_exponent(self):
        assert extract_boxed_scorer(r"\boxed{x^{2}}") == r"x^{2}"

    def test_deep_nesting(self):
        assert extract_boxed_scorer(r"\boxed{\frac{x^{2}+1}{y^{3}}}") == r"\frac{x^{2}+1}{y^{3}}"

    def test_sqrt(self):
        assert extract_boxed_scorer(r"\boxed{\sqrt{2}}") == r"\sqrt{2}"

    def test_nested_sqrt_fraction(self):
        assert extract_boxed_scorer(r"\boxed{\frac{\sqrt{3}}{2}}") == r"\frac{\sqrt{3}}{2}"

    def test_interval_notation(self):
        assert extract_boxed_scorer(r"\boxed{[0,\infty)}") == r"[0,\infty)"

    def test_interval_with_sqrt(self):
        assert extract_boxed_scorer(r"\boxed{(-\sqrt{3}, \sqrt{3})}") == r"(-\sqrt{3}, \sqrt{3})"

    def test_text_command(self):
        assert extract_boxed_scorer(r"\boxed{\text{Evelyn}}") == r"\text{Evelyn}"

    def test_multiple_nested_groups(self):
        # e.g. \boxed{\binom{10}{3}}
        assert extract_boxed_scorer(r"\boxed{\binom{10}{3}}") == r"\binom{10}{3}"

    def test_triple_nesting(self):
        expr = r"\boxed{\frac{1}{\sqrt{x^{2}+1}}}"
        assert extract_boxed_scorer(expr) == r"\frac{1}{\sqrt{x^{2}+1}}"


# ---------------------------------------------------------------------------
# 3. extract_boxed_answer: multiple \boxed{} (should take LAST)
# ---------------------------------------------------------------------------

class TestExtractBoxedMultiple:
    """MATH convention: take the last \\boxed{} expression."""

    def test_two_boxed_takes_last(self):
        text = r"First \boxed{1}, then \boxed{42}"
        assert extract_boxed_scorer(text) == "42"

    def test_three_boxed_takes_last(self):
        text = r"\boxed{a} and \boxed{b} so \boxed{c}"
        assert extract_boxed_scorer(text) == "c"

    def test_boxed_in_work_vs_final(self):
        # Common pattern: intermediate boxed in work, final boxed answer
        text = (
            r"We know $x = \boxed{5}$ from the first equation. "
            r"Substituting, $y = \boxed{3}$. "
            r"Therefore $x + y = \boxed{8}$."
        )
        assert extract_boxed_scorer(text) == "8"


# ---------------------------------------------------------------------------
# 4. extract_boxed_answer: no boxed / malformed
# ---------------------------------------------------------------------------

class TestExtractBoxedNone:
    """Cases where extraction should return None."""

    def test_no_boxed_plain_text(self):
        assert extract_boxed_scorer("The answer is 42.") is None

    def test_no_boxed_garbage(self):
        assert extract_boxed_scorer("blah blah 2 blah") is None

    def test_no_boxed_with_number(self):
        assert extract_boxed_scorer("I think the answer is 7, because reasons") is None

    def test_no_boxed_with_dollar_signs(self):
        assert extract_boxed_scorer("The answer is $42$") is None

    def test_no_boxed_with_answer_pattern(self):
        assert extract_boxed_scorer("The final answer is $\\frac{1}{2}$") is None

    def test_unclosed_boxed(self):
        assert extract_boxed_scorer(r"\boxed{42") is None

    def test_unclosed_nested(self):
        assert extract_boxed_scorer(r"\boxed{\frac{1}{2}") is None

    def test_boxed_no_brace(self):
        assert extract_boxed_scorer(r"\boxed 42") is None

    def test_empty_string(self):
        assert extract_boxed_scorer("") is None

    def test_just_backslash_boxed(self):
        assert extract_boxed_scorer(r"\boxed") is None

    def test_partial_boxed_word(self):
        # "boxed" appears but not as \boxed{
        assert extract_boxed_scorer("I boxed the answer: 42") is None


# ---------------------------------------------------------------------------
# 5. score_response: correct answers (should return True)
# ---------------------------------------------------------------------------

class TestScoreResponseCorrect:
    """Verify correct boxed answers score True."""

    def test_integer_match(self):
        assert score_response_scorer(gold(r"\boxed{2}"), r"The answer is \boxed{2}") is True

    def test_negative_match(self):
        assert score_response_scorer(gold(r"\boxed{-15}"), r"Sum is \boxed{-15}") is True

    def test_zero_match(self):
        assert score_response_scorer(gold(r"\boxed{0}"), r"\boxed{0}") is True

    def test_fraction_match(self):
        assert score_response_scorer(gold(r"\boxed{\frac{1}{2}}"), r"Thus \boxed{\frac{1}{2}}") is True

    def test_dfrac_vs_frac(self):
        """\\dfrac and \\frac should be equivalent."""
        assert score_response_scorer(gold(r"\boxed{\dfrac{9}{7}}"), r"\boxed{\frac{9}{7}}") is True

    def test_negative_fraction(self):
        assert score_response_scorer(gold(r"\boxed{-\frac{1}{8}}"), r"Answer: \boxed{-\frac{1}{8}}") is True

    def test_large_integer(self):
        assert score_response_scorer(gold(r"\boxed{1000}"), r"\boxed{1000}") is True

    def test_sqrt_match(self):
        assert score_response_scorer(gold(r"\boxed{\sqrt{2}}"), r"The answer is \boxed{\sqrt{2}}.") is True

    def test_symbolic_match(self):
        assert score_response_scorer(gold(r"\boxed{i}"), r"So $\boxed{i}$ is our answer") is True


# ---------------------------------------------------------------------------
# 6. score_response: wrong answers (should return False)
# ---------------------------------------------------------------------------

class TestScoreResponseWrong:
    """Verify wrong boxed answers score False."""

    def test_wrong_integer(self):
        assert score_response_scorer(gold(r"\boxed{2}"), r"\boxed{3}") is False

    def test_wrong_fraction(self):
        assert score_response_scorer(gold(r"\boxed{\frac{1}{2}}"), r"\boxed{\frac{1}{3}}") is False

    def test_wrong_sign(self):
        assert score_response_scorer(gold(r"\boxed{5}"), r"\boxed{-5}") is False

    def test_off_by_one(self):
        assert score_response_scorer(gold(r"\boxed{100}"), r"\boxed{99}") is False


# ---------------------------------------------------------------------------
# 7. FALSE POSITIVE ELIMINATION: the critical fix
#    These all scored as "correct" with the old lenient parse() approach.
#    They must ALL score False with the new boxed-required approach.
# ---------------------------------------------------------------------------

class TestFalsePositiveElimination:
    """The whole point of this fix: garbage text with coincidental numbers
    must NOT score as correct, even when the number matches the gold answer."""

    def test_bare_number_matching_gold(self):
        """Gold is 2, garbage contains '2' — was a false positive."""
        assert score_response_scorer(gold(r"\boxed{2}"), "blah blah 2 blah") is False

    def test_answer_pattern_no_boxed(self):
        """'The answer is 2' without \\boxed{} — was a false positive."""
        assert score_response_scorer(gold(r"\boxed{2}"), "The answer is 2") is False

    def test_final_answer_pattern_no_boxed(self):
        """'The final answer is $2$' — was a false positive."""
        assert score_response_scorer(gold(r"\boxed{2}"), "The final answer is $2$") is False

    def test_zero_in_garbage(self):
        """Gold is 0, garbage contains '0'. Most common false positive."""
        assert score_response_scorer(gold(r"\boxed{0}"), "blah 0 blah stuff") is False

    def test_one_in_garbage(self):
        """Gold is 1, garbage mentions '1'."""
        assert score_response_scorer(gold(r"\boxed{1}"), "Step 1: do stuff") is False

    def test_small_int_in_garbage(self):
        """Gold is 5, garbage mentions '5'."""
        assert score_response_scorer(gold(r"\boxed{5}"), "There are 5 items listed") is False

    def test_dollar_sign_number_no_boxed(self):
        """Gold is 42, response has $42$ but no boxed."""
        assert score_response_scorer(gold(r"\boxed{42}"), "We get $42$ as the result") is False

    def test_realistic_garbage_sample(self):
        """Actual garbage from our uncontaminated 344M model."""
        garbage = (
            "Show your answer (b) in the given text. (c) How many vertical "
            "asymptotes in the answer? Discuss your answer in the following "
            "questions: 1. Report the data from bank's website in SACS."
        )
        assert score_response_scorer(gold(r"\boxed{2}"), garbage) is False

    def test_garbage_with_matching_number(self):
        """Garbage that coincidentally contains the right number."""
        garbage = "More total data given: 2 Formats feel free to prepare"
        assert score_response_scorer(gold(r"\boxed{2}"), garbage) is False

    def test_number_in_equation_no_boxed(self):
        """Mathematical-looking text but no \\boxed{}."""
        text = "x = 2, so the answer = 2"
        assert score_response_scorer(gold(r"\boxed{2}"), text) is False


# ---------------------------------------------------------------------------
# 8. score_response: no boxed at all (should return False)
# ---------------------------------------------------------------------------

class TestScoreResponseNoBoxed:
    """Responses without \\boxed{} must always score False."""

    def test_empty_response(self):
        assert score_response_scorer(gold(r"\boxed{42}"), "") is False

    def test_whitespace_only(self):
        assert score_response_scorer(gold(r"\boxed{42}"), "   \n\n  ") is False

    def test_random_text(self):
        assert score_response_scorer(gold(r"\boxed{42}"), "Hello world foo bar baz") is False

    def test_latex_without_boxed(self):
        assert score_response_scorer(gold(r"\boxed{42}"), r"$x = 42$, $y = \frac{1}{2}$") is False


# ---------------------------------------------------------------------------
# 9. Equivalence: both scripts must produce identical results
# ---------------------------------------------------------------------------

class TestEquivalenceBetweenScripts:
    """extract_boxed_answer and score_response must be identical
    in score_pass_at_k.py and incremental_scorer.py."""

    EXTRACTION_CASES = [
        r"\boxed{42}",
        r"The answer is \boxed{\frac{1}{2}}",
        r"\boxed{-\frac{1}{8}}",
        r"blah blah 2 blah",
        "",
        r"\boxed{42",
        r"\boxed{x^{2} + 1}",
        r"\boxed{1} then \boxed{2}",
    ]

    @pytest.mark.parametrize("text", EXTRACTION_CASES)
    def test_extract_boxed_identical(self, text):
        assert extract_boxed_scorer(text) == extract_boxed_incremental(text)

    SCORING_CASES = [
        (r"\boxed{2}", r"\boxed{2}"),
        (r"\boxed{2}", "blah 2 blah"),
        (r"\boxed{2}", r"\boxed{3}"),
        (r"\boxed{\frac{1}{2}}", r"Answer: \boxed{\frac{1}{2}}"),
        (r"\boxed{0}", "0"),
        (r"\boxed{42}", ""),
    ]

    @pytest.mark.parametrize("gold_str,response", SCORING_CASES)
    def test_score_response_identical(self, gold_str, response):
        g = gold(gold_str)
        assert score_response_scorer(g, response) == score_response_incremental(g, response)


# ---------------------------------------------------------------------------
# 10. Integration: verify math-verify equivalences work through our pipeline
# ---------------------------------------------------------------------------

class TestMathVerifyIntegration:
    """Ensure math-verify's equivalence checking still works when we
    re-wrap extracted content as \\boxed{content}."""

    def test_equivalent_fraction_representations(self):
        """\\frac{2}{4} should equal \\frac{1}{2}."""
        g = gold(r"\boxed{\frac{1}{2}}")
        assert score_response_scorer(g, r"\boxed{\frac{2}{4}}") is True

    def test_integer_vs_fraction(self):
        """2 should equal \\frac{4}{2}."""
        g = gold(r"\boxed{2}")
        assert score_response_scorer(g, r"\boxed{\frac{4}{2}}") is True

    def test_negative_fraction_equivalence(self):
        """-\\frac{1}{2} should equal \\frac{-1}{2}."""
        g = gold(r"\boxed{-\frac{1}{2}}")
        assert score_response_scorer(g, r"\boxed{\frac{-1}{2}}") is True

    def test_squared_expression(self):
        g = gold(r"\boxed{4}")
        assert score_response_scorer(g, r"\boxed{2^{2}}") is True

    def test_non_equivalent_fractions(self):
        g = gold(r"\boxed{\frac{1}{2}}")
        assert score_response_scorer(g, r"\boxed{\frac{1}{3}}") is False


# ---------------------------------------------------------------------------
# 11. Edge cases from real MATH benchmark gold solutions
# ---------------------------------------------------------------------------

class TestRealMATHFormats:
    """Test with formats actually found in MATH benchmark gold solutions."""

    def test_boxed_with_trailing_period(self):
        """Common pattern: $\\boxed{4}$."""
        g = gold(r"\boxed{4}")
        response = r"the greatest vertical distance is $\boxed{4}$ (achieved for all $x$ from $8$ to $12$)."
        assert score_response_scorer(g, response) is True

    def test_boxed_with_trailing_period_and_bracket(self):
        """Pattern: \\boxed{4}."""
        g = gold(r"\boxed{4}")
        response = r"The sum is \[0+4=\boxed{4}.\]"
        assert score_response_scorer(g, response) is True

    def test_boxed_in_dollar_signs(self):
        g = gold(r"\boxed{1}")
        response = r"solving to $k=\boxed{1}$."
        assert score_response_scorer(g, response) is True

    def test_long_solution_with_boxed_at_end(self):
        """Realistic: long work followed by boxed answer."""
        g = gold(r"\boxed{2}")
        response = (
            "We need to find the number of vertical asymptotes of "
            r"$\frac{x-1}{x^2+x-6}$. Factoring the denominator, "
            r"$x^2+x-6 = (x+3)(x-2)$. The numerator $x-1$ doesn't "
            r"cancel either factor, so there are vertical asymptotes at "
            r"$x = 2$ and $x = -3$. Therefore, the graph has $\boxed{2}$ "
            "vertical asymptotes."
        )
        assert score_response_scorer(g, response) is True


# ---------------------------------------------------------------------------
# 12. Verify the old scoring behavior was indeed wrong
#     (using raw parse() on garbage text)
# ---------------------------------------------------------------------------

class TestOldScoringWasWrong:
    """Demonstrate that raw parse() on garbage text DOES extract numbers,
    confirming the false positive mechanism we fixed."""

    def test_parse_extracts_bare_number(self):
        """parse() on 'blah 2 blah' extracts 2 — this is the root cause."""
        result = parse("blah blah 2 blah")
        # parse() returns a non-empty list when it finds something
        assert len(result) > 0, "parse() should extract bare numbers (showing the bug)"

    def test_old_approach_false_positive(self):
        """The OLD approach: verify(gold=parse(gold), target=parse(garbage))
        would return True when garbage contains the right number."""
        gold_parsed = parse(r"\boxed{2}")
        garbage_parsed = parse("blah blah 2 blah")
        # This SHOULD be True, demonstrating the false positive
        old_result = bool(verify(gold=gold_parsed, target=garbage_parsed))
        assert old_result is True, "Old approach should produce false positives"

    def test_new_approach_no_false_positive(self):
        """Our fix correctly rejects the same garbage."""
        assert score_response_scorer(gold(r"\boxed{2}"), "blah blah 2 blah") is False


# ---------------------------------------------------------------------------
# 13. Stress test: various \\boxed{} positions and contexts
# ---------------------------------------------------------------------------

class TestBoxedPositionVariants:
    """Ensure extraction works regardless of surrounding context."""

    def test_boxed_after_newlines(self):
        text = "Work:\n\nStep 1\nStep 2\n\n\\boxed{42}"
        assert extract_boxed_scorer(text) == "42"

    def test_boxed_in_align_environment(self):
        text = r"\begin{align*} x &= 3 \\ y &= \boxed{7} \end{align*}"
        assert extract_boxed_scorer(text) == "7"

    def test_boxed_surrounded_by_dollars(self):
        text = r"$\boxed{42}$"
        assert extract_boxed_scorer(text) == "42"

    def test_boxed_in_display_math(self):
        text = r"\[\boxed{42}\]"
        assert extract_boxed_scorer(text) == "42"

    def test_very_long_text_with_boxed_at_end(self):
        text = "x " * 10000 + r"\boxed{42}"
        assert extract_boxed_scorer(text) == "42"

    def test_boxed_with_spaces_inside(self):
        text = r"\boxed{ 42 }"
        assert extract_boxed_scorer(text) == " 42 "
        # And scoring should still work via parse()
        g = gold(r"\boxed{42}")
        assert score_response_scorer(g, text) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
