"""
Tests for parsing utility functions.

This module tests the parsing utilities used for LLM response processing.
"""

import pytest

from src.novaeval.utils.parsing import parse_claims, parse_simple_claims

pytestmark = pytest.mark.unit


class TestParseClaims:
    """Test the parse_claims function."""

    def test_parse_numbered_claims(self):
        """Test parsing numbered claims."""
        text = """
        1. First claim here
        2. Second claim here
        3. Third claim here
        """

        result = parse_claims(text)

        assert len(result) == 3
        assert result[0] == "First claim here"
        assert result[1] == "Second claim here"
        assert result[2] == "Third claim here"

    def test_parse_bullet_dash_claims(self):
        """Test parsing claims with dash bullets."""
        text = """
        - First bullet claim
        - Second bullet claim
        - Third bullet claim
        """

        result = parse_claims(text)

        assert len(result) == 3
        assert result[0] == "First bullet claim"
        assert result[1] == "Second bullet claim"
        assert result[2] == "Third bullet claim"

    def test_parse_bullet_asterisk_claims(self):
        """Test parsing claims with asterisk bullets."""
        text = """
        * First asterisk claim
        * Second asterisk claim
        * Third asterisk claim
        """

        result = parse_claims(text)

        assert len(result) == 3
        assert result[0] == "First asterisk claim"
        assert result[1] == "Second asterisk claim"
        assert result[2] == "Third asterisk claim"

    def test_parse_mixed_format_claims(self):
        """Test parsing claims with mixed formats."""
        text = """
        1. Numbered claim
        - Dash claim
        * Asterisk claim
        2. Another numbered claim
        """

        result = parse_claims(text)

        assert len(result) == 4
        assert result[0] == "Numbered claim"
        assert result[1] == "Dash claim"
        assert result[2] == "Asterisk claim"
        assert result[3] == "Another numbered claim"

    def test_parse_claims_with_empty_lines(self):
        """Test parsing claims with empty lines and whitespace."""
        text = """

        1. First claim

        2. Second claim


        3. Third claim

        """

        result = parse_claims(text)

        assert len(result) == 3
        assert result[0] == "First claim"
        assert result[1] == "Second claim"
        assert result[2] == "Third claim"

    def test_parse_claims_high_numbers(self):
        """Test parsing claims with numbers higher than 10."""
        text = """
        11. Eleventh claim
        12. Twelfth claim
        15. Fifteenth claim
        """

        result = parse_claims(text)

        # Numbers > 10 won't be stripped by the current implementation
        assert len(result) == 3
        assert "11. Eleventh claim" in result[0]  # Number not stripped
        assert "12. Twelfth claim" in result[1]  # Number not stripped
        assert "15. Fifteenth claim" in result[2]  # Number not stripped

    def test_parse_claims_no_claims(self):
        """Test parsing text with no claims."""
        text = """
        This is just regular text without any numbered or bulleted items.
        It has multiple lines but no claims to extract.
        """

        result = parse_claims(text)

        assert result == []

    def test_parse_claims_empty_text(self):
        """Test parsing empty text."""
        result = parse_claims("")
        assert result == []

    def test_parse_claims_whitespace_only(self):
        """Test parsing text with only whitespace."""
        result = parse_claims("   \n  \t  \n   ")
        assert result == []

    def test_parse_claims_empty_claim_lines(self):
        """Test parsing with empty claim lines (just numbers/bullets)."""
        text = """
        1.
        2. Valid claim
        -
        * Another valid claim
        3.
        """

        result = parse_claims(text)

        assert len(result) == 2
        assert result[0] == "Valid claim"
        assert result[1] == "Another valid claim"

    def test_parse_claims_with_extra_whitespace(self):
        """Test parsing claims with extra whitespace around numbers/bullets."""
        text = """
        1.    Claim with extra spaces
        2.	Claim with tab
           3. Indented claim
        """

        result = parse_claims(text)

        assert len(result) == 3
        assert result[0] == "Claim with extra spaces"
        assert result[1] == "Claim with tab"
        assert result[2] == "Indented claim"


class TestParseSimpleClaims:
    """Test the parse_simple_claims function."""

    def test_parse_simple_claims_basic(self):
        """Test basic sentence splitting."""
        text = "This is the first sentence. This is the second sentence. This is the third sentence."

        result = parse_simple_claims(text, min_length=10)

        assert len(result) == 3
        assert result[0] == "This is the first sentence"
        assert result[1] == "This is the second sentence"
        assert result[2] == "This is the third sentence"

    def test_parse_simple_claims_min_length_filter(self):
        """Test minimum length filtering."""
        text = "Short. This is a longer sentence that meets the minimum length. Tiny."

        result = parse_simple_claims(text, min_length=20)

        assert len(result) == 1
        assert result[0] == "This is a longer sentence that meets the minimum length"

    def test_parse_simple_claims_max_claims_limit(self):
        """Test maximum claims limit."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."

        result = parse_simple_claims(text, min_length=5, max_claims=3)

        assert len(result) == 3
        assert result[0] == "First sentence"
        assert result[1] == "Second sentence"
        assert result[2] == "Third sentence"

    def test_parse_simple_claims_no_max_limit(self):
        """Test with no maximum claims limit."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."

        result = parse_simple_claims(text, min_length=5, max_claims=None)

        assert len(result) == 4

    def test_parse_simple_claims_empty_text(self):
        """Test parsing empty text."""
        result = parse_simple_claims("", min_length=10)
        assert result == []

    def test_parse_simple_claims_no_periods(self):
        """Test text without periods."""
        text = "This is text without any periods so it should be one long claim"

        result = parse_simple_claims(text, min_length=10)

        assert len(result) == 1
        assert (
            result[0]
            == "This is text without any periods so it should be one long claim"
        )

    def test_parse_simple_claims_multiple_periods(self):
        """Test text with multiple consecutive periods."""
        text = "First sentence.. Second sentence... Third sentence...."

        result = parse_simple_claims(text, min_length=5)

        # Should split on each period, creating empty strings that get filtered
        expected_claims = ["First sentence", "Second sentence", "Third sentence"]
        assert len(result) >= 3
        for claim in expected_claims:
            assert claim in result

    def test_parse_simple_claims_whitespace_handling(self):
        """Test handling of whitespace around sentences."""
        text = "  First sentence.   Second sentence.  Third sentence.  "

        result = parse_simple_claims(text, min_length=5)

        assert len(result) == 3
        assert result[0] == "First sentence"
        assert result[1] == "Second sentence"
        assert result[2] == "Third sentence"

    def test_parse_simple_claims_zero_min_length(self):
        """Test with zero minimum length."""
        text = "A. B. C. Longer sentence."

        result = parse_simple_claims(text, min_length=0)

        assert len(result) == 4
        assert "A" in result
        assert "B" in result
        assert "C" in result
        assert "Longer sentence" in result

    def test_parse_simple_claims_max_claims_zero(self):
        """Test with max_claims set to zero."""
        text = "First sentence. Second sentence. Third sentence."

        result = parse_simple_claims(text, min_length=5, max_claims=0)

        # When max_claims is 0 (falsy), no slicing occurs, returns all claims
        assert len(result) == 3
        assert result[0] == "First sentence"
        assert result[1] == "Second sentence"
        assert result[2] == "Third sentence"


class TestEdgeCases:
    """Test edge cases for parsing functions."""

    def test_parse_claims_unicode_text(self):
        """Test parsing claims with unicode characters."""
        text = """
        1. Claim with émojis 🚀 and ünïcödé characters
        2. Another claim with 中文 characters
        """

        result = parse_claims(text)

        assert len(result) == 2
        assert "émojis 🚀 and ünïcödé characters" in result[0]
        assert "中文 characters" in result[1]

    def test_parse_simple_claims_unicode_text(self):
        """Test simple claims parsing with unicode."""
        text = "First 🚀 sentence. Second ünïcödé sentence. Third 中文 sentence."

        result = parse_simple_claims(text, min_length=10)

        assert len(result) == 3
        assert "🚀" in result[0]
        assert "ünïcödé" in result[1]
        assert "中文" in result[2]

    def test_parse_claims_very_long_text(self):
        """Test parsing very long text."""
        long_claim = "This is a very long claim " * 100
        text = f"1. {long_claim}\n2. Short claim"

        result = parse_claims(text)

        assert len(result) == 2
        assert len(result[0]) > 1000  # Very long claim
        assert result[1] == "Short claim"
