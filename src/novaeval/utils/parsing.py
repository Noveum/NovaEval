"""Utility functions for parsing LLM responses."""

from typing import Optional


def parse_claims(text: str) -> list[str]:
    """
    Parse claims from LLM response text.

    Extracts numbered or bulleted claims from a text response.
    Supports formats like "1. claim", "2. claim", "- claim", "* claim".

    Args:
        text: The text response containing claims

    Returns:
        List of extracted claims
    """
    claims = []
    lines = text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith("-") or line.startswith("*")):
            # Remove numbering and bullet points
            claim = line
            for prefix in [
                "1.",
                "2.",
                "3.",
                "4.",
                "5.",
                "6.",
                "7.",
                "8.",
                "9.",
                "10.",
                "-",
                "*",
            ]:
                if claim.startswith(prefix):
                    claim = claim[len(prefix) :].strip()
                    break

            if claim:
                claims.append(claim)

    return claims


def parse_simple_claims(
    text: str, min_length: int = 10, max_claims: Optional[int] = None
) -> list[str]:
    """
    Parse claims by simple sentence splitting.

    Args:
        text: The text to parse
        min_length: Minimum length for a claim to be included
        max_claims: Maximum number of claims to return

    Returns:
        List of extracted claims
    """
    sentences = text.split(".")
    claims = [s.strip() for s in sentences if len(s.strip()) > min_length]

    if max_claims:
        claims = claims[:max_claims]

    return claims
