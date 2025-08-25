"""
Centralized RAG (Retrieval-Augmented Generation) prompts for NovaEval.

This module contains all prompts used by RAG-related scorers to ensure consistency,
maintainability, and easy updates across the evaluation system.
"""

from typing import Any


class RAGPrompts:
    """Centralized collection of RAG evaluation prompts."""

    # Numerical relevance scoring prompts (1-5 scale)
    NUMERICAL_CHUNK_RELEVANCE_1_5 = """
Question: {question}

Context chunk: {chunk}

Is this context chunk relevant for answering the question?
Rate the relevance on a scale of 1-5 where:
1 = Not relevant at all
2 = Slightly relevant
3 = Moderately relevant
4 = Highly relevant
5 = Extremely relevant

Provide your rating and a brief explanation.

Format:
Rating: [1-5]
Explanation: [Brief explanation]
"""

    # Numerical relevance scoring prompts (0-10 scale)
    NUMERICAL_CHUNK_RELEVANCE_0_10 = """
Question: {question}

Context chunk: {chunk}

Rate the relevance from 0-10 where:
0: Completely irrelevant
5: Somewhat relevant
10: Highly relevant and directly addresses the query

Determine if this chunk is relevant to answering the query.
A chunk is relevant if it contains information that helps answer the query.

Provide only the numerical score (0-10):

Format:
Rating: [0-10]
"""

    # Total relevant chunks estimation
    ESTIMATE_TOTAL_RELEVANT = """
Based on the query and the retrieved chunks, estimate how many relevant chunks might exist in total.

Query: {query}
Retrieved Chunks: {num_retrieved} chunks

Consider:
1. The complexity of the query
2. The number of retrieved chunks
3. Whether the query likely requires more information than what's retrieved

Respond with a JSON object in this exact format:
{{
    "estimated_total": <number>,
    "reasoning": "brief explanation"
}}
"""

    @classmethod
    def format_prompt(cls, prompt_template: str, **kwargs: Any) -> str:
        """Format a prompt template with the given parameters."""
        return prompt_template.format(**kwargs)

    @classmethod
    def get_numerical_chunk_relevance_1_5(cls, question: str, chunk: str) -> str:
        """Get formatted 1-5 scale chunk relevance prompt."""
        return cls.format_prompt(
            cls.NUMERICAL_CHUNK_RELEVANCE_1_5, question=question, chunk=chunk
        )

    @classmethod
    def get_numerical_chunk_relevance_0_10(cls, question: str, chunk: str) -> str:
        """Get formatted 0-10 scale chunk relevance prompt."""
        return cls.format_prompt(
            cls.NUMERICAL_CHUNK_RELEVANCE_0_10, question=question, chunk=chunk
        )

    @classmethod
    def get_estimate_total_relevant(cls, query: str, num_retrieved: int) -> str:
        """Get formatted total relevant chunks estimation prompt."""
        return cls.format_prompt(
            cls.ESTIMATE_TOTAL_RELEVANT, query=query, num_retrieved=num_retrieved
        )


def get_numerical_chunk_relevance_prompt_1_5(question: str, chunk: str) -> str:
    """Get 1-5 scale chunk relevance prompt (backward compatibility)."""
    return RAGPrompts.get_numerical_chunk_relevance_1_5(question, chunk)


def get_numerical_chunk_relevance_prompt_0_10(question: str, chunk: str) -> str:
    """Get 0-10 scale chunk relevance prompt (backward compatibility)."""
    return RAGPrompts.get_numerical_chunk_relevance_0_10(question, chunk)


def get_estimate_total_relevant_prompt(query: str, num_retrieved: int) -> str:
    """Get total relevant chunks estimation prompt (backward compatibility)."""
    return RAGPrompts.get_estimate_total_relevant(query, num_retrieved)
