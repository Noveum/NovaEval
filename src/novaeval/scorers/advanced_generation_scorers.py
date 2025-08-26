"""
Advanced Generation Evaluation Scorers for RAG.

This module implements sophisticated generation evaluation scorers specifically designed for RAG scenarios,
focusing on context-conditioned generation quality.

Key Features:
- Context-Aware Generation Scorers
- Hallucination Detection
- Answer Quality Enhancement
- Multi-Context Integration
- Domain-Specific Evaluation
"""

import re
from typing import Any, Optional, Union

from novaeval.scorers.base import BaseScorer, ScoreResult
from novaeval.scorers.rag_prompts import RAGPrompts
from novaeval.utils.llm import call_llm
from novaeval.utils.parsing import parse_claims


# CONTEXT-AWARE GENERATION SCORERS
class BiasDetectionScorer(BaseScorer):
    """
    Bias detection in generated content.
    """

    def __init__(
        self, model: Any, threshold: float = 0.8, max_score: float = 5.0, **kwargs: Any
    ) -> None:
        super().__init__(name="BiasDetectionScorer", **kwargs)
        self.threshold = threshold
        self.model = model
        self.max_score = max_score  # Maximum score for inversion calculation

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        if not output_text:
            return ScoreResult(
                score=0.0, passed=False, reasoning="No answer provided", metadata={}
            )

        prompt = RAGPrompts.get_bias_detection_evaluation(
            input_text=input_text, output_text=output_text
        )

        try:
            response = await self._call_model(prompt)
            result = self._parse_json_response(response)
            bias_score = result.get("bias_score", 1.0)

            # Invert the score (lower bias = higher quality)
            quality_score = (self.max_score + 1 - bias_score) / self.max_score
            passed = quality_score >= self.threshold

            reasoning = f"Bias level: {bias_score}/5, Quality: {quality_score:.3f}"

            return ScoreResult(
                score=quality_score,
                passed=passed,
                reasoning=reasoning,
                metadata={
                    "bias_score": bias_score,
                    "bias_types": result.get("bias_types", []),
                    "reasoning": result.get("reasoning", ""),
                    "confidence": result.get("confidence", 0.0),
                    "specific_examples": result.get("specific_examples", []),
                    "recommendations": result.get("recommendations", ""),
                },
            )

        except Exception as e:
            return ScoreResult(
                score=0.0, passed=False, reasoning=f"Error: {e!s}", metadata={}
            )

    async def _call_model(self, prompt: str) -> str:
        import asyncio

        return await asyncio.to_thread(call_llm, self.model, prompt)

    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON response from model."""
        import json
        import re

        # Try to extract JSON from response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: try to parse the entire response as JSON
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            # Final fallback: extract bias score from text
            numbers = re.findall(r"\b(\d+)\b", response)
            bias_score = float(numbers[-1]) if numbers else 1.0
            return {
                "bias_score": bias_score,
                "bias_types": [],
                "reasoning": "Fallback parsing used",
            }

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Union[float, dict[str, float]]:
        import asyncio

        # Extract context from dict if available
        context_text = context.get("context") if context else None

        # Run async evaluation
        result = asyncio.run(
            self.evaluate(
                input_text=ground_truth, output_text=prediction, context=context_text
            )
        )

        if hasattr(result, "score"):
            return {"score": result.score}
        else:
            return {"score": result.score if hasattr(result, "score") else 0.0}


# HALLUCINATION DETECTION SCORERS
class FactualAccuracyScorer(BaseScorer):
    """
    Verify factual claims against contexts.
    """

    def __init__(
        self, model: Any, threshold: float = 0.8, max_score: float = 5.0, **kwargs: Any
    ) -> None:
        super().__init__(name="FactualAccuracyScorer", **kwargs)
        self.threshold = threshold
        self.model = model
        self.max_score = max_score  # Maximum score for normalization

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        if not output_text or not context:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="No answer or context provided",
                metadata={},
            )

        prompt = RAGPrompts.get_factual_accuracy_evaluation(
            context=context, output_text=output_text
        )

        try:
            response = await self._call_model(prompt)
            result = self._parse_json_response(response)
            score = result.get("accuracy_score", 3.0)
            normalized_score = score / self.max_score
            passed = normalized_score >= self.threshold

            reasoning = f"Factual accuracy: {normalized_score:.3f} ({score}/5)"

            return ScoreResult(
                score=normalized_score,
                passed=passed,
                reasoning=reasoning,
                metadata={
                    "raw_score": score,
                    "issues": result.get("issues", []),
                    "reasoning": result.get("reasoning", ""),
                    "confidence": result.get("confidence", 0.0),
                    "supported_claims": result.get("supported_claims", []),
                    "unsupported_claims": result.get("unsupported_claims", []),
                    "recommendations": result.get("recommendations", ""),
                },
            )

        except Exception as e:
            return ScoreResult(
                score=0.0, passed=False, reasoning=f"Error: {e!s}", metadata={}
            )

    async def _call_model(self, prompt: str) -> str:
        import asyncio

        return await asyncio.to_thread(call_llm, self.model, prompt)

    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON response from model."""
        import json
        import re

        # Try to extract JSON from response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: try to parse the entire response as JSON
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            # Final fallback: extract score from text
            numbers = re.findall(r"\b(\d+)\b", response)
            score = float(numbers[-1]) if numbers else 3.0
            return {
                "accuracy_score": score,
                "issues": [],
                "reasoning": "Fallback parsing used",
            }

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Union[float, dict[str, float]]:
        import asyncio

        # Extract context from dict if available
        context_text = context.get("context") if context else None

        # Run async evaluation
        result = asyncio.run(
            self.evaluate(
                input_text=ground_truth, output_text=prediction, context=context_text
            )
        )

        if hasattr(result, "score"):

            return {"score": result.score}

        else:

            return {"score": result.score if hasattr(result, "score") else 0.0}


class ClaimVerificationScorer(BaseScorer):
    """
    Verify specific claims in generated answers.
    """

    def __init__(
        self, model: Any, threshold: float = 0.7, max_score: float = 5.0, **kwargs: Any
    ) -> None:
        super().__init__(name="ClaimVerificationScorer", **kwargs)
        self.threshold = threshold
        self.model = model
        self.max_score = max_score  # Maximum score for normalization

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        if not output_text:
            return ScoreResult(
                score=0.0, passed=False, reasoning="No answer provided", metadata={}
            )

        # Extract claims from the answer
        claims_prompt = RAGPrompts.get_claim_extraction_evaluation(
            output_text=output_text
        )

        try:
            claims_response = await self._call_model(claims_prompt)
            claims_result = self._parse_json_response(claims_response)
            claims = claims_result.get("claims", [])

            if not claims:
                return ScoreResult(
                    score=1.0,
                    passed=True,
                    reasoning="No specific claims found",
                    metadata={"claims": []},
                )

            # Verify each claim
            verified_claims = []
            total_score = 0.0

            for claim in claims:
                verification_prompt = RAGPrompts.get_claim_verification_evaluation(
                    context=context or "No context provided", claim=claim
                )

                verification_response = await self._call_model(verification_prompt)
                verification_result = self._parse_json_response(verification_response)
                score = verification_result.get("verification_score", 3.0)
                total_score += score
                verified_claims.append(
                    {
                        "claim": claim,
                        "score": score,
                        "supported": verification_result.get("supported", False),
                        "reasoning": verification_result.get("reasoning", ""),
                        "confidence": verification_result.get("confidence", 0.0),
                        "supporting_evidence": verification_result.get(
                            "supporting_evidence", []
                        ),
                        "contradicting_evidence": verification_result.get(
                            "contradicting_evidence", []
                        ),
                        "verification_method": verification_result.get(
                            "verification_method", ""
                        ),
                    }
                )

            avg_score = total_score / len(claims) / self.max_score  # Normalize to 0-1
            passed = avg_score >= self.threshold

            reasoning = (
                f"Verified {len(claims)} claims. Average verification: {avg_score:.3f}"
            )

            return ScoreResult(
                score=avg_score,
                passed=passed,
                reasoning=reasoning,
                metadata={
                    "verified_claims": verified_claims,
                    "total_claims": len(claims),
                    "claim_extraction": {
                        "reasoning": claims_result.get("reasoning", ""),
                        "confidence": claims_result.get("confidence", 0.0),
                        "claim_types": claims_result.get("claim_types", []),
                        "total_claims": claims_result.get("total_claims", len(claims)),
                    },
                },
            )

        except Exception as e:
            return ScoreResult(
                score=0.0, passed=False, reasoning=f"Error: {e!s}", metadata={}
            )

    async def _call_model(self, prompt: str) -> str:
        import asyncio

        return await asyncio.to_thread(call_llm, self.model, prompt)

    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON response from model."""
        import json
        import re

        # Try to extract JSON from response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: try to parse the entire response as JSON
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            # Final fallback: extract claims or score from text
            if "claims" in response.lower():
                # Extract claims from numbered list
                claims = []
                lines = response.split("\n")
                for line in lines:
                    if re.match(r"^\d+\.", line.strip()):
                        claim = line.strip().split(".", 1)[1].strip()
                        if claim:
                            claims.append(claim)
                return {"claims": claims, "reasoning": "Fallback parsing used"}
            else:
                # Extract verification score
                numbers = re.findall(r"\b(\d+)\b", response)
                score = float(numbers[-1]) if numbers else 3.0
                return {
                    "verification_score": score,
                    "supported": score >= 3,
                    "reasoning": "Fallback parsing used",
                }

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Union[float, dict[str, float]]:
        import asyncio

        # Extract context from dict if available
        context_text = context.get("context") if context else None

        # Run async evaluation
        result = asyncio.run(
            self.evaluate(
                input_text=ground_truth, output_text=prediction, context=context_text
            )
        )

        if hasattr(result, "score"):

            return {"score": result.score}

        else:

            return {"score": result.score if hasattr(result, "score") else 0.0}


# ANSWER COMPLETENESS AND RELEVANCE SCORERS
class InformationDensityScorer(BaseScorer):
    """
    Information richness evaluation.
    """

    def __init__(
        self, model: Any, threshold: float = 0.6, max_score: float = 5.0, **kwargs: Any
    ) -> None:
        super().__init__(name="InformationDensityScorer", **kwargs)
        self.threshold = threshold
        self.model = model
        self.max_score = max_score  # Maximum score for normalization

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        if not output_text:
            return ScoreResult(
                score=0.0, passed=False, reasoning="No answer provided", metadata={}
            )

        prompt = RAGPrompts.get_information_density_evaluation(
            input_text=input_text, output_text=output_text
        )

        try:
            response = await self._call_model(prompt)
            score = self._parse_density_score(response)
            normalized_score = score / self.max_score
            passed = normalized_score >= self.threshold

            reasoning = f"Information density: {normalized_score:.3f} ({score}/5)"

            return ScoreResult(
                score=normalized_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"raw_score": score},
            )

        except Exception as e:
            return ScoreResult(
                score=0.0, passed=False, reasoning=f"Error: {e!s}", metadata={}
            )

    async def _call_model(self, prompt: str) -> str:
        import asyncio

        return await asyncio.to_thread(call_llm, self.model, prompt)

    def _parse_density_score(self, response: str) -> float:
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 3.0  # Default to neutral

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Union[float, dict[str, float]]:
        import asyncio

        # Extract context from dict if available
        context_text = context.get("context") if context else None

        # Run async evaluation
        result = asyncio.run(
            self.evaluate(
                input_text=ground_truth, output_text=prediction, context=context_text
            )
        )

        if hasattr(result, "score"):

            return {"score": result.score}

        else:

            return {"score": result.score if hasattr(result, "score") else 0.0}


class ClarityAndCoherenceScorer(BaseScorer):
    """
    Answer readability and logic evaluation.
    """

    def __init__(
        self, model: Any, threshold: float = 0.7, max_score: float = 5.0, **kwargs: Any
    ) -> None:
        super().__init__(name="ClarityAndCoherenceScorer", **kwargs)
        self.threshold = threshold
        self.model = model
        self.max_score = max_score  # Maximum score for normalization

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        if not output_text:
            return ScoreResult(
                score=0.0, passed=False, reasoning="No answer provided", metadata={}
            )

        prompt = RAGPrompts.get_clarity_coherence_evaluation(
            input_text="", output_text=output_text
        )

        try:
            response = await self._call_model(prompt)
            score = self._parse_clarity_score(response)
            normalized_score = score / self.max_score
            passed = normalized_score >= self.threshold

            reasoning = f"Clarity and coherence: {normalized_score:.3f} ({score}/5)"

            return ScoreResult(
                score=normalized_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"raw_score": score},
            )

        except Exception as e:
            return ScoreResult(
                score=0.0, passed=False, reasoning=f"Error: {e!s}", metadata={}
            )

    async def _call_model(self, prompt: str) -> str:
        import asyncio

        return await asyncio.to_thread(call_llm, self.model, prompt)

    def _parse_clarity_score(self, response: str) -> float:
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 3.0  # Default to neutral

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Union[float, dict[str, float]]:
        import asyncio

        # Extract context from dict if available
        context_text = context.get("context") if context else None

        # Run async evaluation
        result = asyncio.run(
            self.evaluate(
                input_text=ground_truth, output_text=prediction, context=context_text
            )
        )

        if hasattr(result, "score"):

            return {"score": result.score}

        else:

            return {"score": result.score if hasattr(result, "score") else 0.0}


# MULTI-CONTEXT INTEGRATION SCORERS
class ConflictResolutionScorer(BaseScorer):
    """
    Handling contradictory information across contexts.
    """

    def __init__(
        self, model: Any, threshold: float = 0.7, max_score: float = 5.0, **kwargs: Any
    ) -> None:
        super().__init__(name="ConflictResolutionScorer", **kwargs)
        self.threshold = threshold
        self.model = model
        self.max_score = max_score  # Maximum score for normalization

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        if not context or not output_text:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="No context or output provided",
                metadata={},
            )

        # Split context into chunks to check for conflicts
        context_chunks = context.split("\n\n")
        if len(context_chunks) < 2:
            return ScoreResult(
                score=1.0,
                passed=True,
                reasoning="Single context provided",
                metadata={"chunks": 1},
            )

        prompt = RAGPrompts.get_context_faithfulness_evaluation(
            context=context, output_text=output_text
        )

        try:
            response = await self._call_model(prompt)
            score = self._parse_conflict_score(response)
            normalized_score = score / self.max_score
            passed = normalized_score >= self.threshold

            reasoning = f"Conflict resolution: {normalized_score:.3f} ({score}/5)"

            return ScoreResult(
                score=normalized_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"raw_score": score, "context_chunks": len(context_chunks)},
            )

        except Exception as e:
            return ScoreResult(
                score=0.0, passed=False, reasoning=f"Error: {e!s}", metadata={}
            )

    async def _call_model(self, prompt: str) -> str:
        import asyncio

        return await asyncio.to_thread(call_llm, self.model, prompt)

    def _parse_conflict_score(self, response: str) -> float:
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 3.0  # Default to neutral

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Union[float, dict[str, float]]:
        import asyncio

        # Extract context from dict if available
        context_text = context.get("context") if context else None

        # Run async evaluation
        result = asyncio.run(
            self.evaluate(
                input_text=ground_truth, output_text=prediction, context=context_text
            )
        )

        if hasattr(result, "score"):

            return {"score": result.score}

        else:

            return {"score": result.score if hasattr(result, "score") else 0.0}


class ContextPrioritizationScorer(BaseScorer):
    """
    Appropriate context weighting evaluation.
    """

    def __init__(
        self, model: Any, threshold: float = 0.6, max_score: float = 5.0, **kwargs: Any
    ) -> None:
        super().__init__(name="ContextPrioritizationScorer", **kwargs)
        self.threshold = threshold
        self.model = model
        self.max_score = max_score  # Maximum score for normalization

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        if not context or not output_text:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="No context or output provided",
                metadata={},
            )

        prompt = RAGPrompts.get_context_faithfulness_evaluation(
            context=context, output_text=output_text
        )

        try:
            response = await self._call_model(prompt)
            score = self._parse_prioritization_score(response)
            normalized_score = score / self.max_score
            passed = normalized_score >= self.threshold

            reasoning = f"Context prioritization: {normalized_score:.3f} ({score}/5)"

            return ScoreResult(
                score=normalized_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"raw_score": score},
            )

        except Exception as e:
            return ScoreResult(
                score=0.0, passed=False, reasoning=f"Error: {e!s}", metadata={}
            )

    async def _call_model(self, prompt: str) -> str:
        import asyncio

        return await asyncio.to_thread(call_llm, self.model, prompt)

    def _parse_prioritization_score(self, response: str) -> float:
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 3.0  # Default to neutral

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Union[float, dict[str, float]]:
        import asyncio

        # Extract context from dict if available
        context_text = context.get("context") if context else None

        # Run async evaluation
        result = asyncio.run(
            self.evaluate(
                input_text=ground_truth, output_text=prediction, context=context_text
            )
        )

        if hasattr(result, "score"):

            return {"score": result.score}

        else:

            return {"score": result.score if hasattr(result, "score") else 0.0}


class CitationQualityScorer(BaseScorer):
    """
    Quality of source references evaluation.
    """

    def __init__(
        self, model: Any, threshold: float = 0.6, max_score: float = 5.0, **kwargs: Any
    ) -> None:
        super().__init__(name="CitationQualityScorer", **kwargs)
        self.threshold = threshold
        self.model = model
        self.max_score = max_score  # Maximum score for normalization

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        if not output_text:
            return ScoreResult(
                score=0.0, passed=False, reasoning="No answer provided", metadata={}
            )

        prompt = RAGPrompts.get_source_attribution_evaluation(
            context="", output_text=output_text
        )

        try:
            response = await self._call_model(prompt)
            score = self._parse_citation_score(response)
            normalized_score = score / self.max_score
            passed = normalized_score >= self.threshold

            reasoning = f"Citation quality: {normalized_score:.3f} ({score}/5)"

            return ScoreResult(
                score=normalized_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"raw_score": score},
            )

        except Exception as e:
            return ScoreResult(
                score=0.0, passed=False, reasoning=f"Error: {e!s}", metadata={}
            )

    async def _call_model(self, prompt: str) -> str:
        import asyncio

        return await asyncio.to_thread(call_llm, self.model, prompt)

    def _parse_citation_score(self, response: str) -> float:
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 3.0  # Default to neutral

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Union[float, dict[str, float]]:
        import asyncio

        # Extract context from dict if available
        context_text = context.get("context") if context else None

        # Run async evaluation
        result = asyncio.run(
            self.evaluate(
                input_text=ground_truth, output_text=prediction, context=context_text
            )
        )

        if hasattr(result, "score"):

            return {"score": result.score}

        else:

            return {"score": result.score if hasattr(result, "score") else 0.0}


# DOMAIN-SPECIFIC EVALUATION SCORERS
class ToneConsistencyScorer(BaseScorer):
    """
    Appropriate tone for domain evaluation.
    """

    def __init__(
        self, model: Any, threshold: float = 0.7, max_score: float = 5.0, **kwargs: Any
    ) -> None:
        super().__init__(name="ToneConsistencyScorer", **kwargs)
        self.threshold = threshold
        self.model = model
        self.max_score = max_score  # Maximum score for normalization

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        if not output_text:
            return ScoreResult(
                score=0.0, passed=False, reasoning="No answer provided", metadata={}
            )

        # Use input_text as context if no context is provided
        # evaluation_context = context if context else input_text

        prompt = RAGPrompts.get_clarity_coherence_evaluation(
            input_text=input_text, output_text=output_text
        )

        try:
            response = await self._call_model(prompt)
            score = self._parse_tone_score(response)
            normalized_score = score / self.max_score
            passed = normalized_score >= self.threshold

            reasoning = f"Tone consistency: {normalized_score:.3f} ({score}/5)"

            return ScoreResult(
                score=normalized_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"raw_score": score},
            )

        except Exception as e:
            return ScoreResult(
                score=0.0, passed=False, reasoning=f"Error: {e!s}", metadata={}
            )

    async def _call_model(self, prompt: str) -> str:
        import asyncio

        return await asyncio.to_thread(call_llm, self.model, prompt)

    def _parse_tone_score(self, response: str) -> float:
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 3.0  # Default to neutral

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Union[float, dict[str, float]]:
        import asyncio

        # Extract context from dict if available
        context_text = context.get("context") if context else None

        # Run async evaluation
        result = asyncio.run(
            self.evaluate(
                input_text=ground_truth, output_text=prediction, context=context_text
            )
        )

        if hasattr(result, "score"):

            return {"score": result.score}

        else:

            return {"score": result.score if hasattr(result, "score") else 0.0}


class TerminologyConsistencyScorer(BaseScorer):
    """
    Consistent use of domain terms evaluation.
    """

    def __init__(
        self, model: Any, threshold: float = 0.7, max_score: float = 5.0, **kwargs: Any
    ) -> None:
        super().__init__(name="TerminologyConsistencyScorer", **kwargs)
        self.threshold = threshold
        self.model = model
        self.max_score = max_score  # Maximum score for normalization

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        if not output_text:
            return ScoreResult(
                score=0.0, passed=False, reasoning="No answer provided", metadata={}
            )

        prompt = RAGPrompts.get_clarity_coherence_evaluation(
            input_text="", output_text=output_text
        )

        try:
            response = await self._call_model(prompt)
            score = self._parse_terminology_score(response)
            normalized_score = score / self.max_score
            passed = normalized_score >= self.threshold

            reasoning = f"Terminology consistency: {normalized_score:.3f} ({score}/5)"

            return ScoreResult(
                score=normalized_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"raw_score": score},
            )

        except Exception as e:
            return ScoreResult(
                score=0.0, passed=False, reasoning=f"Error: {e!s}", metadata={}
            )

    async def _call_model(self, prompt: str) -> str:
        import asyncio

        return await asyncio.to_thread(call_llm, self.model, prompt)

    def _parse_terminology_score(self, response: str) -> float:
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 3.0  # Default to neutral

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Union[float, dict[str, float]]:
        import asyncio

        # Extract context from dict if available
        context_text = context.get("context") if context else None

        # Run async evaluation
        result = asyncio.run(
            self.evaluate(
                input_text=ground_truth, output_text=prediction, context=context_text
            )
        )

        if hasattr(result, "score"):

            return {"score": result.score}

        else:

            return {"score": result.score if hasattr(result, "score") else 0.0}


class ContextFaithfulnessScorerPP(BaseScorer):
    """
    Enhanced faithfulness detection with fine-grained analysis.
    Analyzes each claim in the answer against the provided context.
    """

    def __init__(
        self, model: Any, threshold: float = 0.8, max_score: float = 5.0, **kwargs: Any
    ) -> None:
        super().__init__(name="ContextFaithfulnessScorerPP", **kwargs)
        self.threshold = threshold
        self.model = model
        self.max_score = max_score  # Maximum score for normalization

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        if not context or not output_text:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="No context or output provided",
                metadata={},
            )

        # Extract claims from the answer
        claims_prompt = RAGPrompts.get_claim_extraction_evaluation(
            output_text=output_text
        )

        try:
            claims_response = await self._call_model(claims_prompt)
            claims = self._parse_claims(claims_response)

            if not claims:
                return ScoreResult(
                    score=1.0,
                    passed=True,
                    reasoning="No factual claims found",
                    metadata={"claims": []},
                )

            # Verify each claim against context
            verified_claims = []
            total_score = 0.0

            for _i, claim in enumerate(claims):
                verification_prompt = RAGPrompts.get_claim_verification_evaluation(
                    context=context, claim=claim
                )

                verification_response = await self._call_model(verification_prompt)
                score = self._parse_verification_score(verification_response)
                total_score += score
                verified_claims.append({"claim": claim, "score": score})

            avg_score = total_score / len(claims) / self.max_score  # Normalize to 0-1
            passed = avg_score >= self.threshold

            reasoning = (
                f"Verified {len(claims)} claims. Average faithfulness: {avg_score:.3f}"
            )

            return ScoreResult(
                score=avg_score,
                passed=passed,
                reasoning=reasoning,
                metadata={
                    "verified_claims": verified_claims,
                    "total_claims": len(claims),
                },
            )

        except Exception as e:
            return ScoreResult(
                score=0.0, passed=False, reasoning=f"Error: {e!s}", metadata={}
            )

    async def _call_model(self, prompt: str) -> str:
        import asyncio

        return await asyncio.to_thread(call_llm, self.model, prompt)

    def _parse_claims(self, text: str) -> list[str]:
        return parse_claims(text)

    def _parse_verification_score(self, response: str) -> float:
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 1.0  # Default to no hallucinations

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Union[float, dict[str, float]]:
        import asyncio

        # Extract context from dict if available
        context_text = context.get("context") if context else None

        # Run async evaluation
        result = asyncio.run(
            self.evaluate(
                input_text=ground_truth, output_text=prediction, context=context_text
            )
        )

        if hasattr(result, "score"):

            return {"score": result.score}

        else:

            return {"score": result.score if hasattr(result, "score") else 0.0}


class ContextGroundednessScorer(BaseScorer):
    """
    Ensures answers are grounded in provided context.
    Evaluates how well the answer is supported by the given context.
    """

    def __init__(
        self, model: Any, threshold: float = 0.7, max_score: float = 5.0, **kwargs: Any
    ) -> None:
        super().__init__(name="ContextGroundednessScorer", **kwargs)
        self.threshold = threshold
        self.model = model
        self.max_score = max_score  # Maximum score for normalization

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        if not context or not output_text:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="No context or output provided",
                metadata={},
            )

        prompt = RAGPrompts.get_context_faithfulness_evaluation(
            context=context, output_text=output_text
        )

        try:
            response = await self._call_model(prompt)
            score = self._parse_groundedness_score(response)
            normalized_score = score / self.max_score
            passed = normalized_score >= self.threshold

            reasoning = f"Groundedness score: {normalized_score:.3f} ({score}/5)"

            return ScoreResult(
                score=normalized_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"raw_score": score},
            )

        except Exception as e:
            return ScoreResult(
                score=0.0, passed=False, reasoning=f"Error: {e!s}", metadata={}
            )

    async def _call_model(self, prompt: str) -> str:
        import asyncio

        return await asyncio.to_thread(call_llm, self.model, prompt)

    def _parse_groundedness_score(self, response: str) -> float:
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 3.0  # Default to neutral

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Union[float, dict[str, float]]:
        import asyncio

        # Extract context from dict if available
        context_text = context.get("context") if context else None

        # Run async evaluation
        result = asyncio.run(
            self.evaluate(
                input_text=ground_truth, output_text=prediction, context=context_text
            )
        )

        if hasattr(result, "score"):

            return {"score": result.score}

        else:

            return {"score": result.score if hasattr(result, "score") else 0.0}


class ContextCompletenessScorer(BaseScorer):
    """
    Evaluates if context fully supports the answer.
    Checks whether the provided context contains all necessary information.
    """

    def __init__(
        self, model: Any, threshold: float = 0.6, max_score: float = 5.0, **kwargs: Any
    ) -> None:
        super().__init__(name="ContextCompletenessScorer", **kwargs)
        self.threshold = threshold
        self.model = model
        self.max_score = max_score  # Maximum score for normalization

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        if not context or not output_text:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="No context or output provided",
                metadata={},
            )

        prompt = RAGPrompts.get_answer_completeness_evaluation(
            input_text=input_text, context=context, output_text=output_text
        )

        try:
            response = await self._call_model(prompt)
            score = self._parse_completeness_score(response)
            normalized_score = score / self.max_score
            passed = normalized_score >= self.threshold

            reasoning = f"Context completeness: {normalized_score:.3f} ({score}/5)"

            return ScoreResult(
                score=normalized_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"raw_score": score},
            )

        except Exception as e:
            return ScoreResult(
                score=0.0, passed=False, reasoning=f"Error: {e!s}", metadata={}
            )

    async def _call_model(self, prompt: str) -> str:
        import asyncio

        return await asyncio.to_thread(call_llm, self.model, prompt)

    def _parse_completeness_score(self, response: str) -> float:
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 3.0  # Default to neutral

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Union[float, dict[str, float]]:
        import asyncio

        # Extract context from dict if available
        context_text = context.get("context") if context else None

        # Run async evaluation
        result = asyncio.run(
            self.evaluate(
                input_text=ground_truth, output_text=prediction, context=context_text
            )
        )

        if hasattr(result, "score"):

            return {"score": result.score}

        else:

            return {"score": result.score if hasattr(result, "score") else 0.0}


class ContextConsistencyScorer(BaseScorer):
    """
    Consistency across multiple contexts.
    Evaluates if the answer is consistent when multiple contexts are provided.
    """

    def __init__(
        self, model: Any, threshold: float = 0.7, max_score: float = 5.0, **kwargs: Any
    ) -> None:
        super().__init__(name="ContextConsistencyScorer", **kwargs)
        self.threshold = threshold
        self.model = model
        self.max_score = max_score  # Maximum score for normalization

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        if not context or not output_text:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="No context or output provided",
                metadata={},
            )

        # Split context into multiple chunks
        context_chunks = context.split("\n\n")
        if len(context_chunks) < 2:
            return ScoreResult(
                score=1.0,
                passed=True,
                reasoning="Single context provided",
                metadata={"chunks": 1},
            )

        # Evaluate consistency across chunks
        consistency_scores = []
        for _i, chunk in enumerate(context_chunks):
            prompt = RAGPrompts.get_context_faithfulness_evaluation(
                context=chunk, output_text=output_text
            )

            try:
                response = await self._call_model(prompt)
                score = self._parse_consistency_score(response)
                consistency_scores.append(score)
            except Exception:
                consistency_scores.append(3.0)  # Default to neutral

        avg_score = sum(consistency_scores) / len(consistency_scores) / self.max_score
        passed = avg_score >= self.threshold

        reasoning = f"Consistency across {len(context_chunks)} chunks: {avg_score:.3f}"

        return ScoreResult(
            score=avg_score,
            passed=passed,
            reasoning=reasoning,
            metadata={
                "consistency_scores": consistency_scores,
                "chunks": len(context_chunks),
            },
        )

    async def _call_model(self, prompt: str) -> str:
        import asyncio

        return await asyncio.to_thread(call_llm, self.model, prompt)

    def _parse_consistency_score(self, response: str) -> float:
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 3.0  # Default to neutral

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Union[float, dict[str, float]]:
        import asyncio

        # Extract context from dict if available
        context_text = context.get("context") if context else None

        # Run async evaluation
        result = asyncio.run(
            self.evaluate(
                input_text=ground_truth, output_text=prediction, context=context_text
            )
        )

        if hasattr(result, "score"):

            return {"score": result.score}

        else:

            return {"score": result.score if hasattr(result, "score") else 0.0}


class RAGAnswerQualityScorer(BaseScorer):
    """
    Comprehensive RAG generation evaluation.
    Evaluates the overall quality of RAG-generated answers.
    """

    def __init__(
        self, model: Any, threshold: float = 0.7, max_score: float = 5.0, **kwargs: Any
    ) -> None:
        super().__init__(name="RAGAnswerQualityScorer", **kwargs)
        self.threshold = threshold
        self.model = model
        self.max_score = max_score  # Maximum score for normalization

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        if not output_text:
            return ScoreResult(
                score=0.0, passed=False, reasoning="No answer provided", metadata={}
            )

        prompt = RAGPrompts.get_question_answer_alignment_evaluation(
            input_text=input_text, output_text=output_text
        )

        try:
            response = await self._call_model(prompt)
            score = self._parse_quality_score(response)
            normalized_score = score / self.max_score
            passed = normalized_score >= self.threshold

            reasoning = f"Answer quality: {normalized_score:.3f} ({score}/5)"

            return ScoreResult(
                score=normalized_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"raw_score": score},
            )

        except Exception as e:
            return ScoreResult(
                score=0.0, passed=False, reasoning=f"Error: {e!s}", metadata={}
            )

    async def _call_model(self, prompt: str) -> str:
        import asyncio

        return await asyncio.to_thread(call_llm, self.model, prompt)

    def _parse_quality_score(self, response: str) -> float:
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 3.0  # Default to neutral

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Union[float, dict[str, float]]:
        import asyncio

        # Extract context from dict if available
        context_text = context.get("context") if context else None

        # Run async evaluation
        result = asyncio.run(
            self.evaluate(
                input_text=ground_truth, output_text=prediction, context=context_text
            )
        )

        if hasattr(result, "score"):

            return {"score": result.score}

        else:

            return {"score": result.score if hasattr(result, "score") else 0.0}


class HallucinationDetectionScorer(BaseScorer):
    """
    Identify factual inconsistencies in generated answers.
    """

    def __init__(
        self, model: Any, threshold: float = 0.8, max_score: float = 5.0, **kwargs: Any
    ) -> None:
        super().__init__(name="HallucinationDetectionScorer", **kwargs)
        self.threshold = threshold
        self.model = model
        self.max_score = max_score  # Maximum score for inversion calculation

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        if not output_text:
            return ScoreResult(
                score=0.0, passed=False, reasoning="No answer provided", metadata={}
            )

        prompt = RAGPrompts.get_hallucination_detection_evaluation(
            context=context or "No context provided", output_text=output_text
        )

        try:
            response = await self._call_model(prompt)
            hallucination_score = self._parse_hallucination_score(response)
            # Invert the score (lower hallucination = higher quality)
            quality_score = (self.max_score + 1 - hallucination_score) / self.max_score
            passed = quality_score >= self.threshold

            reasoning = f"Hallucination level: {hallucination_score}/5, Quality: {quality_score:.3f}"

            return ScoreResult(
                score=quality_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"hallucination_score": hallucination_score},
            )

        except Exception as e:
            return ScoreResult(
                score=0.0, passed=False, reasoning=f"Error: {e!s}", metadata={}
            )

    async def _call_model(self, prompt: str) -> str:
        import asyncio

        return await asyncio.to_thread(call_llm, self.model, prompt)

    def _parse_hallucination_score(self, response: str) -> float:
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 1.0  # Default to no hallucinations

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Union[float, dict[str, float]]:
        import asyncio

        # Extract context from dict if available
        context_text = context.get("context") if context else None

        # Run async evaluation
        result = asyncio.run(
            self.evaluate(
                input_text=ground_truth, output_text=prediction, context=context_text
            )
        )

        if hasattr(result, "score"):

            return {"score": result.score}

        else:

            return {"score": result.score if hasattr(result, "score") else 0.0}


class SourceAttributionScorer(BaseScorer):
    """
    Proper citation and source attribution evaluation.
    """

    def __init__(
        self, model: Any, threshold: float = 0.6, max_score: float = 5.0, **kwargs: Any
    ) -> None:
        super().__init__(name="SourceAttributionScorer", **kwargs)
        self.threshold = threshold
        self.model = model
        self.max_score = max_score  # Maximum score for normalization

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        if not output_text:
            return ScoreResult(
                score=0.0, passed=False, reasoning="No answer provided", metadata={}
            )

        prompt = RAGPrompts.get_source_attribution_evaluation(
            context=context or "No context provided", output_text=output_text
        )

        try:
            response = await self._call_model(prompt)
            score = self._parse_attribution_score(response)
            normalized_score = score / self.max_score
            passed = normalized_score >= self.threshold

            reasoning = f"Source attribution: {normalized_score:.3f} ({score}/5)"

            return ScoreResult(
                score=normalized_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"raw_score": score},
            )

        except Exception as e:
            return ScoreResult(
                score=0.0, passed=False, reasoning=f"Error: {e!s}", metadata={}
            )

    async def _call_model(self, prompt: str) -> str:
        import asyncio

        return await asyncio.to_thread(call_llm, self.model, prompt)

    def _parse_attribution_score(self, response: str) -> float:
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 3.0  # Default to neutral

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Union[float, dict[str, float]]:
        import asyncio

        # Extract context from dict if available
        context_text = context.get("context") if context else None

        # Run async evaluation
        result = asyncio.run(
            self.evaluate(
                input_text=ground_truth, output_text=prediction, context=context_text
            )
        )

        if hasattr(result, "score"):

            return {"score": result.score}

        else:

            return {"score": result.score if hasattr(result, "score") else 0.0}


class AnswerCompletenessScorer(BaseScorer):
    """
    Comprehensive answer coverage evaluation.
    """

    def __init__(
        self, model: Any, threshold: float = 0.7, max_score: float = 5.0, **kwargs: Any
    ) -> None:
        super().__init__(name="AnswerCompletenessScorer", **kwargs)
        self.threshold = threshold
        self.model = model
        self.max_score = max_score  # Maximum score for normalization

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        if not output_text:
            return ScoreResult(
                score=0.0, passed=False, reasoning="No answer provided", metadata={}
            )

        prompt = RAGPrompts.get_answer_completeness_evaluation(
            input_text=input_text, context="", output_text=output_text
        )

        try:
            response = await self._call_model(prompt)
            score = self._parse_completeness_score(response)
            normalized_score = score / self.max_score
            passed = normalized_score >= self.threshold

            reasoning = f"Answer completeness: {normalized_score:.3f} ({score}/5)"

            return ScoreResult(
                score=normalized_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"raw_score": score},
            )

        except Exception as e:
            return ScoreResult(
                score=0.0, passed=False, reasoning=f"Error: {e!s}", metadata={}
            )

    async def _call_model(self, prompt: str) -> str:
        import asyncio

        return await asyncio.to_thread(call_llm, self.model, prompt)

    def _parse_completeness_score(self, response: str) -> float:
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 3.0  # Default to neutral

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Union[float, dict[str, float]]:
        import asyncio

        # Extract context from dict if available
        context_text = context.get("context") if context else None

        # Run async evaluation
        result = asyncio.run(
            self.evaluate(
                input_text=ground_truth, output_text=prediction, context=context_text
            )
        )

        if hasattr(result, "score"):

            return {"score": result.score}

        else:

            return {"score": result.score if hasattr(result, "score") else 0.0}


class QuestionAnswerAlignmentScorer(BaseScorer):
    """
    Direct question addressing evaluation.
    """

    def __init__(
        self, model: Any, threshold: float = 0.7, max_score: float = 5.0, **kwargs: Any
    ) -> None:
        super().__init__(name="QuestionAnswerAlignmentScorer", **kwargs)
        self.threshold = threshold
        self.model = model
        self.max_score = max_score  # Maximum score for normalization

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        if not output_text:
            return ScoreResult(
                score=0.0, passed=False, reasoning="No answer provided", metadata={}
            )

        prompt = RAGPrompts.get_question_answer_alignment_evaluation(
            input_text=input_text, output_text=output_text
        )

        try:
            response = await self._call_model(prompt)
            score = self._parse_alignment_score(response)
            normalized_score = score / self.max_score
            passed = normalized_score >= self.threshold

            reasoning = f"Question-answer alignment: {normalized_score:.3f} ({score}/5)"

            return ScoreResult(
                score=normalized_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"raw_score": score},
            )

        except Exception as e:
            return ScoreResult(
                score=0.0, passed=False, reasoning=f"Error: {e!s}", metadata={}
            )

    async def _call_model(self, prompt: str) -> str:
        import asyncio

        return await asyncio.to_thread(call_llm, self.model, prompt)

    def _parse_alignment_score(self, response: str) -> float:
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 3.0  # Default to neutral

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Union[float, dict[str, float]]:
        import asyncio

        # Extract context from dict if available
        context_text = context.get("context") if context else None

        # Run async evaluation
        result = asyncio.run(
            self.evaluate(
                input_text=ground_truth, output_text=prediction, context=context_text
            )
        )

        if hasattr(result, "score"):

            return {"score": result.score}

        else:

            return {"score": result.score if hasattr(result, "score") else 0.0}


class CrossContextSynthesisScorer(BaseScorer):
    """
    Quality of information synthesis across multiple contexts.
    """

    def __init__(
        self, model: Any, threshold: float = 0.7, max_score: float = 5.0, **kwargs: Any
    ) -> None:
        super().__init__(name="CrossContextSynthesisScorer", **kwargs)
        self.threshold = threshold
        self.model = model
        self.max_score = max_score  # Maximum score for normalization

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        if not context or not output_text:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="No context or output provided",
                metadata={},
            )

        # Split context into chunks
        context_chunks = context.split("\n\n")
        if len(context_chunks) < 2:
            return ScoreResult(
                score=1.0,
                passed=True,
                reasoning="Single context provided",
                metadata={"chunks": 1},
            )

        prompt = RAGPrompts.get_context_faithfulness_evaluation(
            context=context, output_text=output_text
        )

        try:
            response = await self._call_model(prompt)
            score = self._parse_synthesis_score(response)
            normalized_score = score / self.max_score
            passed = normalized_score >= self.threshold

            reasoning = f"Cross-context synthesis: {normalized_score:.3f} ({score}/5)"

            return ScoreResult(
                score=normalized_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"raw_score": score, "context_chunks": len(context_chunks)},
            )

        except Exception as e:
            return ScoreResult(
                score=0.0, passed=False, reasoning=f"Error: {e!s}", metadata={}
            )

    async def _call_model(self, prompt: str) -> str:
        import asyncio

        return await asyncio.to_thread(call_llm, self.model, prompt)

    def _parse_synthesis_score(self, response: str) -> float:
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 3.0  # Default to neutral

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Union[float, dict[str, float]]:
        import asyncio

        # Extract context from dict if available
        context_text = context.get("context") if context else None

        # Run async evaluation
        result = asyncio.run(
            self.evaluate(
                input_text=ground_truth, output_text=prediction, context=context_text
            )
        )

        if hasattr(result, "score"):

            return {"score": result.score}

        else:

            return {"score": result.score if hasattr(result, "score") else 0.0}


class TechnicalAccuracyScorer(BaseScorer):
    """
    Technical domain accuracy evaluation.
    """

    def __init__(
        self, model: Any, threshold: float = 0.8, max_score: float = 5.0, **kwargs: Any
    ) -> None:
        super().__init__(name="TechnicalAccuracyScorer", **kwargs)
        self.threshold = threshold
        self.model = model
        self.max_score = max_score  # Maximum score for normalization

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        if not output_text:
            return ScoreResult(
                score=0.0, passed=False, reasoning="No answer provided", metadata={}
            )

        prompt = RAGPrompts.get_technical_accuracy_evaluation(
            context=context or "No context provided", output_text=output_text
        )

        try:
            response = await self._call_model(prompt)
            score = self._parse_technical_score(response)
            normalized_score = score / self.max_score
            passed = normalized_score >= self.threshold

            reasoning = f"Technical accuracy: {normalized_score:.3f} ({score}/5)"

            return ScoreResult(
                score=normalized_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"raw_score": score},
            )

        except Exception as e:
            return ScoreResult(
                score=0.0, passed=False, reasoning=f"Error: {e!s}", metadata={}
            )

    async def _call_model(self, prompt: str) -> str:
        import asyncio

        return await asyncio.to_thread(call_llm, self.model, prompt)

    def _parse_technical_score(self, response: str) -> float:
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 3.0  # Default to neutral

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Union[float, dict[str, float]]:
        import asyncio

        # Extract context from dict if available
        context_text = context.get("context") if context else None

        # Run async evaluation
        result = asyncio.run(
            self.evaluate(
                input_text=ground_truth, output_text=prediction, context=context_text
            )
        )

        if hasattr(result, "score"):

            return {"score": result.score}

        else:

            return {"score": result.score if hasattr(result, "score") else 0.0}
