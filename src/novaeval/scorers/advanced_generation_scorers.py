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

# LangChain detection
try:
    import langchain
    from langchain_core.language_models.base import BaseLanguageModel
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False
    BaseLanguageModel = None

from typing import Any, Optional, Union, List, TYPE_CHECKING
from novaeval.scorers.base import BaseScorer, ScoreResult
import re
import numpy as np

# Forward reference for type hints
if TYPE_CHECKING:
    from typing_extensions import TypedDict


def _call_llm(model, prompt: str):
    """
    Helper to call the LLM, supporting both string (OpenAI) and LangChain LLM objects.
    """
    if _HAS_LANGCHAIN and BaseLanguageModel and isinstance(model, BaseLanguageModel):
        # LangChain LLM: use .invoke or .generate
        if hasattr(model, "invoke"):
            return model.invoke(prompt)
        elif hasattr(model, "generate"):
            return model.generate([prompt]).generations[0][0].text
        else:
            raise ValueError("LangChain LLM does not support invoke or generate.")
    elif isinstance(model, str):
        # Built-in: string model name, use OpenAI API
        raise NotImplementedError("String-based LLM calling not implemented. Please provide a LangChain LLM or implement OpenAI call here.")
    else:
        # Assume model has a .generate method (NovaEval style)
        return model.generate(prompt)


# CONTEXT-AWARE GENERATION SCORERS
class BiasDetectionScorer(BaseScorer):
    """
    Bias detection in generated content.
    """
    def __init__(self, model, threshold=0.8, **kwargs):
        super().__init__(name="BiasDetectionScorer", **kwargs)
        self.threshold = threshold
        self.model = model
    
    async def evaluate(self, input_text: str, output_text: str, expected_output: Optional[str] = None, context: Optional[str] = None, **kwargs: Any) -> ScoreResult:
        if not output_text:
            return ScoreResult(score=0.0, passed=False, reasoning="No answer provided", metadata={})
        
        prompt = f"""
        Question: {input_text}
        Answer: {output_text}
        
        Detect any bias in this answer.
        Look for:
        1. Gender bias
        2. Racial bias
        3. Cultural bias
        4. Political bias
        5. Confirmation bias
        6. Stereotyping
        
        Rate the level of bias from 1-5:
        1: No bias detected
        2: Minor bias
        3: Some bias
        4: Significant bias
        5: Major bias
        
        Rating: """
        
        try:
            response = await self._call_model(prompt)
            bias_score = self._parse_bias_score(response)
            # Invert the score (lower bias = higher quality)
            quality_score = (6 - bias_score) / 5.0
            passed = quality_score >= self.threshold
            
            reasoning = f"Bias level: {bias_score}/5, Quality: {quality_score:.3f}"
            
            return ScoreResult(
                score=quality_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"bias_score": bias_score}
            )
            
        except Exception as e:
            return ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}", metadata={})
    
    async def _call_model(self, prompt: str):
        import asyncio
        return await asyncio.to_thread(_call_llm, self.model, prompt)
    
    def _parse_bias_score(self, response: str) -> float:
        import re
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 1.0  # Default to no bias
    
    def score(self, prediction, ground_truth, context=None):
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.evaluate(
                    input_text=ground_truth,
                    output_text=prediction,
                    context=context.get("context", "") if context else None
                )
            )
            loop.close()
            return result.score if hasattr(result, 'score') else result
        except Exception as e:
            return 0.0

# HALLUCINATION DETECTION SCORERS
class FactualAccuracyScorer(BaseScorer):
    """
    Verify factual claims against contexts.
    """
    def __init__(self, model, threshold=0.8, **kwargs):
        super().__init__(name="FactualAccuracyScorer", **kwargs)
        self.threshold = threshold
        self.model = model
    
    async def evaluate(self, input_text: str, output_text: str, expected_output: Optional[str] = None, context: Optional[str] = None, **kwargs: Any) -> ScoreResult:
        if not output_text or not context:
            return ScoreResult(score=0.0, passed=False, reasoning="No answer or context provided", metadata={})
        
        prompt = f"""
        Context: {context}
        Answer: {output_text}
        
        Verify the factual accuracy of this answer against the provided context.
        Check for:
        1. Factual claims that can be verified
        2. Dates, numbers, names, and specific details
        3. Relationships and causal connections
        4. Statistical information and data
        
        Rate the factual accuracy from 1-5:
        1: Completely inaccurate
        2: Mostly inaccurate
        3: Partially accurate
        4: Mostly accurate
        5: Completely accurate
        
        Rating: """
        
        try:
            response = await self._call_model(prompt)
            score = self._parse_factual_score(response)
            normalized_score = score / 5.0
            passed = normalized_score >= self.threshold
            
            reasoning = f"Factual accuracy: {normalized_score:.3f} ({score}/5)"
            
            return ScoreResult(
                score=normalized_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"raw_score": score}
            )
            
        except Exception as e:
            return ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}", metadata={})
    
    async def _call_model(self, prompt: str):
        import asyncio
        return await asyncio.to_thread(_call_llm, self.model, prompt)
    
    def _parse_factual_score(self, response: str) -> float:
        import re
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 3.0  # Default to neutral
    
    def score(self, prediction, ground_truth, context=None):
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.evaluate(
                    input_text=ground_truth,
                    output_text=prediction,
                    context=context.get("context", "") if context else None
                )
            )
            loop.close()
            return result.score if hasattr(result, 'score') else result
        except Exception as e:
            return 0.0

class ClaimVerificationScorer(BaseScorer):
    """
    Verify specific claims in generated answers.
    """
    def __init__(self, model, threshold=0.7, **kwargs):
        super().__init__(name="ClaimVerificationScorer", **kwargs)
        self.threshold = threshold
        self.model = model
    
    async def evaluate(self, input_text: str, output_text: str, expected_output: Optional[str] = None, context: Optional[str] = None, **kwargs: Any) -> ScoreResult:
        if not output_text:
            return ScoreResult(score=0.0, passed=False, reasoning="No answer provided", metadata={})
        
        # Extract claims from the answer
        claims_prompt = f"""
        Extract all specific claims from this answer. Focus on factual statements.
        
        Answer: {output_text}
        
        List each claim as a separate statement:
        1. [Claim 1]
        2. [Claim 2]
        3. [Claim 3]
        """
        
        try:
            claims_response = await self._call_model(claims_prompt)
            claims = self._parse_claims(claims_response)
            
            if not claims:
                return ScoreResult(score=1.0, passed=True, reasoning="No specific claims found", metadata={"claims": []})
            
            # Verify each claim
            verified_claims = []
            total_score = 0.0
            
            for claim in claims:
                verification_prompt = f"""
                Context: {context or "No context provided"}
                
                Claim: {claim}
                
                Can this specific claim be verified or supported? Rate 1-5:
                1: Cannot be verified/contradicts context
                2: Poorly supported
                3: Somewhat supported
                4: Well supported
                5: Fully verified by context
                
                Rating: """
                
                verification_response = await self._call_model(verification_prompt)
                score = self._parse_verification_score(verification_response)
                total_score += score
                verified_claims.append({"claim": claim, "score": score})
            
            avg_score = total_score / len(claims) / 5.0
            passed = avg_score >= self.threshold
            
            reasoning = f"Verified {len(claims)} claims. Average verification: {avg_score:.3f}"
            
            return ScoreResult(
                score=avg_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"verified_claims": verified_claims, "total_claims": len(claims)}
            )
            
        except Exception as e:
            return ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}", metadata={})
    
    async def _call_model(self, prompt: str):
        import asyncio
        return await asyncio.to_thread(_call_llm, self.model, prompt)
    
    def _parse_claims(self, text: str) -> list[str]:
        claims = []
        lines = text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-") or line.startswith("*")):
                for prefix in ["1.", "2.", "3.", "4.", "5.", "-", "*"]:
                    if line.startswith(prefix):
                        claim = line[len(prefix):].strip()
                        if claim:
                            claims.append(claim)
                        break
        return claims
    
    def _parse_verification_score(self, response: str) -> float:
        import re
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 3.0  # Default to neutral
    
    def score(self, prediction, ground_truth, context=None):
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.evaluate(
                    input_text=ground_truth,
                    output_text=prediction,
                    context=context.get("context", "") if context else None
                )
            )
            loop.close()
            return result.score if hasattr(result, 'score') else result
        except Exception as e:
            return 0.0

# ANSWER COMPLETENESS AND RELEVANCE SCORERS
class InformationDensityScorer(BaseScorer):
    """
    Information richness evaluation.
    """
    def __init__(self, model, threshold=0.6, **kwargs):
        super().__init__(name="InformationDensityScorer", **kwargs)
        self.threshold = threshold
        self.model = model
    
    async def evaluate(self, input_text: str, output_text: str, expected_output: Optional[str] = None, context: Optional[str] = None, **kwargs: Any) -> ScoreResult:
        if not output_text:
            return ScoreResult(score=0.0, passed=False, reasoning="No answer provided", metadata={})
        
        prompt = f"""
        Question: {input_text}
        Answer: {output_text}
        
        Evaluate the information density of this answer.
        Consider:
        1. How much useful information is provided?
        2. Is the information concise and focused?
        3. Are there unnecessary repetitions?
        4. Is the information well-structured?
        5. Does it provide valuable insights?
        
        Rate the information density from 1-5:
        1: Very low information density
        2: Low information density
        3: Moderate information density
        4: High information density
        5: Very high information density
        
        Rating: """
        
        try:
            response = await self._call_model(prompt)
            score = self._parse_density_score(response)
            normalized_score = score / 5.0
            passed = normalized_score >= self.threshold
            
            reasoning = f"Information density: {normalized_score:.3f} ({score}/5)"
            
            return ScoreResult(
                score=normalized_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"raw_score": score}
            )
            
        except Exception as e:
            return ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}", metadata={})
    
    async def _call_model(self, prompt: str):
        import asyncio
        return await asyncio.to_thread(_call_llm, self.model, prompt)
    
    def _parse_density_score(self, response: str) -> float:
        import re
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 3.0  # Default to neutral
    
    def score(self, prediction, ground_truth, context=None):
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.evaluate(
                    input_text=ground_truth,
                    output_text=prediction,
                    context=context.get("context", "") if context else None
                )
            )
            loop.close()
            return result.score if hasattr(result, 'score') else result
        except Exception as e:
            return 0.0

class ClarityAndCoherenceScorer(BaseScorer):
    """
    Answer readability and logic evaluation.
    """
    def __init__(self, model, threshold=0.7, **kwargs):
        super().__init__(name="ClarityAndCoherenceScorer", **kwargs)
        self.threshold = threshold
        self.model = model
    
    async def evaluate(self, input_text: str, output_text: str, expected_output: Optional[str] = None, context: Optional[str] = None, **kwargs: Any) -> ScoreResult:
        if not output_text:
            return ScoreResult(score=0.0, passed=False, reasoning="No answer provided", metadata={})
        
        prompt = f"""
        Answer: {output_text}
        
        Evaluate the clarity and coherence of this answer.
        Consider:
        1. Is the language clear and understandable?
        2. Is the logic flow coherent?
        3. Are ideas well-connected?
        4. Is the structure logical?
        5. Are transitions smooth?
        6. Is the writing style appropriate?
        
        Rate from 1-5:
        1: Very unclear and incoherent
        2: Unclear and somewhat incoherent
        3: Somewhat clear and coherent
        4: Clear and mostly coherent
        5: Very clear and coherent
        
        Rating: """
        
        try:
            response = await self._call_model(prompt)
            score = self._parse_clarity_score(response)
            normalized_score = score / 5.0
            passed = normalized_score >= self.threshold
            
            reasoning = f"Clarity and coherence: {normalized_score:.3f} ({score}/5)"
            
            return ScoreResult(
                score=normalized_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"raw_score": score}
            )
            
        except Exception as e:
            return ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}", metadata={})
    
    async def _call_model(self, prompt: str):
        import asyncio
        return await asyncio.to_thread(_call_llm, self.model, prompt)
    
    def _parse_clarity_score(self, response: str) -> float:
        import re
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 3.0  # Default to neutral
    
    def score(self, prediction, ground_truth, context=None):
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.evaluate(
                    input_text=ground_truth,
                    output_text=prediction,
                    context=context.get("context", "") if context else None
                )
            )
            loop.close()
            return result.score if hasattr(result, 'score') else result
        except Exception as e:
            return 0.0

# MULTI-CONTEXT INTEGRATION SCORERS
class ConflictResolutionScorer(BaseScorer):
    """
    Handling contradictory information across contexts.
    """
    def __init__(self, model, threshold=0.7, **kwargs):
        super().__init__(name="ConflictResolutionScorer", **kwargs)
        self.threshold = threshold
        self.model = model
    
    async def evaluate(self, input_text: str, output_text: str, expected_output: Optional[str] = None, context: Optional[str] = None, **kwargs: Any) -> ScoreResult:
        if not context or not output_text:
            return ScoreResult(score=0.0, passed=False, reasoning="No context or output provided", metadata={})
        
        # Split context into chunks to check for conflicts
        context_chunks = context.split("\n\n")
        if len(context_chunks) < 2:
            return ScoreResult(score=1.0, passed=True, reasoning="Single context provided", metadata={"chunks": 1})
        
        prompt = f"""
        Question: {input_text}
        Context chunks: {len(context_chunks)} separate pieces of information
        Answer: {output_text}
        
        Evaluate how well this answer handles potential conflicts between different context pieces.
        Consider:
        1. Does it acknowledge conflicting information?
        2. Does it resolve contradictions appropriately?
        3. Does it present balanced perspectives?
        4. Does it avoid taking sides without evidence?
        
        Rate from 1-5:
        1: Poor conflict resolution
        2: Basic conflict handling
        3: Adequate conflict resolution
        4: Good conflict resolution
        5: Excellent conflict resolution
        
        Rating: """
        
        try:
            response = await self._call_model(prompt)
            score = self._parse_conflict_score(response)
            normalized_score = score / 5.0
            passed = normalized_score >= self.threshold
            
            reasoning = f"Conflict resolution: {normalized_score:.3f} ({score}/5)"
            
            return ScoreResult(
                score=normalized_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"raw_score": score, "context_chunks": len(context_chunks)}
            )
            
        except Exception as e:
            return ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}", metadata={})
    
    async def _call_model(self, prompt: str):
        import asyncio
        return await asyncio.to_thread(_call_llm, self.model, prompt)
    
    def _parse_conflict_score(self, response: str) -> float:
        import re
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 3.0  # Default to neutral
    
    def score(self, prediction, ground_truth, context=None):
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.evaluate(
                    input_text=ground_truth,
                    output_text=prediction,
                    context=context.get("context", "") if context else None
                )
            )
            loop.close()
            return result.score if hasattr(result, 'score') else result
        except Exception as e:
            return 0.0

class ContextPrioritizationScorer(BaseScorer):
    """
    Appropriate context weighting evaluation.
    """
    def __init__(self, model, threshold=0.6, **kwargs):
        super().__init__(name="ContextPrioritizationScorer", **kwargs)
        self.threshold = threshold
        self.model = model
    
    async def evaluate(self, input_text: str, output_text: str, expected_output: Optional[str] = None, context: Optional[str] = None, **kwargs: Any) -> ScoreResult:
        if not context or not output_text:
            return ScoreResult(score=0.0, passed=False, reasoning="No context or output provided", metadata={})
        
        prompt = f"""
        Question: {input_text}
        Context: {context}
        Answer: {output_text}
        
        Evaluate how well this answer prioritizes and weights different parts of the context.
        Consider:
        1. Does it focus on the most relevant context?
        2. Does it appropriately weight important vs. less important information?
        3. Does it avoid over-emphasizing irrelevant details?
        4. Does it balance different context pieces appropriately?
        
        Rate from 1-5:
        1: Poor context prioritization
        2: Basic prioritization
        3: Adequate prioritization
        4: Good prioritization
        5: Excellent prioritization
        
        Rating: """
        
        try:
            response = await self._call_model(prompt)
            score = self._parse_prioritization_score(response)
            normalized_score = score / 5.0
            passed = normalized_score >= self.threshold
            
            reasoning = f"Context prioritization: {normalized_score:.3f} ({score}/5)"
            
            return ScoreResult(
                score=normalized_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"raw_score": score}
            )
            
        except Exception as e:
            return ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}", metadata={})
    
    async def _call_model(self, prompt: str):
        import asyncio
        return await asyncio.to_thread(_call_llm, self.model, prompt)
    
    def _parse_prioritization_score(self, response: str) -> float:
        import re
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 3.0  # Default to neutral
    
    def score(self, prediction, ground_truth, context=None):
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.evaluate(
                    input_text=ground_truth,
                    output_text=prediction,
                    context=context.get("context", "") if context else None
                )
            )
            loop.close()
            return result.score if hasattr(result, 'score') else result
        except Exception as e:
            return 0.0

class CitationQualityScorer(BaseScorer):
    """
    Quality of source references evaluation.
    """
    def __init__(self, model, threshold=0.6, **kwargs):
        super().__init__(name="CitationQualityScorer", **kwargs)
        self.threshold = threshold
        self.model = model
    
    async def evaluate(self, input_text: str, output_text: str, expected_output: Optional[str] = None, context: Optional[str] = None, **kwargs: Any) -> ScoreResult:
        if not output_text:
            return ScoreResult(score=0.0, passed=False, reasoning="No answer provided", metadata={})
        
        prompt = f"""
        Answer: {output_text}
        
        Evaluate the quality of citations and source references in this answer.
        Consider:
        1. Are sources properly cited?
        2. Are citations accurate and relevant?
        3. Is the citation format appropriate?
        4. Are sources credible and authoritative?
        5. Are citations placed appropriately in the text?
        
        Rate from 1-5:
        1: Poor citation quality
        2: Basic citation quality
        3: Adequate citation quality
        4: Good citation quality
        5: Excellent citation quality
        
        Rating: """
        
        try:
            response = await self._call_model(prompt)
            score = self._parse_citation_score(response)
            normalized_score = score / 5.0
            passed = normalized_score >= self.threshold
            
            reasoning = f"Citation quality: {normalized_score:.3f} ({score}/5)"
            
            return ScoreResult(
                score=normalized_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"raw_score": score}
            )
            
        except Exception as e:
            return ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}", metadata={})
    
    async def _call_model(self, prompt: str):
        import asyncio
        return await asyncio.to_thread(_call_llm, self.model, prompt)
    
    def _parse_citation_score(self, response: str) -> float:
        import re
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 3.0  # Default to neutral
    
    def score(self, prediction, ground_truth, context=None):
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.evaluate(
                    input_text=ground_truth,
                    output_text=prediction,
                    context=context.get("context", "") if context else None
                )
            )
            loop.close()
            return result.score if hasattr(result, 'score') else result
        except Exception as e:
            return 0.0

# DOMAIN-SPECIFIC EVALUATION SCORERS
class ToneConsistencyScorer(BaseScorer):
    """
    Appropriate tone for domain evaluation.
    """
    def __init__(self, model, threshold=0.7, **kwargs):
        super().__init__(name="ToneConsistencyScorer", **kwargs)
        self.threshold = threshold
        self.model = model
    
    async def evaluate(self, input_text: str, output_text: str, expected_output: Optional[str] = None, context: Optional[str] = None, **kwargs: Any) -> ScoreResult:
        if not output_text:
            return ScoreResult(score=0.0, passed=False, reasoning="No answer provided", metadata={})
        
        prompt = f"""
        Question: {input_text}
        Answer: {output_text}
        
        Evaluate the appropriateness and consistency of tone for this domain.
        Consider:
        1. Is the tone appropriate for the subject matter?
        2. Is the tone consistent throughout the answer?
        3. Is the formality level appropriate?
        4. Does the tone match the expected audience?
        5. Is the tone professional and respectful?
        
        Rate from 1-5:
        1: Inappropriate and inconsistent tone
        2: Poor tone appropriateness
        3: Adequate tone
        4: Good tone appropriateness
        5: Excellent tone appropriateness
        
        Rating: """
        
        try:
            response = await self._call_model(prompt)
            score = self._parse_tone_score(response)
            normalized_score = score / 5.0
            passed = normalized_score >= self.threshold
            
            reasoning = f"Tone consistency: {normalized_score:.3f} ({score}/5)"
            
            return ScoreResult(
                score=normalized_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"raw_score": score}
            )
            
        except Exception as e:
            return ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}", metadata={})
    
    async def _call_model(self, prompt: str):
        import asyncio
        return await asyncio.to_thread(_call_llm, self.model, prompt)
    
    def _parse_tone_score(self, response: str) -> float:
        import re
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 3.0  # Default to neutral
    
    def score(self, prediction, ground_truth, context=None):
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.evaluate(
                    input_text=ground_truth,
                    output_text=prediction,
                    context=context.get("context", "") if context else None
                )
            )
            loop.close()
            return result.score if hasattr(result, 'score') else result
        except Exception as e:
            return 0.0

class TerminologyConsistencyScorer(BaseScorer):
    """
    Consistent use of domain terms evaluation.
    """
    def __init__(self, model, threshold=0.7, **kwargs):
        super().__init__(name="TerminologyConsistencyScorer", **kwargs)
        self.threshold = threshold
        self.model = model
    
    async def evaluate(self, input_text: str, output_text: str, expected_output: Optional[str] = None, context: Optional[str] = None, **kwargs: Any) -> ScoreResult:
        if not output_text:
            return ScoreResult(score=0.0, passed=False, reasoning="No answer provided", metadata={})
        
        prompt = f"""
        Answer: {output_text}
        
        Evaluate the consistency of terminology and domain-specific language.
        Consider:
        1. Are domain terms used consistently?
        2. Are technical terms defined appropriately?
        3. Is the terminology accurate for the domain?
        4. Are abbreviations used consistently?
        5. Is the language appropriate for the domain?
        
        Rate from 1-5:
        1: Poor terminology consistency
        2: Basic terminology consistency
        3: Adequate terminology consistency
        4: Good terminology consistency
        5: Excellent terminology consistency
        
        Rating: """
        
        try:
            response = await self._call_model(prompt)
            score = self._parse_terminology_score(response)
            normalized_score = score / 5.0
            passed = normalized_score >= self.threshold
            
            reasoning = f"Terminology consistency: {normalized_score:.3f} ({score}/5)"
            
            return ScoreResult(
                score=normalized_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"raw_score": score}
            )
            
        except Exception as e:
            return ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}", metadata={})
    
    async def _call_model(self, prompt: str):
        import asyncio
        return await asyncio.to_thread(_call_llm, self.model, prompt)
    
    def _parse_terminology_score(self, response: str) -> float:
        import re
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 3.0  # Default to neutral
    
    def score(self, prediction, ground_truth, context=None):
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.evaluate(
                    input_text=ground_truth,
                    output_text=prediction,
                    context=context.get("context", "") if context else None
                )
            )
            loop.close()
            return result.score if hasattr(result, 'score') else result
        except Exception as e:
            return 0.0

class ContextFaithfulnessScorerPP(BaseScorer):
    """
    Enhanced faithfulness detection with fine-grained analysis.
    Analyzes each claim in the answer against the provided context.
    """
    def __init__(self, model, threshold=0.8, **kwargs):
        super().__init__(name="ContextFaithfulnessScorerPP", **kwargs)
        self.threshold = threshold
        self.model = model
    
    async def evaluate(self, input_text: str, output_text: str, expected_output: Optional[str] = None, context: Optional[str] = None, **kwargs: Any) -> ScoreResult:
        if not context or not output_text:
            return ScoreResult(score=0.0, passed=False, reasoning="No context or output provided", metadata={})
        
        # Extract claims from the answer
        claims_prompt = f"""
        Extract all factual claims from this answer. List each claim as a separate statement.
        
        Answer: {output_text}
        
        Format as:
        1. [Claim 1]
        2. [Claim 2]
        3. [Claim 3]
        """
        
        try:
            claims_response = await self._call_model(claims_prompt)
            claims = self._parse_claims(claims_response)
            
            if not claims:
                return ScoreResult(score=1.0, passed=True, reasoning="No factual claims found", metadata={"claims": []})
            
            # Verify each claim against context
            verified_claims = []
            total_score = 0.0
            
            for i, claim in enumerate(claims):
                verification_prompt = f"""
                Context: {context}
                
                Claim: {claim}
                
                Can this claim be verified from the provided context? Rate 1-5:
                1: Completely false/contradicts context
                2: Mostly false
                3: Partially supported
                4: Mostly supported
                5: Fully supported by context
                
                Rating: """
                
                verification_response = await self._call_model(verification_prompt)
                score = self._parse_verification_score(verification_response)
                total_score += score
                verified_claims.append({"claim": claim, "score": score})
            
            avg_score = total_score / len(claims) / 5.0  # Normalize to 0-1
            passed = avg_score >= self.threshold
            
            reasoning = f"Verified {len(claims)} claims. Average faithfulness: {avg_score:.3f}"
            
            return ScoreResult(
                score=avg_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"verified_claims": verified_claims, "total_claims": len(claims)}
            )
            
        except Exception as e:
            return ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}", metadata={})
    
    async def _call_model(self, prompt: str):
        import asyncio
        return await asyncio.to_thread(_call_llm, self.model, prompt)
    
    def _parse_claims(self, text: str) -> list[str]:
        claims = []
        lines = text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-") or line.startswith("*")):
                for prefix in ["1.", "2.", "3.", "4.", "5.", "-", "*"]:
                    if line.startswith(prefix):
                        claim = line[len(prefix):].strip()
                        if claim:
                            claims.append(claim)
                        break
        return claims
    
    def _parse_verification_score(self, response: str) -> float:
        import re
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 1.0  # Default to no hallucinations
    
    def score(self, prediction, ground_truth, context=None):
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.evaluate(
                    input_text=ground_truth,
                    output_text=prediction,
                    context=context.get("context", "") if context else None
                )
            )
            loop.close()
            return result.score if hasattr(result, 'score') else result
        except Exception as e:
            return 0.0

class ContextGroundednessScorer(BaseScorer):
    """
    Ensures answers are grounded in provided context.
    Evaluates how well the answer is supported by the given context.
    """
    def __init__(self, model, threshold=0.7, **kwargs):
        super().__init__(name="ContextGroundednessScorer", **kwargs)
        self.threshold = threshold
        self.model = model
    
    async def evaluate(self, input_text: str, output_text: str, expected_output: Optional[str] = None, context: Optional[str] = None, **kwargs: Any) -> ScoreResult:
        if not context or not output_text:
            return ScoreResult(score=0.0, passed=False, reasoning="No context or output provided", metadata={})
        
        prompt = f"""
        Context: {context}
        
        Answer: {output_text}
        
        Evaluate how well this answer is grounded in the provided context.
        Consider:
        1. Are the main points supported by the context?
        2. Are there any claims that go beyond the context?
        3. Is the answer faithful to the information provided?
        
        Rate from 1-5:
        1: Not grounded at all
        2: Poorly grounded
        3: Somewhat grounded
        4: Well grounded
        5: Fully grounded in context
        
        Rating: """
        
        try:
            response = await self._call_model(prompt)
            score = self._parse_groundedness_score(response)
            normalized_score = score / 5.0
            passed = normalized_score >= self.threshold
            
            reasoning = f"Groundedness score: {normalized_score:.3f} ({score}/5)"
            
            return ScoreResult(
                score=normalized_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"raw_score": score}
            )
            
        except Exception as e:
            return ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}", metadata={})
    
    async def _call_model(self, prompt: str):
        import asyncio
        return await asyncio.to_thread(_call_llm, self.model, prompt)
    
    def _parse_groundedness_score(self, response: str) -> float:
        import re
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 3.0  # Default to neutral
    
    def score(self, prediction, ground_truth, context=None):
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.evaluate(
                    input_text=ground_truth,
                    output_text=prediction,
                    context=context.get("context", "") if context else None
                )
            )
            loop.close()
            return result.score if hasattr(result, 'score') else result
        except Exception as e:
            return 0.0

class ContextCompletenessScorer(BaseScorer):
    """
    Evaluates if context fully supports the answer.
    Checks whether the provided context contains all necessary information.
    """
    def __init__(self, model, threshold=0.6, **kwargs):
        super().__init__(name="ContextCompletenessScorer", **kwargs)
        self.threshold = threshold
        self.model = model
    
    async def evaluate(self, input_text: str, output_text: str, expected_output: Optional[str] = None, context: Optional[str] = None, **kwargs: Any) -> ScoreResult:
        if not context or not output_text:
            return ScoreResult(score=0.0, passed=False, reasoning="No context or output provided", metadata={})
        
        prompt = f"""
        Question: {input_text}
        Context: {context}
        Answer: {output_text}
        
        Evaluate if the provided context is complete enough to support this answer.
        Consider:
        1. Does the context contain all the information needed for this answer?
        2. Are there any gaps in the context that prevent a complete answer?
        3. Could a better answer be given with more context?
        
        Rate from 1-5:
        1: Context is completely insufficient
        2: Context has major gaps
        3: Context is partially complete
        4: Context is mostly complete
        5: Context is fully complete for this answer
        
        Rating: """
        
        try:
            response = await self._call_model(prompt)
            score = self._parse_completeness_score(response)
            normalized_score = score / 5.0
            passed = normalized_score >= self.threshold
            
            reasoning = f"Context completeness: {normalized_score:.3f} ({score}/5)"
            
            return ScoreResult(
                score=normalized_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"raw_score": score}
            )
            
        except Exception as e:
            return ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}", metadata={})
    
    async def _call_model(self, prompt: str):
        import asyncio
        return await asyncio.to_thread(_call_llm, self.model, prompt)
    
    def _parse_completeness_score(self, response: str) -> float:
        import re
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 3.0  # Default to neutral
    
    def score(self, prediction, ground_truth, context=None):
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.evaluate(
                    input_text=ground_truth,
                    output_text=prediction,
                    context=context.get("context", "") if context else None
                )
            )
            loop.close()
            return result.score if hasattr(result, 'score') else result
        except Exception as e:
            return 0.0

class ContextConsistencyScorer(BaseScorer):
    """
    Consistency across multiple contexts.
    Evaluates if the answer is consistent when multiple contexts are provided.
    """
    def __init__(self, model, threshold=0.7, **kwargs):
        super().__init__(name="ContextConsistencyScorer", **kwargs)
        self.threshold = threshold
        self.model = model
    
    async def evaluate(self, input_text: str, output_text: str, expected_output: Optional[str] = None, context: Optional[str] = None, **kwargs: Any) -> ScoreResult:
        if not context or not output_text:
            return ScoreResult(score=0.0, passed=False, reasoning="No context or output provided", metadata={})
        
        # Split context into multiple chunks
        context_chunks = context.split("\n\n")
        if len(context_chunks) < 2:
            return ScoreResult(score=1.0, passed=True, reasoning="Single context provided", metadata={"chunks": 1})
        
        # Evaluate consistency across chunks
        consistency_scores = []
        for i, chunk in enumerate(context_chunks):
            prompt = f"""
            Question: {input_text}
            Context chunk {i+1}: {chunk}
            Answer: {output_text}
            
            Evaluate if this answer is consistent with this specific context chunk.
            Rate from 1-5:
            1: Completely inconsistent
            2: Mostly inconsistent
            3: Somewhat consistent
            4: Mostly consistent
            5: Fully consistent
            
            Rating: """
            
            try:
                response = await self._call_model(prompt)
                score = self._parse_consistency_score(response)
                consistency_scores.append(score)
            except Exception as e:
                consistency_scores.append(3.0)  # Default to neutral
        
        avg_score = sum(consistency_scores) / len(consistency_scores) / 5.0
        passed = avg_score >= self.threshold
        
        reasoning = f"Consistency across {len(context_chunks)} chunks: {avg_score:.3f}"
        
        return ScoreResult(
            score=avg_score,
            passed=passed,
            reasoning=reasoning,
            metadata={"consistency_scores": consistency_scores, "chunks": len(context_chunks)}
        )
    
    async def _call_model(self, prompt: str):
        import asyncio
        return await asyncio.to_thread(_call_llm, self.model, prompt)
    
    def _parse_consistency_score(self, response: str) -> float:
        import re
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 3.0  # Default to neutral
    
    def score(self, prediction, ground_truth, context=None):
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.evaluate(
                    input_text=ground_truth,
                    output_text=prediction,
                    context=context.get("context", "") if context else None
                )
            )
            loop.close()
            return result.score if hasattr(result, 'score') else result
        except Exception as e:
            return 0.0

class RAGAnswerQualityScorer(BaseScorer):
    """
    Comprehensive RAG generation evaluation.
    Evaluates the overall quality of RAG-generated answers.
    """
    def __init__(self, model, threshold=0.7, **kwargs):
        super().__init__(name="RAGAnswerQualityScorer", **kwargs)
        self.threshold = threshold
        self.model = model
    
    async def evaluate(self, input_text: str, output_text: str, expected_output: Optional[str] = None, context: Optional[str] = None, **kwargs: Any) -> ScoreResult:
        if not output_text:
            return ScoreResult(score=0.0, passed=False, reasoning="No answer provided", metadata={})
        
        prompt = f"""
        Question: {input_text}
        Context: {context or "No context provided"}
        Answer: {output_text}
        
        Evaluate the overall quality of this RAG-generated answer.
        Consider:
        1. Accuracy and factual correctness
        2. Relevance to the question
        3. Completeness of the answer
        4. Clarity and coherence
        5. Use of provided context
        
        Rate from 1-5:
        1: Poor quality
        2: Below average
        3: Average
        4: Good quality
        5: Excellent quality
        
        Rating: """
        
        try:
            response = await self._call_model(prompt)
            score = self._parse_quality_score(response)
            normalized_score = score / 5.0
            passed = normalized_score >= self.threshold
            
            reasoning = f"Answer quality: {normalized_score:.3f} ({score}/5)"
            
            return ScoreResult(
                score=normalized_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"raw_score": score}
            )
            
        except Exception as e:
            return ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}", metadata={})
    
    async def _call_model(self, prompt: str):
        import asyncio
        return await asyncio.to_thread(_call_llm, self.model, prompt)
    
    def _parse_quality_score(self, response: str) -> float:
        import re
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 3.0  # Default to neutral
    
    def score(self, prediction, ground_truth, context=None):
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.evaluate(
                    input_text=ground_truth,
                    output_text=prediction,
                    context=context.get("context", "") if context else None
                )
            )
            loop.close()
            return result.score if hasattr(result, 'score') else result
        except Exception as e:
            return 0.0

class HallucinationDetectionScorer(BaseScorer):
    """
    Identify factual inconsistencies in generated answers.
    """
    def __init__(self, model, threshold=0.8, **kwargs):
        super().__init__(name="HallucinationDetectionScorer", **kwargs)
        self.threshold = threshold
        self.model = model
    
    async def evaluate(self, input_text: str, output_text: str, expected_output: Optional[str] = None, context: Optional[str] = None, **kwargs: Any) -> ScoreResult:
        if not output_text:
            return ScoreResult(score=0.0, passed=False, reasoning="No answer provided", metadata={})
        
        prompt = f"""
        Context: {context or "No context provided"}
        Answer: {output_text}
        
        Detect any hallucinations (factual inconsistencies) in this answer.
        Look for:
        1. Claims that contradict the context
        2. Information not present in the context
        3. Exaggerated or unsupported statements
        4. Fabricated facts or details
        
        Rate the level of hallucination from 1-5:
        1: No hallucinations detected
        2: Minor hallucinations
        3: Some hallucinations
        4: Significant hallucinations
        5: Major hallucinations
        
        Rating: """
        
        try:
            response = await self._call_model(prompt)
            hallucination_score = self._parse_hallucination_score(response)
            # Invert the score (lower hallucination = higher quality)
            quality_score = (6 - hallucination_score) / 5.0
            passed = quality_score >= self.threshold
            
            reasoning = f"Hallucination level: {hallucination_score}/5, Quality: {quality_score:.3f}"
            
            return ScoreResult(
                score=quality_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"hallucination_score": hallucination_score}
            )
            
        except Exception as e:
            return ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}", metadata={})
    
    async def _call_model(self, prompt: str):
        import asyncio
        return await asyncio.to_thread(_call_llm, self.model, prompt)
    
    def _parse_hallucination_score(self, response: str) -> float:
        import re
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 1.0  # Default to no hallucinations
    
    def score(self, prediction, ground_truth, context=None):
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.evaluate(
                    input_text=ground_truth,
                    output_text=prediction,
                    context=context.get("context", "") if context else None
                )
            )
            loop.close()
            return result.score if hasattr(result, 'score') else result
        except Exception as e:
            return 0.0

class SourceAttributionScorer(BaseScorer):
    """
    Proper citation and source attribution evaluation.
    """
    def __init__(self, model, threshold=0.6, **kwargs):
        super().__init__(name="SourceAttributionScorer", **kwargs)
        self.threshold = threshold
        self.model = model
    
    async def evaluate(self, input_text: str, output_text: str, expected_output: Optional[str] = None, context: Optional[str] = None, **kwargs: Any) -> ScoreResult:
        if not output_text:
            return ScoreResult(score=0.0, passed=False, reasoning="No answer provided", metadata={})
        
        prompt = f"""
        Context: {context or "No context provided"}
        Answer: {output_text}
        
        Evaluate the quality of source attribution and citations in this answer.
        Consider:
        1. Are sources properly cited?
        2. Are claims attributed to specific sources?
        3. Is there appropriate acknowledgment of information sources?
        4. Are citations accurate and relevant?
        
        Rate from 1-5:
        1: No source attribution
        2: Poor attribution
        3: Some attribution
        4: Good attribution
        5: Excellent attribution
        
        Rating: """
        
        try:
            response = await self._call_model(prompt)
            score = self._parse_attribution_score(response)
            normalized_score = score / 5.0
            passed = normalized_score >= self.threshold
            
            reasoning = f"Source attribution: {normalized_score:.3f} ({score}/5)"
            
            return ScoreResult(
                score=normalized_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"raw_score": score}
            )
            
        except Exception as e:
            return ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}", metadata={})
    
    async def _call_model(self, prompt: str):
        import asyncio
        return await asyncio.to_thread(_call_llm, self.model, prompt)
    
    def _parse_attribution_score(self, response: str) -> float:
        import re
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 3.0  # Default to neutral
    
    def score(self, prediction, ground_truth, context=None):
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.evaluate(
                    input_text=ground_truth,
                    output_text=prediction,
                    context=context.get("context", "") if context else None
                )
            )
            loop.close()
            return result.score if hasattr(result, 'score') else result
        except Exception as e:
            return 0.0

class AnswerCompletenessScorer(BaseScorer):
    """
    Comprehensive answer coverage evaluation.
    """
    def __init__(self, model, threshold=0.7, **kwargs):
        super().__init__(name="AnswerCompletenessScorer", **kwargs)
        self.threshold = threshold
        self.model = model
    
    async def evaluate(self, input_text: str, output_text: str, expected_output: Optional[str] = None, context: Optional[str] = None, **kwargs: Any) -> ScoreResult:
        if not output_text:
            return ScoreResult(score=0.0, passed=False, reasoning="No answer provided", metadata={})
        
        prompt = f"""
        Question: {input_text}
        Answer: {output_text}
        Expected: {expected_output or "Not provided"}
        
        Evaluate the completeness of this answer.
        Consider:
        1. Does it address all parts of the question?
        2. Is the answer comprehensive?
        3. Are there missing important details?
        4. Does it provide sufficient depth?
        
        Rate from 1-5:
        1: Very incomplete
        2: Incomplete
        3: Somewhat complete
        4: Mostly complete
        5: Fully complete
        
        Rating: """
        
        try:
            response = await self._call_model(prompt)
            score = self._parse_completeness_score(response)
            normalized_score = score / 5.0
            passed = normalized_score >= self.threshold
            
            reasoning = f"Answer completeness: {normalized_score:.3f} ({score}/5)"
            
            return ScoreResult(
                score=normalized_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"raw_score": score}
            )
            
        except Exception as e:
            return ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}", metadata={})
    
    async def _call_model(self, prompt: str):
        import asyncio
        return await asyncio.to_thread(_call_llm, self.model, prompt)
    
    def _parse_completeness_score(self, response: str) -> float:
        import re
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 3.0  # Default to neutral
    
    def score(self, prediction, ground_truth, context=None):
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.evaluate(
                    input_text=ground_truth,
                    output_text=prediction,
                    context=context.get("context", "") if context else None
                )
            )
            loop.close()
            return result.score if hasattr(result, 'score') else result
        except Exception as e:
            return 0.0

class QuestionAnswerAlignmentScorer(BaseScorer):
    """
    Direct question addressing evaluation.
    """
    def __init__(self, model, threshold=0.7, **kwargs):
        super().__init__(name="QuestionAnswerAlignmentScorer", **kwargs)
        self.threshold = threshold
        self.model = model
    
    async def evaluate(self, input_text: str, output_text: str, expected_output: Optional[str] = None, context: Optional[str] = None, **kwargs: Any) -> ScoreResult:
        if not output_text:
            return ScoreResult(score=0.0, passed=False, reasoning="No answer provided", metadata={})
        
        prompt = f"""
        Question: {input_text}
        Answer: {output_text}
        
        Evaluate how well this answer directly addresses the question.
        Consider:
        1. Does it answer the specific question asked?
        2. Is it relevant to the question?
        3. Does it stay on topic?
        4. Is it focused on the question's intent?
        
        Rate from 1-5:
        1: Completely off-topic
        2: Mostly off-topic
        3: Somewhat relevant
        4: Mostly relevant
        5: Directly addresses the question
        
        Rating: """
        
        try:
            response = await self._call_model(prompt)
            score = self._parse_alignment_score(response)
            normalized_score = score / 5.0
            passed = normalized_score >= self.threshold
            
            reasoning = f"Question-answer alignment: {normalized_score:.3f} ({score}/5)"
            
            return ScoreResult(
                score=normalized_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"raw_score": score}
            )
            
        except Exception as e:
            return ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}", metadata={})
    
    async def _call_model(self, prompt: str):
        import asyncio
        return await asyncio.to_thread(_call_llm, self.model, prompt)
    
    def _parse_alignment_score(self, response: str) -> float:
        import re
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 3.0  # Default to neutral
    
    def score(self, prediction, ground_truth, context=None):
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.evaluate(
                    input_text=ground_truth,
                    output_text=prediction,
                    context=context.get("context", "") if context else None
                )
            )
            loop.close()
            return result.score if hasattr(result, 'score') else result
        except Exception as e:
            return 0.0

class CrossContextSynthesisScorer(BaseScorer):
    """
    Quality of information synthesis across multiple contexts.
    """
    def __init__(self, model, threshold=0.7, **kwargs):
        super().__init__(name="CrossContextSynthesisScorer", **kwargs)
        self.threshold = threshold
        self.model = model
    
    async def evaluate(self, input_text: str, output_text: str, expected_output: Optional[str] = None, context: Optional[str] = None, **kwargs: Any) -> ScoreResult:
        if not context or not output_text:
            return ScoreResult(score=0.0, passed=False, reasoning="No context or output provided", metadata={})
        
        # Split context into chunks
        context_chunks = context.split("\n\n")
        if len(context_chunks) < 2:
            return ScoreResult(score=1.0, passed=True, reasoning="Single context provided", metadata={"chunks": 1})
        
        prompt = f"""
        Question: {input_text}
        Context chunks: {len(context_chunks)} separate pieces of information
        Answer: {output_text}
        
        Evaluate how well this answer synthesizes information from multiple context pieces.
        Consider:
        1. Does it combine information from different sources?
        2. Is the synthesis coherent and logical?
        3. Does it avoid contradictions between sources?
        4. Is the integration seamless?
        
        Rate from 1-5:
        1: Poor synthesis
        2: Basic synthesis
        3: Adequate synthesis
        4: Good synthesis
        5: Excellent synthesis
        
        Rating: """
        
        try:
            response = await self._call_model(prompt)
            score = self._parse_synthesis_score(response)
            normalized_score = score / 5.0
            passed = normalized_score >= self.threshold
            
            reasoning = f"Cross-context synthesis: {normalized_score:.3f} ({score}/5)"
            
            return ScoreResult(
                score=normalized_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"raw_score": score, "context_chunks": len(context_chunks)}
            )
            
        except Exception as e:
            return ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}", metadata={})
    
    async def _call_model(self, prompt: str):
        import asyncio
        return await asyncio.to_thread(_call_llm, self.model, prompt)
    
    def _parse_synthesis_score(self, response: str) -> float:
        import re
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 3.0  # Default to neutral
    
    def score(self, prediction, ground_truth, context=None):
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.evaluate(
                    input_text=ground_truth,
                    output_text=prediction,
                    context=context.get("context", "") if context else None
                )
            )
            loop.close()
            return result.score if hasattr(result, 'score') else result
        except Exception as e:
            return 0.0

class TechnicalAccuracyScorer(BaseScorer):
    """
    Technical domain accuracy evaluation.
    """
    def __init__(self, model, threshold=0.8, **kwargs):
        super().__init__(name="TechnicalAccuracyScorer", **kwargs)
        self.threshold = threshold
        self.model = model
    
    async def evaluate(self, input_text: str, output_text: str, expected_output: Optional[str] = None, context: Optional[str] = None, **kwargs: Any) -> ScoreResult:
        if not output_text:
            return ScoreResult(score=0.0, passed=False, reasoning="No answer provided", metadata={})
        
        prompt = f"""
        Question: {input_text}
        Context: {context or "No context provided"}
        Answer: {output_text}
        
        Evaluate the technical accuracy of this answer.
        Consider:
        1. Are technical concepts correctly explained?
        2. Are technical terms used accurately?
        3. Are technical relationships properly described?
        4. Is the technical information precise and correct?
        
        Rate from 1-5:
        1: Technically incorrect
        2: Mostly incorrect
        3: Somewhat accurate
        4: Mostly accurate
        5: Technically precise
        
        Rating: """
        
        try:
            response = await self._call_model(prompt)
            score = self._parse_technical_score(response)
            normalized_score = score / 5.0
            passed = normalized_score >= self.threshold
            
            reasoning = f"Technical accuracy: {normalized_score:.3f} ({score}/5)"
            
            return ScoreResult(
                score=normalized_score,
                passed=passed,
                reasoning=reasoning,
                metadata={"raw_score": score}
            )
            
        except Exception as e:
            return ScoreResult(score=0.0, passed=False, reasoning=f"Error: {str(e)}", metadata={})
    
    async def _call_model(self, prompt: str):
        import asyncio
        return await asyncio.to_thread(_call_llm, self.model, prompt)
    
    def _parse_technical_score(self, response: str) -> float:
        import re
        match = re.search(r"Rating:\s*(\d+)", response)
        if match:
            return float(match.group(1))
        numbers = re.findall(r"\b(\d+)\b", response)
        if numbers:
            return float(numbers[-1])
        return 3.0  # Default to neutral
    
    def score(self, prediction, ground_truth, context=None):
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.evaluate(
                    input_text=ground_truth,
                    output_text=prediction,
                    context=context.get("context", "") if context else None
                )
            )
            loop.close()
            return result.score if hasattr(result, 'score') else result
        except Exception as e:
            return 0.0 