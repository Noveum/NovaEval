"""
Enhanced RAG (Retrieval-Augmented Generation) evaluation system for NovaEval.

This module implements a comprehensive set of RAG evaluation metrics based on research
from DeepEval, Braintrust/AutoEvals, and academic literature. It provides both
component-level evaluation (retriever and generator) and end-to-end assessment.

Metrics included:
- Retrieval Metrics: Context Precision, Context Relevancy, Context Recall, Context Entity Recall
- Generation Metrics: Answer Relevancy, Answer Similarity, Answer Correctness, Faithfulness
- Composite Metrics: RAGAS Score, RAG Triad Score
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from novaeval.models.base import BaseModel as LLMModel
from novaeval.scorers.base import BaseScorer, ScoreResult


class RAGEvaluationConfig:
    """Configuration class for RAG evaluation settings."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.7,
        faithfulness_threshold: float = 0.8,
        relevancy_threshold: float = 0.7,
        precision_threshold: float = 0.7,
        recall_threshold: float = 0.7,
        answer_correctness_threshold: float = 0.8,
        ragas_weights: Optional[Dict[str, float]] = None,
    ):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.faithfulness_threshold = faithfulness_threshold
        self.relevancy_threshold = relevancy_threshold
        self.precision_threshold = precision_threshold
        self.recall_threshold = recall_threshold
        self.answer_correctness_threshold = answer_correctness_threshold

        # Default RAGAS weights
        self.ragas_weights = ragas_weights or {
            "context_precision": 0.2,
            "context_relevancy": 0.15,
            "context_recall": 0.2,
            "context_entity_recall": 0.1,
            "answer_relevancy": 0.15,
            "answer_similarity": 0.1,
            "answer_correctness": 0.15,
            "faithfulness": 0.25,
        }


class ContextPrecisionScorer(BaseScorer):
    """
    Evaluates whether the reranker ranks more relevant nodes higher than irrelevant ones.

    This metric focuses on the quality of the reranking process and whether
    the most relevant documents appear at the top of the retrieved results.
    """

    def __init__(
        self,
        model: LLMModel,
        config: Optional[RAGEvaluationConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.config = config or RAGEvaluationConfig()
        self.model = model
        self.embedding_model = SentenceTransformer(self.config.embedding_model)

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        """Evaluate context precision."""

        if not context:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="No context provided for context precision evaluation",
                metadata={"error": "no_context"},
            )

        try:
            # Handle both string and list context formats
            if isinstance(context, str):
                context_chunks = self._split_context(context)
            else:
                context_chunks = context

            if not context_chunks:
                return ScoreResult(
                    score=0.0,
                    passed=False,
                    reasoning="No context chunks found",
                    metadata={"error": "no_chunks"},
                )

            # Evaluate relevance of each context chunk with position weighting
            relevance_evaluations = []

            for i, chunk in enumerate(context_chunks):
                relevance_prompt = f"""
                Question: {input_text}
                Context chunk (position {i+1}): {chunk}

                Evaluate the relevance of this context chunk for answering the question.
                Consider:
                1. Does it contain information directly related to the question?
                2. Is the information useful for generating a complete answer?
                3. How specific and detailed is the relevant information?

                Rate the relevance on a scale of 1-5:
                1 = Not relevant at all
                2 = Slightly relevant (tangentially related)
                3 = Moderately relevant (somewhat useful)
                4 = Highly relevant (very useful)
                5 = Extremely relevant (essential for answering)

                Respond in JSON format:
                {{
                    "relevance_score": [1-5],
                    "reasoning": "Brief explanation of the relevance assessment"
                }}
                """

                try:
                    relevance_response = await self.model.generate(relevance_prompt)
                    relevance_data = self._parse_json_response(relevance_response)

                    if relevance_data and "relevance_score" in relevance_data:
                        score = float(relevance_data["relevance_score"])
                        reasoning = relevance_data.get(
                            "reasoning", "No reasoning provided"
                        )
                    else:
                        # Fallback parsing
                        score = self._parse_relevance_score(relevance_response)
                        reasoning = "Parsed from text response"

                    relevance_evaluations.append(
                        {
                            "position": i + 1,
                            "chunk": chunk[:100] + "..." if len(chunk) > 100 else chunk,
                            "relevance_score": score,
                            "reasoning": reasoning,
                        }
                    )

                except Exception as e:
                    relevance_evaluations.append(
                        {
                            "position": i + 1,
                            "chunk": chunk[:100] + "..." if len(chunk) > 100 else chunk,
                            "relevance_score": 1.0,
                            "reasoning": f"Error in evaluation: {e!s}",
                        }
                    )

            # Calculate precision with position-based weighting
            # Higher positions should have higher relevance for good precision
            total_weighted_score = 0.0
            total_weight = 0.0

            for i, eval_result in enumerate(relevance_evaluations):
                # Position weight: earlier positions get higher weight
                position_weight = 1.0 / (i + 1)  # 1.0, 0.5, 0.33, 0.25, ...
                weighted_score = eval_result["relevance_score"] * position_weight
                total_weighted_score += weighted_score
                total_weight += position_weight

            # Normalize to 0-1 scale (relevance scores are 1-5)
            precision_score = (total_weighted_score / total_weight) / 5.0

            # Calculate average relevance for reporting
            avg_relevance = sum(
                eval_result["relevance_score"] for eval_result in relevance_evaluations
            ) / len(relevance_evaluations)

            reasoning = f"""
            Context Precision Analysis:
            - Evaluated {len(context_chunks)} context chunks
            - Position-weighted precision score: {precision_score:.3f}
            - Average relevance score: {avg_relevance:.2f}/5.0

            Individual chunk evaluations:
            {chr(10).join(f'Position {eval_result["position"]}: {eval_result["relevance_score"]}/5 - {eval_result["reasoning"]}' for eval_result in relevance_evaluations)}

            Precision calculation uses position weighting where earlier positions
            (which should contain more relevant content) receive higher weights.
            """

            return ScoreResult(
                score=precision_score,
                passed=precision_score >= self.config.precision_threshold,
                reasoning=reasoning.strip(),
                metadata={
                    "context_chunks_count": len(context_chunks),
                    "relevance_evaluations": relevance_evaluations,
                    "average_relevance": avg_relevance,
                    "position_weighted_score": total_weighted_score / total_weight,
                },
            )

        except Exception as e:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"Context precision evaluation failed: {e!s}",
                metadata={"error": str(e)},
            )

    def _split_context(self, context: str) -> List[str]:
        """Split context into chunks for evaluation."""
        # Try multiple splitting strategies
        chunks = []

        # Strategy 1: Split by double newlines (paragraphs)
        if "\n\n" in context:
            chunks = [chunk.strip() for chunk in context.split("\n\n") if chunk.strip()]

        # Strategy 2: Split by document separators
        elif "---" in context or "###" in context:
            separators = ["---", "###", "***"]
            for sep in separators:
                if sep in context:
                    chunks = [
                        chunk.strip() for chunk in context.split(sep) if chunk.strip()
                    ]
                    break

        # Strategy 3: Split by sentences if no clear structure
        else:
            sentences = re.split(r"[.!?]+", context)
            # Group sentences into chunks of ~200 characters
            current_chunk = ""
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                if len(current_chunk) + len(sentence) < 200:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "

            if current_chunk:
                chunks.append(current_chunk.strip())

        # Filter out very short chunks
        min_length = 30
        filtered_chunks = [chunk for chunk in chunks if len(chunk) >= min_length]

        return filtered_chunks if filtered_chunks else [context]

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response from LLM."""
        try:
            # Try to find JSON in the response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
        except (json.JSONDecodeError, AttributeError):
            pass
        return None

    def _parse_relevance_score(self, response: str) -> float:
        """Parse relevance score from text response."""
        # Look for patterns like "Rating: X", "Score: X", or standalone numbers
        patterns = [
            r"(?:relevance_score|rating|score):\s*(\d+)",
            r"(\d+)\s*(?:/5|out of 5)",
            r"\b([1-5])\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                if 1 <= score <= 5:
                    return float(score)

        return 3.0  # Default middle score


class ContextRelevancyScorer(BaseScorer):
    """
    Evaluates whether text chunk size and top-K retrieve information without much irrelevancy.

    This metric focuses on the overall relevance of the retrieved context
    and whether it contains minimal irrelevant information.
    """

    def __init__(
        self,
        model: LLMModel,
        config: Optional[RAGEvaluationConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.config = config or RAGEvaluationConfig()
        self.model = model

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        """Evaluate context relevancy."""

        if not context:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="No context provided for context relevancy evaluation",
                metadata={"error": "no_context"},
            )

        try:
            # Handle both string and list context formats
            if isinstance(context, str):
                full_context = context
                context_chunks = self._split_context(context)
            else:
                full_context = "\n\n".join(context)
                context_chunks = context

            # Evaluate overall relevancy of the entire context
            relevancy_prompt = f"""
            Question: {input_text}

            Retrieved Context:
            {full_context}

            Evaluate the overall relevancy of this retrieved context for answering the question.
            Consider:
            1. What percentage of the context is directly relevant to the question?
            2. How much irrelevant or tangential information is included?
            3. Is the context focused and on-topic?
            4. Does the context contain noise or off-topic information?

            Provide your assessment in JSON format:
            {{
                "relevant_percentage": [0-100],
                "irrelevant_percentage": [0-100],
                "relevancy_score": [0.0-1.0],
                "key_relevant_points": ["point1", "point2", ...],
                "irrelevant_aspects": ["aspect1", "aspect2", ...],
                "reasoning": "Detailed explanation of the relevancy assessment"
            }}
            """

            relevancy_response = await self.model.generate(relevancy_prompt)
            relevancy_data = self._parse_json_response(relevancy_response)

            if relevancy_data:
                relevancy_score = float(relevancy_data.get("relevancy_score", 0.5))
                relevant_percentage = float(
                    relevancy_data.get("relevant_percentage", 50)
                )
                irrelevant_percentage = float(
                    relevancy_data.get("irrelevant_percentage", 50)
                )
                key_relevant_points = relevancy_data.get("key_relevant_points", [])
                irrelevant_aspects = relevancy_data.get("irrelevant_aspects", [])
                detailed_reasoning = relevancy_data.get(
                    "reasoning", "No detailed reasoning provided"
                )
            else:
                # Fallback: analyze sentence by sentence
                relevancy_score, relevant_percentage, detailed_reasoning = (
                    await self._fallback_relevancy_analysis(input_text, context_chunks)
                )
                irrelevant_percentage = 100 - relevant_percentage
                key_relevant_points = []
                irrelevant_aspects = []

            reasoning = f"""
            Context Relevancy Analysis:
            - Total context length: {len(full_context)} characters
            - Number of context chunks: {len(context_chunks)}
            - Relevant content: {relevant_percentage:.1f}%
            - Irrelevant content: {irrelevant_percentage:.1f}%
            - Overall relevancy score: {relevancy_score:.3f}

            Key relevant points identified:
            {chr(10).join(f'• {point}' for point in key_relevant_points) if key_relevant_points else '• None specifically identified'}

            Irrelevant aspects identified:
            {chr(10).join(f'• {aspect}' for aspect in irrelevant_aspects) if irrelevant_aspects else '• None specifically identified'}

            Detailed Assessment:
            {detailed_reasoning}
            """

            return ScoreResult(
                score=relevancy_score,
                passed=relevancy_score >= self.config.relevancy_threshold,
                reasoning=reasoning.strip(),
                metadata={
                    "relevant_percentage": relevant_percentage,
                    "irrelevant_percentage": irrelevant_percentage,
                    "context_length": len(full_context),
                    "context_chunks_count": len(context_chunks),
                    "key_relevant_points": key_relevant_points,
                    "irrelevant_aspects": irrelevant_aspects,
                },
            )

        except Exception as e:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"Context relevancy evaluation failed: {e!s}",
                metadata={"error": str(e)},
            )

    async def _fallback_relevancy_analysis(
        self, input_text: str, context_chunks: List[str]
    ) -> tuple[float, float, str]:
        """Fallback analysis when JSON parsing fails."""
        relevant_chunks = 0
        total_chunks = len(context_chunks)

        for chunk in context_chunks:
            simple_prompt = f"""
            Question: {input_text}
            Text: {chunk}

            Is this text relevant to answering the question? Reply with only "YES" or "NO".
            """

            try:
                response = await self.model.generate(simple_prompt)
                if "YES" in response.upper():
                    relevant_chunks += 1
            except:
                # If evaluation fails, assume neutral relevance
                relevant_chunks += 0.5

        relevancy_score = relevant_chunks / total_chunks if total_chunks > 0 else 0.0
        relevant_percentage = relevancy_score * 100

        reasoning = f"Fallback analysis: {relevant_chunks}/{total_chunks} chunks deemed relevant"

        return relevancy_score, relevant_percentage, reasoning

    def _split_context(self, context: str) -> List[str]:
        """Split context into chunks for evaluation."""
        # Similar to ContextPrecisionScorer but optimized for relevancy analysis
        chunks = []

        if "\n\n" in context:
            chunks = [chunk.strip() for chunk in context.split("\n\n") if chunk.strip()]
        else:
            # Split into sentences and group
            sentences = re.split(r"[.!?]+", context)
            current_chunk = ""

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                if len(current_chunk) + len(sentence) < 150:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "

            if current_chunk:
                chunks.append(current_chunk.strip())

        return [chunk for chunk in chunks if len(chunk) >= 20]

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response from LLM."""
        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
        except (json.JSONDecodeError, AttributeError):
            pass
        return None


class ContextRecallScorer(BaseScorer):
    """
    Evaluates whether the embedding model can accurately capture and retrieve relevant information.

    This metric measures if all the necessary information from the ground truth
    is present in the retrieved context.
    """

    def __init__(
        self,
        model: LLMModel,
        config: Optional[RAGEvaluationConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.config = config or RAGEvaluationConfig()
        self.model = model

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        """Evaluate context recall."""

        if not context or not expected_output:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="Both context and expected output are required for context recall evaluation",
                metadata={"error": "missing_inputs"},
            )

        try:
            # Handle both string and list context formats
            if isinstance(context, str):
                full_context = context
            else:
                full_context = "\n\n".join(context)

            # Extract key information from the expected output
            key_info_prompt = f"""
            Analyze the following expected answer and extract all key pieces of information
            that would be necessary to generate this answer.

            Expected Answer: {expected_output}

            Extract the key information in JSON format:
            {{
                "key_facts": ["fact1", "fact2", ...],
                "key_concepts": ["concept1", "concept2", ...],
                "key_details": ["detail1", "detail2", ...],
                "essential_information": ["info1", "info2", ...]
            }}
            """

            key_info_response = await self.model.generate(key_info_prompt)
            key_info_data = self._parse_json_response(key_info_response)

            if key_info_data:
                # Combine all types of key information
                all_key_info = []
                for key in [
                    "key_facts",
                    "key_concepts",
                    "key_details",
                    "essential_information",
                ]:
                    if key in key_info_data and isinstance(key_info_data[key], list):
                        all_key_info.extend(key_info_data[key])
            else:
                # Fallback: simple extraction
                all_key_info = await self._extract_key_info_fallback(expected_output)

            if not all_key_info:
                return ScoreResult(
                    score=1.0,  # No key info means perfect recall
                    passed=True,
                    reasoning="No key information extracted from expected output",
                    metadata={"key_information": []},
                )

            # Check presence of each key information in the context
            recall_evaluations = []

            for info in all_key_info:
                presence_prompt = f"""
                Context: {full_context}

                Key information to find: {info}

                Is this information present in the context? Consider:
                1. Direct mentions or statements
                2. Implicit information that can be inferred
                3. Partial information that contributes to the key point

                Respond in JSON format:
                {{
                    "presence_status": "FULLY_PRESENT|PARTIALLY_PRESENT|NOT_PRESENT",
                    "confidence": [0.0-1.0],
                    "supporting_evidence": "Quote or description of where this information appears",
                    "reasoning": "Explanation of the presence assessment"
                }}
                """

                try:
                    presence_response = await self.model.generate(presence_prompt)
                    presence_data = self._parse_json_response(presence_response)

                    if presence_data:
                        status = presence_data.get("presence_status", "NOT_PRESENT")
                        confidence = float(presence_data.get("confidence", 0.0))
                        evidence = presence_data.get(
                            "supporting_evidence", "No evidence provided"
                        )
                        reasoning = presence_data.get(
                            "reasoning", "No reasoning provided"
                        )
                    else:
                        # Fallback parsing
                        status, confidence = self._parse_presence_fallback(
                            presence_response
                        )
                        evidence = "Parsed from text response"
                        reasoning = "Fallback parsing used"

                    recall_evaluations.append(
                        {
                            "key_info": info,
                            "status": status,
                            "confidence": confidence,
                            "evidence": evidence,
                            "reasoning": reasoning,
                        }
                    )

                except Exception as e:
                    recall_evaluations.append(
                        {
                            "key_info": info,
                            "status": "NOT_PRESENT",
                            "confidence": 0.0,
                            "evidence": f"Error in evaluation: {e!s}",
                            "reasoning": "Evaluation failed",
                        }
                    )

            # Calculate recall score
            total_info = len(all_key_info)
            fully_present = sum(
                1
                for eval_result in recall_evaluations
                if eval_result["status"] == "FULLY_PRESENT"
            )
            partially_present = sum(
                1
                for eval_result in recall_evaluations
                if eval_result["status"] == "PARTIALLY_PRESENT"
            )
            not_present = total_info - fully_present - partially_present

            # Weighted recall score: full points for fully present, half points for partially present
            recall_score = (fully_present + 0.5 * partially_present) / total_info

            reasoning = f"""
            Context Recall Analysis:
            - Total key information pieces: {total_info}
            - Fully present in context: {fully_present}
            - Partially present in context: {partially_present}
            - Not present in context: {not_present}
            - Recall score: {recall_score:.3f}

            Information presence details:
            {chr(10).join(f'• {eval_result["key_info"]}: {eval_result["status"]} (confidence: {eval_result["confidence"]:.2f})' for eval_result in recall_evaluations)}

            This metric evaluates whether the retrieval system successfully captured
            all the information necessary to generate the expected answer.
            """

            return ScoreResult(
                score=recall_score,
                passed=recall_score >= self.config.recall_threshold,
                reasoning=reasoning.strip(),
                metadata={
                    "total_key_info": total_info,
                    "fully_present": fully_present,
                    "partially_present": partially_present,
                    "not_present": not_present,
                    "recall_evaluations": recall_evaluations,
                    "key_information_extracted": all_key_info,
                },
            )

        except Exception as e:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"Context recall evaluation failed: {e!s}",
                metadata={"error": str(e)},
            )

    async def _extract_key_info_fallback(self, expected_output: str) -> List[str]:
        """Fallback method to extract key information."""
        simple_prompt = f"""
        Extract the main facts and key points from this text:
        {expected_output}

        List them one per line, starting each with a dash:
        """

        try:
            response = await self.model.generate(simple_prompt)
            lines = response.strip().split("\n")
            key_info = []

            for line in lines:
                line = line.strip()
                if line.startswith("-") or line.startswith("•"):
                    key_info.append(line[1:].strip())
                elif (
                    line
                    and not line.startswith("Extract")
                    and not line.startswith("List")
                ):
                    key_info.append(line)

            return key_info[:10]  # Limit to top 10 items
        except:
            return []

    def _parse_presence_fallback(self, response: str) -> tuple[str, float]:
        """Fallback parsing for presence status."""
        response_upper = response.upper()

        if "FULLY_PRESENT" in response_upper or "FULLY PRESENT" in response_upper:
            return "FULLY_PRESENT", 0.9
        elif (
            "PARTIALLY_PRESENT" in response_upper
            or "PARTIALLY PRESENT" in response_upper
        ):
            return "PARTIALLY_PRESENT", 0.6
        elif "NOT_PRESENT" in response_upper or "NOT PRESENT" in response_upper:
            return "NOT_PRESENT", 0.1
        elif "YES" in response_upper:
            return "FULLY_PRESENT", 0.8
        elif "NO" in response_upper:
            return "NOT_PRESENT", 0.2
        else:
            return "PARTIALLY_PRESENT", 0.5  # Default to partial

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response from LLM."""
        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
        except (json.JSONDecodeError, AttributeError):
            pass
        return None


class ContextEntityRecallScorer(BaseScorer):
    """
    Evaluates entity-level recall in retrieved context.

    This metric specifically focuses on whether important entities
    (people, places, organizations, dates, etc.) from the ground truth
    are present in the retrieved context.
    """

    def __init__(
        self,
        model: LLMModel,
        config: Optional[RAGEvaluationConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.config = config or RAGEvaluationConfig()
        self.model = model

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        """Evaluate context entity recall."""

        if not context or not expected_output:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="Both context and expected output are required for entity recall evaluation",
                metadata={"error": "missing_inputs"},
            )

        try:
            # Handle both string and list context formats
            if isinstance(context, str):
                full_context = context
            else:
                full_context = "\n\n".join(context)

            # Extract entities from expected output
            entity_extraction_prompt = f"""
            Extract all important entities from the following text.
            Focus on entities that are crucial for understanding and answering questions about this content.

            Text: {expected_output}

            Extract entities in JSON format:
            {{
                "persons": ["person1", "person2", ...],
                "organizations": ["org1", "org2", ...],
                "locations": ["location1", "location2", ...],
                "dates": ["date1", "date2", ...],
                "numbers": ["number1", "number2", ...],
                "concepts": ["concept1", "concept2", ...],
                "other_entities": ["entity1", "entity2", ...]
            }}
            """

            entity_response = await self.model.generate(entity_extraction_prompt)
            entity_data = self._parse_json_response(entity_response)

            if entity_data:
                all_entities = []
                entity_types = {}

                for entity_type, entities in entity_data.items():
                    if isinstance(entities, list):
                        for entity in entities:
                            if entity and entity.strip():
                                all_entities.append(entity.strip())
                                entity_types[entity.strip()] = entity_type
            else:
                # Fallback entity extraction
                all_entities, entity_types = await self._extract_entities_fallback(
                    expected_output
                )

            if not all_entities:
                return ScoreResult(
                    score=1.0,  # No entities means perfect recall
                    passed=True,
                    reasoning="No entities extracted from expected output",
                    metadata={"entities": []},
                )

            # Check presence of each entity in the context
            entity_evaluations = []

            for entity in all_entities:
                entity_prompt = f"""
                Context: {full_context}

                Entity to find: {entity}
                Entity type: {entity_types.get(entity, 'unknown')}

                Is this entity present in the context? Consider:
                1. Exact matches
                2. Variations or synonyms
                3. Abbreviated forms
                4. Contextual references

                Respond in JSON format:
                {{
                    "is_present": true/false,
                    "match_type": "EXACT|VARIATION|REFERENCE|NOT_FOUND",
                    "matched_text": "The actual text found in context",
                    "confidence": [0.0-1.0],
                    "reasoning": "Explanation of the finding"
                }}
                """

                try:
                    entity_check_response = await self.model.generate(entity_prompt)
                    entity_check_data = self._parse_json_response(entity_check_response)

                    if entity_check_data:
                        is_present = entity_check_data.get("is_present", False)
                        match_type = entity_check_data.get("match_type", "NOT_FOUND")
                        matched_text = entity_check_data.get("matched_text", "")
                        confidence = float(entity_check_data.get("confidence", 0.0))
                        reasoning = entity_check_data.get(
                            "reasoning", "No reasoning provided"
                        )
                    else:
                        # Fallback: simple text search
                        is_present = entity.lower() in full_context.lower()
                        match_type = "EXACT" if is_present else "NOT_FOUND"
                        matched_text = entity if is_present else ""
                        confidence = 0.8 if is_present else 0.2
                        reasoning = "Simple text search"

                    entity_evaluations.append(
                        {
                            "entity": entity,
                            "entity_type": entity_types.get(entity, "unknown"),
                            "is_present": is_present,
                            "match_type": match_type,
                            "matched_text": matched_text,
                            "confidence": confidence,
                            "reasoning": reasoning,
                        }
                    )

                except Exception as e:
                    entity_evaluations.append(
                        {
                            "entity": entity,
                            "entity_type": entity_types.get(entity, "unknown"),
                            "is_present": False,
                            "match_type": "ERROR",
                            "matched_text": "",
                            "confidence": 0.0,
                            "reasoning": f"Evaluation error: {e!s}",
                        }
                    )

            # Calculate entity recall score
            total_entities = len(all_entities)
            present_entities = sum(
                1 for eval_result in entity_evaluations if eval_result["is_present"]
            )

            # Weight by match quality
            weighted_score = 0.0
            for eval_result in entity_evaluations:
                if eval_result["is_present"]:
                    match_type = eval_result["match_type"]
                    if match_type == "EXACT":
                        weighted_score += 1.0
                    elif match_type == "VARIATION":
                        weighted_score += 0.9
                    elif match_type == "REFERENCE":
                        weighted_score += 0.7
                    else:
                        weighted_score += 0.5

            entity_recall_score = (
                weighted_score / total_entities if total_entities > 0 else 1.0
            )

            # Group entities by type for reporting
            entities_by_type = {}
            for entity, entity_type in entity_types.items():
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = []
                entities_by_type[entity_type].append(entity)

            reasoning = f"""
            Context Entity Recall Analysis:
            - Total entities extracted: {total_entities}
            - Entities present in context: {present_entities}
            - Entity recall score: {entity_recall_score:.3f}

            Entities by type:
            {chr(10).join(f'• {entity_type}: {len(entities)}' for entity_type, entities in entities_by_type.items())}

            Entity presence details:
            {chr(10).join(f'• {eval_result["entity"]} ({eval_result["entity_type"]}): {eval_result["match_type"]} - {eval_result["reasoning"][:50]}...' for eval_result in entity_evaluations)}

            This metric evaluates whether important entities from the expected answer
            are captured in the retrieved context, which is crucial for factual accuracy.
            """

            return ScoreResult(
                score=entity_recall_score,
                passed=entity_recall_score >= self.config.recall_threshold,
                reasoning=reasoning.strip(),
                metadata={
                    "total_entities": total_entities,
                    "present_entities": present_entities,
                    "entities_by_type": entities_by_type,
                    "entity_evaluations": entity_evaluations,
                    "all_entities": all_entities,
                },
            )

        except Exception as e:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"Context entity recall evaluation failed: {e!s}",
                metadata={"error": str(e)},
            )

    async def _extract_entities_fallback(
        self, text: str
    ) -> tuple[List[str], Dict[str, str]]:
        """Fallback entity extraction using simple patterns."""
        entities = []
        entity_types = {}

        # Simple patterns for common entity types
        patterns = {
            "dates": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
            "numbers": r"\b\d+(?:,\d{3})*(?:\.\d+)?\b",
            "organizations": r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|LLC|Ltd|Company|Organization|University|College)\b",
            "locations": r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:City|State|Country|County|Province))\b",
        }

        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            for match in matches[:5]:  # Limit to 5 per type
                if match not in entities:
                    entities.append(match)
                    entity_types[match] = entity_type

        # Extract capitalized words as potential entities
        capitalized_words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        for word in capitalized_words[:10]:  # Limit to 10
            if word not in entities and len(word) > 3:
                entities.append(word)
                entity_types[word] = "other_entities"

        return entities, entity_types

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response from LLM."""
        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
        except (json.JSONDecodeError, AttributeError):
            pass
        return None


# Continue with the remaining scorers in the next part...

# Enhanced RAG Evaluation System - Part 2: Generation Metrics and Composite Scorers


class AnswerRelevancyScorer(BaseScorer):
    """
    Enhanced Answer Relevancy Scorer that measures how relevant the answer is to the given question.

    This metric uses multiple approaches:
    1. Question generation from answer and similarity comparison
    2. Direct relevance assessment
    3. Semantic similarity analysis
    """

    def __init__(
        self,
        model: LLMModel,
        config: Optional[RAGEvaluationConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.config = config or RAGEvaluationConfig()
        self.model = model
        self.embedding_model = SentenceTransformer(self.config.embedding_model)

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        """Evaluate answer relevancy using multiple approaches."""

        try:
            # Approach 1: Question generation and similarity
            question_gen_score = await self._evaluate_via_question_generation(
                input_text, output_text
            )

            # Approach 2: Direct relevance assessment
            direct_relevance_score = await self._evaluate_direct_relevance(
                input_text, output_text
            )

            # Approach 3: Semantic similarity
            semantic_score = self._evaluate_semantic_similarity(input_text, output_text)

            # Combine scores with weights
            weights = {"question_gen": 0.4, "direct_relevance": 0.4, "semantic": 0.2}

            final_score = (
                question_gen_score * weights["question_gen"]
                + direct_relevance_score * weights["direct_relevance"]
                + semantic_score * weights["semantic"]
            )

            reasoning = f"""
            Answer Relevancy Analysis (Multi-approach):

            1. Question Generation Approach: {question_gen_score:.3f}
               - Generated questions from answer and compared similarity with original question

            2. Direct Relevance Assessment: {direct_relevance_score:.3f}
               - LLM-based direct evaluation of answer relevance to question

            3. Semantic Similarity: {semantic_score:.3f}
               - Embedding-based similarity between question and answer

            Combined Score: {final_score:.3f}
            (Weights: Question Gen 40%, Direct 40%, Semantic 20%)

            This multi-approach method provides a more robust assessment of answer relevancy
            by combining different evaluation perspectives.
            """

            return ScoreResult(
                score=final_score,
                passed=final_score >= self.config.relevancy_threshold,
                reasoning=reasoning.strip(),
                metadata={
                    "question_generation_score": question_gen_score,
                    "direct_relevance_score": direct_relevance_score,
                    "semantic_similarity_score": semantic_score,
                    "weights": weights,
                    "approach": "multi_method",
                },
            )

        except Exception as e:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"Answer relevancy evaluation failed: {e!s}",
                metadata={"error": str(e)},
            )

    async def _evaluate_via_question_generation(
        self, input_text: str, output_text: str
    ) -> float:
        """Evaluate relevancy by generating questions from the answer."""
        question_gen_prompt = f"""
        Given the following answer, generate 3-5 questions that this answer could be responding to.
        Make the questions specific and directly related to the content.

        Answer: {output_text}

        Generate questions in JSON format:
        {{
            "questions": ["question1", "question2", "question3", ...]
        }}
        """

        try:
            response = await self.model.generate(question_gen_prompt)
            data = self._parse_json_response(response)

            if data and "questions" in data:
                generated_questions = data["questions"]
            else:
                # Fallback parsing
                generated_questions = self._parse_questions_fallback(response)

            if not generated_questions:
                return 0.5  # Neutral score if no questions generated

            # Calculate semantic similarity
            original_embedding = self.embedding_model.encode([input_text])
            generated_embeddings = self.embedding_model.encode(generated_questions)

            similarities = []
            for gen_embedding in generated_embeddings:
                similarity = np.dot(original_embedding[0], gen_embedding) / (
                    np.linalg.norm(original_embedding[0])
                    * np.linalg.norm(gen_embedding)
                )
                similarities.append(max(0, similarity))  # Ensure non-negative

            return float(np.mean(similarities))

        except Exception:
            return 0.5  # Neutral score on error

    async def _evaluate_direct_relevance(
        self, input_text: str, output_text: str
    ) -> float:
        """Direct LLM-based relevance evaluation."""
        relevance_prompt = f"""
        Question: {input_text}
        Answer: {output_text}

        Evaluate how relevant this answer is to the question on a scale of 0.0 to 1.0.
        Consider:
        1. Does the answer directly address the question?
        2. Is the information provided useful for answering the question?
        3. How well does the answer stay on topic?

        Respond in JSON format:
        {{
            "relevance_score": [0.0-1.0],
            "reasoning": "Brief explanation of the relevance assessment"
        }}
        """

        try:
            response = await self.model.generate(relevance_prompt)
            data = self._parse_json_response(response)

            if data and "relevance_score" in data:
                return float(data["relevance_score"])
            else:
                # Fallback parsing
                return self._parse_score_fallback(response)

        except Exception:
            return 0.5

    def _evaluate_semantic_similarity(self, input_text: str, output_text: str) -> float:
        """Evaluate semantic similarity between question and answer."""
        try:
            question_embedding = self.embedding_model.encode([input_text])
            answer_embedding = self.embedding_model.encode([output_text])

            similarity = np.dot(question_embedding[0], answer_embedding[0]) / (
                np.linalg.norm(question_embedding[0])
                * np.linalg.norm(answer_embedding[0])
            )

            # Normalize to 0-1 range and ensure non-negative
            return max(0, float((similarity + 1) / 2))

        except Exception:
            return 0.5

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response from LLM."""
        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        return None

    def _parse_questions_fallback(self, response: str) -> List[str]:
        """Fallback parsing for questions."""
        questions = []
        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip()
            if line and ("?" in line):
                # Remove numbering and clean up
                clean_line = re.sub(r"^\d+[\.\)]\s*", "", line)
                clean_line = re.sub(r"^[-\*]\s*", "", clean_line)
                if clean_line:
                    questions.append(clean_line)

        return questions[:5]  # Limit to 5 questions

    def _parse_score_fallback(self, response: str) -> float:
        """Fallback parsing for numerical scores."""
        # Look for decimal numbers between 0 and 1
        matches = re.findall(r"\b0?\.\d+\b", response)
        if matches:
            return float(matches[0])

        # Look for percentages
        percent_matches = re.findall(r"\b(\d+)%", response)
        if percent_matches:
            return float(percent_matches[0]) / 100

        return 0.5  # Default neutral score


class AnswerSimilarityScorer(BaseScorer):
    """
    Measures similarity between generated and expected answers using multiple similarity metrics.
    """

    def __init__(
        self,
        model: LLMModel,
        config: Optional[RAGEvaluationConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.config = config or RAGEvaluationConfig()
        self.model = model
        self.embedding_model = SentenceTransformer(self.config.embedding_model)

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        """Evaluate answer similarity using multiple metrics."""

        if not expected_output:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="Expected output is required for answer similarity evaluation",
                metadata={"error": "no_expected_output"},
            )

        try:
            # Calculate different similarity metrics
            semantic_similarity = self._calculate_semantic_similarity(
                output_text, expected_output
            )
            lexical_similarity = self._calculate_lexical_similarity(
                output_text, expected_output
            )
            structural_similarity = await self._calculate_structural_similarity(
                output_text, expected_output
            )

            # Combine similarities with weights
            weights = {"semantic": 0.5, "lexical": 0.3, "structural": 0.2}

            combined_similarity = (
                semantic_similarity * weights["semantic"]
                + lexical_similarity * weights["lexical"]
                + structural_similarity * weights["structural"]
            )

            reasoning = f"""
            Answer Similarity Analysis:

            1. Semantic Similarity: {semantic_similarity:.3f}
               - Embedding-based similarity using {self.config.embedding_model}

            2. Lexical Similarity: {lexical_similarity:.3f}
               - Token overlap and BLEU-like metrics

            3. Structural Similarity: {structural_similarity:.3f}
               - LLM-based assessment of content structure and organization

            Combined Similarity Score: {combined_similarity:.3f}
            (Weights: Semantic 50%, Lexical 30%, Structural 20%)

            Generated Answer Length: {len(output_text)} characters
            Expected Answer Length: {len(expected_output)} characters
            """

            return ScoreResult(
                score=combined_similarity,
                passed=combined_similarity >= self.config.similarity_threshold,
                reasoning=reasoning.strip(),
                metadata={
                    "semantic_similarity": semantic_similarity,
                    "lexical_similarity": lexical_similarity,
                    "structural_similarity": structural_similarity,
                    "weights": weights,
                    "generated_length": len(output_text),
                    "expected_length": len(expected_output),
                },
            )

        except Exception as e:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"Answer similarity evaluation failed: {e!s}",
                metadata={"error": str(e)},
            )

    def _calculate_semantic_similarity(
        self, output_text: str, expected_output: str
    ) -> float:
        """Calculate semantic similarity using embeddings."""
        try:
            output_embedding = self.embedding_model.encode([output_text])
            expected_embedding = self.embedding_model.encode([expected_output])

            similarity = np.dot(output_embedding[0], expected_embedding[0]) / (
                np.linalg.norm(output_embedding[0])
                * np.linalg.norm(expected_embedding[0])
            )

            # Normalize to 0-1 range
            return max(0, float((similarity + 1) / 2))

        except Exception:
            return 0.0

    def _calculate_lexical_similarity(
        self, output_text: str, expected_output: str
    ) -> float:
        """Calculate lexical similarity using token overlap."""
        try:
            # Simple tokenization
            output_tokens = set(output_text.lower().split())
            expected_tokens = set(expected_output.lower().split())

            if not output_tokens and not expected_tokens:
                return 1.0
            if not output_tokens or not expected_tokens:
                return 0.0

            # Jaccard similarity
            intersection = len(output_tokens.intersection(expected_tokens))
            union = len(output_tokens.union(expected_tokens))

            jaccard = intersection / union if union > 0 else 0.0

            # Also calculate precision and recall
            precision = intersection / len(output_tokens) if output_tokens else 0.0
            recall = intersection / len(expected_tokens) if expected_tokens else 0.0

            # F1-like combination
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0

            # Combine Jaccard and F1
            return (jaccard + f1) / 2

        except Exception:
            return 0.0

    async def _calculate_structural_similarity(
        self, output_text: str, expected_output: str
    ) -> float:
        """Calculate structural similarity using LLM assessment."""
        structure_prompt = f"""
        Compare the structure and organization of these two answers:

        Answer 1: {output_text}

        Answer 2: {expected_output}

        Evaluate their structural similarity considering:
        1. Information organization and flow
        2. Key points coverage
        3. Logical structure
        4. Content completeness

        Rate the structural similarity on a scale of 0.0 to 1.0.

        Respond in JSON format:
        {{
            "structural_similarity": [0.0-1.0],
            "reasoning": "Brief explanation of the structural comparison"
        }}
        """

        try:
            response = await self.model.generate(structure_prompt)
            data = self._parse_json_response(response)

            if data and "structural_similarity" in data:
                return float(data["structural_similarity"])
            else:
                return self._parse_score_fallback(response)

        except Exception:
            return 0.5

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response from LLM."""
        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        return None

    def _parse_score_fallback(self, response: str) -> float:
        """Fallback parsing for numerical scores."""
        matches = re.findall(r"\b0?\.\d+\b", response)
        if matches:
            return float(matches[0])
        return 0.5


class AnswerCorrectnessScorer(BaseScorer):
    """
    Evaluates generated answer against the golden/ground truth answer.

    This metric checks each statement in the answer and classifies them as
    true positive, false positive, or false negative.
    """

    def __init__(
        self,
        model: LLMModel,
        config: Optional[RAGEvaluationConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.config = config or RAGEvaluationConfig()
        self.model = model

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        """Evaluate answer correctness."""

        if not expected_output:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="Expected output is required for answer correctness evaluation",
                metadata={"error": "no_expected_output"},
            )

        try:
            # Extract statements from both answers
            generated_statements = await self._extract_statements(
                output_text, "generated"
            )
            expected_statements = await self._extract_statements(
                expected_output, "expected"
            )

            if not generated_statements and not expected_statements:
                return ScoreResult(
                    score=1.0,
                    passed=True,
                    reasoning="No statements found in either answer",
                    metadata={"statements": {"generated": [], "expected": []}},
                )

            # Classify each generated statement
            statement_classifications = []

            for stmt in generated_statements:
                classification = await self._classify_statement(
                    stmt, expected_statements, expected_output
                )
                statement_classifications.append(classification)

            # Calculate correctness metrics
            true_positives = sum(
                1
                for c in statement_classifications
                if c["classification"] == "TRUE_POSITIVE"
            )
            false_positives = sum(
                1
                for c in statement_classifications
                if c["classification"] == "FALSE_POSITIVE"
            )

            # Calculate false negatives (important info in expected but missing in generated)
            false_negatives = await self._calculate_false_negatives(
                expected_statements, output_text
            )

            # Calculate precision, recall, and F1
            total_generated = len(generated_statements)
            total_expected = len(expected_statements)

            precision = true_positives / total_generated if total_generated > 0 else 0.0
            recall = (
                true_positives / (true_positives + false_negatives)
                if (true_positives + false_negatives) > 0
                else 0.0
            )

            if precision + recall > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = 0.0

            # Use F1 score as the main correctness metric
            correctness_score = f1_score

            reasoning = f"""
            Answer Correctness Analysis:

            Statement Classification:
            - True Positives: {true_positives} (correct statements in generated answer)
            - False Positives: {false_positives} (incorrect statements in generated answer)
            - False Negatives: {false_negatives} (missing important statements)

            Metrics:
            - Precision: {precision:.3f} ({true_positives}/{total_generated})
            - Recall: {recall:.3f} ({true_positives}/{true_positives + false_negatives})
            - F1 Score (Correctness): {f1_score:.3f}

            Generated Statements: {total_generated}
            Expected Statements: {total_expected}

            Statement Details:
            {chr(10).join(f'• {c["statement"][:60]}... → {c["classification"]}' for c in statement_classifications)}
            """

            return ScoreResult(
                score=correctness_score,
                passed=correctness_score >= self.config.answer_correctness_threshold,
                reasoning=reasoning.strip(),
                metadata={
                    "true_positives": true_positives,
                    "false_positives": false_positives,
                    "false_negatives": false_negatives,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "statement_classifications": statement_classifications,
                    "generated_statements_count": total_generated,
                    "expected_statements_count": total_expected,
                },
            )

        except Exception as e:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"Answer correctness evaluation failed: {e!s}",
                metadata={"error": str(e)},
            )

    async def _extract_statements(self, text: str, source: str) -> List[str]:
        """Extract factual statements from text."""
        extraction_prompt = f"""
        Extract all factual statements from the following {source} answer.
        Focus on specific, verifiable claims and facts.

        Text: {text}

        Extract statements in JSON format:
        {{
            "statements": ["statement1", "statement2", ...]
        }}
        """

        try:
            response = await self.model.generate(extraction_prompt)
            data = self._parse_json_response(response)

            if data and "statements" in data:
                return data["statements"]
            else:
                # Fallback: split by sentences
                sentences = re.split(r"[.!?]+", text)
                return [
                    s.strip() for s in sentences if s.strip() and len(s.strip()) > 10
                ]

        except Exception:
            return []

    async def _classify_statement(
        self, statement: str, expected_statements: List[str], expected_full: str
    ) -> Dict[str, Any]:
        """Classify a statement as true positive or false positive."""
        classification_prompt = f"""
        Statement to classify: {statement}

        Expected answer: {expected_full}

        Expected statements: {json.dumps(expected_statements)}

        Is this statement correct based on the expected answer?

        Classify as:
        - TRUE_POSITIVE: Statement is factually correct and supported by expected answer
        - FALSE_POSITIVE: Statement is incorrect or not supported by expected answer

        Respond in JSON format:
        {{
            "classification": "TRUE_POSITIVE|FALSE_POSITIVE",
            "confidence": [0.0-1.0],
            "reasoning": "Brief explanation of the classification"
        }}
        """

        try:
            response = await self.model.generate(classification_prompt)
            data = self._parse_json_response(response)

            if data:
                classification = data.get("classification", "FALSE_POSITIVE")
                confidence = float(data.get("confidence", 0.5))
                reasoning = data.get("reasoning", "No reasoning provided")
            else:
                # Fallback: simple keyword matching
                classification = (
                    "TRUE_POSITIVE"
                    if any(
                        word in expected_full.lower()
                        for word in statement.lower().split()
                    )
                    else "FALSE_POSITIVE"
                )
                confidence = 0.6
                reasoning = "Fallback keyword matching"

            return {
                "statement": statement,
                "classification": classification,
                "confidence": confidence,
                "reasoning": reasoning,
            }

        except Exception:
            return {
                "statement": statement,
                "classification": "FALSE_POSITIVE",
                "confidence": 0.0,
                "reasoning": "Classification failed",
            }

    async def _calculate_false_negatives(
        self, expected_statements: List[str], generated_text: str
    ) -> int:
        """Calculate false negatives (missing important information)."""
        false_negatives = 0

        for expected_stmt in expected_statements:
            missing_check_prompt = f"""
            Expected statement: {expected_stmt}
            Generated answer: {generated_text}

            Is the information from the expected statement present in the generated answer?
            Consider paraphrasing and different wordings.

            Respond with only "PRESENT" or "MISSING".
            """

            try:
                response = await self.model.generate(missing_check_prompt)
                if "MISSING" in response.upper():
                    false_negatives += 1
            except Exception:
                false_negatives += 1  # Conservative: assume missing on error

        return false_negatives

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response from LLM."""
        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        return None


class EnhancedFaithfulnessScorer(BaseScorer):
    """
    Enhanced Faithfulness Scorer that evaluates whether the answer is faithful to the provided context.

    This version includes more sophisticated claim extraction and verification.
    """

    def __init__(
        self,
        model: LLMModel,
        config: Optional[RAGEvaluationConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.config = config or RAGEvaluationConfig()
        self.model = model

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        """Evaluate faithfulness to context with enhanced methodology."""

        if not context:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="No context provided for faithfulness evaluation",
                metadata={"error": "no_context"},
            )

        try:
            # Handle both string and list context formats
            if isinstance(context, str):
                full_context = context
            else:
                full_context = "\n\n".join(context)

            # Extract claims with categorization
            claims_data = await self._extract_categorized_claims(output_text)

            if not claims_data["all_claims"]:
                return ScoreResult(
                    score=1.0,  # No claims means no unfaithful content
                    passed=True,
                    reasoning="No factual claims found in the answer",
                    metadata={"claims": []},
                )

            # Verify each claim against the context
            verification_results = []

            for claim in claims_data["all_claims"]:
                verification = await self._verify_claim_enhanced(claim, full_context)
                verification_results.append(verification)

            # Calculate faithfulness score with weighted categories
            category_weights = {
                "factual": 1.0,
                "numerical": 1.0,
                "temporal": 0.9,
                "relational": 0.8,
                "opinion": 0.5,
            }

            total_weighted_score = 0.0
            total_weight = 0.0

            for verification in verification_results:
                claim_category = verification.get("category", "factual")
                weight = category_weights.get(claim_category, 1.0)

                if verification["status"] == "SUPPORTED":
                    score = 1.0
                elif verification["status"] == "PARTIALLY_SUPPORTED":
                    score = 0.6
                else:
                    score = 0.0

                total_weighted_score += score * weight
                total_weight += weight

            faithfulness_score = (
                total_weighted_score / total_weight if total_weight > 0 else 1.0
            )

            # Calculate category-wise statistics
            category_stats = self._calculate_category_stats(verification_results)

            reasoning = f"""
            Enhanced Faithfulness Analysis:
            - Total claims extracted: {len(claims_data["all_claims"])}
            - Weighted faithfulness score: {faithfulness_score:.3f}

            Claims by category:
            {chr(10).join(f'• {cat}: {count}' for cat, count in claims_data["categories"].items())}

            Verification results:
            - Fully supported: {category_stats["supported"]}
            - Partially supported: {category_stats["partial"]}
            - Not supported: {category_stats["not_supported"]}

            Detailed claim verification:
            {chr(10).join(f'• {v["claim"][:50]}... → {v["status"]} ({v.get("category", "unknown")})' for v in verification_results)}

            This enhanced faithfulness metric considers claim categories and applies
            appropriate weights to different types of factual assertions.
            """

            return ScoreResult(
                score=faithfulness_score,
                passed=faithfulness_score >= self.config.faithfulness_threshold,
                reasoning=reasoning.strip(),
                metadata={
                    "total_claims": len(claims_data["all_claims"]),
                    "claims_by_category": claims_data["categories"],
                    "verification_results": verification_results,
                    "category_stats": category_stats,
                    "weighted_score": faithfulness_score,
                },
            )

        except Exception as e:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"Enhanced faithfulness evaluation failed: {e!s}",
                metadata={"error": str(e)},
            )

    async def _extract_categorized_claims(self, text: str) -> Dict[str, Any]:
        """Extract and categorize claims from text."""
        extraction_prompt = f"""
        Extract all factual claims from the following text and categorize them.

        Text: {text}

        Categorize claims as:
        - factual: General factual statements
        - numerical: Claims involving numbers, quantities, measurements
        - temporal: Claims involving dates, times, sequences
        - relational: Claims about relationships between entities
        - opinion: Subjective statements or opinions

        Respond in JSON format:
        {{
            "factual_claims": ["claim1", "claim2", ...],
            "numerical_claims": ["claim1", "claim2", ...],
            "temporal_claims": ["claim1", "claim2", ...],
            "relational_claims": ["claim1", "claim2", ...],
            "opinion_claims": ["claim1", "claim2", ...]
        }}
        """

        try:
            response = await self.model.generate(extraction_prompt)
            data = self._parse_json_response(response)

            if data:
                all_claims = []
                categories = {}

                for category in [
                    "factual_claims",
                    "numerical_claims",
                    "temporal_claims",
                    "relational_claims",
                    "opinion_claims",
                ]:
                    claims = data.get(category, [])
                    categories[category.replace("_claims", "")] = len(claims)
                    all_claims.extend(claims)

                return {"all_claims": all_claims, "categories": categories}
            else:
                # Fallback: simple sentence splitting
                sentences = re.split(r"[.!?]+", text)
                claims = [
                    s.strip() for s in sentences if s.strip() and len(s.strip()) > 10
                ]
                return {"all_claims": claims, "categories": {"factual": len(claims)}}

        except Exception:
            return {"all_claims": [], "categories": {}}

    async def _verify_claim_enhanced(self, claim: str, context: str) -> Dict[str, Any]:
        """Enhanced claim verification with detailed analysis."""
        verification_prompt = f"""
        Context: {context}

        Claim to verify: {claim}

        Verify this claim against the context and provide detailed analysis.

        Respond in JSON format:
        {{
            "status": "SUPPORTED|PARTIALLY_SUPPORTED|NOT_SUPPORTED",
            "confidence": [0.0-1.0],
            "category": "factual|numerical|temporal|relational|opinion",
            "supporting_evidence": "Direct quote or paraphrase from context",
            "reasoning": "Detailed explanation of verification",
            "contradictions": "Any contradictory information found"
        }}
        """

        try:
            response = await self.model.generate(verification_prompt)
            data = self._parse_json_response(response)

            if data:
                return {
                    "claim": claim,
                    "status": data.get("status", "NOT_SUPPORTED"),
                    "confidence": float(data.get("confidence", 0.0)),
                    "category": data.get("category", "factual"),
                    "supporting_evidence": data.get("supporting_evidence", ""),
                    "reasoning": data.get("reasoning", ""),
                    "contradictions": data.get("contradictions", ""),
                }
            else:
                # Fallback: simple text search
                status = (
                    "SUPPORTED" if claim.lower() in context.lower() else "NOT_SUPPORTED"
                )
                return {
                    "claim": claim,
                    "status": status,
                    "confidence": 0.6 if status == "SUPPORTED" else 0.4,
                    "category": "factual",
                    "supporting_evidence": "Simple text matching",
                    "reasoning": "Fallback verification",
                    "contradictions": "",
                }

        except Exception:
            return {
                "claim": claim,
                "status": "NOT_SUPPORTED",
                "confidence": 0.0,
                "category": "factual",
                "supporting_evidence": "",
                "reasoning": "Verification failed",
                "contradictions": "",
            }

    def _calculate_category_stats(
        self, verification_results: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Calculate statistics by verification status."""
        stats = {"supported": 0, "partial": 0, "not_supported": 0}

        for result in verification_results:
            status = result.get("status", "NOT_SUPPORTED")
            if status == "SUPPORTED":
                stats["supported"] += 1
            elif status == "PARTIALLY_SUPPORTED":
                stats["partial"] += 1
            else:
                stats["not_supported"] += 1

        return stats

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response from LLM."""
        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        return None


# Continue with composite scorers...

# Enhanced RAG Evaluation System - Part 3: Composite Scorers and Integration


class EnhancedRAGASScorer(BaseScorer):
    """
    Enhanced RAGAS (Retrieval-Augmented Generation Assessment) scorer.

    Combines all RAG metrics into a comprehensive evaluation with configurable weights
    and detailed component analysis.
    """

    def __init__(
        self,
        model: LLMModel,
        config: Optional[RAGEvaluationConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.config = config or RAGEvaluationConfig()
        self.model = model

        # Initialize all component scorers
        self.context_precision_scorer = ContextPrecisionScorer(model, config)
        self.context_relevancy_scorer = ContextRelevancyScorer(model, config)
        self.context_recall_scorer = ContextRecallScorer(model, config)
        self.context_entity_recall_scorer = ContextEntityRecallScorer(model, config)
        self.answer_relevancy_scorer = AnswerRelevancyScorer(model, config)
        self.answer_similarity_scorer = AnswerSimilarityScorer(model, config)
        self.answer_correctness_scorer = AnswerCorrectnessScorer(model, config)
        self.faithfulness_scorer = EnhancedFaithfulnessScorer(model, config)

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        """Comprehensive RAGAS evaluation."""

        try:
            # Run all evaluations in parallel for efficiency
            evaluation_tasks = [
                (
                    "context_precision",
                    self.context_precision_scorer.evaluate(
                        input_text, output_text, expected_output, context
                    ),
                ),
                (
                    "context_relevancy",
                    self.context_relevancy_scorer.evaluate(
                        input_text, output_text, expected_output, context
                    ),
                ),
                (
                    "context_recall",
                    self.context_recall_scorer.evaluate(
                        input_text, output_text, expected_output, context
                    ),
                ),
                (
                    "context_entity_recall",
                    self.context_entity_recall_scorer.evaluate(
                        input_text, output_text, expected_output, context
                    ),
                ),
                (
                    "answer_relevancy",
                    self.answer_relevancy_scorer.evaluate(
                        input_text, output_text, expected_output, context
                    ),
                ),
                (
                    "answer_similarity",
                    self.answer_similarity_scorer.evaluate(
                        input_text, output_text, expected_output, context
                    ),
                ),
                (
                    "answer_correctness",
                    self.answer_correctness_scorer.evaluate(
                        input_text, output_text, expected_output, context
                    ),
                ),
                (
                    "faithfulness",
                    self.faithfulness_scorer.evaluate(
                        input_text, output_text, expected_output, context
                    ),
                ),
            ]

            # Execute all evaluations
            results = await asyncio.gather(
                *[task[1] for task in evaluation_tasks], return_exceptions=True
            )

            # Process results
            component_scores = {}
            component_details = {}
            failed_components = []

            for i, (metric_name, result) in enumerate(
                zip([task[0] for task in evaluation_tasks], results)
            ):
                if isinstance(result, Exception):
                    component_scores[metric_name] = 0.0
                    failed_components.append(f"{metric_name}: {result!s}")
                elif hasattr(result, "score"):
                    component_scores[metric_name] = result.score
                    component_details[metric_name] = {
                        "score": result.score,
                        "passed": result.passed,
                        "reasoning": result.reasoning,
                        "metadata": result.metadata,
                    }
                else:
                    component_scores[metric_name] = 0.0
                    failed_components.append(f"{metric_name}: Invalid result type")

            # Calculate weighted RAGAS score
            weights = self.config.ragas_weights
            total_weight = sum(weights.values())

            ragas_score = (
                sum(
                    component_scores.get(metric, 0.0) * weight
                    for metric, weight in weights.items()
                )
                / total_weight
            )

            # Calculate component group scores
            retrieval_metrics = [
                "context_precision",
                "context_relevancy",
                "context_recall",
                "context_entity_recall",
            ]
            generation_metrics = [
                "answer_relevancy",
                "answer_similarity",
                "answer_correctness",
                "faithfulness",
            ]

            retrieval_score = sum(
                component_scores.get(m, 0.0) for m in retrieval_metrics
            ) / len(retrieval_metrics)
            generation_score = sum(
                component_scores.get(m, 0.0) for m in generation_metrics
            ) / len(generation_metrics)

            # Determine overall pass/fail
            passed_components = sum(
                1 for details in component_details.values() if details["passed"]
            )
            total_components = len(component_details)
            overall_passed = (
                ragas_score >= 0.7 and passed_components >= total_components * 0.6
            )

            # Generate comprehensive reasoning
            reasoning = f"""
            Enhanced RAGAS Evaluation Results:

            Overall RAGAS Score: {ragas_score:.3f}

            Component Group Scores:
            • Retrieval Pipeline: {retrieval_score:.3f}
            • Generation Pipeline: {generation_score:.3f}

            Individual Component Scores:
            {chr(10).join(f'• {metric.replace("_", " ").title()}: {score:.3f} (weight: {weights.get(metric, 0.0):.2f})' for metric, score in component_scores.items())}

            Component Pass/Fail Status:
            • Passed: {passed_components}/{total_components} components
            • Overall Status: {"PASSED" if overall_passed else "FAILED"}

            {f"Failed Components: {chr(10).join(failed_components)}" if failed_components else "All components evaluated successfully"}

            This comprehensive evaluation assesses both the retrieval and generation
            components of the RAG system using multiple complementary metrics.
            """

            return ScoreResult(
                score=ragas_score,
                passed=overall_passed,
                reasoning=reasoning.strip(),
                metadata={
                    "component_scores": component_scores,
                    "component_details": component_details,
                    "retrieval_score": retrieval_score,
                    "generation_score": generation_score,
                    "weights": weights,
                    "passed_components": passed_components,
                    "total_components": total_components,
                    "failed_components": failed_components,
                },
            )

        except Exception as e:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"Enhanced RAGAS evaluation failed: {e!s}",
                metadata={"error": str(e)},
            )


class RAGTriadScorer(BaseScorer):
    """
    RAG Triad Scorer focusing on the three core aspects of RAG evaluation:
    1. Context Relevance (retrieval quality)
    2. Groundedness (faithfulness to context)
    3. Answer Relevance (response quality)

    This provides a simplified but comprehensive RAG assessment.
    """

    def __init__(
        self,
        model: LLMModel,
        config: Optional[RAGEvaluationConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.config = config or RAGEvaluationConfig()
        self.model = model

        # Initialize core scorers
        self.context_relevancy_scorer = ContextRelevancyScorer(model, config)
        self.faithfulness_scorer = EnhancedFaithfulnessScorer(model, config)
        self.answer_relevancy_scorer = AnswerRelevancyScorer(model, config)

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        """Evaluate using RAG Triad methodology."""

        try:
            # Run the three core evaluations
            context_relevance_result = await self.context_relevancy_scorer.evaluate(
                input_text, output_text, expected_output, context
            )

            groundedness_result = await self.faithfulness_scorer.evaluate(
                input_text, output_text, expected_output, context
            )

            answer_relevance_result = await self.answer_relevancy_scorer.evaluate(
                input_text, output_text, expected_output, context
            )

            # Extract scores
            context_relevance_score = (
                context_relevance_result.score
                if hasattr(context_relevance_result, "score")
                else 0.0
            )
            groundedness_score = (
                groundedness_result.score
                if hasattr(groundedness_result, "score")
                else 0.0
            )
            answer_relevance_score = (
                answer_relevance_result.score
                if hasattr(answer_relevance_result, "score")
                else 0.0
            )

            # Calculate triad score (equal weights)
            triad_score = (
                context_relevance_score + groundedness_score + answer_relevance_score
            ) / 3.0

            # Determine pass/fail for each component
            context_passed = (
                context_relevance_result.passed
                if hasattr(context_relevance_result, "passed")
                else False
            )
            groundedness_passed = (
                groundedness_result.passed
                if hasattr(groundedness_result, "passed")
                else False
            )
            answer_passed = (
                answer_relevance_result.passed
                if hasattr(answer_relevance_result, "passed")
                else False
            )

            # Overall pass requires all three components to pass
            overall_passed = context_passed and groundedness_passed and answer_passed

            reasoning = f"""
            RAG Triad Evaluation Results:

            The RAG Triad evaluates three fundamental aspects of RAG systems:

            1. Context Relevance: {context_relevance_score:.3f} {"✓" if context_passed else "✗"}
               - Measures the quality and relevance of retrieved context
               - Ensures minimal irrelevant information in retrieval

            2. Groundedness (Faithfulness): {groundedness_score:.3f} {"✓" if groundedness_passed else "✗"}
               - Evaluates whether the answer is faithful to the provided context
               - Prevents hallucination and ensures factual accuracy

            3. Answer Relevance: {answer_relevance_score:.3f} {"✓" if answer_passed else "✗"}
               - Assesses how well the answer addresses the original question
               - Ensures the response is on-topic and useful

            Overall Triad Score: {triad_score:.3f}
            Overall Status: {"PASSED" if overall_passed else "FAILED"}

            The RAG Triad provides a balanced assessment covering retrieval quality,
            factual grounding, and response relevance - the three pillars of effective RAG.
            """

            return ScoreResult(
                score=triad_score,
                passed=overall_passed,
                reasoning=reasoning.strip(),
                metadata={
                    "context_relevance": {
                        "score": context_relevance_score,
                        "passed": context_passed,
                        "details": (
                            context_relevance_result.metadata
                            if hasattr(context_relevance_result, "metadata")
                            else {}
                        ),
                    },
                    "groundedness": {
                        "score": groundedness_score,
                        "passed": groundedness_passed,
                        "details": (
                            groundedness_result.metadata
                            if hasattr(groundedness_result, "metadata")
                            else {}
                        ),
                    },
                    "answer_relevance": {
                        "score": answer_relevance_score,
                        "passed": answer_passed,
                        "details": (
                            answer_relevance_result.metadata
                            if hasattr(answer_relevance_result, "metadata")
                            else {}
                        ),
                    },
                    "triad_methodology": True,
                },
            )

        except Exception as e:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"RAG Triad evaluation failed: {e!s}",
                metadata={"error": str(e)},
            )


class RAGEvaluationSuite:
    """
    Comprehensive RAG Evaluation Suite that provides easy access to all RAG metrics
    and evaluation methodologies.
    """

    def __init__(
        self,
        model: LLMModel,
        config: Optional[RAGEvaluationConfig] = None,
    ):
        self.model = model
        self.config = config or RAGEvaluationConfig()

        # Initialize all available scorers
        self.scorers = {
            # Individual component scorers
            "context_precision": ContextPrecisionScorer(model, config),
            "context_relevancy": ContextRelevancyScorer(model, config),
            "context_recall": ContextRecallScorer(model, config),
            "context_entity_recall": ContextEntityRecallScorer(model, config),
            "answer_relevancy": AnswerRelevancyScorer(model, config),
            "answer_similarity": AnswerSimilarityScorer(model, config),
            "answer_correctness": AnswerCorrectnessScorer(model, config),
            "faithfulness": EnhancedFaithfulnessScorer(model, config),
            # Composite scorers
            "ragas": EnhancedRAGASScorer(model, config),
            "rag_triad": RAGTriadScorer(model, config),
        }

    async def evaluate_single_metric(
        self,
        metric_name: str,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        """Evaluate using a single metric."""

        if metric_name not in self.scorers:
            available_metrics = list(self.scorers.keys())
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"Unknown metric '{metric_name}'. Available metrics: {', '.join(available_metrics)}",
                metadata={
                    "error": "unknown_metric",
                    "available_metrics": available_metrics,
                },
            )

        scorer = self.scorers[metric_name]
        return await scorer.evaluate(
            input_text, output_text, expected_output, context, **kwargs
        )

    async def evaluate_retrieval_pipeline(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> Dict[str, ScoreResult]:
        """Evaluate only the retrieval pipeline components."""

        retrieval_metrics = [
            "context_precision",
            "context_relevancy",
            "context_recall",
            "context_entity_recall",
        ]

        results = {}
        for metric in retrieval_metrics:
            results[metric] = await self.evaluate_single_metric(
                metric, input_text, output_text, expected_output, context, **kwargs
            )

        return results

    async def evaluate_generation_pipeline(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> Dict[str, ScoreResult]:
        """Evaluate only the generation pipeline components."""

        generation_metrics = [
            "answer_relevancy",
            "answer_similarity",
            "answer_correctness",
            "faithfulness",
        ]

        results = {}
        for metric in generation_metrics:
            results[metric] = await self.evaluate_single_metric(
                metric, input_text, output_text, expected_output, context, **kwargs
            )

        return results

    async def evaluate_comprehensive(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[Union[str, List[str]]] = None,
        include_individual: bool = True,
        **kwargs: Any,
    ) -> Dict[str, ScoreResult]:
        """Run comprehensive evaluation with all metrics."""

        results = {}

        if include_individual:
            # Run all individual metrics
            individual_metrics = [
                "context_precision",
                "context_relevancy",
                "context_recall",
                "context_entity_recall",
                "answer_relevancy",
                "answer_similarity",
                "answer_correctness",
                "faithfulness",
            ]

            for metric in individual_metrics:
                results[metric] = await self.evaluate_single_metric(
                    metric, input_text, output_text, expected_output, context, **kwargs
                )

        # Run composite metrics
        results["ragas"] = await self.evaluate_single_metric(
            "ragas", input_text, output_text, expected_output, context, **kwargs
        )

        results["rag_triad"] = await self.evaluate_single_metric(
            "rag_triad", input_text, output_text, expected_output, context, **kwargs
        )

        return results

    def get_available_metrics(self) -> List[str]:
        """Get list of all available metrics."""
        return list(self.scorers.keys())

    def get_metric_info(self) -> Dict[str, str]:
        """Get information about each available metric."""
        return {
            "context_precision": "Evaluates whether the reranker ranks more relevant nodes higher than irrelevant ones",
            "context_relevancy": "Evaluates whether text chunk size and top-K retrieve information without much irrelevancy",
            "context_recall": "Evaluates whether the embedding model can accurately capture and retrieve relevant information",
            "context_entity_recall": "Evaluates entity-level recall in retrieved context",
            "answer_relevancy": "Measures how relevant the answer is to the given question",
            "answer_similarity": "Measures similarity between generated and expected answers",
            "answer_correctness": "Evaluates generated answer against the golden/ground truth answer",
            "faithfulness": "Evaluates whether the response is faithful to the provided context",
            "ragas": "Comprehensive RAGAS score combining all individual metrics",
            "rag_triad": "Simplified evaluation focusing on context relevance, groundedness, and answer relevance",
        }


# Utility functions for integration with existing NovaEval architecture


def create_rag_scorer(
    scorer_type: str,
    model: LLMModel,
    config: Optional[RAGEvaluationConfig] = None,
    **kwargs: Any,
) -> BaseScorer:
    """Factory function to create RAG scorers."""

    config = config or RAGEvaluationConfig()

    scorer_mapping = {
        "context_precision": ContextPrecisionScorer,
        "context_relevancy": ContextRelevancyScorer,
        "context_recall": ContextRecallScorer,
        "context_entity_recall": ContextEntityRecallScorer,
        "answer_relevancy": AnswerRelevancyScorer,
        "answer_similarity": AnswerSimilarityScorer,
        "answer_correctness": AnswerCorrectnessScorer,
        "faithfulness": EnhancedFaithfulnessScorer,
        "ragas": EnhancedRAGASScorer,
        "rag_triad": RAGTriadScorer,
    }

    if scorer_type not in scorer_mapping:
        raise ValueError(
            f"Unknown RAG scorer type: {scorer_type}. Available: {list(scorer_mapping.keys())}"
        )

    scorer_class = scorer_mapping[scorer_type]
    return scorer_class(model, config, **kwargs)


def get_default_rag_config() -> RAGEvaluationConfig:
    """Get default RAG evaluation configuration."""
    return RAGEvaluationConfig()


def get_optimized_rag_config(focus: str = "balanced") -> RAGEvaluationConfig:
    """Get optimized RAG configuration for different use cases."""

    if focus == "precision":
        # Focus on precision and accuracy
        return RAGEvaluationConfig(
            faithfulness_threshold=0.9,
            answer_correctness_threshold=0.9,
            precision_threshold=0.8,
            ragas_weights={
                "context_precision": 0.25,
                "context_relevancy": 0.1,
                "context_recall": 0.15,
                "context_entity_recall": 0.1,
                "answer_relevancy": 0.1,
                "answer_similarity": 0.05,
                "answer_correctness": 0.25,
                "faithfulness": 0.3,
            },
        )
    elif focus == "recall":
        # Focus on recall and completeness
        return RAGEvaluationConfig(
            recall_threshold=0.8,
            relevancy_threshold=0.6,
            ragas_weights={
                "context_precision": 0.15,
                "context_relevancy": 0.2,
                "context_recall": 0.25,
                "context_entity_recall": 0.2,
                "answer_relevancy": 0.15,
                "answer_similarity": 0.05,
                "answer_correctness": 0.15,
                "faithfulness": 0.2,
            },
        )
    elif focus == "speed":
        # Optimized for faster evaluation
        return RAGEvaluationConfig(
            embedding_model="all-MiniLM-L6-v2",  # Faster model
            ragas_weights={
                "context_precision": 0.3,
                "context_relevancy": 0.3,
                "context_recall": 0.0,  # Skip slower metrics
                "context_entity_recall": 0.0,
                "answer_relevancy": 0.2,
                "answer_similarity": 0.0,
                "answer_correctness": 0.0,
                "faithfulness": 0.2,
            },
        )
    else:  # balanced
        return RAGEvaluationConfig()


# Export all classes and functions
__all__ = [
    "AnswerCorrectnessScorer",
    "AnswerRelevancyScorer",
    "AnswerSimilarityScorer",
    "ContextEntityRecallScorer",
    "ContextPrecisionScorer",
    "ContextRecallScorer",
    "ContextRelevancyScorer",
    "EnhancedFaithfulnessScorer",
    "EnhancedRAGASScorer",
    "RAGEvaluationConfig",
    "RAGEvaluationSuite",
    "RAGTriadScorer",
    "create_rag_scorer",
    "get_default_rag_config",
    "get_optimized_rag_config",
]
