"""
Conversational AI metrics for NovaEval.

This module implements comprehensive metrics for evaluating conversational AI systems including:
- Knowledge Retention with sophisticated knowledge extraction and tracking
- Conversation Completeness with intention analysis
- Conversation Relevancy with sliding window context
- Role Adherence with detailed role analysis
- Comprehensive conversation-level metrics with outcome-based evaluation

Based on best practices from DeepEval and research in conversational AI evaluation.
"""

import asyncio
import re
import threading
from collections.abc import Coroutine
from typing import Any, Optional, TypeVar

from pydantic import BaseModel, Field

from novaeval.models.base import BaseModel as LLMModel
from novaeval.scorers.base import BaseScorer, ScoreResult

T = TypeVar("T")


def _run_async_in_sync_context(coro: Coroutine[Any, Any, T]) -> T:
    """
    Helper function to run async code from sync context.

    Handles both cases:
    - When called from outside an event loop (uses asyncio.run)
    - When called from within an existing event loop (uses loop.run_until_complete in thread)
    """
    try:
        # Check if we're already in a running event loop
        asyncio.get_running_loop()
    except RuntimeError:
        # No running event loop, we can use asyncio.run
        return asyncio.run(coro)

    # We're in a running loop, we need to run in a separate thread
    # Use a sentinel object to track completion status
    _SENTINEL = object()
    result: Any = _SENTINEL
    exception: Optional[BaseException] = None

    def run_in_thread() -> None:
        nonlocal result, exception
        try:
            # Create a new event loop for this thread
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                result = new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
        except (
            BaseException
        ) as e:  # Catch ALL exceptions, including SystemExit, KeyboardInterrupt
            exception = e

    thread = threading.Thread(target=run_in_thread)
    thread.start()
    thread.join()

    if exception:
        raise exception

    # Ensure thread completed successfully and we have a valid result
    if result is _SENTINEL:
        raise RuntimeError(
            "Thread completed without setting result or raising exception"
        )

    return result  # type: ignore[return-value]  # We've verified result is not sentinel


class ConversationTurn(BaseModel):
    """Represents a single turn in a conversation."""

    speaker: str = Field(description="Speaker identifier (user, assistant, system)")
    message: str = Field(description="The message content")
    timestamp: Optional[str] = Field(default=None, description="Optional timestamp")
    metadata: Optional[dict[str, Any]] = Field(
        default_factory=dict, description="Additional metadata"
    )


class Conversation(BaseModel):
    """Represents a complete conversation with metadata."""

    turns: list[ConversationTurn] = Field(description="List of conversation turns")
    context: Optional[str] = Field(
        default=None, description="Overall conversation context or system role"
    )
    topic: Optional[str] = Field(default=None, description="Conversation topic")
    metadata: Optional[dict[str, Any]] = Field(
        default_factory=dict, description="Additional metadata"
    )


class KnowledgeItem(BaseModel):
    """Represents a piece of knowledge extracted from conversation."""

    content: str = Field(description="The knowledge content")
    turn_index: int = Field(description="Turn where this knowledge was introduced")
    speaker: str = Field(description="Who provided this knowledge")
    confidence: float = Field(description="Confidence in extraction (0-1)")


class KnowledgeRetentionScorer(BaseScorer):
    """
    Evaluates knowledge retention in conversations.

    Uses sophisticated knowledge extraction and tracking to determine if the LLM
    retains information provided by users throughout the conversation.
    """

    def __init__(self, model: LLMModel, window_size: int = 10):
        super().__init__(
            name="Knowledge Retention",
            description="Evaluates knowledge retention in conversations by tracking information recall and consistency across dialogue turns",
        )
        self.model = model
        self.window_size = window_size

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        """
        Evaluate knowledge retention in conversation.

        Args:
            input_text: The input/context for the conversation
            output_text: The model's response to evaluate
            expected_output: Expected response (optional)
            context: Additional context including conversation history

        Returns:
            ScoreResult with score, pass/fail status, and reasoning
        """
        # Input validation with proper type checking
        if not isinstance(output_text, str):
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="Invalid input: output_text must be a string",
                metadata={
                    "error": "type_error",
                    "input_type": type(output_text).__name__,
                },
            )

        if expected_output is not None and not isinstance(expected_output, str):
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="Invalid input: expected_output must be a string or None",
                metadata={
                    "error": "type_error",
                    "expected_type": type(expected_output).__name__,
                },
            )

        # Additional validation for empty strings (safe now that we've checked types)
        if not output_text or not output_text.strip():
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="Empty or whitespace-only output text",
                metadata={"error": "empty_output"},
            )

        if expected_output is not None and (
            not expected_output or not expected_output.strip()
        ):
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="Empty or whitespace-only expected output",
                metadata={"error": "empty_expected"},
            )

        try:
            score = await self._evaluate_knowledge_retention_async(
                output_text, expected_output or "", context
            )
            self._track_score(score)

            passed = score >= 0.7  # Default threshold
            reasoning = self._generate_reasoning(score, output_text, context)

            return ScoreResult(
                score=score,
                passed=passed,
                reasoning=reasoning,
                metadata={
                    "scorer": "knowledge_retention",
                    "model": (
                        self.model.name if hasattr(self.model, "name") else "unknown"
                    ),
                    "window_size": self.window_size,
                },
            )
        except Exception as e:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"Evaluation failed: {e!s}",
                metadata={"error": "evaluation_error", "exception": str(e)},
            )

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> float:
        """
        Legacy synchronous score method for backward compatibility.

        Note: This method is deprecated. Use evaluate() for new code.
        """
        # Input validation with proper type checking
        if not isinstance(prediction, str) or not isinstance(ground_truth, str):
            return 0.0

        # Safe to call strip() now
        if (
            not prediction
            or not prediction.strip()
            or not ground_truth
            or not ground_truth.strip()
        ):
            return 0.0

        try:
            result = _run_async_in_sync_context(
                self.evaluate(
                    input_text="",  # Not used in knowledge retention
                    output_text=prediction,
                    expected_output=ground_truth,
                    context=context,
                )
            )
            return result.score
        except Exception:
            return 0.0

    def _generate_reasoning(
        self, score: float, output_text: str, context: Optional[dict[str, Any]]
    ) -> str:
        """Generate reasoning for the knowledge retention score."""
        if score >= 0.9:
            return f"Excellent knowledge retention (score: {score:.2f}). The response demonstrates strong recall of previously shared information."
        elif score >= 0.7:
            return f"Good knowledge retention (score: {score:.2f}). The response shows adequate recall with minor gaps."
        elif score >= 0.5:
            return f"Moderate knowledge retention (score: {score:.2f}). The response shows some retention but with noticeable gaps."
        elif score >= 0.3:
            return f"Poor knowledge retention (score: {score:.2f}). The response asks for information that was previously provided."
        else:
            return f"Very poor knowledge retention (score: {score:.2f}). The response shows significant failure to retain previously shared information."

    def _evaluate_knowledge_retention_sync(
        self, prediction: str, ground_truth: str, context: Optional[dict[str, Any]]
    ) -> float:
        """Synchronous wrapper for knowledge retention evaluation."""
        try:
            return _run_async_in_sync_context(
                self._evaluate_knowledge_retention_async(
                    prediction, ground_truth, context
                )
            )
        except Exception:
            return 0.0

    async def _evaluate_knowledge_retention_async(
        self, prediction: str, ground_truth: str, context: Optional[dict[str, Any]]
    ) -> float:
        """Async evaluation of knowledge retention."""
        if not context or "conversation" not in context:
            return self._simple_retention_score(prediction, ground_truth)

        conversation = context["conversation"]
        if not isinstance(conversation, Conversation):
            return self._simple_retention_score(prediction, ground_truth)

        # Extract knowledge from conversation history
        knowledge_items = await self._extract_conversation_knowledge(conversation)

        # Check if the current response asks for already provided information
        violations = await self._detect_retention_violations(
            prediction, knowledge_items
        )

        # Calculate retention score
        if not knowledge_items:
            return 1.0  # No knowledge to retain, perfect score

        retention_rate = max(0.0, 1.0 - (len(violations) / len(knowledge_items)))
        return retention_rate

    async def _extract_conversation_knowledge(
        self, conversation: Conversation
    ) -> list[KnowledgeItem]:
        """Extract knowledge items from recent conversation history within the sliding window."""
        knowledge_items = []

        # Apply sliding window to limit the conversation turns processed
        current_turn_index = len(conversation.turns) - 1
        window_start = max(0, current_turn_index - self.window_size)
        relevant_turns = conversation.turns[window_start:]

        for i, turn in enumerate(relevant_turns, start=window_start):
            if turn.speaker == "user":  # Only extract knowledge from user messages
                knowledge_prompt = f"""Extract factual information and personal details from this user message:

User message: "{turn.message}"

Identify specific facts, preferences, personal information, or contextual details that should be remembered.
Format each piece of knowledge as a separate item.

Output as a numbered list:
1. [Knowledge item 1]
2. [Knowledge item 2]
...

If no significant knowledge is present, respond with "None"."""

                try:
                    if hasattr(self.model, "generate") and asyncio.iscoroutinefunction(
                        self.model.generate
                    ):
                        response = await self.model.generate(knowledge_prompt)
                    else:
                        response = self.model.generate(knowledge_prompt)
                    extracted_knowledge = self._parse_knowledge_items(
                        response, i, turn.speaker
                    )
                    knowledge_items.extend(extracted_knowledge)
                except Exception:
                    continue

        return knowledge_items

    def _parse_knowledge_items(
        self, response: str, turn_index: int, speaker: str
    ) -> list[KnowledgeItem]:
        """Parse knowledge items from LLM response."""
        items: list[KnowledgeItem] = []

        if response.strip().lower() == "none":
            return items

        # Extract numbered list items
        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            # Match numbered items like "1. Something" or "1) Something"
            match = re.match(r"^\d+[\.\)]\s*(.+)$", line)
            if match:
                content = match.group(1).strip()
                if content and len(content) > 5:  # Filter out very short items
                    items.append(
                        KnowledgeItem(
                            content=content,
                            turn_index=turn_index,
                            speaker=speaker,
                            confidence=0.8,  # Default confidence
                        )
                    )

        return items

    async def _detect_retention_violations(
        self, response: str, knowledge_items: list[KnowledgeItem]
    ) -> list[str]:
        """Detect if the response asks for information already provided."""
        if not knowledge_items:
            return []

        knowledge_summary = "\n".join([f"- {item.content}" for item in knowledge_items])

        violation_prompt = f"""Analyze if this assistant response asks for information that was already provided.

Previously provided information:
{knowledge_summary}

Assistant response: "{response}"

Does the assistant ask for any information that was already provided? Look for:
- Direct questions about already shared facts
- Requests for clarification on provided details
- Asking to repeat information

Respond with "YES" if there are violations, "NO" if there are none.
If YES, list the specific violations:"""

        try:
            if hasattr(self.model, "generate") and asyncio.iscoroutinefunction(
                self.model.generate
            ):
                violation_response = await self.model.generate(violation_prompt)
            else:
                violation_response = self.model.generate(violation_prompt)
            return self._parse_violations(violation_response)
        except Exception:
            return []

    def _parse_violations(self, response: str) -> list[str]:
        """Parse violations from LLM response."""
        if response.strip().upper().startswith("NO"):
            return []

        violations = []
        lines = response.strip().split("\n")[1:]  # Skip first line with YES/NO
        for line in lines:
            line = line.strip()
            if line and not line.upper().startswith("NO"):
                violations.append(line)

        return violations

    def _simple_retention_score(self, prediction: str, ground_truth: str) -> float:
        """Fallback simple retention scoring when conversation context unavailable."""
        # Simple heuristic: check if prediction asks questions about basic info
        question_patterns = [
            r"\bwhat is your name\b",
            r"\bwho are you\b",
            r"\bwhere are you from\b",
            r"\bhow old are you\b",
            r"\bwhat do you do\b",
        ]

        prediction_lower = prediction.lower()
        for pattern in question_patterns:
            if re.search(pattern, prediction_lower):
                return 0.3  # Low score for asking basic questions

        return 0.7  # Default decent score


class ConversationRelevancyScorer(BaseScorer):
    """
    Evaluates conversation relevancy using sliding window approach.

    Assesses whether responses are relevant to recent conversation context,
    using a sliding window to consider appropriate conversation history.
    """

    def __init__(self, model: LLMModel, window_size: int = 5):
        super().__init__(
            name="Conversation Relevancy",
            description="Measures how relevant and contextually appropriate responses are within the conversation flow",
        )
        self.model = model
        self.window_size = window_size

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        """
        Evaluate conversation relevancy.

        Args:
            input_text: The input/context for the conversation
            output_text: The model's response to evaluate
            expected_output: Expected response (optional)
            context: Additional context including conversation history

        Returns:
            ScoreResult with score, pass/fail status, and reasoning
        """
        # Input validation with proper type checking
        if not isinstance(output_text, str):
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="Invalid input: output_text must be a string",
                metadata={
                    "error": "type_error",
                    "input_type": type(output_text).__name__,
                },
            )

        # Additional validation for empty strings (safe now that we've checked types)
        if not output_text or not output_text.strip():
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="Empty or whitespace-only output text",
                metadata={"error": "empty_output"},
            )

        try:
            score = await self._evaluate_relevancy_async(
                output_text, expected_output or "", context
            )
            self._track_score(score)

            passed = score >= 0.7  # Default threshold
            reasoning = self._generate_relevancy_reasoning(score, output_text, context)

            return ScoreResult(
                score=score,
                passed=passed,
                reasoning=reasoning,
                metadata={
                    "scorer": "conversation_relevancy",
                    "model": (
                        self.model.name if hasattr(self.model, "name") else "unknown"
                    ),
                    "window_size": self.window_size,
                },
            )
        except Exception as e:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"Evaluation failed: {e!s}",
                metadata={"error": "evaluation_error", "exception": str(e)},
            )

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> float:
        """
        Legacy synchronous score method for backward compatibility.

        Note: This method is deprecated. Use evaluate() for new code.
        """
        # Input validation with proper type checking
        if not isinstance(prediction, str) or not isinstance(ground_truth, str):
            return 0.0

        # Safe to call strip() now
        if (
            not prediction
            or not prediction.strip()
            or not ground_truth
            or not ground_truth.strip()
        ):
            return 0.0

        try:
            result = _run_async_in_sync_context(
                self.evaluate(
                    input_text="",
                    output_text=prediction,
                    expected_output=ground_truth,
                    context=context,
                )
            )
            return result.score
        except Exception:
            return 0.0

    def _generate_relevancy_reasoning(
        self, score: float, output_text: str, context: Optional[dict[str, Any]]
    ) -> str:
        """Generate reasoning for the conversation relevancy score."""
        if score >= 0.9:
            return f"Excellent relevancy (score: {score:.2f}). The response is highly relevant to the conversation context."
        elif score >= 0.7:
            return f"Good relevancy (score: {score:.2f}). The response is mostly relevant with minor deviations."
        elif score >= 0.5:
            return f"Moderate relevancy (score: {score:.2f}). The response has some relevance but could be more focused."
        elif score >= 0.3:
            return f"Poor relevancy (score: {score:.2f}). The response is loosely relevant to the conversation."
        else:
            return f"Very poor relevancy (score: {score:.2f}). The response is off-topic or ignores the conversation context."

    def _evaluate_relevancy_sync(
        self, prediction: str, ground_truth: str, context: Optional[dict[str, Any]]
    ) -> float:
        """Synchronous wrapper for relevancy evaluation."""
        try:
            return _run_async_in_sync_context(
                self._evaluate_relevancy_async(prediction, ground_truth, context)
            )
        except Exception:
            return 0.0

    async def _evaluate_relevancy_async(
        self, prediction: str, ground_truth: str, context: Optional[dict[str, Any]]
    ) -> float:
        """Async evaluation of conversation relevancy with sliding window."""
        if not context or "conversation" not in context:
            return self._simple_relevancy_score(prediction, ground_truth)

        conversation = context["conversation"]
        if not isinstance(conversation, Conversation):
            return self._simple_relevancy_score(prediction, ground_truth)

        # Get relevant context window
        current_turn_index = len(conversation.turns) - 1
        window_start = max(0, current_turn_index - self.window_size)
        relevant_turns = conversation.turns[window_start:current_turn_index]

        if not relevant_turns:
            return self._simple_relevancy_score(prediction, ground_truth)

        # Build context summary
        context_summary = self._build_context_summary(relevant_turns)

        relevancy_prompt = f"""Evaluate the relevancy of an assistant's response to the recent conversation context.

Recent conversation context:
{context_summary}

Assistant response: "{prediction}"

Rate the relevancy on a scale of 1-5:
5 = Highly relevant, directly addresses the conversation flow
4 = Mostly relevant with minor tangents
3 = Somewhat relevant but could be more focused
2 = Loosely relevant, partially addresses context
1 = Not relevant, off-topic or ignoring context

Consider:
- Does the response address the most recent user input?
- Does it maintain conversation flow and coherence?
- Is it appropriate given the conversation history?

Respond with just the number (1-5):"""

        try:
            if hasattr(self.model, "generate") and asyncio.iscoroutinefunction(
                self.model.generate
            ):
                response = await self.model.generate(relevancy_prompt)
            else:
                response = self.model.generate(relevancy_prompt)
            score = self._parse_relevancy_score(response)
            return score / 5.0  # Normalize to 0-1
        except Exception:
            return 0.0

    def _build_context_summary(self, turns: list[ConversationTurn]) -> str:
        """Build a summary of conversation turns for context."""
        summary_parts = []
        for _i, turn in enumerate(turns):
            summary_parts.append(f"{turn.speaker}: {turn.message}")
        return "\n".join(summary_parts)

    def _parse_relevancy_score(self, response: str) -> float:
        """Parse relevancy score from LLM response."""
        # Look for numbers 1-5 in the response
        numbers = re.findall(r"\b([1-5])\b", response.strip())
        if numbers:
            return float(numbers[0])
        return 3.0  # Default to middle score

    def _simple_relevancy_score(self, prediction: str, ground_truth: str) -> float:
        """Fallback simple relevancy scoring when conversation context unavailable."""
        # Simple word overlap heuristic
        pred_words = set(prediction.lower().split())
        truth_words = set(ground_truth.lower().split())

        if not pred_words or not truth_words:
            return 0.0

        overlap = len(pred_words.intersection(truth_words))
        union = len(pred_words.union(truth_words))

        if union == 0:
            return 0.0

        return overlap / union


class ConversationCompletenessScorer(BaseScorer):
    """
    Evaluates conversation completeness by analyzing user intentions and fulfillment.

    Determines whether user requests and intentions throughout the conversation
    have been adequately addressed and fulfilled.
    """

    def __init__(self, model: LLMModel):
        super().__init__(
            name="Conversation Completeness",
            description="Assesses whether responses provide comprehensive coverage of topics and address all aspects of user queries",
        )
        self.model = model

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        """
        Evaluate conversation completeness.

        Args:
            input_text: The input/context for the conversation
            output_text: The model's response to evaluate
            expected_output: Expected response (optional)
            context: Additional context including conversation history

        Returns:
            ScoreResult with score, pass/fail status, and reasoning
        """
        # Input validation with proper type checking
        if not isinstance(output_text, str):
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="Invalid input: output_text must be a string",
                metadata={
                    "error": "type_error",
                    "input_type": type(output_text).__name__,
                },
            )

        # Additional validation for empty strings (safe now that we've checked types)
        if not output_text or not output_text.strip():
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="Empty or whitespace-only output text",
                metadata={"error": "empty_output"},
            )

        try:
            score = await self._evaluate_completeness_async(
                output_text, expected_output or "", context
            )
            self._track_score(score)

            passed = score >= 0.7  # Default threshold
            reasoning = self._generate_completeness_reasoning(
                score, output_text, context
            )

            return ScoreResult(
                score=score,
                passed=passed,
                reasoning=reasoning,
                metadata={
                    "scorer": "conversation_completeness",
                    "model": (
                        self.model.name if hasattr(self.model, "name") else "unknown"
                    ),
                },
            )
        except Exception as e:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"Evaluation failed: {e!s}",
                metadata={"error": "evaluation_error", "exception": str(e)},
            )

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> float:
        """
        Legacy synchronous score method for backward compatibility.

        Note: This method is deprecated. Use evaluate() for new code.
        """
        # Input validation with proper type checking
        if not isinstance(prediction, str) or not isinstance(ground_truth, str):
            return 0.0

        # Safe to call strip() now
        if (
            not prediction
            or not prediction.strip()
            or not ground_truth
            or not ground_truth.strip()
        ):
            return 0.0

        try:
            result = _run_async_in_sync_context(
                self.evaluate(
                    input_text="",
                    output_text=prediction,
                    expected_output=ground_truth,
                    context=context,
                )
            )
            return result.score
        except Exception:
            return 0.0

    def _generate_completeness_reasoning(
        self, score: float, output_text: str, context: Optional[dict[str, Any]]
    ) -> str:
        """Generate reasoning for the conversation completeness score."""
        if score >= 0.9:
            return f"Excellent completeness (score: {score:.2f}). All user intentions and requests have been thoroughly addressed."
        elif score >= 0.7:
            return f"Good completeness (score: {score:.2f}). Most user intentions have been addressed with minor gaps."
        elif score >= 0.5:
            return f"Moderate completeness (score: {score:.2f}). Some user intentions addressed but significant gaps remain."
        elif score >= 0.3:
            return f"Poor completeness (score: {score:.2f}). User intentions minimally addressed or partially ignored."
        else:
            return f"Very poor completeness (score: {score:.2f}). User intentions largely unaddressed or ignored."

    def _evaluate_completeness_sync(
        self, prediction: str, ground_truth: str, context: Optional[dict[str, Any]]
    ) -> float:
        """Synchronous wrapper for completeness evaluation."""
        try:
            return _run_async_in_sync_context(
                self._evaluate_completeness_async(prediction, ground_truth, context)
            )
        except Exception:
            return 0.0

    async def _evaluate_completeness_async(
        self, prediction: str, ground_truth: str, context: Optional[dict[str, Any]]
    ) -> float:
        """Async evaluation of conversation completeness."""
        if not context or "conversation" not in context:
            return self._simple_completeness_score(prediction, ground_truth)

        conversation = context["conversation"]
        if not isinstance(conversation, Conversation):
            return self._simple_completeness_score(prediction, ground_truth)

        # Extract user intentions from conversation
        intentions = await self._extract_user_intentions(conversation)

        if not intentions:
            return 1.0  # No specific intentions to fulfill

        # Evaluate fulfillment of each intention
        fulfillment_scores = []
        for intention in intentions:
            fulfillment = await self._evaluate_intention_fulfillment(
                intention, conversation
            )
            fulfillment_scores.append(fulfillment)

        # Calculate overall completeness
        if fulfillment_scores:
            return sum(fulfillment_scores) / len(fulfillment_scores)

        return 0.5  # Default score when unclear

    async def _extract_user_intentions(self, conversation: Conversation) -> list[str]:
        """Extract user intentions from conversation."""
        user_messages = [
            turn.message for turn in conversation.turns if turn.speaker == "user"
        ]

        if not user_messages:
            return []

        combined_messages = "\n".join(user_messages)

        intention_prompt = f"""Analyze the user messages and identify their main intentions or goals.

User messages:
{combined_messages}

What does the user want to achieve? List the main intentions as:
1. [Intention 1]
2. [Intention 2]
...

If no clear intentions are present, respond with "None"."""

        try:
            if hasattr(self.model, "generate") and asyncio.iscoroutinefunction(
                self.model.generate
            ):
                response = await self.model.generate(intention_prompt)
            else:
                response = self.model.generate(intention_prompt)
            return self._parse_intentions(response)
        except Exception:
            return []

    def _parse_intentions(self, response: str) -> list[str]:
        """Parse intentions from LLM response."""
        intentions: list[str] = []

        if response.strip().lower() == "none":
            return intentions

        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            # Match numbered items
            match = re.match(r"^\d+[\.\)]\s*(.+)$", line)
            if match:
                intention = match.group(1).strip()
                if intention:
                    intentions.append(intention)

        return intentions

    async def _evaluate_intention_fulfillment(
        self, intention: str, conversation: Conversation
    ) -> float:
        """Evaluate how well an intention was fulfilled."""
        assistant_responses = [
            turn.message for turn in conversation.turns if turn.speaker == "assistant"
        ]

        if not assistant_responses:
            return 0.0

        combined_responses = "\n".join(assistant_responses)

        fulfillment_prompt = f"""Evaluate how well the assistant fulfilled this user intention:

User intention: "{intention}"

Assistant responses:
{combined_responses}

Rate the fulfillment on a scale of 1-5:
5 = Completely fulfilled, user goal fully achieved
4 = Mostly fulfilled with minor gaps
3 = Partially fulfilled, significant progress made
2 = Minimally fulfilled, some attempt made
1 = Not fulfilled, intention ignored or inadequately addressed

Respond with just the number (1-5):"""

        try:
            if hasattr(self.model, "generate") and asyncio.iscoroutinefunction(
                self.model.generate
            ):
                response = await self.model.generate(fulfillment_prompt)
            else:
                response = self.model.generate(fulfillment_prompt)
            score = self._parse_fulfillment_score(response)
            return score / 5.0  # Normalize to 0-1
        except Exception:
            return 0.0

    def _parse_fulfillment_score(self, response: str) -> float:
        """Parse fulfillment score from LLM response."""
        numbers = re.findall(r"\b([1-5])\b", response.strip())
        if numbers:
            return float(numbers[0])
        return 3.0  # Default to middle score

    def _simple_completeness_score(self, prediction: str, ground_truth: str) -> float:
        """Fallback simple completeness scoring."""
        # Simple heuristic based on response length and content
        if len(prediction.strip()) < 10:
            return 0.2  # Very short responses likely incomplete

        if "sorry" in prediction.lower() or "can't" in prediction.lower():
            return 0.4  # Apologetic or refusing responses partially complete

        return 0.7  # Default decent score for substantial responses


class RoleAdherenceScorer(BaseScorer):
    """
    Evaluates role adherence in conversations.

    Assesses whether the assistant maintains its assigned role throughout
    the conversation and behaves consistently with role expectations.
    """

    def __init__(self, model: LLMModel, expected_role: Optional[str] = None):
        super().__init__(
            name="Role Adherence",
            description="Evaluates how well responses maintain consistency with assigned persona, role, or character throughout conversations",
        )
        self.model = model
        self.expected_role = expected_role

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        """
        Evaluate role adherence.

        Args:
            input_text: The input/context for the conversation
            output_text: The model's response to evaluate
            expected_output: Expected response (optional)
            context: Additional context including conversation history

        Returns:
            ScoreResult with score, pass/fail status, and reasoning
        """
        # Input validation with proper type checking
        if not isinstance(output_text, str):
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="Invalid input: output_text must be a string",
                metadata={
                    "error": "type_error",
                    "input_type": type(output_text).__name__,
                },
            )

        # Additional validation for empty strings (safe now that we've checked types)
        if not output_text or not output_text.strip():
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="Empty or whitespace-only output text",
                metadata={"error": "empty_output"},
            )

        try:
            score = await self._evaluate_role_adherence_async(output_text, context)
            self._track_score(score)

            passed = score >= 0.7  # Default threshold
            reasoning = self._generate_role_reasoning(score, output_text, context)

            return ScoreResult(
                score=score,
                passed=passed,
                reasoning=reasoning,
                metadata={
                    "scorer": "role_adherence",
                    "model": (
                        self.model.name if hasattr(self.model, "name") else "unknown"
                    ),
                    "expected_role": self.expected_role,
                },
            )
        except Exception as e:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"Evaluation failed: {e!s}",
                metadata={"error": "evaluation_error", "exception": str(e)},
            )

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> float:
        """
        Legacy synchronous score method for backward compatibility.

        Note: This method is deprecated. Use evaluate() for new code.
        """
        # Input validation with proper type checking
        if not isinstance(prediction, str) or not isinstance(ground_truth, str):
            return 0.0

        # Safe to call strip() now
        if (
            not prediction
            or not prediction.strip()
            or not ground_truth
            or not ground_truth.strip()
        ):
            return 0.0

        try:
            result = _run_async_in_sync_context(
                self.evaluate(
                    input_text="",
                    output_text=prediction,
                    expected_output=ground_truth,
                    context=context,
                )
            )
            return result.score
        except Exception:
            return 0.0

    def _generate_role_reasoning(
        self, score: float, output_text: str, context: Optional[dict[str, Any]]
    ) -> str:
        """Generate reasoning for the role adherence score."""
        role_name = self.expected_role or "assigned role"
        if score >= 0.9:
            return f"Excellent role adherence (score: {score:.2f}). The response perfectly maintains the {role_name} throughout."
        elif score >= 0.7:
            return f"Good role adherence (score: {score:.2f}). The response mostly maintains the {role_name} with minor deviations."
        elif score >= 0.5:
            return f"Moderate role adherence (score: {score:.2f}). The response somewhat maintains the {role_name} but has noticeable inconsistencies."
        elif score >= 0.3:
            return f"Poor role adherence (score: {score:.2f}). The response poorly maintains the {role_name} with significant role breaks."
        else:
            return f"Very poor role adherence (score: {score:.2f}). The response completely abandons the {role_name}."

    def _evaluate_role_adherence_sync(
        self, prediction: str, ground_truth: str, context: Optional[dict[str, Any]]
    ) -> float:
        """Synchronous wrapper for role adherence evaluation."""
        try:
            return _run_async_in_sync_context(
                self._evaluate_role_adherence_async(prediction, context)
            )
        except Exception:
            return 0.0

    async def _evaluate_role_adherence_async(
        self, prediction: str, context: Optional[dict[str, Any]]
    ) -> float:
        """Async evaluation of role adherence."""
        # Determine expected role
        role = self.expected_role
        if not role and context and "conversation" in context:
            conversation = context["conversation"]
            if isinstance(conversation, Conversation) and conversation.context:
                role = conversation.context

        if not role:
            return 1.0  # No role defined, perfect adherence

        role_prompt = f"""Evaluate how well this response adheres to the expected role:

Expected Role: {role}
Response: "{prediction}"

Rate role adherence from 1-5:
5 = Perfect role adherence, stays completely in character
4 = Good adherence with minor deviations
3 = Adequate adherence, mostly appropriate
2 = Poor adherence, significant role breaks
1 = No adherence, completely out of character

Consider:
- Does the response match the expected personality/expertise?
- Is the tone and language appropriate for the role?
- Does it maintain role consistency?

Respond with just the number (1-5):"""

        try:
            if hasattr(self.model, "generate") and asyncio.iscoroutinefunction(
                self.model.generate
            ):
                response = await self.model.generate(role_prompt)
            else:
                response = self.model.generate(role_prompt)
            score = self._parse_role_score(response)
            return score / 5.0  # Normalize to 0-1
        except Exception:
            return 0.0

    def _parse_role_score(self, response: str) -> float:
        """Parse role adherence score from LLM response."""
        numbers = re.findall(r"\b([1-5])\b", response.strip())
        if numbers:
            return float(numbers[0])
        return 3.0  # Default to middle score


class ConversationalMetricsScorer(BaseScorer):
    """
    Comprehensive conversational metrics scorer.

    Combines multiple conversational metrics to provide a holistic evaluation
    of conversation quality, including knowledge retention, relevancy,
    completeness, and role adherence.
    """

    def __init__(
        self,
        model: LLMModel,
        include_knowledge_retention: bool = True,
        include_relevancy: bool = True,
        include_completeness: bool = True,
        include_role_adherence: bool = True,
        window_size: int = 5,
        expected_role: Optional[str] = None,
    ):
        super().__init__(
            name="Conversational Metrics",
            description="Comprehensive conversational evaluation combining multiple metrics: knowledge retention, relevancy, completeness, and role adherence",
        )
        self.model = model
        self.include_knowledge_retention = include_knowledge_retention
        self.include_relevancy = include_relevancy
        self.include_completeness = include_completeness
        self.include_role_adherence = include_role_adherence

        # Initialize individual scorers
        if include_knowledge_retention:
            self.knowledge_scorer = KnowledgeRetentionScorer(model, window_size)
        if include_relevancy:
            self.relevancy_scorer = ConversationRelevancyScorer(model, window_size)
        if include_completeness:
            self.completeness_scorer = ConversationCompletenessScorer(model)
        if include_role_adherence:
            self.role_scorer = RoleAdherenceScorer(model, expected_role)

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        """
        Evaluate using multiple conversational metrics.

        Args:
            input_text: The input/context for the conversation
            output_text: The model's response to evaluate
            expected_output: Expected response (optional)
            context: Additional context including conversation history

        Returns:
            ScoreResult with combined scores and detailed reasoning
        """
        # Input validation with proper type checking
        if not isinstance(output_text, str):
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="Invalid input: output_text must be a string",
                metadata={
                    "error": "type_error",
                    "input_type": type(output_text).__name__,
                },
            )

        # Additional validation for empty strings (safe now that we've checked types)
        if not output_text or not output_text.strip():
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="Empty or whitespace-only output text",
                metadata={"error": "empty_output"},
            )

        scores = {}
        results = {}

        try:
            # Evaluate each enabled metric
            if self.include_knowledge_retention:
                result = await self.knowledge_scorer.evaluate(
                    input_text, output_text, expected_output, context
                )
                scores["knowledge_retention"] = result.score
                results["knowledge_retention"] = result

            if self.include_relevancy:
                result = await self.relevancy_scorer.evaluate(
                    input_text, output_text, expected_output, context
                )
                scores["relevancy"] = result.score
                results["relevancy"] = result

            if self.include_completeness:
                result = await self.completeness_scorer.evaluate(
                    input_text, output_text, expected_output, context
                )
                scores["completeness"] = result.score
                results["completeness"] = result

            if self.include_role_adherence:
                result = await self.role_scorer.evaluate(
                    input_text, output_text, expected_output, context
                )
                scores["role_adherence"] = result.score
                results["role_adherence"] = result

            # Calculate overall score as average
            overall_score = sum(scores.values()) / len(scores) if scores else 0.0
            passed = overall_score >= 0.7  # Default threshold

            # Generate combined reasoning
            reasoning = self._generate_combined_reasoning(scores, results)

            self._track_score(overall_score)

            return ScoreResult(
                score=overall_score,
                passed=passed,
                reasoning=reasoning,
                metadata={
                    "scorer": "conversational_metrics",
                    "individual_scores": scores,
                    "enabled_metrics": {
                        "knowledge_retention": self.include_knowledge_retention,
                        "relevancy": self.include_relevancy,
                        "completeness": self.include_completeness,
                        "role_adherence": self.include_role_adherence,
                    },
                },
            )

        except Exception as e:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"Evaluation failed: {e!s}",
                metadata={"error": "evaluation_error", "exception": str(e)},
            )

    def score(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, float]:
        """
        Legacy synchronous score method for backward compatibility.

        Note: This method is deprecated. Use evaluate() for new code.

        Returns:
            Dictionary with individual metric scores and overall score
        """
        # Input validation with proper type checking
        if not isinstance(prediction, str) or not isinstance(ground_truth, str):
            return {"overall": 0.0}

        # Safe to call strip() now
        if (
            not prediction
            or not prediction.strip()
            or not ground_truth
            or not ground_truth.strip()
        ):
            return {"overall": 0.0}

        try:
            result = _run_async_in_sync_context(
                self.evaluate(
                    input_text="",
                    output_text=prediction,
                    expected_output=ground_truth,
                    context=context,
                )
            )
            # Extract individual scores from metadata
            individual_scores = result.metadata.get("individual_scores", {})
            individual_scores["overall"] = result.score
            return individual_scores
        except Exception:
            return {"overall": 0.0}

    def _generate_combined_reasoning(
        self, scores: dict[str, float], results: dict[str, ScoreResult]
    ) -> str:
        """Generate combined reasoning from all metric results."""
        overall_score = sum(scores.values()) / len(scores) if scores else 0.0

        reasoning_parts = [
            f"Combined conversational metrics score: {overall_score:.2f}"
        ]
        reasoning_parts.append("\nIndividual metric breakdown:")

        for metric, score in scores.items():
            metric_name = metric.replace("_", " ").title()
            reasoning_parts.append(f"- {metric_name}: {score:.2f}")

        # Add any specific insights from individual results
        if scores:
            if overall_score >= 0.9:
                reasoning_parts.append(
                    "\nExcellent overall conversational performance across all metrics."
                )
            elif overall_score >= 0.7:
                reasoning_parts.append(
                    "\nGood overall conversational performance with room for minor improvements."
                )
            elif overall_score >= 0.5:
                reasoning_parts.append(
                    "\nModerate conversational performance with several areas needing improvement."
                )
            else:
                reasoning_parts.append(
                    "\nPoor conversational performance requiring significant improvements."
                )

        return "\n".join(reasoning_parts)
