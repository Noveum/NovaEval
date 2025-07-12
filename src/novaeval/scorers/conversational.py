"""
Conversational AI metrics for NovaEval.

This module implements metrics for evaluating conversational AI systems including:
- Knowledge Retention
- Conversation Completeness
- Conversation Relevancy
- Role Adherence
- Turn-level and conversation-level metrics
"""

import asyncio
from typing import Any, Optional

from pydantic import BaseModel, Field

from novaeval.models.base import BaseModel as LLMModel
from novaeval.scorers.base import BaseScorer, ScoreResult


class ConversationTurn(BaseModel):
    """Represents a single turn in a conversation."""

    speaker: str = Field(description="Speaker identifier (user, assistant, system)")
    message: str = Field(description="The message content")
    timestamp: Optional[str] = Field(default=None, description="Timestamp of the turn")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class Conversation(BaseModel):
    """Represents a complete conversation."""

    turns: list[ConversationTurn] = Field(description="List of conversation turns")
    context: Optional[str] = Field(
        default=None, description="Conversation context or system prompt"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Conversation metadata"
    )


class KnowledgeRetentionScorer(BaseScorer):
    """
    Evaluates how well the AI retains and uses information from earlier in the conversation.

    This metric measures whether the AI remembers and appropriately references
    information shared in previous turns.
    """

    def __init__(self, model: LLMModel, threshold: float = 0.7, **kwargs):
        super().__init__(threshold=threshold, **kwargs)
        self.model = model

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
        conversation: Optional[Conversation] = None,
        **kwargs,
    ) -> ScoreResult:
        """Evaluate knowledge retention in conversation."""

        if not conversation or len(conversation.turns) < 3:
            return ScoreResult(
                score=1.0,  # No retention needed for short conversations
                passed=True,
                reasoning="Conversation too short to evaluate knowledge retention",
                metadata={"turns": len(conversation.turns) if conversation else 0},
            )

        try:
            # Extract key information from earlier turns
            earlier_turns = conversation.turns[:-2]  # Exclude last 2 turns
            current_turn = conversation.turns[-1]

            # Build conversation history
            history = "\n".join(
                [f"{turn.speaker}: {turn.message}" for turn in earlier_turns]
            )

            # Extract key information that should be retained
            key_info_prompt = f"""
            Analyze the following conversation history and extract key information that should be remembered and potentially referenced in future responses.

            Conversation History:
            {history}

            Extract important facts, preferences, context, or information that an AI assistant should remember. List each piece of information separately.

            Format your response as:
            1. [Key information 1]
            2. [Key information 2]
            3. [Key information 3]
            ...
            """

            key_info_response = await self.model.generate(key_info_prompt)
            key_information = self._parse_information(key_info_response)

            if not key_information:
                return ScoreResult(
                    score=1.0,  # No key info to retain
                    passed=True,
                    reasoning="No key information found in conversation history to retain",
                    metadata={"key_information": []},
                )

            # Check if current response appropriately uses retained information
            retention_prompt = f"""
            Conversation History:
            {history}

            Current Response: {current_turn.message}

            Key Information from History:
            {chr(10).join(f'{i+1}. {info}' for i, info in enumerate(key_information))}

            Evaluate how well the current response demonstrates retention and appropriate use of the key information from the conversation history.

            For each piece of key information, determine if it was:
            - APPROPRIATELY_USED: Referenced or used when relevant
            - IGNORED: Should have been used but wasn't
            - NOT_RELEVANT: Not relevant to the current response

            Then provide an overall retention score from 1-5 where:
            1 = Poor retention, ignored relevant information
            2 = Below average retention
            3 = Average retention
            4 = Good retention, used most relevant information
            5 = Excellent retention, perfectly used all relevant information

            Format your response as:
            Information Analysis:
            1. [Info 1]: [APPROPRIATELY_USED/IGNORED/NOT_RELEVANT] - [explanation]
            2. [Info 2]: [APPROPRIATELY_USED/IGNORED/NOT_RELEVANT] - [explanation]
            ...

            Overall Retention Score: [1-5]
            Reasoning: [Brief explanation of the score]
            """

            retention_response = await self.model.generate(retention_prompt)
            retention_score, reasoning = self._parse_retention_score(retention_response)

            # Normalize score to 0-1 range
            normalized_score = (retention_score - 1) / 4

            detailed_reasoning = f"""
            Knowledge Retention Analysis:
            - Analyzed {len(key_information)} pieces of key information from conversation history
            - Retention evaluation score: {retention_score}/5
            - Normalized score: {normalized_score:.3f}

            Detailed Analysis:
            {reasoning}
            """

            return ScoreResult(
                score=normalized_score,
                passed=normalized_score >= self.threshold,
                reasoning=detailed_reasoning.strip(),
                metadata={
                    "key_information": key_information,
                    "retention_score": retention_score,
                    "conversation_length": len(conversation.turns),
                },
            )

        except Exception as e:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"Knowledge retention evaluation failed: {e!s}",
                metadata={"error": str(e)},
            )

    def _parse_information(self, response: str) -> list[str]:
        """Parse key information from LLM response."""
        information = []
        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip()
            if line and (
                line[0].isdigit() or line.startswith("-") or line.startswith("*")
            ):
                # Remove numbering and bullet points
                info = line
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
                    if info.startswith(prefix):
                        info = info[len(prefix) :].strip()
                        break

                if info:
                    information.append(info)

        return information

    def _parse_retention_score(self, response: str) -> tuple[float, str]:
        """Parse retention score and reasoning from LLM response."""
        import re

        # Look for "Overall Retention Score: X" pattern
        score_match = re.search(r"Overall Retention Score:\s*(\d+)", response)
        if score_match:
            score = float(score_match.group(1))
        else:
            # Look for standalone numbers 1-5
            numbers = re.findall(r"\b([1-5])\b", response)
            score = float(numbers[-1]) if numbers else 3.0

        # Extract reasoning
        reasoning_match = re.search(r"Reasoning:\s*(.+)", response, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else response

        return score, reasoning


class ConversationCompletenessScorer(BaseScorer):
    """
    Evaluates whether the conversation reaches a satisfactory conclusion.

    This metric measures if the AI adequately addresses the user's needs
    and brings the conversation to a natural completion.
    """

    def __init__(self, model: LLMModel, threshold: float = 0.7, **kwargs):
        super().__init__(threshold=threshold, **kwargs)
        self.model = model

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
        conversation: Optional[Conversation] = None,
        **kwargs,
    ) -> ScoreResult:
        """Evaluate conversation completeness."""

        if not conversation:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="No conversation provided for completeness evaluation",
                metadata={"error": "no_conversation"},
            )

        try:
            # Build full conversation
            full_conversation = "\n".join(
                [f"{turn.speaker}: {turn.message}" for turn in conversation.turns]
            )

            # Evaluate completeness
            completeness_prompt = f"""
            Analyze the following conversation and evaluate how complete it is.

            Conversation:
            {full_conversation}

            Evaluate the conversation on the following criteria:
            1. Were the user's questions/requests adequately addressed?
            2. Are there any unresolved issues or hanging threads?
            3. Does the conversation reach a natural conclusion?
            4. Would the user likely be satisfied with the outcome?

            Provide a completeness score from 1-5 where:
            1 = Very incomplete, major issues unresolved
            2 = Somewhat incomplete, some issues unresolved
            3 = Moderately complete, minor issues remain
            4 = Mostly complete, well-addressed
            5 = Fully complete, all issues resolved satisfactorily

            Format your response as:
            Analysis:
            1. Questions/Requests Addressed: [evaluation]
            2. Unresolved Issues: [evaluation]
            3. Natural Conclusion: [evaluation]
            4. User Satisfaction: [evaluation]

            Completeness Score: [1-5]
            Reasoning: [Brief explanation]
            """

            completeness_response = await self.model.generate(completeness_prompt)
            completeness_score, reasoning = self._parse_completeness_score(
                completeness_response
            )

            # Normalize score to 0-1 range
            normalized_score = (completeness_score - 1) / 4

            detailed_reasoning = f"""
            Conversation Completeness Analysis:
            - Conversation length: {len(conversation.turns)} turns
            - Completeness score: {completeness_score}/5
            - Normalized score: {normalized_score:.3f}

            Detailed Analysis:
            {reasoning}
            """

            return ScoreResult(
                score=normalized_score,
                passed=normalized_score >= self.threshold,
                reasoning=detailed_reasoning.strip(),
                metadata={
                    "completeness_score": completeness_score,
                    "conversation_length": len(conversation.turns),
                },
            )

        except Exception as e:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"Conversation completeness evaluation failed: {e!s}",
                metadata={"error": str(e)},
            )

    def _parse_completeness_score(self, response: str) -> tuple[float, str]:
        """Parse completeness score and reasoning from LLM response."""
        import re

        # Look for "Completeness Score: X" pattern
        score_match = re.search(r"Completeness Score:\s*(\d+)", response)
        if score_match:
            score = float(score_match.group(1))
        else:
            # Look for standalone numbers 1-5
            numbers = re.findall(r"\b([1-5])\b", response)
            score = float(numbers[-1]) if numbers else 3.0

        # Extract reasoning
        reasoning_match = re.search(r"Reasoning:\s*(.+)", response, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else response

        return score, reasoning


class ConversationRelevancyScorer(BaseScorer):
    """
    Evaluates how relevant each response is within the conversation context.

    This metric measures whether responses stay on topic and are appropriate
    for the conversation flow.
    """

    def __init__(self, model: LLMModel, threshold: float = 0.7, **kwargs):
        super().__init__(threshold=threshold, **kwargs)
        self.model = model

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
        conversation: Optional[Conversation] = None,
        **kwargs,
    ) -> ScoreResult:
        """Evaluate conversation relevancy."""

        if not conversation:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="No conversation provided for relevancy evaluation",
                metadata={"error": "no_conversation"},
            )

        try:
            # Get conversation context (previous turns)
            if len(conversation.turns) <= 1:
                # First turn, evaluate against system context if available
                context_for_eval = conversation.context or "General conversation"
            else:
                # Build context from previous turns
                previous_turns = conversation.turns[:-1]
                context_for_eval = "\n".join(
                    [
                        f"{turn.speaker}: {turn.message}"
                        for turn in previous_turns[-5:]  # Last 5 turns for context
                    ]
                )

            current_response = conversation.turns[-1].message

            # Evaluate relevancy
            relevancy_prompt = f"""
            Conversation Context:
            {context_for_eval}

            Current Response: {current_response}

            Evaluate how relevant the current response is to the conversation context.
            Consider:
            1. Does it address the immediate question or topic?
            2. Is it appropriate for the conversation flow?
            3. Does it maintain topical coherence?
            4. Is the tone and style consistent?

            Provide a relevancy score from 1-5 where:
            1 = Completely irrelevant or off-topic
            2 = Somewhat relevant but misses the point
            3 = Moderately relevant, addresses some aspects
            4 = Highly relevant, well-aligned with context
            5 = Perfectly relevant, ideal response for the context

            Format your response as:
            Analysis:
            1. Addresses Question/Topic: [evaluation]
            2. Conversation Flow: [evaluation]
            3. Topical Coherence: [evaluation]
            4. Tone/Style Consistency: [evaluation]

            Relevancy Score: [1-5]
            Reasoning: [Brief explanation]
            """

            relevancy_response = await self.model.generate(relevancy_prompt)
            relevancy_score, reasoning = self._parse_relevancy_score(relevancy_response)

            # Normalize score to 0-1 range
            normalized_score = (relevancy_score - 1) / 4

            detailed_reasoning = f"""
            Conversation Relevancy Analysis:
            - Evaluated against {len(conversation.turns)-1} previous turns
            - Relevancy score: {relevancy_score}/5
            - Normalized score: {normalized_score:.3f}

            Detailed Analysis:
            {reasoning}
            """

            return ScoreResult(
                score=normalized_score,
                passed=normalized_score >= self.threshold,
                reasoning=detailed_reasoning.strip(),
                metadata={
                    "relevancy_score": relevancy_score,
                    "context_turns": len(conversation.turns) - 1,
                },
            )

        except Exception as e:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"Conversation relevancy evaluation failed: {e!s}",
                metadata={"error": str(e)},
            )

    def _parse_relevancy_score(self, response: str) -> tuple[float, str]:
        """Parse relevancy score and reasoning from LLM response."""
        import re

        # Look for "Relevancy Score: X" pattern
        score_match = re.search(r"Relevancy Score:\s*(\d+)", response)
        if score_match:
            score = float(score_match.group(1))
        else:
            # Look for standalone numbers 1-5
            numbers = re.findall(r"\b([1-5])\b", response)
            score = float(numbers[-1]) if numbers else 3.0

        # Extract reasoning
        reasoning_match = re.search(r"Reasoning:\s*(.+)", response, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else response

        return score, reasoning


class RoleAdherenceScorer(BaseScorer):
    """
    Evaluates how well the AI adheres to its assigned role or persona.

    This metric measures consistency with system prompts, character traits,
    and behavioral guidelines throughout the conversation.
    """

    def __init__(self, model: LLMModel, threshold: float = 0.8, **kwargs):
        super().__init__(threshold=threshold, **kwargs)
        self.model = model

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
        conversation: Optional[Conversation] = None,
        role_description: Optional[str] = None,
        **kwargs,
    ) -> ScoreResult:
        """Evaluate role adherence."""

        if not conversation:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="No conversation provided for role adherence evaluation",
                metadata={"error": "no_conversation"},
            )

        # Use provided role description or extract from conversation context
        role_desc = role_description or conversation.context

        if not role_desc:
            return ScoreResult(
                score=1.0,  # No role to adhere to
                passed=True,
                reasoning="No role description provided, cannot evaluate adherence",
                metadata={"role_description": None},
            )

        try:
            # Get AI responses from conversation
            ai_responses = [
                turn.message
                for turn in conversation.turns
                if turn.speaker.lower() in ["assistant", "ai", "bot"]
            ]

            if not ai_responses:
                return ScoreResult(
                    score=0.0,
                    passed=False,
                    reasoning="No AI responses found in conversation",
                    metadata={"ai_responses": 0},
                )

            # Evaluate role adherence across all AI responses
            adherence_prompt = f"""
            Role Description/System Prompt:
            {role_desc}

            AI Responses in Conversation:
            {chr(10).join(f'{i+1}. {response}' for i, response in enumerate(ai_responses))}

            Evaluate how well the AI responses adhere to the specified role throughout the conversation.
            Consider:
            1. Consistency with role characteristics
            2. Appropriate tone and language style
            3. Adherence to behavioral guidelines
            4. Maintenance of persona throughout conversation

            Provide an adherence score from 1-5 where:
            1 = Poor adherence, frequently breaks character
            2 = Below average adherence, some inconsistencies
            3 = Average adherence, mostly consistent
            4 = Good adherence, minor deviations
            5 = Excellent adherence, perfect role consistency

            Format your response as:
            Analysis:
            1. Role Characteristics: [evaluation]
            2. Tone/Language Style: [evaluation]
            3. Behavioral Guidelines: [evaluation]
            4. Persona Consistency: [evaluation]

            Adherence Score: [1-5]
            Reasoning: [Brief explanation]
            """

            adherence_response = await self.model.generate(adherence_prompt)
            adherence_score, reasoning = self._parse_adherence_score(adherence_response)

            # Normalize score to 0-1 range
            normalized_score = (adherence_score - 1) / 4

            detailed_reasoning = f"""
            Role Adherence Analysis:
            - Evaluated {len(ai_responses)} AI responses
            - Role adherence score: {adherence_score}/5
            - Normalized score: {normalized_score:.3f}

            Role Description: {role_desc[:200]}{'...' if len(role_desc) > 200 else ''}

            Detailed Analysis:
            {reasoning}
            """

            return ScoreResult(
                score=normalized_score,
                passed=normalized_score >= self.threshold,
                reasoning=detailed_reasoning.strip(),
                metadata={
                    "adherence_score": adherence_score,
                    "ai_responses_count": len(ai_responses),
                    "role_description": role_desc,
                },
            )

        except Exception as e:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"Role adherence evaluation failed: {e!s}",
                metadata={"error": str(e)},
            )

    def _parse_adherence_score(self, response: str) -> tuple[float, str]:
        """Parse adherence score and reasoning from LLM response."""
        import re

        # Look for "Adherence Score: X" pattern
        score_match = re.search(r"Adherence Score:\s*(\d+)", response)
        if score_match:
            score = float(score_match.group(1))
        else:
            # Look for standalone numbers 1-5
            numbers = re.findall(r"\b([1-5])\b", response)
            score = float(numbers[-1]) if numbers else 3.0

        # Extract reasoning
        reasoning_match = re.search(r"Reasoning:\s*(.+)", response, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else response

        return score, reasoning


class ConversationalMetricsScorer(BaseScorer):
    """
    Composite scorer that evaluates multiple conversational metrics.

    Combines knowledge retention, completeness, relevancy, and role adherence
    into a comprehensive conversational evaluation.
    """

    def __init__(
        self,
        model: LLMModel,
        threshold: float = 0.7,
        weights: Optional[dict[str, float]] = None,
        **kwargs,
    ):
        super().__init__(threshold=threshold, **kwargs)
        self.model = model

        # Default weights for different metrics
        self.weights = weights or {
            "knowledge_retention": 0.25,
            "completeness": 0.25,
            "relevancy": 0.25,
            "role_adherence": 0.25,
        }

        # Initialize individual scorers
        self.knowledge_retention_scorer = KnowledgeRetentionScorer(model, threshold=0.7)
        self.completeness_scorer = ConversationCompletenessScorer(model, threshold=0.7)
        self.relevancy_scorer = ConversationRelevancyScorer(model, threshold=0.7)
        self.role_adherence_scorer = RoleAdherenceScorer(model, threshold=0.8)

    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[str] = None,
        conversation: Optional[Conversation] = None,
        **kwargs,
    ) -> ScoreResult:
        """Evaluate using comprehensive conversational metrics."""

        if not conversation:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning="No conversation provided for conversational metrics evaluation",
                metadata={"error": "no_conversation"},
            )

        try:
            # Run all individual evaluations in parallel
            results = await asyncio.gather(
                self.knowledge_retention_scorer.evaluate(
                    input_text,
                    output_text,
                    expected_output,
                    context,
                    conversation=conversation,
                ),
                self.completeness_scorer.evaluate(
                    input_text,
                    output_text,
                    expected_output,
                    context,
                    conversation=conversation,
                ),
                self.relevancy_scorer.evaluate(
                    input_text,
                    output_text,
                    expected_output,
                    context,
                    conversation=conversation,
                ),
                self.role_adherence_scorer.evaluate(
                    input_text,
                    output_text,
                    expected_output,
                    context,
                    conversation=conversation,
                ),
                return_exceptions=True,
            )

            # Extract scores and handle exceptions
            scores = {}
            reasonings = {}

            metric_names = [
                "knowledge_retention",
                "completeness",
                "relevancy",
                "role_adherence",
            ]

            for _i, (metric_name, result) in enumerate(zip(metric_names, results)):
                if isinstance(result, Exception):
                    scores[metric_name] = 0.0
                    reasonings[metric_name] = f"Error: {result!s}"
                else:
                    scores[metric_name] = result.score
                    reasonings[metric_name] = result.reasoning

            # Calculate weighted average
            total_weight = sum(self.weights.values())
            conversational_score = (
                sum(scores[metric] * self.weights[metric] for metric in scores)
                / total_weight
            )

            # Compile comprehensive reasoning
            reasoning = f"""
            Conversational Metrics Evaluation Results:

            Individual Metric Scores:
            • Knowledge Retention: {scores['knowledge_retention']:.3f} (weight: {self.weights['knowledge_retention']})
            • Completeness: {scores['completeness']:.3f} (weight: {self.weights['completeness']})
            • Relevancy: {scores['relevancy']:.3f} (weight: {self.weights['relevancy']})
            • Role Adherence: {scores['role_adherence']:.3f} (weight: {self.weights['role_adherence']})

            Weighted Conversational Score: {conversational_score:.3f}

            Conversation Overview:
            - Total turns: {len(conversation.turns)}
            - AI responses: {len([t for t in conversation.turns if t.speaker.lower() in ['assistant', 'ai', 'bot']])}
            - User messages: {len([t for t in conversation.turns if t.speaker.lower() == 'user'])}

            Detailed Analysis:
            {chr(10).join(f'{metric.replace("_", " ").title()}:{chr(10)}{reasoning}{chr(10)}' for metric, reasoning in reasonings.items())}
            """

            return ScoreResult(
                score=conversational_score,
                passed=conversational_score >= self.threshold,
                reasoning=reasoning.strip(),
                metadata={
                    "individual_scores": scores,
                    "weights": self.weights,
                    "conversational_score": conversational_score,
                    "conversation_stats": {
                        "total_turns": len(conversation.turns),
                        "ai_responses": len(
                            [
                                t
                                for t in conversation.turns
                                if t.speaker.lower() in ["assistant", "ai", "bot"]
                            ]
                        ),
                        "user_messages": len(
                            [
                                t
                                for t in conversation.turns
                                if t.speaker.lower() == "user"
                            ]
                        ),
                    },
                },
            )

        except Exception as e:
            return ScoreResult(
                score=0.0,
                passed=False,
                reasoning=f"Conversational metrics evaluation failed: {e!s}",
                metadata={"error": str(e)},
            )
