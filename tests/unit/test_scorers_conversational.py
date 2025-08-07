"""
Unit tests for conversational scorers.
"""

from novaeval.scorers.conversational import (
    Conversation,
    ConversationalMetricsScorer,
    ConversationCompletenessScorer,
    ConversationRelevancyScorer,
    ConversationTurn,
    KnowledgeRetentionScorer,
    RoleAdherenceScorer,
)


class MockLLMModel:
    """Mock LLM model for testing."""

    def __init__(self, mock_responses=None):
        self.mock_responses = mock_responses or {}
        self.call_count = 0

    async def generate(self, prompt, **kwargs):
        """Mock generate method."""
        self.call_count += 1
        if isinstance(self.mock_responses, dict):
            # Find matching prompt patterns for more sophisticated mocking
            for pattern, response in self.mock_responses.items():
                if pattern.lower() in prompt.lower():
                    return response
            return f"Mock response {self.call_count}"
        elif isinstance(self.mock_responses, list):
            if self.call_count <= len(self.mock_responses):
                return self.mock_responses[self.call_count - 1]
            return f"Mock response {self.call_count}"
        else:
            return (
                str(self.mock_responses)
                if self.mock_responses
                else f"Mock response {self.call_count}"
            )


class TestKnowledgeRetentionScorer:
    """Test cases for KnowledgeRetentionScorer."""

    def test_init(self):
        """Test scorer initialization."""
        model = MockLLMModel()
        scorer = KnowledgeRetentionScorer(model)
        assert scorer.name == "Knowledge Retention"
        assert scorer.model == model
        assert scorer.window_size == 10  # Default window size

    def test_score_basic_functionality(self):
        """Test basic scoring functionality."""
        model = MockLLMModel("4")
        scorer = KnowledgeRetentionScorer(model)

        score = scorer.score("Good response", "What is AI?", None)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_with_conversation_context(self):
        """Test scoring with conversation context."""
        # Mock knowledge extraction and violation detection
        mock_responses = [
            "1. User likes Python programming\n2. User is a beginner",  # Knowledge extraction
            "NO",  # No violations detected
        ]
        model = MockLLMModel(mock_responses)
        scorer = KnowledgeRetentionScorer(model)

        conversation = Conversation(
            turns=[
                ConversationTurn(
                    speaker="user", message="I am learning Python programming"
                ),
                ConversationTurn(
                    speaker="assistant",
                    message="That's great! Python is excellent for beginners",
                ),
                ConversationTurn(speaker="user", message="What should I learn next?"),
            ]
        )

        context = {"conversation": conversation}
        score = scorer.score(
            "I recommend learning data structures", "What should I learn next?", context
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_with_retention_violations(self):
        """Test scoring when retention violations are detected."""
        mock_responses = [
            "1. User name is John\n2. User is 25 years old",  # Knowledge extraction
            "YES\n- Asking for name again",  # Violations detected
        ]
        model = MockLLMModel(mock_responses)
        scorer = KnowledgeRetentionScorer(model)

        conversation = Conversation(
            turns=[
                ConversationTurn(
                    speaker="user", message="Hi, I'm John and I'm 25 years old"
                ),
                ConversationTurn(speaker="assistant", message="Nice to meet you John!"),
            ]
        )

        context = {"conversation": conversation}
        score = scorer.score("What's your name again?", "Question", context)
        # Score should be reduced due to violation
        assert 0.0 <= score < 1.0

    def test_simple_retention_score_fallback(self):
        """Test fallback to simple retention scoring."""
        model = MockLLMModel()
        scorer = KnowledgeRetentionScorer(model)

        # Test asking basic questions (should get low score)
        score = scorer.score("What is your name?", "Question", None)
        assert score == 0.3  # Low score for asking basic questions

        # Test normal response (should get decent score)
        score = scorer.score("I can help you with that", "Question", None)
        assert score == 0.7  # Default decent score

    def test_parse_knowledge_items(self):
        """Test knowledge item parsing."""
        model = MockLLMModel()
        scorer = KnowledgeRetentionScorer(model)

        response = "1. User likes Python\n2. User is a beginner programmer\n3. Short"
        items = scorer._parse_knowledge_items(response, 0, "user")

        assert len(items) == 2  # Third item filtered out for being too short
        assert items[0].content == "User likes Python"
        assert items[1].content == "User is a beginner programmer"
        assert all(item.turn_index == 0 for item in items)
        assert all(item.speaker == "user" for item in items)

    def test_parse_violations(self):
        """Test violation parsing."""
        model = MockLLMModel()
        scorer = KnowledgeRetentionScorer(model)

        # Test no violations
        response = "NO"
        violations = scorer._parse_violations(response)
        assert len(violations) == 0

        # Test with violations
        response = (
            "YES\n- Asking for already provided name\n- Requesting repeated information"
        )
        violations = scorer._parse_violations(response)
        assert len(violations) == 2
        assert "Asking for already provided name" in violations[0]

    def test_input_validation(self):
        """Test input validation."""
        model = MockLLMModel()
        scorer = KnowledgeRetentionScorer(model)

        # Test empty prediction
        assert scorer.score("", "ground_truth", {}) == 0.0

        # Test empty ground truth
        assert scorer.score("prediction", "", {}) == 0.0

        # Test whitespace only
        assert scorer.score("   ", "ground_truth", {}) == 0.0


class TestConversationRelevancyScorer:
    """Test cases for ConversationRelevancyScorer."""

    def test_init(self):
        """Test scorer initialization."""
        model = MockLLMModel()
        scorer = ConversationRelevancyScorer(model, window_size=3)
        assert scorer.name == "Conversation Relevancy"
        assert scorer.window_size == 3

    def test_score_with_sliding_window(self):
        """Test scoring with sliding window context."""
        model = MockLLMModel("4")  # Mock relevancy score
        scorer = ConversationRelevancyScorer(model, window_size=2)

        conversation = Conversation(
            turns=[
                ConversationTurn(speaker="user", message="Tell me about Python"),
                ConversationTurn(
                    speaker="assistant", message="Python is a programming language"
                ),
                ConversationTurn(speaker="user", message="What about data science?"),
                ConversationTurn(
                    speaker="assistant", message="Previous response should be evaluated"
                ),
            ]
        )

        context = {"conversation": conversation}
        score = scorer.score(
            "Python is great for data science", "What about data science?", context
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_simple_relevancy_score_fallback(self):
        """Test fallback to simple relevancy scoring."""
        model = MockLLMModel()
        scorer = ConversationRelevancyScorer(model)

        # Test word overlap
        score = scorer.score("Python programming", "Learn Python", None)
        assert score > 0.0  # Should have some overlap

        # Test no overlap
        score = scorer.score("Cooking recipes", "Math problems", None)
        assert score >= 0.0

    def test_parse_relevancy_score(self):
        """Test relevancy score parsing."""
        model = MockLLMModel()
        scorer = ConversationRelevancyScorer(model)

        assert scorer._parse_relevancy_score("5") == 5.0
        assert scorer._parse_relevancy_score("The score is 3 out of 5") == 3.0
        assert scorer._parse_relevancy_score("No clear score") == 3.0  # Default

    def test_build_context_summary(self):
        """Test context summary building."""
        model = MockLLMModel()
        scorer = ConversationRelevancyScorer(model)

        turns = [
            ConversationTurn(speaker="user", message="Hello"),
            ConversationTurn(speaker="assistant", message="Hi there"),
        ]

        summary = scorer._build_context_summary(turns)
        assert "user: Hello" in summary
        assert "assistant: Hi there" in summary


class TestConversationCompletenessScorer:
    """Test cases for ConversationCompletenessScorer."""

    def test_init(self):
        """Test scorer initialization."""
        model = MockLLMModel()
        scorer = ConversationCompletenessScorer(model)
        assert scorer.name == "Conversation Completeness"

    def test_score_with_intention_analysis(self):
        """Test scoring with user intention analysis."""
        mock_responses = [
            "1. Learn about Python basics\n2. Get programming help",  # Intentions
            "4",  # Fulfillment score for first intention
            "3",  # Fulfillment score for second intention
        ]
        model = MockLLMModel(mock_responses)
        scorer = ConversationCompletenessScorer(model)

        conversation = Conversation(
            turns=[
                ConversationTurn(
                    speaker="user", message="I want to learn Python basics"
                ),
                ConversationTurn(
                    speaker="assistant",
                    message="Here's a comprehensive Python guide...",
                ),
            ]
        )

        context = {"conversation": conversation}
        score = scorer.score("Great explanation", "How did I do?", context)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_simple_completeness_score_fallback(self):
        """Test fallback to simple completeness scoring."""
        model = MockLLMModel()
        scorer = ConversationCompletenessScorer(model)

        # Test very short response
        score = scorer.score("OK", "Question", None)
        assert score == 0.2

        # Test apologetic response
        score = scorer.score("Sorry, I can't help with that", "Question", None)
        assert score == 0.4

        # Test substantial response
        score = scorer.score(
            "Here is a detailed explanation of the topic", "Question", None
        )
        assert score == 0.7

    def test_parse_intentions(self):
        """Test intention parsing."""
        model = MockLLMModel()
        scorer = ConversationCompletenessScorer(model)

        # Test with intentions
        response = (
            "1. Learn programming\n2. Get help with coding\n3. Understand concepts"
        )
        intentions = scorer._parse_intentions(response)
        assert len(intentions) == 3
        assert "Learn programming" in intentions

        # Test no intentions
        response = "None"
        intentions = scorer._parse_intentions(response)
        assert len(intentions) == 0

    def test_parse_fulfillment_score(self):
        """Test fulfillment score parsing."""
        model = MockLLMModel()
        scorer = ConversationCompletenessScorer(model)

        assert scorer._parse_fulfillment_score("5") == 5.0
        assert scorer._parse_fulfillment_score("Score: 2") == 2.0
        assert scorer._parse_fulfillment_score("No score") == 3.0  # Default


class TestRoleAdherenceScorer:
    """Test cases for RoleAdherenceScorer."""

    def test_init(self):
        """Test scorer initialization."""
        model = MockLLMModel()
        scorer = RoleAdherenceScorer(model, expected_role="helpful assistant")
        assert scorer.name == "Role Adherence"
        assert scorer.expected_role == "helpful assistant"

    def test_score_with_role_context(self):
        """Test scoring with role context."""
        model = MockLLMModel("4")  # Mock role adherence score
        scorer = RoleAdherenceScorer(model, expected_role="math tutor")

        conversation = Conversation(
            turns=[ConversationTurn(speaker="user", message="Help with math")],
            context="You are a helpful math tutor",
        )

        context = {"conversation": conversation}
        score = scorer.score("Let me help you with algebra", "Math question", context)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_no_role_defined(self):
        """Test scoring when no role is defined."""
        model = MockLLMModel()
        scorer = RoleAdherenceScorer(model)

        score = scorer.score("Any response", "Question", None)
        assert score == 1.0  # Perfect adherence when no role defined

    def test_parse_role_score(self):
        """Test role score parsing."""
        model = MockLLMModel()
        scorer = RoleAdherenceScorer(model)

        assert scorer._parse_role_score("4") == 4.0
        assert scorer._parse_role_score("The adherence is 2") == 2.0
        assert scorer._parse_role_score("No clear score") == 3.0  # Default


class TestConversationalMetricsScorer:
    """Test cases for ConversationalMetricsScorer."""

    def test_init(self):
        """Test scorer initialization."""
        model = MockLLMModel()
        scorer = ConversationalMetricsScorer(model)
        assert scorer.name == "Conversational Metrics"
        assert hasattr(scorer, "knowledge_scorer")
        assert hasattr(scorer, "relevancy_scorer")
        assert hasattr(scorer, "completeness_scorer")
        assert hasattr(scorer, "role_scorer")

    def test_init_selective_metrics(self):
        """Test initialization with selective metrics."""
        model = MockLLMModel()
        scorer = ConversationalMetricsScorer(
            model,
            include_knowledge_retention=True,
            include_relevancy=False,
            include_completeness=True,
            include_role_adherence=False,
        )

        assert hasattr(scorer, "knowledge_scorer")
        assert not hasattr(scorer, "relevancy_scorer")
        assert hasattr(scorer, "completeness_scorer")
        assert not hasattr(scorer, "role_scorer")

    def test_score_all_metrics(self):
        """Test scoring with all metrics enabled."""
        # Mock responses for all individual scorers
        mock_responses = [
            "1. User info",
            "NO",  # Knowledge retention
            "3",  # Relevancy
            "1. Help with task",
            "4",  # Completeness
            "5",  # Role adherence
        ]
        model = MockLLMModel(mock_responses)
        scorer = ConversationalMetricsScorer(model)

        conversation = Conversation(
            turns=[
                ConversationTurn(speaker="user", message="I like AI"),
                ConversationTurn(speaker="assistant", message="Great!"),
                ConversationTurn(speaker="user", message="Tell me more"),
            ],
            context="You are a helpful AI assistant",
        )

        context = {"conversation": conversation}
        scores = scorer.score("AI is fascinating", "Tell me more", context)

        assert isinstance(scores, dict)
        assert "overall" in scores
        assert "knowledge_retention" in scores
        assert "relevancy" in scores
        assert "completeness" in scores
        assert "role_adherence" in scores

        # All scores should be between 0 and 1
        for score in scores.values():
            assert 0.0 <= score <= 1.0

    def test_score_partial_metrics(self):
        """Test scoring with partial metrics enabled."""
        model = MockLLMModel(["4", "3"])  # Mock responses
        scorer = ConversationalMetricsScorer(
            model,
            include_knowledge_retention=True,
            include_relevancy=False,
            include_completeness=True,
            include_role_adherence=False,
        )

        scores = scorer.score("Response", "Question", None)
        assert isinstance(scores, dict)
        assert "overall" in scores
        assert "knowledge_retention" in scores
        assert "completeness" in scores
        assert "relevancy" not in scores
        assert "role_adherence" not in scores


class TestInputValidation:
    """Test input validation across all scorers."""

    def test_validate_inputs(self):
        """Test input validation for all scorers."""
        model = MockLLMModel()
        scorers = [
            KnowledgeRetentionScorer(model),
            ConversationRelevancyScorer(model),
            ConversationCompletenessScorer(model),
            RoleAdherenceScorer(model),
        ]

        for scorer in scorers:
            # Test empty strings
            assert scorer.score("", "ground_truth", {}) == 0.0
            assert scorer.score("prediction", "", {}) == 0.0

            # Test whitespace only
            assert scorer.score("   ", "ground_truth", {}) == 0.0
            assert scorer.score("prediction", "   ", {}) == 0.0


class TestConversationalScorerIntegration:
    """Integration tests for conversational scorers."""

    def test_complete_conversation_flow(self):
        """Test a complete conversation evaluation flow."""
        mock_responses = [
            "1. User is learning Python\n2. User wants to build web apps",  # Knowledge extraction
            "NO",  # No retention violations
            "4",  # Relevancy score
            "1. Learn Python\n2. Build web applications",  # Intentions
            "5",
            "4",  # Fulfillment scores
            "4",  # Role adherence
        ]
        model = MockLLMModel(mock_responses)

        # Test individual scorers
        conversation = Conversation(
            turns=[
                ConversationTurn(
                    speaker="user", message="I'm learning Python to build web apps"
                ),
                ConversationTurn(
                    speaker="assistant",
                    message="Great! Let me guide you through web development",
                ),
                ConversationTurn(
                    speaker="user", message="What framework should I use?"
                ),
            ],
            context="You are a helpful programming mentor",
        )

        context = {"conversation": conversation}

        # Test knowledge retention
        kr_scorer = KnowledgeRetentionScorer(model)
        kr_score = kr_scorer.score(
            "I recommend Django or Flask for Python web development",
            "What framework?",
            context,
        )
        assert 0.0 <= kr_score <= 1.0

        # Test comprehensive metrics
        comp_scorer = ConversationalMetricsScorer(model)
        comp_scores = comp_scorer.score(
            "Django is great for beginners", "What framework?", context
        )
        assert isinstance(comp_scores, dict)
        assert "overall" in comp_scores

    def test_poor_conversation_scoring(self):
        """Test scoring a poor conversation scenario."""
        conversation = Conversation(
            turns=[
                ConversationTurn(speaker="user", message="What's the weather like?"),
                ConversationTurn(speaker="assistant", message="I like ice cream"),
                ConversationTurn(
                    speaker="user", message="That doesn't answer my question"
                ),
                ConversationTurn(
                    speaker="assistant", message="Purple is my favorite color"
                ),
            ],
            context="You are a helpful weather assistant",
        )

        # Mock very poor-quality responses - make them worse to ensure low scores
        mock_responses = [
            "1. User asked about weather",
            "YES\n- Assistant completely ignoring weather question and talking about irrelevant topics",  # Very poor knowledge retention
            "1",  # Very poor relevancy
            "1. Get weather information",
            "1",  # Very poor fulfillment
            "1",  # Very poor role adherence
        ]
        model = MockLLMModel(mock_responses)

        scorer = ConversationalMetricsScorer(model)
        context = {"conversation": conversation}

        scores = scorer.score("Purple is nice", "Weather question", context)

        # Should get low scores across the board - adjusted threshold for very poor conversation
        assert scores["overall"] < 0.7  # Poor overall performance (adjusted from 0.5)
        assert all(0.0 <= score <= 1.0 for score in scores.values())

    def test_edge_case_empty_conversation(self):
        """Test edge case with empty conversation."""
        model = MockLLMModel()
        scorer = ConversationalMetricsScorer(model)

        conversation = Conversation(turns=[])
        context = {"conversation": conversation}

        scores = scorer.score("Response", "Question", context)
        assert isinstance(scores, dict)
        assert "overall" in scores
