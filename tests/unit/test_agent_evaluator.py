"""
Tests for the AgentEvaluator class.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock

from novaeval.evaluators.agent_evaluator import AgentEvaluator
from novaeval.datasets.agent_dataset import AgentDataset
from novaeval.models.base import BaseModel
from novaeval.scorers.agent_scorers import AgentScorers


class TestAgentEvaluator:
    """Test cases for AgentEvaluator."""

    def test_init(self, tmp_path):
        """Test AgentEvaluator initialization."""
        # Mock dependencies
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]
        agent_scorers = [Mock(spec=AgentScorers)]
        
        # Create evaluator
        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            agent_scorers=agent_scorers,
            output_dir=tmp_path,
            stream=False,
            include_reasoning=True
        )
        
        # Verify initialization
        assert evaluator.agent_dataset == agent_dataset
        assert evaluator.models == models
        assert evaluator.agent_scorers == agent_scorers
        assert evaluator.output_dir == tmp_path
        assert evaluator.stream is False
        assert evaluator.include_reasoning is True
        assert evaluator.results_df is not None
        assert len(evaluator.results_df.columns) > 0

    def test_initialize_dataframe_with_reasoning(self, tmp_path):
        """Test DataFrame initialization with reasoning enabled."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]
        agent_scorers = [Mock(spec=AgentScorers), Mock(spec=AgentScorers)]
        
        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            agent_scorers=agent_scorers,
            output_dir=tmp_path,
            include_reasoning=True
        )
        
        # Check that DataFrame has expected columns
        expected_base_columns = ["user_id", "task_id", "turn_id", "agent_name"]
        expected_scorer_columns = ["scorer_0", "scorer_1"]
        expected_reasoning_columns = ["scorer_0_reasoning", "scorer_1_reasoning"]
        
        all_expected_columns = expected_base_columns + expected_scorer_columns + expected_reasoning_columns
        
        for col in all_expected_columns:
            assert col in evaluator.results_df.columns

    def test_initialize_dataframe_without_reasoning(self, tmp_path):
        """Test DataFrame initialization without reasoning."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]
        agent_scorers = [Mock(spec=AgentScorers)]
        
        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            agent_scorers=agent_scorers,
            output_dir=tmp_path,
            include_reasoning=False
        )
        
        # Check that DataFrame has expected columns (no reasoning columns)
        expected_base_columns = ["user_id", "task_id", "turn_id", "agent_name"]
        expected_scorer_columns = ["scorer_0"]
        
        all_expected_columns = expected_base_columns + expected_scorer_columns
        
        for col in all_expected_columns:
            assert col in evaluator.results_df.columns
        
        # Check that reasoning columns are not present
        assert "scorer_0_reasoning" not in evaluator.results_df.columns

    def test_evaluate_sample(self, tmp_path):
        """Test sample evaluation."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]
        agent_scorers = [Mock(spec=AgentScorers)]
        
        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            agent_scorers=agent_scorers,
            output_dir=tmp_path
        )
        
        # Mock sample
        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"
        
        # Mock scorer result
        mock_score_result = Mock()
        mock_score_result.score = 0.85
        mock_score_result.reasoning = "Good performance"
        
        # Mock the AgentScorers.score_all method
        mock_scorer = Mock(spec=AgentScorers)
        mock_scorer.score_all.return_value = {"scorer_0": mock_score_result}
        
        # Replace the scorer in the evaluator
        evaluator.agent_scorers = [mock_scorer]
        
        # Evaluate sample
        result = evaluator.evaluate_sample(sample)
        
        # Verify result structure
        assert result["user_id"] == "user1"
        assert result["task_id"] == "task1"
        assert result["turn_id"] == "turn1"
        assert result["agent_name"] == "agent1"
        assert "scores" in result
        assert "reasoning" in result
        assert result["error"] is None

    def test_add_result_to_dataframe(self, tmp_path):
        """Test adding result to DataFrame."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]
        agent_scorers = [Mock(spec=AgentScorers)]
        
        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            agent_scorers=agent_scorers,
            output_dir=tmp_path,
            include_reasoning=True
        )
        
        # Sample result
        sample_result = {
            "user_id": "user1",
            "task_id": "task1",
            "turn_id": "turn1",
            "agent_name": "agent1",
            "scores": {"scorer_0": 0.85},
            "reasoning": {"scorer_0": "Good performance"},
            "error": None
        }
        
        # Add to DataFrame
        evaluator._add_result_to_dataframe(sample_result)
        
        # Verify DataFrame has the data
        assert len(evaluator.results_df) == 1
        assert evaluator.results_df.iloc[0]["user_id"] == "user1"
        assert evaluator.results_df.iloc[0]["scorer_0"] == 0.85
        assert evaluator.results_df.iloc[0]["scorer_0_reasoning"] == "Good performance"

    def test_save_results_csv(self, tmp_path):
        """Test saving results as CSV."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]
        agent_scorers = [Mock(spec=AgentScorers)]
        
        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            agent_scorers=agent_scorers,
            output_dir=tmp_path
        )
        
        # Sample results
        results = [
            {
                "user_id": "user1",
                "task_id": "task1",
                "scores": {"scorer_0": 0.85}
            }
        ]
        
        # Save results
        evaluator.save_results(results, file_type="csv")
        
        # Check that file was created
        csv_file = tmp_path / "agent_evaluation_results.csv"
        assert csv_file.exists()

    def test_save_results_json(self, tmp_path):
        """Test saving results as JSON."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]
        agent_scorers = [Mock(spec=AgentScorers)]
        
        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            agent_scorers=agent_scorers,
            output_dir=tmp_path
        )
        
        # Sample results
        results = [
            {
                "user_id": "user1",
                "task_id": "task1",
                "scores": {"scorer_0": 0.85}
            }
        ]
        
        # Save results
        evaluator.save_results(results, file_type="json")
        
        # Check that file was created
        json_file = tmp_path / "agent_evaluation_results.json"
        assert json_file.exists()

    def test_convert_to_json_streaming(self, tmp_path):
        """Test JSON conversion with streaming."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]
        agent_scorers = [Mock(spec=AgentScorers)]
        
        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            agent_scorers=agent_scorers,
            output_dir=tmp_path,
            stream=True
        )
        
        # Add some data to DataFrame
        sample_result = {
            "user_id": "user1",
            "task_id": "task1",
            "turn_id": "turn1",
            "agent_name": "agent1",
            "scores": {"scorer_0": 0.85},
            "reasoning": {"scorer_0": "Good performance"},
            "error": None
        }
        evaluator._add_result_to_dataframe(sample_result)
        
        # Convert to JSON
        evaluator._convert_to_json()
        
        # Check that file was created
        json_file = tmp_path / "agent_evaluation_results.json"
        assert json_file.exists()

    def test_run_method(self, tmp_path):
        """Test the run method."""
        agent_dataset = Mock(spec=AgentDataset)
        agent_dataset.get_datapoint.return_value = []  # Empty dataset
        
        models = [Mock(spec=BaseModel)]
        agent_scorers = [Mock(spec=AgentScorers)]
        
        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            agent_scorers=agent_scorers,
            output_dir=tmp_path
        )
        
        # Run evaluation
        result = evaluator.run()
        
        # Verify result structure
        assert "total_samples" in result
        assert "scorers_used" in result
        assert "models_used" in result
        assert "output_directory" in result
        assert "include_reasoning" in result
        assert "streaming_mode" in result 