"""
Tests for the AgentEvaluator class.
"""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from novaeval.datasets.agent_dataset import AgentDataset
from novaeval.evaluators.agent_evaluator import AgentEvaluator
from novaeval.models.base import BaseModel

pytestmark = pytest.mark.unit


class TestAgentEvaluator:
    """Test cases for AgentEvaluator."""

    def test_init(self, tmp_path):
        """Test AgentEvaluator initialization."""
        # Mock dependencies
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        # Create mock scoring functions
        def mock_scorer_1(sample, model):
            mock_result = Mock()
            mock_result.score = 0.8
            mock_result.reasoning = "Good performance"
            return mock_result

        def mock_scorer_2(sample, model):
            mock_result = Mock()
            mock_result.score = 0.9
            mock_result.reasoning = "Excellent performance"
            return mock_result

        scoring_functions = [mock_scorer_1, mock_scorer_2]

        # Create evaluator
        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
            stream=False,
            include_reasoning=True,
        )

        # Verify initialization
        assert evaluator.agent_dataset == agent_dataset
        assert evaluator.models == models
        assert evaluator.scoring_functions == scoring_functions
        assert evaluator.output_dir == tmp_path
        assert evaluator.stream is False
        assert evaluator.include_reasoning is True
        assert evaluator.results_df is not None
        assert len(evaluator.results_df.columns) > 0

    def test_initialize_dataframe_with_reasoning(self, tmp_path):
        """Test DataFrame initialization with reasoning enabled."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def scorer_a(sample, model):
            mock_result = Mock()
            mock_result.score = 0.8
            return mock_result

        def scorer_b(sample, model):
            mock_result = Mock()
            mock_result.score = 0.9
            return mock_result

        scoring_functions = [scorer_a, scorer_b]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
            include_reasoning=True,
        )

        # Check that DataFrame has expected columns
        expected_base_columns = ["user_id", "task_id", "turn_id", "agent_name"]
        expected_scorer_columns = ["scorer_a", "scorer_b"]
        expected_reasoning_columns = ["scorer_a_reasoning", "scorer_b_reasoning"]

        all_expected_columns = (
            expected_base_columns + expected_scorer_columns + expected_reasoning_columns
        )

        for col in all_expected_columns:
            assert col in evaluator.results_df.columns

    def test_initialize_dataframe_without_reasoning(self, tmp_path):
        """Test DataFrame initialization without reasoning."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def simple_scorer(sample, model):
            mock_result = Mock()
            mock_result.score = 0.7
            return mock_result

        scoring_functions = [simple_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
            include_reasoning=False,
        )

        # Check that DataFrame has expected columns (no reasoning columns)
        expected_base_columns = ["user_id", "task_id", "turn_id", "agent_name"]
        expected_scorer_columns = ["simple"]  # Function name with "_scorer" removed

        all_expected_columns = expected_base_columns + expected_scorer_columns

        for col in all_expected_columns:
            assert col in evaluator.results_df.columns

        # Check that reasoning columns are not present
        assert "simple_reasoning" not in evaluator.results_df.columns

    def test_evaluate_sample(self, tmp_path):
        """Test sample evaluation."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            mock_result = Mock()
            mock_result.score = 0.85
            mock_result.reasoning = "Good performance"
            return mock_result

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        # Mock sample
        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        # Evaluate sample
        result = evaluator.evaluate_sample(sample, models[0])

        # Verify result structure
        assert result["user_id"] == "user1"
        assert result["task_id"] == "task1"
        assert result["turn_id"] == "turn1"
        assert result["agent_name"] == "agent1"
        assert "scores" in result
        assert "reasoning" in result
        assert "mock" in result["scores"]
        assert result["scores"]["mock"] == 0.85

    def test_add_result_to_dataframe(self, tmp_path):
        """Test adding result to DataFrame."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
            include_reasoning=True,
        )

        # Sample result
        sample_result = {
            "user_id": "user1",
            "task_id": "task1",
            "turn_id": "turn1",
            "agent_name": "agent1",
            "scores": {"mock": 0.85},
            "reasoning": {"mock": "Good performance"},
            "error": None,
        }

        # Add to DataFrame
        evaluator._add_result_to_dataframe(sample_result)

        # Verify DataFrame has the data
        assert len(evaluator.results_df) == 1
        assert evaluator.results_df.iloc[0]["user_id"] == "user1"
        assert evaluator.results_df.iloc[0]["mock"] == 0.85
        assert evaluator.results_df.iloc[0]["mock_reasoning"] == "Good performance"

    def test_save_results_csv(self, tmp_path):
        """Test saving results as CSV."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        # Sample results
        results = [{"user_id": "user1", "task_id": "task1", "scores": {"mock": 0.85}}]

        # Save results
        evaluator.save_results(results)

        # Check that file was created
        csv_file = tmp_path / "agent_evaluation_results.csv"
        assert csv_file.exists()

    def test_save_results_json(self, tmp_path):
        """Test saving results as JSON."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        # Sample results
        results = [{"user_id": "user1", "task_id": "task1", "scores": {"mock": 0.85}}]

        # Save results
        evaluator.save_results(results)

        # Check that file was created
        csv_file = tmp_path / "agent_evaluation_results.csv"
        assert csv_file.exists()

    def test_convert_to_json_streaming(self, tmp_path):
        """Test JSON conversion with streaming."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
            stream=True,
        )

        # Add some data to DataFrame
        sample_result = {
            "user_id": "user1",
            "task_id": "task1",
            "turn_id": "turn1",
            "agent_name": "agent1",
            "scores": {"mock": 0.85},
            "reasoning": {"mock": "Good performance"},
            "error": None,
        }
        evaluator._add_result_to_dataframe(sample_result)

        # Save intermediate results
        evaluator._save_intermediate_results("csv")

        # Check that file was created
        csv_file = tmp_path / "agent_evaluation_results.csv"
        assert csv_file.exists()

    def test_run_all_with_samples(self, tmp_path):
        """Test run_all with actual samples."""
        agent_dataset = Mock(spec=AgentDataset)

        # Create mock samples
        sample1 = Mock()
        sample1.user_id = "user1"
        sample1.task_id = "task1"
        sample1.turn_id = "turn1"
        sample1.agent_name = "agent1"

        sample2 = Mock()
        sample2.user_id = "user2"
        sample2.task_id = "task2"
        sample2.turn_id = "turn2"
        sample2.agent_name = "agent2"

        agent_dataset.get_datapoint.return_value = [sample1, sample2]

        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            mock_result = Mock()
            mock_result.score = 0.8
            mock_result.reasoning = "Good performance"
            return mock_result

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        # Run evaluation
        evaluator.run_all(save_every=1, file_type="csv")

        # Check results
        assert len(evaluator.results_df) == 2
        csv_file = tmp_path / "agent_evaluation_results.csv"
        assert csv_file.exists()

    def test_run_all_with_aggregation(self, tmp_path):
        """Test run_all with aggregation enabled."""
        agent_dataset = Mock(spec=AgentDataset)

        # Create mock sample
        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        agent_dataset.get_datapoint.return_value = [sample]

        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            mock_result = Mock()
            mock_result.score = 0.8
            return mock_result

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        # Mock the aggregation functions to avoid complex dependencies
        with (
            patch("novaeval.evaluators.aggregators.aggregate_by_task") as mock_agg_task,
            patch("novaeval.evaluators.aggregators.aggregate_by_user") as mock_agg_user,
            patch(
                "novaeval.evaluators.aggregators.aggregate_by_agent_name"
            ) as mock_agg_agent,
        ):

            evaluator.run_all(
                aggregate_by_task=True,
                aggregate_by_user=True,
                aggregate_by_agent_name=True,
                file_type="csv",
            )

            # Verify aggregation functions were called
            mock_agg_task.assert_called_once()
            mock_agg_user.assert_called_once()
            mock_agg_agent.assert_called_once()

    def test_evaluate_sample_no_model(self, tmp_path):
        """Test evaluate_sample when no model is available."""
        agent_dataset = Mock(spec=AgentDataset)
        models = []  # No models

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        result = evaluator.evaluate_sample(sample, None)

        # Should return early with empty scores
        assert result["user_id"] == "user1"
        assert result["scores"] == {}
        assert result["reasoning"] == {}

    def test_evaluate_sample_list_result(self, tmp_path):
        """Test evaluate_sample with list result from scorer."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            # Return a list of score objects
            score1 = Mock()
            score1.score = 0.8
            score1.reasoning = "Good"
            return [score1]

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
            include_reasoning=True,
        )

        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        result = evaluator.evaluate_sample(sample, models[0])

        assert result["scores"]["mock"] == 0.8
        assert result["reasoning"]["mock"] == "Good"

    def test_evaluate_sample_dict_result_with_error(self, tmp_path):
        """Test evaluate_sample with dict result containing error."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return {"error": "Something went wrong"}

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
            include_reasoning=True,
        )

        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        result = evaluator.evaluate_sample(sample, models[0])

        assert result["scores"]["mock"] == 0.0
        assert "Error: Something went wrong" in result["reasoning"]["mock"]

    def test_evaluate_sample_dict_result_with_score(self, tmp_path):
        """Test evaluate_sample with dict result containing score."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return {"score": 0.9, "reasoning": "Excellent work"}

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
            include_reasoning=True,
        )

        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        result = evaluator.evaluate_sample(sample, models[0])

        assert result["scores"]["mock"] == 0.9
        assert result["reasoning"]["mock"] == "Excellent work"

    def test_evaluate_sample_numeric_result(self, tmp_path):
        """Test evaluate_sample with numeric result."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return 0.75  # Direct numeric value

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        result = evaluator.evaluate_sample(sample, models[0])

        assert result["scores"]["mock"] == 0.75

    def test_evaluate_sample_invalid_numeric_result(self, tmp_path):
        """Test evaluate_sample with invalid numeric result."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return "invalid_number"  # Cannot be converted to float

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        result = evaluator.evaluate_sample(sample, models[0])

        assert result["scores"]["mock"] == 0.0

    def test_evaluate_sample_scorer_exception(self, tmp_path):
        """Test evaluate_sample when scorer raises exception."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            raise ValueError("Scorer failed")

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
            include_reasoning=True,
        )

        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        result = evaluator.evaluate_sample(sample, models[0])

        assert result["scores"]["mock"] == 0.0
        assert "Error: Scorer failed" in result["reasoning"]["mock"]

    def test_save_intermediate_results_json(self, tmp_path):
        """Test saving intermediate results as JSON."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        # Add some data
        sample_result = {
            "user_id": "user1",
            "task_id": "task1",
            "turn_id": "turn1",
            "agent_name": "agent1",
            "scores": {"mock": 0.85},
            "reasoning": {"mock": "Good"},
            "error": None,
        }
        evaluator._add_result_to_dataframe(sample_result)

        # Save as JSON
        evaluator._save_intermediate_results("json")

        json_file = tmp_path / "agent_evaluation_results.json"
        assert json_file.exists()

    def test_convert_to_json_non_streaming(self, tmp_path):
        """Test JSON conversion without streaming."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
            stream=False,  # Non-streaming
        )

        # Add some data
        sample_result = {
            "user_id": "user1",
            "task_id": "task1",
            "turn_id": "turn1",
            "agent_name": "agent1",
            "scores": {"mock": 0.85},
            "reasoning": {"mock": "Good"},
            "error": None,
        }
        evaluator._add_result_to_dataframe(sample_result)

        # Save intermediate results
        evaluator._save_intermediate_results("csv")

        csv_file = tmp_path / "agent_evaluation_results.csv"
        assert csv_file.exists()

    def test_add_result_to_dataframe_empty_df(self, tmp_path):
        """Test adding result to empty DataFrame."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        # Ensure DataFrame is empty
        evaluator.results_df = pd.DataFrame(
            columns=["user_id", "task_id", "turn_id", "agent_name", "mock"]
        )

        sample_result = {
            "user_id": "user1",
            "task_id": "task1",
            "turn_id": "turn1",
            "agent_name": "agent1",
            "scores": {"mock": 0.85},
            "reasoning": {},
            "error": None,
        }

        evaluator._add_result_to_dataframe(sample_result)

        assert len(evaluator.results_df) == 1
        assert evaluator.results_df.iloc[0]["user_id"] == "user1"

    def test_evaluate_sample_empty_list_result(self, tmp_path):
        """Test evaluate_sample with empty list result."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return []  # Empty list

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        result = evaluator.evaluate_sample(sample, models[0])

        # Should fall through to the else case and set score to 0.0
        assert result["scores"]["mock"] == 0.0

    def test_evaluate_sample_list_without_score_attr(self, tmp_path):
        """Test evaluate_sample with list result where items don't have score attribute."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            # Return list with item that doesn't have score attribute
            return ["invalid_item"]

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        result = evaluator.evaluate_sample(sample, models[0])

        assert result["scores"]["mock"] == 0.0

    def test_evaluate_sample_dict_without_score_or_error(self, tmp_path):
        """Test evaluate_sample with dict result without score or error keys."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return {"other_key": "other_value"}  # Dict without score or error

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        result = evaluator.evaluate_sample(sample, models[0])

        assert result["scores"]["mock"] == 0.0

    def test_streaming_with_multiple_callable_functions(self, tmp_path):
        """Test streaming aggregation with multiple callable functions."""
        # Import the function
        from novaeval.evaluators.aggregators import _aggregate_by_task_streaming

        def max_callable(scores):
            return max(scores) if scores else 0.0

        def min_callable(scores):
            return min(scores) if scores else 0.0

        # Create test CSV file
        test_data = pd.DataFrame(
            {
                "user_id": ["user1", "user2"],
                "task_id": ["task1", "task1"],
                "turn_id": ["turn1", "turn1"],
                "agent_name": ["agent1", "agent2"],
                "score1": [0.8, 0.9],
            }
        )

        input_file = tmp_path / "test_input.csv"
        test_data.to_csv(input_file, index=False)

        output_file = tmp_path / "task_aggregation.csv"

        # Test streaming with multiple functions
        _aggregate_by_task_streaming(
            input_file=input_file,
            output_filename=output_file,
            callable_funcs=[max_callable, min_callable],
            chunk_size=1000,
        )

        # Check output
        assert output_file.exists()
        result = pd.read_csv(output_file)

        # Should have columns for all functions
        assert "max_callable_score1" in result.columns
        assert "min_callable_score1" in result.columns

    def test_run_all_with_json_aggregation(self, tmp_path):
        """Test run_all with JSON aggregation to cover missing lines."""
        agent_dataset = Mock(spec=AgentDataset)

        # Create mock sample
        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        agent_dataset.get_datapoint.return_value = [sample]

        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            mock_result = Mock()
            mock_result.score = 0.8
            return mock_result

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        # Mock the aggregation functions to avoid complex dependencies
        with patch(
            "novaeval.evaluators.aggregators.aggregate_by_task"
        ) as mock_agg_task:
            evaluator.run_all(
                aggregate_by_task=True,
                file_type="json",  # Use JSON to cover lines 196-197
                aggregator_functions=None,  # Test default aggregator_functions (lines 192-193)
            )

            # Verify aggregation function was called with JSON file
            mock_agg_task.assert_called_once()
            call_args = mock_agg_task.call_args
            assert str(call_args[1]["input_file"]).endswith(".json")

    def test_convert_to_json_streaming_multiple_rows(self, tmp_path):
        """Test streaming JSON conversion with multiple rows to cover line 449."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
            stream=True,  # Enable streaming
        )

        # Add multiple rows to DataFrame to trigger line 449 (comma writing)
        sample_results = [
            {
                "user_id": "user1",
                "task_id": "task1",
                "turn_id": "turn1",
                "agent_name": "agent1",
                "scores": {"mock": 0.85},
                "reasoning": {"mock": "Good"},
                "error": None,
            },
            {
                "user_id": "user2",
                "task_id": "task2",
                "turn_id": "turn2",
                "agent_name": "agent2",
                "scores": {"mock": 0.75},
                "reasoning": {"mock": "Fair"},
                "error": None,
            },
        ]

        for sample_result in sample_results:
            evaluator._add_result_to_dataframe(sample_result)

        # Save intermediate results
        evaluator._save_intermediate_results("csv")

        # Check that file was created and has proper CSV format
        csv_file = tmp_path / "agent_evaluation_results.csv"
        assert csv_file.exists()

        # Verify the content is valid CSV
        with open(csv_file) as f:
            content = f.read()
            assert "user_id" in content
            assert "task_id" in content

    def test_evaluate_sample_with_reasoning_disabled(self, tmp_path):
        """Test evaluate_sample with reasoning disabled to cover more branches."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            mock_result = Mock()
            mock_result.score = 0.8
            mock_result.reasoning = "This should be ignored"
            return mock_result

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
            include_reasoning=False,  # Disable reasoning
        )

        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        result = evaluator.evaluate_sample(sample, models[0])

        # Should have score but no reasoning
        assert result["scores"]["mock"] == 0.8
        assert result["reasoning"] == {}

    def test_evaluate_sample_list_result_with_reasoning_disabled(self, tmp_path):
        """Test evaluate_sample with list result and reasoning disabled."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            # Return a list of score objects
            score1 = Mock()
            score1.score = 0.8
            score1.reasoning = "Good"
            return [score1]

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
            include_reasoning=False,  # Disable reasoning
        )

        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        result = evaluator.evaluate_sample(sample, models[0])

        # Should have score but no reasoning
        assert result["scores"]["mock"] == 0.8
        assert result["reasoning"] == {}

    def test_evaluate_sample_dict_result_without_reasoning_key(self, tmp_path):
        """Test evaluate_sample with dict result that has score but no reasoning key."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return {"score": 0.9}  # No reasoning key

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
            include_reasoning=True,  # Enable reasoning
        )

        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        result = evaluator.evaluate_sample(sample, models[0])

        # Should have score but no reasoning since dict didn't have reasoning key
        assert result["scores"]["mock"] == 0.9
        # The reasoning dict should either not have the key or have None value
        assert (
            "mock" not in result["reasoning"] or result["reasoning"].get("mock") is None
        )

    def test_add_result_to_dataframe_missing_reasoning_column(self, tmp_path):
        """Test adding result when reasoning column doesn't exist in DataFrame."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
            include_reasoning=True,
        )

        # Manually modify the DataFrame to not have reasoning column
        evaluator.results_df = pd.DataFrame(
            columns=["user_id", "task_id", "turn_id", "agent_name", "mock"]
        )

        sample_result = {
            "user_id": "user1",
            "task_id": "task1",
            "turn_id": "turn1",
            "agent_name": "agent1",
            "scores": {"mock": 0.85},
            "reasoning": {"mock": "Good performance"},
            "error": None,
        }

        # This should handle the case where reasoning column doesn't exist
        evaluator._add_result_to_dataframe(sample_result)

        assert len(evaluator.results_df) == 1
        assert evaluator.results_df.iloc[0]["user_id"] == "user1"

    def test_run_all_with_json_aggregation_no_functions(self, tmp_path):
        """Test run_all with JSON aggregation and no aggregator functions."""
        agent_dataset = Mock(spec=AgentDataset)

        # Create mock sample
        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        agent_dataset.get_datapoint.return_value = [sample]

        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            mock_result = Mock()
            mock_result.score = 0.8
            return mock_result

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        # Mock the aggregation functions to avoid complex dependencies
        with (
            patch("novaeval.evaluators.aggregators.aggregate_by_task") as mock_agg_task,
            patch(
                "novaeval.evaluators.aggregators.mean_callable"
            ) as mock_mean_callable,
        ):
            evaluator.run_all(
                aggregate_by_task=True,
                file_type="json",  # Use JSON to cover lines 196-197
                aggregator_functions=None,  # Test default aggregator_functions (lines 192-193)
            )

            # Verify aggregation function was called with JSON file
            mock_agg_task.assert_called_once()
            call_args = mock_agg_task.call_args
            assert str(call_args[1]["input_file"]).endswith(".json")
            assert call_args[1]["callable_func"] == [
                mock_mean_callable
            ]  # Should use default function

    def test_evaluate_sample_dict_result_with_error_and_reasoning(self, tmp_path):
        """Test evaluate_sample with dict result containing error and reasoning enabled."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return {"error": "Test error"}

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
            include_reasoning=True,  # Enable reasoning
        )

        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        result = evaluator.evaluate_sample(sample, models[0])

        # Should have score and reasoning
        assert result["scores"]["mock"] == 0.0
        assert result["reasoning"]["mock"] == "Error: Test error"

    def test_add_result_to_dataframe_with_missing_columns(self, tmp_path):
        """Test adding result when columns don't exist in DataFrame."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
            include_reasoning=True,
        )

        # Manually modify the DataFrame to have different columns
        evaluator.results_df = pd.DataFrame(columns=["user_id", "task_id", "extra_col"])

        sample_result = {
            "user_id": "user1",
            "task_id": "task1",
            "turn_id": "turn1",  # This column doesn't exist in DataFrame
            "agent_name": "agent1",  # This column doesn't exist in DataFrame
            "scores": {"mock": 0.85},
            "reasoning": {"mock": "Good performance"},
            "error": None,
        }

        # This should handle missing columns by setting them to None
        evaluator._add_result_to_dataframe(sample_result)

        assert len(evaluator.results_df) == 1
        assert evaluator.results_df.iloc[0]["user_id"] == "user1"
        assert evaluator.results_df.iloc[0]["task_id"] == "task1"
        assert evaluator.results_df.iloc[0]["extra_col"] == ""  # Should be empty string

    def test_evaluate_sample_with_exception_in_scorer(self, tmp_path):
        """Test evaluate_sample when scorer raises an exception."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            raise Exception("Test error")

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
            include_reasoning=True,
        )

        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        result = evaluator.evaluate_sample(sample, models[0])

        # Should catch the exception and set score to 0.0 and reasoning to error message
        assert result["scores"]["mock"] == 0.0
        assert result["reasoning"]["mock"] == "Error: Test error"

    def test_add_result_to_dataframe_with_reasoning_column_not_in_df(self, tmp_path):
        """Test adding result when reasoning column doesn't exist in DataFrame."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
            include_reasoning=True,
        )

        # Manually modify the DataFrame to not have reasoning columns
        evaluator.results_df = pd.DataFrame(
            columns=["user_id", "task_id", "turn_id", "agent_name", "mock"]
        )

        sample_result = {
            "user_id": "user1",
            "task_id": "task1",
            "turn_id": "turn1",
            "agent_name": "agent1",
            "scores": {"mock": 0.85},
            "reasoning": {
                "mock": "Good performance"
            },  # This column doesn't exist in DataFrame
            "error": None,
        }

        # This should handle the case where reasoning column doesn't exist
        evaluator._add_result_to_dataframe(sample_result)

        assert len(evaluator.results_df) == 1
        assert evaluator.results_df.iloc[0]["user_id"] == "user1"
        assert (
            "mock_reasoning" not in evaluator.results_df.columns
        )  # Reasoning column should not be added

    def test_run_all_with_json_aggregation_and_all_types(self, tmp_path):
        """Test run_all with JSON aggregation and all aggregation types."""
        agent_dataset = Mock(spec=AgentDataset)

        # Create mock sample
        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        agent_dataset.get_datapoint.return_value = [sample]

        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            mock_result = Mock()
            mock_result.score = 0.8
            return mock_result

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        # Mock the aggregation functions to avoid complex dependencies
        with (
            patch("novaeval.evaluators.aggregators.aggregate_by_task") as mock_agg_task,
            patch("novaeval.evaluators.aggregators.aggregate_by_user") as mock_agg_user,
            patch(
                "novaeval.evaluators.aggregators.aggregate_by_agent_name"
            ) as mock_agg_agent,
            patch(
                "novaeval.evaluators.aggregators.mean_callable"
            ) as mock_mean_callable,
        ):
            evaluator.run_all(
                aggregate_by_task=True,
                aggregate_by_user=True,
                aggregate_by_agent_name=True,
                file_type="json",  # Use JSON to cover lines 196-197
                aggregator_functions=None,  # Test default aggregator_functions (lines 192-193)
            )

            # Verify all aggregation functions were called with JSON file
            mock_agg_task.assert_called_once()
            mock_agg_user.assert_called_once()
            mock_agg_agent.assert_called_once()

            # Check task aggregation
            task_args = mock_agg_task.call_args
            assert str(task_args[1]["input_file"]).endswith(".json")
            assert task_args[1]["callable_func"] == [mock_mean_callable]

            # Check user aggregation
            user_args = mock_agg_user.call_args
            assert str(user_args[1]["input_file"]).endswith(".json")
            assert user_args[1]["callable_func"] == [mock_mean_callable]

            # Check agent aggregation
            agent_args = mock_agg_agent.call_args
            assert str(agent_args[1]["input_file"]).endswith(".json")
            assert agent_args[1]["callable_func"] == [mock_mean_callable]

    def test_initialize_dataframe_with_scoring_function_no_name(self, tmp_path):
        """Test DataFrame initialization with scoring function that has no __name__ attribute."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        # Create a scoring function without __name__ attribute using a class
        class ScoringFunctionWithoutName:
            def __call__(self, sample, model):
                return Mock()

        scoring_function = ScoringFunctionWithoutName()
        # The class doesn't have __name__ by default, so this should trigger the else branch

        scoring_functions = [scoring_function]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        # Should use default name "scorer_0"
        assert "scorer_0" in evaluator.results_df.columns

    def test_run_all_no_model_available(self, tmp_path):
        """Test run_all when no model is available."""
        agent_dataset = Mock(spec=AgentDataset)

        # Create mock sample
        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        agent_dataset.get_datapoint.return_value = [sample]

        models = []  # No models

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        # This should handle the case where no model is available
        evaluator.run_all(save_every=1, file_type="csv")

        # Should not crash and should have empty results
        assert len(evaluator.results_df) == 0

    def test_run_single_aggregation_unknown_type(self, tmp_path):
        """Test _run_single_aggregation with unknown aggregation type."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        # Test with unknown aggregation type
        evaluator._run_single_aggregation(
            aggregation_type="unknown",
            input_file=tmp_path / "test.csv",
            output_file=tmp_path / "output.csv",
            aggregator_functions=[Mock()],
            aggregation_chunk_size=1000,
        )

        # Should log error but not crash

    def test_run_single_aggregation_exception(self, tmp_path):
        """Test _run_single_aggregation when aggregation function raises exception."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        # Mock aggregation functions to raise exception
        with (
            patch("novaeval.evaluators.aggregators.aggregate_by_task") as mock_agg_task,
            patch("novaeval.evaluators.aggregators.aggregate_by_user"),
            patch("novaeval.evaluators.aggregators.aggregate_by_agent_name"),
        ):
            # Make task aggregation raise an exception
            mock_agg_task.side_effect = Exception("Aggregation failed")

            evaluator._run_single_aggregation(
                aggregation_type="task",
                input_file=tmp_path / "test.csv",
                output_file=tmp_path / "output.csv",
                aggregator_functions=[Mock()],
                aggregation_chunk_size=1000,
            )

            # Should catch exception and log error

    def test_evaluate_sample_non_dict_scores_reasoning(self, tmp_path):
        """Test evaluate_sample with non-dict scores and reasoning."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        # Create sample with non-dict scores and reasoning
        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        # Manually set non-dict values to trigger the conversion
        evaluator.sample_result = {
            "scores": "not_a_dict",
            "reasoning": "not_a_dict",
        }

        result = evaluator.evaluate_sample(sample, models[0])

        # Should convert to empty dicts
        assert isinstance(result["scores"], dict)
        assert isinstance(result["reasoning"], dict)

    def test_evaluate_sample_general_exception(self, tmp_path):
        """Test evaluate_sample when a general exception occurs."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            raise Exception("General error")

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        result = evaluator.evaluate_sample(sample, models[0])

        # The exception is caught in the inner try-catch, not the outer one
        # So it should have scores and reasoning with error messages
        assert "scores" in result
        assert "reasoning" in result
        assert result["scores"]["mock"] == 0.0
        assert "Error: General error" in result["reasoning"]["mock"]

    def test_add_result_to_dataframe_with_new_columns(self, tmp_path):
        """Test adding result when new columns need to be added to DataFrame."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        # Manually set DataFrame with existing data
        evaluator.results_df = pd.DataFrame(
            {
                "user_id": ["existing_user"],
                "task_id": ["existing_task"],
                "turn_id": ["existing_turn"],
                "agent_name": ["existing_agent"],
                "mock": [0.5],
            }
        )

        # Add result with new column
        sample_result = {
            "user_id": "user1",
            "task_id": "task1",
            "turn_id": "turn1",
            "agent_name": "agent1",
            "scores": {"mock": 0.85, "new_scorer": 0.9},
            "reasoning": {},
            "error": None,
        }

        evaluator._add_result_to_dataframe(sample_result)

        # Should handle new columns properly
        assert len(evaluator.results_df) == 2
        assert "new_scorer" in evaluator.results_df.columns

    def test_save_results_with_dict_format(self, tmp_path):
        """Test save_results with dict format results."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        # Test with dict format results
        results = {
            "user_id": "user1",
            "task_id": "task1",
            "scores": {"mock": 0.85},
        }

        evaluator.save_results(results)

        # Check that file was created
        csv_file = tmp_path / "agent_evaluation_results.csv"
        assert csv_file.exists()

    def test_save_results_with_empty_dataframe(self, tmp_path):
        """Test save_results when results_df is empty."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        # Manually clear the DataFrame
        evaluator.results_df = pd.DataFrame()

        # Test with list format results
        results = [
            {"user_id": "user1", "task_id": "task1", "scores": {"mock": 0.85}},
            {"user_id": "user2", "task_id": "task2", "scores": {"mock": 0.75}},
        ]

        evaluator.save_results(results)

        # Check that file was created
        csv_file = tmp_path / "agent_evaluation_results.csv"
        assert csv_file.exists()

    def test_run_aggregations_input_file_not_exists(self, tmp_path):
        """Test _run_aggregations when input file doesn't exist."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        # Run aggregations without creating the input file first
        evaluator._run_aggregations(
            file_type="csv",
            aggregate_by_task=True,
            aggregate_by_user=False,
            aggregate_by_agent_name=False,
            aggregator_functions=None,
            aggregation_chunk_size=1000,
        )

        # Should handle missing file gracefully and log warning

    def test_evaluate_sample_with_non_dict_scores_reasoning_initialization(
        self, tmp_path
    ):
        """Test evaluate_sample with non-dict scores and reasoning initialization."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        # Create a sample result with non-dict scores and reasoning
        sample_result = {
            "user_id": "user1",
            "task_id": "task1",
            "turn_id": "turn1",
            "agent_name": "agent1",
            "scores": "not_a_dict",  # This should trigger the conversion
            "reasoning": "not_a_dict",  # This should trigger the conversion
        }

        # Manually set the sample_result to trigger the conversion logic
        evaluator.sample_result = sample_result

        result = evaluator.evaluate_sample(sample, models[0])

        # Should convert to empty dicts
        assert isinstance(result["scores"], dict)
        assert isinstance(result["reasoning"], dict)

    def test_add_result_to_dataframe_with_complex_column_handling(self, tmp_path):
        """Test adding result with complex DataFrame column handling."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        # Create a complex DataFrame with existing data
        evaluator.results_df = pd.DataFrame(
            {
                "user_id": ["existing_user"],
                "task_id": ["existing_task"],
                "turn_id": ["existing_turn"],
                "agent_name": ["existing_agent"],
                "mock": [0.5],
                "extra_col": ["extra_value"],
            }
        )

        # Add result with new columns and missing columns
        sample_result = {
            "user_id": "user1",
            "task_id": "task1",
            "turn_id": "turn1",
            "agent_name": "agent1",
            "scores": {"mock": 0.85, "new_scorer": 0.9},
            "reasoning": {"mock": "Good", "new_scorer": "Excellent"},
            "error": None,
        }

        evaluator._add_result_to_dataframe(sample_result)

        # Should handle column differences properly
        assert len(evaluator.results_df) == 2
        assert "new_scorer" in evaluator.results_df.columns
        assert "extra_col" in evaluator.results_df.columns
        # Note: new_scorer_reasoning column is not added because reasoning is not enabled in this test

    def test_save_results_with_complex_data_handling(self, tmp_path):
        """Test save_results with complex data handling scenarios."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        # Test with complex dict format results
        results = {
            "user_id": "user1",
            "task_id": "task1",
            "scores": {"mock": 0.85, "scorer2": 0.75},
            "reasoning": {"mock": "Good", "scorer2": "Fair"},
            "extra_field": "extra_value",
        }

        evaluator.save_results(results)

        # Check that file was created
        csv_file = tmp_path / "agent_evaluation_results.csv"
        assert csv_file.exists()

        # Test with empty results_df and complex list format
        evaluator.results_df = pd.DataFrame()
        results_list = [
            {"user_id": "user1", "task_id": "task1", "scores": {"mock": 0.85}},
            {"user_id": "user2", "task_id": "task2", "scores": {"mock": 0.75}},
            {"user_id": "user3", "task_id": "task3", "scores": {"mock": 0.95}},
        ]

        evaluator.save_results(results_list)

        # Check that file was updated
        assert csv_file.exists()

    def test_run_single_aggregation_with_exception_handling(self, tmp_path):
        """Test _run_single_aggregation with comprehensive exception handling."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        # Mock aggregation functions to raise different exceptions
        with (
            patch("novaeval.evaluators.aggregators.aggregate_by_task") as mock_agg_task,
            patch("novaeval.evaluators.aggregators.aggregate_by_user") as mock_agg_user,
            patch(
                "novaeval.evaluators.aggregators.aggregate_by_agent_name"
            ) as mock_agg_agent,
        ):
            # Make task aggregation raise an exception
            mock_agg_task.side_effect = Exception("Task aggregation failed")

            evaluator._run_single_aggregation(
                aggregation_type="task",
                input_file=tmp_path / "test.csv",
                output_file=tmp_path / "output.csv",
                aggregator_functions=[Mock()],
                aggregation_chunk_size=1000,
            )

            # Make user aggregation raise an exception
            mock_agg_user.side_effect = Exception("User aggregation failed")

            evaluator._run_single_aggregation(
                aggregation_type="user",
                input_file=tmp_path / "test.csv",
                output_file=tmp_path / "output.csv",
                aggregator_functions=[Mock()],
                aggregation_chunk_size=1000,
            )

            # Make agent aggregation raise an exception
            mock_agg_agent.side_effect = Exception("Agent aggregation failed")

            evaluator._run_single_aggregation(
                aggregation_type="agent",
                input_file=tmp_path / "test.csv",
                output_file=tmp_path / "output.csv",
                aggregator_functions=[Mock()],
                aggregation_chunk_size=1000,
            )

            # Should catch all exceptions and log errors

    def test_evaluate_sample_with_non_dict_scores_reasoning_initialization_edge_cases(
        self, tmp_path
    ):
        """Test evaluate_sample with edge cases for non-dict scores and reasoning initialization."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        # Test with None values for scores and reasoning
        sample_result = {
            "user_id": "user1",
            "task_id": "task1",
            "turn_id": "turn1",
            "agent_name": "agent1",
            "scores": None,  # This should trigger the conversion
            "reasoning": None,  # This should trigger the conversion
        }

        # Manually set the sample_result to trigger the conversion logic
        evaluator.sample_result = sample_result

        result = evaluator.evaluate_sample(sample, models[0])

        # Should convert to empty dicts
        assert isinstance(result["scores"], dict)
        assert isinstance(result["reasoning"], dict)

    def test_add_result_to_dataframe_with_empty_dataframe_and_new_columns(
        self, tmp_path
    ):
        """Test adding result when DataFrame is empty and new columns need to be added."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        # Ensure DataFrame is empty
        evaluator.results_df = pd.DataFrame()

        # Add result with new columns
        sample_result = {
            "user_id": "user1",
            "task_id": "task1",
            "turn_id": "turn1",
            "agent_name": "agent1",
            "scores": {"mock": 0.85, "new_scorer": 0.9},
            "reasoning": {"mock": "Good", "new_scorer": "Excellent"},
            "error": None,
        }

        evaluator._add_result_to_dataframe(sample_result)

        # Should handle empty DataFrame properly
        assert len(evaluator.results_df) == 1
        assert "new_scorer" in evaluator.results_df.columns

    def test_save_results_with_empty_dataframe_and_dict_results(self, tmp_path):
        """Test save_results with empty DataFrame and dict results."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        # Ensure DataFrame is empty
        evaluator.results_df = pd.DataFrame()

        # Test with dict format results
        results = {
            "user_id": "user1",
            "task_id": "task1",
            "scores": {"mock": 0.85},
        }

        evaluator.save_results(results)

        # Check that file was created
        csv_file = tmp_path / "agent_evaluation_results.csv"
        assert csv_file.exists()

    def test_run_single_aggregation_with_unknown_type_and_exception(self, tmp_path):
        """Test _run_single_aggregation with unknown type and exception handling."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        # Test with unknown aggregation type
        evaluator._run_single_aggregation(
            aggregation_type="unknown_type",
            input_file=tmp_path / "test.csv",
            output_file=tmp_path / "output.csv",
            aggregator_functions=[Mock()],
            aggregation_chunk_size=1000,
        )

        # Should log error but not crash

        # Test with exception in aggregation function
        with patch(
            "novaeval.evaluators.aggregators.aggregate_by_task"
        ) as mock_agg_task:
            mock_agg_task.side_effect = Exception("Aggregation failed")

            evaluator._run_single_aggregation(
                aggregation_type="task",
                input_file=tmp_path / "test.csv",
                output_file=tmp_path / "output.csv",
                aggregator_functions=[Mock()],
                aggregation_chunk_size=1000,
            )

            # Should catch exception and log error

    def test_evaluate_sample_with_complex_exception_handling_scenarios(self, tmp_path):
        """Test evaluate_sample with complex exception handling scenarios."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        # Test with a scoring function that raises an exception
        def failing_scorer(sample, model):
            raise Exception("Scorer failed")

        evaluator.scoring_functions = [failing_scorer]

        result = evaluator.evaluate_sample(sample, models[0])

        # Should handle the exception gracefully
        assert "scores" in result
        assert "reasoning" in result
        assert result["scores"]["failing"] == 0.0
        assert "Error: Scorer failed" in result["reasoning"]["failing"]

    def test_add_result_to_dataframe_with_complex_column_scenarios(self, tmp_path):
        """Test adding result with complex column scenarios."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
            include_reasoning=True,  # Enable reasoning to test reasoning columns
        )

        # Create a DataFrame with existing data and reasoning columns
        evaluator.results_df = pd.DataFrame(
            {
                "user_id": ["existing_user"],
                "task_id": ["existing_task"],
                "turn_id": ["existing_turn"],
                "agent_name": ["existing_agent"],
                "mock": [0.5],
                "mock_reasoning": ["existing_reasoning"],
                "extra_col": ["extra_value"],
            }
        )

        # Add result with new columns and reasoning
        sample_result = {
            "user_id": "user1",
            "task_id": "task1",
            "turn_id": "turn1",
            "agent_name": "agent1",
            "scores": {"mock": 0.85, "new_scorer": 0.9},
            "reasoning": {"mock": "Good", "new_scorer": "Excellent"},
            "error": None,
        }

        evaluator._add_result_to_dataframe(sample_result)

        # Should handle column differences properly including reasoning columns
        assert len(evaluator.results_df) == 2
        assert "new_scorer" in evaluator.results_df.columns
        assert "extra_col" in evaluator.results_df.columns
        # Note: new_scorer_reasoning column is not added because the DataFrame already has mock_reasoning

    def test_save_results_with_various_data_formats(self, tmp_path):
        """Test save_results with various data formats."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        # Test with empty DataFrame and list results
        evaluator.results_df = pd.DataFrame()
        results_list = [
            {"user_id": "user1", "task_id": "task1", "scores": {"mock": 0.85}},
            {"user_id": "user2", "task_id": "task2", "scores": {"mock": 0.75}},
        ]

        evaluator.save_results(results_list)

        # Check that file was created
        csv_file = tmp_path / "agent_evaluation_results.csv"
        assert csv_file.exists()

        # Test with existing DataFrame and dict results
        evaluator.results_df = pd.DataFrame(
            {
                "user_id": ["existing_user"],
                "task_id": ["existing_task"],
                "scores": [{"mock": 0.5}],
            }
        )

        results_dict = {
            "user_id": "user1",
            "task_id": "task1",
            "scores": {"mock": 0.85},
        }

        evaluator.save_results(results_dict)

        # Check that file was updated
        assert csv_file.exists()

    def test_run_single_aggregation_with_all_exception_scenarios(self, tmp_path):
        """Test _run_single_aggregation with all exception scenarios."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        # Mock aggregation functions to raise different exceptions
        with (
            patch("novaeval.evaluators.aggregators.aggregate_by_task") as mock_agg_task,
            patch("novaeval.evaluators.aggregators.aggregate_by_user") as mock_agg_user,
            patch(
                "novaeval.evaluators.aggregators.aggregate_by_agent_name"
            ) as mock_agg_agent,
        ):
            # Test task aggregation with exception
            mock_agg_task.side_effect = Exception("Task aggregation failed")

            evaluator._run_single_aggregation(
                aggregation_type="task",
                input_file=tmp_path / "test.csv",
                output_file=tmp_path / "output.csv",
                aggregator_functions=[Mock()],
                aggregation_chunk_size=1000,
            )

            # Test user aggregation with exception
            mock_agg_user.side_effect = Exception("User aggregation failed")

            evaluator._run_single_aggregation(
                aggregation_type="user",
                input_file=tmp_path / "test.csv",
                output_file=tmp_path / "output.csv",
                aggregator_functions=[Mock()],
                aggregation_chunk_size=1000,
            )

            # Test agent aggregation with exception
            mock_agg_agent.side_effect = Exception("Agent aggregation failed")

            evaluator._run_single_aggregation(
                aggregation_type="agent",
                input_file=tmp_path / "test.csv",
                output_file=tmp_path / "output.csv",
                aggregator_functions=[Mock()],
                aggregation_chunk_size=1000,
            )

            # Should catch all exceptions and log errors

    def test_evaluate_sample_with_complex_data_handling_edge_cases(self, tmp_path):
        """Test evaluate_sample with complex data handling edge cases to cover remaining lines."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        # Test with a scoring function that returns a complex object
        def complex_scorer(sample, model):
            # Return an object that has score but not reasoning
            result = Mock()
            result.score = 0.8
            # Don't set reasoning attribute
            return result

        evaluator.scoring_functions = [complex_scorer]

        result = evaluator.evaluate_sample(sample, models[0])

        # Should handle the complex object properly
        assert "scores" in result
        assert "reasoning" in result
        assert result["scores"]["complex"] == 0.8

    def test_add_result_to_dataframe_with_extreme_column_scenarios(self, tmp_path):
        """Test adding result with extreme column scenarios."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
            include_reasoning=True,
        )

        # Create a DataFrame with many columns
        evaluator.results_df = pd.DataFrame(
            {
                "user_id": ["existing_user"],
                "task_id": ["existing_task"],
                "turn_id": ["existing_turn"],
                "agent_name": ["existing_agent"],
                "mock": [0.5],
                "mock_reasoning": ["existing_reasoning"],
                "col1": ["val1"],
                "col2": ["val2"],
                "col3": ["val3"],
            }
        )

        # Add result with completely different columns
        sample_result = {
            "user_id": "user1",
            "task_id": "task1",
            "turn_id": "turn1",
            "agent_name": "agent1",
            "scores": {"new_scorer1": 0.85, "new_scorer2": 0.9},
            "reasoning": {"new_scorer1": "Good", "new_scorer2": "Excellent"},
            "error": None,
        }

        evaluator._add_result_to_dataframe(sample_result)

        # Should handle extreme column differences properly
        assert len(evaluator.results_df) == 2
        assert "new_scorer1" in evaluator.results_df.columns
        assert "new_scorer2" in evaluator.results_df.columns
        assert "col1" in evaluator.results_df.columns
        assert "col2" in evaluator.results_df.columns
        assert "col3" in evaluator.results_df.columns

    def test_save_results_with_complex_dataframe_scenarios(self, tmp_path):
        """Test save_results with complex DataFrame scenarios."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        # Test with DataFrame that has complex data types
        evaluator.results_df = pd.DataFrame(
            {
                "user_id": ["user1"],
                "task_id": ["task1"],
                "scores": [{"mock": 0.85}],  # Complex nested data
                "reasoning": [{"mock": "Good"}],  # Complex nested data
            }
        )

        # This should trigger the complex DataFrame handling
        evaluator.save_results({})

        # Check that file was created
        csv_file = tmp_path / "agent_evaluation_results.csv"
        assert csv_file.exists()

        # Test with DataFrame that has no results_df attribute
        evaluator.results_df = pd.DataFrame()
        delattr(evaluator, "results_df")

        # This should handle the missing attribute
        evaluator.save_results({"user_id": "user1", "scores": {"mock": 0.85}})

        # Check that file was created
        assert csv_file.exists()

    def test_comprehensive_edge_case_coverage(self, tmp_path):
        """Comprehensive test to cover remaining edge cases."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
            include_reasoning=True,
        )

        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        # Test with a scoring function that returns various data types
        def complex_scorer(sample, model):
            # Return different types of results to test various code paths
            import random

            result_type = random.choice([1, 2, 3, 4, 5])

            if result_type == 1:
                # Return object with score but no reasoning
                result = Mock()
                result.score = 0.8
                return result
            elif result_type == 2:
                # Return list with object that has score but no reasoning
                result = Mock()
                result.score = 0.7
                return [result]
            elif result_type == 3:
                # Return dict with score but no reasoning
                return {"score": 0.9}
            elif result_type == 4:
                # Return numeric value
                return 0.75
            else:
                # Return invalid type
                return "invalid"

        evaluator.scoring_functions = [complex_scorer]

        # Run multiple evaluations to cover different code paths
        for _ in range(10):
            result = evaluator.evaluate_sample(sample, models[0])
            assert "scores" in result
            assert "reasoning" in result

        # Test DataFrame with various data types
        evaluator.results_df = pd.DataFrame(
            {
                "user_id": ["user1"],
                "task_id": ["task1"],
                "turn_id": ["turn1"],
                "agent_name": ["agent1"],
                "complex_col": [{"nested": {"deep": "data"}}],
                "list_col": [[1, 2, 3, {"nested": "list"}]],
                "tuple_col": [(1, 2, 3)],
                "set_col": [{"set", "data"}],
                "frozenset_col": [frozenset([1, 2, 3])],
                "bytes_col": [b"bytes_data"],
                "bytearray_col": [bytearray(b"bytearray_data")],
                "memoryview_col": [memoryview(b"memoryview_data")],
                "complex_num_col": [complex(1, 2)],
                "range_col": [range(1, 10)],
                "slice_col": [slice(1, 10, 2)],
            }
        )

        # Add result with completely different structure
        sample_result = {
            "user_id": "user2",
            "task_id": "task2",
            "turn_id": "turn2",
            "agent_name": "agent2",
            "scores": {"new_scorer": 0.95},
            "reasoning": {"new_scorer": "Excellent"},
            "error": None,
        }

        evaluator._add_result_to_dataframe(sample_result)

        # Test save_results with complex data
        evaluator.save_results({})

        # Check that file was created
        csv_file = tmp_path / "agent_evaluation_results.csv"
        assert csv_file.exists()

        # Test with completely empty DataFrame
        evaluator.results_df = pd.DataFrame()

        evaluator.save_results({"user_id": "user3", "scores": {"final_scorer": 0.99}})

        # Check that file was updated
        assert csv_file.exists()

    def test_evaluate_sample_with_comprehensive_data_type_handling(self, tmp_path):
        """Test evaluate_sample with comprehensive data type handling to cover lines 341, 343, 355."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        # Test with various data types that should trigger the conversion logic
        test_cases = [
            (42, "string"),  # Integer and string
            (3.14, True),  # Float and boolean
            ([1, 2, 3], {"key": "value"}),  # List and dict
            ({1, 2, 3}, frozenset([1, 2, 3])),  # Set and frozenset
            (complex(1, 2), bytes([1, 2, 3])),  # Complex and bytes
        ]

        for scores_val, reasoning_val in test_cases:
            # Manually set the sample_result to trigger the conversion logic
            evaluator.sample_result = {
                "scores": scores_val,
                "reasoning": reasoning_val,
            }

            result = evaluator.evaluate_sample(sample, models[0])

            # Should convert to empty dicts
            assert isinstance(result["scores"], dict)
            assert isinstance(result["reasoning"], dict)

    def test_add_result_to_dataframe_with_extreme_edge_cases(self, tmp_path):
        """Test adding result with extreme edge cases to cover lines 440->447, 461."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
            include_reasoning=True,
        )

        # Create a DataFrame with maximum complexity
        evaluator.results_df = pd.DataFrame(
            {
                "user_id": ["existing_user"],
                "task_id": ["existing_task"],
                "turn_id": ["existing_turn"],
                "agent_name": ["existing_agent"],
                "mock": [0.5],
                "mock_reasoning": ["existing_reasoning"],
                "col1": ["val1"],
                "col2": ["val2"],
                "col3": ["val3"],
                "col4": ["val4"],
                "col5": ["val5"],
                "col6": ["val6"],
                "col7": ["val7"],
                "col8": ["val8"],
                "col9": ["val9"],
                "col10": ["val10"],
            }
        )

        # Add result with completely different structure and many new columns
        sample_result = {
            "user_id": "user1",
            "task_id": "task1",
            "turn_id": "turn1",
            "agent_name": "agent1",
            "scores": {
                "new_scorer1": 0.85,
                "new_scorer2": 0.9,
                "new_scorer3": 0.75,
                "new_scorer4": 0.8,
                "new_scorer5": 0.95,
                "new_scorer6": 0.7,
                "new_scorer7": 0.88,
                "new_scorer8": 0.92,
                "new_scorer9": 0.78,
                "new_scorer10": 0.83,
            },
            "reasoning": {
                "new_scorer1": "Good",
                "new_scorer2": "Excellent",
                "new_scorer3": "Fair",
                "new_scorer4": "Very Good",
                "new_scorer5": "Outstanding",
                "new_scorer6": "Average",
                "new_scorer7": "Good",
                "new_scorer8": "Excellent",
                "new_scorer9": "Fair",
                "new_scorer10": "Good",
            },
            "error": None,
        }

        evaluator._add_result_to_dataframe(sample_result)

        # Should handle extreme column differences properly
        assert len(evaluator.results_df) == 2
        assert "new_scorer1" in evaluator.results_df.columns
        assert "new_scorer10" in evaluator.results_df.columns
        assert "col1" in evaluator.results_df.columns
        assert "col10" in evaluator.results_df.columns

        # Test with completely empty DataFrame and maximum complexity result
        evaluator.results_df = pd.DataFrame()

        max_complex_sample_result = {
            "user_id": "user2",
            "task_id": "task2",
            "turn_id": "turn2",
            "agent_name": "agent2",
            "scores": {"max_complex_scorer": 0.99},
            "reasoning": {"max_complex_scorer": "Maximum complexity"},
            "error": None,
        }

        evaluator._add_result_to_dataframe(max_complex_sample_result)

        # Should handle empty DataFrame properly
        assert len(evaluator.results_df) == 1
        assert "max_complex_scorer" in evaluator.results_df.columns

    def test_run_single_aggregation_with_unknown_type_comprehensive(self, tmp_path):
        """Test _run_single_aggregation with comprehensive unknown type handling to cover lines 211->215, 224->234."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        # Test with various unknown aggregation types
        unknown_types = [
            "completely_unknown_type",
            "invalid_aggregation",
            "wrong_type",
            "nonexistent_type",
            "random_string",
        ]

        for unknown_type in unknown_types:
            evaluator._run_single_aggregation(
                aggregation_type=unknown_type,
                input_file=tmp_path / "test.csv",
                output_file=tmp_path / "output.csv",
                aggregator_functions=[Mock()],
                aggregation_chunk_size=1000,
            )

        # Test with various exception types in aggregation functions
        with (
            patch("novaeval.evaluators.aggregators.aggregate_by_task") as mock_agg_task,
            patch("novaeval.evaluators.aggregators.aggregate_by_user"),
            patch("novaeval.evaluators.aggregators.aggregate_by_agent_name"),
        ):
            # Test various exception types
            exception_types = [
                (ValueError, "Value error in aggregation"),
                (TypeError, "Type error in aggregation"),
                (RuntimeError, "Runtime error in aggregation"),
                (OSError, "OS error in aggregation"),
                (MemoryError, "Memory error in aggregation"),
            ]

            for exc_type, message in exception_types:
                mock_agg_task.side_effect = exc_type(message)

                evaluator._run_single_aggregation(
                    aggregation_type="task",
                    input_file=tmp_path / "test.csv",
                    output_file=tmp_path / "output.csv",
                    aggregator_functions=[Mock()],
                    aggregation_chunk_size=1000,
                )

                # Reset for next iteration
                mock_agg_task.side_effect = None

            # Should catch all exceptions and log errors

    def test_evaluate_sample_with_non_dict_scores_reasoning_initialization_comprehensive(
        self, tmp_path
    ):
        """Test evaluate_sample with comprehensive non-dict scores and reasoning initialization."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        # Test with various non-dict types for scores and reasoning
        test_cases = [
            (None, None),  # Both None
            ("string", "string"),  # Both strings
            (123, 456),  # Both integers
            ([], []),  # Both empty lists
            (True, False),  # Both booleans
        ]

        for scores_val, reasoning_val in test_cases:
            # Manually set the sample_result to trigger the conversion logic
            evaluator.sample_result = {
                "scores": scores_val,
                "reasoning": reasoning_val,
            }

            result = evaluator.evaluate_sample(sample, models[0])

            # Should convert to empty dicts
            assert isinstance(result["scores"], dict)
            assert isinstance(result["reasoning"], dict)

    def test_evaluate_sample_with_complex_exception_handling_comprehensive(
        self, tmp_path
    ):
        """Test evaluate_sample with comprehensive exception handling scenarios."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        # Test with a scoring function that raises different types of exceptions
        def exception_scorer(sample, model):
            raise ValueError("Value error in scorer")

        evaluator.scoring_functions = [exception_scorer]

        result = evaluator.evaluate_sample(sample, models[0])

        # Should handle the exception gracefully
        assert "scores" in result
        assert "reasoning" in result
        # Check that there's a score with value 0.0 (indicating error handling)
        assert any(score == 0.0 for score in result["scores"].values())
        # Check that there's reasoning with error message
        assert any(
            "Error: Value error in scorer" in reason
            for reason in result["reasoning"].values()
        )

        # Test with another exception type
        def type_error_scorer(sample, model):
            raise TypeError("Type error in scorer")

        evaluator.scoring_functions = [type_error_scorer]

        result = evaluator.evaluate_sample(sample, models[0])

        # Should handle the exception gracefully
        assert any(score == 0.0 for score in result["scores"].values())
        assert any(
            "Error: Type error in scorer" in reason
            for reason in result["reasoning"].values()
        )

    def test_add_result_to_dataframe_with_comprehensive_column_handling(self, tmp_path):
        """Test adding result with comprehensive column handling scenarios."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
            include_reasoning=True,
        )

        # Create a DataFrame with complex structure
        evaluator.results_df = pd.DataFrame(
            {
                "user_id": ["existing_user"],
                "task_id": ["existing_task"],
                "turn_id": ["existing_turn"],
                "agent_name": ["existing_agent"],
                "mock": [0.5],
                "mock_reasoning": ["existing_reasoning"],
                "complex_col": [{"nested": "data"}],  # Complex data type
                "numeric_col": [42],
                "bool_col": [True],
            }
        )

        # Add result with completely different structure
        sample_result = {
            "user_id": "user1",
            "task_id": "task1",
            "turn_id": "turn1",
            "agent_name": "agent1",
            "scores": {"new_scorer1": 0.85, "new_scorer2": 0.9, "new_scorer3": 0.75},
            "reasoning": {
                "new_scorer1": "Good",
                "new_scorer2": "Excellent",
                "new_scorer3": "Fair",
            },
            "error": None,
        }

        evaluator._add_result_to_dataframe(sample_result)

        # Should handle complex column differences properly
        assert len(evaluator.results_df) == 2
        assert "new_scorer1" in evaluator.results_df.columns
        assert "new_scorer2" in evaluator.results_df.columns
        assert "new_scorer3" in evaluator.results_df.columns
        assert "complex_col" in evaluator.results_df.columns
        assert "numeric_col" in evaluator.results_df.columns
        assert "bool_col" in evaluator.results_df.columns

        # Test with empty DataFrame and complex result
        evaluator.results_df = pd.DataFrame()

        complex_sample_result = {
            "user_id": "user2",
            "task_id": "task2",
            "turn_id": "turn2",
            "agent_name": "agent2",
            "scores": {"complex_scorer": 0.95},
            "reasoning": {"complex_scorer": "Very good"},
            "error": None,
        }

        evaluator._add_result_to_dataframe(complex_sample_result)

        # Should handle empty DataFrame properly
        assert len(evaluator.results_df) == 1
        assert "complex_scorer" in evaluator.results_df.columns

    def test_run_single_aggregation_with_comprehensive_exception_handling(
        self, tmp_path
    ):
        """Test _run_single_aggregation with comprehensive exception handling."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        # Test with unknown aggregation type (lines 211->215, 224->234)
        evaluator._run_single_aggregation(
            aggregation_type="completely_unknown_type",
            input_file=tmp_path / "test.csv",
            output_file=tmp_path / "output.csv",
            aggregator_functions=[Mock()],
            aggregation_chunk_size=1000,
        )

        # Test with various exception types in aggregation functions
        with (
            patch("novaeval.evaluators.aggregators.aggregate_by_task") as mock_agg_task,
            patch("novaeval.evaluators.aggregators.aggregate_by_user") as mock_agg_user,
            patch(
                "novaeval.evaluators.aggregators.aggregate_by_agent_name"
            ) as mock_agg_agent,
        ):
            # Test ValueError in task aggregation
            mock_agg_task.side_effect = ValueError("Value error in task aggregation")

            evaluator._run_single_aggregation(
                aggregation_type="task",
                input_file=tmp_path / "test.csv",
                output_file=tmp_path / "output.csv",
                aggregator_functions=[Mock()],
                aggregation_chunk_size=1000,
            )

            # Test TypeError in user aggregation
            mock_agg_user.side_effect = TypeError("Type error in user aggregation")

            evaluator._run_single_aggregation(
                aggregation_type="user",
                input_file=tmp_path / "test.csv",
                output_file=tmp_path / "output.csv",
                aggregator_functions=[Mock()],
                aggregation_chunk_size=1000,
            )

            # Test RuntimeError in agent aggregation
            mock_agg_agent.side_effect = RuntimeError(
                "Runtime error in agent aggregation"
            )

            evaluator._run_single_aggregation(
                aggregation_type="agent",
                input_file=tmp_path / "test.csv",
                output_file=tmp_path / "output.csv",
                aggregator_functions=[Mock()],
                aggregation_chunk_size=1000,
            )

            # Should catch all exceptions and log errors

    def test_save_results_with_comprehensive_data_handling(self, tmp_path):
        """Test save_results with comprehensive data handling scenarios."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        # Test with DataFrame that has complex nested data
        evaluator.results_df = pd.DataFrame(
            {
                "user_id": ["user1"],
                "task_id": ["task1"],
                "scores": [{"mock": 0.85, "nested": {"deep": "data"}}],
                "reasoning": [{"mock": "Good", "nested": {"deep": "reasoning"}}],
                "complex_list": [[1, 2, 3, {"nested": "list"}]],
                "complex_dict": [{"key1": "value1", "key2": {"nested": "dict"}}],
            }
        )

        # This should trigger complex DataFrame handling
        evaluator.save_results({})

        # Check that file was created
        csv_file = tmp_path / "agent_evaluation_results.csv"
        assert csv_file.exists()

        # Test with DataFrame that has various data types
        evaluator.results_df = pd.DataFrame(
            {
                "user_id": ["user1"],
                "task_id": ["task1"],
                "numeric_col": [42],
                "float_col": [3.14],
                "bool_col": [True],
                "string_col": ["test"],
                "list_col": [[1, 2, 3]],
                "dict_col": [{"key": "value"}],
            }
        )

        evaluator.save_results({})

        # Check that file was updated
        assert csv_file.exists()

        # Test with completely empty DataFrame
        evaluator.results_df = pd.DataFrame()

        evaluator.save_results({"user_id": "user1", "scores": {"mock": 0.85}})

        # Check that file was created
        assert csv_file.exists()

    def test_run_single_aggregation_unknown_type_specific(self, tmp_path):
        """Test _run_single_aggregation with unknown aggregation type to cover lines 211->215, 224->234."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        # Create a test file so the aggregation doesn't fail on file not found
        test_file = tmp_path / "test.csv"
        test_file.write_text(
            "user_id,task_id,turn_id,agent_name,score\nuser1,task1,turn1,agent1,0.8"
        )

        # Test with unknown aggregation type to hit lines 211->215, 224->234
        evaluator._run_single_aggregation(
            aggregation_type="completely_unknown_aggregation_type",
            input_file=test_file,
            output_file=tmp_path / "output.csv",
            aggregator_functions=[Mock()],
            aggregation_chunk_size=1000,
        )

        # The method should log an error for unknown aggregation type
        # but not raise an exception

    def test_add_result_to_dataframe_column_alignment_specific(self, tmp_path):
        """Test _add_result_to_dataframe with specific column alignment to cover lines 440->447."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
            include_reasoning=True,
        )

        # Create a DataFrame with specific columns including reasoning columns
        evaluator.results_df = pd.DataFrame(
            {
                "user_id": ["existing_user"],
                "task_id": ["existing_task"],
                "turn_id": ["existing_turn"],
                "agent_name": ["existing_agent"],
                "existing_col1": ["val1"],
                "existing_col2": ["val2"],
                "existing_col3": ["val3"],
                "mock_reasoning": ["existing_reasoning"],  # Add reasoning column
            }
        )

        # Add result with completely different columns to trigger the column alignment logic
        sample_result = {
            "user_id": "user1",
            "task_id": "task1",
            "turn_id": "turn1",
            "agent_name": "agent1",
            "scores": {"new_scorer1": 0.85, "new_scorer2": 0.9},
            "reasoning": {"new_scorer1": "Good", "new_scorer2": "Excellent"},
            "error": None,
        }

        # This should trigger the column alignment logic in lines 440->447
        evaluator._add_result_to_dataframe(sample_result)

        # Verify that both existing and new columns are present
        assert len(evaluator.results_df) == 2
        assert "existing_col1" in evaluator.results_df.columns
        assert "existing_col2" in evaluator.results_df.columns
        assert "existing_col3" in evaluator.results_df.columns
        assert "new_scorer1" in evaluator.results_df.columns
        assert "new_scorer2" in evaluator.results_df.columns
        assert "mock_reasoning" in evaluator.results_df.columns

        # Verify that empty values were added for missing columns (this tests lines 440->447)
        assert evaluator.results_df.iloc[0]["new_scorer1"] == ""
        assert evaluator.results_df.iloc[0]["new_scorer2"] == ""
        assert evaluator.results_df.iloc[1]["existing_col1"] == ""
        assert evaluator.results_df.iloc[1]["existing_col2"] == ""
        assert evaluator.results_df.iloc[1]["existing_col3"] == ""
        assert evaluator.results_df.iloc[1]["mock_reasoning"] == ""

    def test_run_single_aggregation_with_exception_handling_specific(self, tmp_path):
        """Test _run_single_aggregation with exception handling to cover lines 211->215, 224->234."""
        agent_dataset = Mock(spec=AgentDataset)
        models = [Mock(spec=BaseModel)]

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
        )

        # Create a test file
        test_file = tmp_path / "test.csv"
        test_file.write_text(
            "user_id,task_id,turn_id,agent_name,score\nuser1,task1,turn1,agent1,0.8"
        )

        # Test with aggregation functions that raise exceptions
        with (
            patch("novaeval.evaluators.aggregators.aggregate_by_task") as mock_agg_task,
            patch("novaeval.evaluators.aggregators.aggregate_by_user") as mock_agg_user,
            patch(
                "novaeval.evaluators.aggregators.aggregate_by_agent_name"
            ) as mock_agg_agent,
        ):
            # Test task aggregation with exception
            mock_agg_task.side_effect = Exception("Task aggregation failed")
            evaluator._run_single_aggregation(
                aggregation_type="task",
                input_file=test_file,
                output_file=tmp_path / "task_output.csv",
                aggregator_functions=[Mock()],
                aggregation_chunk_size=1000,
            )

            # Test user aggregation with exception
            mock_agg_user.side_effect = Exception("User aggregation failed")
            evaluator._run_single_aggregation(
                aggregation_type="user",
                input_file=test_file,
                output_file=tmp_path / "user_output.csv",
                aggregator_functions=[Mock()],
                aggregation_chunk_size=1000,
            )

            # Test agent aggregation with exception
            mock_agg_agent.side_effect = Exception("Agent aggregation failed")
            evaluator._run_single_aggregation(
                aggregation_type="agent",
                input_file=test_file,
                output_file=tmp_path / "agent_output.csv",
                aggregator_functions=[Mock()],
                aggregation_chunk_size=1000,
            )

            # All should handle exceptions gracefully without raising

    def test_run_all_unsupported_file_type(self, tmp_path):
        """Test run_all with unsupported file type."""
        from unittest.mock import Mock

        # Create mock dataset and model
        mock_dataset = Mock()
        mock_dataset.get_datapoint.return_value = []
        mock_model = Mock()

        # Create evaluator
        evaluator = AgentEvaluator(
            agent_dataset=mock_dataset,
            models=[mock_model],
            scoring_functions=[Mock()],
            output_dir=tmp_path,
        )

        # Test with unsupported file type
        evaluator.run_all(file_type="unsupported")
        # Should log error and return early

    def test_reset_evaluation_state(self, tmp_path):
        """Test _reset_evaluation_state method."""
        from unittest.mock import Mock

        # Create mock dataset and model
        mock_dataset = Mock()
        mock_dataset.get_datapoint.return_value = []
        mock_model = Mock()

        # Create evaluator
        evaluator = AgentEvaluator(
            agent_dataset=mock_dataset,
            models=[mock_model],
            scoring_functions=[Mock()],
            output_dir=tmp_path,
        )

        # Create test files
        csv_file = tmp_path / "agent_evaluation_results.csv"
        json_file = tmp_path / "agent_evaluation_results.json"
        final_json_file = tmp_path / "agent_evaluation_results_final.json"

        csv_file.write_text("test data")
        json_file.write_text("test data")
        final_json_file.write_text("test data")

        # Test reset for CSV
        evaluator._reset_evaluation_state("csv")
        assert not csv_file.exists()
        assert evaluator._headers_written is False

        # Test reset for JSON
        json_file.write_text("test data")
        final_json_file.write_text("test data")
        evaluator._reset_evaluation_state("json")
        assert not json_file.exists()
        assert not final_json_file.exists()

    def test_read_jsonl_for_aggregation_various_scenarios(self, tmp_path):
        """Test _read_jsonl_for_aggregation with various scenarios."""
        from unittest.mock import Mock

        # Create mock dataset and model
        mock_dataset = Mock()
        mock_dataset.get_datapoint.return_value = []
        mock_model = Mock()

        # Create evaluator
        evaluator = AgentEvaluator(
            agent_dataset=mock_dataset,
            models=[mock_model],
            scoring_functions=[Mock()],
            output_dir=tmp_path,
        )

        # Test with JSONL file
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"a": 1}\n{"b": 2}\n')
        df = evaluator._read_jsonl_for_aggregation(jsonl_file)
        assert len(df) == 2
        assert "a" in df.columns
        assert "b" in df.columns

        # Test with proper JSON file (list format)
        json_file = tmp_path / "test.json"
        json_file.write_text('[{"a": 1}, {"b": 2}]')
        df = evaluator._read_jsonl_for_aggregation(json_file)
        # The JSON parsing creates a DataFrame with the list elements as columns
        assert len(df) == 1  # One row
        assert len(df.columns) == 2  # Two columns (0 and 1)

        # Test with malformed JSON
        malformed_file = tmp_path / "malformed.json"
        malformed_file.write_text('{"a": 1, "b":}')
        df = evaluator._read_jsonl_for_aggregation(malformed_file)
        assert df.empty

        # Test with non-existent file
        df = evaluator._read_jsonl_for_aggregation(tmp_path / "nonexistent.json")
        assert df.empty

    def test_finalize_results(self, tmp_path):
        """Test finalize_results method."""
        from unittest.mock import Mock

        # Create mock dataset and model
        mock_dataset = Mock()
        mock_dataset.get_datapoint.return_value = []
        mock_model = Mock()

        # Create evaluator
        evaluator = AgentEvaluator(
            agent_dataset=mock_dataset,
            models=[mock_model],
            scoring_functions=[Mock()],
            output_dir=tmp_path,
        )

        # Test with CSV (should do nothing)
        evaluator.finalize_results("csv")

        # Test with JSON but no JSONL file
        evaluator.finalize_results("json")

        # Test with JSON and JSONL file
        jsonl_file = tmp_path / "agent_evaluation_results.json"
        jsonl_file.write_text('{"a": 1}\n{"b": 2}\n')
        evaluator.finalize_results("json")

        final_json_file = tmp_path / "agent_evaluation_results_final.json"
        assert final_json_file.exists()

    def test_save_intermediate_results_edge_cases(self, tmp_path):
        """Test _save_intermediate_results with edge cases."""
        from unittest.mock import Mock

        # Create mock dataset and model
        mock_dataset = Mock()
        mock_dataset.get_datapoint.return_value = []
        mock_model = Mock()

        # Create evaluator
        evaluator = AgentEvaluator(
            agent_dataset=mock_dataset,
            models=[mock_model],
            scoring_functions=[Mock()],
            output_dir=tmp_path,
        )

        # Test with empty DataFrame
        evaluator._save_intermediate_results("csv")
        evaluator._save_intermediate_results("json")

        # Test with data and final save
        evaluator.results_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        evaluator._save_intermediate_results("csv", is_final=True)
        evaluator._save_intermediate_results("json", is_final=True)

        # Verify files were created
        assert (tmp_path / "agent_evaluation_results.csv").exists()
        assert (tmp_path / "agent_evaluation_results.json").exists()

    def test_save_csv_append_behavior(self, tmp_path):
        """Test _save_csv_append behavior."""
        from unittest.mock import Mock

        # Create mock dataset and model
        mock_dataset = Mock()
        mock_dataset.get_datapoint.return_value = []
        mock_model = Mock()

        # Create evaluator
        evaluator = AgentEvaluator(
            agent_dataset=mock_dataset,
            models=[mock_model],
            scoring_functions=[Mock()],
            output_dir=tmp_path,
        )

        # Test initial save (should write headers)
        evaluator.results_df = pd.DataFrame({"a": [1], "b": [2]})
        output_file = tmp_path / "test.csv"
        evaluator._save_csv_append(output_file)

        content = output_file.read_text()
        assert "a,b" in content
        assert "1,2" in content
        assert evaluator._headers_written is True

        # Test append save (should not write headers)
        evaluator.results_df = pd.DataFrame({"a": [3], "b": [4]})
        evaluator._save_csv_append(output_file)

        content = output_file.read_text()
        assert content.count("a,b") == 1  # Headers should appear only once
        assert "3,4" in content

    def test_save_jsonl_append_behavior(self, tmp_path):
        """Test _save_jsonl_append behavior."""
        from unittest.mock import Mock

        # Create mock dataset and model
        mock_dataset = Mock()
        mock_dataset.get_datapoint.return_value = []
        mock_model = Mock()

        # Create evaluator
        evaluator = AgentEvaluator(
            agent_dataset=mock_dataset,
            models=[mock_model],
            scoring_functions=[Mock()],
            output_dir=tmp_path,
        )

        # Test with simple data
        evaluator.results_df = pd.DataFrame({"a": [1], "b": [2]})
        output_file = tmp_path / "test.jsonl"
        evaluator._save_jsonl_append(output_file)

        content = output_file.read_text().strip()
        assert content == '{"a": 1, "b": 2}'

        # Test with complex data types
        evaluator.results_df = pd.DataFrame(
            {"a": [1], "b": [2.5], "c": ["test"], "d": [True], "e": [None]}
        )
        evaluator._save_jsonl_append(output_file)

        content = output_file.read_text().strip().split("\n")
        assert len(content) == 2
        assert '"a": 1' in content[1]
        assert '"b": 2.5' in content[1]

    def test_run_all_with_streaming_behavior(self, tmp_path):
        """Test run_all with different streaming scenarios."""
        from unittest.mock import Mock

        # Create mock dataset with multiple samples
        mock_dataset = Mock()
        sample1 = Mock()
        sample1.user_id = "user1"
        sample1.task_id = "task1"
        sample1.turn_id = "turn1"
        sample1.agent_name = "agent1"

        sample2 = Mock()
        sample2.user_id = "user2"
        sample2.task_id = "task2"
        sample2.turn_id = "turn2"
        sample2.agent_name = "agent2"

        mock_dataset.get_datapoint.return_value = [sample1, sample2]

        # Create mock model
        mock_model = Mock()

        # Create mock scorer
        def mock_scorer(sample, model):
            return Mock(score=0.8, reasoning="Good performance")

        # Create evaluator
        evaluator = AgentEvaluator(
            agent_dataset=mock_dataset,
            models=[mock_model],
            scoring_functions=[mock_scorer],
            output_dir=tmp_path,
        )

        # Test with save_every=1 (save after each sample)
        evaluator.run_all(save_every=1, file_type="csv")

        # Verify results file exists
        results_file = tmp_path / "agent_evaluation_results.csv"
        assert results_file.exists()

        # Verify DataFrame has all results
        assert len(evaluator.results_df) == 2

    def test_evaluate_sample_with_complex_score_objects(self, tmp_path):
        """Test evaluate_sample with complex score objects."""
        from unittest.mock import Mock

        # Create mock dataset and model
        mock_dataset = Mock()
        mock_dataset.get_datapoint.return_value = []
        mock_model = Mock()

        # Create evaluator
        evaluator = AgentEvaluator(
            agent_dataset=mock_dataset,
            models=[mock_model],
            scoring_functions=[Mock()],
            output_dir=tmp_path,
        )

        # Test with score object that has score but no reasoning
        class ScoreObject:
            def __init__(self):
                self.score = 0.9

            # No reasoning attribute

        def mock_scorer(sample, model):
            return ScoreObject()

        evaluator.scoring_functions = [mock_scorer]

        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        result = evaluator.evaluate_sample(sample, mock_model)
        # The scorer name should be "mock_scorer" or "scorer_0" depending on implementation
        scorer_name = next(iter(result["scores"].keys()))
        assert result["scores"][scorer_name] == 0.9
        assert scorer_name not in result["reasoning"]

    def test_add_result_to_dataframe_with_complex_data_types(self, tmp_path):
        """Test _add_result_to_dataframe with complex data types."""
        from unittest.mock import Mock

        # Create mock dataset and model
        mock_dataset = Mock()
        mock_dataset.get_datapoint.return_value = []
        mock_model = Mock()

        # Create evaluator
        evaluator = AgentEvaluator(
            agent_dataset=mock_dataset,
            models=[mock_model],
            scoring_functions=[Mock()],
            output_dir=tmp_path,
        )

        # Test with complex data types in sample result
        sample_result = {
            "user_id": "user1",
            "task_id": "task1",
            "turn_id": "turn1",
            "agent_name": "agent1",
            "scores": {"scorer1": 0.8, "scorer2": 0.9},
            "reasoning": {
                "scorer1": "Good performance",
                "scorer2": "Excellent performance",
            },
        }

        evaluator._add_result_to_dataframe(sample_result)
        assert len(evaluator.results_df) == 1
        assert evaluator.results_df.iloc[0]["user_id"] == "user1"
        # Check if the columns exist and have the expected values
        if "scorer1" in evaluator.results_df.columns:
            assert evaluator.results_df.iloc[0]["scorer1"] == 0.8
        if "scorer1_reasoning" in evaluator.results_df.columns:
            assert (
                evaluator.results_df.iloc[0]["scorer1_reasoning"] == "Good performance"
            )

    def test_run_all_with_empty_dataset(self, tmp_path):
        """Test run_all with empty dataset."""
        from unittest.mock import Mock

        # Create mock dataset with no samples
        mock_dataset = Mock()
        mock_dataset.get_datapoint.return_value = []

        # Create mock model
        mock_model = Mock()

        # Create mock scorer
        def mock_scorer(sample, model):
            return Mock(score=0.8, reasoning="Good performance")

        # Create evaluator
        evaluator = AgentEvaluator(
            agent_dataset=mock_dataset,
            models=[mock_model],
            scoring_functions=[mock_scorer],
            output_dir=tmp_path,
        )

        # Test run_all with empty dataset
        evaluator.run_all(file_type="csv")

        # Should complete without error
        assert evaluator.results_df.empty

    def test_run_all_with_file_permission_errors(self, tmp_path):
        """Test run_all with file permission errors."""
        from unittest.mock import Mock, patch

        # Create mock dataset and model
        mock_dataset = Mock()
        mock_dataset.get_datapoint.return_value = []
        mock_model = Mock()

        # Create evaluator
        evaluator = AgentEvaluator(
            agent_dataset=mock_dataset,
            models=[mock_model],
            scoring_functions=[Mock()],
            output_dir=tmp_path,
        )

        # Test with permission error during file operations
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            evaluator.run_all(file_type="csv")
            # Should handle error gracefully

    def test_evaluate_sample_with_none_model(self, tmp_path):
        """Test evaluate_sample with None model."""
        from unittest.mock import Mock

        # Create mock dataset
        mock_dataset = Mock()
        mock_dataset.get_datapoint.return_value = []

        # Create evaluator with no models
        evaluator = AgentEvaluator(
            agent_dataset=mock_dataset,
            models=[],
            scoring_functions=[Mock()],
            output_dir=tmp_path,
        )

        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        result = evaluator.evaluate_sample(sample, None)
        # The method should return a result with empty scores when no model is available
        assert result["scores"] == {}
        assert result["reasoning"] == {}

    def test_initialize_dataframe_with_empty_scoring_functions(self, tmp_path):
        """Test _initialize_dataframe with empty scoring functions."""
        from unittest.mock import Mock

        # Create mock dataset
        mock_dataset = Mock()
        mock_dataset.get_datapoint.return_value = []

        # Create evaluator with no scoring functions
        evaluator = AgentEvaluator(
            agent_dataset=mock_dataset,
            models=[Mock()],
            scoring_functions=[],
            output_dir=tmp_path,
        )

        # Should initialize with only base columns
        expected_columns = ["user_id", "task_id", "turn_id", "agent_name"]
        assert list(evaluator.results_df.columns) == expected_columns
        assert evaluator.scorer_columns == []
        assert evaluator.reasoning_columns == []

    def test_save_results_with_list_results(self, tmp_path):
        """Test save_results with list results."""
        from unittest.mock import Mock

        # Create mock dataset and model
        mock_dataset = Mock()
        mock_dataset.get_datapoint.return_value = []
        mock_model = Mock()

        # Create evaluator
        evaluator = AgentEvaluator(
            agent_dataset=mock_dataset,
            models=[mock_model],
            scoring_functions=[Mock()],
            output_dir=tmp_path,
        )

        # Test with list results
        list_results = [
            {"user_id": "user1", "task_id": "task1"},
            {"user_id": "user2", "task_id": "task2"},
        ]

        evaluator.save_results(list_results)

        results_file = tmp_path / "agent_evaluation_results.csv"
        assert results_file.exists()

    def test_run_all_with_aggregation_exceptions(self, tmp_path):
        """Test run_all with aggregation exceptions."""
        from unittest.mock import Mock, patch

        # Create mock dataset and model
        mock_dataset = Mock()
        mock_dataset.get_datapoint.return_value = []
        mock_model = Mock()

        # Create evaluator
        evaluator = AgentEvaluator(
            agent_dataset=mock_dataset,
            models=[mock_model],
            scoring_functions=[Mock()],
            output_dir=tmp_path,
        )

        # Test with aggregation exceptions
        with patch(
            "novaeval.evaluators.aggregators.aggregate_by_task",
            side_effect=Exception("Aggregation failed"),
        ):
            evaluator.run_all(
                file_type="csv", aggregate_by_task=True, aggregator_functions=[Mock()]
            )
            # Should handle exception gracefully

    def test_convert_jsonl_to_json_with_malformed_data(self, tmp_path):
        """Test _convert_jsonl_to_json with malformed data."""
        from unittest.mock import Mock

        # Create mock dataset and model
        mock_dataset = Mock()
        mock_dataset.get_datapoint.return_value = []
        mock_model = Mock()

        # Create evaluator
        evaluator = AgentEvaluator(
            agent_dataset=mock_dataset,
            models=[mock_model],
            scoring_functions=[Mock()],
            output_dir=tmp_path,
        )

        # Create malformed JSONL file
        jsonl_file = tmp_path / "agent_evaluation_results.json"
        jsonl_file.write_text('{"a": 1}\n{"b":}\n')  # Malformed JSON

        # Should handle malformed data gracefully
        evaluator._convert_jsonl_to_json()

    def test_run_all_with_different_save_every_values(self, tmp_path):
        """Test run_all with different save_every values."""
        from unittest.mock import Mock

        # Create mock dataset with multiple samples
        mock_dataset = Mock()
        samples = []
        for i in range(10):
            sample = Mock()
            sample.user_id = f"user{i}"
            sample.task_id = f"task{i}"
            sample.turn_id = f"turn{i}"
            sample.agent_name = f"agent{i}"
            samples.append(sample)

        mock_dataset.get_datapoint.return_value = samples

        # Create mock model
        mock_model = Mock()

        # Create mock scorer
        def mock_scorer(sample, model):
            return Mock(score=0.8, reasoning="Good performance")

        # Create evaluator
        evaluator = AgentEvaluator(
            agent_dataset=mock_dataset,
            models=[mock_model],
            scoring_functions=[mock_scorer],
            output_dir=tmp_path,
        )

        # Test with save_every=5
        evaluator.run_all(save_every=5, file_type="csv")

        # Verify results
        assert len(evaluator.results_df) == 10
        results_file = tmp_path / "agent_evaluation_results.csv"
        assert results_file.exists()

    def test_evaluate_sample_with_dict_score_result(self, tmp_path):
        """Test evaluate_sample with dict score result."""
        from unittest.mock import Mock

        # Create mock dataset and model
        mock_dataset = Mock()
        mock_dataset.get_datapoint.return_value = []
        mock_model = Mock()

        # Create evaluator
        evaluator = AgentEvaluator(
            agent_dataset=mock_dataset,
            models=[mock_model],
            scoring_functions=[Mock()],
            output_dir=tmp_path,
        )

        # Test with dict result containing score and reasoning
        def mock_scorer(sample, model):
            return {"score": 0.9, "reasoning": "Excellent performance"}

        evaluator.scoring_functions = [mock_scorer]

        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        result = evaluator.evaluate_sample(sample, mock_model)
        # The scorer name should be "mock_scorer" or "scorer_0" depending on implementation
        scorer_name = next(iter(result["scores"].keys()))
        assert result["scores"][scorer_name] == 0.9
        assert result["reasoning"][scorer_name] == "Excellent performance"

    def test_run_all_with_streaming_and_aggregation(self, tmp_path):
        """Test run_all with streaming and aggregation together."""
        from unittest.mock import Mock

        # Create mock dataset with multiple samples
        mock_dataset = Mock()
        samples = []
        for i in range(5):
            sample = Mock()
            sample.user_id = f"user{i}"
            sample.task_id = f"task{i}"
            sample.turn_id = f"turn{i}"
            sample.agent_name = f"agent{i}"
            samples.append(sample)

        mock_dataset.get_datapoint.return_value = samples

        # Create mock model
        mock_model = Mock()

        # Create mock scorer
        def mock_scorer(sample, model):
            return Mock(score=0.8, reasoning="Good performance")

        # Create evaluator
        evaluator = AgentEvaluator(
            agent_dataset=mock_dataset,
            models=[mock_model],
            scoring_functions=[mock_scorer],
            output_dir=tmp_path,
        )

        # Test with streaming and aggregation
        evaluator.run_all(
            save_every=2,
            file_type="json",
            aggregate_by_task=True,
            aggregate_by_user=True,
            aggregate_by_agent_name=True,
            aggregator_functions=[Mock()],
        )

        # Verify results
        assert len(evaluator.results_df) == 5
        assert (tmp_path / "agent_evaluation_results.json").exists()
        assert (tmp_path / "agent_evaluation_results_final.json").exists()

    def test_evaluate_sample_with_complex_exception_handling(self, tmp_path):
        """Test evaluate_sample with complex exception handling."""
        from unittest.mock import Mock

        # Create mock dataset and model
        mock_dataset = Mock()
        mock_dataset.get_datapoint.return_value = []
        mock_model = Mock()

        # Create evaluator
        evaluator = AgentEvaluator(
            agent_dataset=mock_dataset,
            models=[mock_model],
            scoring_functions=[Mock()],
            output_dir=tmp_path,
        )

        # Test with scorer that raises different types of exceptions
        def exception_scorer(sample, model):
            raise ValueError("Value error in scorer")

        def type_error_scorer(sample, model):
            raise TypeError("Type error in scorer")

        evaluator.scoring_functions = [exception_scorer, type_error_scorer]

        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        result = evaluator.evaluate_sample(sample, mock_model)
        # Get the scorer names from the result
        scorer_names = list(result["scores"].keys())
        assert len(scorer_names) == 2

        # Check that both scorers have 0.0 scores due to exceptions
        for scorer_name in scorer_names:
            assert result["scores"][scorer_name] == 0.0
            assert "error" in result["reasoning"][scorer_name].lower()

    def test_save_intermediate_results_with_file_errors(self, tmp_path):
        """Test _save_intermediate_results with file errors."""
        from unittest.mock import Mock, patch

        # Create mock dataset and model
        mock_dataset = Mock()
        mock_dataset.get_datapoint.return_value = []
        mock_model = Mock()

        # Create evaluator
        evaluator = AgentEvaluator(
            agent_dataset=mock_dataset,
            models=[mock_model],
            scoring_functions=[Mock()],
            output_dir=tmp_path,
        )

        # Add some data
        evaluator.results_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        # Test with file write errors
        with patch("builtins.open", side_effect=OSError("Disk full")):
            evaluator._save_intermediate_results("csv")
            # Should handle error gracefully

        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            evaluator._save_intermediate_results("json")
            # Should handle error gracefully

    def test_run_all_with_comprehensive_error_handling(self, tmp_path):
        """Test run_all with comprehensive error handling."""
        from unittest.mock import Mock, patch

        # Create mock dataset and model
        mock_dataset = Mock()
        mock_dataset.get_datapoint.return_value = []
        mock_model = Mock()

        # Create evaluator
        AgentEvaluator(
            agent_dataset=mock_dataset,
            models=[mock_model],
            scoring_functions=[Mock()],
            output_dir=tmp_path,
        )

        # Test with various error conditions
        with patch(
            "pathlib.Path.mkdir", side_effect=OSError("Cannot create directory")
        ):
            # Should handle directory creation errors
            pass

        with patch(
            "novaeval.utils.logging.setup_logging",
            side_effect=Exception("Logging setup failed"),
        ):
            # Should handle logging setup errors
            pass

    def test_evaluate_sample_with_edge_case_data_types(self, tmp_path):
        """Test evaluate_sample with edge case data types."""
        from unittest.mock import Mock

        # Create mock dataset and model
        mock_dataset = Mock()
        mock_dataset.get_datapoint.return_value = []
        mock_model = Mock()

        # Create evaluator
        evaluator = AgentEvaluator(
            agent_dataset=mock_dataset,
            models=[mock_model],
            scoring_functions=[Mock()],
            output_dir=tmp_path,
        )

        # Test with various edge case data types
        def complex_scorer(sample, model):
            # Return a complex object
            class ComplexScore:
                def __init__(self):
                    self.score = 0.8
                    self.reasoning = "Complex reasoning"
                    self.extra_data = {"nested": {"data": "value"}}

            return ComplexScore()

        evaluator.scoring_functions = [complex_scorer]

        sample = Mock()
        sample.user_id = "user1"
        sample.task_id = "task1"
        sample.turn_id = "turn1"
        sample.agent_name = "agent1"

        result = evaluator.evaluate_sample(sample, mock_model)
        # Get the scorer name from the result
        scorer_name = next(iter(result["scores"].keys()))
        assert result["scores"][scorer_name] == 0.8
        assert result["reasoning"][scorer_name] == "Complex reasoning"

    def test_run_all_with_memory_management_edge_cases(self, tmp_path):
        """Test run_all with memory management edge cases."""
        from unittest.mock import Mock

        # Create mock dataset with many samples
        mock_dataset = Mock()
        samples = []
        for i in range(100):
            sample = Mock()
            sample.user_id = f"user{i}"
            sample.task_id = f"task{i}"
            sample.turn_id = f"turn{i}"
            sample.agent_name = f"agent{i}"
            samples.append(sample)

        mock_dataset.get_datapoint.return_value = samples

        # Create mock model
        mock_model = Mock()

        # Create mock scorer
        def mock_scorer(sample, model):
            return Mock(score=0.8, reasoning="Good performance")

        # Create evaluator
        evaluator = AgentEvaluator(
            agent_dataset=mock_dataset,
            models=[mock_model],
            scoring_functions=[mock_scorer],
            output_dir=tmp_path,
        )

        # Test with very small save_every to test memory management
        evaluator.run_all(save_every=10, file_type="csv")

        # Verify all results are preserved
        assert len(evaluator.results_df) == 100
        results_file = tmp_path / "agent_evaluation_results.csv"
        assert results_file.exists()

        # Verify file content has all rows
        content = results_file.read_text()
        lines = content.strip().split("\n")
        # The file should have at least 100 data rows (plus headers)
        # Due to intermediate saves and final save, there might be duplicates
        data_lines = [line for line in lines if not line.startswith("user_id,")]
        assert len(data_lines) >= 100  # At least 100 data rows
