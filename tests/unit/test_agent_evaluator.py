"""
Tests for the AgentEvaluator class.
"""

from unittest.mock import Mock, patch

import pandas as pd

from novaeval.datasets.agent_dataset import AgentDataset
from novaeval.evaluators.agent_evaluator import AgentEvaluator
from novaeval.models.base import BaseModel


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
        result = evaluator.evaluate_sample(sample)

        # Verify result structure
        assert result["user_id"] == "user1"
        assert result["task_id"] == "task1"
        assert result["turn_id"] == "turn1"
        assert result["agent_name"] == "agent1"
        assert "scores" in result
        assert "reasoning" in result
        assert result["error"] is None
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
        evaluator.save_results(results, file_type="csv")

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
        evaluator.save_results(results, file_type="json")

        # Check that file was created
        json_file = tmp_path / "agent_evaluation_results.json"
        assert json_file.exists()

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

        def mock_scorer(sample, model):
            return Mock()

        scoring_functions = [mock_scorer]

        evaluator = AgentEvaluator(
            agent_dataset=agent_dataset,
            models=models,
            scoring_functions=scoring_functions,
            output_dir=tmp_path,
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

        result = evaluator.evaluate_sample(sample)

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

        result = evaluator.evaluate_sample(sample)

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

        result = evaluator.evaluate_sample(sample)

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

        result = evaluator.evaluate_sample(sample)

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

        result = evaluator.evaluate_sample(sample)

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

        result = evaluator.evaluate_sample(sample)

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

        result = evaluator.evaluate_sample(sample)

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

        # Convert to JSON
        evaluator._convert_to_json()

        json_file = tmp_path / "agent_evaluation_results.json"
        assert json_file.exists()

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

        result = evaluator.evaluate_sample(sample)

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

        result = evaluator.evaluate_sample(sample)

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

        result = evaluator.evaluate_sample(sample)

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

        # Convert to JSON - this should trigger line 449 for the comma
        evaluator._convert_to_json()

        # Check that file was created and has proper JSON array format
        json_file = tmp_path / "agent_evaluation_results.json"
        assert json_file.exists()

        # Verify the content is valid JSON array
        with open(json_file) as f:
            content = f.read()
            assert content.startswith("[\n")
            assert content.endswith("\n]")
            assert ",\n" in content  # Should have comma separator

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

        result = evaluator.evaluate_sample(sample)

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

        result = evaluator.evaluate_sample(sample)

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

        result = evaluator.evaluate_sample(sample)

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

        result = evaluator.evaluate_sample(sample)

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
        assert pd.isna(evaluator.results_df.iloc[0]["extra_col"])  # Should be None

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

        result = evaluator.evaluate_sample(sample)

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
