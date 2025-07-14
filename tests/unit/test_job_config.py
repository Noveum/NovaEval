"""
Unit tests for job configuration functionality.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, mock_open, patch

import pytest
import yaml

from novaeval.config.job_config import (
    DatasetFactory,
    JobConfigLoader,
    JobRunner,
    ModelFactory,
    ScorerFactory,
    main,
)
from novaeval.config.schema import (
    DatasetType,
    EvaluationJobConfig,
    ModelProvider,
    ScorerType,
)


class TestJobConfigLoader:
    """Test cases for JobConfigLoader."""

    def test_load_from_file_success(self) -> None:
        """Test successful loading from YAML file."""
        config_data = {
            "name": "test_job",
            "models": [{"provider": "openai", "model_name": "gpt-3.5-turbo"}],
            "datasets": [{"type": "mmlu", "subset": "abstract_algebra"}],
            "scorers": [{"type": "accuracy"}],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = JobConfigLoader.load_from_file(config_path)
            assert config.name == "test_job"
            assert len(config.models) == 1
            assert len(config.datasets) == 1
            assert len(config.scorers) == 1
        finally:
            Path(config_path).unlink()

    def test_load_from_file_not_found(self) -> None:
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            JobConfigLoader.load_from_file("/nonexistent/config.yaml")

    def test_load_from_file_invalid_yaml(self) -> None:
        """Test loading invalid YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid YAML"):
                JobConfigLoader.load_from_file(config_path)
        finally:
            Path(config_path).unlink()

    def test_load_from_file_invalid_config(self) -> None:
        """Test loading file with invalid configuration."""
        config_data = {"invalid_field": "value"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid configuration"):
                JobConfigLoader.load_from_file(config_path)
        finally:
            Path(config_path).unlink()

    def test_load_from_dict_success(self) -> None:
        """Test successful loading from dictionary."""
        config_dict = {
            "name": "test_job",
            "models": [{"provider": "openai", "model_name": "gpt-3.5-turbo"}],
            "datasets": [{"type": "mmlu", "subset": "abstract_algebra"}],
            "scorers": [{"type": "accuracy"}],
        }

        config = JobConfigLoader.load_from_dict(config_dict)
        assert config.name == "test_job"
        assert len(config.models) == 1

    def test_load_from_dict_invalid(self) -> None:
        """Test loading invalid dictionary."""
        with pytest.raises(ValueError, match="Invalid configuration"):
            JobConfigLoader.load_from_dict({"invalid": "config"})

    def test_load_from_file_with_path_object(self) -> None:
        """Test loading from Path object."""
        config_data = {
            "name": "test_job",
            "models": [{"provider": "openai", "model_name": "gpt-3.5-turbo"}],
            "datasets": [{"type": "mmlu", "subset": "abstract_algebra"}],
            "scorers": [{"type": "accuracy"}],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)

        try:
            config = JobConfigLoader.load_from_file(config_path)
            assert config.name == "test_job"
        finally:
            config_path.unlink()


class TestModelFactory:
    """Test cases for ModelFactory."""

    def test_create_openai_model(self) -> None:
        """Test creating OpenAI model."""
        model_config = Mock()
        model_config.provider = ModelProvider.OPENAI
        model_config.model_name = "gpt-3.5-turbo"
        model_config.api_key = "test-key"
        model_config.api_base = None
        model_config.temperature = 0.7
        model_config.max_tokens = 1000
        model_config.timeout = 30
        model_config.retry_attempts = 3
        model_config.additional_params = {}

        with patch("novaeval.models.openai.OpenAIModel") as mock_model:
            ModelFactory.create_model(model_config)

            mock_model.assert_called_once_with(
                model_name="gpt-3.5-turbo",
                api_key="test-key",
                api_base=None,
                temperature=0.7,
                max_tokens=1000,
                timeout=30,
                retry_attempts=3,
            )

    def test_create_anthropic_model(self) -> None:
        """Test creating Anthropic model."""
        model_config = Mock()
        model_config.provider = ModelProvider.ANTHROPIC
        model_config.model_name = "claude-3-opus"
        model_config.api_key = "test-key"
        model_config.temperature = 0.5
        model_config.max_tokens = 2000
        model_config.timeout = 60
        model_config.retry_attempts = 2
        model_config.additional_params = {}

        with patch("novaeval.models.anthropic.AnthropicModel") as mock_model:
            ModelFactory.create_model(model_config)

            mock_model.assert_called_once_with(
                model_name="claude-3-opus",
                api_key="test-key",
                temperature=0.5,
                max_tokens=2000,
                timeout=60,
                retry_attempts=2,
            )

    def test_create_noveum_model(self) -> None:
        """Test creating Noveum model (fallback to OpenAI)."""
        model_config = Mock()
        model_config.provider = ModelProvider.NOVEUM
        model_config.model_name = "custom-model"
        model_config.api_key = "noveum-key"
        model_config.api_base = "https://api.noveum.ai"
        model_config.temperature = 0.8
        model_config.max_tokens = 1500
        model_config.timeout = 45
        model_config.retry_attempts = 3
        model_config.additional_params = {}

        with patch("novaeval.models.openai.OpenAIModel") as mock_model:
            ModelFactory.create_model(model_config)

            mock_model.assert_called_once_with(
                model_name="custom-model",
                api_key="noveum-key",
                api_base="https://api.noveum.ai",
                temperature=0.8,
                max_tokens=1500,
                timeout=45,
                retry_attempts=3,
            )

    def test_create_model_with_env_vars(self) -> None:
        """Test creating model with environment variables."""
        model_config = Mock()
        model_config.provider = ModelProvider.OPENAI
        model_config.model_name = "gpt-4"
        model_config.api_key = None
        model_config.api_base = None
        model_config.temperature = 0.0
        model_config.max_tokens = 500
        model_config.timeout = 30
        model_config.retry_attempts = 3
        model_config.additional_params = {}

        with (
            patch("novaeval.models.openai.OpenAIModel") as mock_model,
            patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}),
        ):
            ModelFactory.create_model(model_config)

            mock_model.assert_called_once_with(
                model_name="gpt-4",
                api_key="env-key",
                api_base=None,
                temperature=0.0,
                max_tokens=500,
                timeout=30,
                retry_attempts=3,
            )

    def test_create_model_unsupported_provider(self) -> None:
        """Test creating model with unsupported provider."""
        model_config = Mock()
        model_config.provider = "unsupported"

        with pytest.raises(ValueError, match="Unsupported model provider"):
            ModelFactory.create_model(model_config)


class TestDatasetFactory:
    """Test cases for DatasetFactory."""

    def test_create_mmlu_dataset(self) -> None:
        """Test creating MMLU dataset."""
        dataset_config = Mock()
        dataset_config.type = DatasetType.MMLU
        dataset_config.subset = "abstract_algebra"
        dataset_config.split = "test"
        dataset_config.limit = 100
        dataset_config.shuffle = True
        dataset_config.seed = 42

        with patch("novaeval.datasets.mmlu.MMLUDataset") as mock_dataset:
            DatasetFactory.create_dataset(dataset_config)

            mock_dataset.assert_called_once_with(
                subset="abstract_algebra",
                split="test",
                limit=100,
                shuffle=True,
                seed=42,
            )

    def test_create_huggingface_dataset(self) -> None:
        """Test creating HuggingFace dataset."""
        dataset_config = Mock()
        dataset_config.type = DatasetType.HUGGINGFACE
        dataset_config.name = "squad"
        dataset_config.subset = None
        dataset_config.split = "validation"
        dataset_config.limit = 50
        dataset_config.shuffle = False
        dataset_config.seed = 123

        with patch("novaeval.datasets.huggingface.HuggingFaceDataset") as mock_dataset:
            DatasetFactory.create_dataset(dataset_config)

            mock_dataset.assert_called_once_with(
                dataset_name="squad",
                subset=None,
                split="validation",
                limit=50,
                shuffle=False,
                seed=123,
            )

    def test_create_custom_dataset(self) -> None:
        """Test creating custom dataset."""
        dataset_config = Mock()
        dataset_config.type = DatasetType.CUSTOM
        dataset_config.path = "/path/to/data.jsonl"
        dataset_config.limit = 200
        dataset_config.shuffle = True
        dataset_config.seed = 456
        dataset_config.preprocessing = None

        with patch("novaeval.datasets.custom.CustomDataset") as mock_dataset:
            DatasetFactory.create_dataset(dataset_config)

            mock_dataset.assert_called_once_with(
                data_source="/path/to/data.jsonl",
                format="custom",
                limit=200,
                shuffle=True,
                seed=456,
                preprocessing=None,
            )

    def test_create_json_dataset(self) -> None:
        """Test creating JSON dataset."""
        dataset_config = Mock()
        dataset_config.type = DatasetType.JSON
        dataset_config.path = "/path/to/data.json"
        dataset_config.limit = 150
        dataset_config.shuffle = False
        dataset_config.seed = 789
        dataset_config.preprocessing = {"normalize": True}

        with patch("novaeval.datasets.custom.CustomDataset") as mock_dataset:
            DatasetFactory.create_dataset(dataset_config)

            mock_dataset.assert_called_once_with(
                data_source="/path/to/data.json",
                format="json",
                limit=150,
                shuffle=False,
                seed=789,
                preprocessing={"normalize": True},
            )

    def test_create_dataset_unsupported_type(self) -> None:
        """Test creating dataset with unsupported type."""
        dataset_config = Mock()
        dataset_config.type = "unsupported"

        with pytest.raises(ValueError, match="Unsupported dataset type"):
            DatasetFactory.create_dataset(dataset_config)


class TestScorerFactory:
    """Test cases for ScorerFactory."""

    def test_create_accuracy_scorer(self) -> None:
        """Test creating accuracy scorer."""
        scorer_config = Mock()
        scorer_config.type = ScorerType.ACCURACY
        scorer_config.threshold = 0.8
        scorer_config.parameters = {"case_sensitive": False}

        model = Mock()

        with patch("novaeval.scorers.accuracy.AccuracyScorer") as mock_scorer:
            ScorerFactory.create_scorer(scorer_config, model)

            mock_scorer.assert_called_once_with(threshold=0.8, case_sensitive=False)

    def test_create_g_eval_scorer(self) -> None:
        """Test creating G-Eval scorer."""
        scorer_config = Mock()
        scorer_config.type = ScorerType.G_EVAL
        scorer_config.threshold = 0.7
        scorer_config.parameters = {"criteria": "coherence"}

        model = Mock()

        with patch("novaeval.scorers.g_eval.GEvalScorer") as mock_scorer:
            ScorerFactory.create_scorer(scorer_config, model)

            mock_scorer.assert_called_once_with(
                model=model, threshold=0.7, criteria="coherence"
            )

    def test_create_rag_answer_relevancy_scorer(self) -> None:
        """Test creating RAG answer relevancy scorer."""
        scorer_config = Mock()
        scorer_config.type = ScorerType.RAG_ANSWER_RELEVANCY
        scorer_config.threshold = 0.75
        scorer_config.parameters = {"use_embeddings": True}

        model = Mock()

        with patch("novaeval.scorers.rag.AnswerRelevancyScorer") as mock_scorer:
            ScorerFactory.create_scorer(scorer_config, model)

            mock_scorer.assert_called_once_with(
                model=model, threshold=0.75, use_embeddings=True
            )

    def test_create_rag_faithfulness_scorer(self) -> None:
        """Test creating RAG faithfulness scorer."""
        scorer_config = Mock()
        scorer_config.type = ScorerType.RAG_FAITHFULNESS
        scorer_config.threshold = 0.85
        scorer_config.parameters = {"strict_mode": True}

        model = Mock()

        with patch("novaeval.scorers.rag.FaithfulnessScorer") as mock_scorer:
            ScorerFactory.create_scorer(scorer_config, model)

            mock_scorer.assert_called_once_with(
                model=model, threshold=0.85, strict_mode=True
            )

    def test_create_ragas_scorer(self) -> None:
        """Test creating RAGAS scorer."""
        scorer_config = Mock()
        scorer_config.type = ScorerType.RAGAS
        scorer_config.threshold = 0.9
        scorer_config.parameters = {"metrics": ["answer_relevancy", "faithfulness"]}

        model = Mock()

        with patch("novaeval.scorers.rag.RAGASScorer") as mock_scorer:
            ScorerFactory.create_scorer(scorer_config, model)

            mock_scorer.assert_called_once_with(
                model=model,
                threshold=0.9,
                metrics=["answer_relevancy", "faithfulness"],
            )

    def test_create_conversational_metrics_scorer(self) -> None:
        """Test creating conversational metrics scorer."""
        scorer_config = Mock()
        scorer_config.type = ScorerType.CONVERSATIONAL_METRICS
        scorer_config.threshold = 0.8
        scorer_config.parameters = {"turn_level": True}

        model = Mock()

        with patch(
            "novaeval.scorers.conversational.ConversationalMetricsScorer"
        ) as mock_scorer:
            ScorerFactory.create_scorer(scorer_config, model)

            mock_scorer.assert_called_once_with(
                model=model, threshold=0.8, turn_level=True
            )

    def test_create_scorer_unsupported_type(self) -> None:
        """Test creating scorer with unsupported type."""
        scorer_config = Mock()
        scorer_config.type = "unsupported"

        model = Mock()

        with pytest.raises(ValueError, match="Unsupported scorer type"):
            ScorerFactory.create_scorer(scorer_config, model)


class TestJobRunner:
    """Test cases for JobRunner."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_config = Mock(spec=EvaluationJobConfig)
        self.mock_config.name = "test_job"
        self.mock_config.environment = {"TEST_VAR": "test_value"}

        # Create proper mock models with providers
        mock_model = Mock()
        mock_model.provider = ModelProvider.OPENAI
        mock_model.model_name = "gpt-3.5-turbo"
        self.mock_config.models = [mock_model]

        # Create proper mock datasets
        mock_dataset = Mock()
        mock_dataset.type = DatasetType.MMLU
        self.mock_config.datasets = [mock_dataset]

        # Create proper mock scorers
        mock_scorer = Mock()
        mock_scorer.type = ScorerType.ACCURACY
        self.mock_config.scorers = [mock_scorer]

        self.mock_config.parallel_models = False
        self.mock_config.output = Mock()
        self.mock_config.output.directory = "/tmp/output"
        self.mock_config.output.filename_prefix = "test_results"
        self.mock_config.output.formats = ["json"]
        self.mock_config.ci = Mock()
        self.mock_config.ci.fail_threshold = 0.7
        self.mock_config.ci.fail_on_threshold = True
        self.mock_config.dict.return_value = {"test": "config"}

    def test_init(self) -> None:
        """Test JobRunner initialization."""
        with patch.dict(os.environ, clear=True):
            runner = JobRunner(self.mock_config)

            assert runner.config == self.mock_config
            assert runner.start_time is None
            assert runner.end_time is None
            assert runner.results == {}
            assert os.environ["TEST_VAR"] == "test_value"

    @pytest.mark.asyncio
    async def test_run_success_sequential(self) -> None:
        """Test successful job run in sequential mode."""
        self.mock_config.parallel_models = False

        with (
            patch.object(ModelFactory, "create_model") as mock_create_model,
            patch.object(DatasetFactory, "create_dataset") as mock_create_dataset,
            patch.object(JobRunner, "_evaluate_model_dataset") as mock_evaluate,
            patch.object(JobRunner, "_compile_results") as mock_compile,
            patch.object(JobRunner, "_generate_outputs"),
            patch.object(JobRunner, "_check_ci_requirements") as mock_ci,
        ):
            mock_create_model.return_value = Mock()
            mock_create_dataset.return_value = Mock()
            mock_evaluate.return_value = {"test": "result"}
            mock_compile.return_value = {"compiled": "results"}
            mock_ci.return_value = {"passed": True}

            runner = JobRunner(self.mock_config)
            result = await runner.run()

            assert result["status"] == "completed"
            assert result["job_name"] == "test_job"
            assert "duration" in result
            assert result["ci_status"]["passed"] is True
            mock_evaluate.assert_called_once()
            mock_compile.assert_called_once()
            mock_ci.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_success_parallel(self) -> None:
        """Test successful job run in parallel mode."""
        self.mock_config.parallel_models = True

        with (
            patch.object(ModelFactory, "create_model") as mock_create_model,
            patch.object(DatasetFactory, "create_dataset") as mock_create_dataset,
            patch.object(JobRunner, "_evaluate_model_dataset") as mock_evaluate,
            patch.object(JobRunner, "_compile_results") as mock_compile,
            patch.object(JobRunner, "_generate_outputs"),
            patch.object(JobRunner, "_check_ci_requirements") as mock_ci,
            patch("asyncio.gather", new_callable=AsyncMock) as mock_gather,
        ):
            mock_create_model.return_value = Mock()
            mock_create_dataset.return_value = Mock()
            mock_evaluate.return_value = {"test": "result"}
            mock_gather.return_value = [{"test": "result"}]
            mock_compile.return_value = {"compiled": "results"}
            mock_ci.return_value = {"passed": True}

            runner = JobRunner(self.mock_config)
            result = await runner.run()

            assert result["status"] == "completed"
            assert result["job_name"] == "test_job"
            mock_gather.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_evaluation_exception(self) -> None:
        """Test job run with evaluation exception in sequential mode."""
        self.mock_config.parallel_models = False

        with (
            patch.object(ModelFactory, "create_model") as mock_create_model,
            patch.object(DatasetFactory, "create_dataset") as mock_create_dataset,
            patch.object(JobRunner, "_evaluate_model_dataset") as mock_evaluate,
            patch.object(JobRunner, "_compile_results") as mock_compile,
            patch.object(JobRunner, "_generate_outputs"),
            patch.object(JobRunner, "_check_ci_requirements") as mock_ci,
        ):
            mock_create_model.return_value = Mock()
            mock_create_dataset.return_value = Mock()
            mock_evaluate.side_effect = RuntimeError("Evaluation failed")
            mock_compile.return_value = {"compiled": "results"}
            mock_ci.return_value = {"passed": True}

            runner = JobRunner(self.mock_config)
            result = await runner.run()

            # Should continue despite evaluation errors in sequential mode
            assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_run_with_parallel_exceptions(self) -> None:
        """Test job run with exceptions in parallel mode."""
        self.mock_config.parallel_models = True

        with (
            patch.object(ModelFactory, "create_model") as mock_create_model,
            patch.object(DatasetFactory, "create_dataset") as mock_create_dataset,
            patch.object(JobRunner, "_evaluate_model_dataset") as mock_evaluate,
            patch("asyncio.gather", new_callable=AsyncMock) as mock_gather,
            patch.object(JobRunner, "_compile_results") as mock_compile,
            patch.object(JobRunner, "_generate_outputs"),
            patch.object(JobRunner, "_check_ci_requirements") as mock_ci,
        ):
            mock_create_model.return_value = Mock()
            mock_create_dataset.return_value = Mock()
            mock_evaluate.return_value = {"test": "result"}
            mock_gather.return_value = [RuntimeError("Test error"), {"test": "result"}]
            mock_compile.return_value = {"compiled": "results"}
            mock_ci.return_value = {"passed": True}

            runner = JobRunner(self.mock_config)
            result = await runner.run()

            assert result["status"] == "completed"
            # Should handle exceptions gracefully

    @pytest.mark.asyncio
    async def test_run_failure(self) -> None:
        """Test job run failure."""
        # Make it fail during model creation
        with patch.object(
            ModelFactory, "create_model", side_effect=Exception("Fatal error")
        ):
            runner = JobRunner(self.mock_config)
            result = await runner.run()

            assert result["status"] == "failed"
            assert "Fatal error" in result["error"]
            assert result["ci_status"]["passed"] is False

    @pytest.mark.asyncio
    async def test_evaluate_model_dataset(self) -> None:
        """Test single model-dataset evaluation."""
        model_config = Mock()
        model_config.provider = ModelProvider.OPENAI
        model_config.model_name = "gpt-3.5-turbo"

        dataset_config = Mock()
        dataset_config.type = DatasetType.MMLU
        dataset_config.name = "test_dataset"

        model = Mock()
        dataset = Mock()

        scorer_config = Mock()
        scorer_config.type = ScorerType.ACCURACY
        scorer_config.name = "accuracy_scorer"
        scorer_config.threshold = 0.8
        scorer_config.weight = 1.0

        self.mock_config.scorers = [scorer_config]

        mock_evaluation_result = Mock()
        mock_evaluation_result.dict.return_value = {"score": 0.85}

        with (
            patch.object(ScorerFactory, "create_scorer") as mock_create_scorer,
            patch("novaeval.config.job_config.Evaluator") as mock_evaluator_class,
        ):
            mock_scorer = Mock()
            mock_create_scorer.return_value = mock_scorer

            mock_evaluator = Mock()
            mock_evaluator.run_evaluation = AsyncMock(
                return_value=mock_evaluation_result
            )
            mock_evaluator_class.return_value = mock_evaluator

            runner = JobRunner(self.mock_config)
            result = await runner._evaluate_model_dataset(
                model_config, model, dataset_config, dataset
            )

            assert result["model"]["provider"] == ModelProvider.OPENAI
            assert result["model"]["name"] == "gpt-3.5-turbo"
            assert result["dataset"]["type"] == DatasetType.MMLU
            assert result["results"]["score"] == 0.85

            mock_create_scorer.assert_called_once_with(scorer_config, model)
            mock_evaluator_class.assert_called_once_with(
                models=[model], dataset=dataset, scorers=[mock_scorer]
            )

    def test_compile_results(self) -> None:
        """Test results compilation."""
        evaluation_results = [
            {
                "model": {"provider": "openai", "name": "gpt-3.5-turbo"},
                "dataset": {"name": "test_dataset"},
                "results": {"overall_score": 0.85},
            },
            {
                "model": {"provider": "anthropic", "name": "claude-3"},
                "dataset": {"name": "test_dataset"},
                "results": {"overall_score": 0.90},
            },
        ]

        runner = JobRunner(self.mock_config)
        runner.start_time = 1000.0
        runner.end_time = 1010.0

        result = runner._compile_results(evaluation_results)

        assert result["summary"]["total_evaluations"] == 2
        assert result["summary"]["models_evaluated"] == 2
        assert result["summary"]["datasets_used"] == 1
        assert result["summary"]["average_score"] == 0.875
        assert result["summary"]["min_score"] == 0.85
        assert result["summary"]["max_score"] == 0.90
        assert result["execution_metadata"]["duration"] == 10.0

    def test_compile_results_no_scores(self) -> None:
        """Test results compilation with no overall scores."""
        evaluation_results = [
            {
                "model": {"provider": "openai", "name": "gpt-3.5-turbo"},
                "dataset": {"name": "test_dataset"},
                "results": {"some_metric": 0.85},
            }
        ]

        runner = JobRunner(self.mock_config)
        result = runner._compile_results(evaluation_results)

        assert result["summary"]["average_score"] == 0.0
        assert result["summary"]["min_score"] == 0.0
        assert result["summary"]["max_score"] == 0.0

    @pytest.mark.asyncio
    async def test_generate_outputs_json(self) -> None:
        """Test JSON output generation."""
        results = {"test": "results"}

        with (
            patch("pathlib.Path.mkdir") as mock_mkdir,
            patch("builtins.open", mock_open()) as mock_file,
            patch("json.dump") as mock_json_dump,
            patch("time.time", return_value=1234567890),
        ):
            runner = JobRunner(self.mock_config)
            await runner._generate_outputs(results)

            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
            mock_file.assert_called_once()
            mock_json_dump.assert_called_once_with(
                results, mock_file.return_value.__enter__(), indent=2, default=str
            )

    @pytest.mark.asyncio
    async def test_generate_outputs_html(self) -> None:
        """Test HTML output generation."""
        self.mock_config.output.formats = ["html"]
        results = {"test": "results"}

        with (
            patch("pathlib.Path.mkdir"),
            patch("builtins.open", mock_open()) as mock_file,
            patch.object(JobRunner, "_generate_html_report") as mock_html,
        ):
            mock_html.return_value = "<html>test</html>"

            runner = JobRunner(self.mock_config)
            await runner._generate_outputs(results)

            mock_html.assert_called_once_with(results)
            mock_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_outputs_junit_xml(self) -> None:
        """Test JUnit XML output generation."""
        self.mock_config.output.formats = ["junit_xml"]
        results = {"test": "results"}

        with (
            patch("pathlib.Path.mkdir"),
            patch("builtins.open", mock_open()) as mock_file,
            patch.object(JobRunner, "_generate_junit_xml") as mock_xml,
        ):
            mock_xml.return_value = "<xml>test</xml>"

            runner = JobRunner(self.mock_config)
            await runner._generate_outputs(results)

            mock_xml.assert_called_once_with(results)
            mock_file.assert_called_once()

    def test_check_ci_requirements_pass(self) -> None:
        """Test CI requirements check with passing scores."""
        results = {
            "raw_results": [
                {
                    "model": {"name": "gpt-3.5-turbo"},
                    "dataset": {"name": "test_dataset"},
                    "results": {"overall_score": 0.8},
                }
            ],
            "summary": {"average_score": 0.8},
        }

        runner = JobRunner(self.mock_config)
        ci_status = runner._check_ci_requirements(results)

        assert ci_status["passed"] is True
        assert ci_status["passed_evaluations"] == 1
        assert ci_status["failed_evaluations"] == 0
        assert ci_status["recommendation"] == "Deploy"

    def test_check_ci_requirements_fail(self) -> None:
        """Test CI requirements check with failing scores."""
        results = {
            "raw_results": [
                {
                    "model": {"name": "gpt-3.5-turbo"},
                    "dataset": {"name": "test_dataset"},
                    "results": {"overall_score": 0.6},
                }
            ],
            "summary": {"average_score": 0.6},
        }

        runner = JobRunner(self.mock_config)
        ci_status = runner._check_ci_requirements(results)

        assert ci_status["passed"] is False
        assert ci_status["passed_evaluations"] == 0
        assert ci_status["failed_evaluations"] == 1
        assert "Do not deploy" in ci_status["recommendation"]

    def test_check_ci_requirements_no_fail_on_threshold(self) -> None:
        """Test CI requirements when fail_on_threshold is False."""
        self.mock_config.ci.fail_on_threshold = False

        results = {
            "raw_results": [
                {
                    "model": {"name": "gpt-3.5-turbo"},
                    "dataset": {"name": "test_dataset"},
                    "results": {"overall_score": 0.6},
                }
            ],
            "summary": {"average_score": 0.6},
        }

        runner = JobRunner(self.mock_config)
        ci_status = runner._check_ci_requirements(results)

        assert ci_status["passed"] is True

    def test_generate_html_report(self) -> None:
        """Test HTML report generation."""
        results = {
            "summary": {
                "total_evaluations": 2,
                "models_evaluated": 1,
                "datasets_used": 1,
                "average_score": 0.85,
                "min_score": 0.8,
                "max_score": 0.9,
            }
        }

        with patch("time.strftime", return_value="2023-01-01 12:00:00"):
            runner = JobRunner(self.mock_config)
            html_content = runner._generate_html_report(results)

            assert "test_job" in html_content
            assert "2023-01-01 12:00:00" in html_content
            assert "Total Evaluations: 2" in html_content
            assert "Average Score: 0.850" in html_content

    def test_generate_junit_xml(self) -> None:
        """Test JUnit XML generation."""
        results = {
            "raw_results": [
                {
                    "model": {"name": "gpt-3.5-turbo"},
                    "dataset": {"name": "test_dataset"},
                    "results": {"overall_score": 0.8},
                },
                {
                    "model": {"name": "claude-3"},
                    "dataset": {"name": "test_dataset"},
                    "results": {"overall_score": 0.6},
                },
            ],
            "execution_metadata": {"duration": 10.5},
        }

        runner = JobRunner(self.mock_config)
        xml_content = runner._generate_junit_xml(results)

        assert 'tests="2"' in xml_content
        assert 'failures="1"' in xml_content
        assert 'time="10.5"' in xml_content
        assert "gpt-3.5-turbo_on_test_dataset" in xml_content
        assert "claude-3_on_test_dataset" in xml_content
        assert "failure message" in xml_content


class TestMainCLI:
    """Test cases for main CLI function."""

    def test_main_dry_run(self) -> None:
        """Test main function with dry run."""
        mock_config = Mock()
        mock_config.name = "test_job"
        mock_config.models = [Mock()]
        mock_config.datasets = [Mock()]
        mock_config.scorers = [Mock()]

        test_args = ["test_script", "config.yaml", "--dry-run"]

        with (
            patch("sys.argv", test_args),
            patch.object(JobConfigLoader, "load_from_file") as mock_load,
            patch("builtins.print") as mock_print,
        ):
            mock_load.return_value = mock_config

            result = main()

            assert result == 0
            mock_load.assert_called_once_with("config.yaml")
            mock_print.assert_called()

    def test_main_successful_run(self) -> None:
        """Test main function with successful run."""
        mock_config = Mock()
        mock_config.environment = {}
        mock_job_result = {
            "status": "completed",
            "job_name": "test_job",
            "duration": 10.5,
            "ci_status": {"passed": True, "recommendation": "Deploy"},
        }

        test_args = ["test_script", "config.yaml"]

        with (
            patch("sys.argv", test_args),
            patch.object(JobConfigLoader, "load_from_file") as mock_load,
            patch("novaeval.config.job_config.JobRunner") as mock_job_runner_class,
            patch("builtins.print") as mock_print,
        ):
            mock_load.return_value = mock_config
            mock_job_runner = Mock()
            mock_job_runner.run = AsyncMock(return_value=mock_job_result)
            mock_job_runner_class.return_value = mock_job_runner

            result = main()

            assert result == 0
            mock_job_runner_class.assert_called_once_with(mock_config)
            mock_print.assert_called()

    def test_main_failed_run(self) -> None:
        """Test main function with failed run."""
        mock_config = Mock()
        mock_job_result = {
            "status": "completed",
            "job_name": "test_job",
            "duration": 5.2,
            "ci_status": {"passed": False, "recommendation": "Do not deploy"},
        }

        test_args = ["test_script", "config.yaml"]

        with (
            patch("sys.argv", test_args),
            patch.object(JobConfigLoader, "load_from_file") as mock_load,
            patch("asyncio.run") as mock_asyncio_run,
            patch("builtins.print") as mock_print,
        ):
            mock_load.return_value = mock_config
            mock_asyncio_run.return_value = mock_job_result

            result = main()

            assert result == 1
            mock_print.assert_called()

    def test_main_job_failure(self) -> None:
        """Test main function with job failure."""
        mock_config = Mock()
        mock_job_result = {"status": "failed", "error": "Something went wrong"}

        test_args = ["test_script", "config.yaml"]

        with (
            patch("sys.argv", test_args),
            patch.object(JobConfigLoader, "load_from_file") as mock_load,
            patch("asyncio.run") as mock_asyncio_run,
            patch("builtins.print") as mock_print,
        ):
            mock_load.return_value = mock_config
            mock_asyncio_run.return_value = mock_job_result

            result = main()

            assert result == 1
            mock_print.assert_called()

    def test_main_exception(self) -> None:
        """Test main function with exception."""
        test_args = ["test_script", "config.yaml"]

        with (
            patch("sys.argv", test_args),
            patch.object(
                JobConfigLoader, "load_from_file", side_effect=Exception("Config error")
            ),
            patch("builtins.print") as mock_print,
        ):
            result = main()

            assert result == 1
            mock_print.assert_called()

    def test_main_verbose_flag(self) -> None:
        """Test main function with verbose flag."""
        mock_config = Mock()
        mock_config.name = "test_job"
        mock_config.models = [Mock()]
        mock_config.datasets = [Mock()]
        mock_config.scorers = [Mock()]

        test_args = ["test_script", "config.yaml", "--verbose", "--dry-run"]

        with (
            patch("sys.argv", test_args),
            patch.object(JobConfigLoader, "load_from_file") as mock_load,
            patch("builtins.print"),
        ):
            mock_load.return_value = mock_config

            result = main()

            assert result == 0
            mock_load.assert_called_once_with("config.yaml")
