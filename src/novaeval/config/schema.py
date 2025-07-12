"""
Configuration schema for NovaEval evaluation jobs.

This module defines the structure and validation for YAML-based
evaluation job configurations used in CI/CD pipelines.
"""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, validator


class ModelProvider(str, Enum):
    """Supported model providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    NOVEUM = "noveum"
    AWS_BEDROCK = "aws_bedrock"
    AZURE_OPENAI = "azure_openai"
    GOOGLE_VERTEX = "google_vertex"


class DatasetType(str, Enum):
    """Supported dataset types."""

    MMLU = "mmlu"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"
    JSON = "json"
    CSV = "csv"
    JSONL = "jsonl"


class ScorerType(str, Enum):
    """Supported scorer types."""

    ACCURACY = "accuracy"
    G_EVAL = "g_eval"
    RAG_ANSWER_RELEVANCY = "rag_answer_relevancy"
    RAG_FAITHFULNESS = "rag_faithfulness"
    RAG_CONTEXTUAL_PRECISION = "rag_contextual_precision"
    RAG_CONTEXTUAL_RECALL = "rag_contextual_recall"
    RAGAS = "ragas"
    CONVERSATIONAL_KNOWLEDGE_RETENTION = "conversational_knowledge_retention"
    CONVERSATIONAL_COMPLETENESS = "conversational_completeness"
    CONVERSATIONAL_RELEVANCY = "conversational_relevancy"
    CONVERSATIONAL_ROLE_ADHERENCE = "conversational_role_adherence"
    CONVERSATIONAL_METRICS = "conversational_metrics"
    SIMILARITY = "similarity"
    CUSTOM = "custom"


class OutputFormat(str, Enum):
    """Supported output formats."""

    JSON = "json"
    CSV = "csv"
    HTML = "html"
    MARKDOWN = "markdown"
    JUNIT_XML = "junit_xml"


class ModelConfig(BaseModel):
    """Configuration for a model."""

    provider: ModelProvider = Field(description="Model provider")
    model_name: str = Field(description="Name of the model")
    api_key: Optional[str] = Field(
        default=None, description="API key (use env var if not provided)"
    )
    api_base: Optional[str] = Field(default=None, description="Custom API base URL")
    temperature: float = Field(default=0.0, description="Model temperature")
    max_tokens: Optional[int] = Field(
        default=None, description="Maximum tokens to generate"
    )
    timeout: int = Field(default=60, description="Request timeout in seconds")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    additional_params: dict[str, Any] = Field(
        default_factory=dict, description="Additional model parameters"
    )


class DatasetConfig(BaseModel):
    """Configuration for a dataset."""

    type: DatasetType = Field(description="Dataset type")
    name: Optional[str] = Field(
        default=None, description="Dataset name (for HuggingFace)"
    )
    path: Optional[str] = Field(default=None, description="Path to dataset file")
    subset: Optional[str] = Field(default=None, description="Dataset subset")
    split: str = Field(default="test", description="Dataset split to use")
    limit: Optional[int] = Field(default=None, description="Limit number of samples")
    shuffle: bool = Field(default=False, description="Shuffle dataset")
    seed: Optional[int] = Field(default=None, description="Random seed for shuffling")
    preprocessing: dict[str, Any] = Field(
        default_factory=dict, description="Preprocessing configuration"
    )


class ScorerConfig(BaseModel):
    """Configuration for a scorer."""

    type: ScorerType = Field(description="Scorer type")
    name: Optional[str] = Field(default=None, description="Custom scorer name")
    threshold: float = Field(default=0.7, description="Pass/fail threshold")
    weight: float = Field(default=1.0, description="Weight in composite scoring")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Scorer-specific parameters"
    )

    @validator("threshold")
    def validate_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        return v

    @validator("weight")
    def validate_weight(cls, v):
        if v < 0.0:
            raise ValueError("Weight must be non-negative")
        return v


class OutputConfig(BaseModel):
    """Configuration for output generation."""

    formats: list[OutputFormat] = Field(
        default=[OutputFormat.JSON], description="Output formats"
    )
    directory: str = Field(default="./novaeval_results", description="Output directory")
    filename_prefix: str = Field(default="evaluation", description="Filename prefix")
    include_raw_results: bool = Field(
        default=True, description="Include raw evaluation results"
    )
    include_summary: bool = Field(
        default=True, description="Include summary statistics"
    )
    include_metadata: bool = Field(
        default=True, description="Include evaluation metadata"
    )


class CIConfig(BaseModel):
    """Configuration for CI/CD integration."""

    fail_on_threshold: bool = Field(
        default=True, description="Fail CI if any scorer below threshold"
    )
    fail_threshold: float = Field(default=0.7, description="Global fail threshold")
    generate_badges: bool = Field(
        default=True, description="Generate evaluation badges"
    )
    post_to_pr: bool = Field(default=False, description="Post results to PR comments")
    upload_artifacts: bool = Field(
        default=True, description="Upload results as CI artifacts"
    )
    notify_on_regression: bool = Field(
        default=True, description="Notify on performance regression"
    )

    @validator("fail_threshold")
    def validate_fail_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Fail threshold must be between 0.0 and 1.0")
        return v


class EvaluationJobConfig(BaseModel):
    """Complete configuration for an evaluation job."""

    # Job metadata
    name: str = Field(description="Job name")
    description: Optional[str] = Field(default=None, description="Job description")
    version: str = Field(default="1.0", description="Configuration version")

    # Core configuration
    models: list[ModelConfig] = Field(description="Models to evaluate")
    datasets: list[DatasetConfig] = Field(description="Datasets to use")
    scorers: list[ScorerConfig] = Field(description="Scorers to apply")

    # Optional configuration
    output: OutputConfig = Field(
        default_factory=OutputConfig, description="Output configuration"
    )
    ci: CIConfig = Field(default_factory=CIConfig, description="CI/CD configuration")

    # Execution configuration
    parallel_models: bool = Field(
        default=True, description="Evaluate models in parallel"
    )
    parallel_datasets: bool = Field(
        default=True, description="Process datasets in parallel"
    )
    max_workers: int = Field(default=4, description="Maximum number of worker threads")
    timeout: int = Field(default=3600, description="Job timeout in seconds")

    # Environment configuration
    environment: dict[str, str] = Field(
        default_factory=dict, description="Environment variables"
    )

    @validator("models")
    def validate_models(cls, v):
        if not v:
            raise ValueError("At least one model must be specified")
        return v

    @validator("datasets")
    def validate_datasets(cls, v):
        if not v:
            raise ValueError("At least one dataset must be specified")
        return v

    @validator("scorers")
    def validate_scorers(cls, v):
        if not v:
            raise ValueError("At least one scorer must be specified")
        return v


class ConfigSchema:
    """Schema utilities for configuration validation."""

    @staticmethod
    def get_json_schema() -> dict[str, Any]:
        """Get JSON schema for validation."""
        return EvaluationJobConfig.schema()

    @staticmethod
    def validate_config(config_dict: dict[str, Any]) -> EvaluationJobConfig:
        """Validate and parse configuration dictionary."""
        return EvaluationJobConfig(**config_dict)

    @staticmethod
    def get_example_config() -> dict[str, Any]:
        """Get example configuration."""
        return {
            "name": "AI Model Evaluation",
            "description": "Comprehensive evaluation of AI models for production deployment",
            "version": "1.0",
            "models": [
                {
                    "provider": "openai",
                    "model_name": "gpt-4",
                    "temperature": 0.0,
                    "max_tokens": 1000,
                },
                {
                    "provider": "anthropic",
                    "model_name": "claude-3-sonnet-20240229",
                    "temperature": 0.0,
                    "max_tokens": 1000,
                },
            ],
            "datasets": [
                {"type": "mmlu", "subset": "all", "split": "test", "limit": 100},
                {"type": "custom", "path": "./test_data.jsonl", "split": "test"},
            ],
            "scorers": [
                {"type": "accuracy", "threshold": 0.8, "weight": 1.0},
                {
                    "type": "g_eval",
                    "threshold": 0.7,
                    "weight": 1.0,
                    "parameters": {"criteria": "correctness", "use_cot": True},
                },
            ],
            "output": {
                "formats": ["json", "html", "junit_xml"],
                "directory": "./evaluation_results",
                "filename_prefix": "ai_model_eval",
            },
            "ci": {
                "fail_on_threshold": True,
                "fail_threshold": 0.75,
                "generate_badges": True,
                "upload_artifacts": True,
            },
        }
