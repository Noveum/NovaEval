from novaeval import Evaluator
from novaeval.datasets import CustomDataset
from novaeval.models import AnthropicModel, OpenAIModel
from novaeval.scorers.panel_judge import (
    AggregationMethod,
    JudgeConfig,
    PanelOfJudgesScorer,
)

# Set up the models to be evaluated
models_to_evaluate = [
    OpenAIModel(model_name="gpt-4", temperature=0.0),
    AnthropicModel(model_name="claude-3-sonnet-20240229", temperature=0.0),
]

# Configure the judge panel
judge_panel = [
    JudgeConfig(
        model=OpenAIModel(model_name="gpt-4", temperature=0.0),
        weight=1.0,
        name="GPT-4 Judge",
        specialty="accuracy",
    ),
    JudgeConfig(
        model=AnthropicModel(model_name="claude-3-sonnet-20240229", temperature=0.0),
        weight=1.0,
        name="Claude Judge",
        specialty="clarity",
    ),
    JudgeConfig(
        model=OpenAIModel(model_name="gpt-3.5-turbo", temperature=0.1),
        weight=0.8,
        name="GPT-3.5 Judge",
        specialty="completeness",
    ),
]

# Create the panel scorer
panel_scorer = PanelOfJudgesScorer(
    judges=judge_panel,
    aggregation_method=AggregationMethod.WEIGHTED_MEAN,
    threshold=0.8,
    require_consensus=True,
    consensus_threshold=0.7,
    evaluation_criteria="overall quality, factual accuracy, and user helpfulness",
)

# Set up dataset
dataset = CustomDataset(path="./test_data/qa_dataset.jsonl")

# Run evaluation
evaluator = Evaluator(
    dataset=dataset,
    models=models_to_evaluate,
    scorers=[panel_scorer],
    output_dir="./results/panel_evaluation",
)

results = evaluator.run()
