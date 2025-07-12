"""
Command-line interface for NovaEval.

This module provides the main CLI entry point for running evaluations.
"""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from novaeval import __version__
from novaeval.evaluators.standard import Evaluator
from novaeval.utils.config import Config
from novaeval.utils.logging import setup_logging

console = Console()


@click.group()
@click.version_option(version=__version__)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help="Set the logging level",
)
@click.option("--log-file", type=click.Path(), help="Log file path")
def cli(log_level: str, log_file: Optional[str]) -> None:
    """
    NovaEval: A comprehensive AI model evaluation framework.
    """
    setup_logging(level=log_level, log_file=log_file)


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option(
    "--output-dir", "-o", type=click.Path(), help="Output directory for results"
)
@click.option(
    "--dry-run", is_flag=True, help="Validate configuration without running evaluation"
)
def run(config_file: str, output_dir: Optional[str], dry_run: bool) -> None:
    """
    Run evaluation from configuration file.

    CONFIG_FILE: Path to YAML or JSON configuration file
    """
    try:
        # Load configuration
        config = Config.load(config_file)

        if output_dir:
            config.set("output.directory", output_dir)

        if dry_run:
            console.print("[green]✓[/green] Configuration is valid")
            _display_config_summary(config)
            return

        # Create and run evaluator
        evaluator = Evaluator.from_config(config_file)

        console.print("[blue]Starting evaluation...[/blue]")
        results = evaluator.run()

        console.print("[green]✓[/green] Evaluation completed successfully")
        _display_results_summary(results)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command()
@click.option(
    "--dataset", "-d", required=True, help="Dataset name (e.g., 'mmlu', 'hellaswag')"
)
@click.option(
    "--model",
    "-m",
    required=True,
    multiple=True,
    help="Model name (can specify multiple)",
)
@click.option(
    "--scorer",
    "-s",
    multiple=True,
    default=["accuracy"],
    help="Scorer name (can specify multiple)",
)
@click.option("--num-samples", "-n", type=int, help="Number of samples to evaluate")
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="./results",
    help="Output directory for results",
)
def quick(
    dataset: str,
    model: tuple,
    scorer: tuple,
    num_samples: Optional[int],
    output_dir: str,
) -> None:
    """
    Quick evaluation with minimal configuration.
    """
    try:
        # Create configuration
        {
            "dataset": {"type": dataset, "num_samples": num_samples},
            "models": [{"type": m} for m in model],
            "scorers": [{"type": s} for s in scorer],
            "output": {"directory": output_dir},
        }

        console.print("[blue]Starting quick evaluation...[/blue]")
        console.print(f"Dataset: {dataset}")
        console.print(f"Models: {', '.join(model)}")
        console.print(f"Scorers: {', '.join(scorer)}")

        # This would create and run the evaluator
        # For now, it's a placeholder
        console.print("[yellow]Quick evaluation not yet implemented[/yellow]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command()
def list_datasets() -> None:
    """List available datasets."""
    datasets = [
        ("mmlu", "Massive Multitask Language Understanding"),
        ("hellaswag", "Commonsense reasoning"),
        ("truthful_qa", "Truthfulness evaluation"),
        ("squad", "Reading comprehension"),
        ("glue", "General Language Understanding"),
    ]

    table = Table(title="Available Datasets")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")

    for name, description in datasets:
        table.add_row(name, description)

    console.print(table)


@cli.command()
def list_models() -> None:
    """List available model providers."""
    models = [
        ("openai", "OpenAI GPT models", "gpt-4, gpt-3.5-turbo"),
        ("anthropic", "Anthropic Claude models", "claude-3-opus, claude-3-sonnet"),
        ("bedrock", "AWS Bedrock models", "Various providers"),
        ("noveum", "Noveum AI Gateway", "Multiple providers"),
    ]

    table = Table(title="Available Model Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Examples", style="dim")

    for provider, description, examples in models:
        table.add_row(provider, description, examples)

    console.print(table)


@cli.command()
def list_scorers() -> None:
    """List available scorers."""
    scorers = [
        ("accuracy", "Classification accuracy"),
        ("exact_match", "Exact string matching"),
        ("f1", "Token-level F1 score"),
        ("semantic_similarity", "Embedding-based similarity"),
        ("bert_score", "BERT-based evaluation"),
        ("code_execution", "Code execution validation"),
        ("llm_judge", "LLM-as-a-judge scoring"),
    ]

    table = Table(title="Available Scorers")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")

    for name, description in scorers:
        table.add_row(name, description)

    console.print(table)


@cli.command()
@click.argument("output_file", type=click.Path())
def generate_config(output_file: str) -> None:
    """
    Generate a sample configuration file.

    OUTPUT_FILE: Path for the generated configuration file
    """
    sample_config = {
        "dataset": {"type": "mmlu", "subset": "abstract_algebra", "num_samples": 100},
        "models": [{"type": "openai", "model_name": "gpt-4", "temperature": 0.0}],
        "scorers": [{"type": "accuracy"}],
        "output": {"directory": "./results", "formats": ["json", "csv", "html"]},
        "evaluation": {"max_workers": 4, "batch_size": 1},
    }

    config = Config(sample_config)

    # Determine format from file extension
    output_path = Path(output_file)
    if output_path.suffix.lower() in [".yaml", ".yml"]:
        config.save(output_file, format="yaml")
    else:
        config.save(output_file, format="json")

    console.print(f"[green]✓[/green] Sample configuration saved to {output_file}")


def _display_config_summary(config: Config) -> None:
    """Display configuration summary."""
    table = Table(title="Configuration Summary")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")

    # Add key configuration items
    dataset_info = config.get("dataset", {})
    table.add_row("Dataset", dataset_info.get("type", "Unknown"))

    models = config.get("models", [])
    model_names = [m.get("model_name", m.get("type", "Unknown")) for m in models]
    table.add_row("Models", ", ".join(model_names))

    scorers = config.get("scorers", [])
    scorer_names = [s.get("type", "Unknown") for s in scorers]
    table.add_row("Scorers", ", ".join(scorer_names))

    output_dir = config.get("output.directory", "./results")
    table.add_row("Output Directory", str(output_dir))

    console.print(table)


def _display_results_summary(results: dict) -> None:
    """Display results summary."""
    table = Table(title="Evaluation Results")
    table.add_column("Model", style="cyan")
    table.add_column("Scorer", style="white")
    table.add_column("Score", style="green")

    for model_name, model_results in results.get("model_results", {}).items():
        for scorer_name, score_info in model_results.get("scores", {}).items():
            if isinstance(score_info, dict) and "mean" in score_info:
                score = f"{score_info['mean']:.4f}"
            else:
                score = str(score_info)
            table.add_row(model_name, scorer_name, score)

    console.print(table)


def main() -> None:
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
