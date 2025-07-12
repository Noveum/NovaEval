"""
Configuration-based evaluation example for NovaEval.

This example demonstrates how to run evaluations using
configuration files.
"""

from pathlib import Path

from novaeval.utils.config import Config


def create_sample_config():
    """Create a sample configuration file."""

    config_data = {
        "dataset": {
            "type": "mmlu",
            "subset": "abstract_algebra",
            "num_samples": 20,
            "split": "test",
        },
        "models": [
            {
                "type": "openai",
                "model_name": "gpt-3.5-turbo",
                "temperature": 0.0,
                "max_tokens": 10,
            },
            {
                "type": "openai",
                "model_name": "gpt-4",
                "temperature": 0.0,
                "max_tokens": 10,
            },
        ],
        "scorers": [
            {"type": "accuracy", "extract_answer": True},
            {"type": "exact_match", "case_sensitive": False},
        ],
        "output": {
            "directory": "./results/config_example",
            "formats": ["json", "csv", "html"],
        },
        "evaluation": {"max_workers": 2, "batch_size": 1},
    }

    config = Config(config_data)
    config_path = Path("examples/sample_config.yaml")
    config_path.parent.mkdir(exist_ok=True)
    config.save(config_path, format="yaml")

    return config_path


def main():
    """Run a configuration-based evaluation example."""

    # Create sample configuration
    print("Creating sample configuration...")
    config_path = create_sample_config()
    print(f"Configuration saved to: {config_path}")

    # Load configuration
    print("Loading configuration...")
    config = Config.load(config_path)

    # Display configuration summary
    print("\nConfiguration Summary:")
    print("-" * 30)
    print(f"Dataset: {config.get('dataset.type')} ({config.get('dataset.subset')})")
    print(f"Models: {len(config.get('models', []))} models")
    print(f"Scorers: {len(config.get('scorers', []))} scorers")
    print(f"Samples: {config.get('dataset.num_samples')}")
    print(f"Output: {config.get('output.directory')}")

    # Note: The actual Evaluator.from_config() method would need to be implemented
    # to parse the configuration and create the appropriate objects
    print("\nNote: Configuration-based evaluation requires implementation")
    print("of the Evaluator.from_config() method and configuration parsers.")
    print("This is a placeholder example showing the intended usage.")

    # For now, we'll just validate the configuration structure
    required_keys = ["dataset", "models", "scorers"]
    for key in required_keys:
        if key not in config:
            print(f"Error: Missing required configuration key: {key}")
            return

    print("\n✓ Configuration is valid!")
    print(
        "Run 'novaeval run examples/sample_config.yaml' when implementation is complete."
    )


if __name__ == "__main__":
    main()
