from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from novaeval import Evaluator
from novaeval.datasets import MMLUDataset
from novaeval.models import OpenAIModel
from novaeval.scorers import AccuracyScorer

"""
MMLU evaluation example for NovaEval using OpenAI models.

Environment variables:
- NOVEUM_OPENAI_API_KEY: OpenAI API key to authenticate requests
- MAX_TOKENS:            Optional override for max tokens (default: 5000)
- NUM_SAMPLES:           Optional override for number of samples per subset (default: 50)

Notes:
- This mirrors the functionality and CSV outputs of `examples/ollama_mmlu_evaluation.py`,
  but without think modes. It enforces a global max_tokens override to match the
  Ollama example's behavior.
"""

# Configuration defaults and subsets (mirrors ollama_mmlu_evaluation.py)
SUBSETS = [
    # "abstract_algebra",
    # "college_chemistry",
    # "college_mathematics",
    # "college_physics",
    # "conceptual_physics",
    # "elementary_mathematics",
    # "high_school_chemistry",
    # "high_school_mathematics",
    # "high_school_physics",
    "high_school_statistics",
]

NUM_SAMPLES = 50
DEFAULT_MAX_TOKENS = 5000

# Handle a list of model names; hardcoded length 1 for now
MODEL_NAMES: list[str] = [
    "o3",
]


class ConfiguredOpenAIModel(OpenAIModel):
    def __init__(
        self,
        *args,
        default_max_tokens: int | None = None,
        timeout: float = 600.0,  # Increased timeout to 600 seconds (10 minutes)
        **kwargs: Any,
    ) -> None:
        # Inject API key from env var NOVEUM_OPENAI_API_KEY if not provided
        api_key = kwargs.get("api_key") or os.getenv("NOVEUM_OPENAI_API_KEY")
        kwargs["api_key"] = api_key
        # Set the timeout parameter
        kwargs["timeout"] = timeout
        super().__init__(*args, **kwargs)
        self.default_max_tokens = default_max_tokens

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: str | list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        # Enforce default max tokens regardless of dataset-provided kwargs
        if self.default_max_tokens is not None:
            max_tokens = self.default_max_tokens
        return super().generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            **kwargs,
        )


class OpenAIMMLUEvaluator(Evaluator):
    def evaluate_sample(self, sample: dict, model: OpenAIModel, scorers: list) -> dict:
        sample_result = {
            "sample_id": sample.get("id", "unknown"),
            "input": sample.get("input", ""),
            "expected": sample.get("expected", ""),
            "prediction": None,
            "scores": {},
            "metadata": {},
            "error": None,
        }

        try:
            before_cost = float(getattr(model, "total_cost", 0.0) or 0.0)

            prediction = model.generate(
                sample["input"], **sample.get("generation_kwargs", {})
            )

            after_cost = float(getattr(model, "total_cost", before_cost) or before_cost)
            estimated_cost = after_cost - before_cost

            sample_result["prediction"] = prediction

            # Apply scorers
            for scorer in scorers:
                try:
                    score = scorer.score(
                        prediction=prediction,
                        ground_truth=sample.get("expected", ""),
                        context=sample,
                    )
                    sample_result["scores"][scorer.name] = score
                except Exception as e:  # keep going even if a scorer fails

                    sample_result["error"] = str(e)
                    raise e

            # Metadata (mirror ollama fields; set thinking_tokens=0)
            sample_result["metadata"] = {
                "model_name": model.name,
                "timestamp": time.time(),
                "estimated_cost_usd": float(estimated_cost),
                "thinking_tokens": 0,
            }
        except Exception as e:
            sample_result["error"] = str(e)
            raise e
        return sample_result


def main() -> None:
    # Configurable generation counts/budget (mirrors ollama example)
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))
    NUM_SAMPLES_LOCAL = int(os.getenv("NUM_SAMPLES", str(NUM_SAMPLES)))

    # Prepare timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_run_dir = Path("examples") / f"openai_run_{timestamp}"
    base_run_dir.mkdir(parents=True, exist_ok=True)

    # Scorer
    scorer = AccuracyScorer(extract_answer=True)

    # Iterate over model names (list length 1 for now)
    for model_name in MODEL_NAMES:
        print("\n" + "=" * 50)
        print(f"Running evaluation (model: {model_name})")
        print("=" * 50)

        # Initialize model with enforced max_tokens
        print("Initializing OpenAI model...")
        print(f"Using model '{model_name}'")
        print(f"⚠️  Using max_tokens={MAX_TOKENS}")
        print("⚠️  Using timeout=600 seconds (10 minutes) for o3 model")
        model = ConfiguredOpenAIModel(
            model_name=model_name,
            temperature=0.0,
            default_max_tokens=MAX_TOKENS,
        )

        # Optional connectivity check
        ok = model.validate_connection()
        print(f"Connection valid: {ok}")

        # Create evaluator in a mode-specific subdirectory (mirror ollama)
        mode_dir = base_run_dir / "mode_unspecified"
        mode_dir.mkdir(parents=True, exist_ok=True)

        # Aggregate across all subsets for this model
        all_rows: list[dict[str, Any]] = []
        subset_durations: dict[str, float] = {}

        for subset in SUBSETS:
            # Fresh dataset per subset
            dataset = MMLUDataset(
                subset=subset,
                num_samples=NUM_SAMPLES_LOCAL,
                split="test",
            )

            evaluator = OpenAIMMLUEvaluator(
                dataset=dataset,
                models=[model],
                scorers=[scorer],
                output_dir=mode_dir,
            )

            # Run evaluation
            results = evaluator.run()
            subset_durations[subset] = float(
                results.get("metadata", {}).get("duration", 0.0)
            )

            # Collect per-sample rows with subset column
            for model_name_key, model_results in results.get(
                "model_results", {}
            ).items():
                for sample in model_results.get("samples", []):
                    row: dict[str, Any] = {
                        "subset": subset,
                        "model": model_name_key,
                        "sample_id": sample.get("sample_id", "unknown"),
                    }
                    meta = sample.get("metadata", {}) or {}
                    if isinstance(meta, dict):
                        row["estimated_cost_usd"] = meta.get("estimated_cost_usd")
                        row["thinking_tokens"] = meta.get("thinking_tokens")
                    scores = sample.get("scores", {}) or {}
                    for scorer_name, score_val in scores.items():
                        if isinstance(score_val, dict):
                            # Flatten dict scores
                            for k, v in score_val.items():
                                if isinstance(v, (int, float)):
                                    row[f"score_{scorer_name}_{k}"] = v
                        else:
                            row[f"score_{scorer_name}"] = score_val
                    all_rows.append(row)

        # Build and write CSV for this model
        if all_rows:
            df = pd.DataFrame(all_rows)
            score_cols = [c for c in df.columns if c.startswith("score_")]

            # Compute per-subset means and append a MEAN row per subset
            mean_rows = []
            for subset in SUBSETS:
                df_subset = df[df["subset"] == subset]
                if df_subset.empty:
                    continue
                mean_series = (
                    df_subset[score_cols].mean(numeric_only=True)
                    if score_cols
                    else pd.Series()
                )
                mean_row = dict.fromkeys(df.columns, "")
                mean_row["subset"] = subset
                mean_row["sample_id"] = "MEAN"
                mean_row["run_duration_sec"] = subset_durations.get(subset, 0.0)
                for c in score_cols:
                    mean_row[c] = float(mean_series.get(c, 0.0))
                mean_rows.append(mean_row)

            if mean_rows:
                df = pd.concat([df, pd.DataFrame(mean_rows)], ignore_index=True)

            csv_path = base_run_dir / "unspecified.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved CSV for model '{model_name}' to: {csv_path}")

    print(f"\nAll outputs saved under: {base_run_dir}")


if __name__ == "__main__":
    main()
