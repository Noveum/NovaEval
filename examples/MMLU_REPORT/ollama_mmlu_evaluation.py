"""
MMLU evaluation example for NovaEval using an Ollama endpoint.

Environment variables (optional):
- OLLAMA_BASE_URL: The Ollama endpoint URL, e.g. https://api.your-ollama-host.com
- OLLAMA_HOST:      Alternative env var for endpoint URL
- OLLAMA_API_KEY:   Optional API key; sent as Authorization: Bearer <key>
- OLLAMA_MODEL:     Model name to use (default: gpt-oss:20b)
- OLLAMA_GPU_COST_PER_SEC: Optional cost per GPU-second for cost estimation
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from novaeval import Evaluator
from novaeval.datasets import MMLUDataset
from novaeval.models import OllamaModel
from novaeval.scorers import AccuracyScorer, MultiPatternAccuracyScorer

# Configuration defaults and subsets
SUBSETS = [
    "abstract_algebra",
    "college_chemistry",
    "college_mathematics",
    "college_physics",
    "conceptual_physics",
    "elementary_mathematics",
    "high_school_chemistry",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics",
]

NUM_SAMPLES = 50
DEFAULT_MAX_TOKENS = 5000
DEFAULT_OLLAMA_HOST = "http://34.121.64.12:8001"
DEFAULT_OLLAMA_MODEL = "gpt-oss:20b"
DEFAULT_GPU_COST_PER_SEC = 1

THINK_MODES = [
    (None, "unspecified"),
    ("low", "low"),
    ("medium", "medium"),
    ("high", "high"),
]


class ThinkOllamaModel(OllamaModel):
    def __init__(
        self,
        *args,
        default_think: str | None = None,
        default_max_tokens: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.default_think = default_think
        self.default_max_tokens = default_max_tokens

    def generate(
        self,
        prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: str | list[str] | None = None,
        messages: list[dict] | None = None,
        **kwargs,
    ) -> str:
        # Enforce default max tokens regardless of dataset-provided kwargs
        if self.default_max_tokens is not None:
            max_tokens = self.default_max_tokens
        # Inject default think if not explicitly provided
        if self.default_think is not None and "think" not in kwargs:
            kwargs["think"] = self.default_think
        return super().generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            messages=messages,
            **kwargs,
        )

    def thinking_generate(
        self,
        prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: str | list[str] | None = None,
        messages: list[dict] | None = None,
        **kwargs,
    ) -> tuple[str, str]:
        if self.default_max_tokens is not None:
            max_tokens = self.default_max_tokens
        if self.default_think is not None and "think" not in kwargs:
            kwargs["think"] = self.default_think
        return super().generate_with_thought(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            messages=messages,
            **kwargs,
        )


class OllamaEvaluator(Evaluator):
    def evaluate_sample(self, sample: dict, model: OllamaModel, scorers: list) -> dict:
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

            if hasattr(model, "thinking_generate"):
                prediction, thinking_text = model.thinking_generate(
                    sample["input"], **sample.get("generation_kwargs", {})
                )
            else:
                prediction = model.generate(
                    sample["input"], **sample.get("generation_kwargs", {})
                )
                thinking_text = ""

            after_cost = float(getattr(model, "total_cost", before_cost) or before_cost)
            estimated_cost = after_cost - before_cost
            thinking_tokens = model.count_tokens(thinking_text) if thinking_text else 0

            sample_result["prediction"] = prediction

            for scorer in scorers:
                try:
                    score = scorer.score(
                        prediction=prediction,
                        ground_truth=sample.get("expected", ""),
                        context=sample,
                    )
                    sample_result["scores"][scorer.name] = score
                except Exception as e:
                    sample_result["error"] = str(e)

            sample_result["metadata"] = {
                "model_name": model.name,
                "timestamp": time.time(),
                "estimated_cost_usd": float(estimated_cost),
                "thinking_tokens": int(thinking_tokens),
            }
        except Exception as e:
            sample_result["error"] = str(e)

        return sample_result


def main() -> None:
    """
    Run a basic MMLU evaluation against an Ollama endpoint.

    Note: This example uses max_tokens=5000 for potential reasoning, but the
    MMLU dataset internally configures per-sample generation to return just
    the answer choice efficiently. Adjust MAX_TOKENS based on your needs.
    """

    # Configurable generation counts/budget
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))
    NUM_SAMPLES_LOCAL = int(os.getenv("NUM_SAMPLES", str(NUM_SAMPLES)))

    # Prepare timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_run_dir = Path("examples") / f"ollama_run_{timestamp}"
    base_run_dir.mkdir(parents=True, exist_ok=True)

    # Resolve Ollama connection details from environment
    base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv(
        "OLLAMA_HOST", DEFAULT_OLLAMA_HOST
    )
    api_key = os.getenv("OLLAMA_API_KEY")
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    model_name = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)

    # Run evaluations across think modes and save per-mode CSVs
    think_modes = THINK_MODES

    for think_value, label in think_modes:
        print("\n" + "=" * 50)
        print(f"Running evaluation (think mode: {label})")
        print("=" * 50)

        # Initialize Ollama model for this think mode
        print(f"Initializing Ollama model (think: {label})...")
        print(f"Using model '{model_name}' at {base_url}")
        print(f"⚠️  Using max_tokens={MAX_TOKENS}")
        model = ThinkOllamaModel(
            model_name=model_name,
            base_url=base_url,
            headers=headers,
            gpu_cost_per_sec=float(
                os.getenv("OLLAMA_GPU_COST_PER_SEC", str(DEFAULT_GPU_COST_PER_SEC))
            ),
            temperature=0.0,
            default_think=think_value,
            default_max_tokens=MAX_TOKENS,
        )
        # Make name reflect the think mode
        model.name = f"{model.name}_think_{label}"

        # Optional connectivity check
        ok = model.validate_connection()
        print(f"Connection valid: {ok}")

        # Create evaluator in a mode-specific subdirectory
        mode_dir = base_run_dir / f"mode_{label}"
        mode_dir.mkdir(parents=True, exist_ok=True)
        scorer = AccuracyScorer(extract_answer=True)
        multi_pattern_scorer = MultiPatternAccuracyScorer()

        # Aggregate across all subsets for this think mode
        all_rows = []
        subset_durations: dict[str, float] = {}

        for subset in SUBSETS:
            # Fresh dataset per subset
            dataset = MMLUDataset(
                subset=subset,
                num_samples=NUM_SAMPLES_LOCAL,
                split="test",
            )

            evaluator = OllamaEvaluator(
                dataset=dataset,
                models=[model],
                scorers=[scorer, multi_pattern_scorer],
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
                    row = {
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

            csv_path = base_run_dir / f"{label}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved CSV for think mode '{label}' to: {csv_path}")

    print(f"\nAll outputs saved under: {base_run_dir}")


if __name__ == "__main__":
    main()
