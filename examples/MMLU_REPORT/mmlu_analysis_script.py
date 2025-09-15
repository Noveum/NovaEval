#!/usr/bin/env python3
"""
Comprehensive MMLU Analysis Script
Analyzes OpenAI and GPT-OSS model performance across different thinking modes
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# Set style for plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class MMLUAnalyzer:
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.output_file = "insights.txt"
        self.plots_dir = Path("plots")
        self.plots_dir.mkdir(exist_ok=True)

        # Define the folders to analyze
        self.folders = [
            "multi_pattern_gpt_5_openai_run_20250810_145132",
            "multi_pattern_gpt_4o_mini_openai_run_20250810_091625",
            "multi_pattern_o3_concatenated_results",
            "multi_pattern_gpt_oss_cleaned_ollama_run_20250812_180842",
        ]

        # Initialize data storage
        self.all_data = {}
        self.insights = []

    def load_data(self):
        """Load all CSV files from the specified folders"""
        print("Loading data from CSV files...")

        # Files to ignore
        ignored_files = [
            "diff_results_summary.csv",
            "diff_results.csv",
            "regex_log.csv",
        ]

        for folder in self.folders:
            folder_path = self.base_path / folder
            if not folder_path.exists():
                print(f"Warning: Folder {folder} not found")
                continue

            csv_files = list(folder_path.glob("*.csv"))
            print(f"\nDiscovering CSV files in {folder}:")

            for csv_file in csv_files:
                if csv_file.name in ignored_files:
                    print(f"  Ignoring: {csv_file.name}")
                    continue

                print(f"  Loading: {csv_file.name}")
                try:
                    df = pd.read_csv(csv_file)
                    # Extract model name and thinking mode from filename
                    thinking_mode = csv_file.stem
                    df["thinking_mode"] = thinking_mode
                    df["source_folder"] = folder

                    # Ensure model column is populated for all rows
                    if "model" in df.columns:
                        # Get the first non-NaN model name
                        first_model = (
                            df["model"].dropna().iloc[0]
                            if not df["model"].dropna().empty
                            else None
                        )
                        if first_model:
                            # Log NaN count before fill
                            nan_count_before = df["model"].isna().sum()
                            print(f"  Before fill, NaN count: {nan_count_before}")

                            # Show the actual NaN rows to investigate
                            if nan_count_before > 0:
                                nan_rows = df[df["model"].isna()]
                                print("  NaN rows details:")
                                for idx, row in nan_rows.iterrows():
                                    print(
                                        f"    Row {idx}: sample_id='{row.get('sample_id', 'N/A')}', subset='{row.get('subset', 'N/A')}', score_accuracy={row.get('score_accuracy', 'N/A')}"
                                    )

                            print(
                                f"  Filling model column for {csv_file.name} with: {first_model}"
                            )
                            df["model"] = df["model"].fillna(first_model)
                            # Verify the fill worked
                            nan_count = df["model"].isna().sum()
                            print(f"  After fill, NaN count: {nan_count}")

                            # Show the same rows after filling to confirm the change
                            if nan_count_before > 0:
                                print("  After fill, same rows now contain:")
                                # Get the rows by their original indices from the updated dataframe
                                for idx in nan_rows.index:
                                    row = df.loc[idx]
                                    print(
                                        f"    Row {idx}: sample_id='{row.get('sample_id', 'N/A')}', subset='{row.get('subset', 'N/A')}', score_accuracy={row.get('score_accuracy', 'N/A')}, model='{row.get('model', 'N/A')}'"
                                    )
                        else:
                            print(f"No model found in {csv_file.name}")
                    else:
                        print(f"No model column in {csv_file.name}")

                    # Fix missing MEAN row data
                    df = self._fix_missing_mean_data(df)

                    # Store data with key
                    key = f"{folder}_{thinking_mode}"
                    self.all_data[key] = df
                    print(f"Loaded: {key} - {len(df)} rows")

                except Exception as e:
                    print(f"Error loading {csv_file}: {e}")

    def _fix_missing_mean_data(self, df):
        """Fix missing score_accuracy values in MEAN rows by calculating from individual data"""
        # Get MEAN rows
        mean_rows = df[df["sample_id"] == "MEAN"].copy()
        if mean_rows.empty:
            return df

        # Get individual question rows (non-MEAN)
        individual_rows = df[df["sample_id"] != "MEAN"].copy()
        if individual_rows.empty:
            return df

        # Create a copy to avoid modifying original
        df_fixed = df.copy()

        # Track data quality issues
        data_quality_issues = []

        for _, mean_row in mean_rows.iterrows():
            subset = mean_row["subset"]

            # Check if score_accuracy is missing for this MEAN row
            if pd.isna(mean_row["score_accuracy"]):
                # Calculate score from individual questions for this subject
                subject_data = individual_rows[individual_rows["subset"] == subset]
                if not subject_data.empty:
                    # Calculate accuracy from individual scores
                    valid_scores = subject_data["score_accuracy"].dropna()
                    if len(valid_scores) > 0:
                        calculated_accuracy = valid_scores.mean()

                        # Update the MEAN row in the fixed dataframe
                        mask = (df_fixed["sample_id"] == "MEAN") & (
                            df_fixed["subset"] == subset
                        )
                        df_fixed.loc[mask, "score_accuracy"] = calculated_accuracy

                        print(
                            f"  Fixed missing score_accuracy for {subset}: calculated {calculated_accuracy:.4f}"
                        )
                    else:
                        # No valid scores found for this subject
                        data_quality_issues.append(
                            f"{subset}: No valid individual question scores found"
                        )
                        print(
                            f"  Warning: No valid scores found for {subset} - subject will be excluded from analysis"
                        )

                        # Remove the problematic MEAN row
                        mask = (df_fixed["sample_id"] == "MEAN") & (
                            df_fixed["subset"] == subset
                        )
                        df_fixed = df_fixed[~mask]

                else:
                    data_quality_issues.append(
                        f"{subset}: No individual question data found"
                    )
                    print(
                        f"  Warning: No individual data found for {subset} - subject will be excluded from analysis"
                    )

                    # Remove the problematic MEAN row
                    mask = (df_fixed["sample_id"] == "MEAN") & (
                        df_fixed["subset"] == subset
                    )
                    df_fixed = df_fixed[~mask]

        # Print summary of data quality issues
        if data_quality_issues:
            print(
                f"  Data quality issues found: {len(data_quality_issues)} subjects excluded"
            )
            for issue in data_quality_issues:
                print(f"    - {issue}")

        return df_fixed

    def analyze_overall_model_performance(self):
        """Analyze overall model-wise performance"""
        print("\n=== Overall Model Performance Analysis ===")

        # Aggregate performance by model
        model_performance = {}

        for key, df in self.all_data.items():
            # Skip MEAN rows
            df_clean = df[df["sample_id"] != "MEAN"].copy()
            if df_clean.empty:
                continue

            # Extract model name - use the actual model name from CSV or fallback to key-based extraction
            if "model" in df_clean.columns and not df_clean["model"].isna().all():
                model_name = df_clean["model"].iloc[0]
            else:
                # Extract model name from the key as fallback
                if "ollama_run" in key:
                    model_name = "ollama_gpt-oss:20b"
                elif "gpt_5_openai" in key:
                    model_name = "openai_gpt-5"
                elif "gpt_4o_mini_openai" in key:
                    model_name = "openai_gpt-4o-mini"
                elif "o3_openai" in key:
                    model_name = "openai_o3"
                else:
                    model_name = key
            thinking_mode = df_clean["thinking_mode"].iloc[0]

            # Calculate metrics - only count rows with valid scores
            valid_scores = df_clean["score_accuracy"].dropna()
            total_questions = len(valid_scores)
            correct_answers = valid_scores.sum()
            accuracy = correct_answers / total_questions if total_questions > 0 else 0

            # For GPT-OSS models, calculate thinking token stats
            thinking_tokens_mean = 0
            thinking_tokens_std = 0
            if "thinking_tokens" in df_clean.columns:
                thinking_tokens_mean = df_clean["thinking_tokens"].mean()
                thinking_tokens_std = df_clean["thinking_tokens"].std()

            # For GPT-OSS models, calculate time stats
            cost_mean = 0
            cost_std = 0
            if "estimated_cost_usd" in df_clean.columns:
                cost_mean = df_clean["estimated_cost_usd"].mean()
                cost_std = df_clean["estimated_cost_usd"].std()

            model_performance[key] = {
                "model_name": model_name,
                "thinking_mode": thinking_mode,
                "total_questions": total_questions,
                "correct_answers": correct_answers,
                "accuracy": accuracy,
                "thinking_tokens_mean": thinking_tokens_mean,
                "thinking_tokens_std": thinking_tokens_std,
                "cost_mean": cost_mean,
                "cost_std": cost_std,
            }

        # Output results
        self.insights.append("=== OVERALL MODEL PERFORMANCE ===")
        self.insights.append(f"Total models analyzed: {len(model_performance)}")
        self.insights.append("")

        for key, perf in model_performance.items():
            self.insights.append(f"Model: {key}")
            self.insights.append(f"  Model Name: {perf['model_name']}")
            self.insights.append(f"  Thinking Mode: {perf['thinking_mode']}")
            self.insights.append(f"  Total Questions: {perf['total_questions']}")
            self.insights.append(f"  Correct Answers: {perf['correct_answers']}")
            self.insights.append(f"  Accuracy: {perf['accuracy']:.4f}")
            if perf["thinking_tokens_mean"] > 0:
                self.insights.append(
                    f"  Avg Thinking Tokens: {perf['thinking_tokens_mean']:.2f} ± {perf['thinking_tokens_std']:.2f}"
                )
            if perf["cost_mean"] > 0:
                self.insights.append(
                    f"  Avg Time: {perf['cost_mean']:.2f}s ± {perf['cost_std']:.2f}s"
                )
            self.insights.append("")

        return model_performance

    def analyze_subject_wise_performance(self):
        """Analyze subject-wise model performance"""
        print("\n=== Subject-wise Performance Analysis ===")

        subject_performance = {}

        for key, df in self.all_data.items():
            # Get MEAN rows for subject-wise analysis
            df_mean = df[df["sample_id"] == "MEAN"].copy()
            if df_mean.empty:
                continue

            # Get model name from the original data or use folder name as fallback
            if "model" in df_mean.columns and not df_mean["model"].isna().all():
                model_name = df_mean["model"].iloc[0]
                print(f"  Using model from CSV: {model_name}")
            else:
                # Extract model name from the key as fallback
                if "ollama_run" in key:
                    model_name = "ollama_gpt-oss:20b"
                elif "gpt_5_openai" in key:
                    model_name = "openai_gpt-5"
                elif "gpt_4o_mini_openai" in key:
                    model_name = "openai_gpt-4o-mini"
                elif "o3_openai" in key:
                    model_name = "openai_o3"
                else:
                    model_name = key
                print(f"  Using fallback model name: {model_name}")

            # Debug: check what's in the model column
            print(
                f"  Debug - key: {key}, model column values: {df_mean['model'].unique()}"
            )
            print(f"  Debug - model column NaN count: {df_mean['model'].isna().sum()}")
            thinking_mode = df_mean["thinking_mode"].iloc[0]

            for _, row in df_mean.iterrows():
                subset = row["subset"]
                accuracy = row["score_accuracy"]
                run_duration = row.get("run_duration_sec", 0)

                if subset not in subject_performance:
                    subject_performance[subset] = []

                subject_performance[subset].append(
                    {
                        "model_key": key,
                        "model_name": model_name,
                        "thinking_mode": thinking_mode,
                        "accuracy": accuracy,
                        "run_duration": run_duration,
                    }
                )
                print(
                    f"    Added to {subset}: {model_name} ({thinking_mode}): {accuracy}"
                )

        # Output results
        self.insights.append("=== SUBJECT-WISE PERFORMANCE ===")
        self.insights.append("")

        for subject, performances in subject_performance.items():
            self.insights.append(f"Subject: {subject}")
            self.insights.append(f"  Models tested: {len(performances)}")

            # Sort by accuracy
            sorted_perf = sorted(
                performances, key=lambda x: x["accuracy"], reverse=True
            )

            for i, perf in enumerate(sorted_perf):
                rank = i + 1
                print(
                    f"    Writing to insights: {rank}. {perf['model_name']} ({perf['thinking_mode']}): {perf['accuracy']:.4f}"
                )
                self.insights.append(
                    f"  {rank}. {perf['model_name']} ({perf['thinking_mode']}): {perf['accuracy']:.4f}"
                )
                if perf["run_duration"] > 0:
                    self.insights.append(f"      Duration: {perf['run_duration']:.2f}s")

            self.insights.append("")

        return subject_performance

    def analyze_thinking_mode_effects(self):
        """Analyze effects of thinking modes on performance and token usage"""
        print("\n=== Thinking Mode Effects Analysis ===")

        # Focus on GPT-OSS models with different thinking modes
        ollama_data = {k: v for k, v in self.all_data.items() if "ollama_run" in k}

        thinking_mode_analysis = {}

        for _key, df in ollama_data.items():
            thinking_mode = df["thinking_mode"].iloc[0]

            # Get individual question results (not MEAN rows)
            df_clean = df[df["sample_id"] != "MEAN"].copy()
            if df_clean.empty:
                continue

            # Calculate metrics
            accuracy = df_clean["score_accuracy"].mean()
            thinking_tokens_mean = df_clean["thinking_tokens"].mean()
            thinking_tokens_std = df_clean["thinking_tokens"].std()
            cost_mean = df_clean["estimated_cost_usd"].mean()

            thinking_mode_analysis[thinking_mode] = {
                "accuracy": accuracy,
                "thinking_tokens_mean": thinking_tokens_mean,
                "thinking_tokens_std": thinking_tokens_std,
                "cost_mean": cost_mean,
                "sample_size": len(df_clean),
            }

        # Output results
        self.insights.append("=== THINKING MODE EFFECTS ===")
        self.insights.append("")

        for mode, analysis in thinking_mode_analysis.items():
            self.insights.append(f"Thinking Mode: {mode}")
            self.insights.append(f"  Accuracy: {analysis['accuracy']:.4f}")
            self.insights.append(
                f"  Avg Thinking Tokens: {analysis['thinking_tokens_mean']:.2f} ± {analysis['thinking_tokens_std']:.2f}"
            )
            self.insights.append(f"  Avg Time: {analysis['cost_mean']:.2f}s")
            self.insights.append(f"  Sample Size: {analysis['sample_size']}")
            self.insights.append("")

        return thinking_mode_analysis

    def analyze_thinking_token_correlation(self):
        """Analyze correlation between thinking tokens and accuracy"""
        print("\n=== Thinking Token Correlation Analysis ===")

        # Focus on GPT-OSS models
        ollama_data = {k: v for k, v in self.all_data.items() if "ollama_run" in k}

        correlations = {}

        for key, df in ollama_data.items():
            # Get individual question results
            df_clean = df[df["sample_id"] != "MEAN"].copy()
            if df_clean.empty or "thinking_tokens" not in df_clean.columns:
                continue

            thinking_mode = df["thinking_mode"].iloc[0]

            # Calculate correlation
            tokens = df_clean["thinking_tokens"]
            accuracy = df_clean["score_accuracy"]

            if len(tokens) > 1 and tokens.std() > 0:
                correlation, p_value = stats.pearsonr(tokens, accuracy)
                spearman_corr, spearman_p = stats.spearmanr(tokens, accuracy)

                correlations[key] = {
                    "thinking_mode": thinking_mode,
                    "pearson_corr": correlation,
                    "pearson_p": p_value,
                    "spearman_corr": spearman_corr,
                    "spearman_p": spearman_p,
                    "sample_size": len(df_clean),
                }

        # Output results
        self.insights.append("=== THINKING TOKEN CORRELATION ANALYSIS ===")
        self.insights.append("")

        for key, corr in correlations.items():
            self.insights.append(f"Model: {key}")
            self.insights.append(f"  Thinking Mode: {corr['thinking_mode']}")
            self.insights.append(
                f"  Pearson Correlation: {corr['pearson_corr']:.4f} (p={corr['pearson_p']:.4f})"
            )
            self.insights.append(
                f"  Spearman Correlation: {corr['spearman_corr']:.4f} (p={corr['spearman_p']:.4f})"
            )
            self.insights.append(f"  Sample Size: {corr['sample_size']}")
            self.insights.append("")

        return correlations

    def analyze_subject_difficulty(self):
        """Analyze subject difficulty patterns across all models"""
        print("\n=== Subject Difficulty Analysis ===")

        subject_difficulty = {}

        for _key, df in self.all_data.items():
            # Get MEAN rows
            df_mean = df[df["sample_id"] == "MEAN"].copy()
            if df_mean.empty:
                continue

            for _, row in df_mean.iterrows():
                subset = row["subset"]
                accuracy = row["score_accuracy"]

                if subset not in subject_difficulty:
                    subject_difficulty[subset] = []

                subject_difficulty[subset].append(accuracy)

        # Calculate difficulty metrics
        difficulty_analysis = {}
        for subject, accuracies in subject_difficulty.items():
            if len(accuracies) > 1:
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                difficulty_analysis[subject] = {
                    "mean_accuracy": mean_acc,
                    "std_accuracy": std_acc,
                    "sample_size": len(accuracies),
                    "difficulty_score": 1 - mean_acc,  # Higher score = more difficult
                }

        # Sort by difficulty
        sorted_difficulty = sorted(
            difficulty_analysis.items(),
            key=lambda x: x[1]["difficulty_score"],
            reverse=True,
        )

        # Output results
        self.insights.append("=== SUBJECT DIFFICULTY ANALYSIS ===")
        self.insights.append("")

        for subject, analysis in sorted_difficulty:
            self.insights.append(f"Subject: {subject}")
            self.insights.append(
                f"  Difficulty Score: {analysis['difficulty_score']:.4f}"
            )
            self.insights.append(
                f"  Mean Accuracy: {analysis['mean_accuracy']:.4f} ± {analysis['std_accuracy']:.4f}"
            )
            self.insights.append(f"  Models Tested: {analysis['sample_size']}")
            self.insights.append("")

        return difficulty_analysis

    def detect_outliers(self, difficulty_analysis):
        """Detect outlier subjects using z-score"""
        print("\n=== Outlier Detection Analysis ===")

        if not difficulty_analysis:
            return {}

        # Calculate z-scores for mean accuracy
        accuracies = [
            analysis["mean_accuracy"] for analysis in difficulty_analysis.values()
        ]
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)

        outliers = {}
        for subject, analysis in difficulty_analysis.items():
            z_score = (
                (analysis["mean_accuracy"] - mean_acc) / std_acc if std_acc > 0 else 0
            )

            if abs(z_score) > 1.96:  # 95% confidence interval
                outliers[subject] = {
                    "z_score": z_score,
                    "mean_accuracy": analysis["mean_accuracy"],
                    "outlier_type": (
                        "high_performer" if z_score > 0 else "low_performer"
                    ),
                }

        # Output results
        self.insights.append("=== OUTLIER DETECTION (Z-score > 1.96) ===")
        self.insights.append("")

        if outliers:
            for subject, outlier_info in outliers.items():
                self.insights.append(f"Subject: {subject}")
                self.insights.append(f"  Z-score: {outlier_info['z_score']:.4f}")
                self.insights.append(
                    f"  Mean Accuracy: {outlier_info['mean_accuracy']:.4f}"
                )
                self.insights.append(f"  Outlier Type: {outlier_info['outlier_type']}")
                self.insights.append("")
        else:
            self.insights.append(
                "No outliers detected (all subjects within normal range)"
            )
            self.insights.append("")

        return outliers

    def analyze_model_consistency(self):
        """Analyze model consistency across subjects"""
        print("\n=== Model Consistency Analysis ===")

        model_consistency = {}

        for key, df in self.all_data.items():
            # Get MEAN rows
            df_mean = df[df["sample_id"] == "MEAN"].copy()
            if df_mean.empty:
                continue

            accuracies = df_mean["score_accuracy"].tolist()

            if len(accuracies) > 1:
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                cv = (
                    std_acc / mean_acc if mean_acc > 0 else float("inf")
                )  # Coefficient of variation

                model_consistency[key] = {
                    "mean_accuracy": mean_acc,
                    "std_accuracy": std_acc,
                    "cv": cv,
                    "min_accuracy": min(accuracies),
                    "max_accuracy": max(accuracies),
                    "range": max(accuracies) - min(accuracies),
                    "subjects_tested": len(accuracies),
                }

        # Sort by consistency (lower CV = more consistent)
        sorted_consistency = sorted(model_consistency.items(), key=lambda x: x[1]["cv"])

        # Output results
        self.insights.append("=== MODEL CONSISTENCY ANALYSIS ===")
        self.insights.append("(Lower CV = more consistent performance across subjects)")
        self.insights.append("")

        for key, consistency in sorted_consistency:
            self.insights.append(f"Model: {key}")
            self.insights.append(f"  Mean Accuracy: {consistency['mean_accuracy']:.4f}")
            self.insights.append(f"  Std Accuracy: {consistency['std_accuracy']:.4f}")
            self.insights.append(f"  Coefficient of Variation: {consistency['cv']:.4f}")
            self.insights.append(
                f"  Accuracy Range: {consistency['min_accuracy']:.4f} - {consistency['max_accuracy']:.4f}"
            )
            self.insights.append(f"  Subjects Tested: {consistency['subjects_tested']}")
            self.insights.append("")

        return model_consistency

    def analyze_efficiency_curves(self):
        """Analyze efficiency curves for thinking modes"""
        print("\n=== Efficiency Curve Analysis ===")

        # Focus on GPT-OSS models
        ollama_data = {k: v for k, v in self.all_data.items() if "ollama_run" in k}

        efficiency_data = {}

        for _key, df in ollama_data.items():
            thinking_mode = df["thinking_mode"].iloc[0]

            # Get individual question results
            df_clean = df[df["sample_id"] != "MEAN"].copy()
            if df_clean.empty or "thinking_tokens" not in df_clean.columns:
                continue

            # Group by thinking token ranges and calculate accuracy
            token_ranges = pd.cut(df_clean["thinking_tokens"], bins=10)
            range_accuracy = df_clean.groupby(token_ranges)["score_accuracy"].agg(
                ["mean", "count"]
            )

            efficiency_data[thinking_mode] = {
                "token_ranges": range_accuracy.index.tolist(),
                "accuracies": range_accuracy["mean"].tolist(),
                "counts": range_accuracy["count"].tolist(),
            }

        # Output results
        self.insights.append("=== EFFICIENCY CURVE ANALYSIS ===")
        self.insights.append("")

        for mode, data in efficiency_data.items():
            self.insights.append(f"Thinking Mode: {mode}")
            self.insights.append("  Token Range -> Accuracy (Sample Count):")

            for _i, (token_range, accuracy, count) in enumerate(
                zip(data["token_ranges"], data["accuracies"], data["counts"])
            ):
                self.insights.append(f"    {token_range}: {accuracy:.4f} (n={count})")
            self.insights.append("")

        return efficiency_data

    def create_plots(self):
        """Create all analysis plots"""
        print("\n=== Creating Analysis Plots ===")

        # 1. Overall Model Performance Comparison
        self.plot_model_performance()

        # 2. Subject-wise Performance Heatmap
        self.plot_subject_performance_heatmap()

        # 3. Thinking Mode Effects
        self.plot_thinking_mode_effects()

        # 4. Thinking Token Distribution
        self.plot_thinking_token_distribution()

        # 5. Efficiency Curves
        self.plot_efficiency_curves()

        # 6. Subject Difficulty Ranking
        self.plot_subject_difficulty()

        # 7. Model Consistency Analysis
        self.plot_model_consistency()

        # 8. Time vs Performance Analysis (GPT-OSS only)
        self.plot_time_vs_performance()

    def plot_model_performance(self):
        """Plot overall model performance comparison"""
        plt.figure(figsize=(14, 8))

        # Prepare data
        models = []
        accuracies = []
        thinking_modes = []

        for _key, df in self.all_data.items():
            df_clean = df[df["sample_id"] != "MEAN"].copy()
            if df_clean.empty:
                continue

            accuracy = df_clean["score_accuracy"].mean()
            thinking_mode = df["thinking_mode"].iloc[0]
            model_name = (
                df_clean["model"].iloc[0] if "model" in df_clean.columns else "unknown"
            )

            models.append(f"{model_name}\n({thinking_mode})")
            accuracies.append(accuracy)
            thinking_modes.append(thinking_mode)

        # Create color map based on thinking modes
        colors = [
            "skyblue" if "gpt-oss" in mode else "lightcoral" for mode in thinking_modes
        ]

        bars = plt.bar(range(len(models)), accuracies, color=colors, alpha=0.7)
        plt.xlabel("Model and Thinking Mode")
        plt.ylabel("Accuracy")
        plt.title("Overall Model Performance Comparison")
        plt.xticks(range(len(models)), models, rotation=45, ha="right")
        plt.ylim(0, 1)

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{acc:.3f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "overall_model_performance_comparison_bar_chart.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def plot_subject_performance_heatmap(self):
        """Plot subject-wise performance heatmap"""
        # Prepare data matrix
        subjects = set()
        models = set()

        for key, df in self.all_data.items():
            df_mean = df[df["sample_id"] == "MEAN"].copy()
            if df_mean.empty:
                continue

            for _, row in df_mean.iterrows():
                subjects.add(row["subset"])
                models.add(key)

        subjects = sorted(subjects)
        models = sorted(models)

        # Create performance matrix
        performance_matrix = np.full((len(subjects), len(models)), np.nan)

        for i, subject in enumerate(subjects):
            for j, model in enumerate(models):
                if model in self.all_data:
                    df_mean = self.all_data[model][
                        self.all_data[model]["sample_id"] == "MEAN"
                    ]
                    subject_data = df_mean[df_mean["subset"] == subject]
                    if not subject_data.empty:
                        performance_matrix[i, j] = subject_data["score_accuracy"].iloc[
                            0
                        ]

        # Check if we have any data
        if np.isnan(performance_matrix).all():
            print("Warning: No performance data found for heatmap")
            return

        # Create heatmap
        plt.figure(figsize=(16, 10))
        sns.heatmap(
            performance_matrix,
            xticklabels=[m[:30] + "..." if len(m) > 30 else m for m in models],
            yticklabels=subjects,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            cbar_kws={"label": "Accuracy"},
        )
        plt.title("Subject-wise Performance Heatmap")
        plt.xlabel("Model and Thinking Mode")
        plt.ylabel("Subject")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "subject_wise_performance_heatmap.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def plot_thinking_mode_effects(self):
        """Plot thinking mode effects on performance and tokens"""
        # Focus on GPT-OSS models
        ollama_data = {k: v for k, v in self.all_data.items() if "ollama_run" in k}

        if not ollama_data:
            return

        _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Performance comparison
        modes = []
        accuracies = []
        token_counts = []

        for _key, df in ollama_data.items():
            df_clean = df[df["sample_id"] != "MEAN"].copy()
            if df_clean.empty:
                continue

            mode = df["thinking_mode"].iloc[0]
            accuracy = df_clean["score_accuracy"].mean()
            tokens = df_clean["thinking_tokens"].mean()

            modes.append(mode)
            accuracies.append(accuracy)
            token_counts.append(tokens)

        # Plot 1: Accuracy by thinking mode
        bars1 = ax1.bar(modes, accuracies, color="skyblue", alpha=0.7)
        ax1.set_xlabel("Thinking Mode")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Accuracy by Thinking Mode")
        ax1.set_ylim(0, 1)

        for bar, acc in zip(bars1, accuracies):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{acc:.3f}",
                ha="center",
                va="bottom",
            )

        # Plot 2: Thinking tokens by thinking mode
        bars2 = ax2.bar(modes, token_counts, color="lightgreen", alpha=0.7)
        ax2.set_xlabel("Thinking Mode")
        ax2.set_ylabel("Average Thinking Tokens")
        ax2.set_title("Thinking Token Usage by Mode")

        for bar, tokens in zip(bars2, token_counts):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 10,
                f"{tokens:.0f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "thinking_mode_effects_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def plot_thinking_token_distribution(self):
        """Plot thinking token distribution for GPT-OSS models"""
        # Focus on GPT-OSS models
        ollama_data = {k: v for k, v in self.all_data.items() if "ollama_run" in k}

        if not ollama_data:
            return

        plt.figure(figsize=(14, 8))

        for _key, df in ollama_data.items():
            df_clean = df[df["sample_id"] != "MEAN"].copy()
            if df_clean.empty or "thinking_tokens" not in df_clean.columns:
                continue

            thinking_mode = df["thinking_mode"].iloc[0]
            tokens = df_clean["thinking_tokens"]

            plt.hist(tokens, bins=30, alpha=0.6, label=thinking_mode, density=True)

        plt.xlabel("Thinking Tokens")
        plt.ylabel("Density")
        plt.title("Thinking Token Distribution by Mode")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "thinking_token_distribution_by_mode.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def plot_efficiency_curves(self):
        """Plot efficiency curves for thinking modes"""
        # Focus on GPT-OSS models
        ollama_data = {k: v for k, v in self.all_data.items() if "ollama_run" in k}

        if not ollama_data:
            return

        plt.figure(figsize=(12, 8))

        for _key, df in ollama_data.items():
            df_clean = df[df["sample_id"] != "MEAN"].copy()
            if df_clean.empty or "thinking_tokens" not in df_clean.columns:
                continue

            thinking_mode = df["thinking_mode"].iloc[0]

            # Create efficiency curve
            tokens = df_clean["thinking_tokens"]
            accuracy = df_clean["score_accuracy"]

            # Sort by tokens and calculate cumulative accuracy
            sorted_data = sorted(zip(tokens, accuracy))
            tokens_sorted, accuracy_sorted = zip(*sorted_data)

            # Calculate cumulative accuracy
            cumulative_acc = np.cumsum(accuracy_sorted) / np.arange(
                1, len(accuracy_sorted) + 1
            )

            plt.plot(
                tokens_sorted,
                cumulative_acc,
                marker="o",
                label=thinking_mode,
                alpha=0.7,
            )

        plt.xlabel("Thinking Tokens")
        plt.ylabel("Cumulative Accuracy")
        plt.title("Efficiency Curves: Accuracy vs Thinking Tokens")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "efficiency_curves_accuracy_vs_tokens.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def plot_subject_difficulty(self):
        """Plot subject difficulty ranking"""
        # Calculate subject difficulty
        subject_difficulty = {}

        for _key, df in self.all_data.items():
            df_mean = df[df["sample_id"] == "MEAN"].copy()
            if df_mean.empty:
                continue

            for _, row in df_mean.iterrows():
                subset = row["subset"]
                accuracy = row["score_accuracy"]

                if subset not in subject_difficulty:
                    subject_difficulty[subset] = []

                subject_difficulty[subset].append(accuracy)

        # Calculate mean difficulty for each subject
        difficulty_scores = {}
        for subject, accuracies in subject_difficulty.items():
            if len(accuracies) > 1:
                mean_acc = np.mean(accuracies)
                difficulty_scores[subject] = 1 - mean_acc  # Higher = more difficult

        # Sort by difficulty
        sorted_difficulty = sorted(
            difficulty_scores.items(), key=lambda x: x[1], reverse=True
        )

        subjects, difficulties = zip(*sorted_difficulty)

        plt.figure(figsize=(12, 8))
        bars = plt.barh(subjects, difficulties, color="lightcoral", alpha=0.7)
        plt.xlabel("Difficulty Score (1 - Mean Accuracy)")
        plt.ylabel("Subject")
        plt.title("Subject Difficulty Ranking")
        plt.xlim(0, max(difficulties) * 1.1)

        # Add value labels
        for bar, diff in zip(bars, difficulties):
            plt.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{diff:.3f}",
                ha="left",
                va="center",
            )

        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "subject_difficulty_ranking_horizontal_bar.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def plot_model_consistency(self):
        """Plot model consistency analysis"""
        # Calculate consistency metrics
        model_consistency = {}

        for key, df in self.all_data.items():
            df_mean = df[df["sample_id"] == "MEAN"].copy()
            if df_mean.empty:
                continue

            accuracies = df_mean["score_accuracy"].tolist()

            if len(accuracies) > 1:
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                cv = std_acc / mean_acc if mean_acc > 0 else float("inf")

                model_consistency[key] = {"mean_accuracy": mean_acc, "cv": cv}

        if not model_consistency:
            return

        # Create scatter plot
        models = list(model_consistency.keys())
        mean_accs = [model_consistency[m]["mean_accuracy"] for m in models]
        cvs = [model_consistency[m]["cv"] for m in models]

        plt.figure(figsize=(12, 8))

        # Color code by model type
        colors = ["skyblue" if "gpt-oss" in m else "lightcoral" for m in models]

        plt.scatter(mean_accs, cvs, c=colors, s=100, alpha=0.7)

        # Add labels
        for i, model in enumerate(models):
            plt.annotate(
                model[:30] + "..." if len(model) > 30 else model,
                (mean_accs[i], cvs[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        plt.xlabel("Mean Accuracy")
        plt.ylabel("Coefficient of Variation (Lower = More Consistent)")
        plt.title("Model Consistency: Accuracy vs Variability")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "model_consistency_scatter_plot.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def plot_time_vs_performance(self):
        """Plot time vs performance analysis for GPT-OSS models"""
        # Focus on GPT-OSS models
        ollama_data = {k: v for k, v in self.all_data.items() if "ollama_run" in k}

        if not ollama_data:
            return

        plt.figure(figsize=(12, 8))

        models = []
        accuracies = []
        times = []
        thinking_modes = []

        for key, df in ollama_data.items():
            df_clean = df[df["sample_id"] != "MEAN"].copy()
            if df_clean.empty or "estimated_cost_usd" not in df_clean.columns:
                continue

            accuracy = df_clean["score_accuracy"].mean()
            time_taken = df_clean["estimated_cost_usd"].mean()
            thinking_mode = df["thinking_mode"].iloc[0]

            models.append(key)
            accuracies.append(accuracy)
            times.append(time_taken)
            thinking_modes.append(thinking_mode)

        if not models:
            return

        # Create scatter plot
        colors = ["skyblue", "lightgreen", "lightcoral", "gold"]
        color_map = {
            mode: colors[i % len(colors)] for i, mode in enumerate(set(thinking_modes))
        }

        # Create a list to track which modes we've already labeled
        labeled_modes = set()

        for _i, (model, acc, time_taken, mode) in enumerate(
            zip(models, accuracies, times, thinking_modes)
        ):
            # Only label if we haven't seen this mode before
            should_label = mode not in labeled_modes
            if should_label:
                labeled_modes.add(mode)

            plt.scatter(
                time_taken,
                acc,
                c=color_map[mode],
                s=100,
                alpha=0.7,
                label=mode if should_label else "",
            )
            plt.annotate(
                model[:20] + "..." if len(model) > 20 else model,
                (time_taken, acc),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        plt.xlabel("Average Time per Question (seconds)")
        plt.ylabel("Accuracy")
        plt.title("Time vs Performance Analysis (GPT-OSS Models)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "time_vs_performance_gpt_oss_scatter.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def save_insights(self):
        """Save all insights to text file"""
        print(f"\n=== Saving Insights to {self.output_file} ===")

        with open(self.output_file, "w") as f:
            f.write("MMLU EVALUATION ANALYSIS INSIGHTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(
                "Generated on: "
                + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                + "\n\n"
            )

            for insight in self.insights:
                f.write(insight + "\n")

            # Add plot descriptions
            f.write("\n" + "=" * 50 + "\n")
            f.write("PLOT DESCRIPTIONS\n")
            f.write("=" * 50 + "\n\n")

            plot_descriptions = [
                "overall_model_performance_comparison_bar_chart.png - Bar chart comparing overall accuracy across all models and thinking modes, with GPT-OSS models in blue and OpenAI models in red",
                "subject_wise_performance_heatmap.png - Heatmap showing performance matrix with subjects on Y-axis and models on X-axis, color-coded by accuracy (green=high, red=low)",
                "thinking_mode_effects_comparison.png - Side-by-side bar charts showing accuracy and thinking token usage for different GPT-OSS thinking modes",
                "thinking_token_distribution_by_mode.png - Histogram showing distribution of thinking tokens across different thinking modes for GPT-OSS models",
                "efficiency_curves_accuracy_vs_tokens.png - Line plots showing cumulative accuracy vs thinking tokens for different thinking modes, demonstrating efficiency curves",
                "subject_difficulty_ranking_horizontal_bar.png - Horizontal bar chart ranking subjects by difficulty score (1 - mean accuracy), higher bars indicate more difficult subjects",
                "model_consistency_scatter_plot.png - Scatter plot showing mean accuracy vs coefficient of variation for each model, with GPT-OSS models in blue and OpenAI models in red",
                "time_vs_performance_gpt_oss_scatter.png - Scatter plot showing time vs performance relationship for GPT-OSS models only, with different colors for thinking modes",
            ]

            for desc in plot_descriptions:
                f.write(desc + "\n")

        print(f"Insights saved to {self.output_file}")

    def run_analysis(self):
        """Run complete analysis"""
        print("Starting MMLU Analysis...")

        # Load data
        self.load_data()

        # Run analyses
        self.analyze_overall_model_performance()
        self.analyze_subject_wise_performance()
        self.analyze_thinking_mode_effects()
        self.analyze_thinking_token_correlation()
        subject_difficulty = self.analyze_subject_difficulty()
        self.detect_outliers(subject_difficulty)
        self.analyze_model_consistency()
        self.analyze_efficiency_curves()

        # Create plots
        self.create_plots()

        # Save insights
        self.save_insights()

        print(
            f"\nAnalysis complete! Check {self.output_file} for insights and {self.plots_dir}/ for plots."
        )


if __name__ == "__main__":
    analyzer = MMLUAnalyzer()
    analyzer.run_analysis()
