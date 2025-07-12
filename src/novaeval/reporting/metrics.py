"""
Metrics calculation for NovaEval evaluations.

This module provides comprehensive metrics calculation including
performance, cost, and quality metrics aligned with Noveum.ai platform.
"""

import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class RequestMetrics:
    """Metrics for a single request."""

    latency: float  # Total latency in milliseconds
    ttfb: Optional[float] = None  # Time to first byte in milliseconds
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    tokens_total: Optional[int] = None
    cost: Optional[float] = None
    success: bool = True
    error_type: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None


@dataclass
class AggregatedMetrics:
    """Aggregated metrics for a set of requests."""

    # Required performance metrics
    avg_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    success_rate: float
    error_rate: float

    # Required counts
    total_requests: int
    successful_requests: int
    failed_requests: int

    # Optional performance metrics
    timeout_rate: float = 0.0
    avg_ttfb: Optional[float] = None
    p50_ttfb: Optional[float] = None
    p95_ttfb: Optional[float] = None
    p99_ttfb: Optional[float] = None

    # Optional token metrics
    total_tokens: int = 0
    avg_tokens_per_request: float = 0.0
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0

    # Optional cost metrics
    total_cost: float = 0.0
    avg_cost_per_request: float = 0.0
    cost_per_1k_tokens: float = 0.0

    # Optional distribution data
    latency_distribution: Optional[dict[str, int]] = None
    error_breakdown: Optional[dict[str, int]] = None


class MetricsCalculator:
    """
    Calculator for evaluation metrics and analytics.

    Provides comprehensive metrics calculation including performance,
    cost, and quality metrics.
    """

    def __init__(self):
        """Initialize metrics calculator."""
        self.request_metrics: list[RequestMetrics] = []

    def add_request_metric(self, metric: RequestMetrics) -> None:
        """Add a request metric to the collection."""
        self.request_metrics.append(metric)

    def add_request_data(
        self,
        latency: float,
        success: bool = True,
        ttfb: Optional[float] = None,
        tokens_input: Optional[int] = None,
        tokens_output: Optional[int] = None,
        cost: Optional[float] = None,
        error_type: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """
        Add request data as individual parameters.

        Args:
            latency: Request latency in milliseconds
            success: Whether request was successful
            ttfb: Time to first byte in milliseconds
            tokens_input: Number of input tokens
            tokens_output: Number of output tokens
            cost: Request cost in USD
            error_type: Type of error if failed
            provider: Model provider
            model: Model name
        """
        tokens_total = None
        if tokens_input is not None and tokens_output is not None:
            tokens_total = tokens_input + tokens_output

        metric = RequestMetrics(
            latency=latency,
            ttfb=ttfb,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            tokens_total=tokens_total,
            cost=cost,
            success=success,
            error_type=error_type,
            provider=provider,
            model=model,
        )
        self.add_request_metric(metric)

    def calculate_aggregated_metrics(
        self, filter_provider: Optional[str] = None, filter_model: Optional[str] = None
    ) -> AggregatedMetrics:
        """
        Calculate aggregated metrics for all requests.

        Args:
            filter_provider: Filter by provider
            filter_model: Filter by model

        Returns:
            Aggregated metrics
        """
        # Filter requests if needed
        filtered_metrics = self.request_metrics
        if filter_provider:
            filtered_metrics = [
                m for m in filtered_metrics if m.provider == filter_provider
            ]
        if filter_model:
            filtered_metrics = [m for m in filtered_metrics if m.model == filter_model]

        if not filtered_metrics:
            return self._empty_metrics()

        # Extract data for calculations
        latencies = [m.latency for m in filtered_metrics]
        ttfbs = [m.ttfb for m in filtered_metrics if m.ttfb is not None]
        successful = [m for m in filtered_metrics if m.success]
        failed = [m for m in filtered_metrics if not m.success]

        # Performance metrics
        avg_latency = statistics.mean(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)

        # TTFB metrics
        avg_ttfb = statistics.mean(ttfbs) if ttfbs else None
        p50_ttfb = np.percentile(ttfbs, 50) if ttfbs else None
        p95_ttfb = np.percentile(ttfbs, 95) if ttfbs else None
        p99_ttfb = np.percentile(ttfbs, 99) if ttfbs else None

        # Success metrics
        total_requests = len(filtered_metrics)
        successful_requests = len(successful)
        failed_requests = len(failed)
        success_rate = successful_requests / total_requests * 100
        error_rate = failed_requests / total_requests * 100

        # Token metrics
        token_metrics = self._calculate_token_metrics(filtered_metrics)

        # Cost metrics
        cost_metrics = self._calculate_cost_metrics(filtered_metrics)

        # Distribution data
        latency_distribution = self._calculate_latency_distribution(latencies)
        error_breakdown = self._calculate_error_breakdown(failed)

        return AggregatedMetrics(
            avg_latency=avg_latency,
            p50_latency=p50_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            avg_ttfb=avg_ttfb,
            p50_ttfb=p50_ttfb,
            p95_ttfb=p95_ttfb,
            p99_ttfb=p99_ttfb,
            success_rate=success_rate,
            error_rate=error_rate,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            latency_distribution=latency_distribution,
            error_breakdown=error_breakdown,
            **token_metrics,
            **cost_metrics,
        )

    def calculate_provider_comparison(self) -> dict[str, AggregatedMetrics]:
        """
        Calculate metrics grouped by provider.

        Returns:
            Dictionary mapping provider names to their metrics
        """
        providers = {m.provider for m in self.request_metrics if m.provider}
        return {
            provider: self.calculate_aggregated_metrics(filter_provider=provider)
            for provider in providers
        }

    def calculate_model_comparison(self) -> dict[str, AggregatedMetrics]:
        """
        Calculate metrics grouped by model.

        Returns:
            Dictionary mapping model names to their metrics
        """
        models = {m.model for m in self.request_metrics if m.model}
        return {
            model: self.calculate_aggregated_metrics(filter_model=model)
            for model in models
        }

    def _calculate_token_metrics(self, metrics: list[RequestMetrics]) -> dict[str, Any]:
        """Calculate token-related metrics."""
        total_tokens = 0
        input_tokens = []
        output_tokens = []

        for m in metrics:
            if m.tokens_total is not None:
                total_tokens += m.tokens_total
            if m.tokens_input is not None:
                input_tokens.append(m.tokens_input)
            if m.tokens_output is not None:
                output_tokens.append(m.tokens_output)

        return {
            "total_tokens": total_tokens,
            "avg_tokens_per_request": total_tokens / len(metrics) if metrics else 0,
            "avg_input_tokens": statistics.mean(input_tokens) if input_tokens else 0,
            "avg_output_tokens": statistics.mean(output_tokens) if output_tokens else 0,
        }

    def _calculate_cost_metrics(self, metrics: list[RequestMetrics]) -> dict[str, Any]:
        """Calculate cost-related metrics."""
        costs = [m.cost for m in metrics if m.cost is not None]
        total_cost = sum(costs) if costs else 0
        avg_cost_per_request = statistics.mean(costs) if costs else 0

        # Calculate cost per 1K tokens
        total_tokens = sum(
            m.tokens_total for m in metrics if m.tokens_total is not None
        )
        cost_per_1k_tokens = (
            (total_cost / total_tokens * 1000) if total_tokens > 0 else 0
        )

        return {
            "total_cost": total_cost,
            "avg_cost_per_request": avg_cost_per_request,
            "cost_per_1k_tokens": cost_per_1k_tokens,
        }

    def _calculate_latency_distribution(self, latencies: list[float]) -> dict[str, int]:
        """Calculate latency distribution buckets."""
        if not latencies:
            return {}

        # Define buckets in milliseconds
        buckets = [
            ("0-100ms", 0, 100),
            ("100-200ms", 100, 200),
            ("200-300ms", 200, 300),
            ("300-400ms", 300, 400),
            ("400-500ms", 400, 500),
            ("500-600ms", 500, 600),
            ("600-700ms", 600, 700),
            ("700-800ms", 700, 800),
            ("800-900ms", 800, 900),
            ("900ms+", 900, float("inf")),
        ]

        distribution = {}
        for label, min_val, max_val in buckets:
            count = sum(1 for lat in latencies if min_val <= lat < max_val)
            distribution[label] = count

        return distribution

    def _calculate_error_breakdown(
        self, failed_metrics: list[RequestMetrics]
    ) -> dict[str, int]:
        """Calculate error type breakdown."""
        error_counts = defaultdict(int)
        for metric in failed_metrics:
            error_type = metric.error_type or "unknown"
            error_counts[error_type] += 1
        return dict(error_counts)

    def _empty_metrics(self) -> AggregatedMetrics:
        """Return empty metrics for when no data is available."""
        return AggregatedMetrics(
            avg_latency=0.0,
            p50_latency=0.0,
            p95_latency=0.0,
            p99_latency=0.0,
            success_rate=0.0,
            error_rate=0.0,
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
        )

    def export_metrics_summary(self) -> dict[str, Any]:
        """
        Export comprehensive metrics summary.

        Returns:
            Dictionary containing all calculated metrics
        """
        overall_metrics = self.calculate_aggregated_metrics()
        provider_metrics = self.calculate_provider_comparison()
        model_metrics = self.calculate_model_comparison()

        return {
            "overall": overall_metrics.__dict__,
            "by_provider": {k: v.__dict__ for k, v in provider_metrics.items()},
            "by_model": {k: v.__dict__ for k, v in model_metrics.items()},
            "summary": {
                "total_requests": len(self.request_metrics),
                "unique_providers": len(
                    {m.provider for m in self.request_metrics if m.provider}
                ),
                "unique_models": len(
                    {m.model for m in self.request_metrics if m.model}
                ),
                "evaluation_duration": self._calculate_evaluation_duration(),
            },
        }

    def _calculate_evaluation_duration(self) -> Optional[float]:
        """Calculate total evaluation duration if timestamps are available."""
        # This would need timestamp data to be meaningful
        # For now, return None as placeholder
        return None

    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        self.request_metrics.clear()
