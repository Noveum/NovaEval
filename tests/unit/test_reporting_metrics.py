"""
Unit tests for metrics calculation functionality.
"""

import pytest

from novaeval.reporting.metrics import (
    AggregatedMetrics,
    MetricsCalculator,
    RequestMetrics,
)


class TestRequestMetrics:
    """Test cases for RequestMetrics dataclass."""

    def test_init_minimal(self):
        """Test initialization with minimal parameters."""
        metrics = RequestMetrics(latency=100.0)

        assert metrics.latency == 100.0
        assert metrics.success is True
        assert metrics.ttfb is None
        assert metrics.tokens_input is None
        assert metrics.tokens_output is None
        assert metrics.tokens_total is None
        assert metrics.cost is None
        assert metrics.error_type is None
        assert metrics.provider is None
        assert metrics.model is None

    def test_init_complete(self):
        """Test initialization with all parameters."""
        metrics = RequestMetrics(
            latency=150.0,
            ttfb=50.0,
            tokens_input=100,
            tokens_output=50,
            tokens_total=150,
            cost=0.05,
            success=True,
            error_type=None,
            provider="openai",
            model="gpt-4",
        )

        assert metrics.latency == 150.0
        assert metrics.ttfb == 50.0
        assert metrics.tokens_input == 100
        assert metrics.tokens_output == 50
        assert metrics.tokens_total == 150
        assert metrics.cost == 0.05
        assert metrics.success is True
        assert metrics.error_type is None
        assert metrics.provider == "openai"
        assert metrics.model == "gpt-4"

    def test_init_failed_request(self):
        """Test initialization for failed request."""
        metrics = RequestMetrics(
            latency=5000.0,
            success=False,
            error_type="timeout",
            provider="openai",
            model="gpt-4",
        )

        assert metrics.latency == 5000.0
        assert metrics.success is False
        assert metrics.error_type == "timeout"
        assert metrics.provider == "openai"
        assert metrics.model == "gpt-4"


class TestAggregatedMetrics:
    """Test cases for AggregatedMetrics dataclass."""

    def test_init_required_fields(self):
        """Test initialization with required fields."""
        metrics = AggregatedMetrics(
            avg_latency=100.0,
            p50_latency=90.0,
            p95_latency=180.0,
            p99_latency=200.0,
            success_rate=95.0,
            error_rate=5.0,
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
        )

        assert metrics.avg_latency == 100.0
        assert metrics.p50_latency == 90.0
        assert metrics.p95_latency == 180.0
        assert metrics.p99_latency == 200.0
        assert metrics.success_rate == 95.0
        assert metrics.error_rate == 5.0
        assert metrics.total_requests == 100
        assert metrics.successful_requests == 95
        assert metrics.failed_requests == 5

    def test_init_with_optional_fields(self):
        """Test initialization with optional fields."""
        metrics = AggregatedMetrics(
            avg_latency=100.0,
            p50_latency=90.0,
            p95_latency=180.0,
            p99_latency=200.0,
            success_rate=95.0,
            error_rate=5.0,
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            timeout_rate=1.0,
            avg_ttfb=30.0,
            total_tokens=5000,
            avg_tokens_per_request=50.0,
            total_cost=10.0,
            avg_cost_per_request=0.1,
            cost_per_1k_tokens=2.0,
        )

        assert metrics.timeout_rate == 1.0
        assert metrics.avg_ttfb == 30.0
        assert metrics.total_tokens == 5000
        assert metrics.avg_tokens_per_request == 50.0
        assert metrics.total_cost == 10.0
        assert metrics.avg_cost_per_request == 0.1
        assert metrics.cost_per_1k_tokens == 2.0


class TestMetricsCalculator:
    """Test cases for MetricsCalculator class."""

    def test_init(self):
        """Test initialization."""
        calculator = MetricsCalculator()
        assert calculator.request_metrics == []

    def test_add_request_metric(self):
        """Test adding request metric."""
        calculator = MetricsCalculator()
        metric = RequestMetrics(latency=100.0)

        calculator.add_request_metric(metric)

        assert len(calculator.request_metrics) == 1
        assert calculator.request_metrics[0] == metric

    def test_add_request_data(self):
        """Test adding request data."""
        calculator = MetricsCalculator()

        calculator.add_request_data(
            latency=150.0,
            success=True,
            ttfb=50.0,
            tokens_input=100,
            tokens_output=50,
            cost=0.05,
            provider="openai",
            model="gpt-4",
        )

        assert len(calculator.request_metrics) == 1
        metric = calculator.request_metrics[0]
        assert metric.latency == 150.0
        assert metric.success is True
        assert metric.ttfb == 50.0
        assert metric.tokens_input == 100
        assert metric.tokens_output == 50
        assert metric.tokens_total == 150  # Auto-calculated
        assert metric.cost == 0.05
        assert metric.provider == "openai"
        assert metric.model == "gpt-4"

    def test_add_request_data_minimal(self):
        """Test adding minimal request data."""
        calculator = MetricsCalculator()

        calculator.add_request_data(latency=100.0)

        assert len(calculator.request_metrics) == 1
        metric = calculator.request_metrics[0]
        assert metric.latency == 100.0
        assert metric.success is True
        assert metric.tokens_total is None

    def test_add_request_data_failed(self):
        """Test adding failed request data."""
        calculator = MetricsCalculator()

        calculator.add_request_data(
            latency=5000.0,
            success=False,
            error_type="timeout",
            provider="openai",
            model="gpt-4",
        )

        assert len(calculator.request_metrics) == 1
        metric = calculator.request_metrics[0]
        assert metric.latency == 5000.0
        assert metric.success is False
        assert metric.error_type == "timeout"

    def test_calculate_aggregated_metrics_empty(self):
        """Test calculating metrics with no data."""
        calculator = MetricsCalculator()

        metrics = calculator.calculate_aggregated_metrics()

        assert metrics.avg_latency == 0.0
        assert metrics.p50_latency == 0.0
        assert metrics.success_rate == 0.0
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0

    def test_calculate_aggregated_metrics_single_request(self):
        """Test calculating metrics with single request."""
        calculator = MetricsCalculator()
        calculator.add_request_data(
            latency=100.0,
            success=True,
            ttfb=30.0,
            tokens_input=50,
            tokens_output=25,
            cost=0.02,
            provider="openai",
            model="gpt-4",
        )

        metrics = calculator.calculate_aggregated_metrics()

        assert metrics.avg_latency == 100.0
        assert metrics.p50_latency == 100.0
        assert metrics.p95_latency == 100.0
        assert metrics.p99_latency == 100.0
        assert metrics.success_rate == 100.0
        assert metrics.error_rate == 0.0
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 0
        assert metrics.avg_ttfb == 30.0
        assert metrics.total_tokens == 75
        assert metrics.avg_tokens_per_request == 75.0
        assert metrics.total_cost == 0.02
        assert metrics.avg_cost_per_request == 0.02

    def test_calculate_aggregated_metrics_multiple_requests(self):
        """Test calculating metrics with multiple requests."""
        calculator = MetricsCalculator()

        # Add successful requests
        calculator.add_request_data(latency=100.0, success=True, cost=0.01)
        calculator.add_request_data(latency=200.0, success=True, cost=0.02)
        calculator.add_request_data(latency=150.0, success=True, cost=0.015)

        # Add failed request
        calculator.add_request_data(latency=5000.0, success=False, error_type="timeout")

        metrics = calculator.calculate_aggregated_metrics()

        assert metrics.avg_latency == 1362.5  # (100+200+150+5000)/4
        assert metrics.success_rate == 75.0  # 3/4
        assert metrics.error_rate == 25.0  # 1/4
        assert metrics.total_requests == 4
        assert metrics.successful_requests == 3
        assert metrics.failed_requests == 1
        assert metrics.total_cost == 0.045  # 0.01+0.02+0.015
        assert (
            metrics.avg_cost_per_request == 0.015
        )  # 0.045/3 (only successful requests)

    def test_calculate_aggregated_metrics_with_ttfb(self):
        """Test calculating metrics with TTFB data."""
        calculator = MetricsCalculator()

        calculator.add_request_data(latency=100.0, ttfb=20.0)
        calculator.add_request_data(latency=200.0, ttfb=40.0)
        calculator.add_request_data(latency=150.0, ttfb=30.0)

        metrics = calculator.calculate_aggregated_metrics()

        assert metrics.avg_ttfb == 30.0  # (20+40+30)/3
        assert metrics.p50_ttfb == 30.0
        assert abs(metrics.p95_ttfb - 39.0) < 0.1  # Expected percentile for 3 values
        assert abs(metrics.p99_ttfb - 39.8) < 0.1

    def test_calculate_aggregated_metrics_with_tokens(self):
        """Test calculating metrics with token data."""
        calculator = MetricsCalculator()

        calculator.add_request_data(
            latency=100.0, tokens_input=50, tokens_output=25, cost=0.01
        )
        calculator.add_request_data(
            latency=200.0, tokens_input=100, tokens_output=50, cost=0.03
        )

        metrics = calculator.calculate_aggregated_metrics()

        assert metrics.total_tokens == 225  # (50+25) + (100+50)
        assert metrics.avg_tokens_per_request == 112.5  # 225/2
        assert metrics.avg_input_tokens == 75.0  # (50+100)/2
        assert metrics.avg_output_tokens == 37.5  # (25+50)/2
        assert metrics.cost_per_1k_tokens == pytest.approx(0.04 / 225 * 1000, rel=1e-3)

    def test_calculate_aggregated_metrics_filter_provider(self):
        """Test calculating metrics filtered by provider."""
        calculator = MetricsCalculator()

        calculator.add_request_data(latency=100.0, provider="openai")
        calculator.add_request_data(latency=200.0, provider="anthropic")
        calculator.add_request_data(latency=150.0, provider="openai")

        metrics = calculator.calculate_aggregated_metrics(filter_provider="openai")

        assert metrics.total_requests == 2
        assert metrics.avg_latency == 125.0  # (100+150)/2

    def test_calculate_aggregated_metrics_filter_model(self):
        """Test calculating metrics filtered by model."""
        calculator = MetricsCalculator()

        calculator.add_request_data(latency=100.0, model="gpt-4")
        calculator.add_request_data(latency=200.0, model="gpt-3.5-turbo")
        calculator.add_request_data(latency=150.0, model="gpt-4")

        metrics = calculator.calculate_aggregated_metrics(filter_model="gpt-4")

        assert metrics.total_requests == 2
        assert metrics.avg_latency == 125.0  # (100+150)/2

    def test_calculate_aggregated_metrics_filter_none_match(self):
        """Test calculating metrics with filter that matches nothing."""
        calculator = MetricsCalculator()

        calculator.add_request_data(latency=100.0, provider="openai")
        calculator.add_request_data(latency=200.0, provider="anthropic")

        metrics = calculator.calculate_aggregated_metrics(filter_provider="nonexistent")

        assert metrics.total_requests == 0
        assert metrics.avg_latency == 0.0

    def test_calculate_provider_comparison(self):
        """Test calculating provider comparison metrics."""
        calculator = MetricsCalculator()

        calculator.add_request_data(latency=100.0, provider="openai")
        calculator.add_request_data(latency=200.0, provider="anthropic")
        calculator.add_request_data(latency=150.0, provider="openai")
        calculator.add_request_data(latency=250.0, provider="anthropic")

        comparison = calculator.calculate_provider_comparison()

        assert "openai" in comparison
        assert "anthropic" in comparison
        assert comparison["openai"].total_requests == 2
        assert comparison["openai"].avg_latency == 125.0  # (100+150)/2
        assert comparison["anthropic"].total_requests == 2
        assert comparison["anthropic"].avg_latency == 225.0  # (200+250)/2

    def test_calculate_model_comparison(self):
        """Test calculating model comparison metrics."""
        calculator = MetricsCalculator()

        calculator.add_request_data(latency=100.0, model="gpt-4")
        calculator.add_request_data(latency=200.0, model="gpt-3.5-turbo")
        calculator.add_request_data(latency=150.0, model="gpt-4")
        calculator.add_request_data(latency=250.0, model="gpt-3.5-turbo")

        comparison = calculator.calculate_model_comparison()

        assert "gpt-4" in comparison
        assert "gpt-3.5-turbo" in comparison
        assert comparison["gpt-4"].total_requests == 2
        assert comparison["gpt-4"].avg_latency == 125.0  # (100+150)/2
        assert comparison["gpt-3.5-turbo"].total_requests == 2
        assert comparison["gpt-3.5-turbo"].avg_latency == 225.0  # (200+250)/2

    def test_calculate_latency_distribution(self):
        """Test latency distribution calculation."""
        calculator = MetricsCalculator()

        # Add requests with various latencies
        calculator.add_request_data(latency=50.0)  # 0-100ms
        calculator.add_request_data(latency=150.0)  # 100-200ms
        calculator.add_request_data(latency=250.0)  # 200-300ms
        calculator.add_request_data(latency=950.0)  # 900ms+

        metrics = calculator.calculate_aggregated_metrics()

        assert metrics.latency_distribution["0-100ms"] == 1
        assert metrics.latency_distribution["100-200ms"] == 1
        assert metrics.latency_distribution["200-300ms"] == 1
        assert metrics.latency_distribution["300-400ms"] == 0
        assert metrics.latency_distribution["900ms+"] == 1

    def test_calculate_error_breakdown(self):
        """Test error breakdown calculation."""
        calculator = MetricsCalculator()

        calculator.add_request_data(latency=100.0, success=True)
        calculator.add_request_data(latency=5000.0, success=False, error_type="timeout")
        calculator.add_request_data(
            latency=1000.0, success=False, error_type="rate_limit"
        )
        calculator.add_request_data(latency=2000.0, success=False, error_type="timeout")
        calculator.add_request_data(latency=3000.0, success=False)  # No error_type

        metrics = calculator.calculate_aggregated_metrics()

        assert metrics.error_breakdown["timeout"] == 2
        assert metrics.error_breakdown["rate_limit"] == 1
        assert metrics.error_breakdown["unknown"] == 1

    def test_export_metrics_summary(self):
        """Test exporting comprehensive metrics summary."""
        calculator = MetricsCalculator()

        calculator.add_request_data(latency=100.0, provider="openai", model="gpt-4")
        calculator.add_request_data(
            latency=200.0, provider="anthropic", model="claude-3"
        )
        calculator.add_request_data(latency=150.0, provider="openai", model="gpt-3.5")

        summary = calculator.export_metrics_summary()

        assert "overall" in summary
        assert "by_provider" in summary
        assert "by_model" in summary
        assert "summary" in summary

        # Check overall metrics
        assert summary["overall"]["total_requests"] == 3
        assert summary["overall"]["avg_latency"] == 150.0

        # Check provider breakdown
        assert "openai" in summary["by_provider"]
        assert "anthropic" in summary["by_provider"]
        assert summary["by_provider"]["openai"]["total_requests"] == 2
        assert summary["by_provider"]["anthropic"]["total_requests"] == 1

        # Check model breakdown
        assert "gpt-4" in summary["by_model"]
        assert "claude-3" in summary["by_model"]
        assert "gpt-3.5" in summary["by_model"]

        # Check summary stats
        assert summary["summary"]["total_requests"] == 3
        assert summary["summary"]["unique_providers"] == 2
        assert summary["summary"]["unique_models"] == 3
        assert summary["summary"]["evaluation_duration"] is None

    def test_clear_metrics(self):
        """Test clearing all metrics."""
        calculator = MetricsCalculator()

        calculator.add_request_data(latency=100.0)
        calculator.add_request_data(latency=200.0)
        assert len(calculator.request_metrics) == 2

        calculator.clear_metrics()
        assert len(calculator.request_metrics) == 0

    def test_calculate_token_metrics_empty(self):
        """Test token metrics calculation with empty data."""
        calculator = MetricsCalculator()

        result = calculator._calculate_token_metrics([])

        assert result["total_tokens"] == 0
        assert result["avg_tokens_per_request"] == 0
        assert result["avg_input_tokens"] == 0
        assert result["avg_output_tokens"] == 0

    def test_calculate_cost_metrics_empty(self):
        """Test cost metrics calculation with empty data."""
        calculator = MetricsCalculator()

        result = calculator._calculate_cost_metrics([])

        assert result["total_cost"] == 0
        assert result["avg_cost_per_request"] == 0
        assert result["cost_per_1k_tokens"] == 0

    def test_calculate_cost_metrics_no_tokens(self):
        """Test cost metrics calculation with no token data."""
        calculator = MetricsCalculator()

        metrics = [
            RequestMetrics(latency=100.0, cost=0.01),
            RequestMetrics(latency=200.0, cost=0.02),
        ]

        result = calculator._calculate_cost_metrics(metrics)

        assert result["total_cost"] == 0.03
        assert result["avg_cost_per_request"] == 0.015
        assert result["cost_per_1k_tokens"] == 0  # No tokens

    def test_calculate_latency_distribution_empty(self):
        """Test latency distribution with empty data."""
        calculator = MetricsCalculator()

        result = calculator._calculate_latency_distribution([])

        assert result == {}

    def test_calculate_error_breakdown_empty(self):
        """Test error breakdown with empty data."""
        calculator = MetricsCalculator()

        result = calculator._calculate_error_breakdown([])

        assert result == {}

    def test_provider_comparison_empty(self):
        """Test provider comparison with no provider data."""
        calculator = MetricsCalculator()

        calculator.add_request_data(latency=100.0)  # No provider

        comparison = calculator.calculate_provider_comparison()

        assert comparison == {}

    def test_model_comparison_empty(self):
        """Test model comparison with no model data."""
        calculator = MetricsCalculator()

        calculator.add_request_data(latency=100.0)  # No model

        comparison = calculator.calculate_model_comparison()

        assert comparison == {}

    def test_percentile_calculations(self):
        """Test percentile calculations with various data."""
        calculator = MetricsCalculator()

        # Add data with known percentiles
        latencies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for latency in latencies:
            calculator.add_request_data(latency=latency)

        metrics = calculator.calculate_aggregated_metrics()

        # With 10 values, p50 should be 55, p95 should be ~95, p99 should be ~99
        assert metrics.p50_latency == 55.0
        assert (
            abs(metrics.p95_latency - 95.0) < 0.6
        )  # Allow for numpy percentile calculation differences
        assert abs(metrics.p99_latency - 99.0) < 0.2

    def test_metrics_with_none_values(self):
        """Test metrics calculation with None values."""
        calculator = MetricsCalculator()

        calculator.add_request_data(
            latency=100.0,
            ttfb=None,
            tokens_input=None,
            tokens_output=None,
            cost=None,
        )

        metrics = calculator.calculate_aggregated_metrics()

        assert metrics.avg_ttfb is None
        assert metrics.total_tokens == 0
        assert metrics.total_cost == 0.0
