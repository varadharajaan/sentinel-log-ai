"""
Ablation testing framework for component contribution analysis.

This module provides tools for running ablation studies to understand
how different components and configurations affect clustering quality.

Ablation studies help answer questions like:
- What happens if we disable normalization?
- How does model A compare to model B?
- What is the impact of different min_cluster_size values?

Design Patterns:
- Strategy Pattern: Pluggable component configurations
- Template Method: Ablation study workflow
- Factory Pattern: Configuration creation
- Observer Pattern: Real-time result tracking

SOLID Principles:
- Single Responsibility: Each class handles one concern
- Open/Closed: Extensible via new component types
- Liskov Substitution: All configurations implement same interface
- Interface Segregation: Minimal configuration interface
- Dependency Inversion: Depends on abstractions
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

from sentinel_ml.evaluation.metrics import ClusteringQualityResult, QualityEvaluator
from sentinel_ml.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from numpy.typing import NDArray

logger = get_logger(__name__)


class ComponentType(str, Enum):
    """Types of components that can be ablated."""

    NORMALIZATION = "normalization"
    EMBEDDING_MODEL = "embedding_model"
    CLUSTERING_ALGORITHM = "clustering_algorithm"
    MIN_CLUSTER_SIZE = "min_cluster_size"
    MIN_SAMPLES = "min_samples"
    CLUSTER_EPSILON = "cluster_epsilon"
    CUSTOM = "custom"


class ComparisonMode(str, Enum):
    """Mode for comparing ablation results."""

    PAIRWISE = "pairwise"  # Compare each config to baseline
    ALL_PAIRS = "all_pairs"  # Compare all configurations
    BASELINE_ONLY = "baseline_only"  # Only compare to baseline


@dataclass
class ComponentConfig:
    """
    Configuration for a component in an ablation study.

    Attributes:
        name: Human-readable name for this configuration.
        component_type: Type of component being configured.
        enabled: Whether the component is enabled.
        value: Configuration value (type depends on component).
        description: Description of this configuration.
        metadata: Additional metadata.
    """

    name: str
    component_type: ComponentType
    enabled: bool = True
    value: Any = None
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "component_type": self.component_type.value,
            "enabled": self.enabled,
            "value": self.value,
            "description": self.description,
            "metadata": self.metadata,
        }


@dataclass
class AblationConfig:
    """
    Configuration for an ablation study.

    Attributes:
        name: Name of the ablation study.
        baseline_config: Baseline configuration for comparison.
        test_configs: Configurations to test against baseline.
        comparison_mode: How to compare results.
        n_runs: Number of runs per configuration for statistical significance.
        random_seed: Random seed for reproducibility.
        metrics_to_track: List of metrics to track.
    """

    name: str
    baseline_config: ComponentConfig
    test_configs: list[ComponentConfig] = field(default_factory=list)
    comparison_mode: ComparisonMode = ComparisonMode.PAIRWISE
    n_runs: int = 3
    random_seed: int = 42
    metrics_to_track: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Set default metrics if not provided."""
        if not self.metrics_to_track:
            self.metrics_to_track = [
                "silhouette",
                "davies_bouldin",
                "calinski_harabasz",
            ]

    @property
    def n_configs(self) -> int:
        """Total number of configurations including baseline."""
        return 1 + len(self.test_configs)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "baseline_config": self.baseline_config.to_dict(),
            "test_configs": [c.to_dict() for c in self.test_configs],
            "comparison_mode": self.comparison_mode.value,
            "n_runs": self.n_runs,
            "random_seed": self.random_seed,
            "metrics_to_track": self.metrics_to_track,
        }


@dataclass
class AblationResult:
    """
    Result of a single ablation configuration run.

    Attributes:
        config: Configuration that was tested.
        quality_result: Clustering quality result.
        run_index: Which run this is (for multi-run studies).
        execution_time_seconds: Time taken for this run.
        timestamp: When the run was executed.
        error: Error message if run failed.
    """

    config: ComponentConfig
    quality_result: ClusteringQualityResult | None = None
    run_index: int = 0
    execution_time_seconds: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """Whether the run succeeded."""
        return self.error is None and self.quality_result is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config": self.config.to_dict(),
            "quality_result": (self.quality_result.to_dict() if self.quality_result else None),
            "run_index": self.run_index,
            "execution_time_seconds": round(self.execution_time_seconds, 4),
            "timestamp": self.timestamp.isoformat(),
            "error": self.error,
            "succeeded": self.succeeded,
        }


@dataclass
class ConfigComparison:
    """
    Comparison between two configurations.

    Attributes:
        config_a: First configuration.
        config_b: Second configuration (usually baseline).
        metric_diffs: Difference in metrics (a - b).
        winner: Name of configuration with better overall quality.
        significance: Statistical significance if multiple runs.
    """

    config_a: str
    config_b: str
    metric_diffs: dict[str, float] = field(default_factory=dict)
    winner: str = ""
    significance: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config_a": self.config_a,
            "config_b": self.config_b,
            "metric_diffs": {k: round(v, 6) for k, v in self.metric_diffs.items()},
            "winner": self.winner,
            "significance": {k: round(v, 4) for k, v in self.significance.items()},
        }


@dataclass
class AblationStudy:
    """
    Complete ablation study with all results.

    Attributes:
        config: Study configuration.
        results: Results for each configuration and run.
        comparisons: Comparisons between configurations.
        summary: Summary of findings.
        start_time: When study started.
        end_time: When study ended.
    """

    config: AblationConfig
    results: list[AblationResult] = field(default_factory=list)
    comparisons: list[ConfigComparison] = field(default_factory=list)
    summary: str = ""
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime | None = None

    @property
    def duration_seconds(self) -> float:
        """Total study duration."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    @property
    def n_successful(self) -> int:
        """Number of successful runs."""
        return sum(1 for r in self.results if r.succeeded)

    @property
    def n_failed(self) -> int:
        """Number of failed runs."""
        return sum(1 for r in self.results if not r.succeeded)

    def get_results_for_config(self, config_name: str) -> list[AblationResult]:
        """Get all results for a specific configuration."""
        return [r for r in self.results if r.config.name == config_name]

    def get_best_config(self) -> str | None:
        """Get the configuration with best average quality."""
        if not self.results:
            return None

        config_scores: dict[str, list[float]] = {}
        for result in self.results:
            if result.succeeded and result.quality_result:
                name = result.config.name
                if name not in config_scores:
                    config_scores[name] = []
                config_scores[name].append(result.quality_result.overall_quality)

        if not config_scores:
            return None

        best_name = max(
            config_scores.keys(),
            key=lambda k: sum(config_scores[k]) / len(config_scores[k]),
        )
        return best_name

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config": self.config.to_dict(),
            "results": [r.to_dict() for r in self.results],
            "comparisons": [c.to_dict() for c in self.comparisons],
            "summary": self.summary,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": round(self.duration_seconds, 2),
            "n_successful": self.n_successful,
            "n_failed": self.n_failed,
            "best_config": self.get_best_config(),
        }


class AblationRunner:
    """
    Runner for ablation studies.

    Coordinates the ablation study workflow including:
    - Running each configuration multiple times
    - Computing quality metrics
    - Comparing configurations
    - Generating summary

    Usage:
        runner = AblationRunner()
        study = runner.run(config, clustering_func)
    """

    def __init__(
        self,
        evaluator: QualityEvaluator | None = None,
    ) -> None:
        """
        Initialize the ablation runner.

        Args:
            evaluator: Quality evaluator. Uses default if not provided.
        """
        self._evaluator = evaluator or QualityEvaluator()

        logger.info("ablation_runner_initialized")

    def run(
        self,
        config: AblationConfig,
        cluster_func: Callable[[ComponentConfig, NDArray[np.float32]], NDArray[np.int32]],
        embeddings: NDArray[np.float32],
    ) -> AblationStudy:
        """
        Run an ablation study.

        Args:
            config: Ablation study configuration.
            cluster_func: Function that takes config and embeddings, returns labels.
            embeddings: Embeddings to cluster.

        Returns:
            Completed AblationStudy.
        """
        study = AblationStudy(config=config)

        logger.info(
            "ablation_study_started",
            name=config.name,
            n_configs=config.n_configs,
            n_runs=config.n_runs,
        )

        # Run baseline
        study.results.extend(
            self._run_config(config.baseline_config, cluster_func, embeddings, config.n_runs)
        )

        # Run test configurations
        for test_config in config.test_configs:
            study.results.extend(
                self._run_config(test_config, cluster_func, embeddings, config.n_runs)
            )

        # Compute comparisons
        study.comparisons = self._compute_comparisons(study, config)

        # Generate summary
        study.summary = self._generate_summary(study)

        study.end_time = datetime.now(timezone.utc)

        logger.info(
            "ablation_study_completed",
            duration_seconds=round(study.duration_seconds, 2),
            n_successful=study.n_successful,
            n_failed=study.n_failed,
            best_config=study.get_best_config(),
        )

        return study

    def _run_config(
        self,
        config: ComponentConfig,
        cluster_func: Callable[[ComponentConfig, NDArray[np.float32]], NDArray[np.int32]],
        embeddings: NDArray[np.float32],
        n_runs: int,
    ) -> list[AblationResult]:
        """Run a configuration multiple times."""
        results = []

        for run_idx in range(n_runs):
            start_time = time.perf_counter()

            try:
                # Run clustering
                labels = cluster_func(config, embeddings)

                # Evaluate quality
                quality_result = self._evaluator.evaluate(embeddings, labels)

                result = AblationResult(
                    config=config,
                    quality_result=quality_result,
                    run_index=run_idx,
                    execution_time_seconds=time.perf_counter() - start_time,
                )

                logger.debug(
                    "ablation_run_completed",
                    config=config.name,
                    run=run_idx + 1,
                    quality=round(quality_result.overall_quality, 4),
                )

            except Exception as e:
                result = AblationResult(
                    config=config,
                    run_index=run_idx,
                    execution_time_seconds=time.perf_counter() - start_time,
                    error=str(e),
                )

                logger.error(
                    "ablation_run_failed",
                    config=config.name,
                    run=run_idx + 1,
                    error=str(e),
                )

            results.append(result)

        return results

    def _compute_comparisons(
        self,
        study: AblationStudy,
        config: AblationConfig,
    ) -> list[ConfigComparison]:
        """Compute comparisons between configurations."""
        comparisons = []

        baseline_name = config.baseline_config.name
        baseline_results = study.get_results_for_config(baseline_name)

        if config.comparison_mode == ComparisonMode.PAIRWISE:
            # Compare each test config to baseline
            for test_config in config.test_configs:
                test_results = study.get_results_for_config(test_config.name)
                comparison = self._compare_configs(
                    test_config.name, test_results, baseline_name, baseline_results
                )
                comparisons.append(comparison)

        elif config.comparison_mode == ComparisonMode.ALL_PAIRS:
            # Compare all pairs
            all_configs = [config.baseline_config, *config.test_configs]
            for i, cfg_a in enumerate(all_configs):
                for cfg_b in all_configs[i + 1 :]:
                    results_a = study.get_results_for_config(cfg_a.name)
                    results_b = study.get_results_for_config(cfg_b.name)
                    comparison = self._compare_configs(cfg_a.name, results_a, cfg_b.name, results_b)
                    comparisons.append(comparison)

        return comparisons

    def _compare_configs(
        self,
        name_a: str,
        results_a: list[AblationResult],
        name_b: str,
        results_b: list[AblationResult],
    ) -> ConfigComparison:
        """Compare two configurations."""
        # Get successful results
        success_a = [r for r in results_a if r.succeeded and r.quality_result]
        success_b = [r for r in results_b if r.succeeded and r.quality_result]

        metric_diffs: dict[str, float] = {}
        significance: dict[str, float] = {}

        if success_a and success_b:
            # Compare overall quality
            avg_a = sum(
                r.quality_result.overall_quality for r in success_a if r.quality_result
            ) / len(success_a)
            avg_b = sum(
                r.quality_result.overall_quality for r in success_b if r.quality_result
            ) / len(success_b)
            metric_diffs["overall_quality"] = avg_a - avg_b

            # Compare individual metrics
            for result_a in success_a:
                if result_a.quality_result:
                    for metric in result_a.quality_result.metrics:
                        if metric.metric_type.value not in metric_diffs:
                            metric_diffs[metric.metric_type.value] = 0.0

            # Compute p-values if enough samples
            if len(success_a) >= 3 and len(success_b) >= 3:
                significance = self._compute_significance(success_a, success_b)

        # Determine winner
        winner = ""
        if "overall_quality" in metric_diffs:
            winner = name_a if metric_diffs["overall_quality"] > 0 else name_b

        return ConfigComparison(
            config_a=name_a,
            config_b=name_b,
            metric_diffs=metric_diffs,
            winner=winner,
            significance=significance,
        )

    def _compute_significance(
        self,
        results_a: list[AblationResult],
        results_b: list[AblationResult],
    ) -> dict[str, float]:
        """Compute statistical significance using t-test."""
        from scipy import stats

        significance: dict[str, float] = {}

        scores_a = [r.quality_result.overall_quality for r in results_a if r.quality_result]
        scores_b = [r.quality_result.overall_quality for r in results_b if r.quality_result]

        if len(scores_a) >= 2 and len(scores_b) >= 2:
            _, p_value = stats.ttest_ind(scores_a, scores_b)
            significance["overall_quality_p_value"] = float(p_value)

        return significance

    def _generate_summary(self, study: AblationStudy) -> str:
        """Generate summary of ablation study."""
        parts = []

        parts.append(f"Ablation study '{study.config.name}' completed.")
        parts.append(
            f"Tested {study.config.n_configs} configurations with {study.config.n_runs} runs each."
        )
        parts.append(f"Successful runs: {study.n_successful}, Failed: {study.n_failed}.")

        best_config = study.get_best_config()
        if best_config:
            parts.append(f"Best performing configuration: {best_config}.")

        # Highlight key findings
        if study.comparisons:
            for comparison in study.comparisons:
                if comparison.winner:
                    diff = comparison.metric_diffs.get("overall_quality", 0)
                    parts.append(
                        f"{comparison.winner} outperformed "
                        f"{'baseline' if comparison.config_b == study.config.baseline_config.name else comparison.config_b} "
                        f"by {abs(diff):.2%} overall quality."
                    )

        return " ".join(parts)


def create_parameter_sweep_configs(
    component_type: ComponentType,
    parameter_name: str,
    values: list[Any],
    baseline_value: Any,
) -> tuple[ComponentConfig, list[ComponentConfig]]:
    """
    Create configurations for a parameter sweep.

    Args:
        component_type: Type of component being configured.
        parameter_name: Name of the parameter being swept.
        values: Values to test.
        baseline_value: Baseline value for comparison.

    Returns:
        Tuple of (baseline_config, test_configs).
    """
    baseline = ComponentConfig(
        name=f"{parameter_name}={baseline_value} (baseline)",
        component_type=component_type,
        value=baseline_value,
        description=f"Baseline configuration with {parameter_name}={baseline_value}",
    )

    test_configs = []
    for value in values:
        if value != baseline_value:
            test_configs.append(
                ComponentConfig(
                    name=f"{parameter_name}={value}",
                    component_type=component_type,
                    value=value,
                    description=f"Test configuration with {parameter_name}={value}",
                )
            )

    return baseline, test_configs


def create_toggle_config(
    component_type: ComponentType,
    component_name: str,
    enabled: bool = True,
) -> ComponentConfig:
    """
    Create a configuration for toggling a component on/off.

    Args:
        component_type: Type of component.
        component_name: Name of the component.
        enabled: Whether component is enabled.

    Returns:
        ComponentConfig for the toggle.
    """
    state = "enabled" if enabled else "disabled"
    return ComponentConfig(
        name=f"{component_name}_{state}",
        component_type=component_type,
        enabled=enabled,
        description=f"{component_name} is {state}",
    )
