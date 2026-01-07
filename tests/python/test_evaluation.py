"""
Comprehensive unit tests for the evaluation module.

Tests cover:
- ClusteringQualityMetrics: Silhouette, Davies-Bouldin, Calinski-Harabasz
- QualityEvaluator: Aggregate quality evaluation
- GoldenDataset: Dataset management and regression testing
- EvaluationReport: Report generation and formatting
- AblationStudy: Ablation testing framework
- LabelingTool: Human labeling infrastructure
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from sentinel_ml.evaluation.ablation import (
    AblationConfig,
    AblationResult,
    AblationRunner,
    AblationStudy,
    ComponentConfig,
    ComponentType,
    create_parameter_sweep_configs,
)
from sentinel_ml.evaluation.golden_dataset import (
    ComparisonStatus,
    ExpectedCluster,
    GoldenDataset,
    GoldenDatasetManager,
    GoldenRecord,
    RegressionRunner,
)
from sentinel_ml.evaluation.labeling import (
    ClusterPair,
    InterRaterAgreement,
    LabelingResult,
    LabelingSession,
    LabelingTool,
    LabelType,
)
from sentinel_ml.evaluation.metrics import (
    CalinskiHarabaszMetric,
    ClusteringQualityResult,
    DaviesBouldinMetric,
    MetricResult,
    MetricType,
    QualityEvaluator,
    SilhouetteMetric,
)
from sentinel_ml.evaluation.report import (
    EvaluationReportConfig,
    EvaluationReportGenerator,
    ReportFormat,
    TrendAnalysis,
    TrendDirection,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """Create sample embeddings with distinct clusters."""
    np.random.seed(42)

    # Create 3 well-separated clusters
    cluster1 = np.random.randn(30, 32).astype(np.float32) + np.array([10, 0] + [0] * 30)
    cluster2 = np.random.randn(30, 32).astype(np.float32) + np.array([-10, 0] + [0] * 30)
    cluster3 = np.random.randn(30, 32).astype(np.float32) + np.array([0, 10] + [0] * 30)

    return np.vstack([cluster1, cluster2, cluster3])


@pytest.fixture
def sample_labels() -> np.ndarray:
    """Create sample cluster labels matching embeddings."""
    return np.array([0] * 30 + [1] * 30 + [2] * 30, dtype=np.int32)


@pytest.fixture
def sample_labels_with_noise() -> np.ndarray:
    """Create sample cluster labels with noise points."""
    labels = [0] * 25 + [-1] * 5 + [1] * 25 + [-1] * 5 + [2] * 25 + [-1] * 5
    return np.array(labels, dtype=np.int32)


@pytest.fixture
def golden_dataset() -> GoldenDataset:
    """Create a sample golden dataset."""
    records = [
        GoldenRecord(
            id=f"rec-{i}",
            message=f"Test message {i}",
            normalized="Test message <num>",
            expected_cluster_id=f"cluster-{i // 10}",
            source="test.log",
            level="INFO",
        )
        for i in range(30)
    ]

    clusters = [
        ExpectedCluster(
            id=f"cluster-{i}",
            name=f"Test Cluster {i}",
            description=f"Cluster containing test pattern {i}",
            expected_size=10,
        )
        for i in range(3)
    ]

    return GoldenDataset(
        name="test_dataset",
        version="1.0",
        description="Test golden dataset",
        records=records,
        expected_clusters=clusters,
    )


@pytest.fixture
def tmp_path_factory_session(tmp_path: Path) -> Path:
    """Create a temporary directory for session tests."""
    session_dir = tmp_path / "labeling_sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


# ============================================================================
# Metrics Tests
# ============================================================================


class TestMetricResult:
    """Tests for MetricResult dataclass."""

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        result = MetricResult(
            metric_type=MetricType.SILHOUETTE,
            value=0.75,
            interpretation="Good cluster separation",
            optimal_direction="higher",
            computation_time_seconds=0.123,
            metadata={"n_samples": 100},
        )

        data = result.to_dict()

        assert data["metric_type"] == "silhouette"
        assert data["value"] == 0.75
        assert data["interpretation"] == "Good cluster separation"
        assert data["optimal_direction"] == "higher"
        assert data["computation_time_seconds"] == 0.123
        assert data["metadata"]["n_samples"] == 100


class TestSilhouetteMetric:
    """Tests for SilhouetteMetric."""

    def test_compute_valid_clusters(
        self, sample_embeddings: np.ndarray, sample_labels: np.ndarray
    ) -> None:
        """Test computation with valid cluster data."""
        metric = SilhouetteMetric()
        result = metric.compute(sample_embeddings, sample_labels)

        assert result.metric_type == MetricType.SILHOUETTE
        assert -1.0 <= result.value <= 1.0
        assert result.optimal_direction == "higher"
        assert result.computation_time_seconds > 0
        assert "n_samples" in result.metadata
        assert "n_clusters" in result.metadata

    def test_compute_with_noise(
        self, sample_embeddings: np.ndarray, sample_labels_with_noise: np.ndarray
    ) -> None:
        """Test computation handles noise points correctly."""
        metric = SilhouetteMetric()
        result = metric.compute(sample_embeddings, sample_labels_with_noise)

        assert result.metric_type == MetricType.SILHOUETTE
        assert not math.isnan(result.value)
        # Noise points should be filtered
        assert result.metadata["n_samples"] < len(sample_labels_with_noise)

    def test_compute_insufficient_clusters(self, sample_embeddings: np.ndarray) -> None:
        """Test handling of single cluster."""
        metric = SilhouetteMetric()
        single_cluster_labels = np.zeros(len(sample_embeddings), dtype=np.int32)

        result = metric.compute(sample_embeddings, single_cluster_labels)

        assert math.isnan(result.value)
        assert "error" in result.metadata

    def test_interpret_score_excellent(self) -> None:
        """Test interpretation for excellent score."""
        metric = SilhouetteMetric()
        interpretation = metric._interpret_score(0.85)
        assert "Excellent" in interpretation

    def test_interpret_score_poor(self) -> None:
        """Test interpretation for poor score."""
        metric = SilhouetteMetric()
        interpretation = metric._interpret_score(-0.3)
        assert "Poor" in interpretation


class TestDaviesBouldinMetric:
    """Tests for DaviesBouldinMetric."""

    def test_compute_valid_clusters(
        self, sample_embeddings: np.ndarray, sample_labels: np.ndarray
    ) -> None:
        """Test computation with valid cluster data."""
        metric = DaviesBouldinMetric()
        result = metric.compute(sample_embeddings, sample_labels)

        assert result.metric_type == MetricType.DAVIES_BOULDIN
        assert result.value >= 0
        assert result.optimal_direction == "lower"
        assert result.computation_time_seconds > 0

    def test_interpret_score_excellent(self) -> None:
        """Test interpretation for excellent score."""
        metric = DaviesBouldinMetric()
        interpretation = metric._interpret_score(0.3)
        assert "Excellent" in interpretation


class TestCalinskiHarabaszMetric:
    """Tests for CalinskiHarabaszMetric."""

    def test_compute_valid_clusters(
        self, sample_embeddings: np.ndarray, sample_labels: np.ndarray
    ) -> None:
        """Test computation with valid cluster data."""
        metric = CalinskiHarabaszMetric()
        result = metric.compute(sample_embeddings, sample_labels)

        assert result.metric_type == MetricType.CALINSKI_HARABASZ
        assert result.value >= 0
        assert result.optimal_direction == "higher"


class TestQualityEvaluator:
    """Tests for QualityEvaluator."""

    def test_evaluate_all_metrics(
        self, sample_embeddings: np.ndarray, sample_labels: np.ndarray
    ) -> None:
        """Test evaluation with all default metrics."""
        evaluator = QualityEvaluator()
        result = evaluator.evaluate(sample_embeddings, sample_labels)

        assert isinstance(result, ClusteringQualityResult)
        assert len(result.metrics) == 3
        assert 0.0 <= result.overall_quality <= 1.0
        assert result.n_samples == len(sample_labels)
        assert result.n_clusters == 3
        assert result.evaluation_time_seconds > 0

    def test_evaluate_with_custom_metrics(
        self, sample_embeddings: np.ndarray, sample_labels: np.ndarray
    ) -> None:
        """Test evaluation with custom metric subset."""
        evaluator = QualityEvaluator(metrics=[SilhouetteMetric()])
        result = evaluator.evaluate(sample_embeddings, sample_labels)

        assert len(result.metrics) == 1
        assert result.metrics[0].metric_type == MetricType.SILHOUETTE

    def test_get_metric_by_type(
        self, sample_embeddings: np.ndarray, sample_labels: np.ndarray
    ) -> None:
        """Test retrieving specific metric from result."""
        evaluator = QualityEvaluator()
        result = evaluator.evaluate(sample_embeddings, sample_labels)

        silhouette = result.get_metric(MetricType.SILHOUETTE)
        assert silhouette is not None
        assert silhouette.metric_type == MetricType.SILHOUETTE

    def test_add_metric(self) -> None:
        """Test adding a metric to evaluator."""
        evaluator = QualityEvaluator(metrics=[])
        assert len(evaluator.metrics) == 0

        evaluator.add_metric(SilhouetteMetric())
        assert len(evaluator.metrics) == 1

    def test_remove_metric(self) -> None:
        """Test removing a metric from evaluator."""
        evaluator = QualityEvaluator()
        initial_count = len(evaluator.metrics)

        removed = evaluator.remove_metric(MetricType.SILHOUETTE)
        assert removed is True
        assert len(evaluator.metrics) == initial_count - 1

        # Try removing non-existent metric
        removed_again = evaluator.remove_metric(MetricType.SILHOUETTE)
        assert removed_again is False


class TestClusteringQualityResult:
    """Tests for ClusteringQualityResult."""

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        result = ClusteringQualityResult(
            metrics=[
                MetricResult(
                    metric_type=MetricType.SILHOUETTE,
                    value=0.5,
                    interpretation="Good",
                    optimal_direction="higher",
                )
            ],
            overall_quality=0.7,
            n_samples=100,
            n_clusters=5,
            n_noise=10,
        )

        data = result.to_dict()

        assert data["overall_quality"] == 0.7
        assert data["n_samples"] == 100
        assert data["n_clusters"] == 5
        assert data["n_noise"] == 10
        assert len(data["metrics"]) == 1


# ============================================================================
# Golden Dataset Tests
# ============================================================================


class TestGoldenRecord:
    """Tests for GoldenRecord."""

    def test_to_dict_and_from_dict(self) -> None:
        """Test round-trip serialization."""
        record = GoldenRecord(
            id="test-1",
            message="Test message",
            normalized="Test <var>",
            expected_cluster_id="cluster-1",
            source="app.log",
            level="ERROR",
            attrs={"key": "value"},
        )

        data = record.to_dict()
        restored = GoldenRecord.from_dict(data)

        assert restored.id == record.id
        assert restored.message == record.message
        assert restored.normalized == record.normalized
        assert restored.expected_cluster_id == record.expected_cluster_id
        assert restored.source == record.source
        assert restored.level == record.level
        assert restored.attrs == record.attrs


class TestGoldenDataset:
    """Tests for GoldenDataset."""

    def test_properties(self, golden_dataset: GoldenDataset) -> None:
        """Test dataset properties."""
        assert golden_dataset.n_records == 30
        assert golden_dataset.n_clusters == 3
        assert golden_dataset.name == "test_dataset"

    def test_checksum_consistency(self, golden_dataset: GoldenDataset) -> None:
        """Test that checksum is consistent."""
        checksum1 = golden_dataset.checksum
        checksum2 = golden_dataset.checksum
        assert checksum1 == checksum2

    def test_checksum_changes_on_modification(self, golden_dataset: GoldenDataset) -> None:
        """Test that checksum changes when data changes."""
        original_checksum = golden_dataset.checksum

        golden_dataset.add_record(
            GoldenRecord(
                id="new-record",
                message="New message",
                normalized="New message",
                expected_cluster_id="cluster-0",
            )
        )

        assert golden_dataset.checksum != original_checksum

    def test_save_and_load(self, golden_dataset: GoldenDataset, tmp_path: Path) -> None:
        """Test saving and loading dataset."""
        path = tmp_path / "test_dataset.json"
        golden_dataset.save(path)

        loaded = GoldenDataset.load(path)

        assert loaded.name == golden_dataset.name
        assert loaded.version == golden_dataset.version
        assert loaded.n_records == golden_dataset.n_records
        assert loaded.n_clusters == golden_dataset.n_clusters

    def test_get_expected_labels(self, golden_dataset: GoldenDataset) -> None:
        """Test getting expected label mapping."""
        labels = golden_dataset.get_expected_labels()

        assert len(labels) == 30
        assert labels["rec-0"] == "cluster-0"
        assert labels["rec-15"] == "cluster-1"

    def test_get_cluster_by_id(self, golden_dataset: GoldenDataset) -> None:
        """Test getting cluster by ID."""
        cluster = golden_dataset.get_cluster_by_id("cluster-0")
        assert cluster is not None
        assert cluster.id == "cluster-0"

        missing = golden_dataset.get_cluster_by_id("nonexistent")
        assert missing is None


class TestGoldenDatasetManager:
    """Tests for GoldenDatasetManager."""

    def test_save_and_load(self, golden_dataset: GoldenDataset, tmp_path: Path) -> None:
        """Test manager save and load operations."""
        manager = GoldenDatasetManager(tmp_path)

        path = manager.save(golden_dataset)
        assert path.exists()

        loaded = manager.load(golden_dataset.name, golden_dataset.version)
        assert loaded.name == golden_dataset.name

    def test_list_all(self, golden_dataset: GoldenDataset, tmp_path: Path) -> None:
        """Test listing all datasets."""
        manager = GoldenDatasetManager(tmp_path)
        manager.save(golden_dataset)

        datasets = list(manager.list_all())
        assert len(datasets) == 1
        assert datasets[0][0] == golden_dataset.name

    def test_exists(self, golden_dataset: GoldenDataset, tmp_path: Path) -> None:
        """Test existence check."""
        manager = GoldenDatasetManager(tmp_path)

        assert not manager.exists(golden_dataset.name)

        manager.save(golden_dataset)
        assert manager.exists(golden_dataset.name)
        assert manager.exists(golden_dataset.name, golden_dataset.version)

    def test_delete(self, golden_dataset: GoldenDataset, tmp_path: Path) -> None:
        """Test deleting datasets."""
        manager = GoldenDatasetManager(tmp_path)
        manager.save(golden_dataset)

        assert manager.exists(golden_dataset.name)

        deleted = manager.delete(golden_dataset.name, golden_dataset.version)
        assert deleted is True
        assert not manager.exists(golden_dataset.name, golden_dataset.version)


class TestRegressionRunner:
    """Tests for RegressionRunner."""

    def test_run_regression(
        self,
        golden_dataset: GoldenDataset,
        sample_embeddings: np.ndarray,
    ) -> None:
        """Test running regression test."""
        runner = RegressionRunner()

        # Use labels that match the golden dataset structure
        actual_labels = np.array([i // 10 for i in range(30)], dtype=np.int32)

        result = runner.run(golden_dataset, actual_labels, sample_embeddings)

        assert result.dataset_name == golden_dataset.name
        assert result.dataset_version == golden_dataset.version
        assert result.status in ComparisonStatus
        assert 0.0 <= result.accuracy <= 1.0


# ============================================================================
# Report Tests
# ============================================================================


class TestTrendAnalysis:
    """Tests for TrendAnalysis."""

    def test_to_dict(self) -> None:
        """Test serialization."""
        trend = TrendAnalysis(
            metric_name="silhouette",
            current_value=0.75,
            previous_value=0.70,
            direction=TrendDirection.IMPROVING,
            change_percent=7.14,
            samples_count=5,
        )

        data = trend.to_dict()

        assert data["metric_name"] == "silhouette"
        assert data["current_value"] == 0.75
        assert data["direction"] == "improving"


class TestEvaluationReportGenerator:
    """Tests for EvaluationReportGenerator."""

    def test_generate_report(
        self, sample_embeddings: np.ndarray, sample_labels: np.ndarray
    ) -> None:
        """Test report generation."""
        evaluator = QualityEvaluator()
        quality_result = evaluator.evaluate(sample_embeddings, sample_labels)

        generator = EvaluationReportGenerator()
        report = generator.generate(quality_result)

        assert report.quality_result == quality_result
        assert len(report.summary) > 0
        assert report.generated_at is not None

    def test_format_json(self, sample_embeddings: np.ndarray, sample_labels: np.ndarray) -> None:
        """Test JSON formatting."""
        evaluator = QualityEvaluator()
        quality_result = evaluator.evaluate(sample_embeddings, sample_labels)

        config = EvaluationReportConfig(output_format=ReportFormat.JSON)
        generator = EvaluationReportGenerator(config)
        report = generator.generate(quality_result)

        formatted = generator.format(report)

        # Should be valid JSON
        data = json.loads(formatted)
        assert "title" in data
        assert "quality" in data

    def test_format_markdown(
        self, sample_embeddings: np.ndarray, sample_labels: np.ndarray
    ) -> None:
        """Test Markdown formatting."""
        evaluator = QualityEvaluator()
        quality_result = evaluator.evaluate(sample_embeddings, sample_labels)

        config = EvaluationReportConfig(output_format=ReportFormat.MARKDOWN)
        generator = EvaluationReportGenerator(config)
        report = generator.generate(quality_result)

        formatted = generator.format(report)

        assert formatted.startswith("#")
        assert "## Clustering Quality Metrics" in formatted

    def test_format_html(self, sample_embeddings: np.ndarray, sample_labels: np.ndarray) -> None:
        """Test HTML formatting."""
        evaluator = QualityEvaluator()
        quality_result = evaluator.evaluate(sample_embeddings, sample_labels)

        config = EvaluationReportConfig(output_format=ReportFormat.HTML)
        generator = EvaluationReportGenerator(config)
        report = generator.generate(quality_result)

        formatted = generator.format(report)

        assert "<!DOCTYPE html>" in formatted
        assert "<html>" in formatted

    def test_save_report(
        self, sample_embeddings: np.ndarray, sample_labels: np.ndarray, tmp_path: Path
    ) -> None:
        """Test saving report to file."""
        evaluator = QualityEvaluator()
        quality_result = evaluator.evaluate(sample_embeddings, sample_labels)

        config = EvaluationReportConfig(output_format=ReportFormat.MARKDOWN)
        generator = EvaluationReportGenerator(config)
        report = generator.generate(quality_result)

        output_path = tmp_path / "report.md"
        saved_path = generator.save(report, output_path)

        assert saved_path.exists()
        content = saved_path.read_text()
        assert "Clustering Quality" in content

    def test_add_to_history(self, sample_embeddings: np.ndarray, sample_labels: np.ndarray) -> None:
        """Test adding results to history."""
        evaluator = QualityEvaluator()
        quality_result = evaluator.evaluate(sample_embeddings, sample_labels)

        generator = EvaluationReportGenerator()

        generator.add_to_history(quality_result)
        assert len(generator.history) == 1

        generator.add_to_history(quality_result)
        assert len(generator.history) == 2


# ============================================================================
# Ablation Tests
# ============================================================================


class TestComponentConfig:
    """Tests for ComponentConfig."""

    def test_to_dict(self) -> None:
        """Test serialization."""
        config = ComponentConfig(
            name="test_config",
            component_type=ComponentType.MIN_CLUSTER_SIZE,
            enabled=True,
            value=5,
            description="Test configuration",
        )

        data = config.to_dict()

        assert data["name"] == "test_config"
        assert data["component_type"] == "min_cluster_size"
        assert data["value"] == 5


class TestAblationStudy:
    """Tests for AblationStudy."""

    def test_get_best_config(self) -> None:
        """Test finding best configuration."""
        config = AblationConfig(
            name="test_study",
            baseline_config=ComponentConfig(
                name="baseline",
                component_type=ComponentType.MIN_CLUSTER_SIZE,
                value=5,
            ),
        )

        study = AblationStudy(config=config)

        # Add some results
        quality_good = ClusteringQualityResult(overall_quality=0.8, n_samples=100)
        quality_bad = ClusteringQualityResult(overall_quality=0.4, n_samples=100)

        study.results.append(
            AblationResult(
                config=ComponentConfig(
                    name="config_a", component_type=ComponentType.MIN_CLUSTER_SIZE
                ),
                quality_result=quality_good,
            )
        )
        study.results.append(
            AblationResult(
                config=ComponentConfig(
                    name="config_b", component_type=ComponentType.MIN_CLUSTER_SIZE
                ),
                quality_result=quality_bad,
            )
        )

        best = study.get_best_config()
        assert best == "config_a"


class TestAblationRunner:
    """Tests for AblationRunner."""

    def test_run_ablation_study(
        self, sample_embeddings: np.ndarray, sample_labels: np.ndarray
    ) -> None:
        """Test running an ablation study."""
        baseline = ComponentConfig(
            name="baseline",
            component_type=ComponentType.MIN_CLUSTER_SIZE,
            value=5,
        )
        test_config = ComponentConfig(
            name="test",
            component_type=ComponentType.MIN_CLUSTER_SIZE,
            value=10,
        )

        config = AblationConfig(
            name="test_study",
            baseline_config=baseline,
            test_configs=[test_config],
            n_runs=1,
        )

        runner = AblationRunner()

        # Simple mock clustering function
        def cluster_func(cfg: ComponentConfig, embeddings: np.ndarray) -> np.ndarray:
            return sample_labels

        study = runner.run(config, cluster_func, sample_embeddings)

        assert len(study.results) == 2  # baseline + 1 test
        assert study.n_successful == 2
        assert study.n_failed == 0


class TestCreateParameterSweepConfigs:
    """Tests for create_parameter_sweep_configs helper."""

    def test_creates_baseline_and_test_configs(self) -> None:
        """Test creating parameter sweep configurations."""
        baseline, tests = create_parameter_sweep_configs(
            component_type=ComponentType.MIN_CLUSTER_SIZE,
            parameter_name="min_cluster_size",
            values=[3, 5, 7, 10],
            baseline_value=5,
        )

        assert baseline.value == 5
        assert len(tests) == 3  # All values except baseline
        assert all(t.value != 5 for t in tests)


# ============================================================================
# Labeling Tests
# ============================================================================


class TestClusterPair:
    """Tests for ClusterPair."""

    def test_to_dict_and_from_dict(self) -> None:
        """Test round-trip serialization."""
        pair = ClusterPair(
            id="pair-1",
            cluster_a_id="cluster-1",
            cluster_b_id="cluster-2",
            cluster_a_samples=["sample 1", "sample 2"],
            cluster_b_samples=["sample 3", "sample 4"],
            expected_label=LabelType.SAME,
        )

        data = pair.to_dict()
        restored = ClusterPair.from_dict(data)

        assert restored.id == pair.id
        assert restored.cluster_a_id == pair.cluster_a_id
        assert restored.expected_label == pair.expected_label


class TestLabelingResult:
    """Tests for LabelingResult."""

    def test_to_dict_and_from_dict(self) -> None:
        """Test round-trip serialization."""
        result = LabelingResult(
            pair_id="pair-1",
            label=LabelType.SAME,
            labeler_id="user-1",
            confidence=0.9,
            time_taken_seconds=5.5,
            notes="Clear match",
        )

        data = result.to_dict()
        restored = LabelingResult.from_dict(data)

        assert restored.pair_id == result.pair_id
        assert restored.label == result.label
        assert restored.confidence == result.confidence


class TestLabelingSession:
    """Tests for LabelingSession."""

    def test_session_progress(self) -> None:
        """Test session progress tracking."""
        pairs = [
            ClusterPair(id=f"pair-{i}", cluster_a_id=f"a-{i}", cluster_b_id=f"b-{i}")
            for i in range(10)
        ]

        session = LabelingSession(
            id="session-1",
            labeler_id="user-1",
            pairs=pairs,
        )

        assert session.n_total == 10
        assert session.n_labeled == 0
        assert session.n_remaining == 10
        assert session.progress_percent == 0.0
        assert not session.is_complete

    def test_add_result(self) -> None:
        """Test adding labeling results."""
        pairs = [
            ClusterPair(id=f"pair-{i}", cluster_a_id=f"a-{i}", cluster_b_id=f"b-{i}")
            for i in range(5)
        ]

        session = LabelingSession(
            id="session-1",
            labeler_id="user-1",
            pairs=pairs,
        )

        result = LabelingResult(
            pair_id="pair-0",
            label=LabelType.SAME,
            labeler_id="user-1",
        )

        session.add_result(result)

        assert session.n_labeled == 1
        assert session.current_index == 1
        assert session.progress_percent == 20.0

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Test saving and loading session."""
        pairs = [
            ClusterPair(id=f"pair-{i}", cluster_a_id=f"a-{i}", cluster_b_id=f"b-{i}")
            for i in range(3)
        ]

        session = LabelingSession(
            id="session-1",
            labeler_id="user-1",
            pairs=pairs,
        )

        session.add_result(
            LabelingResult(pair_id="pair-0", label=LabelType.SAME, labeler_id="user-1")
        )

        path = tmp_path / "session.json"
        session.save(path)

        loaded = LabelingSession.load(path)

        assert loaded.id == session.id
        assert loaded.n_labeled == 1
        assert loaded.current_index == 1

    def test_get_label_distribution(self) -> None:
        """Test getting label distribution."""
        session = LabelingSession(
            id="session-1",
            labeler_id="user-1",
            pairs=[],
            results=[
                LabelingResult(pair_id="p1", label=LabelType.SAME, labeler_id="u1"),
                LabelingResult(pair_id="p2", label=LabelType.SAME, labeler_id="u1"),
                LabelingResult(pair_id="p3", label=LabelType.DIFFERENT, labeler_id="u1"),
            ],
        )

        distribution = session.get_label_distribution()

        assert distribution["same"] == 2
        assert distribution["different"] == 1
        assert distribution["skip"] == 0


class TestInterRaterAgreement:
    """Tests for InterRaterAgreement."""

    def test_cohens_kappa_perfect_agreement(self) -> None:
        """Test Kappa with perfect agreement."""
        labels_a = [LabelType.SAME, LabelType.DIFFERENT, LabelType.SAME]
        labels_b = [LabelType.SAME, LabelType.DIFFERENT, LabelType.SAME]

        kappa = InterRaterAgreement.cohens_kappa(labels_a, labels_b)

        assert kappa == 1.0

    def test_cohens_kappa_no_agreement(self) -> None:
        """Test Kappa with complete disagreement (worse than chance)."""
        # When all labels are different, Kappa should be <= 0
        # With only 2 categories and systematic disagreement, Kappa is 0
        # because expected agreement equals observed agreement
        labels_a = [LabelType.SAME, LabelType.SAME, LabelType.SAME]
        labels_b = [LabelType.DIFFERENT, LabelType.DIFFERENT, LabelType.DIFFERENT]

        kappa = InterRaterAgreement.cohens_kappa(labels_a, labels_b)

        # With systematic disagreement and uniform distributions,
        # Kappa is 0 or negative depending on implementation
        assert kappa <= 0.01  # Allow small floating point variance

    def test_agreement_rate(self) -> None:
        """Test simple agreement rate."""
        labels_a = [LabelType.SAME, LabelType.DIFFERENT, LabelType.SAME]
        labels_b = [LabelType.SAME, LabelType.SAME, LabelType.SAME]

        rate = InterRaterAgreement.agreement_rate(labels_a, labels_b)

        assert rate == pytest.approx(2 / 3)

    def test_interpret_kappa(self) -> None:
        """Test Kappa interpretation."""
        assert "perfect" in InterRaterAgreement.interpret_kappa(0.85).lower()
        assert "substantial" in InterRaterAgreement.interpret_kappa(0.7).lower()
        assert "moderate" in InterRaterAgreement.interpret_kappa(0.5).lower()


class TestLabelingTool:
    """Tests for LabelingTool."""

    def test_create_session(self, tmp_path: Path) -> None:
        """Test creating a labeling session."""
        tool = LabelingTool(session_dir=tmp_path)

        pairs = [
            ClusterPair(id=f"pair-{i}", cluster_a_id=f"a-{i}", cluster_b_id=f"b-{i}")
            for i in range(5)
        ]

        session = tool.create_session(
            labeler_id="user-1",
            pairs=pairs,
            shuffle=False,
        )

        assert session.labeler_id == "user-1"
        assert session.n_total == 5
        assert len(session.id) == 8

    def test_record_label(self, tmp_path: Path) -> None:
        """Test recording a label."""
        tool = LabelingTool(session_dir=tmp_path)

        pairs = [ClusterPair(id="pair-1", cluster_a_id="a-1", cluster_b_id="b-1")]

        session = tool.create_session(labeler_id="user-1", pairs=pairs)

        result = tool.record_label(
            session=session,
            label=LabelType.SAME,
            confidence=0.9,
            notes="Clear match",
        )

        assert result.label == LabelType.SAME
        assert result.confidence == 0.9
        assert session.n_labeled == 1

    def test_save_and_load_session(self, tmp_path: Path) -> None:
        """Test saving and loading session via tool."""
        tool = LabelingTool(session_dir=tmp_path)

        pairs = [ClusterPair(id="pair-1", cluster_a_id="a-1", cluster_b_id="b-1")]

        session = tool.create_session(labeler_id="user-1", pairs=pairs)
        tool.record_label(session, LabelType.SAME)

        saved_path = tool.save_session(session)
        assert saved_path.exists()

        loaded = tool.load_session(session.id)
        assert loaded is not None
        assert loaded.n_labeled == 1

    def test_export_labels(self, tmp_path: Path) -> None:
        """Test exporting labels."""
        tool = LabelingTool(session_dir=tmp_path)

        pairs = [ClusterPair(id="pair-1", cluster_a_id="a-1", cluster_b_id="b-1")]

        session = tool.create_session(labeler_id="user-1", pairs=pairs)
        tool.record_label(session, LabelType.SAME)

        export_path = tmp_path / "labels.json"
        count = tool.export_labels([session], export_path)

        assert count == 1
        assert export_path.exists()

    def test_compute_agreement(self, tmp_path: Path) -> None:
        """Test computing inter-rater agreement."""
        tool = LabelingTool(session_dir=tmp_path)

        pairs = [
            ClusterPair(id=f"pair-{i}", cluster_a_id=f"a-{i}", cluster_b_id=f"b-{i}")
            for i in range(3)
        ]

        session_a = tool.create_session(labeler_id="user-a", pairs=pairs)
        session_b = tool.create_session(labeler_id="user-b", pairs=pairs)

        # Same labels for both sessions - label all pairs for session_a first
        labels = [LabelType.SAME, LabelType.DIFFERENT, LabelType.SAME]
        for label in labels:
            tool.record_label(session_a, label)

        # Then label all pairs for session_b with same labels
        for label in labels:
            tool.record_label(session_b, label)

        agreement = tool.compute_agreement(session_a, session_b)

        assert agreement["kappa"] == 1.0
        assert agreement["agreement_rate"] == 1.0

    def test_generate_pairs_from_clusters(self, tmp_path: Path) -> None:
        """Test generating cluster pairs for labeling."""
        tool = LabelingTool(session_dir=tmp_path)

        cluster_samples = {
            "cluster-0": ["msg 0-1", "msg 0-2", "msg 0-3"],
            "cluster-1": ["msg 1-1", "msg 1-2", "msg 1-3"],
            "cluster-2": ["msg 2-1", "msg 2-2", "msg 2-3"],
        }

        pairs = tool.generate_pairs_from_clusters(cluster_samples, n_pairs=5)

        assert len(pairs) <= 5
        for pair in pairs:
            assert pair.cluster_a_id != pair.cluster_b_id
            assert len(pair.cluster_a_samples) > 0
