"""
Integration tests for the evaluation module.

Tests end-to-end workflows including:
- Quality evaluation with real clustering
- Golden dataset regression testing
- Report generation and export
- Ablation study execution
- Multi-rater labeling workflow
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from sentinel_ml.evaluation import (
    AblationConfig,
    AblationRunner,
    ComponentConfig,
    ComponentType,
    EvaluationReportConfig,
    EvaluationReportGenerator,
    ExpectedCluster,
    GoldenDataset,
    GoldenDatasetManager,
    GoldenRecord,
    LabelingTool,
    LabelType,
    QualityEvaluator,
    RegressionRunner,
    ReportFormat,
)

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def realistic_embeddings() -> np.ndarray:
    """Create realistic embeddings simulating log patterns."""
    np.random.seed(42)

    # 4 patterns with 50 samples each, 64-dimensional embeddings
    samples_per_pattern = 50
    embedding_dim = 64

    embeddings = []

    # Pattern 1: HTTP 200 responses
    base1 = np.random.randn(embedding_dim).astype(np.float32) * 0.1
    for _ in range(samples_per_pattern):
        embedding = base1 + np.random.randn(embedding_dim).astype(np.float32) * 0.05
        embeddings.append(embedding)

    # Pattern 2: Database connection errors
    base2 = np.random.randn(embedding_dim).astype(np.float32) * 0.1 + 2.0
    for _ in range(samples_per_pattern):
        embedding = base2 + np.random.randn(embedding_dim).astype(np.float32) * 0.05
        embeddings.append(embedding)

    # Pattern 3: Authentication failures
    base3 = np.random.randn(embedding_dim).astype(np.float32) * 0.1 - 2.0
    for _ in range(samples_per_pattern):
        embedding = base3 + np.random.randn(embedding_dim).astype(np.float32) * 0.05
        embeddings.append(embedding)

    # Pattern 4: System health checks
    base4 = np.random.randn(embedding_dim).astype(np.float32) * 0.1
    base4[0] = 3.0
    for _ in range(samples_per_pattern):
        embedding = base4 + np.random.randn(embedding_dim).astype(np.float32) * 0.05
        embeddings.append(embedding)

    return np.array(embeddings, dtype=np.float32)


@pytest.fixture
def expected_labels() -> np.ndarray:
    """Expected cluster labels for realistic embeddings."""
    return np.array(
        [0] * 50 + [1] * 50 + [2] * 50 + [3] * 50,
        dtype=np.int32,
    )


@pytest.fixture
def golden_dataset_with_embeddings(
    realistic_embeddings: np.ndarray,
) -> tuple[GoldenDataset, np.ndarray]:
    """Create a golden dataset with corresponding embeddings."""
    records = []
    patterns = [
        ("HTTP 200 OK", "http_success"),
        ("Database connection failed", "db_error"),
        ("Authentication failed for user", "auth_failure"),
        ("Health check passed", "health_ok"),
    ]

    for pattern_idx, (pattern_msg, cluster_id) in enumerate(patterns):
        for i in range(50):
            records.append(
                GoldenRecord(
                    id=f"rec-{pattern_idx}-{i}",
                    message=f"{pattern_msg} {i}",
                    normalized=f"{pattern_msg} <num>",
                    expected_cluster_id=cluster_id,
                    source="app.log",
                    level="INFO" if pattern_idx in (0, 3) else "ERROR",
                )
            )

    clusters = [
        ExpectedCluster(
            id="http_success",
            name="HTTP Success",
            description="Successful HTTP responses",
            expected_size=50,
        ),
        ExpectedCluster(
            id="db_error",
            name="Database Errors",
            description="Database connection failures",
            expected_size=50,
        ),
        ExpectedCluster(
            id="auth_failure",
            name="Auth Failures",
            description="Authentication failures",
            expected_size=50,
        ),
        ExpectedCluster(
            id="health_ok",
            name="Health Checks",
            description="System health checks",
            expected_size=50,
        ),
    ]

    dataset = GoldenDataset(
        name="integration_test_dataset",
        version="1.0",
        description="Dataset for integration testing",
        records=records,
        expected_clusters=clusters,
        quality_thresholds={
            "silhouette": 0.3,
            "davies_bouldin": 2.0,
            "adjusted_rand_index": 0.7,
        },
    )

    return dataset, realistic_embeddings


# ============================================================================
# Integration Tests
# ============================================================================


class TestEvaluationWorkflow:
    """Test complete evaluation workflow."""

    def test_quality_evaluation_with_mock_clustering(
        self,
        realistic_embeddings: np.ndarray,
        expected_labels: np.ndarray,
    ) -> None:
        """Test quality evaluation with well-defined clusters."""
        evaluator = QualityEvaluator()
        result = evaluator.evaluate(realistic_embeddings, expected_labels)

        # With well-separated clusters, we expect good quality
        assert result.overall_quality >= 0.5
        assert result.n_clusters == 4
        assert result.n_samples == 200

        # Check individual metrics are computed
        silhouette = result.get_metric("silhouette")
        assert silhouette is not None
        assert silhouette.value > 0.3  # Should be decent separation

    def test_full_evaluation_pipeline(
        self,
        realistic_embeddings: np.ndarray,
        expected_labels: np.ndarray,
        tmp_path: Path,
    ) -> None:
        """Test complete evaluation pipeline from metrics to report."""
        # Step 1: Evaluate quality
        evaluator = QualityEvaluator()
        quality_result = evaluator.evaluate(realistic_embeddings, expected_labels)

        # Step 2: Generate report
        config = EvaluationReportConfig(
            title="Integration Test Report",
            description="Testing full evaluation pipeline",
            output_format=ReportFormat.MARKDOWN,
        )
        generator = EvaluationReportGenerator(config)
        report = generator.generate(quality_result)

        # Step 3: Save report
        report_path = tmp_path / "eval_report.md"
        saved_path = generator.save(report, report_path)

        # Verify
        assert saved_path.exists()
        content = saved_path.read_text()
        assert "Integration Test Report" in content
        assert "Clustering Quality Metrics" in content


class TestGoldenDatasetWorkflow:
    """Test golden dataset management and regression testing."""

    def test_dataset_lifecycle(self, tmp_path: Path) -> None:
        """Test creating, saving, loading, and deleting datasets."""
        manager = GoldenDatasetManager(tmp_path / "golden")

        # Create dataset
        records = [
            GoldenRecord(
                id=f"rec-{i}",
                message=f"Test message {i}",
                normalized="Test message <num>",
                expected_cluster_id=f"cluster-{i % 3}",
            )
            for i in range(30)
        ]

        dataset = GoldenDataset(
            name="lifecycle_test",
            version="1.0",
            records=records,
        )

        # Save
        path = manager.save(dataset)
        assert path.exists()

        # List
        datasets = list(manager.list_all())
        assert len(datasets) == 1
        assert datasets[0][0] == "lifecycle_test"

        # Load
        loaded = manager.load("lifecycle_test", "1.0")
        assert loaded.n_records == 30

        # Update and save new version
        loaded.version = "1.1"
        loaded.add_record(
            GoldenRecord(
                id="rec-new",
                message="New message",
                normalized="New message",
                expected_cluster_id="cluster-0",
            )
        )
        manager.save(loaded)

        # Verify both versions exist
        datasets = list(manager.list_all())
        assert len(datasets) == 2

        # Delete specific version
        manager.delete("lifecycle_test", "1.0")
        assert not manager.exists("lifecycle_test", "1.0")
        assert manager.exists("lifecycle_test", "1.1")

    def test_regression_testing(
        self,
        golden_dataset_with_embeddings: tuple[GoldenDataset, np.ndarray],
    ) -> None:
        """Test regression testing against golden dataset."""
        dataset, embeddings = golden_dataset_with_embeddings

        # Use exact expected labels (perfect match)
        actual_labels = np.array(
            [i // 50 for i in range(200)],
            dtype=np.int32,
        )

        runner = RegressionRunner(tolerance=0.1)
        result = runner.run(dataset, actual_labels, embeddings)

        assert result.dataset_name == dataset.name
        # With exact match, should pass
        assert result.accuracy >= 0.9


class TestAblationWorkflow:
    """Test ablation study workflow."""

    def test_parameter_sweep_ablation(
        self,
        realistic_embeddings: np.ndarray,
    ) -> None:
        """Test ablation study with parameter sweep."""
        baseline = ComponentConfig(
            name="min_size_5",
            component_type=ComponentType.MIN_CLUSTER_SIZE,
            value=5,
            description="Baseline with min_cluster_size=5",
        )

        test_configs = [
            ComponentConfig(
                name="min_size_10",
                component_type=ComponentType.MIN_CLUSTER_SIZE,
                value=10,
                description="Test with min_cluster_size=10",
            ),
            ComponentConfig(
                name="min_size_3",
                component_type=ComponentType.MIN_CLUSTER_SIZE,
                value=3,
                description="Test with min_cluster_size=3",
            ),
        ]

        config = AblationConfig(
            name="min_cluster_size_sweep",
            baseline_config=baseline,
            test_configs=test_configs,
            n_runs=2,
        )

        runner = AblationRunner()

        # Mock clustering function
        def cluster_func(cfg: ComponentConfig, embeddings: np.ndarray) -> np.ndarray:
            # Simple mock: divide embeddings into 4 clusters
            n = len(embeddings)
            return np.array([i // (n // 4) for i in range(n)], dtype=np.int32)

        study = runner.run(config, cluster_func, realistic_embeddings)

        assert study.n_successful == 6  # 3 configs x 2 runs
        assert study.n_failed == 0
        assert len(study.comparisons) == 2  # 2 test configs vs baseline
        assert study.get_best_config() is not None


class TestLabelingWorkflow:
    """Test human labeling workflow."""

    def test_multi_rater_labeling_session(self, tmp_path: Path) -> None:
        """Test multi-rater labeling with agreement calculation."""
        tool = LabelingTool(session_dir=tmp_path)

        # Generate cluster pairs
        # With 3 clusters, we can have at most 3 unique pairs (3 choose 2)
        cluster_samples = {
            "cluster-0": ["HTTP 200 OK", "Request succeeded", "Response 200"],
            "cluster-1": ["Database error", "Connection failed", "DB timeout"],
            "cluster-2": ["Auth failed", "Login denied", "Unauthorized"],
        }

        pairs = tool.generate_pairs_from_clusters(cluster_samples, n_pairs=3)
        assert len(pairs) == 3  # 3 clusters = 3 unique pairs

        # Rater 1
        session_1 = tool.create_session(
            labeler_id="rater-1",
            pairs=pairs,
            shuffle=False,
        )

        # Rater 2
        session_2 = tool.create_session(
            labeler_id="rater-2",
            pairs=pairs,
            shuffle=False,
        )

        # Both raters label all 3 pairs with same labels
        labels = [LabelType.DIFFERENT, LabelType.DIFFERENT, LabelType.DIFFERENT]

        # Rater 1 labels all pairs first
        for label in labels:
            tool.record_label(session_1, label)

        # Rater 2 labels all pairs with same labels
        for label in labels:
            tool.record_label(session_2, label)

        # Calculate agreement
        agreement = tool.compute_agreement(session_1, session_2)

        assert agreement["kappa"] == 1.0  # Perfect agreement
        assert agreement["n_common_pairs"] == 3

        # Export labels
        export_path = tmp_path / "all_labels.json"
        count = tool.export_labels([session_1, session_2], export_path)

        assert count == 6  # 3 labels x 2 raters
        assert export_path.exists()


class TestReportGeneration:
    """Test report generation with various formats."""

    def test_generate_all_formats(
        self,
        realistic_embeddings: np.ndarray,
        expected_labels: np.ndarray,
        tmp_path: Path,
    ) -> None:
        """Test generating reports in all supported formats."""
        evaluator = QualityEvaluator()
        quality_result = evaluator.evaluate(realistic_embeddings, expected_labels)

        formats = [
            (ReportFormat.JSON, ".json"),
            (ReportFormat.MARKDOWN, ".md"),
            (ReportFormat.HTML, ".html"),
        ]

        for fmt, ext in formats:
            config = EvaluationReportConfig(
                title=f"Report in {fmt.value} format",
                output_format=fmt,
            )
            generator = EvaluationReportGenerator(config)
            report = generator.generate(quality_result)

            output_path = tmp_path / f"report{ext}"
            saved_path = generator.save(report, output_path)

            assert saved_path.exists()
            content = saved_path.read_text()
            assert len(content) > 0

    def test_trend_analysis_with_history(
        self,
        realistic_embeddings: np.ndarray,
        expected_labels: np.ndarray,
    ) -> None:
        """Test trend analysis with historical data."""
        evaluator = QualityEvaluator()

        # Generate multiple evaluation results to simulate history
        history = []
        for _ in range(5):
            result = evaluator.evaluate(realistic_embeddings, expected_labels)
            history.append(result)

        # Generate report with history
        config = EvaluationReportConfig(include_trends=True)
        generator = EvaluationReportGenerator(config)

        current_result = evaluator.evaluate(realistic_embeddings, expected_labels)
        report = generator.generate(current_result, history=history)

        # With consistent results, trends should be stable
        assert len(report.trends) > 0
        for trend in report.trends:
            assert trend.samples_count > 0


class TestEndToEndScenario:
    """Test complete end-to-end scenarios."""

    def test_complete_evaluation_cycle(
        self,
        golden_dataset_with_embeddings: tuple[GoldenDataset, np.ndarray],
        tmp_path: Path,
    ) -> None:
        """Test a complete evaluation cycle from clustering to report."""
        dataset, embeddings = golden_dataset_with_embeddings

        # 1. Evaluate clustering quality
        actual_labels = np.array([i // 50 for i in range(200)], dtype=np.int32)

        evaluator = QualityEvaluator()
        quality_result = evaluator.evaluate(embeddings, actual_labels)

        assert quality_result.overall_quality > 0.0

        # 2. Run regression test
        regression_runner = RegressionRunner()
        regression_result = regression_runner.run(dataset, actual_labels, embeddings)

        # 3. Generate comprehensive report
        config = EvaluationReportConfig(
            title="Complete Evaluation Cycle Report",
            include_trends=True,
            include_recommendations=True,
            output_format=ReportFormat.MARKDOWN,
        )

        generator = EvaluationReportGenerator(config)
        report = generator.generate(
            quality_result=quality_result,
            regression_result=regression_result,
        )

        # 4. Save report
        report_path = tmp_path / "complete_report.md"
        generator.save(report, report_path)

        # Verify report content
        content = report_path.read_text()
        assert "Complete Evaluation Cycle Report" in content
        assert "Clustering Quality Metrics" in content
        assert "Regression Test Results" in content

        # 5. Save golden dataset for future use
        manager = GoldenDatasetManager(tmp_path / "golden")
        manager.save(dataset)

        # Verify dataset was saved
        assert manager.exists(dataset.name)
