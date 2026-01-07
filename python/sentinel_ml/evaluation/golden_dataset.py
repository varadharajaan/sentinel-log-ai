"""
Golden dataset management for regression testing.

This module provides infrastructure for creating, managing, and testing against
golden datasets to ensure clustering quality consistency over time.

Design Patterns:
- Repository Pattern: GoldenDatasetManager for CRUD operations
- Factory Pattern: Dataset creation with configuration
- Strategy Pattern: Pluggable comparison strategies
- Template Method: Regression testing workflow

SOLID Principles:
- Single Responsibility: Each class handles one concern
- Open/Closed: Extensible comparison strategies
- Liskov Substitution: All comparators implement same interface
- Interface Segregation: Minimal interfaces
- Dependency Inversion: Depends on abstractions
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from sentinel_ml.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from numpy.typing import NDArray

logger = get_logger(__name__)


class ComparisonStatus(str, Enum):
    """Status of a regression comparison."""

    PASSED = "passed"
    FAILED = "failed"
    DEGRADED = "degraded"  # Metrics worse but within tolerance
    IMPROVED = "improved"
    SKIPPED = "skipped"


@dataclass
class GoldenRecord:
    """
    A single record in a golden dataset.

    Attributes:
        id: Unique identifier for this record.
        message: The log message.
        normalized: Normalized/masked version of the message.
        expected_cluster_id: Expected cluster assignment.
        source: Source identifier.
        level: Log level.
        attrs: Additional attributes.
    """

    id: str
    message: str
    normalized: str
    expected_cluster_id: str
    source: str = ""
    level: str = ""
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "message": self.message,
            "normalized": self.normalized,
            "expected_cluster_id": self.expected_cluster_id,
            "source": self.source,
            "level": self.level,
            "attrs": self.attrs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoldenRecord:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            message=data["message"],
            normalized=data["normalized"],
            expected_cluster_id=data["expected_cluster_id"],
            source=data.get("source", ""),
            level=data.get("level", ""),
            attrs=data.get("attrs", {}),
        )


@dataclass
class ExpectedCluster:
    """
    Expected cluster definition in a golden dataset.

    Attributes:
        id: Unique cluster identifier.
        name: Human-readable cluster name.
        description: Description of what logs this cluster contains.
        representative_message: Most representative log message.
        expected_size: Expected number of records in this cluster.
        keywords: Keywords that characterize this cluster.
        min_similarity: Minimum expected cohesion score.
    """

    id: str
    name: str
    description: str = ""
    representative_message: str = ""
    expected_size: int = 0
    keywords: list[str] = field(default_factory=list)
    min_similarity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "representative_message": self.representative_message,
            "expected_size": self.expected_size,
            "keywords": self.keywords,
            "min_similarity": self.min_similarity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExpectedCluster:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            representative_message=data.get("representative_message", ""),
            expected_size=data.get("expected_size", 0),
            keywords=data.get("keywords", []),
            min_similarity=data.get("min_similarity", 0.0),
        )


@dataclass
class GoldenDataset:
    """
    A golden dataset for regression testing.

    Contains curated log samples with expected cluster assignments
    and quality thresholds for regression testing.

    Attributes:
        name: Dataset name.
        version: Dataset version string.
        description: Dataset description.
        records: List of golden records.
        expected_clusters: Expected cluster definitions.
        quality_thresholds: Expected minimum quality metrics.
        created_at: When dataset was created.
        updated_at: When dataset was last updated.
        metadata: Additional metadata.
    """

    name: str
    version: str
    description: str = ""
    records: list[GoldenRecord] = field(default_factory=list)
    expected_clusters: list[ExpectedCluster] = field(default_factory=list)
    quality_thresholds: dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set default quality thresholds if not provided."""
        if not self.quality_thresholds:
            self.quality_thresholds = {
                "silhouette": 0.3,
                "davies_bouldin": 2.0,
                "adjusted_rand_index": 0.7,
            }

    @property
    def n_records(self) -> int:
        """Number of records in the dataset."""
        return len(self.records)

    @property
    def n_clusters(self) -> int:
        """Number of expected clusters."""
        return len(self.expected_clusters)

    @property
    def checksum(self) -> str:
        """Compute checksum for dataset integrity verification."""
        data = json.dumps(
            {
                "name": self.name,
                "version": self.version,
                "records": [r.to_dict() for r in self.records],
                "expected_clusters": [c.to_dict() for c in self.expected_clusters],
            },
            sort_keys=True,
        )
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def add_record(self, record: GoldenRecord) -> None:
        """Add a record to the dataset."""
        self.records.append(record)
        self.updated_at = datetime.now(timezone.utc)

    def add_cluster(self, cluster: ExpectedCluster) -> None:
        """Add an expected cluster definition."""
        self.expected_clusters.append(cluster)
        self.updated_at = datetime.now(timezone.utc)

    def get_expected_labels(self) -> dict[str, str]:
        """Get mapping of record ID to expected cluster ID."""
        return {r.id: r.expected_cluster_id for r in self.records}

    def get_cluster_by_id(self, cluster_id: str) -> ExpectedCluster | None:
        """Get an expected cluster by ID."""
        for cluster in self.expected_clusters:
            if cluster.id == cluster_id:
                return cluster
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "records": [r.to_dict() for r in self.records],
            "expected_clusters": [c.to_dict() for c in self.expected_clusters],
            "quality_thresholds": self.quality_thresholds,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoldenDataset:
        """Create from dictionary."""
        records = [GoldenRecord.from_dict(r) for r in data.get("records", [])]
        expected_clusters = [
            ExpectedCluster.from_dict(c) for c in data.get("expected_clusters", [])
        ]

        dataset = cls(
            name=data["name"],
            version=data["version"],
            description=data.get("description", ""),
            records=records,
            expected_clusters=expected_clusters,
            quality_thresholds=data.get("quality_thresholds", {}),
            metadata=data.get("metadata", {}),
        )

        if "created_at" in data:
            dataset.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            dataset.updated_at = datetime.fromisoformat(data["updated_at"])

        return dataset

    def save(self, path: Path) -> None:
        """
        Save dataset to JSON file.

        Args:
            path: Path to save to.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(
            "golden_dataset_saved",
            path=str(path),
            n_records=self.n_records,
            n_clusters=self.n_clusters,
        )

    @classmethod
    def load(cls, path: Path) -> GoldenDataset:
        """
        Load dataset from JSON file.

        Args:
            path: Path to load from.

        Returns:
            Loaded GoldenDataset.
        """
        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        dataset = cls.from_dict(data)

        logger.info(
            "golden_dataset_loaded",
            path=str(path),
            name=dataset.name,
            n_records=dataset.n_records,
        )

        return dataset


@dataclass
class RegressionResult:
    """
    Result of a regression test run.

    Attributes:
        dataset_name: Name of the golden dataset.
        dataset_version: Version of the golden dataset.
        status: Overall status of the regression test.
        quality_metrics: Computed quality metrics.
        cluster_metrics: Per-cluster accuracy metrics.
        adjusted_rand_index: ARI score comparing to expected labels.
        normalized_mutual_info: NMI score comparing to expected labels.
        misclassified_records: Records assigned to wrong clusters.
        new_clusters: Clusters not in expected set.
        missing_clusters: Expected clusters not found.
        execution_time_seconds: Time taken to run regression.
        timestamp: When test was run.
        details: Additional result details.
    """

    dataset_name: str
    dataset_version: str
    status: ComparisonStatus = ComparisonStatus.PASSED
    quality_metrics: dict[str, float] = field(default_factory=dict)
    cluster_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    adjusted_rand_index: float = 0.0
    normalized_mutual_info: float = 0.0
    misclassified_records: list[str] = field(default_factory=list)
    new_clusters: list[str] = field(default_factory=list)
    missing_clusters: list[str] = field(default_factory=list)
    execution_time_seconds: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        """Compute accuracy as percentage of correctly classified records."""
        total = self.details.get("n_records", 0)
        misclassified = len(self.misclassified_records)
        if total == 0:
            return 0.0
        return float((total - misclassified) / total)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "status": self.status.value,
            "quality_metrics": self.quality_metrics,
            "cluster_metrics": self.cluster_metrics,
            "adjusted_rand_index": round(self.adjusted_rand_index, 4),
            "normalized_mutual_info": round(self.normalized_mutual_info, 4),
            "accuracy": round(self.accuracy, 4),
            "misclassified_count": len(self.misclassified_records),
            "new_clusters_count": len(self.new_clusters),
            "missing_clusters_count": len(self.missing_clusters),
            "execution_time_seconds": round(self.execution_time_seconds, 4),
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


class RegressionRunner:
    """
    Runner for regression tests against golden datasets.

    Coordinates the regression testing workflow including:
    - Loading golden dataset
    - Running clustering on dataset records
    - Comparing results to expected clusters
    - Computing quality and accuracy metrics
    - Generating regression result

    Usage:
        runner = RegressionRunner()
        result = await runner.run(dataset, clustering_service)
    """

    def __init__(
        self,
        tolerance: float = 0.1,
    ) -> None:
        """
        Initialize regression runner.

        Args:
            tolerance: Tolerance for metric degradation (0-1).
        """
        self._tolerance = tolerance

        logger.info(
            "regression_runner_initialized",
            tolerance=tolerance,
        )

    def run(
        self,
        dataset: GoldenDataset,
        actual_labels: NDArray[np.int32],
        embeddings: NDArray[np.float32] | None = None,
    ) -> RegressionResult:
        """
        Run regression test against a golden dataset.

        Args:
            dataset: Golden dataset with expected cluster assignments.
            actual_labels: Actual cluster labels from clustering.
            embeddings: Optional embeddings for quality metric computation.

        Returns:
            RegressionResult with comparison details.
        """
        start_time = time.perf_counter()

        logger.info(
            "regression_test_started",
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            n_records=dataset.n_records,
        )

        # Build expected labels array
        expected_labels = self._build_expected_labels(dataset)

        # Compute clustering comparison metrics
        ari, nmi = self._compute_clustering_metrics(expected_labels, actual_labels)

        # Identify misclassified records
        misclassified = self._find_misclassified(dataset, expected_labels, actual_labels)

        # Find new and missing clusters
        new_clusters, missing_clusters = self._compare_cluster_sets(dataset, actual_labels)

        # Compute quality metrics if embeddings provided
        quality_metrics: dict[str, float] = {}
        if embeddings is not None:
            quality_metrics = self._compute_quality_metrics(embeddings, actual_labels)

        # Determine overall status
        status = self._determine_status(dataset, ari, quality_metrics, len(misclassified))

        execution_time = time.perf_counter() - start_time

        result = RegressionResult(
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            status=status,
            quality_metrics=quality_metrics,
            adjusted_rand_index=ari,
            normalized_mutual_info=nmi,
            misclassified_records=misclassified,
            new_clusters=new_clusters,
            missing_clusters=missing_clusters,
            execution_time_seconds=execution_time,
            details={
                "n_records": dataset.n_records,
                "n_expected_clusters": dataset.n_clusters,
                "n_actual_clusters": len(set(actual_labels) - {-1}),
                "tolerance": self._tolerance,
            },
        )

        logger.info(
            "regression_test_completed",
            status=status.value,
            ari=round(ari, 4),
            accuracy=round(result.accuracy, 4),
            execution_time=round(execution_time, 4),
        )

        return result

    def _build_expected_labels(
        self,
        dataset: GoldenDataset,
    ) -> NDArray[np.int32]:
        """Build expected labels array from dataset."""
        # Map cluster IDs to numeric labels
        cluster_id_to_label = {cluster.id: i for i, cluster in enumerate(dataset.expected_clusters)}

        labels = []
        for record in dataset.records:
            label = cluster_id_to_label.get(record.expected_cluster_id, -1)
            labels.append(label)

        return np.array(labels, dtype=np.int32)

    def _compute_clustering_metrics(
        self,
        expected: NDArray[np.int32],
        actual: NDArray[np.int32],
    ) -> tuple[float, float]:
        """Compute ARI and NMI between expected and actual labels."""
        # Filter out noise from both arrays for fair comparison
        mask = (expected >= 0) & (actual >= 0)
        if mask.sum() < 2:
            return 0.0, 0.0

        expected_filtered = expected[mask]
        actual_filtered = actual[mask]

        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        ari = float(adjusted_rand_score(expected_filtered, actual_filtered))
        nmi = float(normalized_mutual_info_score(expected_filtered, actual_filtered))

        return ari, nmi

    def _find_misclassified(
        self,
        dataset: GoldenDataset,
        expected: NDArray[np.int32],
        actual: NDArray[np.int32],
    ) -> list[str]:
        """Find records that were assigned to wrong clusters."""
        misclassified = []

        for i, record in enumerate(dataset.records):
            if expected[i] != actual[i] and expected[i] >= 0:
                misclassified.append(record.id)

        return misclassified

    def _compare_cluster_sets(
        self,
        dataset: GoldenDataset,
        actual_labels: NDArray[np.int32],
    ) -> tuple[list[str], list[str]]:
        """Compare expected and actual cluster sets."""
        expected_ids = {c.id for c in dataset.expected_clusters}
        actual_ids = {str(label) for label in set(actual_labels) if label >= 0}

        # Note: This is a simplified comparison since actual labels are numeric
        # In practice, you would need to map actual clusters to expected clusters
        new_clusters = list(actual_ids - expected_ids)
        missing_clusters = list(expected_ids - actual_ids)

        return new_clusters, missing_clusters

    def _compute_quality_metrics(
        self,
        embeddings: NDArray[np.float32],
        labels: NDArray[np.int32],
    ) -> dict[str, float]:
        """Compute quality metrics for actual clustering."""
        from sentinel_ml.evaluation.metrics import QualityEvaluator

        evaluator = QualityEvaluator()
        result = evaluator.evaluate(embeddings, labels)

        metrics = {}
        for metric in result.metrics:
            if not np.isnan(metric.value):
                metrics[metric.metric_type.value] = metric.value

        return metrics

    def _determine_status(
        self,
        dataset: GoldenDataset,
        ari: float,
        quality_metrics: dict[str, float],
        _n_misclassified: int,
    ) -> ComparisonStatus:
        """Determine overall regression test status."""
        ari_threshold = dataset.quality_thresholds.get("adjusted_rand_index", 0.7)

        # Check ARI threshold
        if ari < ari_threshold - self._tolerance:
            return ComparisonStatus.FAILED

        if ari < ari_threshold:
            return ComparisonStatus.DEGRADED

        # Check quality metric thresholds
        for metric_name, threshold in dataset.quality_thresholds.items():
            if metric_name in quality_metrics:
                actual = quality_metrics[metric_name]
                # Handle both higher-is-better and lower-is-better metrics
                if metric_name == "davies_bouldin":
                    if actual > threshold + self._tolerance:
                        return ComparisonStatus.DEGRADED
                elif actual < threshold - self._tolerance:
                    return ComparisonStatus.DEGRADED

        # Check if metrics improved
        if ari > ari_threshold + self._tolerance:
            return ComparisonStatus.IMPROVED

        return ComparisonStatus.PASSED


class GoldenDatasetManager:
    """
    Manager for golden datasets (Repository pattern).

    Handles CRUD operations for golden datasets stored on disk.

    Usage:
        manager = GoldenDatasetManager(Path("./golden"))
        manager.save(dataset)
        loaded = manager.load("my_dataset")
        all_datasets = list(manager.list_all())
    """

    def __init__(self, base_path: Path) -> None:
        """
        Initialize the manager.

        Args:
            base_path: Base directory for storing golden datasets.
        """
        self._base_path = base_path
        self._base_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "golden_dataset_manager_initialized",
            base_path=str(base_path),
        )

    def save(self, dataset: GoldenDataset) -> Path:
        """
        Save a golden dataset.

        Args:
            dataset: Dataset to save.

        Returns:
            Path where dataset was saved.
        """
        filename = f"{dataset.name}_v{dataset.version}.json"
        path = self._base_path / filename
        dataset.save(path)
        return path

    def load(self, name: str, version: str | None = None) -> GoldenDataset:
        """
        Load a golden dataset.

        Args:
            name: Dataset name.
            version: Optional version. If not provided, loads latest.

        Returns:
            Loaded GoldenDataset.

        Raises:
            FileNotFoundError: If dataset not found.
        """
        if version:
            filename = f"{name}_v{version}.json"
            path = self._base_path / filename
        else:
            # Find latest version
            path = self._find_latest_version(name)

        if not path.exists():
            msg = f"Golden dataset not found: {name}"
            raise FileNotFoundError(msg)

        return GoldenDataset.load(path)

    def _find_latest_version(self, name: str) -> Path:
        """Find the latest version of a dataset."""
        pattern = f"{name}_v*.json"
        matches = sorted(self._base_path.glob(pattern))

        if not matches:
            return self._base_path / f"{name}.json"

        return matches[-1]

    def delete(self, name: str, version: str | None = None) -> bool:
        """
        Delete a golden dataset.

        Args:
            name: Dataset name.
            version: Optional version. If not provided, deletes all versions.

        Returns:
            True if any files were deleted.
        """
        if version:
            filename = f"{name}_v{version}.json"
            path = self._base_path / filename
            if path.exists():
                path.unlink()
                logger.info("golden_dataset_deleted", path=str(path))
                return True
            return False

        # Delete all versions
        pattern = f"{name}_v*.json"
        deleted = False
        for path in self._base_path.glob(pattern):
            path.unlink()
            logger.info("golden_dataset_deleted", path=str(path))
            deleted = True

        return deleted

    def list_all(self) -> Iterator[tuple[str, str, Path]]:
        """
        List all golden datasets.

        Yields:
            Tuples of (name, version, path).
        """
        for path in sorted(self._base_path.glob("*.json")):
            try:
                # Parse name and version from filename
                stem = path.stem
                if "_v" in stem:
                    name, version = stem.rsplit("_v", 1)
                else:
                    name = stem
                    version = "1.0"
                yield name, version, path

            except Exception as e:
                logger.warning(
                    "golden_dataset_parse_error",
                    path=str(path),
                    error=str(e),
                )

    def exists(self, name: str, version: str | None = None) -> bool:
        """
        Check if a golden dataset exists.

        Args:
            name: Dataset name.
            version: Optional version.

        Returns:
            True if dataset exists.
        """
        if version:
            filename = f"{name}_v{version}.json"
            return (self._base_path / filename).exists()

        pattern = f"{name}_v*.json"
        return bool(list(self._base_path.glob(pattern)))

    @property
    def base_path(self) -> Path:
        """Return the base path for datasets."""
        return self._base_path
