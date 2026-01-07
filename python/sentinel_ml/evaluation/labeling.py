"""
Human labeling tool for cluster quality assessment.

This module provides a CLI-based tool for human evaluation of clustering
quality, including cluster pair comparison and inter-rater agreement tracking.

Use cases:
- Mark clusters as same/different
- Rate cluster quality
- Track inter-rater agreement (Cohen's Kappa)
- Export labels for training/validation

Design Patterns:
- Command Pattern: Labeling actions as commands
- Observer Pattern: Label event tracking
- Repository Pattern: Label storage and retrieval
- State Pattern: Labeling session state management

SOLID Principles:
- Single Responsibility: Each class handles one concern
- Open/Closed: Extensible via new label types
- Liskov Substitution: All label types implement same interface
- Interface Segregation: Minimal interfaces
- Dependency Inversion: Depends on abstractions
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sentinel_ml.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = get_logger(__name__)


class LabelType(str, Enum):
    """Types of labels for cluster pairs."""

    SAME = "same"  # Clusters contain same pattern
    DIFFERENT = "different"  # Clusters contain different patterns
    SIMILAR = "similar"  # Clusters are related but distinct
    UNSURE = "unsure"  # Labeler is unsure
    SKIP = "skip"  # Pair was skipped


class QualityRating(str, Enum):
    """Quality ratings for individual clusters."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    VERY_POOR = "very_poor"


@dataclass
class ClusterPair:
    """
    A pair of clusters for comparison.

    Attributes:
        id: Unique identifier for this pair.
        cluster_a_id: ID of first cluster.
        cluster_b_id: ID of second cluster.
        cluster_a_samples: Sample messages from cluster A.
        cluster_b_samples: Sample messages from cluster B.
        expected_label: Optional expected label for validation.
        metadata: Additional metadata.
    """

    id: str
    cluster_a_id: str
    cluster_b_id: str
    cluster_a_samples: list[str] = field(default_factory=list)
    cluster_b_samples: list[str] = field(default_factory=list)
    expected_label: LabelType | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "cluster_a_id": self.cluster_a_id,
            "cluster_b_id": self.cluster_b_id,
            "cluster_a_samples": self.cluster_a_samples,
            "cluster_b_samples": self.cluster_b_samples,
            "expected_label": self.expected_label.value if self.expected_label else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClusterPair:
        """Create from dictionary."""
        expected = None
        if data.get("expected_label"):
            expected = LabelType(data["expected_label"])

        return cls(
            id=data["id"],
            cluster_a_id=data["cluster_a_id"],
            cluster_b_id=data["cluster_b_id"],
            cluster_a_samples=data.get("cluster_a_samples", []),
            cluster_b_samples=data.get("cluster_b_samples", []),
            expected_label=expected,
            metadata=data.get("metadata", {}),
        )


@dataclass
class LabelingResult:
    """
    Result of a labeling decision.

    Attributes:
        pair_id: ID of the cluster pair.
        label: Assigned label.
        labeler_id: ID of the labeler.
        confidence: Labeler's confidence (0-1).
        time_taken_seconds: Time taken to make decision.
        notes: Optional notes from labeler.
        timestamp: When label was assigned.
    """

    pair_id: str
    label: LabelType
    labeler_id: str
    confidence: float = 1.0
    time_taken_seconds: float = 0.0
    notes: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pair_id": self.pair_id,
            "label": self.label.value,
            "labeler_id": self.labeler_id,
            "confidence": round(self.confidence, 2),
            "time_taken_seconds": round(self.time_taken_seconds, 2),
            "notes": self.notes,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LabelingResult:
        """Create from dictionary."""
        return cls(
            pair_id=data["pair_id"],
            label=LabelType(data["label"]),
            labeler_id=data["labeler_id"],
            confidence=data.get("confidence", 1.0),
            time_taken_seconds=data.get("time_taken_seconds", 0.0),
            notes=data.get("notes", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


@dataclass
class LabelingSession:
    """
    A labeling session with progress tracking.

    Attributes:
        id: Session identifier.
        labeler_id: ID of the labeler.
        pairs: Cluster pairs to label.
        results: Labeling results.
        start_time: When session started.
        end_time: When session ended.
        current_index: Current position in pairs.
        settings: Session settings.
    """

    id: str
    labeler_id: str
    pairs: list[ClusterPair] = field(default_factory=list)
    results: list[LabelingResult] = field(default_factory=list)
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime | None = None
    current_index: int = 0
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def n_total(self) -> int:
        """Total number of pairs."""
        return len(self.pairs)

    @property
    def n_labeled(self) -> int:
        """Number of labeled pairs."""
        return len(self.results)

    @property
    def n_remaining(self) -> int:
        """Number of remaining pairs."""
        return self.n_total - self.n_labeled

    @property
    def progress_percent(self) -> float:
        """Progress percentage."""
        if self.n_total == 0:
            return 0.0
        return (self.n_labeled / self.n_total) * 100

    @property
    def is_complete(self) -> bool:
        """Whether all pairs have been labeled."""
        return self.n_labeled >= self.n_total

    @property
    def current_pair(self) -> ClusterPair | None:
        """Get current pair to label."""
        if self.current_index < len(self.pairs):
            return self.pairs[self.current_index]
        return None

    def add_result(self, result: LabelingResult) -> None:
        """Add a labeling result."""
        self.results.append(result)
        self.current_index = min(self.current_index + 1, len(self.pairs))

    def skip_current(self) -> None:
        """Skip the current pair."""
        if self.current_pair:
            result = LabelingResult(
                pair_id=self.current_pair.id,
                label=LabelType.SKIP,
                labeler_id=self.labeler_id,
            )
            self.add_result(result)

    def go_back(self) -> bool:
        """Go back to previous pair."""
        if self.current_index > 0:
            self.current_index -= 1
            # Remove the last result if it was for the current pair
            if self.results and self.results[-1].pair_id == self.pairs[self.current_index].id:
                self.results.pop()
            return True
        return False

    def get_label_distribution(self) -> dict[str, int]:
        """Get distribution of assigned labels."""
        distribution: dict[str, int] = {lt.value: 0 for lt in LabelType}
        for result in self.results:
            distribution[result.label.value] += 1
        return distribution

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "labeler_id": self.labeler_id,
            "pairs": [p.to_dict() for p in self.pairs],
            "results": [r.to_dict() for r in self.results],
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "current_index": self.current_index,
            "settings": self.settings,
            "n_total": self.n_total,
            "n_labeled": self.n_labeled,
            "progress_percent": round(self.progress_percent, 1),
        }

    def save(self, path: Path) -> None:
        """Save session to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(
            "labeling_session_saved",
            path=str(path),
            n_labeled=self.n_labeled,
        )

    @classmethod
    def load(cls, path: Path) -> LabelingSession:
        """Load session from file."""
        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        session = cls(
            id=data["id"],
            labeler_id=data["labeler_id"],
            pairs=[ClusterPair.from_dict(p) for p in data.get("pairs", [])],
            results=[LabelingResult.from_dict(r) for r in data.get("results", [])],
            current_index=data.get("current_index", 0),
            settings=data.get("settings", {}),
        )

        if data.get("start_time"):
            session.start_time = datetime.fromisoformat(data["start_time"])
        if data.get("end_time"):
            session.end_time = datetime.fromisoformat(data["end_time"])

        return session


class InterRaterAgreement:
    """
    Calculator for inter-rater agreement metrics.

    Computes Cohen's Kappa and other agreement metrics
    between multiple labelers.
    """

    @staticmethod
    def cohens_kappa(
        labels_a: list[LabelType],
        labels_b: list[LabelType],
    ) -> float:
        """
        Compute Cohen's Kappa coefficient.

        Args:
            labels_a: Labels from first rater.
            labels_b: Labels from second rater.

        Returns:
            Kappa coefficient (-1 to 1). 1 = perfect agreement,
            0 = agreement by chance, <0 = less than chance.
        """
        if len(labels_a) != len(labels_b):
            msg = "Label lists must have same length"
            raise ValueError(msg)

        if len(labels_a) == 0:
            return 0.0

        # Get all possible labels
        all_labels = list(set(labels_a) | set(labels_b))
        n = len(labels_a)

        # Build confusion matrix
        matrix: dict[tuple[LabelType, LabelType], int] = {}
        for label_1 in all_labels:
            for label_2 in all_labels:
                matrix[(label_1, label_2)] = 0

        for a, b in zip(labels_a, labels_b, strict=True):
            matrix[(a, b)] += 1

        # Compute observed agreement
        observed = sum(matrix[(label, label)] for label in all_labels) / n

        # Compute expected agreement
        expected = 0.0
        for label in all_labels:
            count_a = sum(1 for x in labels_a if x == label)
            count_b = sum(1 for x in labels_b if x == label)
            expected += (count_a / n) * (count_b / n)

        # Compute Kappa
        if expected == 1.0:
            return 1.0 if observed == 1.0 else 0.0

        kappa = (observed - expected) / (1 - expected)
        return float(kappa)

    @staticmethod
    def interpret_kappa(kappa: float) -> str:
        """Provide interpretation of Kappa value."""
        if kappa >= 0.81:
            return "Almost perfect agreement"
        if kappa >= 0.61:
            return "Substantial agreement"
        if kappa >= 0.41:
            return "Moderate agreement"
        if kappa >= 0.21:
            return "Fair agreement"
        if kappa >= 0.0:
            return "Slight agreement"
        return "Poor agreement (less than chance)"

    @staticmethod
    def agreement_rate(
        labels_a: list[LabelType],
        labels_b: list[LabelType],
    ) -> float:
        """Compute simple agreement rate (percentage of matching labels)."""
        if len(labels_a) != len(labels_b) or len(labels_a) == 0:
            return 0.0

        matches = sum(1 for a, b in zip(labels_a, labels_b, strict=True) if a == b)
        return matches / len(labels_a)


class LabelingTool:
    """
    CLI-based labeling tool for cluster quality assessment.

    Manages labeling sessions and provides an interactive
    interface for labeling cluster pairs.

    Usage:
        tool = LabelingTool(session_dir=Path("./labeling"))
        session = tool.create_session(labeler_id="user1", pairs=pairs)

        # Interactive labeling
        while not session.is_complete:
            pair = session.current_pair
            # Display pair to user...
            label = LabelType.SAME  # Get from user input
            tool.record_label(session, label)

        tool.save_session(session)
    """

    def __init__(
        self,
        session_dir: Path = Path("./labeling_sessions"),
    ) -> None:
        """
        Initialize the labeling tool.

        Args:
            session_dir: Directory for storing sessions.
        """
        self._session_dir = session_dir
        self._session_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "labeling_tool_initialized",
            session_dir=str(session_dir),
        )

    def create_session(
        self,
        labeler_id: str,
        pairs: list[ClusterPair],
        shuffle: bool = True,
        random_seed: int | None = None,
    ) -> LabelingSession:
        """
        Create a new labeling session.

        Args:
            labeler_id: ID of the labeler.
            pairs: Cluster pairs to label.
            shuffle: Whether to shuffle pairs.
            random_seed: Random seed for shuffling.

        Returns:
            New LabelingSession.
        """
        import uuid

        session_id = str(uuid.uuid4())[:8]

        # Optionally shuffle pairs
        session_pairs = list(pairs)
        if shuffle:
            if random_seed is not None:
                random.seed(random_seed)
            random.shuffle(session_pairs)

        session = LabelingSession(
            id=session_id,
            labeler_id=labeler_id,
            pairs=session_pairs,
            settings={
                "shuffled": shuffle,
                "random_seed": random_seed,
            },
        )

        logger.info(
            "labeling_session_created",
            session_id=session_id,
            labeler_id=labeler_id,
            n_pairs=len(pairs),
        )

        return session

    def record_label(
        self,
        session: LabelingSession,
        label: LabelType,
        confidence: float = 1.0,
        notes: str = "",
        time_taken: float | None = None,
    ) -> LabelingResult:
        """
        Record a labeling decision.

        Args:
            session: Current session.
            label: Assigned label.
            confidence: Labeler confidence (0-1).
            notes: Optional notes.
            time_taken: Time taken in seconds.

        Returns:
            The recorded LabelingResult.
        """
        pair = session.current_pair
        if pair is None:
            msg = "No current pair to label"
            raise ValueError(msg)

        result = LabelingResult(
            pair_id=pair.id,
            label=label,
            labeler_id=session.labeler_id,
            confidence=confidence,
            time_taken_seconds=time_taken or 0.0,
            notes=notes,
        )

        session.add_result(result)

        logger.debug(
            "label_recorded",
            session_id=session.id,
            pair_id=pair.id,
            label=label.value,
        )

        return result

    def save_session(self, session: LabelingSession) -> Path:
        """
        Save a labeling session.

        Args:
            session: Session to save.

        Returns:
            Path where session was saved.
        """
        filename = f"session_{session.id}_{session.labeler_id}.json"
        path = self._session_dir / filename
        session.save(path)
        return path

    def load_session(self, session_id: str) -> LabelingSession | None:
        """
        Load a labeling session by ID.

        Args:
            session_id: Session ID to load.

        Returns:
            Loaded session or None if not found.
        """
        for path in self._session_dir.glob(f"session_{session_id}_*.json"):
            return LabelingSession.load(path)
        return None

    def list_sessions(self) -> Iterator[tuple[str, str, Path]]:
        """
        List all saved sessions.

        Yields:
            Tuples of (session_id, labeler_id, path).
        """
        for path in sorted(self._session_dir.glob("session_*.json")):
            try:
                stem = path.stem
                parts = stem.split("_")
                if len(parts) >= 3:
                    session_id = parts[1]
                    labeler_id = "_".join(parts[2:])
                    yield session_id, labeler_id, path
            except Exception as e:
                logger.warning(
                    "session_file_parse_error",
                    path=str(path),
                    error=str(e),
                )

    def export_labels(
        self,
        sessions: list[LabelingSession],
        output_path: Path,
    ) -> int:
        """
        Export labels from multiple sessions to a file.

        Args:
            sessions: Sessions to export.
            output_path: Output file path.

        Returns:
            Number of labels exported.
        """
        all_results = []
        for session in sessions:
            for result in session.results:
                all_results.append(
                    {
                        "session_id": session.id,
                        "labeler_id": session.labeler_id,
                        **result.to_dict(),
                    }
                )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        logger.info(
            "labels_exported",
            path=str(output_path),
            n_labels=len(all_results),
        )

        return len(all_results)

    def compute_agreement(
        self,
        session_a: LabelingSession,
        session_b: LabelingSession,
    ) -> dict[str, Any]:
        """
        Compute inter-rater agreement between two sessions.

        Args:
            session_a: First session.
            session_b: Second session.

        Returns:
            Agreement metrics including Kappa and agreement rate.
        """
        # Build label maps
        labels_a: dict[str, LabelType] = {}
        for result in session_a.results:
            labels_a[result.pair_id] = result.label

        labels_b: dict[str, LabelType] = {}
        for result in session_b.results:
            labels_b[result.pair_id] = result.label

        # Find common pairs
        common_pairs = set(labels_a.keys()) & set(labels_b.keys())

        if not common_pairs:
            return {
                "n_common_pairs": 0,
                "kappa": 0.0,
                "agreement_rate": 0.0,
                "interpretation": "No common pairs to compare",
            }

        list_a = [labels_a[pair_id] for pair_id in sorted(common_pairs)]
        list_b = [labels_b[pair_id] for pair_id in sorted(common_pairs)]

        kappa = InterRaterAgreement.cohens_kappa(list_a, list_b)
        agreement_rate = InterRaterAgreement.agreement_rate(list_a, list_b)

        return {
            "labeler_a": session_a.labeler_id,
            "labeler_b": session_b.labeler_id,
            "n_common_pairs": len(common_pairs),
            "kappa": round(kappa, 4),
            "agreement_rate": round(agreement_rate, 4),
            "interpretation": InterRaterAgreement.interpret_kappa(kappa),
        }

    def generate_pairs_from_clusters(
        self,
        cluster_samples: dict[str, list[str]],
        n_pairs: int = 50,
        samples_per_cluster: int = 3,
    ) -> list[ClusterPair]:
        """
        Generate cluster pairs for labeling from cluster data.

        Args:
            cluster_samples: Mapping of cluster ID to sample messages.
            n_pairs: Number of pairs to generate.
            samples_per_cluster: Samples to show per cluster.

        Returns:
            List of ClusterPair objects.
        """
        import uuid

        cluster_ids = list(cluster_samples.keys())
        if len(cluster_ids) < 2:
            return []

        pairs: list[ClusterPair] = []
        used_pairs: set[tuple[str, str]] = set()

        # Calculate max possible pairs (n choose 2)
        max_possible_pairs = (len(cluster_ids) * (len(cluster_ids) - 1)) // 2
        target_pairs = min(n_pairs, max_possible_pairs)

        # If we need most/all pairs, generate exhaustively for reliability
        if target_pairs >= max_possible_pairs * 0.7:
            for i, cluster_a in enumerate(cluster_ids):
                for cluster_b in cluster_ids[i + 1 :]:
                    if len(pairs) >= target_pairs:
                        break
                    samples_a = cluster_samples[cluster_a][:samples_per_cluster]
                    samples_b = cluster_samples[cluster_b][:samples_per_cluster]
                    pairs.append(
                        ClusterPair(
                            id=str(uuid.uuid4())[:8],
                            cluster_a_id=cluster_a,
                            cluster_b_id=cluster_b,
                            cluster_a_samples=samples_a,
                            cluster_b_samples=samples_b,
                        )
                    )
                if len(pairs) >= target_pairs:
                    break
        else:
            # Random sampling for larger cluster sets
            for _ in range(n_pairs * 3):
                if len(pairs) >= n_pairs:
                    break

                idx_a, idx_b = random.sample(range(len(cluster_ids)), 2)
                cluster_a = cluster_ids[idx_a]
                cluster_b = cluster_ids[idx_b]

                pair_key = (min(cluster_a, cluster_b), max(cluster_a, cluster_b))
                if pair_key in used_pairs:
                    continue
                used_pairs.add(pair_key)

                samples_a = cluster_samples[cluster_a][:samples_per_cluster]
                samples_b = cluster_samples[cluster_b][:samples_per_cluster]

                pairs.append(
                    ClusterPair(
                        id=str(uuid.uuid4())[:8],
                        cluster_a_id=cluster_a,
                        cluster_b_id=cluster_b,
                        cluster_a_samples=samples_a,
                        cluster_b_samples=samples_b,
                    )
                )

        logger.info(
            "cluster_pairs_generated",
            n_pairs=len(pairs),
            n_clusters=len(cluster_ids),
        )

        return pairs

    @property
    def session_dir(self) -> Path:
        """Return the session directory."""
        return self._session_dir
