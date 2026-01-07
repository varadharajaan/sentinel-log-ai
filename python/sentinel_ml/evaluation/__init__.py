"""
Evaluation and quality metrics module for Sentinel Log AI.

This module provides comprehensive tools for evaluating clustering quality,
creating golden datasets, generating evaluation reports, and running
ablation studies.

Supports the following use cases:
- Clustering quality assessment (silhouette, Davies-Bouldin index)
- Golden dataset creation and management for regression testing
- Automated evaluation report generation with trend analysis
- Ablation testing for component contribution analysis
- Human labeling tool for cluster quality assessment

Design Patterns:
- Strategy Pattern: Pluggable quality metrics
- Template Method: Common evaluation workflow
- Factory Pattern: Metric and report creation
- Observer Pattern: Real-time evaluation tracking
- Repository Pattern: Golden dataset management

SOLID Principles:
- Single Responsibility: Each class handles one evaluation concern
- Open/Closed: Extensible via MetricStrategy interface
- Liskov Substitution: All metrics implement same interface
- Interface Segregation: Minimal interfaces for specific capabilities
- Dependency Inversion: Depends on abstractions not implementations
"""

from sentinel_ml.evaluation.ablation import (
    AblationConfig,
    AblationResult,
    AblationRunner,
    AblationStudy,
    ComponentConfig,
)
from sentinel_ml.evaluation.golden_dataset import (
    ExpectedCluster,
    GoldenDataset,
    GoldenDatasetManager,
    GoldenRecord,
    RegressionResult,
    RegressionRunner,
)
from sentinel_ml.evaluation.labeling import (
    ClusterPair,
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
    MetricStrategy,
    MetricType,
    QualityEvaluator,
    SilhouetteMetric,
)
from sentinel_ml.evaluation.report import (
    EvaluationReport,
    EvaluationReportConfig,
    EvaluationReportGenerator,
    TrendAnalysis,
    TrendDirection,
)

__all__ = [
    # Ablation
    "AblationConfig",
    "AblationResult",
    "AblationRunner",
    "AblationStudy",
    "CalinskiHarabaszMetric",
    # Metrics
    "ClusteringQualityResult",
    "ClusterPair",
    "ComponentConfig",
    "DaviesBouldinMetric",
    "ExpectedCluster",
    "EvaluationReport",
    # Report
    "EvaluationReportConfig",
    "EvaluationReportGenerator",
    # Golden Dataset
    "GoldenDataset",
    "GoldenDatasetManager",
    "GoldenRecord",
    # Labeling
    "LabelingResult",
    "LabelingSession",
    "LabelingTool",
    "LabelType",
    "MetricResult",
    "MetricStrategy",
    "MetricType",
    "QualityEvaluator",
    "RegressionResult",
    "RegressionRunner",
    "SilhouetteMetric",
    "TrendAnalysis",
    "TrendDirection",
]
