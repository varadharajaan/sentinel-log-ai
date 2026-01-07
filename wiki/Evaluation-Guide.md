# Evaluation and Quality Guide

This guide covers the evaluation and quality assessment framework for Sentinel Log AI.

## Overview

The evaluation module provides tools for assessing and improving clustering quality:

| Component | Purpose |
|-----------|---------|
| Quality Metrics | Quantitative clustering assessment |
| Golden Datasets | Regression testing |
| Reports | Automated report generation |
| Ablation Studies | Component analysis |
| Human Labeling | Manual quality assessment |

## Quick Start

### Evaluate Clustering Quality

```python
from sentinel_ml.evaluation import QualityEvaluator

evaluator = QualityEvaluator()
result = evaluator.evaluate(embeddings, labels)

print(f"Overall quality: {result.overall_quality}")
print(f"Clusters: {result.n_clusters}")
```

### Generate Reports

```python
from sentinel_ml.evaluation import EvaluationReportGenerator
from pathlib import Path

generator = EvaluationReportGenerator()
report = generator.generate(result)
generator.save(report, Path("report.md"))
```

## Quality Metrics

### Available Metrics

**Silhouette Score** (-1 to 1, higher is better)
- Measures how similar objects are to their own cluster
- Values above 0.5 indicate good clustering

**Davies-Bouldin Index** (0 to infinity, lower is better)
- Ratio of within-cluster to between-cluster distances
- Values below 1.0 indicate good clustering

**Calinski-Harabasz Index** (0 to infinity, higher is better)
- Ratio of between-cluster to within-cluster variance
- Higher values indicate better-defined clusters

### Custom Metrics

Add custom metrics by implementing `MetricStrategy`:

```python
from sentinel_ml.evaluation.metrics import MetricStrategy, MetricResult, MetricType

class CustomMetric(MetricStrategy):
    @property
    def metric_type(self) -> MetricType:
        return MetricType.CUSTOM
    
    @property
    def optimal_direction(self) -> str:
        return "higher"
    
    def compute(self, embeddings, labels) -> MetricResult:
        # Compute your metric
        value = compute_custom_metric(embeddings, labels)
        return MetricResult(
            metric_type=self.metric_type,
            value=value,
            interpretation="Custom metric interpretation",
            optimal_direction=self.optimal_direction,
        )
```

## Golden Datasets

Golden datasets provide stable test cases for regression testing.

### Create Dataset

```python
from sentinel_ml.evaluation import GoldenDataset, GoldenRecord

records = [
    GoldenRecord(
        id="rec-1",
        message="Error connecting to database",
        normalized="Error connecting to database",
        expected_cluster_id="db-errors",
    ),
]

dataset = GoldenDataset(
    name="core-patterns-v1",
    records=records,
    quality_thresholds={"adjusted_rand_index": 0.8},
)
```

### Run Regression Test

```python
from sentinel_ml.evaluation import RegressionRunner

runner = RegressionRunner(tolerance=0.05)
result = runner.run(dataset, embeddings, predicted_labels)

if result.status.value == "passed":
    print("Regression test passed")
else:
    print(f"Failed: {result.n_misclassified} misclassified")
```

## Ablation Studies

Analyze how different components affect clustering quality.

### Parameter Sweep

```python
from sentinel_ml.evaluation import AblationRunner
from sentinel_ml.evaluation.ablation import create_parameter_sweep_configs

config = create_parameter_sweep_configs(
    component_type=ComponentType.MIN_CLUSTER_SIZE,
    baseline_value=5,
    test_values=[3, 7, 10],
)

runner = AblationRunner()
study = runner.run(config, cluster_func, embeddings)

print(f"Best config: {study.get_best_config()}")
```

### Component Toggle

Test enabling/disabling components:

```python
from sentinel_ml.evaluation.ablation import create_toggle_config

config = create_toggle_config(
    component_type=ComponentType.NORMALIZATION,
    component_name="IP Masking",
)

study = runner.run(config, cluster_func, embeddings)
```

## Human Labeling

Collect human judgments on cluster quality.

### Create Session

```python
from sentinel_ml.evaluation import LabelingTool, LabelType

tool = LabelingTool(session_dir=Path("sessions"))

pairs = tool.generate_pairs_from_clusters(cluster_samples, n_pairs=50)
session = tool.create_session(labeler_id="analyst-1", pairs=pairs)

# Record labels
tool.record_label(session, LabelType.SAME)
tool.save_session(session)
```

### Measure Agreement

```python
session_a = tool.load_session("session-a-id")
session_b = tool.load_session("session-b-id")

agreement = tool.compute_agreement(session_a, session_b)
print(f"Kappa: {agreement['kappa']}")
```

## Best Practices

1. **Regular Evaluation**: Run quality metrics after each clustering run
2. **Golden Datasets**: Maintain curated datasets for regression testing
3. **Trend Analysis**: Track metrics over time to detect degradation
4. **Multiple Raters**: Use multiple labelers and measure agreement
5. **Ablation Testing**: Validate component contributions systematically

## Related Topics

- [[Architecture Overview|Architecture-Overview]]
- [[Configuration Reference|Configuration-Reference]]
- [[Testing Guide|Testing-Guide]]
