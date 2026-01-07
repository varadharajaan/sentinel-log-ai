# Evaluation and Quality Metrics

This document describes the evaluation and quality assessment framework for Sentinel Log AI clustering.

## Overview

The evaluation module provides comprehensive tools for:

1. **Clustering Quality Metrics**: Quantitative assessment of clustering quality
2. **Golden Dataset Management**: Regression testing with curated datasets
3. **Evaluation Reports**: Automated report generation with trend analysis
4. **Ablation Studies**: Component contribution analysis
5. **Human Labeling**: Manual cluster quality assessment with inter-rater agreement

## Module Structure

```
python/sentinel_ml/evaluation/
    __init__.py          # Module exports
    metrics.py           # Quality metrics (Silhouette, Davies-Bouldin, etc.)
    golden_dataset.py    # Golden dataset management
    report.py            # Report generation
    ablation.py          # Ablation testing framework
    labeling.py          # Human labeling tool
```

## Clustering Quality Metrics

### Available Metrics

| Metric | Range | Optimal | Description |
|--------|-------|---------|-------------|
| Silhouette Score | -1 to 1 | Higher | Measures cluster cohesion and separation |
| Davies-Bouldin Index | 0 to infinity | Lower | Ratio of within-cluster to between-cluster distances |
| Calinski-Harabasz Index | 0 to infinity | Higher | Ratio of between-cluster to within-cluster variance |

### Usage

```python
from sentinel_ml.evaluation import (
    QualityEvaluator,
    SilhouetteMetric,
    DaviesBouldinMetric,
    CalinskiHarabaszMetric,
)

# Create evaluator with default metrics
evaluator = QualityEvaluator()

# Or specify custom metrics
evaluator = QualityEvaluator(
    metrics=[SilhouetteMetric(), DaviesBouldinMetric()]
)

# Evaluate clustering quality
result = evaluator.evaluate(embeddings, labels)

print(f"Overall quality: {result.overall_quality}")
print(f"Number of clusters: {result.n_clusters}")
print(f"Noise points: {result.n_noise}")

# Access individual metrics
silhouette = result.get_metric(MetricType.SILHOUETTE)
if silhouette:
    print(f"Silhouette: {silhouette.value} - {silhouette.interpretation}")
```

### Metric Interpretation

**Silhouette Score**:
- 0.7 to 1.0: Excellent cluster separation
- 0.5 to 0.7: Good cluster separation
- 0.25 to 0.5: Moderate cluster separation
- Below 0.25: Poor cluster separation

**Davies-Bouldin Index**:
- Below 0.5: Excellent cluster definition
- 0.5 to 1.0: Good cluster definition
- 1.0 to 2.0: Moderate cluster definition
- Above 2.0: Poor cluster definition

## Golden Dataset Management

Golden datasets provide curated test cases for regression testing.

### Creating a Golden Dataset

```python
from sentinel_ml.evaluation import (
    GoldenDataset,
    GoldenRecord,
    GoldenDatasetManager,
)

# Create records
records = [
    GoldenRecord(
        id="rec-1",
        message="HTTP 200 OK response",
        normalized="HTTP <status> OK response",
        expected_cluster_id="http-success",
    ),
    GoldenRecord(
        id="rec-2",
        message="Database connection timeout",
        normalized="Database connection timeout",
        expected_cluster_id="db-errors",
    ),
]

# Create dataset
dataset = GoldenDataset(
    name="log-patterns-v1",
    description="Core log pattern test cases",
    records=records,
    quality_thresholds={"adjusted_rand_index": 0.8},
)

# Save dataset
manager = GoldenDatasetManager(storage_path=Path("golden_datasets"))
manager.save(dataset)
```

### Regression Testing

```python
from sentinel_ml.evaluation import RegressionRunner

runner = RegressionRunner(tolerance=0.05)

# Run regression test
result = runner.run(
    dataset=dataset,
    embeddings=current_embeddings,
    predicted_labels=predicted_cluster_ids,
)

print(f"Status: {result.status.value}")
print(f"ARI: {result.adjusted_rand_index}")
print(f"Misclassified: {result.n_misclassified}")
```

## Evaluation Reports

Generate reports in multiple formats with trend analysis.

### Report Generation

```python
from sentinel_ml.evaluation import (
    EvaluationReportGenerator,
    EvaluationReportConfig,
    ReportFormat,
)
from pathlib import Path

# Configure report
config = EvaluationReportConfig(
    title="Clustering Quality Report",
    output_format=ReportFormat.MARKDOWN,
    include_trends=True,
    history_window=10,
)

# Generate report
generator = EvaluationReportGenerator(config=config)
report = generator.generate(quality_result)

# Save report
generator.save(report, Path("reports/quality_report.md"))

# Or format for different output
json_content = generator.format(report, ReportFormat.JSON)
html_content = generator.format(report, ReportFormat.HTML)
```

### Report Contents

Reports include:
- Quality metrics summary
- Trend analysis (improving/stable/degrading)
- Recommendations for improvement
- Regression test results (if provided)
- Historical comparison

## Ablation Studies

Analyze component contributions through systematic testing.

### Parameter Sweep

```python
from sentinel_ml.evaluation import (
    AblationRunner,
    AblationConfig,
    ComponentConfig,
    ComponentType,
)
from sentinel_ml.evaluation.ablation import create_parameter_sweep_configs

# Create parameter sweep configurations
config = create_parameter_sweep_configs(
    component_type=ComponentType.MIN_CLUSTER_SIZE,
    baseline_value=5,
    test_values=[3, 7, 10, 15],
    n_runs=3,
)

# Run ablation study
runner = AblationRunner()

def cluster_func(embeddings, config_value):
    # Your clustering logic here
    return labels

study = runner.run(config, cluster_func, embeddings)

# Analyze results
print(f"Best configuration: {study.get_best_config()}")
for comparison in study.comparisons:
    print(f"{comparison.config_a} vs {comparison.config_b}")
    print(f"  Difference: {comparison.mean_difference}")
    print(f"  Significant: {comparison.is_significant}")
```

### Component Toggle

```python
from sentinel_ml.evaluation.ablation import create_toggle_config

# Test with/without normalization
config = create_toggle_config(
    component_type=ComponentType.NORMALIZATION,
    component_name="IP Address Masking",
    n_runs=5,
)

study = runner.run(config, cluster_func, embeddings)
```

## Human Labeling Tool

Collect human judgments on cluster quality.

### Creating Labeling Sessions

```python
from sentinel_ml.evaluation import (
    LabelingTool,
    LabelType,
    ClusterPair,
)

# Initialize tool
tool = LabelingTool(session_dir=Path("labeling_sessions"))

# Generate pairs from cluster data
cluster_samples = {
    "cluster-0": ["Sample message 1", "Sample message 2"],
    "cluster-1": ["Different pattern 1", "Different pattern 2"],
}

pairs = tool.generate_pairs_from_clusters(
    cluster_samples,
    n_pairs=50,
    samples_per_cluster=3,
)

# Create session for a labeler
session = tool.create_session(
    labeler_id="analyst-1",
    pairs=pairs,
)

# Record labels
for pair in pairs:
    # Show pair to labeler, get decision
    tool.record_label(session, LabelType.SAME)  # or DIFFERENT, SIMILAR, UNSURE

# Save session
tool.save_session(session)
```

### Inter-Rater Agreement

```python
from sentinel_ml.evaluation import InterRaterAgreement

# Compare two labelers
session_a = tool.load_session("session-a-id")
session_b = tool.load_session("session-b-id")

agreement = tool.compute_agreement(session_a, session_b)

print(f"Cohen's Kappa: {agreement['kappa']}")
print(f"Agreement Rate: {agreement['agreement_rate']}")
print(f"Interpretation: {agreement['interpretation']}")
```

### Kappa Interpretation

| Kappa Value | Interpretation |
|-------------|----------------|
| 0.81 - 1.00 | Almost perfect agreement |
| 0.61 - 0.80 | Substantial agreement |
| 0.41 - 0.60 | Moderate agreement |
| 0.21 - 0.40 | Fair agreement |
| 0.00 - 0.20 | Slight agreement |
| Below 0.00 | Poor agreement |

## Design Patterns

The evaluation module uses several design patterns:

1. **Strategy Pattern**: Pluggable quality metrics (MetricStrategy interface)
2. **Template Method**: Common evaluation workflow
3. **Factory Pattern**: Metric and report creation
4. **Repository Pattern**: Golden dataset storage and retrieval
5. **Observer Pattern**: Real-time evaluation tracking
6. **Command Pattern**: Labeling actions

## SOLID Principles

- **Single Responsibility**: Each class handles one evaluation concern
- **Open/Closed**: Extensible via MetricStrategy interface
- **Liskov Substitution**: All metrics implement same interface
- **Interface Segregation**: Minimal interfaces for specific capabilities
- **Dependency Inversion**: Depends on abstractions not implementations

## Dependencies

Required packages:
- `scikit-learn`: Clustering quality metrics
- `scipy`: Statistical testing for ablation studies
- `numpy`: Numerical operations

Install with:
```bash
pip install scikit-learn scipy numpy
```

## Related Documentation

- [Architecture Overview](architecture.md)
- [Data Flow](data-flow.md)
- [Configuration Reference](../wiki/Configuration-Reference.md)
