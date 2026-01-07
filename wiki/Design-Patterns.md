# Design Patterns

Sentinel Log AI employs industry-standard design patterns to ensure maintainability, testability, and extensibility. This document catalogs the patterns used across the codebase.

## Pattern Overview

| Pattern | Location | Purpose |
|---------|----------|---------|
| Strategy | Embeddings, Clustering, Novelty, LLM, Themes, Formatters | Interchangeable algorithms |
| Facade | Console, Services | Simplified unified interface |
| Factory | Parser Registry, Provider Factory | Object creation abstraction |
| Template Method | Report Generation | Base algorithm with customization |
| Decorator | Profiler, Logging | Add behavior without modification |
| Observer | Progress Tracking | Event notification |
| Singleton | Global Logger, Profiler | Single shared instance |
| Builder | Config, Request builders | Complex object construction |
| Repository | VectorStore | Data access abstraction |

---

## Strategy Pattern

### Description
Defines a family of algorithms, encapsulates each one, and makes them interchangeable. Allows the algorithm to vary independently from clients that use it.

### Implementation: Embedding Providers

```python
# Abstract Strategy
class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for texts."""
        pass

# Concrete Strategies
class SentenceTransformerProvider(EmbeddingProvider):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    
    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts)

class MockEmbeddingProvider(EmbeddingProvider):
    def embed(self, texts: list[str]) -> np.ndarray:
        # Deterministic mock for testing
        return np.random.default_rng(42).random((len(texts), 384))

# Context
class EmbeddingService:
    def __init__(self, provider: EmbeddingProvider):
        self._provider = provider
    
    def generate(self, texts: list[str]) -> np.ndarray:
        return self._provider.embed(texts)
```

### Benefits
- **Testability**: Mock providers for unit tests
- **Flexibility**: Swap algorithms without code changes
- **Open/Closed**: Add new providers without modifying existing code

### Other Strategy Implementations
- `ClusteringAlgorithm` (HDBSCAN, KMeans, DBSCAN)
- `NoveltyDetector` (KNN, LOF, IsolationForest)
- `LLMProvider` (Ollama, OpenAI, Mock)
- `Theme` (Dark, Light, Minimal, Colorblind)
- `Formatter` (JSON, Table, Cluster, Novelty)

---

## Facade Pattern

### Description
Provides a unified interface to a set of interfaces in a subsystem, making the subsystem easier to use.

### Implementation: Console

```python
class Console:
    """
    Facade that unifies all CLI output operations.
    Clients interact with Console rather than individual formatters,
    theme managers, progress trackers, etc.
    """
    
    def __init__(self, config: ConsoleConfig):
        self._config = config
        self._theme = get_theme(config.theme)
        self._json_formatter = JSONFormatter()
        self._table_formatter = TableFormatter()
        self._cluster_formatter = ClusterFormatter()
        self._progress_tracker = ProgressTracker()
    
    # Unified interface methods
    def info(self, message: str) -> None: ...
    def success(self, message: str) -> None: ...
    def error(self, message: str) -> None: ...
    def print_table(self, data: TableData) -> None: ...
    def print_clusters(self, clusters: list[ClusterSummary]) -> None: ...
    def print_json(self, data: Any) -> None: ...
    def progress(self, total: int) -> ProgressTracker: ...
```

### Benefits
- **Simplicity**: Single entry point for all CLI operations
- **Decoupling**: Clients don't need to know internal components
- **Consistency**: Centralized theme and format handling

---

## Factory Pattern

### Description
Defines an interface for creating objects but lets subclasses decide which class to instantiate.

### Implementation: Parser Registry

```go
// Parser Registry in Go
type ParserRegistry struct {
    parsers []Parser
}

func NewParserRegistry() *ParserRegistry {
    return &ParserRegistry{
        parsers: []Parser{
            NewJSONParser(),
            NewSyslogParser(),
            NewNginxParser(),
            NewTracebackParser(),
            NewCommonParser(), // Fallback
        },
    }
}

func (r *ParserRegistry) Parse(line string) (*LogRecord, error) {
    for _, parser := range r.parsers {
        if parser.CanParse(line) {
            return parser.Parse(line)
        }
    }
    return nil, ErrNoParserMatch
}
```

### Implementation: LLM Provider Factory

```python
class LLMProviderFactory:
    @staticmethod
    def create(config: LLMConfig) -> LLMProvider:
        match config.provider:
            case "ollama":
                return OllamaProvider(config)
            case "openai":
                return OpenAIProvider(config)
            case "mock":
                return MockLLMProvider()
            case _:
                raise ValueError(f"Unknown provider: {config.provider}")
```

---

## Template Method Pattern

### Description
Defines the skeleton of an algorithm in a method, deferring some steps to subclasses.

### Implementation: Report Generation

```python
class Reporter(ABC):
    """Base reporter with template method."""
    
    def generate(self, data: ReportData) -> str:
        """Template method defining report structure."""
        sections = []
        sections.append(self._generate_header(data))
        sections.append(self._generate_toc(data))
        sections.append(self._generate_summary(data))
        sections.append(self._generate_clusters(data))
        sections.append(self._generate_novelty(data))
        sections.append(self._generate_explanations(data))
        sections.append(self._generate_footer(data))
        return self._join_sections(sections)
    
    # Abstract methods for subclasses to implement
    @abstractmethod
    def _generate_header(self, data: ReportData) -> str: ...
    
    @abstractmethod
    def _generate_clusters(self, data: ReportData) -> str: ...

class MarkdownReporter(Reporter):
    def _generate_header(self, data: ReportData) -> str:
        return f"# {data.title}\n\n*Generated: {data.timestamp}*"
    
    def _generate_clusters(self, data: ReportData) -> str:
        # Markdown-specific cluster formatting
        ...

class HTMLReporter(Reporter):
    def _generate_header(self, data: ReportData) -> str:
        return f"<h1>{data.title}</h1><p>Generated: {data.timestamp}</p>"
    
    def _generate_clusters(self, data: ReportData) -> str:
        # HTML-specific cluster formatting with CSS
        ...
```

---

## Decorator Pattern

### Description
Attaches additional responsibilities to an object dynamically without modifying its structure.

### Implementation: Profiler

```python
class Profiler:
    """Decorator that adds timing instrumentation."""
    
    def profile(self, name: str):
        """Decorator for timing function execution."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.measure(name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    @contextmanager
    def measure(self, name: str):
        """Context manager for timing code blocks."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self._record(name, duration)

# Usage
profiler = Profiler()

@profiler.profile("embed_logs")
def embed_logs(logs: list[str]) -> np.ndarray:
    return embedding_service.embed(logs)

# Or with context manager
with profiler.measure("clustering"):
    clusters = clustering_service.cluster(embeddings)
```

---

## Observer Pattern

### Description
Defines a one-to-many dependency between objects so that when one object changes state, all its dependents are notified.

### Implementation: Progress Tracking

```python
class ProgressTracker:
    """Observable that notifies on progress updates."""
    
    def __init__(self):
        self._observers: list[ProgressObserver] = []
        self._current = 0
        self._total = 0
    
    def add_observer(self, observer: ProgressObserver) -> None:
        self._observers.append(observer)
    
    def update(self, current: int, status: str = "") -> None:
        self._current = current
        for observer in self._observers:
            observer.on_progress(current, self._total, status)
    
    def complete(self) -> None:
        for observer in self._observers:
            observer.on_complete(self._total)

class ConsoleProgressObserver(ProgressObserver):
    """Observer that renders progress to console."""
    
    def on_progress(self, current: int, total: int, status: str) -> None:
        percentage = (current / total) * 100
        bar = self._render_bar(percentage)
        print(f"\r{bar} {percentage:.1f}% {status}", end="")
    
    def on_complete(self, total: int) -> None:
        print(f"\n✓ Completed {total} items")
```

---

## Repository Pattern

### Description
Mediates between the domain and data mapping layers, acting like an in-memory collection of domain objects.

### Implementation: VectorStore

```python
class VectorStore:
    """Repository for vector embeddings with CRUD operations."""
    
    def __init__(self, config: VectorStoreConfig):
        self._index = self._create_index(config)
        self._metadata: dict[str, LogRecord] = {}
    
    # Repository CRUD operations
    def add(self, ids: list[str], vectors: np.ndarray, 
            records: list[LogRecord]) -> None:
        """Add vectors with associated metadata."""
        self._index.add(vectors)
        for id_, record in zip(ids, records):
            self._metadata[id_] = record
    
    def get(self, id_: str) -> LogRecord | None:
        """Retrieve a record by ID."""
        return self._metadata.get(id_)
    
    def search(self, query: np.ndarray, k: int) -> list[SearchResult]:
        """Find k nearest neighbors."""
        distances, indices = self._index.search(query, k)
        return self._build_results(distances, indices)
    
    def delete(self, ids: list[str]) -> None:
        """Remove vectors by ID."""
        for id_ in ids:
            self._metadata.pop(id_, None)
    
    def save(self, path: Path) -> None:
        """Persist to disk."""
        faiss.write_index(self._index, str(path / "index.faiss"))
        # Save metadata separately
    
    def load(self, path: Path) -> None:
        """Load from disk."""
        self._index = faiss.read_index(str(path / "index.faiss"))
```

---

## SOLID Principles

### Single Responsibility Principle (SRP)
Each module has one clear responsibility:

| Module | Single Responsibility |
|--------|----------------------|
| `parser.go` | Parse log lines |
| `source.go` | Read from log sources |
| `normalization.py` | Mask sensitive data |
| `embedding.py` | Generate vector embeddings |
| `clustering.py` | Group similar vectors |

### Open/Closed Principle (OCP)
- Parser Registry: Add parsers without modifying registry
- Provider Strategy: Add new LLM providers without changing service
- Formatter Strategy: Add output formats without changing console

### Liskov Substitution Principle (LSP)
- All `EmbeddingProvider` implementations are interchangeable
- All `ClusteringAlgorithm` implementations produce valid results
- All `Reporter` subclasses generate valid reports

### Interface Segregation Principle (ISP)
- Small, focused interfaces (`Parser`, `Source`, `Provider`)
- No forced implementation of unused methods

### Dependency Inversion Principle (DIP)
- Services depend on abstract providers, not concrete implementations
- Configuration injected via dependency injection
- Mock implementations for testing

---

*See also: [[Architecture Overview|Architecture-Overview]], [[Testing Guide|Testing-Guide]]*
