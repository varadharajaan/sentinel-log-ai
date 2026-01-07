# Contributing Guide

Thank you for your interest in contributing to Sentinel Log AI! This guide will help you get started.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and constructive in all interactions.

## How to Contribute

### Reporting Bugs

1. **Check existing issues** - Search [GitHub Issues](https://github.com/varadharajaan/sentinel-log-ai/issues) first
2. **Create a new issue** using the bug report template
3. **Include**:
   - Clear description of the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python/Go version)
   - Relevant logs or error messages

### Suggesting Features

1. **Check existing issues** for similar requests
2. **Create a feature request** using the template
3. **Include**:
   - Use case description
   - Proposed solution
   - Alternatives considered
   - Willingness to contribute

### Contributing Code

#### 1. Fork and Clone

```bash
# Fork via GitHub UI, then:
git clone https://github.com/YOUR_USERNAME/sentinel-log-ai.git
cd sentinel-log-ai
git remote add upstream https://github.com/varadharajaan/sentinel-log-ai.git
```

#### 2. Create a Branch

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
```

#### 3. Set Up Development Environment

```bash
# Install all dependencies
make install

# Or manually:
# Go
go mod download

# Python
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

#### 4. Make Changes

- Follow the [[Code Style]] guidelines
- Write tests for new functionality
- Update documentation as needed
- Keep commits small and focused

#### 5. Run Tests and Linters

```bash
# Run all checks
make check

# Or individually:
make lint          # Run linters
make test          # Run tests
make format        # Format code
```

#### 6. Commit Changes

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Format: <type>(<scope>): <description>

# Types:
# feat:     New feature
# fix:      Bug fix
# docs:     Documentation
# style:    Formatting (no code change)
# refactor: Code restructuring
# test:     Adding tests
# chore:    Maintenance

# Examples:
git commit -m "feat(cli): add HTML report generation"
git commit -m "fix(clustering): handle empty embeddings array"
git commit -m "docs: update configuration reference"
git commit -m "test(novelty): add edge case tests"
```

#### 7. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request via GitHub.

---

## Pull Request Guidelines

### PR Checklist

- [ ] Follows code style guidelines
- [ ] Includes tests for new functionality
- [ ] All tests pass (`make test`)
- [ ] Linters pass (`make lint`)
- [ ] Documentation updated if needed
- [ ] Commit messages follow conventional commits
- [ ] PR description explains changes clearly

### PR Title Format

```
<type>(<scope>): <description>

Examples:
feat(embedding): add caching for repeated embeddings
fix(grpc): handle connection timeout gracefully
docs(wiki): add troubleshooting guide
```

### PR Description Template

```markdown
## Summary
Brief description of what this PR does.

## Changes
- Change 1
- Change 2
- Change 3

## Testing
How were these changes tested?

## Related Issues
Closes #123
Relates to #456

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Lint checks pass
```

---

## Development Workflow

### Branch Strategy

```
main
 │
 ├── feature/m6-cli-ux          # Feature branches
 ├── feature/m7-performance
 ├── fix/grpc-timeout
 └── docs/configuration-guide
```

- `main` - Stable, release-ready code
- `feature/*` - New features
- `fix/*` - Bug fixes
- `docs/*` - Documentation updates

### Milestones

Work is organized into milestones:

| Milestone | Focus |
|-----------|-------|
| M0 | Project scaffolding |
| M1 | Ingestion & preprocessing |
| M2 | Embeddings & vector store |
| M3 | Clustering & patterns |
| M4 | Novelty detection |
| M5 | LLM explanations |
| M6 | CLI & UX |
| M7+ | Future work |

### Labels

PRs and issues should use appropriate labels:

| Label | Description |
|-------|-------------|
| `core` | Core functionality |
| `ml` | Machine learning |
| `ux` | User experience |
| `docs` | Documentation |
| `testing` | Test coverage |
| `bug` | Bug fix |
| `enhancement` | New feature |
| `breaking` | Breaking change |

---

## Code Review Process

### For Reviewers

1. **Functionality** - Does the code do what it claims?
2. **Tests** - Are there adequate tests?
3. **Style** - Does it follow our conventions?
4. **Performance** - Any performance concerns?
5. **Security** - Any security implications?
6. **Documentation** - Is it well documented?

### Review Comments

- Be constructive and specific
- Explain the "why" behind suggestions
- Distinguish between blocking and non-blocking issues
- Acknowledge good work

---

## Testing Requirements

### Coverage Targets

| Component | Minimum |
|-----------|---------|
| Python modules | 85% |
| Go packages | 80% |
| Critical paths | 95% |

### Test Types

- **Unit tests** - Required for all new code
- **Integration tests** - For component interactions
- **End-to-end tests** - For critical workflows

### Running Tests

```bash
# All tests
make test

# Python only
make test-python
pytest tests/python/ -v

# Go only
make test-go
go test ./... -v

# With coverage
pytest --cov=sentinel_ml --cov-report=html
```

---

## Documentation

### What to Document

- Public APIs and functions
- Configuration options
- Command-line usage
- Architecture decisions
- Design patterns used

### Documentation Locations

| Type | Location |
|------|----------|
| API docs | Docstrings in code |
| User guide | `docs/` |
| Wiki | `wiki/` |
| README | Project root |

### Docstring Format

Python (Google style):
```python
def process_logs(
    logs: list[str],
    config: Config | None = None,
) -> ProcessingResult:
    """
    Process log messages through the ML pipeline.

    Args:
        logs: List of raw log messages to process.
        config: Optional configuration. Uses defaults if not provided.

    Returns:
        ProcessingResult containing clusters, novelty scores, and metadata.

    Raises:
        ProcessingError: If embedding or clustering fails.
        ConfigError: If configuration is invalid.

    Example:
        >>> result = process_logs(["log1", "log2"])
        >>> print(f"Found {len(result.clusters)} clusters")
    """
```

Go:
```go
// ProcessLogs processes log messages through the ML pipeline.
//
// It takes raw log messages and returns clusters, novelty scores,
// and metadata about the processing.
//
// Parameters:
//   - logs: Slice of raw log messages
//   - opts: Optional processing options
//
// Returns ProcessingResult or error if processing fails.
func ProcessLogs(logs []string, opts ...Option) (*ProcessingResult, error) {
```

---

## Release Process

1. All tests pass on `main`
2. Version bumped in `__init__.py` and `go.mod`
3. CHANGELOG updated
4. Tag created: `git tag v0.X.0`
5. GitHub Release created
6. Packages published

---

## Getting Help

- **Questions**: Open a [Discussion](https://github.com/varadharajaan/sentinel-log-ai/discussions)
- **Bugs**: Open an [Issue](https://github.com/varadharajaan/sentinel-log-ai/issues)
- **Chat**: Join our community (coming soon)

---

*Thank you for contributing! ❤️*

*— Varad*
