# Packaging Guide

This guide covers building, packaging, and releasing Sentinel Log AI for different platforms and distribution channels.

## Overview

Sentinel Log AI supports multiple distribution formats:

| Format | Use Case | Platform |
|--------|----------|----------|
| Python Package (wheel) | pip/pipx installation | All |
| Docker Image | Containerized deployment | Linux, macOS |
| Windows Executable | Standalone CLI | Windows |
| Go Binary | Agent deployment | All |

## Installation Methods

### Using pipx (Recommended)

pipx installs the package in an isolated environment:

```bash
# Install pipx if not already installed
pip install pipx
pipx ensurepath

# Install sentinel-ml
pipx install sentinel-ml
```

### Using pip

For integration with existing projects:

```bash
# Basic installation
pip install sentinel-ml

# With ML dependencies
pip install "sentinel-ml[ml]"

# With LLM support
pip install "sentinel-ml[ml,llm]"
```

### Using Docker

Pull the latest image:

```bash
docker pull ghcr.io/varadharajaan/sentinel-log-ai:latest
```

Run analysis:

```bash
docker run --rm -v $(pwd)/logs:/app/logs \
  ghcr.io/varadharajaan/sentinel-log-ai:latest \
  analyze /app/logs/sample.jsonl
```

### Windows Executable

Download `sentinel-ml.exe` from the [releases page](https://github.com/varadharajaan/sentinel-log-ai/releases).

```powershell
# Run directly
.\sentinel-ml.exe analyze logs\sample.jsonl
```

## Building from Source

### Prerequisites

- Python 3.10 or later
- Go 1.22 or later
- Docker (for container builds)
- PyInstaller (for Windows executable)

### Build Python Package

```bash
# Install build tools
pip install build

# Build wheel and sdist
python -m build

# Artifacts in dist/
ls dist/
# sentinel_ml-0.11.0-py3-none-any.whl
# sentinel_ml-0.11.0.tar.gz
```

### Build Docker Image

```bash
# Build image
docker build -t sentinel-log-ai .

# Run locally
docker run --rm sentinel-log-ai --help
```

### Build Windows Executable

```powershell
# Install PyInstaller
pip install pyinstaller

# Build executable
pyinstaller sentinel-ml.spec

# Output in dist/
dir dist\sentinel-ml.exe
```

### Build Go Binary

```bash
# Build for current platform
go build -o bin/sentinel-log-ai ./cmd/agent

# Cross-compile for Linux
GOOS=linux GOARCH=amd64 go build -o bin/sentinel-log-ai-linux ./cmd/agent

# Cross-compile for Windows
GOOS=windows GOARCH=amd64 go build -o bin/sentinel-log-ai.exe ./cmd/agent
```

## Version Management

Sentinel Log AI uses [Semantic Versioning](https://semver.org/).

### Version Files

The version is maintained in:
- `VERSION` - Single source of truth
- `pyproject.toml` - Uses dynamic versioning from VERSION

### Bumping Versions

Using the packaging module:

```python
from sentinel_ml.packaging import VersionManager, VersionBumpType

manager = VersionManager(project_root=".")

# Bump patch version (1.0.0 -> 1.0.1)
manager.bump_version(VersionBumpType.PATCH)

# Bump minor version (1.0.1 -> 1.1.0)
manager.bump_version(VersionBumpType.MINOR)

# Bump major version (1.1.0 -> 2.0.0)
manager.bump_version(VersionBumpType.MAJOR)
```

### Creating Git Tags

```bash
# Read current version
VERSION=$(cat VERSION)

# Create annotated tag
git tag -a "v${VERSION}" -m "Release v${VERSION}"

# Push tag to trigger release workflow
git push origin "v${VERSION}"
```

## Changelog Management

The changelog follows [Keep a Changelog](https://keepachangelog.com/) format.

### Adding Entries

Using conventional commits:

```bash
# Feature (goes to Added section)
git commit -m "feat(parser): add YAML support"

# Bug fix (goes to Fixed section)
git commit -m "fix: resolve memory leak in embeddings"

# Breaking change (highlighted in Changed section)
git commit -m "feat!: new API for vectorstore"
```

Using the changelog module:

```python
from sentinel_ml.packaging import ChangelogManager, ChangelogEntry, ChangelogEntryType

manager = ChangelogManager(project_root=".")

entry = ChangelogEntry(
    entry_type=ChangelogEntryType.ADDED,
    description="Support for Parquet log files",
    scope="parser",
)
manager.add_entry(entry)
```

### Entry Types

| Type | Conventional Commit | Description |
|------|-------------------|-------------|
| Added | `feat:` | New features |
| Changed | `refactor:`, `perf:` | Changes to existing functionality |
| Deprecated | `deprecate:` | Features to be removed |
| Removed | `remove:` | Removed features |
| Fixed | `fix:` | Bug fixes |
| Security | `security:` | Security fixes |

## Release Workflow

The release workflow is automated via GitHub Actions.

### Triggering a Release

1. Update CHANGELOG.md with release notes
2. Bump version in VERSION file
3. Commit changes
4. Create and push a tag

```bash
# Update changelog and version
vim CHANGELOG.md VERSION
git add CHANGELOG.md VERSION
git commit -m "chore: prepare release 0.11.0"

# Create and push tag
git tag -a v0.11.0 -m "Release v0.11.0"
git push origin main --tags
```

### Workflow Steps

The release workflow:

1. **Build Python Package** - Creates wheel and sdist
2. **Build Go Binaries** - Cross-compiles for all platforms
3. **Build Windows Executable** - Creates PyInstaller bundle
4. **Build Docker Image** - Multi-platform container image
5. **Create GitHub Release** - Attaches all artifacts
6. **Publish to PyPI** - For stable releases only

### Pre-release Versions

Tag with alpha, beta, or rc suffix:

```bash
git tag -a v0.12.0-alpha.1 -m "Pre-release v0.12.0-alpha.1"
```

Pre-releases:
- Appear on GitHub Releases as pre-release
- Are NOT published to PyPI
- Include all build artifacts

## Verification

### Verify Installation

Using the installation verifier:

```python
from sentinel_ml.packaging import InstallationVerifier, DependencyChecker

# Check dependencies
checker = DependencyChecker()
deps = checker.check_all(["numpy", "structlog", "click"])
for dep in deps:
    print(f"{dep.name}: {dep.status.value} ({dep.installed_version})")

# Verify installation
verifier = InstallationVerifier()
result = verifier.verify()
print(verifier.generate_report(result))
```

### Verify Package Integrity

```bash
# Download checksums
curl -LO https://github.com/.../SHA256SUMS.txt

# Verify
sha256sum -c SHA256SUMS.txt
```

## Troubleshooting

### pip Installation Fails

```bash
# Upgrade pip
pip install --upgrade pip

# Install with verbose output
pip install -v sentinel-ml
```

### Docker Build Fails

```bash
# Clean build cache
docker builder prune

# Build with no cache
docker build --no-cache -t sentinel-log-ai .
```

### PyInstaller Issues

```powershell
# Clean previous builds
Remove-Item -Recurse -Force build, dist

# Build with debug output
pyinstaller --debug all sentinel-ml.spec
```

## Development Setup

For contributors building locally:

```bash
# Clone repository
git clone https://github.com/varadharajaan/sentinel-log-ai.git
cd sentinel-log-ai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install in development mode
pip install -e ".[dev,ml,llm]"

# Run tests
pytest tests/

# Build package
python -m build
```

## Related Documentation

- [Quick Start](Quick-Start.md) - Getting started guide
- [Configuration Reference](Configuration-Reference.md) - All configuration options
- [Architecture Overview](Architecture-Overview.md) - System architecture
- [Contributing](Contributing.md) - Contribution guidelines
