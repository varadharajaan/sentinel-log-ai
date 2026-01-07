# Packaging Architecture

This document describes the packaging module architecture, including version management, changelog automation, build configuration, release management, and installation verification.

## Module Structure

```
python/sentinel_ml/packaging/
    __init__.py          # Public API exports
    version.py           # Semantic versioning
    changelog.py         # Changelog generation
    build.py             # Build configuration
    release.py           # Release management
    installer.py         # Installation verification
```

## Component Diagram

```
+------------------+     +-------------------+     +------------------+
|  VersionManager  |---->| ChangelogManager  |---->| ReleaseManager   |
|                  |     |                   |     |                  |
| - read_version() |     | - add_entry()     |     | - create()       |
| - write_version()|     | - read_changelog()|     | - publish()      |
| - bump_version() |     | - create_release()|     | - verify()       |
+------------------+     +-------------------+     +------------------+
         |                        |                        |
         v                        v                        v
+------------------+     +-------------------+     +------------------+
| SemanticVersion  |     | ChangelogEntry    |     | ReleaseArtifact  |
| (frozen)         |     | ChangelogRelease  |     | ReleaseConfig    |
+------------------+     +-------------------+     +------------------+

+------------------+     +-------------------+
|   BuildRunner    |---->| BuildValidator    |
|                  |     |                   |
| - build()        |     | - validate()      |
| - clean()        |     | - verify_checksum |
+------------------+     +-------------------+
         |                        |
         v                        v
+------------------+     +-------------------+
|   BuildConfig    |     |  BuildArtifact    |
|   BuildTarget    |     |  (with checksum)  |
+------------------+     +-------------------+

+----------------------+     +----------------------+
| InstallationVerifier |---->|  DependencyChecker   |
|                      |     |                      |
| - verify_import()    |     | - check_dependency() |
| - verify_cli()       |     | - check_all()        |
| - generate_report()  |     +----------------------+
+----------------------+              |
                                      v
                         +----------------------+
                         |   DependencyInfo     |
                         |   DependencyStatus   |
                         +----------------------+
```

## Design Patterns

### Immutable Value Objects

`SemanticVersion` is implemented as a frozen dataclass to ensure immutability:

```python
@dataclass(frozen=True)
class SemanticVersion:
    major: int
    minor: int
    patch: int
    prerelease: str | None = None
    build_metadata: str | None = None
```

This ensures version objects cannot be accidentally modified and can be safely used as dictionary keys or set members.

### Factory Methods

Static factory methods for object creation:

```python
class SemanticVersion:
    @staticmethod
    def parse(version_string: str) -> SemanticVersion:
        """Parse version from string."""
        ...
```

### Strategy Pattern

Version bumping uses strategy pattern through enum:

```python
class VersionBumpType(Enum):
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"

def bump(self, bump_type: VersionBumpType) -> SemanticVersion:
    match bump_type:
        case VersionBumpType.MAJOR:
            return SemanticVersion(self.major + 1, 0, 0)
        case VersionBumpType.MINOR:
            return SemanticVersion(self.major, self.minor + 1, 0)
        case VersionBumpType.PATCH:
            return SemanticVersion(self.major, self.minor, self.patch + 1)
```

### Builder Pattern

`ReleaseConfig` and `BuildConfig` use builder-style configuration:

```python
config = ReleaseConfig(
    version="1.0.0",
    project_root=Path("."),
    pypi_upload=True,
    docker_push=True,
    github_release=True,
)
```

### Template Method

`ChangelogGenerator` uses template method for parsing different commit formats:

```python
class ChangelogGenerator:
    def parse_conventional_commit(self, message: str) -> ChangelogEntry | None:
        """Template method for parsing commits."""
        ...
```

## Data Flow

### Release Flow

```
1. Developer commits changes
           |
           v
2. ChangelogGenerator parses commits
           |
           v
3. ChangelogManager updates CHANGELOG.md
           |
           v
4. VersionManager bumps version
           |
           v
5. Developer creates git tag
           |
           v
6. GitHub Actions triggers release workflow
           |
           +---> BuildRunner builds artifacts
           |            |
           |            v
           |     BuildValidator verifies
           |            |
           v            v
7. ReleaseManager creates GitHub release
           |
           +---> Publishes to PyPI
           +---> Pushes to Docker registry
```

### Version Resolution

```
VERSION file (source of truth)
       |
       v
pyproject.toml (dynamic versioning)
       |
       +---> Python package __version__
       |
       v
Docker build ARG
       |
       v
Go build -ldflags
```

## Build Targets

| Target | Output | Tool | Use Case |
|--------|--------|------|----------|
| WHEEL | .whl | build | pip install |
| SDIST | .tar.gz | build | Source distribution |
| DOCKER | image | docker | Container deployment |
| PYINSTALLER | .exe | pyinstaller | Windows standalone |
| GO_BINARY | binary | go build | Agent deployment |

## Platform Matrix

| Platform | Python | Go | Docker | PyInstaller |
|----------|--------|----|----|-------------|
| linux/amd64 | Yes | Yes | Yes | No |
| linux/arm64 | Yes | Yes | Yes | No |
| darwin/amd64 | Yes | Yes | No | No |
| darwin/arm64 | Yes | Yes | No | No |
| windows/amd64 | Yes | Yes | No | Yes |

## Logging

All packaging operations use structured logging:

```python
import structlog

logger = structlog.get_logger(__name__)

def bump_version(self, bump_type: VersionBumpType) -> SemanticVersion:
    current = self.read_version()
    new_version = current.bump(bump_type)
    
    logger.info(
        "version_bumped",
        from_version=str(current),
        to_version=str(new_version),
        bump_type=bump_type.value,
    )
    
    self.write_version(new_version)
    return new_version
```

## Error Handling

The module uses specific exceptions:

```python
class VersionParseError(ValueError):
    """Raised when version string is invalid."""
    pass

class BuildError(RuntimeError):
    """Raised when build fails."""
    pass

class ReleaseError(RuntimeError):
    """Raised when release fails."""
    pass
```

## Configuration

### pyproject.toml Integration

```toml
[project]
name = "sentinel-ml"
dynamic = ["version"]

[tool.hatch.version]
path = "VERSION"
```

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| SENTINEL_VERSION | Override version | VERSION file |
| PYPI_API_TOKEN | PyPI authentication | Required for publish |
| GITHUB_TOKEN | GitHub API access | From Actions |

## Testing

Unit tests cover:

1. **SemanticVersion** - Parsing, formatting, comparison, bumping
2. **VersionManager** - File I/O, version bumping
3. **ChangelogEntry** - Markdown generation
4. **ChangelogGenerator** - Commit parsing
5. **BuildConfig** - Configuration validation
6. **BuildArtifact** - Checksum verification
7. **ReleaseConfig** - Safe defaults
8. **InstallationVerifier** - Import and CLI checks
9. **DependencyChecker** - Package detection

Integration tests cover:

1. Version bump + changelog update workflow
2. Build + validate workflow
3. Release notes generation
4. Dependency verification

## Security Considerations

1. **Checksum Verification** - All artifacts include SHA256 checksums
2. **Signed Tags** - Git tags should be GPG signed for releases
3. **Token Security** - API tokens stored as GitHub secrets
4. **Non-root Container** - Docker image runs as non-root user

## Related Documentation

- [Packaging Guide](../wiki/Packaging-Guide.md) - User guide
- [Release Workflow](.github/workflows/release.yml) - CI/CD configuration
- [Architecture Overview](architecture.md) - System architecture
