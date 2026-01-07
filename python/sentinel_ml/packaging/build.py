"""
Build Configuration and Validation for Sentinel Log AI.

This module provides tools for configuring and validating builds
across different platforms and targets.

Design Patterns:
- Builder Pattern: BuildConfig construction
- Strategy Pattern: Platform-specific build strategies
- Factory Pattern: BuildArtifact creation
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

import structlog

logger = structlog.get_logger(__name__)


class BuildTarget(Enum):
    """
    Supported build targets.

    Attributes:
        PYTHON_WHEEL: Python wheel package.
        PYTHON_SDIST: Python source distribution.
        DOCKER_IMAGE: Docker container image.
        WINDOWS_EXE: Windows executable (PyInstaller).
        LINUX_BINARY: Linux binary (PyInstaller).
        MACOS_APP: macOS application bundle.
        GO_BINARY: Go binary.
    """

    PYTHON_WHEEL = "wheel"
    PYTHON_SDIST = "sdist"
    DOCKER_IMAGE = "docker"
    WINDOWS_EXE = "windows-exe"
    LINUX_BINARY = "linux-binary"
    MACOS_APP = "macos-app"
    GO_BINARY = "go-binary"


class BuildPlatform(Enum):
    """Build platform identifiers."""

    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "darwin"
    ANY = "any"


@dataclass
class BuildArtifact:
    """
    A build artifact with metadata.

    Attributes:
        name: Artifact name.
        path: Path to the artifact.
        target: Build target type.
        platform: Target platform.
        size_bytes: File size in bytes.
        checksum_sha256: SHA-256 checksum.
        created_at: Creation timestamp.
        metadata: Additional metadata.
    """

    name: str
    path: Path
    target: BuildTarget
    platform: BuildPlatform
    size_bytes: int
    checksum_sha256: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "path": str(self.path),
            "target": self.target.value,
            "platform": self.platform.value,
            "size_bytes": self.size_bytes,
            "checksum_sha256": self.checksum_sha256,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_file(
        cls,
        path: Path,
        target: BuildTarget,
        build_platform: BuildPlatform,
        metadata: dict[str, Any] | None = None,
    ) -> BuildArtifact:
        """
        Create an artifact from an existing file.

        Args:
            path: Path to the file.
            target: Build target type.
            build_platform: Target platform.
            metadata: Additional metadata.

        Returns:
            BuildArtifact instance.
        """
        if not path.exists():
            raise FileNotFoundError(f"Artifact file not found: {path}")

        size_bytes = path.stat().st_size
        checksum = cls._compute_checksum(path)

        return cls(
            name=path.name,
            path=path,
            target=target,
            platform=build_platform,
            size_bytes=size_bytes,
            checksum_sha256=checksum,
            metadata=metadata or {},
        )

    @staticmethod
    def _compute_checksum(path: Path) -> str:
        """Compute SHA-256 checksum of a file."""
        sha256 = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def verify_checksum(self) -> bool:
        """Verify the artifact checksum."""
        if not self.path.exists():
            return False
        computed = self._compute_checksum(self.path)
        return computed == self.checksum_sha256


@dataclass
class BuildConfig:
    """
    Build configuration.

    Attributes:
        project_root: Root directory of the project.
        output_dir: Directory for build outputs.
        targets: Build targets to generate.
        version: Version string to embed.
        clean_build: Whether to clean before building.
        optimize: Optimization level.
        include_debug: Include debug symbols.
    """

    project_root: Path
    output_dir: Path
    targets: list[BuildTarget] = field(default_factory=list)
    version: str = "0.0.0"
    clean_build: bool = True
    optimize: int = 2
    include_debug: bool = False
    env_vars: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.project_root.exists():
            raise ValueError(f"Project root does not exist: {self.project_root}")

    @classmethod
    def from_pyproject(cls, project_root: Path) -> BuildConfig:
        """
        Create configuration from pyproject.toml.

        Args:
            project_root: Root directory of the project.

        Returns:
            BuildConfig instance.
        """
        import re

        pyproject_path = project_root / "pyproject.toml"
        if not pyproject_path.exists():
            raise FileNotFoundError("pyproject.toml not found")

        content = pyproject_path.read_text(encoding="utf-8")

        version_match = re.search(r'version\s*=\s*"([^"]+)"', content)
        version = version_match.group(1) if version_match else "0.0.0"

        return cls(
            project_root=project_root,
            output_dir=project_root / "dist",
            version=version,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "project_root": str(self.project_root),
            "output_dir": str(self.output_dir),
            "targets": [t.value for t in self.targets],
            "version": self.version,
            "clean_build": self.clean_build,
            "optimize": self.optimize,
            "include_debug": self.include_debug,
        }


class BuildValidator:
    """
    Validates build environment and configuration.

    This class checks that all required tools and dependencies
    are available before starting a build.
    """

    REQUIRED_TOOLS: ClassVar[dict[BuildTarget, list[str]]] = {
        BuildTarget.PYTHON_WHEEL: ["python", "pip"],
        BuildTarget.PYTHON_SDIST: ["python", "pip"],
        BuildTarget.DOCKER_IMAGE: ["docker"],
        BuildTarget.WINDOWS_EXE: ["python", "pyinstaller"],
        BuildTarget.LINUX_BINARY: ["python", "pyinstaller"],
        BuildTarget.MACOS_APP: ["python", "pyinstaller"],
        BuildTarget.GO_BINARY: ["go"],
    }

    def __init__(self, config: BuildConfig) -> None:
        """
        Initialize the validator.

        Args:
            config: Build configuration to validate.
        """
        self.config = config
        logger.info("build_validator_initialized")

    def validate(self) -> list[str]:
        """
        Validate the build environment.

        Returns:
            List of validation errors, empty if valid.
        """
        errors: list[str] = []

        errors.extend(self._validate_project_structure())
        errors.extend(self._validate_tools())
        errors.extend(self._validate_dependencies())

        if errors:
            logger.warning("build_validation_failed", errors=errors)
        else:
            logger.info("build_validation_passed")

        return errors

    def _validate_project_structure(self) -> list[str]:
        """Validate project structure."""
        errors: list[str] = []
        required_files = ["pyproject.toml", "README.md"]

        for filename in required_files:
            if not (self.config.project_root / filename).exists():
                errors.append(f"Required file missing: {filename}")

        return errors

    def _validate_tools(self) -> list[str]:
        """Validate required tools are available."""
        errors: list[str] = []

        for target in self.config.targets:
            required = self.REQUIRED_TOOLS.get(target, [])
            for tool in required:
                if not self._tool_available(tool):
                    errors.append(f"Tool required for {target.value} not found: {tool}")

        return errors

    def _validate_dependencies(self) -> list[str]:
        """Validate Python dependencies."""
        errors: list[str] = []

        try:
            result = subprocess.run(
                ["python", "-m", "pip", "check"],
                capture_output=True,
                text=True,
                cwd=self.config.project_root,
            )
            if result.returncode != 0:
                errors.append(f"Dependency check failed: {result.stdout}")
        except subprocess.SubprocessError as e:
            errors.append(f"Could not validate dependencies: {e}")

        return errors

    @staticmethod
    def _tool_available(tool: str) -> bool:
        """Check if a tool is available in PATH."""
        return shutil.which(tool) is not None

    def get_current_platform(self) -> BuildPlatform:
        """Get the current build platform."""
        system = platform.system().lower()
        platform_map = {
            "windows": BuildPlatform.WINDOWS,
            "linux": BuildPlatform.LINUX,
            "darwin": BuildPlatform.MACOS,
        }
        return platform_map.get(system, BuildPlatform.LINUX)


class BuildRunner:
    """
    Executes builds for different targets.

    This class coordinates the build process for various
    target types and platforms.
    """

    def __init__(self, config: BuildConfig) -> None:
        """
        Initialize the build runner.

        Args:
            config: Build configuration.
        """
        self.config = config
        self._validator = BuildValidator(config)
        self._artifacts: list[BuildArtifact] = []

        logger.info(
            "build_runner_initialized",
            project_root=str(config.project_root),
            targets=[t.value for t in config.targets],
        )

    def build(self, validate: bool = True) -> list[BuildArtifact]:
        """
        Execute the build.

        Args:
            validate: Whether to validate before building.

        Returns:
            List of generated artifacts.

        Raises:
            RuntimeError: If validation fails.
        """
        if validate:
            errors = self._validator.validate()
            if errors:
                raise RuntimeError(f"Build validation failed: {errors}")

        if self.config.clean_build:
            self._clean()

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        for target in self.config.targets:
            logger.info("building_target", target=target.value)
            artifact = self._build_target(target)
            if artifact:
                self._artifacts.append(artifact)

        logger.info(
            "build_completed",
            artifact_count=len(self._artifacts),
            artifacts=[a.name for a in self._artifacts],
        )

        return self._artifacts

    def _clean(self) -> None:
        """Clean build artifacts."""
        if self.config.output_dir.exists():
            shutil.rmtree(self.config.output_dir)
            logger.info("build_output_cleaned", path=str(self.config.output_dir))

    def _build_target(self, target: BuildTarget) -> BuildArtifact | None:
        """Build a specific target."""
        build_methods = {
            BuildTarget.PYTHON_WHEEL: self._build_wheel,
            BuildTarget.PYTHON_SDIST: self._build_sdist,
            BuildTarget.DOCKER_IMAGE: self._build_docker,
            BuildTarget.WINDOWS_EXE: self._build_pyinstaller,
            BuildTarget.GO_BINARY: self._build_go,
        }

        method = build_methods.get(target)
        if method:
            return method()

        logger.warning("unsupported_build_target", target=target.value)
        return None

    def _build_wheel(self) -> BuildArtifact | None:
        """Build Python wheel."""
        try:
            subprocess.run(
                ["python", "-m", "build", "--wheel", "-o", str(self.config.output_dir)],
                capture_output=True,
                text=True,
                cwd=self.config.project_root,
                check=True,
            )

            wheel_files = list(self.config.output_dir.glob("*.whl"))
            if wheel_files:
                return BuildArtifact.from_file(
                    wheel_files[0],
                    BuildTarget.PYTHON_WHEEL,
                    BuildPlatform.ANY,
                )

        except subprocess.CalledProcessError as e:
            logger.error("wheel_build_failed", error=e.stderr)

        return None

    def _build_sdist(self) -> BuildArtifact | None:
        """Build Python source distribution."""
        try:
            subprocess.run(
                ["python", "-m", "build", "--sdist", "-o", str(self.config.output_dir)],
                capture_output=True,
                text=True,
                cwd=self.config.project_root,
                check=True,
            )

            sdist_files = list(self.config.output_dir.glob("*.tar.gz"))
            if sdist_files:
                return BuildArtifact.from_file(
                    sdist_files[0],
                    BuildTarget.PYTHON_SDIST,
                    BuildPlatform.ANY,
                )

        except subprocess.CalledProcessError as e:
            logger.error("sdist_build_failed", error=e.stderr)

        return None

    def _build_docker(self) -> BuildArtifact | None:
        """Build Docker image."""
        dockerfile = self.config.project_root / "Dockerfile"
        if not dockerfile.exists():
            logger.warning("dockerfile_not_found")
            return None

        image_name = f"sentinel-log-ai:{self.config.version}"

        try:
            subprocess.run(
                [
                    "docker", "build",
                    "-t", image_name,
                    "-f", str(dockerfile),
                    str(self.config.project_root),
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            tar_path = self.config.output_dir / f"sentinel-log-ai-{self.config.version}.tar"
            subprocess.run(
                ["docker", "save", "-o", str(tar_path), image_name],
                check=True,
            )

            return BuildArtifact.from_file(
                tar_path,
                BuildTarget.DOCKER_IMAGE,
                BuildPlatform.ANY,
                metadata={"image_name": image_name},
            )

        except subprocess.CalledProcessError as e:
            logger.error("docker_build_failed", error=str(e))

        return None

    def _build_pyinstaller(self) -> BuildArtifact | None:
        """Build executable with PyInstaller."""
        spec_file = self.config.project_root / "sentinel-ml.spec"
        if not spec_file.exists():
            logger.warning("pyinstaller_spec_not_found")
            return None

        current_platform = self._validator.get_current_platform()

        try:
            subprocess.run(
                [
                    "pyinstaller",
                    "--distpath", str(self.config.output_dir),
                    str(spec_file),
                ],
                capture_output=True,
                text=True,
                cwd=self.config.project_root,
                check=True,
            )

            if current_platform == BuildPlatform.WINDOWS:
                exe_pattern = "*.exe"
                target = BuildTarget.WINDOWS_EXE
            else:
                exe_pattern = "sentinel-ml"
                target = BuildTarget.LINUX_BINARY

            exe_files = list(self.config.output_dir.glob(exe_pattern))
            if exe_files:
                return BuildArtifact.from_file(exe_files[0], target, current_platform)

        except subprocess.CalledProcessError as e:
            logger.error("pyinstaller_build_failed", error=e.stderr)

        return None

    def _build_go(self) -> BuildArtifact | None:
        """Build Go binary."""
        current_platform = self._validator.get_current_platform()
        output_name = "sentinel-log-ai"
        if current_platform == BuildPlatform.WINDOWS:
            output_name += ".exe"

        output_path = self.config.output_dir / output_name

        env = os.environ.copy()
        env.update(self.config.env_vars)

        try:
            subprocess.run(
                [
                    "go", "build",
                    "-o", str(output_path),
                    "-ldflags", f"-X main.version={self.config.version}",
                    "./cmd/agent",
                ],
                capture_output=True,
                text=True,
                cwd=self.config.project_root,
                env=env,
                check=True,
            )

            return BuildArtifact.from_file(
                output_path,
                BuildTarget.GO_BINARY,
                current_platform,
            )

        except subprocess.CalledProcessError as e:
            logger.error("go_build_failed", error=e.stderr)

        return None

    @property
    def artifacts(self) -> list[BuildArtifact]:
        """Get list of generated artifacts."""
        return self._artifacts.copy()

    def to_dict(self) -> dict[str, Any]:
        """Convert runner state to dictionary."""
        return {
            "config": self.config.to_dict(),
            "artifacts": [a.to_dict() for a in self._artifacts],
        }
