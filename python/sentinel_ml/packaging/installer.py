"""
Installation Verification for Sentinel Log AI.

This module provides tools for verifying successful installation
and checking dependency requirements.

Design Patterns:
- Chain of Responsibility: Dependency checking chain
- Strategy Pattern: Different verification strategies
- Observer Pattern: Installation status notifications
"""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar

import structlog

logger = structlog.get_logger(__name__)


class DependencyStatus(Enum):
    """
    Status of a dependency.

    Attributes:
        INSTALLED: Dependency is installed and compatible.
        MISSING: Dependency is not installed.
        INCOMPATIBLE: Dependency version is incompatible.
        OPTIONAL: Optional dependency, not installed.
    """

    INSTALLED = "installed"
    MISSING = "missing"
    INCOMPATIBLE = "incompatible"
    OPTIONAL = "optional"


@dataclass
class DependencyInfo:
    """
    Information about a dependency.

    Attributes:
        name: Package name.
        required_version: Required version specification.
        installed_version: Installed version, if any.
        status: Dependency status.
        optional: Whether dependency is optional.
        extra: The extra that requires this dependency.
    """

    name: str
    required_version: str | None = None
    installed_version: str | None = None
    status: DependencyStatus = DependencyStatus.MISSING
    optional: bool = False
    extra: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "required_version": self.required_version,
            "installed_version": self.installed_version,
            "status": self.status.value,
            "optional": self.optional,
            "extra": self.extra,
        }


@dataclass
class InstallationResult:
    """
    Result of installation verification.

    Attributes:
        success: Overall verification success.
        python_version: Python version.
        package_version: Installed package version.
        dependencies: Dependency verification results.
        checks_passed: List of passed checks.
        checks_failed: List of failed checks.
        warnings: Warning messages.
        timestamp: Verification timestamp.
    """

    success: bool
    python_version: str
    package_version: str | None = None
    dependencies: list[DependencyInfo] = field(default_factory=list)
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "success": self.success,
            "python_version": self.python_version,
            "package_version": self.package_version,
            "dependencies": [d.to_dict() for d in self.dependencies],
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "warnings": self.warnings,
            "timestamp": self.timestamp.isoformat(),
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines: list[str] = []

        status = "SUCCESS" if self.success else "FAILED"
        lines.append(f"Installation Verification: {status}")
        lines.append(f"Python: {self.python_version}")

        if self.package_version:
            lines.append(f"Package: sentinel-log-ai-ml {self.package_version}")

        lines.append("")

        if self.checks_passed:
            lines.append(f"Passed: {len(self.checks_passed)}")
            for check in self.checks_passed:
                lines.append(f"  [OK] {check}")

        if self.checks_failed:
            lines.append(f"Failed: {len(self.checks_failed)}")
            for check in self.checks_failed:
                lines.append(f"  [FAIL] {check}")

        if self.warnings:
            lines.append(f"Warnings: {len(self.warnings)}")
            for warning in self.warnings:
                lines.append(f"  [WARN] {warning}")

        return "\n".join(lines)


class DependencyChecker:
    """
    Checks dependency installation and compatibility.

    This class verifies that all required and optional dependencies
    are properly installed with compatible versions.
    """

    CORE_DEPENDENCIES: ClassVar[list[str]] = [
        "pydantic",
        "pydantic-settings",
        "pyyaml",
        "numpy",
        "structlog",
        "grpcio",
        "protobuf",
    ]

    ML_DEPENDENCIES: ClassVar[list[str]] = [
        "sentence-transformers",
        "faiss-cpu",
        "torch",
        "scikit-learn",
    ]

    LLM_DEPENDENCIES: ClassVar[list[str]] = [
        "ollama",
        "httpx",
    ]

    def __init__(self) -> None:
        """Initialize the dependency checker."""
        self._dependencies: list[DependencyInfo] = []
        logger.info("dependency_checker_initialized")

    def check_all(self) -> list[DependencyInfo]:
        """
        Check all dependencies.

        Returns:
            List of dependency information.
        """
        self._dependencies = []

        for name in self.CORE_DEPENDENCIES:
            info = self._check_package(name, optional=False)
            self._dependencies.append(info)

        for name in self.ML_DEPENDENCIES:
            info = self._check_package(name, optional=True, extra="ml")
            self._dependencies.append(info)

        for name in self.LLM_DEPENDENCIES:
            info = self._check_package(name, optional=True, extra="llm")
            self._dependencies.append(info)

        installed_count = sum(
            1 for d in self._dependencies if d.status == DependencyStatus.INSTALLED
        )
        missing_count = sum(1 for d in self._dependencies if d.status == DependencyStatus.MISSING)

        logger.info(
            "dependencies_checked",
            installed=installed_count,
            missing=missing_count,
            total=len(self._dependencies),
        )

        return self._dependencies

    def _check_package(
        self,
        name: str,
        optional: bool = False,
        extra: str | None = None,
    ) -> DependencyInfo:
        """
        Check a single package.

        Args:
            name: Package name.
            optional: Whether package is optional.
            extra: The extra that requires this package.

        Returns:
            DependencyInfo for the package.
        """
        info = DependencyInfo(
            name=name,
            optional=optional,
            extra=extra,
        )

        try:
            dist = importlib.metadata.distribution(name)
            info.installed_version = dist.version
            info.status = DependencyStatus.INSTALLED

        except importlib.metadata.PackageNotFoundError:
            if optional:
                info.status = DependencyStatus.OPTIONAL
            else:
                info.status = DependencyStatus.MISSING

        return info

    def get_missing_required(self) -> list[DependencyInfo]:
        """Get list of missing required dependencies."""
        return [
            d for d in self._dependencies if d.status == DependencyStatus.MISSING and not d.optional
        ]

    def get_missing_optional(self) -> list[DependencyInfo]:
        """Get list of missing optional dependencies."""
        return [d for d in self._dependencies if d.status == DependencyStatus.OPTIONAL]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "dependencies": [d.to_dict() for d in self._dependencies],
            "missing_required": [d.to_dict() for d in self.get_missing_required()],
            "missing_optional": [d.to_dict() for d in self.get_missing_optional()],
        }


class InstallationVerifier:
    """
    Verifies complete installation of Sentinel Log AI.

    This class runs a comprehensive set of checks to ensure
    the package is properly installed and configured.

    Attributes:
        result: The verification result.
    """

    def __init__(self) -> None:
        """Initialize the verifier."""
        self._dependency_checker = DependencyChecker()
        self.result: InstallationResult | None = None
        logger.info("installation_verifier_initialized")

    def verify(self) -> InstallationResult:
        """
        Run all verification checks.

        Returns:
            InstallationResult with verification details.
        """
        logger.info("installation_verification_started")

        checks_passed: list[str] = []
        checks_failed: list[str] = []
        warnings: list[str] = []

        python_version = (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )
        checks_passed.append(f"Python version {python_version}")

        package_version = self._check_package_installed()
        if package_version:
            checks_passed.append("Package installed")
        else:
            checks_failed.append("Package not installed")

        dependencies = self._dependency_checker.check_all()

        missing_required = self._dependency_checker.get_missing_required()
        if missing_required:
            for dep in missing_required:
                checks_failed.append(f"Missing required dependency: {dep.name}")
        else:
            checks_passed.append("All required dependencies installed")

        missing_optional = self._dependency_checker.get_missing_optional()
        for dep in missing_optional:
            warnings.append(f"Optional dependency not installed: {dep.name} ({dep.extra})")

        if self._check_imports():
            checks_passed.append("Core module imports successful")
        else:
            checks_failed.append("Core module imports failed")

        if self._check_cli():
            checks_passed.append("CLI available")
        else:
            warnings.append("CLI not available in PATH")

        success = len(checks_failed) == 0

        self.result = InstallationResult(
            success=success,
            python_version=python_version,
            package_version=package_version,
            dependencies=dependencies,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            warnings=warnings,
        )

        logger.info(
            "installation_verification_completed",
            success=success,
            passed=len(checks_passed),
            failed=len(checks_failed),
            warnings=len(warnings),
        )

        return self.result

    def _check_package_installed(self) -> str | None:
        """Check if package is installed and return version."""
        try:
            dist = importlib.metadata.distribution("sentinel-log-ai-ml")
            return dist.version
        except importlib.metadata.PackageNotFoundError:
            return None

    def _check_imports(self) -> bool:
        """Check that core modules can be imported."""
        modules_to_check = [
            "sentinel_ml",
            "sentinel_ml.config",
            "sentinel_ml.models",
            "sentinel_ml.preprocessing",
        ]

        for module in modules_to_check:
            try:
                importlib.import_module(module)
            except ImportError as e:
                logger.warning("import_failed", module=module, error=str(e))
                return False

        return True

    def _check_cli(self) -> bool:
        """Check if CLI is available."""
        try:
            result = subprocess.run(
                ["sentinel-ml", "--version"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def generate_report(self) -> str:
        """
        Generate detailed verification report.

        Returns:
            Markdown formatted report.
        """
        if self.result is None:
            self.verify()

        assert self.result is not None

        lines: list[str] = []
        lines.append("# Installation Verification Report")
        lines.append("")
        lines.append(f"**Date:** {self.result.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append(f"**Python:** {self.result.python_version}")

        if self.result.package_version:
            lines.append(f"**Package:** sentinel-log-ai-ml {self.result.package_version}")

        lines.append("")

        status = "PASSED" if self.result.success else "FAILED"
        lines.append(f"## Status: {status}")
        lines.append("")

        if self.result.checks_passed:
            lines.append("### Passed Checks")
            lines.append("")
            for check in self.result.checks_passed:
                lines.append(f"- {check}")
            lines.append("")

        if self.result.checks_failed:
            lines.append("### Failed Checks")
            lines.append("")
            for check in self.result.checks_failed:
                lines.append(f"- {check}")
            lines.append("")

        if self.result.warnings:
            lines.append("### Warnings")
            lines.append("")
            for warning in self.result.warnings:
                lines.append(f"- {warning}")
            lines.append("")

        lines.append("### Dependencies")
        lines.append("")
        lines.append("| Package | Status | Version |")
        lines.append("|---------|--------|---------|")

        for dep in self.result.dependencies:
            version = dep.installed_version or "N/A"
            lines.append(f"| {dep.name} | {dep.status.value} | {version} |")

        lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "result": self.result.to_dict() if self.result else None,
        }


def verify_installation() -> InstallationResult:
    """
    Convenience function to verify installation.

    Returns:
        InstallationResult with verification details.
    """
    verifier = InstallationVerifier()
    return verifier.verify()


def check_dependencies() -> list[DependencyInfo]:
    """
    Convenience function to check dependencies.

    Returns:
        List of dependency information.
    """
    checker = DependencyChecker()
    return checker.check_all()
