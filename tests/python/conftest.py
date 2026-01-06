"""Pytest configuration for Python tests."""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "ml: marks tests that require ML dependencies")
    config.addinivalue_line("markers", "llm: marks tests that require LLM dependencies")
    config.addinivalue_line("markers", "grpc: marks tests that require gRPC server")
