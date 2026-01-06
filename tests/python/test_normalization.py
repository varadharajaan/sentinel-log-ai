"""Tests for the normalization module."""

import re

import pytest

from sentinel_ml.normalization import MaskingRule, NormalizationPipeline, normalize


class TestNormalizationPipeline:
    """Test cases for the NormalizationPipeline."""

    @pytest.fixture
    def pipeline(self) -> NormalizationPipeline:
        """Create a fresh pipeline for each test."""
        return NormalizationPipeline()

    def test_normalize_ipv4(self, pipeline: NormalizationPipeline) -> None:
        """Test IPv4 address masking."""
        msg = "Connection from 192.168.1.100 to 10.0.0.1"
        result = pipeline.normalize(msg)
        assert result == "Connection from <ip> to <ip>"

    def test_normalize_uuid(self, pipeline: NormalizationPipeline) -> None:
        """Test UUID masking."""
        msg = "Request ID: 550e8400-e29b-41d4-a716-446655440000"
        result = pipeline.normalize(msg)
        assert result == "Request ID: <uuid>"

    def test_normalize_iso_timestamp(self, pipeline: NormalizationPipeline) -> None:
        """Test ISO timestamp masking."""
        msg = "Event at 2024-01-15T10:30:00.123Z"
        result = pipeline.normalize(msg)
        assert result == "Event at <ts>"

    def test_normalize_hex_token(self, pipeline: NormalizationPipeline) -> None:
        """Test hex token masking."""
        msg = "Token: 0x1234abcdef5678"
        result = pipeline.normalize(msg)
        assert result == "Token: <hex>"

    def test_normalize_long_number(self, pipeline: NormalizationPipeline) -> None:
        """Test long number masking."""
        msg = "Process 1234567890 started"
        result = pipeline.normalize(msg)
        assert result == "Process <num> started"

    def test_normalize_url(self, pipeline: NormalizationPipeline) -> None:
        """Test URL masking."""
        msg = "Fetching https://api.example.com/v1/users"
        result = pipeline.normalize(msg)
        assert result == "Fetching <url>"

    def test_normalize_email(self, pipeline: NormalizationPipeline) -> None:
        """Test email masking."""
        msg = "User email: user@example.com"
        result = pipeline.normalize(msg)
        assert result == "User email: <email>"

    def test_normalize_complex_log(self, pipeline: NormalizationPipeline) -> None:
        """Test normalization of a complex log line."""
        msg = (
            "2024-01-15T10:30:00Z ERROR [pid=12345] "
            "Connection from 192.168.1.100 failed for user@example.com "
            "request_id=550e8400-e29b-41d4-a716-446655440000"
        )
        result = pipeline.normalize(msg)
        # Should mask timestamp, PID, IP, email, UUID
        assert "<ts>" in result
        assert "<ip>" in result
        assert "<email>" in result
        assert "<uuid>" in result

    def test_normalize_preserves_structure(self, pipeline: NormalizationPipeline) -> None:
        """Test that normalization preserves log structure."""
        msg = "ERROR: Database connection failed"
        result = pipeline.normalize(msg)
        assert result == "ERROR: Database connection failed"

    def test_add_custom_rule(self, pipeline: NormalizationPipeline) -> None:
        """Test adding a custom masking rule."""
        # Insert at beginning to take priority over number masking
        pipeline.rules.insert(
            0,
            MaskingRule(
                name="custom_id",
                pattern=re.compile(r"ID-\d+"),
                replacement="<custom_id>",
            ),
        )
        msg = "Processing ID-12345"
        result = pipeline.normalize(msg)
        assert result == "Processing <custom_id>"

    def test_disable_rule(self, pipeline: NormalizationPipeline) -> None:
        """Test disabling a masking rule."""
        pipeline.disable_rule("ipv4")
        msg = "Connection from 192.168.1.100"
        result = pipeline.normalize(msg)
        assert "192.168.1.100" in result

    def test_global_normalize_function(self) -> None:
        """Test the global normalize convenience function."""
        msg = "Error from 192.168.1.1"
        result = normalize(msg)
        assert result == "Error from <ip>"
