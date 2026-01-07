"""
Comprehensive unit tests for LLM module.

Tests cover:
- LLMStats data class
- Explanation data class
- LLMProvider ABC and implementations
- OllamaProvider (with mocking)
- MockLLMProvider
- LLMService high-level API
- Prompt templates
- Error handling
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from sentinel_ml.config import LLMConfig
from sentinel_ml.exceptions import LLMError
from sentinel_ml.llm import (
    CLUSTER_EXPLANATION_PROMPT,
    ERROR_ANALYSIS_PROMPT,
    NOVELTY_EXPLANATION_PROMPT,
    SUMMARY_PROMPT,
    Explanation,
    ExplanationType,
    LLMService,
    LLMStats,
    MockLLMProvider,
    OllamaProvider,
    Severity,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def llm_config() -> LLMConfig:
    """Create a test LLM configuration."""
    return LLMConfig(
        provider="ollama",
        model="llama3.2",
        base_url="http://localhost:11434",
        timeout=60,
        max_retries=3,
        temperature=0.1,
    )


@pytest.fixture
def mock_cluster_summary() -> MagicMock:
    """Create a mock cluster summary for testing."""
    summary = MagicMock()
    summary.id = "cluster_123"
    summary.label = 0
    summary.size = 50
    summary.representative_messages = [
        "Connection timeout to database server",
        "Failed to connect to database",
        "Database connection refused",
    ]
    summary.common_level = "ERROR"
    summary.time_range_start = datetime(2026, 1, 7, 10, 0, 0, tzinfo=timezone.utc)
    summary.time_range_end = datetime(2026, 1, 7, 11, 0, 0, tzinfo=timezone.utc)
    return summary


@pytest.fixture
def mock_novelty_score() -> MagicMock:
    """Create a mock novelty score for testing."""
    score = MagicMock()
    score.score = 0.85
    score.is_novel = True
    score.index = 42
    score.message = "Unusual memory allocation pattern detected"
    return score


@pytest.fixture
def mock_log_record() -> MagicMock:
    """Create a mock log record for testing."""
    record = MagicMock()
    record.id = "record_456"
    record.message = "NullPointerException in UserService.getUser"
    record.level = "ERROR"
    record.source = "app.log"
    record.timestamp = datetime(2026, 1, 7, 12, 0, 0, tzinfo=timezone.utc)
    record.attrs = {"user_id": "12345", "request_id": "req-abc"}
    return record


# ============================================================================
# LLMStats Tests
# ============================================================================


class TestLLMStats:
    """Tests for LLMStats data class."""

    def test_default_values(self) -> None:
        """Test default values are correctly initialized."""
        stats = LLMStats()

        assert stats.total_requests == 0
        assert stats.successful_requests == 0
        assert stats.failed_requests == 0
        assert stats.total_tokens_used == 0
        assert stats.total_prompt_tokens == 0
        assert stats.total_completion_tokens == 0
        assert stats.avg_response_time_seconds == 0.0
        assert stats.last_request_time is None

    def test_record_successful_request(self) -> None:
        """Test recording a successful request."""
        stats = LLMStats()

        stats.record_request(
            success=True,
            response_time=1.5,
            prompt_tokens=100,
            completion_tokens=50,
        )

        assert stats.total_requests == 1
        assert stats.successful_requests == 1
        assert stats.failed_requests == 0
        assert stats.total_tokens_used == 150
        assert stats.total_prompt_tokens == 100
        assert stats.total_completion_tokens == 50
        assert stats.avg_response_time_seconds == 1.5
        assert stats.last_request_time is not None

    def test_record_failed_request(self) -> None:
        """Test recording a failed request."""
        stats = LLMStats()

        stats.record_request(success=False, response_time=0.5)

        assert stats.total_requests == 1
        assert stats.successful_requests == 0
        assert stats.failed_requests == 1
        assert stats.total_tokens_used == 0

    def test_multiple_requests_average(self) -> None:
        """Test average calculation with multiple requests."""
        stats = LLMStats()

        stats.record_request(success=True, response_time=1.0, prompt_tokens=100)
        stats.record_request(success=True, response_time=2.0, prompt_tokens=100)
        stats.record_request(success=True, response_time=3.0, prompt_tokens=100)

        assert stats.total_requests == 3
        assert stats.successful_requests == 3
        assert stats.avg_response_time_seconds == 2.0  # (1+2+3)/3

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        stats = LLMStats()
        stats.record_request(
            success=True,
            response_time=1.5,
            prompt_tokens=100,
            completion_tokens=50,
        )

        result = stats.to_dict()

        assert result["total_requests"] == 1
        assert result["successful_requests"] == 1
        assert result["failed_requests"] == 0
        assert result["success_rate"] == 1.0
        assert result["total_tokens_used"] == 150
        assert result["avg_response_time_seconds"] == 1.5
        assert result["last_request_time"] is not None

    def test_success_rate_zero_requests(self) -> None:
        """Test success rate with no requests."""
        stats = LLMStats()
        result = stats.to_dict()

        assert result["success_rate"] == 0.0


# ============================================================================
# Explanation Tests
# ============================================================================


class TestExplanation:
    """Tests for Explanation data class."""

    def test_default_values(self) -> None:
        """Test default values are correctly initialized."""
        explanation = Explanation()

        assert explanation.id is not None  # UUID generated
        assert explanation.explanation_type == ExplanationType.CLUSTER
        assert explanation.summary == ""
        assert explanation.root_cause is None
        assert explanation.suggested_actions == []
        assert explanation.severity == Severity.INFO
        assert explanation.confidence == 0.0
        assert explanation.related_patterns == []
        assert explanation.model == ""
        assert explanation.created_at is not None

    def test_full_creation(self) -> None:
        """Test creating explanation with all fields."""
        explanation = Explanation(
            explanation_type=ExplanationType.NOVELTY,
            summary="This is an unusual pattern",
            root_cause="Memory leak in application",
            suggested_actions=["Restart service", "Check memory usage"],
            severity=Severity.HIGH,
            confidence=0.85,
            related_patterns=["memory_leak", "oom"],
            model="llama3.2",
            prompt_tokens=100,
            completion_tokens=50,
            response_time_seconds=1.5,
            metadata={"cluster_id": "123"},
        )

        assert explanation.explanation_type == ExplanationType.NOVELTY
        assert explanation.summary == "This is an unusual pattern"
        assert explanation.root_cause == "Memory leak in application"
        assert len(explanation.suggested_actions) == 2
        assert explanation.severity == Severity.HIGH
        assert explanation.confidence == 0.85
        assert explanation.model == "llama3.2"

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        explanation = Explanation(
            explanation_type=ExplanationType.ERROR_ANALYSIS,
            summary="Test summary",
            root_cause="Test cause",
            suggested_actions=["Action 1"],
            severity=Severity.CRITICAL,
            confidence=0.9,
            model="test-model",
            prompt_tokens=50,
            completion_tokens=25,
            response_time_seconds=0.5,
        )

        result = explanation.to_dict()

        assert result["explanation_type"] == "error_analysis"
        assert result["summary"] == "Test summary"
        assert result["root_cause"] == "Test cause"
        assert result["severity"] == "critical"
        assert result["confidence"] == 0.9
        assert result["model"] == "test-model"
        assert result["prompt_tokens"] == 50
        assert result["completion_tokens"] == 25
        assert result["response_time_seconds"] == 0.5


# ============================================================================
# ExplanationType and Severity Enum Tests
# ============================================================================


class TestEnums:
    """Tests for enumeration types."""

    def test_explanation_type_values(self) -> None:
        """Test ExplanationType enum values."""
        assert ExplanationType.CLUSTER.value == "cluster"
        assert ExplanationType.NOVELTY.value == "novelty"
        assert ExplanationType.ERROR_ANALYSIS.value == "error_analysis"
        assert ExplanationType.ROOT_CAUSE.value == "root_cause"
        assert ExplanationType.SUMMARY.value == "summary"

    def test_severity_values(self) -> None:
        """Test Severity enum values."""
        assert Severity.CRITICAL.value == "critical"
        assert Severity.HIGH.value == "high"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.LOW.value == "low"
        assert Severity.INFO.value == "info"


# ============================================================================
# MockLLMProvider Tests
# ============================================================================


class TestMockLLMProvider:
    """Tests for MockLLMProvider."""

    def test_initialization(self) -> None:
        """Test mock provider initialization."""
        provider = MockLLMProvider(
            model="test-model",
            response_delay=0.0,
            fail_rate=0.0,
        )

        assert provider.name == "mock"
        assert provider.model == "test-model"
        assert provider.is_available() is True

    def test_generate_returns_json(self) -> None:
        """Test that generate returns valid JSON."""
        provider = MockLLMProvider(model="test", response_delay=0.0)

        response, prompt_tokens, completion_tokens = provider.generate(
            "Test prompt",
            temperature=0.1,
        )

        # Should be valid JSON
        data = json.loads(response)
        assert "summary" in data
        assert "root_cause" in data
        assert "severity" in data
        assert "suggested_actions" in data
        assert "confidence" in data

        # Should have token counts
        assert prompt_tokens > 0
        assert completion_tokens > 0

    def test_generate_severity_based_on_prompt(self) -> None:
        """Test that severity is based on prompt content."""
        provider = MockLLMProvider(model="test", response_delay=0.0)

        # Error prompt should get high severity
        response, _, _ = provider.generate("This is an error message")
        data = json.loads(response)
        assert data["severity"] == "high"

        # Warning prompt
        response, _, _ = provider.generate("This is a warning message")
        data = json.loads(response)
        assert data["severity"] == "medium"

        # Info prompt
        response, _, _ = provider.generate("This is an info message")
        data = json.loads(response)
        assert data["severity"] == "low"

    def test_generate_with_failure(self) -> None:
        """Test generation with simulated failure."""
        provider = MockLLMProvider(
            model="test",
            response_delay=0.0,
            fail_rate=1.0,  # Always fail
        )

        with pytest.raises(LLMError) as exc_info:
            provider.generate("Test prompt")

        assert "Simulated failure" in str(exc_info.value)

    def test_is_available(self) -> None:
        """Test that mock provider is always available."""
        provider = MockLLMProvider()
        assert provider.is_available() is True


# ============================================================================
# OllamaProvider Tests
# ============================================================================


class TestOllamaProvider:
    """Tests for OllamaProvider."""

    def test_initialization(self) -> None:
        """Test Ollama provider initialization."""
        provider = OllamaProvider(
            model="llama3.2",
            base_url="http://localhost:11434",
            timeout=60,
            max_retries=3,
        )

        assert provider.name == "ollama"
        assert provider.model == "llama3.2"

    @patch("urllib.request.urlopen")
    def test_generate_success(self, mock_urlopen: MagicMock) -> None:
        """Test successful generation with mocked HTTP."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "response": '{"summary": "Test", "confidence": 0.9}',
                "prompt_eval_count": 100,
                "eval_count": 50,
            }
        ).encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        provider = OllamaProvider(model="llama3.2")
        response, prompt_tokens, completion_tokens = provider.generate("Test prompt")

        assert '{"summary": "Test"' in response
        assert prompt_tokens == 100
        assert completion_tokens == 50

    @patch("urllib.request.urlopen")
    def test_generate_retry_on_failure(self, mock_urlopen: MagicMock) -> None:
        """Test retry logic on failure."""
        import urllib.error

        # First two calls fail, third succeeds
        mock_urlopen.side_effect = [
            urllib.error.URLError("Connection refused"),
            urllib.error.URLError("Connection refused"),
            MagicMock(
                read=lambda: json.dumps({"response": "success"}).encode("utf-8"),
                __enter__=lambda s: s,
                __exit__=lambda *_args: False,
            ),
        ]

        provider = OllamaProvider(model="test", max_retries=3)
        response, _, _ = provider.generate("Test")

        assert response == "success"
        assert mock_urlopen.call_count == 3

    @patch("urllib.request.urlopen")
    def test_generate_all_retries_fail(self, mock_urlopen: MagicMock) -> None:
        """Test that all retries failing raises error."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        provider = OllamaProvider(model="test", max_retries=2)

        with pytest.raises(LLMError) as exc_info:
            provider.generate("Test")

        assert "Failed after 2 attempts" in str(exc_info.value)

    @patch("urllib.request.urlopen")
    def test_is_available_true(self, mock_urlopen: MagicMock) -> None:
        """Test is_available when Ollama is running."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {"models": [{"name": "llama3.2:latest"}]}
        ).encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        provider = OllamaProvider(model="llama3.2")
        assert provider.is_available() is True

    @patch("urllib.request.urlopen")
    def test_is_available_false(self, mock_urlopen: MagicMock) -> None:
        """Test is_available when Ollama is not running."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        provider = OllamaProvider(model="llama3.2")
        assert provider.is_available() is False


# ============================================================================
# LLMService Tests
# ============================================================================


class TestLLMService:
    """Tests for LLMService high-level API."""

    def test_from_config_mock(self, llm_config: LLMConfig) -> None:
        """Test creating service with mock provider."""
        service = LLMService.from_config(config=llm_config, use_mock=True)

        assert service.provider.name == "mock"
        assert service.temperature == llm_config.temperature

    def test_from_config_ollama(self, llm_config: LLMConfig) -> None:
        """Test creating service with Ollama provider."""
        service = LLMService.from_config(config=llm_config, use_mock=False)

        assert service.provider.name == "ollama"
        assert service.provider.model == "llama3.2"

    def test_from_config_default(self) -> None:
        """Test creating service with default config."""
        service = LLMService.from_config(use_mock=True)

        assert service.provider is not None
        assert service.provider.name == "mock"

    def test_explain_cluster(
        self,
        llm_config: LLMConfig,
        mock_cluster_summary: MagicMock,
    ) -> None:
        """Test cluster explanation generation."""
        service = LLMService.from_config(config=llm_config, use_mock=True)

        explanation = service.explain_cluster(mock_cluster_summary)

        assert explanation.explanation_type == ExplanationType.CLUSTER
        assert explanation.summary != ""
        assert explanation.confidence > 0
        assert "cluster_id" in explanation.metadata
        assert explanation.model == "mock-model"

    def test_explain_cluster_with_messages(
        self,
        llm_config: LLMConfig,
        mock_cluster_summary: MagicMock,
    ) -> None:
        """Test cluster explanation with explicit messages."""
        service = LLMService.from_config(config=llm_config, use_mock=True)

        messages = ["Custom message 1", "Custom message 2"]
        explanation = service.explain_cluster(mock_cluster_summary, messages=messages)

        assert explanation.explanation_type == ExplanationType.CLUSTER
        assert explanation.summary != ""

    def test_explain_novelty(
        self,
        llm_config: LLMConfig,
        mock_novelty_score: MagicMock,
    ) -> None:
        """Test novelty explanation generation."""
        service = LLMService.from_config(config=llm_config, use_mock=True)

        explanation = service.explain_novelty(
            mock_novelty_score,
            threshold=0.7,
            n_reference=100,
        )

        assert explanation.explanation_type == ExplanationType.NOVELTY
        assert explanation.summary != ""
        assert "novelty_score" in explanation.metadata
        assert explanation.metadata["novelty_score"] == 0.85

    def test_explain_novelty_with_message(
        self,
        llm_config: LLMConfig,
        mock_novelty_score: MagicMock,
    ) -> None:
        """Test novelty explanation with explicit message."""
        service = LLMService.from_config(config=llm_config, use_mock=True)

        explanation = service.explain_novelty(
            mock_novelty_score,
            message="Custom novelty message",
            threshold=0.8,
        )

        assert explanation.explanation_type == ExplanationType.NOVELTY

    def test_explain_error(
        self,
        llm_config: LLMConfig,
        mock_log_record: MagicMock,
    ) -> None:
        """Test error explanation generation."""
        service = LLMService.from_config(config=llm_config, use_mock=True)

        explanation = service.explain_error(mock_log_record)

        assert explanation.explanation_type == ExplanationType.ERROR_ANALYSIS
        assert explanation.summary != ""
        assert "record_id" in explanation.metadata

    def test_explain_error_with_context(
        self,
        llm_config: LLMConfig,
        mock_log_record: MagicMock,
    ) -> None:
        """Test error explanation with additional context."""
        service = LLMService.from_config(config=llm_config, use_mock=True)

        context = {"stack_trace": "at line 42", "environment": "production"}
        explanation = service.explain_error(mock_log_record, context=context)

        assert explanation.explanation_type == ExplanationType.ERROR_ANALYSIS

    def test_generate_summary(self, llm_config: LLMConfig) -> None:
        """Test summary generation."""
        service = LLMService.from_config(config=llm_config, use_mock=True)

        explanation = service.generate_summary(
            total_logs=1000,
            n_clusters=5,
            n_novel=3,
            cluster_summaries=["Cluster 1 summary", "Cluster 2 summary"],
            novelty_summaries=["Novel pattern 1"],
        )

        assert explanation.explanation_type == ExplanationType.SUMMARY
        assert explanation.summary != ""
        assert explanation.metadata["total_logs"] == 1000
        assert explanation.metadata["n_clusters"] == 5
        assert explanation.metadata["n_novel"] == 3

    def test_stats_tracking(
        self,
        llm_config: LLMConfig,
        mock_cluster_summary: MagicMock,
    ) -> None:
        """Test statistics tracking."""
        service = LLMService.from_config(config=llm_config, use_mock=True)

        service.explain_cluster(mock_cluster_summary)
        service.explain_cluster(mock_cluster_summary)

        stats = service.get_stats()
        assert stats.total_requests == 2
        assert stats.successful_requests == 2
        assert stats.total_tokens_used > 0

    def test_reset_stats(
        self,
        llm_config: LLMConfig,
        mock_cluster_summary: MagicMock,
    ) -> None:
        """Test statistics reset."""
        service = LLMService.from_config(config=llm_config, use_mock=True)

        service.explain_cluster(mock_cluster_summary)
        assert service.get_stats().total_requests == 1

        service.reset_stats()
        assert service.get_stats().total_requests == 0

    def test_is_available(self, llm_config: LLMConfig) -> None:
        """Test availability check."""
        service = LLMService.from_config(config=llm_config, use_mock=True)

        assert service.is_available() is True


# ============================================================================
# Prompt Template Tests
# ============================================================================


class TestPromptTemplates:
    """Tests for prompt templates."""

    def test_cluster_prompt_formatting(self) -> None:
        """Test cluster prompt template formatting."""
        prompt = CLUSTER_EXPLANATION_PROMPT.format(
            n_logs=5,
            log_messages="- Message 1\n- Message 2",
            cluster_size=100,
            common_level="ERROR",
            time_range="2026-01-07T10:00:00Z to 2026-01-07T11:00:00Z",
        )

        assert "5 similar log messages" in prompt
        assert "Message 1" in prompt
        assert "Cluster Size: 100" in prompt
        assert "Common Log Level: ERROR" in prompt
        assert "JSON format" in prompt

    def test_novelty_prompt_formatting(self) -> None:
        """Test novelty prompt template formatting."""
        prompt = NOVELTY_EXPLANATION_PROMPT.format(
            log_message="Unusual pattern detected",
            novelty_score=0.85,
            threshold=0.7,
            n_reference=100,
        )

        assert "Unusual pattern detected" in prompt
        assert "0.85" in prompt
        assert "threshold: 0.7" in prompt
        assert "100 known patterns" in prompt

    def test_error_prompt_formatting(self) -> None:
        """Test error prompt template formatting."""
        prompt = ERROR_ANALYSIS_PROMPT.format(
            error_message="NullPointerException",
            log_level="ERROR",
            source="app.log",
            timestamp="2026-01-07T12:00:00Z",
            context='{"key": "value"}',
        )

        assert "NullPointerException" in prompt
        assert "ERROR" in prompt
        assert "app.log" in prompt
        assert "2026-01-07T12:00:00Z" in prompt

    def test_summary_prompt_formatting(self) -> None:
        """Test summary prompt template formatting."""
        prompt = SUMMARY_PROMPT.format(
            total_logs=1000,
            n_clusters=5,
            n_novel=3,
            cluster_summaries="- Cluster 1\n- Cluster 2",
            novelty_summaries="- Novel 1",
        )

        assert "Total Logs Analyzed: 1000" in prompt
        assert "Clusters Found: 5" in prompt
        assert "Novel Patterns Detected: 3" in prompt


# ============================================================================
# Response Parsing Tests
# ============================================================================


class TestResponseParsing:
    """Tests for LLM response parsing."""

    def test_parse_valid_json(self, llm_config: LLMConfig) -> None:
        """Test parsing valid JSON response."""
        service = LLMService.from_config(config=llm_config, use_mock=True)

        # The mock provider returns valid JSON
        response = '{"summary": "Test", "severity": "high", "confidence": 0.9}'
        explanation = service._parse_response(
            response_text=response,
            explanation_type=ExplanationType.CLUSTER,
            model="test",
            prompt_tokens=10,
            completion_tokens=5,
            response_time=0.5,
        )

        assert explanation.summary == "Test"
        assert explanation.severity == Severity.HIGH
        assert explanation.confidence == 0.9

    def test_parse_json_in_markdown(self, llm_config: LLMConfig) -> None:
        """Test parsing JSON wrapped in markdown code block."""
        service = LLMService.from_config(config=llm_config, use_mock=True)

        response = """Here's the analysis:
```json
{"summary": "Test", "severity": "medium", "confidence": 0.8}
```
"""
        explanation = service._parse_response(
            response_text=response,
            explanation_type=ExplanationType.CLUSTER,
            model="test",
            prompt_tokens=10,
            completion_tokens=5,
            response_time=0.5,
        )

        assert explanation.summary == "Test"

    def test_parse_invalid_json(self, llm_config: LLMConfig) -> None:
        """Test parsing invalid JSON raises error."""
        service = LLMService.from_config(config=llm_config, use_mock=True)

        response = "This is not JSON at all"

        with pytest.raises(LLMError) as exc_info:
            service._parse_response(
                response_text=response,
                explanation_type=ExplanationType.CLUSTER,
                model="test",
                prompt_tokens=10,
                completion_tokens=5,
                response_time=0.5,
            )

        assert "No JSON object found" in str(exc_info.value)

    def test_parse_invalid_severity(self, llm_config: LLMConfig) -> None:
        """Test parsing with invalid severity defaults to INFO."""
        service = LLMService.from_config(config=llm_config, use_mock=True)

        response = '{"summary": "Test", "severity": "invalid_severity", "confidence": 0.5}'
        explanation = service._parse_response(
            response_text=response,
            explanation_type=ExplanationType.CLUSTER,
            model="test",
            prompt_tokens=10,
            completion_tokens=5,
            response_time=0.5,
        )

        assert explanation.severity == Severity.INFO


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestLLMErrorHandling:
    """Tests for error handling in LLM operations."""

    def test_provider_error_creation(self) -> None:
        """Test LLMError.provider_error creation."""
        error = LLMError.provider_error("ollama", "Connection refused")

        assert "ollama" in str(error)
        assert "Connection refused" in str(error)
        assert error.is_retryable is True

    def test_rate_limited_error(self) -> None:
        """Test LLMError.rate_limited creation."""
        error = LLMError.rate_limited("ollama", retry_after=30.0)

        assert "Rate limited" in str(error)
        assert error.context["retry_after"] == 30.0
        assert error.is_retryable is True

    def test_context_too_long_error(self) -> None:
        """Test LLMError.context_too_long creation."""
        error = LLMError.context_too_long(tokens=10000, max_tokens=4096)

        assert "10000" in str(error)
        assert "4096" in str(error)
        assert error.is_retryable is False

    def test_invalid_response_error(self) -> None:
        """Test LLMError.invalid_response creation."""
        error = LLMError.invalid_response("Malformed JSON")

        assert "Invalid LLM response" in str(error)
        assert "Malformed JSON" in str(error)
        assert error.is_retryable is True

    def test_stats_updated_on_error(self) -> None:
        """Test that stats are updated when errors occur."""
        provider = MockLLMProvider(fail_rate=1.0, response_delay=0.0)
        service = LLMService(provider=provider, temperature=0.1)

        with pytest.raises(LLMError):
            service._generate_explanation(
                prompt="Test",
                explanation_type=ExplanationType.CLUSTER,
            )

        stats = service.get_stats()
        assert stats.total_requests == 1
        assert stats.failed_requests == 1
        assert stats.successful_requests == 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestLLMIntegration:
    """Integration tests for LLM module."""

    def test_full_cluster_workflow(
        self,
        llm_config: LLMConfig,
        mock_cluster_summary: MagicMock,
    ) -> None:
        """Test complete cluster explanation workflow."""
        service = LLMService.from_config(config=llm_config, use_mock=True)

        # Generate explanation
        explanation = service.explain_cluster(mock_cluster_summary)

        # Verify explanation structure
        assert explanation.id is not None
        assert explanation.explanation_type == ExplanationType.CLUSTER
        assert len(explanation.suggested_actions) > 0
        assert explanation.confidence > 0

        # Verify serialization
        explanation_dict = explanation.to_dict()
        assert isinstance(explanation_dict, dict)
        assert "summary" in explanation_dict
        assert "severity" in explanation_dict

        # Verify stats
        stats = service.get_stats()
        assert stats.total_requests == 1
        assert stats.successful_requests == 1

    def test_multiple_explanation_types(
        self,
        llm_config: LLMConfig,
        mock_cluster_summary: MagicMock,
        mock_novelty_score: MagicMock,
        mock_log_record: MagicMock,
    ) -> None:
        """Test generating multiple types of explanations."""
        service = LLMService.from_config(config=llm_config, use_mock=True)

        cluster_exp = service.explain_cluster(mock_cluster_summary)
        novelty_exp = service.explain_novelty(mock_novelty_score)
        error_exp = service.explain_error(mock_log_record)
        summary_exp = service.generate_summary(
            total_logs=100,
            n_clusters=3,
            n_novel=2,
            cluster_summaries=["C1", "C2"],
            novelty_summaries=["N1"],
        )

        assert cluster_exp.explanation_type == ExplanationType.CLUSTER
        assert novelty_exp.explanation_type == ExplanationType.NOVELTY
        assert error_exp.explanation_type == ExplanationType.ERROR_ANALYSIS
        assert summary_exp.explanation_type == ExplanationType.SUMMARY

        stats = service.get_stats()
        assert stats.total_requests == 4
        assert stats.successful_requests == 4
