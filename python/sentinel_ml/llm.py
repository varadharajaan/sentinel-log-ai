"""
LLM integration for log explanation and analysis.

This module provides LLM-powered explanations for log clusters and novel patterns,
using local models via Ollama or cloud providers.

Design Patterns:
- Strategy Pattern: Pluggable LLM providers (Ollama, OpenAI, Mock)
- Factory Pattern: Service creation with configuration
- Template Pattern: Prompt templates for different explanation types
- Builder Pattern: Flexible prompt construction

SOLID Principles:
- Single Responsibility: Each class handles one concern
- Open/Closed: Extensible via LLMProvider interface
- Liskov Substitution: All providers implement same interface
- Interface Segregation: Minimal interfaces for specific capabilities
- Dependency Inversion: Depends on abstractions not implementations
"""

from __future__ import annotations

import json
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeAlias

from sentinel_ml.config import LLMConfig, get_config
from sentinel_ml.exceptions import LLMError
from sentinel_ml.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sentinel_ml.clustering import ClusterSummary
    from sentinel_ml.models import LogRecord
    from sentinel_ml.novelty import NoveltyScore

logger = get_logger(__name__)

# Type aliases
PromptTemplate: TypeAlias = str


class ExplanationType(str, Enum):
    """Types of explanations the LLM can generate."""

    CLUSTER = "cluster"
    NOVELTY = "novelty"
    ERROR_ANALYSIS = "error_analysis"
    ROOT_CAUSE = "root_cause"
    SUMMARY = "summary"


class Severity(str, Enum):
    """Severity levels for log explanations."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class LLMStats:
    """Statistics for LLM operations."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_used: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    avg_response_time_seconds: float = 0.0
    last_request_time: datetime | None = None

    def record_request(
        self,
        success: bool,
        response_time: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Record a request for statistics tracking."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens_used = self.total_prompt_tokens + self.total_completion_tokens

        # Update running average
        if self.successful_requests > 0:
            prev_avg = self.avg_response_time_seconds
            prev_count = self.successful_requests - 1 if success else self.successful_requests
            if success:
                self.avg_response_time_seconds = (
                    prev_avg * prev_count + response_time
                ) / self.successful_requests

        self.last_request_time = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary for logging."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": round(self.successful_requests / self.total_requests, 4)
            if self.total_requests > 0
            else 0.0,
            "total_tokens_used": self.total_tokens_used,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "avg_response_time_seconds": round(self.avg_response_time_seconds, 3),
            "last_request_time": (
                self.last_request_time.isoformat() if self.last_request_time else None
            ),
        }


@dataclass
class Explanation:
    """
    Structured explanation from the LLM.

    Contains the explanation text, confidence score, and metadata
    for understanding and acting on log patterns.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    explanation_type: ExplanationType = ExplanationType.CLUSTER
    summary: str = ""
    root_cause: str | None = None
    suggested_actions: list[str] = field(default_factory=list)
    severity: Severity = Severity.INFO
    confidence: float = 0.0
    related_patterns: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    response_time_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "explanation_type": self.explanation_type.value,
            "summary": self.summary,
            "root_cause": self.root_cause,
            "suggested_actions": self.suggested_actions,
            "severity": self.severity.value,
            "confidence": round(self.confidence, 4),
            "related_patterns": self.related_patterns,
            "created_at": self.created_at.isoformat(),
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "response_time_seconds": round(self.response_time_seconds, 3),
            "metadata": self.metadata,
        }


# ============================================================================
# Prompt Templates
# ============================================================================

CLUSTER_EXPLANATION_PROMPT = """You are an expert log analyst helping on-call engineers understand log patterns.

Analyze the following cluster of {n_logs} similar log messages:

**Representative Log Messages:**
{log_messages}

**Cluster Metadata:**
- Cluster Size: {cluster_size}
- Common Log Level: {common_level}
- Time Range: {time_range}

Provide a structured analysis in the following JSON format:
{{
    "summary": "Brief 1-2 sentence description of what this log pattern represents",
    "root_cause": "Most likely root cause of these log messages",
    "severity": "critical|high|medium|low|info",
    "suggested_actions": ["action 1", "action 2", "action 3"],
    "confidence": 0.0 to 1.0
}}

Important:
- Be concise and actionable
- Focus on what an on-call engineer needs to know
- Suggest specific debugging steps
- Rate severity based on production impact

Respond with ONLY the JSON object, no additional text."""

NOVELTY_EXPLANATION_PROMPT = """You are an expert log analyst helping on-call engineers identify unusual patterns.

A novel log pattern has been detected that doesn't match known patterns:

**Novel Log Message:**
{log_message}

**Novelty Score:** {novelty_score:.2f} (higher = more unusual, threshold: {threshold})

**Context:**
- Detection Algorithm: k-NN density-based
- This log is significantly different from the baseline of {n_reference} known patterns

Provide a structured analysis in the following JSON format:
{{
    "summary": "Brief explanation of what makes this log unusual",
    "root_cause": "Possible cause for this unexpected pattern",
    "severity": "critical|high|medium|low|info",
    "suggested_actions": ["action 1", "action 2", "action 3"],
    "confidence": 0.0 to 1.0
}}

Important:
- Explain WHY this log is unusual compared to normal patterns
- Consider if this might indicate a new type of error or attack
- Suggest investigation steps
- Rate severity based on potential impact

Respond with ONLY the JSON object, no additional text."""

ERROR_ANALYSIS_PROMPT = """You are an expert log analyst diagnosing errors in production systems.

Analyze the following error log:

**Error Message:**
{error_message}

**Error Details:**
- Log Level: {log_level}
- Source: {source}
- Timestamp: {timestamp}
- Additional Context: {context}

Provide a structured analysis in the following JSON format:
{{
    "summary": "Brief explanation of the error",
    "root_cause": "Most likely cause of this error",
    "severity": "critical|high|medium|low|info",
    "suggested_actions": ["action 1", "action 2", "action 3"],
    "related_patterns": ["pattern 1", "pattern 2"],
    "confidence": 0.0 to 1.0
}}

Important:
- Focus on actionable debugging steps
- Consider common causes for this type of error
- Suggest both immediate actions and long-term fixes

Respond with ONLY the JSON object, no additional text."""

SUMMARY_PROMPT = """You are an expert log analyst creating executive summaries for incident reviews.

Summarize the following log analysis results:

**Overview:**
- Total Logs Analyzed: {total_logs}
- Clusters Found: {n_clusters}
- Novel Patterns Detected: {n_novel}

**Top Clusters:**
{cluster_summaries}

**Novel Patterns:**
{novelty_summaries}

Provide a structured summary in the following JSON format:
{{
    "summary": "Executive summary of the log analysis (2-3 sentences)",
    "key_findings": ["finding 1", "finding 2", "finding 3"],
    "severity": "critical|high|medium|low|info",
    "suggested_actions": ["action 1", "action 2", "action 3"],
    "confidence": 0.0 to 1.0
}}

Important:
- Focus on the most important patterns requiring attention
- Prioritize by severity and frequency
- Make recommendations actionable

Respond with ONLY the JSON object, no additional text."""


# ============================================================================
# LLM Provider Interface (Strategy Pattern)
# ============================================================================


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Implements the Strategy pattern for pluggable LLM backends.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""
        ...

    @property
    @abstractmethod
    def model(self) -> str:
        """Return the model name."""
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> tuple[str, int, int]:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: The prompt to complete.
            temperature: Sampling temperature (0.0-1.0).
            max_tokens: Maximum tokens to generate (None for model default).

        Returns:
            Tuple of (response_text, prompt_tokens, completion_tokens).

        Raises:
            LLMError: If generation fails.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and configured."""
        ...


class OllamaProvider(LLMProvider):
    """
    Ollama LLM provider for local model inference.

    Uses the official Ollama Python client for generating completions
    with locally-hosted models.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize the Ollama provider.

        Args:
            model: Model name (e.g., llama3.2, mistral, codellama).
            base_url: Ollama server base URL.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts for failed requests.
        """
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._client: Any = None

        logger.info(
            "ollama_provider_initialized",
            model=model,
            base_url=base_url,
            timeout=timeout,
        )

    def _get_client(self) -> Any:
        """Get or create the Ollama client."""
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self._base_url, timeout=self._timeout)
            except ImportError as e:
                raise LLMError.provider_error(
                    "ollama",
                    "ollama package not installed. Run: pip install ollama",
                ) from e
        return self._client

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "ollama"

    @property
    def model(self) -> str:
        """Return the model name."""
        return self._model

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> tuple[str, int, int]:
        """
        Generate a completion using Ollama.

        Args:
            prompt: The prompt to complete.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Tuple of (response_text, prompt_tokens, completion_tokens).

        Raises:
            LLMError: If generation fails.
        """
        options: dict[str, Any] = {
            "temperature": temperature,
        }

        if max_tokens is not None:
            options["num_predict"] = max_tokens

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                client = self._get_client()
                response = client.generate(
                    model=self._model,
                    prompt=prompt,
                    options=options,
                )

                response_text = response.get("response", "")
                prompt_tokens = response.get("prompt_eval_count", 0)
                completion_tokens = response.get("eval_count", 0)

                logger.debug(
                    "ollama_generation_complete",
                    model=self._model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    response_length=len(response_text),
                )

                return response_text, prompt_tokens, completion_tokens

            except Exception as e:
                last_error = e
                logger.warning(
                    "ollama_request_failed",
                    attempt=attempt + 1,
                    max_retries=self._max_retries,
                    error=str(e),
                )
                if attempt < self._max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff

        raise LLMError.provider_error(
            "ollama",
            f"Failed after {self._max_retries} attempts: {last_error}",
        )

    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            client = self._get_client()
            models_response = client.list()
            models = [m.get("name", "") for m in models_response.get("models", [])]
            # Check if our model is available
            return any(self._model in m for m in models)
        except Exception:
            return False


class MockLLMProvider(LLMProvider):
    """
    Mock LLM provider for testing.

    Generates deterministic responses for testing purposes.
    """

    def __init__(
        self,
        model: str = "mock-model",
        response_delay: float = 0.1,
        fail_rate: float = 0.0,
    ) -> None:
        """
        Initialize the mock provider.

        Args:
            model: Model name for identification.
            response_delay: Simulated response delay in seconds.
            fail_rate: Probability of simulated failure (0.0-1.0).
        """
        self._model = model
        self._response_delay = response_delay
        self._fail_rate = fail_rate
        self._request_count = 0

        logger.debug(
            "mock_llm_provider_initialized",
            model=model,
            response_delay=response_delay,
            fail_rate=fail_rate,
        )

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "mock"

    @property
    def model(self) -> str:
        """Return the model name."""
        return self._model

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,  # noqa: ARG002
        max_tokens: int | None = None,  # noqa: ARG002
    ) -> tuple[str, int, int]:
        """
        Generate a mock response.

        Returns a deterministic JSON response based on the prompt content.
        """
        import random

        self._request_count += 1

        # Simulate delay
        if self._response_delay > 0:
            time.sleep(self._response_delay)

        # Simulate failures
        if self._fail_rate > 0 and random.random() < self._fail_rate:
            raise LLMError.provider_error("mock", "Simulated failure")

        # Generate deterministic response based on prompt content
        severity = "medium"
        if "error" in prompt.lower() or "critical" in prompt.lower():
            severity = "high"
        elif "warning" in prompt.lower():
            severity = "medium"
        elif "info" in prompt.lower():
            severity = "low"

        response = {
            "summary": "Mock analysis of the provided log patterns.",
            "root_cause": "This is a mock root cause analysis for testing purposes.",
            "severity": severity,
            "suggested_actions": [
                "Check application logs for more context",
                "Review recent deployments",
                "Monitor system metrics",
            ],
            "related_patterns": ["connection_error", "timeout"],
            "confidence": 0.85,
        }

        response_text = json.dumps(response)
        # Estimate tokens (rough approximation)
        prompt_tokens = len(prompt.split()) * 2
        completion_tokens = len(response_text.split()) * 2

        logger.debug(
            "mock_llm_generation_complete",
            request_count=self._request_count,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        return response_text, prompt_tokens, completion_tokens

    def is_available(self) -> bool:
        """Mock provider is always available."""
        return True


# ============================================================================
# LLM Service (High-Level API)
# ============================================================================


@dataclass
class LLMService:
    """
    High-level service for LLM-powered log explanations.

    Provides a unified interface for generating explanations with:
    - Configurable LLM providers
    - Prompt template management
    - Response parsing and validation
    - Statistics tracking

    Usage:
        service = LLMService.from_config()
        explanation = service.explain_cluster(cluster_summary, log_records)
        print(explanation.summary)
    """

    provider: LLMProvider
    temperature: float = 0.1
    stats: LLMStats = field(default_factory=LLMStats)

    @classmethod
    def from_config(
        cls,
        config: LLMConfig | None = None,
        use_mock: bool = False,
    ) -> LLMService:
        """
        Create an LLMService from configuration.

        Args:
            config: LLM configuration. Uses default if None.
            use_mock: If True, use mock provider regardless of config.

        Returns:
            Configured LLMService instance.
        """
        config = config or get_config().llm

        if use_mock:
            provider: LLMProvider = MockLLMProvider(model="mock-model")
            logger.info("llm_service_created_with_mock_provider")
        elif config.provider == "ollama":
            provider = OllamaProvider(
                model=config.model,
                base_url=config.base_url,
                timeout=config.timeout,
                max_retries=config.max_retries,
            )
            logger.info(
                "llm_service_created_with_ollama",
                model=config.model,
                base_url=config.base_url,
            )
        else:
            # Default to mock for unsupported providers
            logger.warning(
                "unsupported_llm_provider_using_mock",
                provider=config.provider,
            )
            provider = MockLLMProvider(model="fallback-mock")

        return cls(
            provider=provider,
            temperature=config.temperature,
        )

    def explain_cluster(
        self,
        summary: ClusterSummary,
        records: Sequence[LogRecord] | None = None,
        messages: list[str] | None = None,
    ) -> Explanation:
        """
        Generate an explanation for a log cluster.

        Args:
            summary: Cluster summary with representative samples.
            records: Optional log records for additional context.
            messages: Optional list of log messages (alternative to records).

        Returns:
            Structured explanation for the cluster.
        """
        # Build log messages for prompt
        if messages:
            log_messages = messages[:10]  # Limit to 10 for context length
        elif summary.representative_messages:
            log_messages = summary.representative_messages[:10]
        elif records:
            log_messages = [r.message for r in list(records)[:10]]
        else:
            log_messages = ["No log messages available"]

        # Format time range
        if summary.time_range_start and summary.time_range_end:
            time_range = (
                f"{summary.time_range_start.isoformat()} to {summary.time_range_end.isoformat()}"
            )
        else:
            time_range = "Unknown"

        # Build prompt
        prompt = CLUSTER_EXPLANATION_PROMPT.format(
            n_logs=len(log_messages),
            log_messages="\n".join(f"- {msg}" for msg in log_messages),
            cluster_size=summary.size,
            common_level=summary.common_level or "Unknown",
            time_range=time_range,
        )

        logger.info(
            "generating_cluster_explanation",
            cluster_id=summary.id,
            cluster_size=summary.size,
            n_messages=len(log_messages),
        )

        return self._generate_explanation(
            prompt=prompt,
            explanation_type=ExplanationType.CLUSTER,
            metadata={
                "cluster_id": summary.id,
                "cluster_size": summary.size,
            },
        )

    def explain_novelty(
        self,
        novelty_score: NoveltyScore,
        message: str | None = None,
        threshold: float = 0.7,
        n_reference: int = 0,
    ) -> Explanation:
        """
        Generate an explanation for a novel log pattern.

        Args:
            novelty_score: Novelty score object with score and metadata.
            message: Log message (uses novelty_score.message if not provided).
            threshold: Novelty threshold used for detection.
            n_reference: Number of reference patterns in baseline.

        Returns:
            Structured explanation for the novel pattern.
        """
        log_message = message or novelty_score.message or "Unknown message"

        prompt = NOVELTY_EXPLANATION_PROMPT.format(
            log_message=log_message,
            novelty_score=novelty_score.score,
            threshold=threshold,
            n_reference=n_reference,
        )

        logger.info(
            "generating_novelty_explanation",
            novelty_score=round(novelty_score.score, 4),
            threshold=threshold,
        )

        return self._generate_explanation(
            prompt=prompt,
            explanation_type=ExplanationType.NOVELTY,
            metadata={
                "novelty_score": novelty_score.score,
                "threshold": threshold,
                "index": novelty_score.index,
            },
        )

    def explain_error(
        self,
        record: LogRecord,
        context: dict[str, Any] | None = None,
    ) -> Explanation:
        """
        Generate an explanation for an error log.

        Args:
            record: Log record containing the error.
            context: Additional context for the error.

        Returns:
            Structured explanation for the error.
        """
        prompt = ERROR_ANALYSIS_PROMPT.format(
            error_message=record.message,
            log_level=record.level or "ERROR",
            source=record.source,
            timestamp=record.timestamp.isoformat() if record.timestamp else "Unknown",
            context=json.dumps(context or record.attrs, indent=2),
        )

        logger.info(
            "generating_error_explanation",
            source=record.source,
            level=record.level,
        )

        return self._generate_explanation(
            prompt=prompt,
            explanation_type=ExplanationType.ERROR_ANALYSIS,
            metadata={
                "record_id": record.id,
                "source": record.source,
                "level": record.level,
            },
        )

    def generate_summary(
        self,
        total_logs: int,
        n_clusters: int,
        n_novel: int,
        cluster_summaries: list[str],
        novelty_summaries: list[str],
    ) -> Explanation:
        """
        Generate an executive summary of log analysis.

        Args:
            total_logs: Total number of logs analyzed.
            n_clusters: Number of clusters found.
            n_novel: Number of novel patterns detected.
            cluster_summaries: Brief descriptions of top clusters.
            novelty_summaries: Brief descriptions of novel patterns.

        Returns:
            Structured executive summary.
        """
        prompt = SUMMARY_PROMPT.format(
            total_logs=total_logs,
            n_clusters=n_clusters,
            n_novel=n_novel,
            cluster_summaries="\n".join(f"- {s}" for s in cluster_summaries[:5]),
            novelty_summaries="\n".join(f"- {s}" for s in novelty_summaries[:5]),
        )

        logger.info(
            "generating_analysis_summary",
            total_logs=total_logs,
            n_clusters=n_clusters,
            n_novel=n_novel,
        )

        return self._generate_explanation(
            prompt=prompt,
            explanation_type=ExplanationType.SUMMARY,
            metadata={
                "total_logs": total_logs,
                "n_clusters": n_clusters,
                "n_novel": n_novel,
            },
        )

    def _generate_explanation(
        self,
        prompt: str,
        explanation_type: ExplanationType,
        metadata: dict[str, Any] | None = None,
    ) -> Explanation:
        """
        Generate and parse an explanation from the LLM.

        Args:
            prompt: The formatted prompt to send.
            explanation_type: Type of explanation being generated.
            metadata: Additional metadata to include.

        Returns:
            Parsed Explanation object.

        Raises:
            LLMError: If generation or parsing fails.
        """
        start_time = time.time()

        try:
            response_text, prompt_tokens, completion_tokens = self.provider.generate(
                prompt=prompt,
                temperature=self.temperature,
            )
            response_time = time.time() - start_time

            # Parse JSON response
            explanation = self._parse_response(
                response_text=response_text,
                explanation_type=explanation_type,
                model=self.provider.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                response_time=response_time,
                metadata=metadata,
            )

            # Update stats
            self.stats.record_request(
                success=True,
                response_time=response_time,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

            logger.info(
                "explanation_generated",
                explanation_type=explanation_type.value,
                model=self.provider.model,
                response_time_seconds=round(response_time, 3),
                confidence=explanation.confidence,
            )

            return explanation

        except LLMError:
            response_time = time.time() - start_time
            self.stats.record_request(success=False, response_time=response_time)
            raise

        except Exception as e:
            response_time = time.time() - start_time
            self.stats.record_request(success=False, response_time=response_time)
            logger.error(
                "explanation_generation_failed",
                explanation_type=explanation_type.value,
                error=str(e),
            )
            raise LLMError.provider_error(self.provider.name, str(e)) from e

    def _parse_response(
        self,
        response_text: str,
        explanation_type: ExplanationType,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        response_time: float,
        metadata: dict[str, Any] | None = None,
    ) -> Explanation:
        """
        Parse the LLM response into an Explanation object.

        Args:
            response_text: Raw response from the LLM.
            explanation_type: Type of explanation.
            model: Model name used.
            prompt_tokens: Number of prompt tokens.
            completion_tokens: Number of completion tokens.
            response_time: Response time in seconds.
            metadata: Additional metadata.

        Returns:
            Parsed Explanation object.

        Raises:
            LLMError: If parsing fails.
        """
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if not json_match:
                raise LLMError.invalid_response("No JSON object found in response")

            data = json.loads(json_match.group())

            # Parse severity
            severity_str = data.get("severity", "info").lower()
            try:
                severity = Severity(severity_str)
            except ValueError:
                severity = Severity.INFO

            return Explanation(
                explanation_type=explanation_type,
                summary=data.get("summary", ""),
                root_cause=data.get("root_cause"),
                suggested_actions=data.get("suggested_actions", []),
                severity=severity,
                confidence=float(data.get("confidence", 0.5)),
                related_patterns=data.get("related_patterns", []),
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                response_time_seconds=response_time,
                metadata=metadata or {},
            )

        except json.JSONDecodeError as e:
            logger.error(
                "llm_response_parse_failed",
                error=str(e),
                response_preview=response_text[:200],
            )
            raise LLMError.invalid_response(f"Invalid JSON: {e}") from e

    def get_stats(self) -> LLMStats:
        """Get current statistics."""
        return self.stats

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = LLMStats()
        logger.debug("llm_stats_reset")

    def is_available(self) -> bool:
        """Check if the LLM provider is available."""
        return self.provider.is_available()
