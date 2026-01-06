"""
gRPC server for the ML engine.

This server exposes the ML capabilities to the Go agent via gRPC.
Implements the MLService as defined in proto/ml/v1/ml_service.proto.

Design Patterns:
- Facade Pattern: Simple interface to complex ML subsystems
- Service Locator: Lazy initialization of ML components
- Observer: Health monitoring

SOLID Principles:
- Single Responsibility: Server handles only gRPC concerns
- Dependency Inversion: Components injected via config
"""

from __future__ import annotations

import signal
import sys
import threading
import time
from concurrent import futures
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import grpc

from sentinel_ml.config import Config, get_config
from sentinel_ml.logging import get_logger, setup_logging
from sentinel_ml.models import LogRecord
from sentinel_ml.preprocessing import PreprocessingService

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


@dataclass
class ServerMetrics:
    """Metrics for the gRPC server."""

    requests_total: int = 0
    requests_success: int = 0
    requests_failed: int = 0
    records_processed: int = 0
    start_time: float = field(default_factory=time.time)
    last_request_time: float | None = None

    def record_request(self, success: bool, record_count: int = 0) -> None:
        """Record a request."""
        self.requests_total += 1
        if success:
            self.requests_success += 1
            self.records_processed += record_count
        else:
            self.requests_failed += 1
        self.last_request_time = time.time()

    @property
    def uptime_seconds(self) -> float:
        """Get server uptime."""
        return time.time() - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "requests_total": self.requests_total,
            "requests_success": self.requests_success,
            "requests_failed": self.requests_failed,
            "records_processed": self.records_processed,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "last_request_time": self.last_request_time,
        }


@dataclass
class ComponentHealth:
    """Health status of a component."""

    name: str
    healthy: bool
    message: str = ""


class MLServiceServicer:
    """
    Implementation of the ML gRPC service.

    This class handles all ML-related requests from the Go agent.
    Currently implements preprocessing; embedding/clustering/etc.
    will be added in subsequent milestones.
    """

    VERSION = "0.1.0"

    def __init__(self, config: Config) -> None:
        """
        Initialize the servicer.

        Args:
            config: Server configuration.
        """
        self.config = config
        self._preprocessing = PreprocessingService()
        self._metrics = ServerMetrics()
        self._lock = threading.Lock()

        # Lazy-loaded components (for future milestones)
        self._embedding_model = None
        self._vector_store = None
        self._clusterer = None
        self._llm_client = None

        logger.info(
            "ml_servicer_initialized",
            version=self.VERSION,
        )

    def preprocess_records(
        self,
        records: list[dict[str, Any]],
    ) -> list[LogRecord]:
        """
        Preprocess raw log records.

        Args:
            records: List of raw record dictionaries.

        Returns:
            List of preprocessed LogRecord objects.
        """
        log_records = []
        for data in records:
            record = LogRecord(
                id=data.get("id"),
                message=data.get("message", data.get("raw", "")),
                raw=data.get("raw", data.get("message", "")),
                normalized=data.get("normalized"),
                level=data.get("level"),
                source=data.get("source", "unknown"),
                timestamp=data.get("timestamp"),
                attrs=data.get("attrs", {}),
            )
            log_records.append(record)

        processed = self._preprocessing.preprocess_batch(log_records)

        with self._lock:
            self._metrics.record_request(True, len(processed))

        logger.info(
            "records_preprocessed",
            input_count=len(records),
            output_count=len(processed),
        )

        return processed

    def health_check(self, detailed: bool = False) -> dict[str, Any]:
        """
        Check service health.

        Args:
            detailed: Include component-level health.

        Returns:
            Health status dictionary.
        """
        components = []
        overall_healthy = True

        # Check preprocessing service
        components.append(
            ComponentHealth(
                name="preprocessing",
                healthy=True,
                message="Ready",
            )
        )

        # Check embedding model (future)
        if self._embedding_model is not None:
            components.append(
                ComponentHealth(
                    name="embedding_model",
                    healthy=True,
                    message="Model loaded",
                )
            )
        else:
            components.append(
                ComponentHealth(
                    name="embedding_model",
                    healthy=True,  # Not loaded is OK for M1
                    message="Not initialized (M2)",
                )
            )

        # Check vector store (future)
        if self._vector_store is not None:
            components.append(
                ComponentHealth(
                    name="vector_store",
                    healthy=True,
                    message="Store ready",
                )
            )
        else:
            components.append(
                ComponentHealth(
                    name="vector_store",
                    healthy=True,  # Not loaded is OK for M1
                    message="Not initialized (M2)",
                )
            )

        result: dict[str, Any] = {
            "healthy": overall_healthy,
            "version": self.VERSION,
        }

        if detailed:
            result["components"] = [
                {"name": c.name, "healthy": c.healthy, "message": c.message} for c in components
            ]
            result["metrics"] = self._metrics.to_dict()

        return result

    def get_metrics(self) -> dict[str, Any]:
        """Get server metrics."""
        with self._lock:
            return self._metrics.to_dict()


class GRPCServer:
    """
    High-level gRPC server wrapper.

    Provides lifecycle management, graceful shutdown, and health monitoring.
    """

    def __init__(
        self,
        config: Config | None = None,
        servicer: MLServiceServicer | None = None,
    ) -> None:
        """
        Initialize the server.

        Args:
            config: Server configuration.
            servicer: Custom servicer instance.
        """
        self.config = config or get_config()
        self.servicer = servicer or MLServiceServicer(self.config)
        self._server: grpc.Server | None = None
        self._shutdown_event = threading.Event()
        self._started = False

        logger.info(
            "grpc_server_created",
            host=self.config.server.host,
            port=self.config.server.port,
        )

    def _create_server(self) -> grpc.Server:
        """Create the underlying gRPC server."""
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.config.server.max_workers),
            options=[
                ("grpc.max_send_message_length", self.config.server.max_message_size),
                ("grpc.max_receive_message_length", self.config.server.max_message_size),
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 10000),
                ("grpc.keepalive_permit_without_calls", True),
            ],
        )

        # TODO: Add servicer once proto is compiled
        # ml_pb2_grpc.add_MLServiceServicer_to_server(self.servicer, server)

        address = f"{self.config.server.host}:{self.config.server.port}"
        server.add_insecure_port(address)

        return server

    def start(self) -> None:
        """Start the server."""
        if self._started:
            logger.warning("server_already_started")
            return

        self._server = self._create_server()
        self._server.start()
        self._started = True

        logger.info(
            "grpc_server_started",
            host=self.config.server.host,
            port=self.config.server.port,
            max_workers=self.config.server.max_workers,
        )

    def stop(self, grace: float = 5.0) -> None:
        """
        Stop the server gracefully.

        Args:
            grace: Grace period in seconds.
        """
        if not self._started or self._server is None:
            return

        logger.info("grpc_server_stopping", grace_period=grace)
        self._server.stop(grace=grace)
        self._started = False
        self._shutdown_event.set()
        logger.info("grpc_server_stopped")

    def wait_for_termination(self, timeout: float | None = None) -> bool:
        """
        Wait for the server to terminate.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            True if terminated, False if timeout.
        """
        if self._server is None:
            return True
        return self._shutdown_event.wait(timeout=timeout)

    def is_running(self) -> bool:
        """Check if the server is running."""
        return self._started

    def health_check(self, detailed: bool = False) -> dict[str, Any]:
        """Get health status."""
        return self.servicer.health_check(detailed)


def create_server(config: Config | None = None) -> GRPCServer:
    """
    Create a new gRPC server instance.

    Args:
        config: Server configuration.

    Returns:
        Configured GRPCServer instance.
    """
    return GRPCServer(config)


def serve(
    config: Config | None = None,
    shutdown_handler: Callable[[], None] | None = None,
) -> None:
    """
    Start the gRPC server and block until shutdown.

    Args:
        config: Server configuration.
        shutdown_handler: Optional callback on shutdown.
    """
    config = config or get_config()
    setup_logging()

    server = create_server(config)
    server.start()

    # Handle graceful shutdown
    def signal_handler(_signum: int, _frame: object) -> None:
        logger.info("shutdown_signal_received")
        server.stop(grace=5.0)
        if shutdown_handler:
            shutdown_handler()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Block until shutdown
        while server.is_running():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("keyboard_interrupt_received")
        server.stop(grace=5.0)


def main() -> None:
    """Entry point for the ML server."""
    logger.info("starting_ml_server")
    try:
        serve()
    except Exception as e:
        logger.error("server_failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
