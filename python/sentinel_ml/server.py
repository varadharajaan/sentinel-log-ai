"""
gRPC server for the ML engine.

This server exposes the ML capabilities to the Go agent via gRPC.
"""

from __future__ import annotations

import signal
import sys
from concurrent import futures

import grpc

from sentinel_ml.config import Config, get_config
from sentinel_ml.logging import get_logger, setup_logging

logger = get_logger(__name__)


class MLServiceServicer:
    """
    Implementation of the ML gRPC service.

    This class handles all ML-related requests from the Go agent.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._embedding_model = None
        self._vector_store = None
        self._clusterer = None
        logger.info("MLServiceServicer initialized")

    # TODO: Implement gRPC methods once proto is compiled
    # These will be implemented in subsequent issues:
    # - Embed / EmbedStream
    # - Search
    # - Cluster
    # - DetectNovelty
    # - Explain
    # - Health


def create_server(config: Config | None = None) -> grpc.Server:
    """Create and configure the gRPC server."""
    config = config or get_config()

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=config.server.max_workers),
        options=[
            ("grpc.max_send_message_length", config.server.max_message_size),
            ("grpc.max_receive_message_length", config.server.max_message_size),
        ],
    )

    # TODO: Add servicer once proto is compiled
    # ml_pb2_grpc.add_MLServiceServicer_to_server(MLServiceServicer(config), server)

    server.add_insecure_port(f"{config.server.host}:{config.server.port}")

    return server


def serve(config: Config | None = None) -> None:
    """Start the gRPC server and block until shutdown."""
    config = config or get_config()
    setup_logging()

    server = create_server(config)
    server.start()

    logger.info(
        "ML gRPC server started",
        host=config.server.host,
        port=config.server.port,
    )

    # Handle graceful shutdown
    def shutdown_handler(_signum: int, _frame: object) -> None:
        logger.info("Shutting down server...")
        server.stop(grace=5)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    server.wait_for_termination()


def main() -> None:
    """Entry point for the ML server."""
    serve()


if __name__ == "__main__":
    main()
