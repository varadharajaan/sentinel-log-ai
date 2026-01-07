"""
Health check endpoint for alerting system.

Provides health status monitoring for the watch daemon
and all configured notifiers.
"""

from __future__ import annotations

import http.server
import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from sentinel_ml.alerting.base import BaseNotifier
    from sentinel_ml.alerting.watch import WatchDaemon

logger = structlog.get_logger(__name__)


class HealthStatus(str, Enum):
    """Health status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthConfig:
    """
    Configuration for health check endpoint.

    Attributes:
        host: Host to bind to.
        port: Port to listen on.
        path: URL path for health endpoint.
        include_details: Include component details in response.
        enabled: Whether health check is enabled.
    """

    host: str = "0.0.0.0"
    port: int = 8080
    path: str = "/health"
    include_details: bool = True
    enabled: bool = True


class HealthCheck:
    """
    Health check endpoint server.

    Provides HTTP endpoint for monitoring the alerting system health,
    including watch daemon status and notifier availability.
    """

    def __init__(
        self,
        config: HealthConfig,
        watch_daemon: WatchDaemon | None = None,
        notifiers: list[BaseNotifier] | None = None,
    ) -> None:
        """
        Initialize health check.

        Args:
            config: Health check configuration.
            watch_daemon: Watch daemon to monitor.
            notifiers: Notifiers to monitor.
        """
        self._config = config
        self._watch_daemon = watch_daemon
        self._notifiers = notifiers or []
        self._logger = logger.bind(component="health-check")

        self._server: http.server.HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._started_at: datetime | None = None

    def add_notifier(self, notifier: BaseNotifier) -> None:
        """Add a notifier to monitor."""
        self._notifiers.append(notifier)

    def set_watch_daemon(self, daemon: WatchDaemon) -> None:
        """Set the watch daemon to monitor."""
        self._watch_daemon = daemon

    def start(self) -> None:
        """Start the health check server."""
        if not self._config.enabled:
            self._logger.info("health_check_disabled")
            return

        handler = self._create_handler()
        self._server = http.server.HTTPServer(
            (self._config.host, self._config.port),
            handler,
        )

        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="health-check-server",
            daemon=True,
        )
        self._thread.start()
        self._started_at = datetime.now(tz=timezone.utc)

        self._logger.info(
            "health_check_started",
            host=self._config.host,
            port=self._config.port,
            path=self._config.path,
        )

    def stop(self) -> None:
        """Stop the health check server."""
        if self._server:
            self._server.shutdown()
            self._server = None

        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

        self._logger.info("health_check_stopped")

    def get_health(self) -> dict[str, Any]:
        """
        Get current health status.

        Returns:
            Health status dictionary.
        """
        components: dict[str, dict[str, Any]] = {}
        overall_status = HealthStatus.HEALTHY

        # Check watch daemon
        if self._watch_daemon:
            daemon_status = self._check_watch_daemon()
            components["watch_daemon"] = daemon_status
            if daemon_status["status"] == HealthStatus.UNHEALTHY.value:
                overall_status = HealthStatus.UNHEALTHY
            elif (
                daemon_status["status"] == HealthStatus.DEGRADED.value
                and overall_status == HealthStatus.HEALTHY
            ):
                overall_status = HealthStatus.DEGRADED

        # Check notifiers
        if self._notifiers:
            notifier_status = self._check_notifiers()
            components["notifiers"] = notifier_status
            if notifier_status["status"] == HealthStatus.UNHEALTHY.value:
                overall_status = HealthStatus.UNHEALTHY
            elif (
                notifier_status["status"] == HealthStatus.DEGRADED.value
                and overall_status == HealthStatus.HEALTHY
            ):
                overall_status = HealthStatus.DEGRADED

        response: dict[str, Any] = {
            "status": overall_status.value,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }

        if self._started_at:
            uptime = (datetime.now(tz=timezone.utc) - self._started_at).total_seconds()
            response["uptime_seconds"] = uptime

        if self._config.include_details:
            response["components"] = components

        return response

    def _check_watch_daemon(self) -> dict[str, Any]:
        """Check watch daemon health."""
        from sentinel_ml.alerting.watch import WatchState

        if not self._watch_daemon:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "message": "Watch daemon not configured",
            }

        state = self._watch_daemon.state
        stats = self._watch_daemon.stats

        if state == WatchState.RUNNING:
            status = HealthStatus.HEALTHY
            message = "Running normally"
        elif state == WatchState.STARTING:
            status = HealthStatus.DEGRADED
            message = "Starting up"
        elif state == WatchState.ERROR:
            status = HealthStatus.UNHEALTHY
            message = "Error state"
        else:
            status = HealthStatus.DEGRADED
            message = f"State: {state.value}"

        return {
            "status": status.value,
            "message": message,
            "state": state.value,
            "stats": {
                "poll_count": stats.poll_count,
                "lines_processed": stats.lines_processed,
                "novel_count": stats.novel_count,
                "alerts_sent": stats.alerts_sent,
                "error_count": stats.error_count,
            },
        }

    def _check_notifiers(self) -> dict[str, Any]:
        """Check notifiers health."""
        if not self._notifiers:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "message": "No notifiers configured",
            }

        healthy_count = 0
        total_count = len(self._notifiers)
        details: dict[str, dict[str, Any]] = {}

        for notifier in self._notifiers:
            try:
                is_healthy = notifier.health_check()
                if is_healthy:
                    healthy_count += 1
                details[notifier.name] = {
                    "healthy": is_healthy,
                    "enabled": notifier.is_enabled,
                    "stats": notifier.stats,
                }
            except Exception as e:
                details[notifier.name] = {
                    "healthy": False,
                    "error": str(e),
                }

        if healthy_count == total_count:
            status = HealthStatus.HEALTHY
            message = "All notifiers healthy"
        elif healthy_count > 0:
            status = HealthStatus.DEGRADED
            message = f"{healthy_count}/{total_count} notifiers healthy"
        else:
            status = HealthStatus.UNHEALTHY
            message = "No healthy notifiers"

        return {
            "status": status.value,
            "message": message,
            "healthy": healthy_count,
            "total": total_count,
            "details": details,
        }

    def _create_handler(self) -> type[http.server.BaseHTTPRequestHandler]:
        """Create HTTP request handler."""
        health_check = self
        config = self._config

        class HealthHandler(http.server.BaseHTTPRequestHandler):
            """HTTP handler for health endpoint."""

            def log_message(self, format: str, *args: Any) -> None:
                """Suppress default logging."""
                pass

            def do_GET(self) -> None:
                """Handle GET requests."""
                if self.path == config.path:
                    self._send_health_response()
                elif self.path == "/ready":
                    self._send_ready_response()
                elif self.path == "/live":
                    self._send_live_response()
                else:
                    self.send_error(404, "Not Found")

            def _send_health_response(self) -> None:
                """Send health check response."""
                health = health_check.get_health()
                status_code = 200

                if health["status"] == HealthStatus.UNHEALTHY.value:
                    status_code = 503
                elif health["status"] == HealthStatus.DEGRADED.value:
                    status_code = 200  # Still accessible

                self._send_json(health, status_code)

            def _send_ready_response(self) -> None:
                """Send readiness probe response."""
                health = health_check.get_health()
                if health["status"] == HealthStatus.UNHEALTHY.value:
                    self._send_json({"ready": False}, 503)
                else:
                    self._send_json({"ready": True}, 200)

            def _send_live_response(self) -> None:
                """Send liveness probe response."""
                self._send_json({"alive": True}, 200)

            def _send_json(self, data: dict[str, Any], status_code: int) -> None:
                """Send JSON response."""
                body = json.dumps(data, indent=2).encode("utf-8")
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return HealthHandler
