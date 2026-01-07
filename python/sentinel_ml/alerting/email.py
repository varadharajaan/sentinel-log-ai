"""
Email SMTP notifier for alert delivery.

Sends formatted email alerts via SMTP with HTML and plain text support.
Includes configurable recipients, templates, and TLS encryption.
"""

from __future__ import annotations

import smtplib
import ssl
from dataclasses import dataclass, field
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import structlog

from sentinel_ml.alerting.base import (
    AlertEvent,
    AlertPriority,
    BaseNotifier,
    NotifierConfig,
    NotifierFactory,
)

logger = structlog.get_logger(__name__)


# Priority to subject prefix mapping
PRIORITY_PREFIXES: dict[AlertPriority, str] = {
    AlertPriority.CRITICAL: "[CRITICAL]",
    AlertPriority.HIGH: "[HIGH]",
    AlertPriority.MEDIUM: "[MEDIUM]",
    AlertPriority.LOW: "[LOW]",
    AlertPriority.INFO: "[INFO]",
}


@dataclass
class EmailConfig(NotifierConfig):
    """
    Configuration for Email notifier.

    Attributes:
        smtp_host: SMTP server hostname.
        smtp_port: SMTP server port.
        smtp_username: SMTP authentication username.
        smtp_password: SMTP authentication password.
        use_tls: Whether to use TLS encryption.
        use_ssl: Whether to use SSL encryption.
        from_address: Sender email address.
        from_name: Sender display name.
        to_addresses: Default recipient addresses.
        cc_addresses: CC recipient addresses.
        subject_prefix: Prefix for email subjects.
        include_html: Whether to include HTML body.
    """

    smtp_host: str = "localhost"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    use_tls: bool = True
    use_ssl: bool = False
    from_address: str = ""
    from_name: str = "Sentinel ML Alerts"
    to_addresses: list[str] = field(default_factory=list)
    cc_addresses: list[str] = field(default_factory=list)
    subject_prefix: str = "[Sentinel ML]"
    include_html: bool = True

    def __post_init__(self) -> None:
        if not self.name or self.name == "base-notifier":
            self.name = "email-notifier"


class EmailNotifier(BaseNotifier):
    """
    Email SMTP notifier.

    Sends alert emails via SMTP with support for HTML formatting,
    TLS encryption, and multiple recipients.
    """

    def __init__(self, config: EmailConfig) -> None:
        """
        Initialize Email notifier.

        Args:
            config: Email-specific configuration.
        """
        super().__init__(config)
        self._email_config = config
        self._logger = logger.bind(
            notifier=config.name,
            smtp_host=config.smtp_host,
        )

    def _send(self, event: AlertEvent) -> dict[str, Any]:
        """
        Send alert via email.

        Args:
            event: Alert event to send.

        Returns:
            Response data with send status.

        Raises:
            ValueError: If configuration is incomplete.
            smtplib.SMTPException: If sending fails.
        """
        if not self._email_config.from_address:
            raise ValueError("From address is required")
        if not self._email_config.to_addresses:
            raise ValueError("At least one recipient is required")

        message = self._build_message(event)
        recipients = self._get_recipients(event)

        self._send_smtp(message, recipients)

        return {
            "status": "sent",
            "recipients": len(recipients),
            "subject": message["Subject"],
        }

    def _build_message(self, event: AlertEvent) -> MIMEMultipart:
        """
        Build email message.

        Args:
            event: Alert event to format.

        Returns:
            MIME message ready to send.
        """
        message = MIMEMultipart("alternative")

        # Build subject
        prefix = PRIORITY_PREFIXES.get(event.priority, "")
        subject_parts = []
        if self._email_config.subject_prefix:
            subject_parts.append(self._email_config.subject_prefix)
        if prefix:
            subject_parts.append(prefix)
        subject_parts.append(event.title)
        message["Subject"] = " ".join(subject_parts)

        # Set headers
        from_header = self._email_config.from_address
        if self._email_config.from_name:
            from_header = f"{self._email_config.from_name} <{self._email_config.from_address}>"
        message["From"] = from_header
        message["To"] = ", ".join(self._email_config.to_addresses)
        if self._email_config.cc_addresses:
            message["Cc"] = ", ".join(self._email_config.cc_addresses)
        message["X-Priority"] = self._get_x_priority(event.priority)
        message["X-Sentinel-Event-ID"] = event.event_id

        # Plain text body
        plain_body = self._build_plain_body(event)
        message.attach(MIMEText(plain_body, "plain", "utf-8"))

        # HTML body
        if self._email_config.include_html:
            html_body = self._build_html_body(event)
            message.attach(MIMEText(html_body, "html", "utf-8"))

        return message

    def _build_plain_body(self, event: AlertEvent) -> str:
        """Build plain text email body."""
        lines = [
            f"Alert: {event.title}",
            "",
            f"Priority: {event.priority.value.upper()}",
            f"Source: {event.source}",
            f"Time: {event.timestamp.isoformat()}",
            f"Event ID: {event.event_id}",
            "",
            "Message:",
            event.message,
        ]

        if event.tags:
            lines.extend(["", f"Tags: {', '.join(event.tags)}"])

        if event.metadata:
            lines.extend(["", "Metadata:"])
            for key, value in event.metadata.items():
                lines.append(f"  {key}: {value}")

        lines.extend([
            "",
            "---",
            "This is an automated message from Sentinel ML.",
        ])

        return "\n".join(lines)

    def _build_html_body(self, event: AlertEvent) -> str:
        """Build HTML email body."""
        priority_colors = {
            AlertPriority.CRITICAL: "#dc3545",
            AlertPriority.HIGH: "#fd7e14",
            AlertPriority.MEDIUM: "#ffc107",
            AlertPriority.LOW: "#17a2b8",
            AlertPriority.INFO: "#6c757d",
        }
        color = priority_colors.get(event.priority, "#6c757d")

        metadata_rows = ""
        if event.metadata:
            for key, value in event.metadata.items():
                metadata_rows += f"<tr><td><strong>{key}</strong></td><td>{value}</td></tr>"

        tags_html = ""
        if event.tags:
            tag_spans = [f'<span style="background:#e9ecef;padding:2px 6px;margin:2px;border-radius:3px;">{tag}</span>' for tag in event.tags]
            tags_html = f'<p><strong>Tags:</strong> {"".join(tag_spans)}</p>'

        return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family: Arial, sans-serif; margin: 0; padding: 20px;">
    <div style="max-width: 600px; margin: 0 auto;">
        <div style="background: {color}; color: white; padding: 15px; border-radius: 4px 4px 0 0;">
            <h2 style="margin: 0;">{event.title}</h2>
        </div>
        <div style="border: 1px solid #ddd; border-top: none; padding: 20px; border-radius: 0 0 4px 4px;">
            <table style="width: 100%; margin-bottom: 15px;">
                <tr>
                    <td><strong>Priority:</strong></td>
                    <td><span style="color: {color}; font-weight: bold;">{event.priority.value.upper()}</span></td>
                </tr>
                <tr>
                    <td><strong>Source:</strong></td>
                    <td>{event.source}</td>
                </tr>
                <tr>
                    <td><strong>Time:</strong></td>
                    <td>{event.timestamp.isoformat()}</td>
                </tr>
                <tr>
                    <td><strong>Event ID:</strong></td>
                    <td style="font-family: monospace;">{event.event_id}</td>
                </tr>
            </table>
            <h3>Message</h3>
            <p style="background: #f8f9fa; padding: 15px; border-radius: 4px;">{event.message}</p>
            {tags_html}
            {f'<h3>Metadata</h3><table style="width: 100%;">{metadata_rows}</table>' if metadata_rows else ''}
        </div>
        <p style="color: #6c757d; font-size: 12px; margin-top: 20px;">
            This is an automated message from Sentinel ML.
        </p>
    </div>
</body>
</html>"""

    def _get_recipients(self, event: AlertEvent) -> list[str]:  # noqa: ARG002
        """Get all recipients for the alert."""
        recipients = list(self._email_config.to_addresses)
        recipients.extend(self._email_config.cc_addresses)
        return recipients

    def _get_x_priority(self, priority: AlertPriority) -> str:
        """Map alert priority to X-Priority header value."""
        mapping = {
            AlertPriority.CRITICAL: "1",
            AlertPriority.HIGH: "2",
            AlertPriority.MEDIUM: "3",
            AlertPriority.LOW: "4",
            AlertPriority.INFO: "5",
        }
        return mapping.get(priority, "3")

    def _send_smtp(self, message: MIMEMultipart, recipients: list[str]) -> None:
        """
        Send message via SMTP.

        Args:
            message: MIME message to send.
            recipients: List of recipient addresses.

        Raises:
            smtplib.SMTPException: If sending fails.
        """
        config = self._email_config

        self._logger.debug(
            "smtp_connecting",
            host=config.smtp_host,
            port=config.smtp_port,
        )

        context = ssl.create_default_context() if (config.use_tls or config.use_ssl) else None

        try:
            if config.use_ssl:
                server = smtplib.SMTP_SSL(
                    config.smtp_host,
                    config.smtp_port,
                    timeout=config.timeout_seconds,
                    context=context,
                )
            else:
                server = smtplib.SMTP(
                    config.smtp_host,
                    config.smtp_port,
                    timeout=config.timeout_seconds,
                )
                if config.use_tls and context:
                    server.starttls(context=context)

            try:
                if config.smtp_username and config.smtp_password:
                    server.login(config.smtp_username, config.smtp_password)

                server.sendmail(
                    config.from_address,
                    recipients,
                    message.as_string(),
                )

                self._logger.debug(
                    "smtp_sent",
                    recipients=len(recipients),
                )
            finally:
                server.quit()

        except smtplib.SMTPException as e:
            self._logger.error(
                "smtp_error",
                error=str(e),
            )
            raise

    def validate_config(self) -> list[str]:
        """Validate email configuration."""
        errors = super().validate_config()

        if not self._email_config.smtp_host:
            errors.append("SMTP host is required")
        if not self._email_config.from_address:
            errors.append("From address is required")
        if not self._email_config.to_addresses:
            errors.append("At least one recipient address is required")
        if self._email_config.use_ssl and self._email_config.use_tls:
            errors.append("Cannot use both SSL and TLS simultaneously")

        return errors

    def health_check(self) -> bool:
        """Check if email notifier is operational."""
        if not super().health_check():
            return False

        try:
            config = self._email_config
            if config.use_ssl:
                server = smtplib.SMTP_SSL(
                    config.smtp_host,
                    config.smtp_port,
                    timeout=5,
                )
            else:
                server = smtplib.SMTP(
                    config.smtp_host,
                    config.smtp_port,
                    timeout=5,
                )
            server.quit()
            return True
        except Exception:
            return False


# Register with factory
NotifierFactory.register("email", EmailNotifier)
