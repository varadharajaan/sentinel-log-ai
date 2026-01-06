"""
Log line parsing for common log formats.

Supports syslog, nginx, Python traceback, JSON, and generic formats.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from sentinel_ml.models import LogRecord


class Parser(ABC):
    """Abstract base class for log parsers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the parser name."""
        pass

    @abstractmethod
    def can_parse(self, line: str) -> bool:
        """Return True if this parser can handle the line."""
        pass

    @abstractmethod
    def parse(self, line: str, source: str) -> LogRecord | None:
        """Parse a log line into a LogRecord. Returns None if parsing fails."""
        pass


class JSONParser(Parser):
    """Parser for JSON-formatted log lines."""

    @property
    def name(self) -> str:
        return "json"

    def can_parse(self, line: str) -> bool:
        trimmed = line.strip()
        return trimmed.startswith("{") and trimmed.endswith("}")

    def parse(self, line: str, source: str) -> LogRecord | None:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        # Extract common fields
        message = data.get("message") or data.get("msg") or ""
        level = data.get("level") or data.get("lvl")
        timestamp = None

        if ts := data.get("timestamp") or data.get("time") or data.get("ts"):
            timestamp = self._parse_timestamp(ts)

        return LogRecord(
            message=message,
            level=level.upper() if level else None,
            source=source,
            raw=line,
            timestamp=timestamp,
            attrs=data,
        )

    def _parse_timestamp(self, ts: Any) -> datetime | None:
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts)
        if isinstance(ts, str):
            for fmt in [
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S",
            ]:
                try:
                    return datetime.strptime(ts, fmt)
                except ValueError:
                    continue
        return None


class SyslogParser(Parser):
    """Parser for syslog-formatted log lines."""

    # Jan 15 10:30:00 hostname program[pid]: message
    _pattern = re.compile(
        r"^([A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+"
        r"(\S+)\s+"
        r"(\S+?)(?:\[(\d+)\])?:\s*(.*)$"
    )

    @property
    def name(self) -> str:
        return "syslog"

    def can_parse(self, line: str) -> bool:
        return self._pattern.match(line) is not None

    def parse(self, line: str, source: str) -> LogRecord | None:
        match = self._pattern.match(line)
        if not match:
            return None

        ts_str, hostname, program, pid, message = match.groups()

        # Parse timestamp (add current year)
        try:
            ts = datetime.strptime(ts_str, "%b %d %H:%M:%S")
            ts = ts.replace(year=datetime.now().year)
        except ValueError:
            ts = None

        attrs: dict[str, Any] = {
            "hostname": hostname,
            "program": program,
        }
        if pid:
            attrs["pid"] = int(pid)

        # Try to extract level from message
        level = self._extract_level(message)

        return LogRecord(
            message=message,
            level=level,
            source=source,
            raw=line,
            timestamp=ts,
            attrs=attrs,
        )

    def _extract_level(self, message: str) -> str | None:
        upper = message.upper()
        if "ERROR" in upper or "FAIL" in upper:
            return "ERROR"
        if "WARN" in upper:
            return "WARN"
        if "DEBUG" in upper:
            return "DEBUG"
        if "INFO" in upper:
            return "INFO"
        if "FATAL" in upper or "CRITICAL" in upper:
            return "FATAL"
        return None


class NginxParser(Parser):
    """Parser for nginx access and error log lines."""

    # Combined log format
    _access_pattern = re.compile(
        r'^(\S+)\s+\S+\s+(\S+)\s+\[([^\]]+)\]\s+"([^"]+)"\s+(\d+)\s+(\d+)\s+"([^"]*)"\s+"([^"]*)"'
    )

    # Error log format
    _error_pattern = re.compile(
        r"^(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})\s+\[(\w+)\]\s+(\d+)#(\d+):\s*(?:\*\d+\s+)?(.*)$"
    )

    @property
    def name(self) -> str:
        return "nginx"

    def can_parse(self, line: str) -> bool:
        return (
            self._access_pattern.match(line) is not None
            or self._error_pattern.match(line) is not None
        )

    def parse(self, line: str, source: str) -> LogRecord | None:
        # Try access log first
        if match := self._access_pattern.match(line):
            return self._parse_access_log(match, line, source)

        # Try error log
        if match := self._error_pattern.match(line):
            return self._parse_error_log(match, line, source)

        return None

    def _parse_access_log(self, match: re.Match[str], line: str, source: str) -> LogRecord:
        client_ip, user, ts_str, request, status, size, referer, user_agent = match.groups()

        try:
            ts = datetime.strptime(ts_str, "%d/%b/%Y:%H:%M:%S %z")
        except ValueError:
            ts = None

        level = self._status_to_level(status)

        return LogRecord(
            message=request,
            level=level,
            source=source,
            raw=line,
            timestamp=ts,
            attrs={
                "client_ip": client_ip,
                "user": user,
                "status_code": int(status),
                "bytes": int(size),
                "referer": referer,
                "user_agent": user_agent,
            },
        )

    def _parse_error_log(self, match: re.Match[str], line: str, source: str) -> LogRecord:
        ts_str, level, pid, tid, message = match.groups()

        try:
            ts = datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S")
        except ValueError:
            ts = None

        return LogRecord(
            message=message,
            level=level.upper(),
            source=source,
            raw=line,
            timestamp=ts,
            attrs={
                "pid": int(pid),
                "tid": int(tid),
            },
        )

    def _status_to_level(self, status: str) -> str:
        if not status:
            return "INFO"
        first = status[0]
        if first == "2" or first == "3":
            return "INFO"
        if first == "4":
            return "WARN"
        if first == "5":
            return "ERROR"
        return "INFO"


class PythonTracebackParser(Parser):
    """Parser for Python traceback lines."""

    _error_pattern = re.compile(
        r"^(\w+Error|\w+Exception|Traceback \(most recent call last\)):?\s*(.*)$"
    )

    @property
    def name(self) -> str:
        return "python_traceback"

    def can_parse(self, line: str) -> bool:
        trimmed = line.strip()
        return (
            self._error_pattern.match(line) is not None
            or trimmed.startswith('File "')
            or (line.startswith("    ") and len(trimmed) > 0)
        )

    def parse(self, line: str, source: str) -> LogRecord | None:
        match = self._error_pattern.match(line)
        if match:
            exception_type, exception_msg = match.groups()
            return LogRecord(
                message=line,
                level="ERROR",
                source=source,
                raw=line,
                attrs={
                    "exception_type": exception_type,
                    "exception_msg": exception_msg,
                    "parser": "python_traceback",
                },
            )

        # Context lines (file references, code)
        return LogRecord(
            message=line,
            level="ERROR",
            source=source,
            raw=line,
            attrs={"parser": "python_traceback"},
        )


class CommonLogParser(Parser):
    """Parser for generic log formats."""

    _pattern = re.compile(
        r"(?i)^(?:(\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\s+)?"
        r"(?:\[?(DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL|CRITICAL)\]?[:\s]+)?"
        r"(.+)$"
    )

    @property
    def name(self) -> str:
        return "common"

    def can_parse(self, _line: str) -> bool:
        return True  # Fallback parser

    def parse(self, line: str, source: str) -> LogRecord | None:
        match = self._pattern.match(line)
        if not match:
            return LogRecord(message=line, source=source, raw=line)

        ts_str, level, message = match.groups()

        timestamp = None
        if ts_str:
            timestamp = self._parse_timestamp(ts_str)

        level = self._normalize_level(level) if level else self._extract_level(message)

        return LogRecord(
            message=message or line,
            level=level,
            source=source,
            raw=line,
            timestamp=timestamp,
        )

    def _parse_timestamp(self, ts_str: str) -> datetime | None:
        for fmt in [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
        ]:
            try:
                return datetime.strptime(ts_str, fmt)
            except ValueError:
                continue
        return None

    def _normalize_level(self, level: str) -> str:
        upper = level.upper()
        if upper == "WARNING":
            return "WARN"
        if upper == "CRITICAL":
            return "FATAL"
        return upper

    def _extract_level(self, message: str) -> str | None:
        upper = message.upper()
        if "ERROR" in upper or "FAIL" in upper:
            return "ERROR"
        if "WARN" in upper:
            return "WARN"
        if "DEBUG" in upper:
            return "DEBUG"
        if "INFO" in upper:
            return "INFO"
        if "FATAL" in upper or "CRITICAL" in upper:
            return "FATAL"
        return None


class ParserRegistry:
    """Registry for log parsers. Routes log lines to appropriate parsers."""

    def __init__(self) -> None:
        self._parsers: list[Parser] = [
            JSONParser(),
            SyslogParser(),
            NginxParser(),
            PythonTracebackParser(),
            CommonLogParser(),  # Fallback
        ]

    def register(self, parser: Parser, priority: int = 0) -> None:
        """Register a parser. Lower priority number = higher priority."""
        self._parsers.insert(priority, parser)

    def parse(self, line: str, source: str) -> LogRecord:
        """Parse a log line using the first matching parser."""
        for parser in self._parsers:
            if parser.can_parse(line):
                result = parser.parse(line, source)
                if result is not None:
                    return result

        # Ultimate fallback
        return LogRecord(message=line, source=source, raw=line)

    @property
    def parsers(self) -> list[Parser]:
        """Return list of registered parsers."""
        return list(self._parsers)


# Convenience function
_registry: ParserRegistry | None = None


def get_parser_registry() -> ParserRegistry:
    """Get the global parser registry."""
    global _registry
    if _registry is None:
        _registry = ParserRegistry()
    return _registry


def parse_log_line(line: str, source: str = "unknown") -> LogRecord:
    """Parse a log line using the global registry."""
    return get_parser_registry().parse(line, source)
