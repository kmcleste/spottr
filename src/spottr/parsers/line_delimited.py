import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from spottr.core.models import LogEntry, LogFormat
from spottr.parsers.base import BaseLogParser


class LineDelimitedParser(BaseLogParser):
    """Parses line-delimited plain text logs into structured LogEntry objects."""

    def __init__(self):
        super().__init__()
        self.format_type = LogFormat.LINE_DELIMITED

        # Common timestamp patterns
        self.timestamp_patterns = [
            r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})",  # 2024-01-15 14:30:25
            r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})",  # 2024-01-15T14:30:25
            r"(\w{3} \d{2} \d{2}:\d{2}:\d{2})",  # Jan 15 14:30:25
            r"(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})",  # 01/15/2024 14:30:25
        ]

        # Log level patterns
        self.level_pattern = r"\b(DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL|TRACE)\b"

        # Component/logger patterns
        self.component_pattern = (
            r"\[([^\]]+)\]|\b([A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*):|\b([A-Za-z0-9_]+)\s*-"
        )

    def can_parse(self, sample_lines: List[str]) -> bool:
        """Determine if this parser can handle line-delimited logs."""
        if not sample_lines:
            return False

        # Check if lines look like traditional log format
        traditional_indicators = 0
        for line in sample_lines[:10]:  # Check first 10 lines
            line = line.strip()
            if not line:
                continue

            # Look for timestamp patterns
            has_timestamp = any(
                re.search(pattern, line) for pattern in self.timestamp_patterns
            )
            # Look for log levels
            has_level = re.search(self.level_pattern, line, re.IGNORECASE)
            # Look for component patterns
            has_component = re.search(self.component_pattern, line)

            if has_timestamp or has_level or has_component:
                traditional_indicators += 1

        # If more than 30% of lines have traditional log indicators, we can parse this
        return (
            traditional_indicators
            >= len([line_item for line_item in sample_lines[:10] if line_item.strip()])
            * 0.3
        )

    def get_format_info(self) -> Dict[str, Any]:
        """Get information about line-delimited format."""
        return {
            "format_type": self.format_type.value,
            "description": "Line-delimited plain text logs",
            "timestamp_patterns": len(self.timestamp_patterns),
            "supports_levels": True,
            "supports_components": True,
        }

    def parse_line(self, line: str, line_number: int) -> LogEntry:
        """Parse a single log line into a LogEntry object."""
        if not line or line.isspace():
            # Return minimal entry for empty/whitespace lines
            return LogEntry(
                raw_line=line.strip() if line else "",
                line_number=line_number,
                message=line.strip() if line else "",
                metadata={"note": "empty_or_whitespace_line"},
            )

        entry = LogEntry(raw_line=line.strip(), line_number=line_number)

        try:
            # Extract timestamp
            entry.timestamp = self._extract_timestamp(line)

            # Extract log level
            entry.level = self._extract_level(line)

            # Extract component/logger name
            entry.component = self._extract_component(line)

            # Extract message (everything after structured parts)
            entry.message = self._extract_message(line)

            # Ensure message is not empty
            if not entry.message:
                entry.message = line.strip()

        except Exception as e:
            # If parsing fails, at least preserve the raw content
            entry.message = line.strip()
            entry.metadata = {"parse_error": str(e)}

        return entry

    def _extract_timestamp(self, line: str) -> Optional[datetime]:
        """Extract timestamp from log line."""
        for pattern in self.timestamp_patterns:
            match = re.search(pattern, line)
            if match:
                timestamp_str = match.group(1)
                # Try different datetime formats
                formats = [
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S",
                    "%b %d %H:%M:%S",
                    "%m/%d/%Y %H:%M:%S",
                ]

                for fmt in formats:
                    try:
                        return datetime.strptime(timestamp_str, fmt)
                    except ValueError:
                        continue
        return None

    def _extract_level(self, line: str) -> Optional[str]:
        """Extract log level from log line."""
        match = re.search(self.level_pattern, line, re.IGNORECASE)
        return match.group(1).upper() if match else None

    def _extract_component(self, line: str) -> Optional[str]:
        """Extract component/logger name from log line."""
        match = re.search(self.component_pattern, line)
        if match:
            # Return first non-None group
            return next((g for g in match.groups() if g), None)
        return None

    def _extract_message(self, line: str) -> str:
        """Extract the main message content."""
        # Remove timestamp, level, and component info to get core message
        cleaned = line

        # Remove timestamp
        for pattern in self.timestamp_patterns:
            cleaned = re.sub(pattern, "", cleaned)

        # Remove level
        cleaned = re.sub(self.level_pattern, "", cleaned, flags=re.IGNORECASE)

        # Remove component patterns
        cleaned = re.sub(self.component_pattern, "", cleaned)

        # Clean up extra whitespace and separators
        cleaned = re.sub(r"^\s*[-:\s]+", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)

        return cleaned.strip()
