import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from spottr.core.models import LogEntry, LogFormat
from spottr.parsers.base import BaseLogParser


class JSONLogParser(BaseLogParser):
    """Parses JSON-formatted logs into structured LogEntry objects."""

    def __init__(self):
        super().__init__()
        self.format_type = LogFormat.JSON

        # Common field mappings for different JSON log formats
        self.field_mappings = {
            "timestamp": ["timestamp", "time", "@timestamp", "ts", "datetime", "date"],
            "level": ["level", "severity", "log_level", "priority", "loglevel"],
            "component": ["component", "service", "logger", "source", "module", "name"],
            "message": ["message", "msg", "text", "content", "description", "event"],
        }

        # Timestamp formats to try
        self.timestamp_formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",  # 2024-01-15T14:30:25.123Z
            "%Y-%m-%dT%H:%M:%SZ",  # 2024-01-15T14:30:25Z
            "%Y-%m-%dT%H:%M:%S.%f",  # 2024-01-15T14:30:25.123
            "%Y-%m-%dT%H:%M:%S",  # 2024-01-15T14:30:25
            "%Y-%m-%d %H:%M:%S.%f",  # 2024-01-15 14:30:25.123
            "%Y-%m-%d %H:%M:%S",  # 2024-01-15 14:30:25
            "%m/%d/%Y %H:%M:%S",  # 01/15/2024 14:30:25
        ]

    def can_parse(self, sample_lines: List[str]) -> bool:
        """Determine if this parser can handle JSON logs."""
        if not sample_lines:
            return False

        json_lines = 0
        for line in sample_lines[:10]:  # Check first 10 lines
            line = line.strip()
            if not line:
                continue

            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    json_lines += 1
            except (json.JSONDecodeError, ValueError):
                continue

        # If more than 70% of lines are valid JSON objects, we can parse this
        non_empty_lines = len([line for line in sample_lines[:10] if line.strip()])
        return json_lines >= non_empty_lines * 0.7 if non_empty_lines > 0 else False

    def parse_line(self, line: str, line_number: int) -> LogEntry:
        """Parse a single JSON log line into a LogEntry object."""
        if not line or line.isspace():
            return LogEntry(
                raw_line=line.strip() if line else "",
                line_number=line_number,
                message=line.strip() if line else "",
                metadata={"note": "empty_or_whitespace_line"},
            )

        try:
            data = json.loads(line.strip())
            if not isinstance(data, dict):
                raise ValueError("JSON line is not an object")
        except (json.JSONDecodeError, ValueError) as e:
            # Return basic entry for invalid JSON
            return LogEntry(
                raw_line=line.strip(),
                line_number=line_number,
                message=line.strip(),
                metadata={"parse_error": f"Invalid JSON: {str(e)}"},
            )

        # Create LogEntry with extracted fields
        entry = LogEntry(raw_line=line.strip(), line_number=line_number)

        try:
            # Extract timestamp
            entry.timestamp = self._extract_timestamp_from_json(data)

            # Extract log level
            entry.level = self._extract_level_from_json(data)

            # Extract component
            entry.component = self._extract_component_from_json(data)

            # Extract message
            entry.message = self._extract_message_from_json(data)

            # Ensure message is not empty
            if not entry.message:
                entry.message = line.strip()

            # Store original JSON data in metadata
            entry.metadata = {"original_json": data, "format": "json"}

            # Add any additional fields not captured above
            standard_fields = {
                "timestamp",
                "level",
                "component",
                "message",
                "time",
                "@timestamp",
                "ts",
                "datetime",
                "date",
                "severity",
                "log_level",
                "priority",
                "loglevel",
                "service",
                "logger",
                "source",
                "module",
                "name",
                "msg",
                "text",
                "content",
                "description",
                "event",
            }
            additional_fields = {
                k: v for k, v in data.items() if k not in standard_fields
            }
            if additional_fields:
                entry.metadata["additional_fields"] = additional_fields

        except Exception as e:
            # If extraction fails, preserve what we can
            entry.message = line.strip()
            entry.metadata = {"parse_error": f"Field extraction failed: {str(e)}"}

        return entry

    def _extract_timestamp_from_json(self, data: Dict) -> Optional[datetime]:
        """Extract timestamp from JSON data."""
        for field_name in self.field_mappings["timestamp"]:
            if field_name in data:
                timestamp_value = data[field_name]

                # Handle different timestamp formats
                if isinstance(timestamp_value, (int, float)):
                    # Unix timestamp (seconds or milliseconds)
                    if timestamp_value > 1e10:  # Milliseconds
                        return datetime.fromtimestamp(timestamp_value / 1000.0)
                    else:  # Seconds
                        return datetime.fromtimestamp(timestamp_value)

                elif isinstance(timestamp_value, str):
                    # String timestamp - try different formats
                    for fmt in self.timestamp_formats:
                        try:
                            return datetime.strptime(timestamp_value, fmt)
                        except ValueError:
                            continue

        return None

    def _extract_level_from_json(self, data: Dict) -> Optional[str]:
        """Extract log level from JSON data."""
        for field_name in self.field_mappings["level"]:
            if field_name in data:
                level = str(data[field_name]).upper()
                # Normalize common level variations
                level_mapping = {
                    "WARN": "WARNING",
                    "ERR": "ERROR",
                    "CRIT": "CRITICAL",
                    "FATAL": "CRITICAL",
                }
                return level_mapping.get(level, level)

        return None

    def _extract_component_from_json(self, data: Dict) -> Optional[str]:
        """Extract component/service name from JSON data."""
        for field_name in self.field_mappings["component"]:
            if field_name in data:
                return str(data[field_name])

        return None

    def _extract_message_from_json(self, data: Dict) -> str:
        """Extract message from JSON data."""
        for field_name in self.field_mappings["message"]:
            if field_name in data:
                message = data[field_name]
                if isinstance(message, str):
                    return message
                else:
                    return str(message)

        # If no standard message field, create a summary from available fields
        excluded_fields = set()
        for field_list in self.field_mappings.values():
            excluded_fields.update(field_list)

        message_parts = []
        for key, value in data.items():
            if key not in excluded_fields and value is not None:
                message_parts.append(f"{key}={value}")

        return " ".join(message_parts) if message_parts else "No message content"

    def get_format_info(self) -> Dict[str, Any]:
        """Get information about JSON format."""
        return {
            "format_type": self.format_type.value,
            "description": "JSON-formatted logs (one JSON object per line)",
            "field_mappings": self.field_mappings,
            "timestamp_formats": len(self.timestamp_formats),
            "supports_levels": True,
            "supports_components": True,
        }
