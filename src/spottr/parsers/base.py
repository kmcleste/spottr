from abc import ABC, abstractmethod
from typing import Any, Dict, List

from spottr.core.models import LogEntry


class BaseLogParser(ABC):
    """Abstract base class for all log parsers."""

    def __init__(self):
        self.format_type = None
        self.parsed_count = 0
        self.error_count = 0

    @abstractmethod
    def can_parse(self, sample_lines: List[str]) -> bool:
        """
        Determine if this parser can handle the given log format.
        Args:
            sample_lines: First few lines of the log file for format detection
        Returns:
            True if this parser can handle the format
        """
        pass

    @abstractmethod
    def parse_line(self, line: str, line_number: int) -> LogEntry:
        """
        Parse a single log line into a LogEntry object.
        Args:
            line: Raw log line
            line_number: Line number in the file
        Returns:
            LogEntry object
        """
        pass

    @abstractmethod
    def get_format_info(self) -> Dict[str, Any]:
        """
        Get information about this parser's format.
        Returns:
            Dictionary with format metadata
        """
        pass

    def parse_file(self, file_path: str) -> List[LogEntry]:
        """
        Parse an entire log file.
        Args:
            file_path: Path to the log file
        Returns:
            List of LogEntry objects
        """
        entries = []
        self.parsed_count = 0
        self.error_count = 0

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():  # Skip empty lines
                        try:
                            entry = self.parse_line(line, line_num)
                            # Ensure we have a valid entry
                            if entry is not None:
                                entries.append(entry)
                                self.parsed_count += 1
                            else:
                                # Create fallback entry if parse_line returned None
                                self.error_count += 1
                                entry = LogEntry(
                                    raw_line=line.strip(),
                                    line_number=line_num,
                                    message=line.strip(),
                                    metadata={
                                        "parse_error": "parse_line returned None"
                                    },
                                )
                                entries.append(entry)
                        except Exception as e:
                            self.error_count += 1
                            # Create a minimal entry for unparseable lines
                            entry = LogEntry(
                                raw_line=line.strip(),
                                line_number=line_num,
                                message=line.strip(),
                                metadata={"parse_error": str(e)},
                            )
                            entries.append(entry)

        except Exception as e:
            raise Exception(f"Error reading file {file_path}: {e}")

        return entries

    def get_parsing_stats(self) -> Dict[str, int]:
        """Get parsing statistics."""
        return {
            "parsed_count": self.parsed_count,
            "error_count": self.error_count,
            "success_rate": self.parsed_count / (self.parsed_count + self.error_count)
            if (self.parsed_count + self.error_count) > 0
            else 0,
        }
