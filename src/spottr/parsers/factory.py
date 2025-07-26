from typing import List

from spottr.core.models import LogFormat
from spottr.parsers.base import BaseLogParser
from spottr.parsers.json_parser import JSONLogParser
from spottr.parsers.line_delimited import LineDelimitedParser


class LogParserFactory:
    """Factory for creating appropriate log parsers based on content analysis."""

    def __init__(self):
        self.parsers = [
            JSONLogParser(),
            LineDelimitedParser(),
        ]

    def detect_format(self, file_path: str, sample_lines: int = 20) -> BaseLogParser:
        """
        Detect the log format and return appropriate parser.
        Args:
            file_path: Path to the log file
            sample_lines: Number of lines to sample for detection
        Returns:
            Appropriate parser instance
        """
        # Read sample lines
        sample = []
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    if i >= sample_lines:
                        break
                    sample.append(line)
        except Exception as e:
            raise Exception(f"Error reading file for format detection: {e}")

        # Try each parser to see which can handle the format
        for parser in self.parsers:
            if parser.can_parse(sample):
                print(f"Detected format: {parser.format_type.value}")
                return parser

        # Default to line-delimited parser
        print("Format detection inconclusive, defaulting to line-delimited parser")
        return LineDelimitedParser()

    def get_parser(self, format_type: LogFormat) -> BaseLogParser:
        """
        Get a specific parser by format type.
        Args:
            format_type: The desired log format
        Returns:
            Parser instance for the specified format
        """
        parser_map = {
            LogFormat.JSON: JSONLogParser,
            LogFormat.LINE_DELIMITED: LineDelimitedParser,
        }

        parser_class = parser_map.get(format_type)
        if parser_class:
            return parser_class()
        else:
            raise ValueError(f"No parser available for format: {format_type}")

    def get_supported_formats(self) -> List[LogFormat]:
        """Get list of supported log formats."""
        return [parser.format_type for parser in self.parsers]
