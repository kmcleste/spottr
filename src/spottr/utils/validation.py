from typing import List

from spottr.core.models import LogEntry


def validate_log_entries(entries: List[LogEntry]) -> List[LogEntry]:
    """Validate and clean log entries list."""
    valid_entries = []
    for i, entry in enumerate(entries):
        if entry is None:
            print(f"Warning: Skipping None entry at index {i}")
            continue
        if not hasattr(entry, "raw_line") or entry.raw_line is None:
            print(f"Warning: Entry at index {i} has no raw_line, fixing...")
            entry.raw_line = getattr(entry, "message", "") or f"Entry {i}"
        valid_entries.append(entry)
    return valid_entries
