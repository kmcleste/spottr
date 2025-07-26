from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


@dataclass
class LogEntry:
    """Represents a single log entry with parsed components."""

    raw_line: str
    timestamp: Optional[datetime] = None
    level: Optional[str] = None
    component: Optional[str] = None
    message: str = ""
    line_number: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        # Ensure raw_line is never None
        if self.raw_line is None:
            self.raw_line = ""
        # Ensure message has a fallback
        if not self.message:
            self.message = self.raw_line


@dataclass
class Insight:
    """Represents an extracted insight from log analysis."""

    rule_name: str
    confidence: float
    description: str
    evidence: List[LogEntry]
    timestamp: datetime
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    category: str  # PERFORMANCE, ERROR, SECURITY, RESOURCE
    metadata: Dict[str, Any] = None

    # New entailment fields
    entailment_scores: List[float] = None  # Entailment scores for evidence
    avg_entailment_score: float = 0.0  # Average entailment score
    target_statement: str = ""  # Target statement used for entailment

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.entailment_scores is None:
            self.entailment_scores = []


class LogFormat(Enum):
    """Supported log formats."""

    LINE_DELIMITED = "line_delimited"
    JSON = "json"
    SYSLOG = "syslog"
    CUSTOM = "custom"
