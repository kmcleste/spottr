"""Spottr - A library for detecting insights from application logs."""

from spottr.analysis.analyzer import LogAnalyzer
from spottr.analysis.entailment import EntailmentScorer
from spottr.analysis.rule_engine import RuleEngine
from spottr.core.models import Insight, LogEntry, LogFormat
from spottr.parsers.factory import LogParserFactory

__version__ = "0.1.0"
__all__ = [
    "LogEntry",
    "Insight",
    "LogFormat",
    "LogAnalyzer",
    "LogParserFactory",
    "RuleEngine",
    "EntailmentScorer",
]
