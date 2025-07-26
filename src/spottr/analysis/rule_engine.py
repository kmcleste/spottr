import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

from spottr.core.models import Insight, LogEntry


class RuleEngine:
    """Rule-based pattern extraction engine for log analysis."""

    def __init__(self):
        self.rules = self._load_default_rules()

    def _load_default_rules(self) -> Dict[str, Dict]:
        """Load default rule set for common log patterns."""
        return {
            "memory_exhaustion": {
                "patterns": [
                    r"OutOfMemoryError",
                    r"java\.lang\.OutOfMemoryError",
                    r"memory.*exhausted",
                    r"heap.*full",
                    r"cannot allocate memory",
                ],
                "severity": "CRITICAL",
                "category": "RESOURCE",
                "description": "Memory exhaustion detected",
            },
            "high_error_rate": {
                "patterns": [r"ERROR", r"FATAL", r"Exception", r"failed", r"error"],
                "severity": "HIGH",
                "category": "ERROR",
                "description": "High error rate detected",
                "threshold": 0.1,  # 10% error rate threshold
                "window_minutes": 5,
            },
            "slow_queries": {
                "patterns": [
                    r"slow query",
                    r"query.*took.*\d+ms",
                    r"execution time.*\d+ms",
                    r"timeout.*query",
                ],
                "severity": "MEDIUM",
                "category": "PERFORMANCE",
                "description": "Slow database queries detected",
            },
            "authentication_failures": {
                "patterns": [
                    r"authentication.*failed",
                    r"login.*failed",
                    r"invalid.*credentials",
                    r"unauthorized.*access",
                    r"access.*denied",
                ],
                "severity": "HIGH",
                "category": "SECURITY",
                "description": "Authentication failures detected",
            },
            "connection_issues": {
                "patterns": [
                    r"connection.*refused",
                    r"connection.*timeout",
                    r"connection.*reset",
                    r"unable to connect",
                    r"network.*unreachable",
                ],
                "severity": "HIGH",
                "category": "PERFORMANCE",
                "description": "Network connectivity issues detected",
            },
            "resource_warnings": {
                "patterns": [
                    r"disk.*full",
                    r"storage.*exhausted",
                    r"cpu.*high",
                    r"load.*average.*high",
                    r"thread.*pool.*exhausted",
                ],
                "severity": "MEDIUM",
                "category": "RESOURCE",
                "description": "Resource utilization warnings",
            },
        }

    def add_rule(self, name: str, rule_config: Dict):
        """Add a custom rule to the engine."""
        self.rules[name] = rule_config

    def apply_rules(
        self, log_entries: List[LogEntry], time_window_minutes: int = 5
    ) -> List[Insight]:
        """Apply all rules to log entries and extract insights."""
        insights = []

        for rule_name, rule_config in self.rules.items():
            rule_insights = self._apply_single_rule(
                rule_name, rule_config, log_entries, time_window_minutes
            )
            insights.extend(rule_insights)

        return insights

    def _apply_single_rule(
        self,
        rule_name: str,
        rule_config: Dict,
        log_entries: List[LogEntry],
        time_window_minutes: int,
    ) -> List[Insight]:
        """Apply a single rule to log entries."""
        patterns = rule_config["patterns"]
        matching_entries = []

        # Find all entries matching the patterns
        for entry in log_entries:
            # Skip None entries or entries without raw_line
            if (
                entry is None
                or not hasattr(entry, "raw_line")
                or entry.raw_line is None
            ):
                continue

            for pattern in patterns:
                try:
                    if re.search(pattern, entry.raw_line, re.IGNORECASE):
                        matching_entries.append(entry)
                        break
                except Exception as e:
                    # Log the error but continue processing
                    print(f"Warning: Error applying pattern '{pattern}' to entry: {e}")
                    continue

        if not matching_entries:
            return []

        # Handle threshold-based rules (like error rates)
        if "threshold" in rule_config:
            return self._handle_threshold_rule(
                rule_name,
                rule_config,
                matching_entries,
                log_entries,
                time_window_minutes,
            )

        # Handle pattern-based rules
        return self._handle_pattern_rule(rule_name, rule_config, matching_entries)

    def _handle_pattern_rule(
        self, rule_name: str, rule_config: Dict, matching_entries: List[LogEntry]
    ) -> List[Insight]:
        """Handle simple pattern-based rules."""
        if not matching_entries:
            return []

        # Group by time windows for better insight generation
        insights = []

        # For now, create one insight per cluster of related entries
        insight = Insight(
            rule_name=rule_name,
            confidence=min(
                1.0, len(matching_entries) / 10.0
            ),  # Simple confidence scoring
            description=f"{rule_config['description']} ({len(matching_entries)} occurrences)",
            evidence=matching_entries[:10],  # Limit evidence to prevent overwhelming
            timestamp=matching_entries[0].timestamp or datetime.now(),
            severity=rule_config["severity"],
            category=rule_config["category"],
            metadata={
                "total_matches": len(matching_entries),
                "pattern": rule_config["patterns"][
                    0
                ],  # Show first pattern for reference
            },
        )

        insights.append(insight)
        return insights

    def _handle_threshold_rule(
        self,
        rule_name: str,
        rule_config: Dict,
        matching_entries: List[LogEntry],
        all_entries: List[LogEntry],
        time_window_minutes: int,
    ) -> List[Insight]:
        """Handle threshold-based rules like error rates."""
        threshold = rule_config["threshold"]
        window_minutes = rule_config.get("window_minutes", time_window_minutes)

        # Group entries by time windows
        time_buckets = defaultdict(list)

        for entry in all_entries:
            if entry.timestamp:
                bucket_key = entry.timestamp.replace(second=0, microsecond=0)
                # Round to window_minutes
                bucket_key = bucket_key.replace(
                    minute=(bucket_key.minute // window_minutes) * window_minutes
                )
                time_buckets[bucket_key].append(entry)

        insights = []

        for bucket_time, bucket_entries in time_buckets.items():
            if len(bucket_entries) < 10:  # Skip buckets with too few entries
                continue

            matching_in_bucket = [e for e in bucket_entries if e in matching_entries]
            error_rate = len(matching_in_bucket) / len(bucket_entries)

            if error_rate > threshold:
                insight = Insight(
                    rule_name=rule_name,
                    confidence=min(1.0, error_rate / threshold),
                    description=f"{rule_config['description']} (rate: {error_rate:.1%})",
                    evidence=matching_in_bucket[:5],
                    timestamp=bucket_time,
                    severity=rule_config["severity"],
                    category=rule_config["category"],
                    metadata={
                        "error_rate": error_rate,
                        "threshold": threshold,
                        "total_entries": len(bucket_entries),
                        "matching_entries": len(matching_in_bucket),
                    },
                )
                insights.append(insight)

        return insights
