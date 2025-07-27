from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from spottr.core.models import Insight, LogEntry


class TemporalPattern(Enum):
    ESCALATION = "escalation"  # Errors increasing over time
    BURST = "burst"  # Sudden spike in activity
    PERIODIC = "periodic"  # Recurring patterns
    DEGRADATION = "degradation"  # Performance getting worse
    RECOVERY = "recovery"  # System recovering from issues
    CORRELATION = "correlation"  # Events happening together


@dataclass
class TemporalWindow:
    """Represents a time window for analysis."""

    start_time: datetime
    end_time: datetime
    duration: timedelta
    entries: List[LogEntry]

    @property
    def entry_count(self) -> int:
        return len(self.entries)

    @property
    def error_rate(self) -> float:
        if not self.entries:
            return 0.0
        error_levels = {"ERROR", "FATAL", "CRITICAL"}
        error_count = sum(1 for entry in self.entries if entry.level in error_levels)
        return error_count / len(self.entries)


@dataclass
class TemporalInsight(Insight):
    """Enhanced insight with temporal information."""

    pattern_type: TemporalPattern = TemporalPattern.ESCALATION
    time_windows: List[TemporalWindow] = None
    trend_direction: str = ""
    velocity: float = 0.0
    periodicity: Optional[timedelta] = None
    correlation_lag: Optional[timedelta] = None


class TemporalAnalyzer:
    """Analyzes temporal patterns in log data."""

    def __init__(self, window_size_minutes: int = 5, overlap_percentage: float = 0.5):
        self.window_size = timedelta(minutes=window_size_minutes)
        self.overlap = timedelta(minutes=window_size_minutes * overlap_percentage)

    def create_time_windows(self, entries: List[LogEntry]) -> List[TemporalWindow]:
        """Create overlapping time windows from log entries."""
        if not entries:
            return []

        # Sort entries by timestamp
        sorted_entries = sorted(
            [e for e in entries if e.timestamp], key=lambda x: x.timestamp
        )

        if not sorted_entries:
            return []

        windows = []
        start_time = sorted_entries[0].timestamp
        end_time = sorted_entries[-1].timestamp

        current_start = start_time
        while current_start < end_time:
            current_end = current_start + self.window_size

            # Get entries in this window
            window_entries = [
                entry
                for entry in sorted_entries
                if current_start <= entry.timestamp < current_end
            ]

            if window_entries:  # Only create window if it has entries
                windows.append(
                    TemporalWindow(
                        start_time=current_start,
                        end_time=current_end,
                        duration=self.window_size,
                        entries=window_entries,
                    )
                )

            current_start += self.window_size - self.overlap

        return windows

    def detect_escalation_patterns(
        self, windows: List[TemporalWindow]
    ) -> List[TemporalInsight]:
        """Detect error escalation patterns."""
        insights = []

        if len(windows) < 3:
            return insights

        # Calculate error rates for each window
        error_rates = [w.error_rate for w in windows]

        # Look for increasing trends
        for i in range(len(windows) - 2):
            window_slice = windows[i : i + 3]
            rate_slice = error_rates[i : i + 3]

            # Check for consistent increase
            if all(
                rate_slice[j] < rate_slice[j + 1] for j in range(len(rate_slice) - 1)
            ):
                velocity = (rate_slice[-1] - rate_slice[0]) / len(rate_slice)

                if velocity > 0.05:  # Significant increase
                    evidence_entries = []
                    for w in window_slice:
                        evidence_entries.extend(
                            [
                                e
                                for e in w.entries
                                if e.level in {"ERROR", "FATAL", "CRITICAL"}
                            ]
                        )

                    insight = TemporalInsight(
                        rule_name="temporal_escalation",
                        confidence=min(
                            1.0, velocity * 10
                        ),  # Scale velocity to confidence
                        description=f"Error escalation detected: {rate_slice[0]:.1%} → {rate_slice[-1]:.1%}",
                        evidence=evidence_entries[:10],
                        timestamp=window_slice[0].start_time,
                        severity="HIGH" if velocity > 0.15 else "MEDIUM",
                        category="TEMPORAL",
                        pattern_type=TemporalPattern.ESCALATION,
                        time_windows=window_slice,
                        trend_direction="increasing",
                        velocity=velocity,
                        metadata={
                            "start_error_rate": rate_slice[0],
                            "end_error_rate": rate_slice[-1],
                            "windows_analyzed": len(window_slice),
                        },
                    )
                    insights.append(insight)

        return insights

    def detect_burst_patterns(
        self, windows: List[TemporalWindow]
    ) -> List[TemporalInsight]:
        """Detect sudden activity bursts."""
        insights = []

        if len(windows) < 3:
            return insights

        entry_counts = [w.entry_count for w in windows]
        mean_count = np.mean(entry_counts)
        std_count = np.std(entry_counts)

        threshold = mean_count + (2 * std_count)  # 2 standard deviations

        for i, window in enumerate(windows):
            if window.entry_count > threshold and window.entry_count > mean_count * 2:
                # Check if this is a isolated burst (neighbors are normal)
                neighbors_normal = True
                for j in range(max(0, i - 1), min(len(windows), i + 2)):
                    if j != i and windows[j].entry_count > mean_count * 1.5:
                        neighbors_normal = False
                        break

                if neighbors_normal:
                    burst_intensity = window.entry_count / mean_count

                    insight = TemporalInsight(
                        rule_name="temporal_burst",
                        confidence=min(1.0, (burst_intensity - 2) / 3),
                        description=f"Activity burst detected: {window.entry_count} entries ({burst_intensity:.1f}x normal)",
                        evidence=window.entries[:15],
                        timestamp=window.start_time,
                        severity="HIGH" if burst_intensity > 5 else "MEDIUM",
                        category="TEMPORAL",
                        pattern_type=TemporalPattern.BURST,
                        time_windows=[window],
                        trend_direction="spike",
                        velocity=burst_intensity,
                        metadata={
                            "burst_intensity": burst_intensity,
                            "normal_average": mean_count,
                            "burst_count": window.entry_count,
                        },
                    )
                    insights.append(insight)

        return insights

    def detect_periodic_patterns(
        self, windows: List[TemporalWindow]
    ) -> List[TemporalInsight]:
        """Detect periodic/recurring patterns."""
        insights = []

        if len(windows) < 6:  # Need enough data for pattern detection
            return insights

        # Look for recurring error spikes
        error_rates = [w.error_rate for w in windows]

        # Simple periodicity detection using autocorrelation
        for period in range(
            2, min(len(windows) // 2, 12)
        ):  # Check periods up to 12 windows
            correlations = []
            for offset in range(len(windows) - period):
                if offset + period < len(error_rates):
                    correlations.append(
                        error_rates[offset] * error_rates[offset + period]
                    )

            if correlations and np.mean(correlations) > 0.1:  # Significant correlation
                # Find the recurring pattern instances
                pattern_windows = []
                for i in range(0, len(windows), period):
                    if i + period <= len(windows):
                        pattern_windows.extend(windows[i : i + period])

                if len(pattern_windows) >= period * 2:  # At least 2 full cycles
                    evidence_entries = []
                    for w in pattern_windows:
                        evidence_entries.extend(
                            [e for e in w.entries if e.level in {"ERROR", "WARNING"}]
                        )

                    period_duration = timedelta(
                        minutes=self.window_size.total_seconds() / 60 * period
                    )

                    insight = TemporalInsight(
                        rule_name="temporal_periodic",
                        confidence=min(1.0, np.mean(correlations) * 5),
                        description=f"Periodic pattern detected: recurring every {period_duration}",
                        evidence=evidence_entries[:10],
                        timestamp=windows[0].start_time,
                        severity="MEDIUM",
                        category="TEMPORAL",
                        pattern_type=TemporalPattern.PERIODIC,
                        time_windows=pattern_windows[
                            : period * 2
                        ],  # Show first 2 cycles
                        trend_direction="periodic",
                        velocity=np.mean(correlations),
                        periodicity=period_duration,
                        metadata={
                            "period_windows": period,
                            "correlation_strength": np.mean(correlations),
                            "cycles_detected": len(pattern_windows) // period,
                        },
                    )
                    insights.append(insight)
                    break  # Only report the strongest period

        return insights

    def analyze_temporal_patterns(
        self, entries: List[LogEntry]
    ) -> List[TemporalInsight]:
        """Main method to analyze all temporal patterns."""
        windows = self.create_time_windows(entries)

        if not windows:
            return []

        all_insights = []

        # Detect different pattern types
        all_insights.extend(self.detect_escalation_patterns(windows))
        all_insights.extend(self.detect_burst_patterns(windows))
        all_insights.extend(self.detect_periodic_patterns(windows))

        # Sort by confidence and timestamp
        all_insights.sort(key=lambda x: (x.confidence, x.timestamp), reverse=True)

        return all_insights


# Integration with existing LogAnalyzer
class TemporalLogAnalyzer:
    """Enhanced LogAnalyzer with temporal capabilities."""

    def __init__(self, base_analyzer, window_size_minutes: int = 5):
        self.base_analyzer = base_analyzer
        self.temporal_analyzer = TemporalAnalyzer(window_size_minutes)

    def analyze_with_temporal(
        self, entries: List[LogEntry], target_statements: List[str] = None
    ) -> Tuple[List[Insight], List[TemporalInsight]]:
        """Analyze logs with both standard and temporal patterns."""

        # Get standard insights
        if target_statements:
            standard_insights = self.base_analyzer.rule_engine.apply_rules(entries)
            # Add entailment if available
            if (
                self.base_analyzer.use_entailment
                and self.base_analyzer.entailment_scorer
            ):
                enhanced_insights = self.base_analyzer._add_entailment_scoring(
                    entries, standard_insights, target_statements
                )
                entailment_insights = self.base_analyzer._generate_entailment_insights(
                    entries, target_statements
                )
                standard_insights = enhanced_insights + entailment_insights
        else:
            standard_insights = self.base_analyzer.rule_engine.apply_rules(entries)

        # Get temporal insights
        temporal_insights = self.temporal_analyzer.analyze_temporal_patterns(entries)

        return standard_insights, temporal_insights
