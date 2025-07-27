import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from spottr.analysis.temporal import TemporalAnalyzer
from spottr.core.models import Insight, LogEntry


class CorrelationType(Enum):
    TEMPORAL = "temporal"  # Events happening at similar times
    CAUSAL = "causal"  # One event causing another
    CASCADE = "cascade"  # Failure propagating through systems
    SYNCHRONIZED = "synchronized"  # Coordinated events across services
    ROOT_CAUSE = "root_cause"  # Multiple symptoms from same cause


@dataclass
class LogSource:
    """Represents a single log file/source."""

    file_path: str
    service_name: str
    entries: List[LogEntry]
    insights: List[Insight]
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        # Auto-detect service name from file path if not provided
        if not self.service_name and self.file_path:
            path = Path(self.file_path)
            # Extract service name from filename patterns
            if "user" in path.name.lower():
                self.service_name = "user-service"
            elif "payment" in path.name.lower():
                self.service_name = "payment-service"
            elif "auth" in path.name.lower():
                self.service_name = "auth-service"
            elif "database" in path.name.lower() or "db" in path.name.lower():
                self.service_name = "database"
            else:
                self.service_name = path.stem


@dataclass
class CorrelatedInsight:
    """Represents insights correlated across multiple sources."""

    correlation_id: str
    correlation_type: CorrelationType
    primary_insight: Insight
    related_insights: List[Tuple[LogSource, Insight]]
    confidence: float
    description: str
    time_window: Tuple[datetime, datetime]
    affected_services: Set[str]
    root_cause_analysis: Optional[str] = None
    propagation_path: Optional[List[str]] = None
    metadata: Dict = field(default_factory=dict)


class ServiceDependencyGraph:
    """Models dependencies between services for correlation analysis."""

    def __init__(self):
        self.dependencies = defaultdict(set)  # service -> set of dependencies
        self.reverse_deps = defaultdict(set)  # service -> set of dependents

    def add_dependency(self, service: str, depends_on: str):
        """Add a dependency relationship."""
        self.dependencies[service].add(depends_on)
        self.reverse_deps[depends_on].add(service)

    def get_downstream_services(self, service: str, max_hops: int = 3) -> Set[str]:
        """Get all services that could be affected by this service failing."""
        downstream = set()
        to_visit = [(service, 0)]
        visited = set()

        while to_visit:
            current_service, hops = to_visit.pop(0)
            if current_service in visited or hops >= max_hops:
                continue

            visited.add(current_service)
            if hops > 0:  # Don't include the starting service
                downstream.add(current_service)

            # Add direct dependents
            for dependent in self.reverse_deps[current_service]:
                if dependent not in visited:
                    to_visit.append((dependent, hops + 1))

        return downstream

    def get_upstream_services(self, service: str, max_hops: int = 3) -> Set[str]:
        """Get all services that this service depends on."""
        upstream = set()
        to_visit = [(service, 0)]
        visited = set()

        while to_visit:
            current_service, hops = to_visit.pop(0)
            if current_service in visited or hops >= max_hops:
                continue

            visited.add(current_service)
            if hops > 0:
                upstream.add(current_service)

            # Add direct dependencies
            for dependency in self.dependencies[current_service]:
                if dependency not in visited:
                    to_visit.append((dependency, hops + 1))

        return upstream


class MultiFileCorrelationEngine:
    """Correlates insights across multiple log sources."""

    def __init__(self, correlation_window_minutes: int = 10):
        self.correlation_window = timedelta(minutes=correlation_window_minutes)
        self.dependency_graph = ServiceDependencyGraph()
        self._setup_default_dependencies()

    def _setup_default_dependencies(self):
        """Setup common microservice dependencies."""
        # Common patterns - can be customized
        self.dependency_graph.add_dependency("user-service", "database")
        self.dependency_graph.add_dependency("user-service", "auth-service")
        self.dependency_graph.add_dependency("payment-service", "database")
        self.dependency_graph.add_dependency("payment-service", "user-service")
        self.dependency_graph.add_dependency("order-service", "payment-service")
        self.dependency_graph.add_dependency("order-service", "inventory-service")
        self.dependency_graph.add_dependency("inventory-service", "database")
        self.dependency_graph.add_dependency("api-gateway", "user-service")
        self.dependency_graph.add_dependency("api-gateway", "payment-service")

    def add_log_source(
        self,
        file_path: str,
        entries: List[LogEntry],
        insights: List[Insight],
        service_name: str = None,
    ) -> LogSource:
        """Add a log source for correlation analysis."""
        return LogSource(
            file_path=file_path,
            service_name=service_name or "",
            entries=entries,
            insights=insights,
        )

    def correlate_insights(
        self, log_sources: List[LogSource]
    ) -> List[CorrelatedInsight]:
        """Find correlated insights across multiple log sources."""
        correlations = []

        # Different correlation strategies
        correlations.extend(self._find_temporal_correlations(log_sources))
        correlations.extend(self._find_cascade_failures(log_sources))
        correlations.extend(self._find_synchronized_events(log_sources))
        correlations.extend(self._find_root_cause_patterns(log_sources))

        # Remove duplicates and sort by confidence
        unique_correlations = self._deduplicate_correlations(correlations)
        return sorted(unique_correlations, key=lambda x: x.confidence, reverse=True)

    def _find_temporal_correlations(
        self, log_sources: List[LogSource]
    ) -> List[CorrelatedInsight]:
        """Find insights that occur close in time across different sources."""
        correlations = []

        # Group insights by time windows
        time_buckets = defaultdict(list)  # time_bucket -> [(source, insight)]

        for source in log_sources:
            for insight in source.insights:
                if insight.timestamp:
                    # Round to correlation window
                    bucket_time = self._round_to_window(insight.timestamp)
                    time_buckets[bucket_time].append((source, insight))

        # Find buckets with insights from multiple sources
        for bucket_time, source_insights in time_buckets.items():
            if len(source_insights) < 2:
                continue

            # Group by source to avoid duplicates from same source
            sources_in_bucket = defaultdict(list)
            for source, insight in source_insights:
                sources_in_bucket[source.service_name].append((source, insight))

            if len(sources_in_bucket) >= 2:  # At least 2 different services
                # Pick the highest confidence insight as primary
                primary_source, primary_insight = max(
                    source_insights, key=lambda x: x[1].confidence
                )
                related = [
                    (s, i)
                    for s, i in source_insights
                    if s.service_name != primary_source.service_name
                ]

                if related:
                    affected_services = {primary_source.service_name} | {
                        s.service_name for s, _ in related
                    }

                    correlation = CorrelatedInsight(
                        correlation_id=self._generate_correlation_id(
                            bucket_time, affected_services
                        ),
                        correlation_type=CorrelationType.TEMPORAL,
                        primary_insight=primary_insight,
                        related_insights=related[:5],  # Limit to top 5
                        confidence=self._calculate_temporal_confidence(source_insights),
                        description=f"Temporal correlation across {len(affected_services)} services",
                        time_window=(
                            bucket_time,
                            bucket_time + self.correlation_window,
                        ),
                        affected_services=affected_services,
                        metadata={
                            "bucket_time": bucket_time,
                            "total_insights": len(source_insights),
                            "services_count": len(affected_services),
                        },
                    )
                    correlations.append(correlation)

        return correlations

    def _find_cascade_failures(
        self, log_sources: List[LogSource]
    ) -> List[CorrelatedInsight]:
        """Find failure cascades following dependency chains."""
        correlations = []

        # Look for patterns where upstream service fails, then downstream services fail
        for source in log_sources:
            for insight in source.insights:
                if insight.severity in {"HIGH", "CRITICAL"} and insight.timestamp:
                    # Find downstream services that could be affected
                    downstream = self.dependency_graph.get_downstream_services(
                        source.service_name
                    )

                    if downstream:
                        # Look for failures in downstream services within time window
                        cascade_failures = []
                        window_start = insight.timestamp
                        window_end = window_start + self.correlation_window

                        for other_source in log_sources:
                            if other_source.service_name in downstream:
                                for other_insight in other_source.insights:
                                    if (
                                        other_insight.timestamp
                                        and window_start
                                        <= other_insight.timestamp
                                        <= window_end
                                        and other_insight.severity
                                        in {"MEDIUM", "HIGH", "CRITICAL"}
                                    ):
                                        cascade_failures.append(
                                            (other_source, other_insight)
                                        )

                        if cascade_failures:
                            # Build propagation path
                            propagation_path = [source.service_name]
                            propagation_path.extend(
                                [s.service_name for s, _ in cascade_failures]
                            )

                            affected_services = {source.service_name} | {
                                s.service_name for s, _ in cascade_failures
                            }

                            correlation = CorrelatedInsight(
                                correlation_id=self._generate_correlation_id(
                                    insight.timestamp, affected_services
                                ),
                                correlation_type=CorrelationType.CASCADE,
                                primary_insight=insight,
                                related_insights=cascade_failures,
                                confidence=self._calculate_cascade_confidence(
                                    insight, cascade_failures
                                ),
                                description=f"Cascade failure: {source.service_name} → {len(cascade_failures)} downstream services",
                                time_window=(window_start, window_end),
                                affected_services=affected_services,
                                propagation_path=propagation_path,
                                root_cause_analysis=f"Root cause appears to be in {source.service_name}: {insight.description}",
                                metadata={
                                    "cascade_depth": len(cascade_failures),
                                    "propagation_time": (
                                        window_end - window_start
                                    ).total_seconds(),
                                },
                            )
                            correlations.append(correlation)

        return correlations

    def _find_synchronized_events(
        self, log_sources: List[LogSource]
    ) -> List[CorrelatedInsight]:
        """Find synchronized events that suggest coordinated activity."""
        correlations = []

        # Look for very similar insights happening simultaneously
        for i, source1 in enumerate(log_sources):
            for j, source2 in enumerate(log_sources[i + 1 :], i + 1):
                # Compare insights between these two sources
                for insight1 in source1.insights:
                    for insight2 in source2.insights:
                        if (
                            insight1.timestamp
                            and insight2.timestamp
                            and abs(
                                (
                                    insight1.timestamp - insight2.timestamp
                                ).total_seconds()
                            )
                            < 60
                        ):  # Within 1 minute
                            # Check for similarity in categories or patterns
                            if (
                                insight1.category == insight2.category
                                or insight1.rule_name == insight2.rule_name
                                or self._insights_semantically_similar(
                                    insight1, insight2
                                )
                            ):
                                affected_services = {
                                    source1.service_name,
                                    source2.service_name,
                                }

                                correlation = CorrelatedInsight(
                                    correlation_id=self._generate_correlation_id(
                                        insight1.timestamp, affected_services
                                    ),
                                    correlation_type=CorrelationType.SYNCHRONIZED,
                                    primary_insight=insight1
                                    if insight1.confidence >= insight2.confidence
                                    else insight2,
                                    related_insights=[
                                        (
                                            source2
                                            if insight1.confidence
                                            >= insight2.confidence
                                            else source1,
                                            insight2
                                            if insight1.confidence
                                            >= insight2.confidence
                                            else insight1,
                                        )
                                    ],
                                    confidence=min(
                                        insight1.confidence, insight2.confidence
                                    )
                                    * 1.2,  # Boost for correlation
                                    description=f"Synchronized {insight1.category.lower()} events across services",
                                    time_window=(
                                        min(insight1.timestamp, insight2.timestamp),
                                        max(insight1.timestamp, insight2.timestamp),
                                    ),
                                    affected_services=affected_services,
                                    metadata={
                                        "time_delta_seconds": abs(
                                            (
                                                insight1.timestamp - insight2.timestamp
                                            ).total_seconds()
                                        ),
                                        "pattern_similarity": "category"
                                        if insight1.category == insight2.category
                                        else "rule",
                                    },
                                )
                                correlations.append(correlation)

        return correlations

    def _find_root_cause_patterns(
        self, log_sources: List[LogSource]
    ) -> List[CorrelatedInsight]:
        """Find patterns suggesting common root causes."""
        correlations = []

        # Group insights by potential root causes
        potential_roots = {
            "database": ["database", "connection", "query", "timeout", "pool"],
            "memory": ["memory", "heap", "OutOfMemoryError", "allocation"],
            "network": ["network", "connection", "timeout", "unreachable"],
            "authentication": ["auth", "login", "credentials", "unauthorized"],
        }

        for root_cause, keywords in potential_roots.items():
            related_insights = []

            for source in log_sources:
                for insight in source.insights:
                    # Check if insight description contains root cause keywords
                    description_lower = insight.description.lower()
                    if any(keyword in description_lower for keyword in keywords):
                        related_insights.append((source, insight))

            # If we have multiple services showing the same type of issue
            if len(related_insights) >= 2:
                services_involved = {s.service_name for s, _ in related_insights}

                if len(services_involved) >= 2:  # At least 2 different services
                    # Find time window covering all related insights
                    timestamps = [
                        i.timestamp for _, i in related_insights if i.timestamp
                    ]
                    if timestamps:
                        time_window = (min(timestamps), max(timestamps))

                        # Pick most confident insight as primary
                        primary_source, primary_insight = max(
                            related_insights, key=lambda x: x[1].confidence
                        )
                        related = [
                            (s, i) for s, i in related_insights if s != primary_source
                        ]

                        correlation = CorrelatedInsight(
                            correlation_id=self._generate_correlation_id(
                                primary_insight.timestamp, services_involved
                            ),
                            correlation_type=CorrelationType.ROOT_CAUSE,
                            primary_insight=primary_insight,
                            related_insights=related,
                            confidence=self._calculate_root_cause_confidence(
                                related_insights
                            ),
                            description=f"Common root cause pattern: {root_cause} issues across {len(services_involved)} services",
                            time_window=time_window,
                            affected_services=services_involved,
                            root_cause_analysis=f"Multiple services showing {root_cause}-related issues suggests common infrastructure problem",
                            metadata={
                                "root_cause_type": root_cause,
                                "affected_insights_count": len(related_insights),
                                "time_span_minutes": (
                                    time_window[1] - time_window[0]
                                ).total_seconds()
                                / 60,
                            },
                        )
                        correlations.append(correlation)

        return correlations

    def _round_to_window(self, timestamp: datetime) -> datetime:
        """Round timestamp to correlation window boundary."""
        minutes = self.correlation_window.total_seconds() / 60
        rounded_minute = (timestamp.minute // minutes) * minutes
        return timestamp.replace(minute=int(rounded_minute), second=0, microsecond=0)

    def _generate_correlation_id(self, timestamp: datetime, services: Set[str]) -> str:
        """Generate unique correlation ID."""
        content = f"{timestamp.isoformat()}_{sorted(services)}"
        return hashlib.md5(content.encode()).hexdigest()[:8]

    def _calculate_temporal_confidence(
        self, source_insights: List[Tuple[LogSource, Insight]]
    ) -> float:
        """Calculate confidence for temporal correlations."""
        if not source_insights:
            return 0.0

        # Base confidence on average insight confidence and number of services
        avg_confidence = sum(
            insight.confidence for _, insight in source_insights
        ) / len(source_insights)
        service_count = len({source.service_name for source, _ in source_insights})

        # Boost confidence for more services involved
        service_boost = min(1.0, (service_count - 1) * 0.2)
        return min(1.0, avg_confidence + service_boost)

    def _calculate_cascade_confidence(
        self,
        primary_insight: Insight,
        cascade_failures: List[Tuple[LogSource, Insight]],
    ) -> float:
        """Calculate confidence for cascade failures."""
        base_confidence = primary_insight.confidence

        # Boost based on number of affected services and their severity
        severity_weights = {"LOW": 0.1, "MEDIUM": 0.3, "HIGH": 0.5, "CRITICAL": 0.7}
        cascade_score = sum(
            severity_weights.get(insight.severity, 0.2)
            for _, insight in cascade_failures
        )

        # Normalize and combine
        cascade_boost = min(
            0.4, cascade_score / len(cascade_failures) if cascade_failures else 0
        )
        return min(1.0, base_confidence + cascade_boost)

    def _calculate_root_cause_confidence(
        self, related_insights: List[Tuple[LogSource, Insight]]
    ) -> float:
        """Calculate confidence for root cause correlations."""
        if not related_insights:
            return 0.0

        # Higher confidence when more services show similar issues
        service_count = len({source.service_name for source, _ in related_insights})
        avg_confidence = sum(
            insight.confidence for _, insight in related_insights
        ) / len(related_insights)

        # Boost confidence based on service diversity
        diversity_boost = min(0.3, (service_count - 1) * 0.1)
        return min(1.0, avg_confidence + diversity_boost)

    def _insights_semantically_similar(
        self, insight1: Insight, insight2: Insight
    ) -> bool:
        """Check if two insights are semantically similar."""
        # Simple keyword-based similarity - could be enhanced with embeddings
        words1 = set(insight1.description.lower().split())
        words2 = set(insight2.description.lower().split())

        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return (intersection / union) > 0.3 if union > 0 else False

    def _deduplicate_correlations(
        self, correlations: List[CorrelatedInsight]
    ) -> List[CorrelatedInsight]:
        """Remove duplicate correlations."""
        seen_ids = set()
        unique_correlations = []

        for correlation in correlations:
            if correlation.correlation_id not in seen_ids:
                seen_ids.add(correlation.correlation_id)
                unique_correlations.append(correlation)

        return unique_correlations


# Enhanced CLI interface for multi-file analysis
class MultiFileAnalyzer:
    """Orchestrates analysis across multiple log files."""

    def __init__(self, base_analyzer, correlation_window_minutes: int = 10):
        self.base_analyzer = base_analyzer
        self.correlation_engine = MultiFileCorrelationEngine(correlation_window_minutes)
        self.temporal_analyzer = TemporalAnalyzer()

    def analyze_multiple_files(
        self,
        file_paths: List[str],
        target_statements: List[str] = None,
        service_names: List[str] = None,
    ) -> Dict:
        """Analyze multiple log files and find correlations."""

        log_sources = []
        all_temporal_insights = []

        print(f"Analyzing {len(file_paths)} log files...")

        # Analyze each file individually
        for i, file_path in enumerate(file_paths):
            service_name = (
                service_names[i] if service_names and i < len(service_names) else None
            )

            print(f"Processing {file_path}...")

            # Parse and analyze single file
            entries = self.base_analyzer.parse_log_file(file_path)

            if target_statements:
                _, insights = self.base_analyzer.analyze_file_with_targets(
                    file_path, target_statements
                )
            else:
                _, insights = self.base_analyzer.analyze_file(file_path)

            # Add temporal analysis
            temporal_insights = self.temporal_analyzer.analyze_temporal_patterns(
                entries
            )
            all_temporal_insights.extend(temporal_insights)

            # Create log source
            log_source = self.correlation_engine.add_log_source(
                file_path, entries, insights, service_name
            )
            log_sources.append(log_source)

            print(
                f"  Found {len(insights)} insights, {len(temporal_insights)} temporal patterns"
            )

        # Find correlations across files
        print("Finding correlations across log files...")
        correlations = self.correlation_engine.correlate_insights(log_sources)

        # Generate comprehensive report
        return self._generate_multi_file_report(
            log_sources, correlations, all_temporal_insights
        )

    def _generate_multi_file_report(
        self,
        log_sources: List[LogSource],
        correlations: List[CorrelatedInsight],
        temporal_insights: List,
    ) -> Dict:
        """Generate comprehensive multi-file analysis report."""

        # Aggregate statistics
        total_entries = sum(len(source.entries) for source in log_sources)
        total_insights = sum(len(source.insights) for source in log_sources)

        # Service-level statistics
        service_stats = {}
        for source in log_sources:
            service_stats[source.service_name] = {
                "log_entries": len(source.entries),
                "insights": len(source.insights),
                "file_path": source.file_path,
                "avg_confidence": sum(i.confidence for i in source.insights)
                / len(source.insights)
                if source.insights
                else 0,
            }

        # Correlation statistics
        correlation_stats = {
            "total_correlations": len(correlations),
            "by_type": {},
            "affected_services": set(),
            "high_confidence_correlations": 0,
        }

        for corr in correlations:
            corr_type = corr.correlation_type.value
            correlation_stats["by_type"][corr_type] = (
                correlation_stats["by_type"].get(corr_type, 0) + 1
            )
            correlation_stats["affected_services"].update(corr.affected_services)
            if corr.confidence > 0.7:
                correlation_stats["high_confidence_correlations"] += 1

        correlation_stats["affected_services"] = list(
            correlation_stats["affected_services"]
        )

        # Top correlations for summary
        top_correlations = sorted(
            correlations, key=lambda x: x.confidence, reverse=True
        )[:5]

        report = {
            "summary": {
                "files_analyzed": len(log_sources),
                "total_log_entries": total_entries,
                "total_insights": total_insights,
                "temporal_patterns": len(temporal_insights),
                "correlations_found": len(correlations),
                "services_involved": list(service_stats.keys()),
            },
            "service_analysis": service_stats,
            "correlation_analysis": correlation_stats,
            "top_correlations": [
                self._correlation_to_dict(corr) for corr in top_correlations
            ],
            "temporal_insights": [
                self._temporal_insight_to_dict(ti) for ti in temporal_insights[:10]
            ],
            "individual_insights": {
                source.service_name: [
                    self._insight_to_dict(insight) for insight in source.insights[:5]
                ]
                for source in log_sources
            },
        }

        return report

    def _correlation_to_dict(self, correlation: CorrelatedInsight) -> Dict:
        """Convert correlation to dictionary for JSON serialization."""
        return {
            "correlation_id": correlation.correlation_id,
            "type": correlation.correlation_type.value,
            "confidence": correlation.confidence,
            "description": correlation.description,
            "affected_services": list(correlation.affected_services),
            "time_window": [
                correlation.time_window[0].isoformat(),
                correlation.time_window[1].isoformat(),
            ],
            "primary_insight": correlation.primary_insight.description,
            "related_insights_count": len(correlation.related_insights),
            "root_cause_analysis": correlation.root_cause_analysis,
            "propagation_path": correlation.propagation_path,
            "metadata": correlation.metadata,
        }

    def _temporal_insight_to_dict(self, temporal_insight) -> Dict:
        """Convert temporal insight to dictionary."""
        return {
            "pattern_type": temporal_insight.pattern_type.value,
            "description": temporal_insight.description,
            "confidence": temporal_insight.confidence,
            "trend_direction": temporal_insight.trend_direction,
            "velocity": temporal_insight.velocity,
            "time_windows_analyzed": len(temporal_insight.time_windows),
            "periodicity": str(temporal_insight.periodicity)
            if temporal_insight.periodicity
            else None,
            "severity": temporal_insight.severity,
            "metadata": temporal_insight.metadata,
        }

    def _insight_to_dict(self, insight: Insight) -> Dict:
        """Convert standard insight to dictionary."""
        return {
            "rule_name": insight.rule_name,
            "confidence": insight.confidence,
            "description": insight.description,
            "severity": insight.severity,
            "category": insight.category,
            "timestamp": insight.timestamp.isoformat() if insight.timestamp else None,
            "evidence_count": len(insight.evidence),
        }


# Usage example and integration
def create_enhanced_cli():
    """Create enhanced CLI with temporal and correlation capabilities."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced Log Analysis with Temporal Patterns and Multi-File Correlation"
    )

    # File inputs
    parser.add_argument("log_files", nargs="+", help="Log files to analyze")
    parser.add_argument(
        "--service-names", nargs="*", help="Service names corresponding to log files"
    )

    # Analysis options
    parser.add_argument(
        "-t",
        "--targets",
        nargs="+",
        default=[
            "system performance issues",
            "authentication problems",
            "memory and resource exhaustion",
            "database performance issues",
        ],
        help="Target statements for entailment scoring",
    )

    parser.add_argument(
        "--correlation-window",
        type=int,
        default=10,
        help="Correlation time window in minutes (default: 10)",
    )

    parser.add_argument(
        "--temporal-window",
        type=int,
        default=5,
        help="Temporal analysis window in minutes (default: 5)",
    )

    # Output options
    parser.add_argument("-o", "--output", help="Output file for JSON report")
    parser.add_argument(
        "--include-temporal",
        action="store_true",
        default=True,
        help="Include temporal pattern analysis",
    )
    parser.add_argument(
        "--include-correlation",
        action="store_true",
        default=True,
        help="Include multi-file correlation analysis",
    )

    return parser


# Example usage
if __name__ == "__main__":
    # This would be integrated into the main CLI
    print("Enhanced Spottr with Temporal Analysis and Multi-File Correlation")
    print("Usage examples:")
    print("  # Analyze multiple service logs")
    print("  spottr logs/user-service.log logs/payment-service.log logs/database.log")
    print("  ")
    print("  # With service names and custom correlation window")
    print(
        "  spottr logs/*.log --service-names user payment database --correlation-window 15"
    )
    print("  ")
    print("  # Focus on specific patterns")
    print(
        "  spottr logs/*.log -t 'authentication failures' 'payment processing errors'"
    )
