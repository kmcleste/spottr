import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spottr.analysis.correlation import (
    CorrelatedInsight,
    LogSource,
    MultiFileCorrelationEngine,
)
from spottr.analysis.entailment import EntailmentScorer
from spottr.analysis.rule_engine import RuleEngine
from spottr.analysis.temporal import TemporalAnalyzer, TemporalInsight
from spottr.core.models import Insight, LogEntry, LogFormat
from spottr.parsers.factory import LogParserFactory
from spottr.utils.validation import validate_log_entries


class EnhancedLogAnalyzer:
    """Enhanced log analysis orchestrator with temporal patterns and multi-file correlation."""

    def __init__(
        self,
        use_entailment=True,
        entailment_model_id="NorGLM/Entailment",
        log_format: Optional[LogFormat] = None,
        temporal_window_minutes: int = 5,
        correlation_window_minutes: int = 10,
    ):
        # Existing components
        self.parser_factory = LogParserFactory()
        self.rule_engine = RuleEngine()
        self.logger = logging.getLogger(__name__)
        self.log_format = log_format

        self.use_entailment = use_entailment
        self.entailment_scorer = None

        if use_entailment:
            try:
                self.entailment_scorer = EntailmentScorer(entailment_model_id)
            except Exception as e:
                self.logger.warning(f"Failed to load entailment model: {e}")
                self.logger.warning("Continuing without entailment scoring")
                self.use_entailment = False

        # New components
        self.temporal_analyzer = TemporalAnalyzer(temporal_window_minutes)
        self.correlation_engine = MultiFileCorrelationEngine(correlation_window_minutes)

    # Existing methods remain the same
    def analyze_file(
        self, file_path: str, time_window_minutes: int = 5
    ) -> Tuple[List[LogEntry], List[Insight]]:
        """Analyze a single log file (backward compatibility)."""
        entries = self.parse_log_file(file_path)
        insights = self.rule_engine.apply_rules(entries, time_window_minutes)
        return entries, insights

    def analyze_file_with_targets(
        self, file_path: str, target_statements: List[str], time_window_minutes: int = 5
    ) -> Tuple[List[LogEntry], List[Insight]]:
        """Analyze a single file with entailment targets (backward compatibility)."""
        # Use existing implementation
        try:
            entries = self.parse_log_file(file_path)
            print(f"Parsed {len(entries)} log entries")

            rule_insights = self.rule_engine.apply_rules(entries, time_window_minutes)
            print(f"Found {len(rule_insights)} rule-based insights")

            if self.use_entailment and self.entailment_scorer and target_statements:
                print("Starting entailment scoring...")
                try:
                    enhanced_insights = self._add_entailment_scoring(
                        entries, rule_insights, target_statements
                    )
                    entailment_insights = self._generate_entailment_insights(
                        entries, target_statements
                    )
                    all_insights = enhanced_insights + entailment_insights
                except Exception as e:
                    print(f"Error during entailment scoring: {e}")
                    all_insights = rule_insights
            else:
                all_insights = rule_insights

            return entries, all_insights

        except Exception as e:
            print(f"Error in analyze_file_with_targets: {e}")
            import traceback

            traceback.print_exc()
            raise

    # NEW: Enhanced single-file analysis with temporal patterns
    def analyze_file_enhanced(
        self,
        file_path: str,
        target_statements: List[str] = None,
        include_temporal: bool = True,
        time_window_minutes: int = 5,
    ) -> Dict[str, Any]:
        """Enhanced single-file analysis with temporal patterns."""

        # Get standard analysis
        if target_statements:
            entries, standard_insights = self.analyze_file_with_targets(
                file_path, target_statements, time_window_minutes
            )
        else:
            entries, standard_insights = self.analyze_file(
                file_path, time_window_minutes
            )

        result = {
            "entries": entries,
            "standard_insights": standard_insights,
            "temporal_insights": [],
            "summary": self._generate_enhanced_summary(entries, standard_insights, []),
        }

        # Add temporal analysis if requested
        if include_temporal:
            temporal_insights = self.temporal_analyzer.analyze_temporal_patterns(
                entries
            )
            result["temporal_insights"] = temporal_insights
            result["summary"] = self._generate_enhanced_summary(
                entries, standard_insights, temporal_insights
            )
            print(f"Found {len(temporal_insights)} temporal patterns")

        return result

    # NEW: Multi-file analysis with correlation
    def analyze_multiple_files(
        self,
        file_paths: List[str],
        target_statements: List[str] = None,
        service_names: List[str] = None,
        include_temporal: bool = True,
        include_correlation: bool = True,
        time_window_minutes: int = 5,
    ) -> Dict[str, Any]:
        """Analyze multiple log files with correlation detection."""

        log_sources = []
        all_temporal_insights = []

        print(f"Analyzing {len(file_paths)} log files...")

        # Analyze each file
        for i, file_path in enumerate(file_paths):
            service_name = (
                service_names[i] if service_names and i < len(service_names) else None
            )

            print(f"Processing {Path(file_path).name}...")

            # Get standard analysis
            if target_statements:
                entries, insights = self.analyze_file_with_targets(
                    file_path, target_statements, time_window_minutes
                )
            else:
                entries, insights = self.analyze_file(file_path, time_window_minutes)

            # Add temporal analysis if requested
            temporal_insights = []
            if include_temporal:
                temporal_insights = self.temporal_analyzer.analyze_temporal_patterns(
                    entries
                )
                all_temporal_insights.extend(temporal_insights)

            # Create log source for correlation
            log_source = self.correlation_engine.add_log_source(
                file_path, entries, insights, service_name
            )
            log_sources.append(log_source)

            print(
                f"  Found {len(insights)} insights, {len(temporal_insights)} temporal patterns"
            )

        # Find correlations if requested
        correlations = []
        if include_correlation and len(log_sources) > 1:
            print("Finding correlations across log files...")
            correlations = self.correlation_engine.correlate_insights(log_sources)
            print(f"Found {len(correlations)} correlations")

        # Generate comprehensive report
        return self._generate_multi_file_report(
            log_sources, correlations, all_temporal_insights
        )

    # NEW: Batch analysis for directories
    def analyze_directory(
        self,
        directory_path: str,
        file_pattern: str = "*.log",
        target_statements: List[str] = None,
        include_temporal: bool = True,
        include_correlation: bool = True,
        time_window_minutes: int = 5,
    ) -> Dict[str, Any]:
        """Analyze all log files in a directory."""

        directory = Path(directory_path)
        log_files = list(directory.glob(file_pattern))

        if not log_files:
            raise ValueError(
                f"No log files found matching pattern '{file_pattern}' in {directory_path}"
            )

        print(f"Found {len(log_files)} log files in {directory_path}")

        # Auto-detect service names from filenames
        service_names = []
        for file_path in log_files:
            # Extract potential service name from filename
            stem = file_path.stem.lower()
            if "user" in stem:
                service_names.append("user-service")
            elif "payment" in stem:
                service_names.append("payment-service")
            elif "auth" in stem:
                service_names.append("auth-service")
            elif "database" in stem or "db" in stem:
                service_names.append("database")
            elif "gateway" in stem or "api" in stem:
                service_names.append("api-gateway")
            elif "inventory" in stem:
                service_names.append("inventory-service")
            elif "notification" in stem:
                service_names.append("notification-service")
            else:
                service_names.append(stem)

        return self.analyze_multiple_files(
            [str(f) for f in log_files],
            target_statements=target_statements,
            service_names=service_names,
            include_temporal=include_temporal,
            include_correlation=include_correlation,
            time_window_minutes=time_window_minutes,
        )

    def _generate_enhanced_summary(
        self,
        entries: List[LogEntry],
        standard_insights: List[Insight],
        temporal_insights: List[TemporalInsight],
    ) -> Dict[str, Any]:
        """Generate enhanced summary including temporal patterns."""

        # Standard summary
        summary = {
            "total_entries": len(entries),
            "total_insights": len(standard_insights),
            "temporal_patterns": len(temporal_insights),
        }

        if standard_insights:
            summary["avg_confidence"] = sum(
                i.confidence for i in standard_insights
            ) / len(standard_insights)

            # Category and severity breakdown
            by_category = defaultdict(int)
            by_severity = defaultdict(int)

            for insight in standard_insights:
                by_category[insight.category] += 1
                by_severity[insight.severity] += 1

            summary["categories"] = dict(by_category)
            summary["severities"] = dict(by_severity)

        # Temporal pattern summary
        if temporal_insights:
            temporal_by_type = defaultdict(int)
            for insight in temporal_insights:
                temporal_by_type[insight.pattern_type.value] += 1

            summary["temporal_patterns_by_type"] = dict(temporal_by_type)
            summary["avg_temporal_confidence"] = sum(
                i.confidence for i in temporal_insights
            ) / len(temporal_insights)

        return summary

    def _generate_multi_file_report(
        self,
        log_sources: List[LogSource],
        correlations: List[CorrelatedInsight],
        temporal_insights: List[TemporalInsight],
    ) -> Dict[str, Any]:
        """Generate comprehensive multi-file analysis report."""

        # Aggregate statistics
        total_entries = sum(len(source.entries) for source in log_sources)
        total_insights = sum(len(source.insights) for source in log_sources)

        # Service-level analysis
        service_analysis = {}
        for source in log_sources:
            service_analysis[source.service_name] = {
                "file_path": source.file_path,
                "log_entries": len(source.entries),
                "insights_found": len(source.insights),
                "avg_confidence": (
                    sum(i.confidence for i in source.insights) / len(source.insights)
                    if source.insights
                    else 0
                ),
                "categories": self._categorize_insights(source.insights),
                "severities": self._severity_breakdown(source.insights),
            }

        # Correlation analysis
        correlation_analysis = self._analyze_correlations(correlations)

        # Temporal analysis
        temporal_analysis = self._analyze_temporal_patterns(temporal_insights)

        # Risk assessment
        risk_assessment = self._assess_risks(
            log_sources, correlations, temporal_insights
        )

        return {
            "summary": {
                "files_analyzed": len(log_sources),
                "services_involved": list(service_analysis.keys()),
                "total_log_entries": total_entries,
                "total_insights": total_insights,
                "correlations_found": len(correlations),
                "temporal_patterns": len(temporal_insights),
                "analysis_timestamp": datetime.now().isoformat(),
            },
            "service_analysis": service_analysis,
            "correlation_analysis": correlation_analysis,
            "temporal_analysis": temporal_analysis,
            "risk_assessment": risk_assessment,
            "detailed_correlations": [
                self._correlation_to_dict(c) for c in correlations[:10]
            ],
            "top_insights": self._get_top_insights_across_services(log_sources, 10),
        }

    # Helper methods for report generation
    def _categorize_insights(self, insights: List[Insight]) -> Dict[str, int]:
        """Categorize insights by type."""
        categories = defaultdict(int)
        for insight in insights:
            categories[insight.category] += 1
        return dict(categories)

    def _severity_breakdown(self, insights: List[Insight]) -> Dict[str, int]:
        """Break down insights by severity."""
        severities = defaultdict(int)
        for insight in insights:
            severities[insight.severity] += 1
        return dict(severities)

    def _analyze_correlations(
        self, correlations: List[CorrelatedInsight]
    ) -> Dict[str, Any]:
        """Analyze correlation patterns."""
        if not correlations:
            return {"total": 0, "types": {}, "high_confidence": 0}

        by_type = defaultdict(int)
        high_confidence = 0
        all_services = set()

        for corr in correlations:
            by_type[corr.correlation_type.value] += 1
            if corr.confidence > 0.7:
                high_confidence += 1
            all_services.update(corr.affected_services)

        return {
            "total": len(correlations),
            "types": dict(by_type),
            "high_confidence": high_confidence,
            "avg_confidence": sum(c.confidence for c in correlations)
            / len(correlations),
            "services_involved": list(all_services),
            "most_affected_services": self._find_most_affected_services(correlations),
        }

    def _analyze_temporal_patterns(
        self, temporal_insights: List[TemporalInsight]
    ) -> Dict[str, Any]:
        """Analyze temporal pattern distribution."""
        if not temporal_insights:
            return {"total": 0, "patterns": {}}

        by_pattern = defaultdict(int)
        for insight in temporal_insights:
            by_pattern[insight.pattern_type.value] += 1

        return {
            "total": len(temporal_insights),
            "patterns": dict(by_pattern),
            "avg_confidence": sum(i.confidence for i in temporal_insights)
            / len(temporal_insights),
            "escalations": by_pattern.get("escalation", 0),
            "bursts": by_pattern.get("burst", 0),
            "periodic": by_pattern.get("periodic", 0),
        }

    def _assess_risks(
        self,
        log_sources: List[LogSource],
        correlations: List[CorrelatedInsight],
        temporal_insights: List[TemporalInsight],
    ) -> Dict[str, Any]:
        """Assess overall system risks based on analysis."""

        risk_factors = {
            "cascade_failures": len(
                [c for c in correlations if c.correlation_type.value == "cascade"]
            ),
            "escalating_errors": len(
                [t for t in temporal_insights if t.pattern_type.value == "escalation"]
            ),
            "critical_insights": sum(
                len([i for i in source.insights if i.severity == "CRITICAL"])
                for source in log_sources
            ),
            "multi_service_impacts": len(
                [c for c in correlations if len(c.affected_services) > 2]
            ),
        }

        # Calculate overall risk score
        risk_score = (
            risk_factors["cascade_failures"] * 0.4
            + risk_factors["escalating_errors"] * 0.3
            + risk_factors["critical_insights"] * 0.2
            + risk_factors["multi_service_impacts"] * 0.1
        )

        if risk_score > 5:
            risk_level = "HIGH"
        elif risk_score > 2:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "overall_risk_level": risk_level,
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "recommendations": self._generate_recommendations(
                risk_factors, correlations
            ),
        }

    def _find_most_affected_services(
        self, correlations: List[CorrelatedInsight]
    ) -> List[str]:
        """Find services most frequently involved in correlations."""
        service_counts = defaultdict(int)
        for corr in correlations:
            for service in corr.affected_services:
                service_counts[service] += 1

        return sorted(service_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    def _get_top_insights_across_services(
        self, log_sources: List[LogSource], limit: int = 10
    ) -> List[Dict]:
        """Get top insights across all services."""
        all_insights = []
        for source in log_sources:
            for insight in source.insights:
                all_insights.append(
                    {
                        "service": source.service_name,
                        "insight": self._insight_to_dict(insight),
                    }
                )

        # Sort by confidence and return top N
        all_insights.sort(key=lambda x: x["insight"]["confidence"], reverse=True)
        return all_insights[:limit]

    def _generate_recommendations(
        self, risk_factors: Dict, correlations: List[CorrelatedInsight]
    ) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        if risk_factors["cascade_failures"] > 0:
            recommendations.append(
                "Implement circuit breakers to prevent cascade failures"
            )
            recommendations.append(
                "Review service dependencies and add proper fallbacks"
            )

        if risk_factors["escalating_errors"] > 0:
            recommendations.append("Set up automated alerting for error rate increases")
            recommendations.append(
                "Investigate root causes of escalating error patterns"
            )

        if risk_factors["critical_insights"] > 2:
            recommendations.append(
                "Immediate investigation required for critical issues"
            )
            recommendations.append("Consider emergency response procedures")

        # Service-specific recommendations based on correlations
        database_issues = any(
            "database" in str(c.affected_services) for c in correlations
        )
        if database_issues:
            recommendations.append(
                "Review database connection pooling and query performance"
            )
            recommendations.append("Consider database scaling or optimization")

        auth_issues = any(
            "auth" in str(c.affected_services).lower() for c in correlations
        )
        if auth_issues:
            recommendations.append(
                "Review authentication system stability and security"
            )
            recommendations.append("Implement additional monitoring for auth services")

        if not recommendations:
            recommendations.append(
                "System appears stable - continue regular monitoring"
            )

        return recommendations

    # Existing helper methods (unchanged)
    def parse_log_file(self, file_path: str) -> List[LogEntry]:
        """Parse a log file using appropriate parser."""
        try:
            if self.log_format:
                parser = self.parser_factory.get_parser(self.log_format)
            else:
                parser = self.parser_factory.detect_format(file_path)

            entries = parser.parse_file(file_path)
            entries = validate_log_entries(entries)

            stats = parser.get_parsing_stats()
            format_info = parser.get_format_info()

            self.logger.info(
                f"Parsed {len(entries)} valid entries using {format_info['format_type']} parser"
            )
            if stats["error_count"] > 0:
                self.logger.warning(
                    f"Failed to parse {stats['error_count']} lines (success rate: {stats['success_rate']:.1%})"
                )

            return entries

        except Exception as e:
            self.logger.error(f"Error parsing file {file_path}: {e}")
            raise

    def generate_report(self, insights: List[Insight]) -> Dict[str, Any]:
        """Generate a summary report from insights (backward compatibility)."""
        if not insights:
            return {"summary": "No insights found", "insights": []}

        by_category = defaultdict(list)
        by_severity = defaultdict(list)

        for insight in insights:
            by_category[insight.category].append(insight)
            by_severity[insight.severity].append(insight)

        total_insights = len(insights)
        avg_confidence = sum(i.confidence for i in insights) / total_insights

        report = {
            "summary": {
                "total_insights": total_insights,
                "average_confidence": round(avg_confidence, 3),
                "categories": {
                    cat: len(insights) for cat, insights in by_category.items()
                },
                "severities": {
                    sev: len(insights) for sev, insights in by_severity.items()
                },
            },
            "insights": [self._insight_to_dict(insight) for insight in insights],
        }

        return report

    def _add_entailment_scoring(
        self,
        entries: List[LogEntry],
        insights: List[Insight],
        target_statements: List[str],
    ) -> List[Insight]:
        """Add entailment scores to existing rule-based insights."""
        enhanced_insights = []

        for insight in insights:
            best_target, best_scores = self._find_best_target_match(
                insight, target_statements
            )

            if best_target:
                evidence_texts = [entry.message for entry in insight.evidence]
                if evidence_texts:
                    scores = self.entailment_scorer.compute_entailment_scores(
                        evidence_texts, best_target
                    )
                    avg_score = np.mean(scores) if scores else 0.0

                    insight.entailment_scores = scores
                    insight.avg_entailment_score = avg_score
                    insight.target_statement = best_target

                    original_confidence = insight.confidence
                    insight.confidence = (original_confidence + avg_score) / 2.0
                    insight.metadata["original_rule_confidence"] = original_confidence
                    insight.metadata["entailment_boost"] = avg_score

            enhanced_insights.append(insight)

        return enhanced_insights

    def _find_best_target_match(
        self, insight: Insight, target_statements: List[str]
    ) -> Tuple[str, List[float]]:
        """Find the target statement that best matches an insight."""
        if not target_statements:
            return "", []

        insight_text = insight.description.lower()
        best_target = ""
        best_keyword_matches = 0

        for target in target_statements:
            target_lower = target.lower()
            insight_words = set(insight_text.split())
            target_words = set(target_lower.split())
            matches = len(insight_words.intersection(target_words))

            if matches > best_keyword_matches:
                best_keyword_matches = matches
                best_target = target

        return best_target, []

    def _generate_entailment_insights(
        self, entries: List[LogEntry], target_statements: List[str]
    ) -> List[Insight]:
        """Generate insights purely based on entailment scoring."""
        try:
            entailment_insights = []
            log_messages = [entry.message for entry in entries if entry.message.strip()]

            for i, target_statement in enumerate(target_statements):
                max_entries = 1000
                limited_messages = log_messages[:max_entries]
                limited_entries = entries[:max_entries]

                scores = self.entailment_scorer.compute_entailment_scores(
                    limited_messages, target_statement
                )

                if not scores:
                    continue

                high_score_indices = [
                    i for i, score in enumerate(scores) if score > 0.7
                ]

                if high_score_indices:
                    high_score_entries = [
                        limited_entries[i] for i in high_score_indices
                    ]
                    high_scores = [scores[i] for i in high_score_indices]
                    avg_score = np.mean(high_scores)

                    insight = Insight(
                        rule_name="entailment_based",
                        confidence=avg_score,
                        description=f"High entailment with: '{target_statement}' ({len(high_score_entries)} matches)",
                        evidence=high_score_entries[:10],
                        timestamp=high_score_entries[0].timestamp or datetime.now(),
                        severity=self._score_to_severity(avg_score),
                        category="SEMANTIC",
                        entailment_scores=high_scores,
                        avg_entailment_score=avg_score,
                        target_statement=target_statement,
                        metadata={
                            "method": "pure_entailment",
                            "total_matches": len(high_score_entries),
                            "score_threshold": 0.7,
                            "max_entries_processed": max_entries,
                        },
                    )

                    entailment_insights.append(insight)

            return entailment_insights

        except Exception as e:
            print(f"Error in _generate_entailment_insights: {e}")
            import traceback

            traceback.print_exc()
            return []

    def _score_to_severity(self, score: float) -> str:
        """Convert entailment score to severity level."""
        if score >= 0.9:
            return "CRITICAL"
        elif score >= 0.8:
            return "HIGH"
        elif score >= 0.7:
            return "MEDIUM"
        else:
            return "LOW"

    def _insight_to_dict(self, insight: Insight) -> Dict[str, Any]:
        """Convert an Insight object to dictionary for JSON serialization."""
        result = {
            "rule_name": insight.rule_name,
            "confidence": insight.confidence,
            "description": insight.description,
            "timestamp": insight.timestamp.isoformat() if insight.timestamp else None,
            "severity": insight.severity,
            "category": insight.category,
            "evidence_count": len(insight.evidence),
            "evidence_sample": [entry.raw_line for entry in insight.evidence[:3]],
            "metadata": insight.metadata,
        }

        if insight.entailment_scores:
            result.update(
                {
                    "avg_entailment_score": insight.avg_entailment_score,
                    "target_statement": insight.target_statement,
                    "entailment_scores": insight.entailment_scores[:5],
                }
            )

        return result

    def _correlation_to_dict(self, correlation: CorrelatedInsight) -> Dict[str, Any]:
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


# Backward compatibility wrapper
class LogAnalyzer(EnhancedLogAnalyzer):
    """Backward compatible LogAnalyzer that inherits enhanced functionality."""

    def __init__(self, *args, **kwargs):
        # Remove new parameters if they exist to maintain compatibility
        kwargs.pop("temporal_window_minutes", None)
        kwargs.pop("correlation_window_minutes", None)
        super().__init__(*args, **kwargs)
