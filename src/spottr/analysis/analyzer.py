import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spottr.analysis.entailment import EntailmentScorer
from spottr.analysis.rule_engine import RuleEngine
from spottr.core.models import Insight, LogEntry, LogFormat
from spottr.parsers.factory import LogParserFactory
from spottr.utils.validation import validate_log_entries


class LogAnalyzer:
    """Main log analysis orchestrator with entailment scoring."""

    def __init__(
        self,
        use_entailment=True,
        entailment_model_id="NorGLM/Entailment",
        log_format: Optional[LogFormat] = None,
    ):
        self.parser_factory = LogParserFactory()
        self.rule_engine = RuleEngine()
        self.logger = logging.getLogger(__name__)
        self.log_format = log_format  # If None, auto-detect

        self.use_entailment = use_entailment
        self.entailment_scorer = None

        if use_entailment:
            try:
                self.entailment_scorer = EntailmentScorer(entailment_model_id)
            except Exception as e:
                self.logger.warning(f"Failed to load entailment model: {e}")
                self.logger.warning("Continuing without entailment scoring")
                self.use_entailment = False

    def analyze_file(
        self, file_path: str, time_window_minutes: int = 5
    ) -> Tuple[List[LogEntry], List[Insight]]:
        """Analyze a log file and return parsed entries and insights."""
        entries = self.parse_log_file(file_path)
        insights = self.rule_engine.apply_rules(entries, time_window_minutes)
        return entries, insights

    def analyze_file_with_targets(
        self, file_path: str, target_statements: List[str], time_window_minutes: int = 5
    ) -> Tuple[List[LogEntry], List[Insight]]:
        """
        Analyze a log file with specific target statements for entailment scoring.
        """
        try:
            entries = self.parse_log_file(file_path)
            print(f"Parsed {len(entries)} log entries")

            # Get rule-based insights
            print("Applying rule-based analysis...")
            rule_insights = self.rule_engine.apply_rules(entries, time_window_minutes)
            print(f"Found {len(rule_insights)} rule-based insights")

            # Add entailment scoring if enabled
            if self.use_entailment and self.entailment_scorer and target_statements:
                print("Starting entailment scoring...")
                try:
                    enhanced_insights = self._add_entailment_scoring(
                        entries, rule_insights, target_statements
                    )
                    print(
                        f"Enhanced {len(enhanced_insights)} insights with entailment scores"
                    )

                    # Generate pure entailment-based insights
                    print("Generating pure entailment insights...")
                    entailment_insights = self._generate_entailment_insights(
                        entries, target_statements
                    )
                    print(
                        f"Generated {len(entailment_insights)} pure entailment insights"
                    )

                    # Combine rule-based and entailment-based insights
                    all_insights = enhanced_insights + entailment_insights
                except Exception as e:
                    print(f"Error during entailment scoring: {e}")
                    print("Falling back to rule-based insights only")
                    all_insights = rule_insights
            else:
                all_insights = rule_insights

            print(f"Total insights: {len(all_insights)}")
            return entries, all_insights

        except Exception as e:
            print(f"Error in analyze_file_with_targets: {e}")
            import traceback

            traceback.print_exc()
            raise

    def parse_log_file(self, file_path: str) -> List[LogEntry]:
        """Parse a log file using appropriate parser."""
        try:
            # Get the appropriate parser
            if self.log_format:
                parser = self.parser_factory.get_parser(self.log_format)
            else:
                parser = self.parser_factory.detect_format(file_path)

            # Parse the file
            entries = parser.parse_file(file_path)

            # Validate entries
            entries = validate_log_entries(entries)

            # Log parsing statistics
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
        """Generate a summary report from insights."""
        if not insights:
            return {"summary": "No insights found", "insights": []}

        # Group insights by category and severity
        by_category = defaultdict(list)
        by_severity = defaultdict(list)

        for insight in insights:
            by_category[insight.category].append(insight)
            by_severity[insight.severity].append(insight)

        # Calculate summary statistics
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
            # Find the best matching target statement for this insight
            best_target, best_scores = self._find_best_target_match(
                insight, target_statements
            )

            if best_target:
                # Calculate entailment scores for evidence
                evidence_texts = [entry.message for entry in insight.evidence]
                if evidence_texts:
                    scores = self.entailment_scorer.compute_entailment_scores(
                        evidence_texts, best_target
                    )
                    avg_score = np.mean(scores) if scores else 0.0

                    # Update insight with entailment information
                    insight.entailment_scores = scores
                    insight.avg_entailment_score = avg_score
                    insight.target_statement = best_target

                    # Enhance confidence with entailment score
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

        # Simple keyword matching for now - could be enhanced with embeddings
        insight_text = insight.description.lower()

        best_target = ""
        best_keyword_matches = 0

        for target in target_statements:
            target_lower = target.lower()
            # Count keyword overlaps
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

            # Score all log messages against each target statement
            log_messages = [entry.message for entry in entries if entry.message.strip()]
            print(
                f"Scoring {len(log_messages)} log messages against {len(target_statements)} targets"
            )

            for i, target_statement in enumerate(target_statements):
                print(
                    f"Processing target {i + 1}/{len(target_statements)}: '{target_statement}'"
                )

                # Limit the number of log entries to process for performance
                max_entries = 1000  # Process only first 1000 entries
                limited_messages = log_messages[:max_entries]
                limited_entries = entries[:max_entries]

                scores = self.entailment_scorer.compute_entailment_scores(
                    limited_messages, target_statement
                )

                if not scores:
                    print(f"No scores returned for target: {target_statement}")
                    continue

                # Find high-scoring entries (threshold > 0.7)
                high_score_indices = [
                    i for i, score in enumerate(scores) if score > 0.7
                ]
                print(
                    f"Found {len(high_score_indices)} high-scoring entries (>0.7) for target"
                )

                if high_score_indices:
                    high_score_entries = [
                        limited_entries[i] for i in high_score_indices
                    ]
                    high_scores = [scores[i] for i in high_score_indices]
                    avg_score = np.mean(high_scores)

                    # Create entailment-based insight
                    insight = Insight(
                        rule_name="entailment_based",
                        confidence=avg_score,
                        description=f"High entailment with: '{target_statement}' ({len(high_score_entries)} matches)",
                        evidence=high_score_entries[:10],  # Limit evidence
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

        # Add entailment information if available
        if insight.entailment_scores:
            result.update(
                {
                    "avg_entailment_score": insight.avg_entailment_score,
                    "target_statement": insight.target_statement,
                    "entailment_scores": insight.entailment_scores[
                        :5
                    ],  # Sample of scores
                }
            )

        return result
