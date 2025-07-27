"""
LLM-powered analysis for enhanced log insights using OpenAI.
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pydantic import BaseModel

from spottr.core.models import Insight, LogEntry


class ExtractedInsight(BaseModel):
    """Structured output model for LLM-extracted insights."""

    category: str
    confidence: float  # 0.0 to 1.0
    description: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    tags: List[str]
    business_impact: Optional[str] = None
    recommended_actions: List[str] = []
    root_cause_hypothesis: Optional[str] = None


class LogQualityAssessment(BaseModel):
    """Assessment of log quality and completeness."""

    completeness_score: float  # 0.0 to 1.0
    missing_context: List[str]
    suggestions: List[str]
    timestamp_consistency: bool
    message_clarity_score: float


class RootCauseAnalysis(BaseModel):
    """Root cause analysis for correlated insights."""

    primary_cause: str
    contributing_factors: List[str]
    evidence_summary: str
    confidence: float
    timeline_analysis: str
    impact_scope: List[str]


@dataclass
class TagRepository:
    """Manages categorical tags for log insights."""

    def __init__(self):
        self.tags: Dict[str, Dict[str, Any]] = {}
        self.tag_usage_count: Dict[str, int] = {}

    def add_tag(self, tag: str, description: str, category: str = "general"):
        """Add or update a tag."""
        self.tags[tag] = {
            "description": description,
            "category": category,
            "created_at": datetime.now().isoformat(),
            "last_used": datetime.now().isoformat(),
        }
        self.tag_usage_count[tag] = self.tag_usage_count.get(tag, 0)

    def use_tag(self, tag: str) -> bool:
        """Mark a tag as used. Returns True if tag exists."""
        if tag in self.tags:
            self.tags[tag]["last_used"] = datetime.now().isoformat()
            self.tag_usage_count[tag] += 1
            return True
        return False

    def find_similar_tags(self, description: str, threshold: float = 0.7) -> List[str]:
        """Find existing tags similar to the description."""
        # Simple keyword-based similarity
        desc_words = set(description.lower().split())
        similar_tags = []

        for tag, tag_info in self.tags.items():
            tag_words = set(tag_info["description"].lower().split())
            if desc_words and tag_words:
                similarity = len(desc_words.intersection(tag_words)) / len(
                    desc_words.union(tag_words)
                )
                if similarity >= threshold:
                    similar_tags.append(tag)

        return similar_tags

    def get_popular_tags(self, limit: int = 10) -> List[tuple[str, int]]:
        """Get most frequently used tags."""
        return sorted(self.tag_usage_count.items(), key=lambda x: x[1], reverse=True)[
            :limit
        ]


class LLMLogAnalyzer:
    """LLM-powered log analysis using OpenAI."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided via parameter or OPENAI_API_KEY environment variable"
            )

        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        self.model = model
        self.logger = logging.getLogger(__name__)
        self.tag_repository = TagRepository()

        # Initialize with common tags
        self._initialize_common_tags()

    def _initialize_common_tags(self):
        """Initialize repository with common log analysis tags."""
        common_tags = [
            (
                "authentication_failure",
                "Failed login or authentication attempts",
                "security",
            ),
            (
                "performance_degradation",
                "Slow response times or performance issues",
                "performance",
            ),
            (
                "memory_exhaustion",
                "Out of memory errors or high memory usage",
                "resource",
            ),
            (
                "database_issues",
                "Database connection or query problems",
                "infrastructure",
            ),
            (
                "network_connectivity",
                "Network timeouts or connection failures",
                "infrastructure",
            ),
            ("payment_processing", "Payment or transaction related issues", "business"),
            ("user_activity", "Normal user operations and activities", "activity"),
            ("system_startup", "Application or service startup events", "lifecycle"),
            ("configuration_error", "Configuration related problems", "configuration"),
            (
                "external_service",
                "Third-party service integration issues",
                "integration",
            ),
        ]

        for tag, description, category in common_tags:
            self.tag_repository.add_tag(tag, description, category)

    def extract_insights_from_logs(
        self, log_entries: List[LogEntry], max_entries: int = 50
    ) -> List[ExtractedInsight]:
        """Extract structured insights from log entries using LLM."""

        if not log_entries:
            return []

        # Sample log entries to avoid token limits
        sample_entries = log_entries[:max_entries]
        log_text = "\n".join(
            entry.raw_line for entry in sample_entries if entry.raw_line
        )

        if not log_text.strip():
            return []

        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert log analyzer. Analyze the provided log entries and extract meaningful insights. 
                        Focus on:
                        1. Error patterns and anomalies
                        2. Performance issues
                        3. Security concerns
                        4. Business impact events
                        5. System health indicators
                        
                        For each insight, provide appropriate tags that categorize the finding.
                        Consider severity levels: LOW (informational), MEDIUM (needs attention), HIGH (urgent), CRITICAL (immediate action required).
                        """,
                    },
                    {
                        "role": "user",
                        "content": f"Analyze these log entries and extract insights:\n\n{log_text[:8000]}",  # Limit to avoid token issues
                    },
                ],
                response_format=ExtractedInsight,
            )

            insight = completion.choices[0].message.parsed

            # Process tags through repository
            processed_tags = self._process_tags(insight.tags, insight.description)
            insight.tags = processed_tags

            return [insight]

        except Exception as e:
            self.logger.error(f"Error extracting insights with LLM: {e}")
            return []

    def _process_tags(self, proposed_tags: List[str], description: str) -> List[str]:
        """Process proposed tags, finding existing ones or creating new ones."""
        final_tags = []

        for tag in proposed_tags:
            # Check if tag already exists
            if self.tag_repository.use_tag(tag):
                final_tags.append(tag)
            else:
                # Look for similar existing tags
                similar_tags = self.tag_repository.find_similar_tags(description)
                if similar_tags:
                    # Use the most popular similar tag
                    best_tag = max(
                        similar_tags,
                        key=lambda t: self.tag_repository.tag_usage_count.get(t, 0),
                    )
                    self.tag_repository.use_tag(best_tag)
                    final_tags.append(best_tag)
                else:
                    # Create new tag
                    self.tag_repository.add_tag(tag, description)
                    final_tags.append(tag)

        return final_tags

    def analyze_log_quality(self, log_entries: List[LogEntry]) -> LogQualityAssessment:
        """Assess the quality and completeness of log entries."""

        if not log_entries:
            return LogQualityAssessment(
                completeness_score=0.0,
                missing_context=[],
                suggestions=["No log entries to analyze"],
                timestamp_consistency=False,
                message_clarity_score=0.0,
            )

        # Sample for analysis
        sample_entries = log_entries[:20]
        log_sample = "\n".join(
            entry.raw_line for entry in sample_entries if entry.raw_line
        )

        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a log quality expert. Assess the quality of the provided log entries.
                        Consider:
                        1. Completeness of information (timestamps, severity levels, context)
                        2. Message clarity and informativeness
                        3. Consistency in formatting
                        4. Missing context that would be helpful for debugging
                        5. Overall usefulness for troubleshooting
                        
                        Provide scores from 0.0 to 1.0 and specific suggestions for improvement.
                        """,
                    },
                    {
                        "role": "user",
                        "content": f"Assess the quality of these log entries:\n\n{log_sample[:4000]}",
                    },
                ],
                response_format=LogQualityAssessment,
            )

            return completion.choices[0].message.parsed

        except Exception as e:
            self.logger.error(f"Error assessing log quality with LLM: {e}")
            return LogQualityAssessment(
                completeness_score=0.5,
                missing_context=["Unable to analyze due to LLM error"],
                suggestions=["Check LLM configuration"],
                timestamp_consistency=True,
                message_clarity_score=0.5,
            )

    def perform_root_cause_analysis(
        self, insights: List[Insight], correlation_context: Optional[str] = None
    ) -> RootCauseAnalysis:
        """Perform root cause analysis on correlated insights."""

        if not insights:
            return RootCauseAnalysis(
                primary_cause="No insights to analyze",
                contributing_factors=[],
                evidence_summary="",
                confidence=0.0,
                timeline_analysis="",
                impact_scope=[],
            )

        # Create summary of insights for analysis
        insights_summary = []
        for insight in insights[:10]:  # Limit to avoid token issues
            insights_summary.append(
                f"[{insight.severity}] {insight.description} (confidence: {insight.confidence:.2f})"
            )

        insights_text = "\n".join(insights_summary)

        prompt = f"""Analyze these correlated insights to determine root cause:

{insights_text}
"""

        if correlation_context:
            prompt += f"\nAdditional context: {correlation_context}"

        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert system analyst specializing in root cause analysis.
                        Analyze the provided insights to determine:
                        1. The most likely primary cause of the issues
                        2. Contributing factors that may have made it worse
                        3. Timeline analysis of how issues developed
                        4. Scope of impact on system components
                        
                        Provide high-confidence analysis based on the evidence patterns.
                        """,
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=RootCauseAnalysis,
            )

            return completion.choices[0].message.parsed

        except Exception as e:
            self.logger.error(f"Error performing root cause analysis with LLM: {e}")
            return RootCauseAnalysis(
                primary_cause="Analysis failed due to LLM error",
                contributing_factors=[],
                evidence_summary="Unable to analyze",
                confidence=0.0,
                timeline_analysis="",
                impact_scope=[],
            )

    def suggest_new_rules(self, log_entries: List[LogEntry]) -> List[Dict[str, Any]]:
        """Suggest new rule patterns based on log analysis."""

        if not log_entries:
            return []

        # Sample entries for pattern analysis
        sample_entries = log_entries[:30]
        log_text = "\n".join(
            entry.raw_line for entry in sample_entries if entry.raw_line
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert at creating log analysis rules. 
                        Analyze the provided logs and suggest regex patterns and rules that could 
                        automatically detect similar issues or patterns in the future.
                        
                        Return suggestions as JSON with this format:
                        {
                            "suggested_rules": [
                                {
                                    "rule_name": "descriptive_name",
                                    "pattern": "regex_pattern",
                                    "severity": "HIGH|MEDIUM|LOW",
                                    "category": "ERROR|PERFORMANCE|SECURITY|RESOURCE",
                                    "description": "what this rule detects"
                                }
                            ]
                        }
                        """,
                    },
                    {
                        "role": "user",
                        "content": f"Suggest rules for these log patterns:\n\n{log_text[:4000]}",
                    },
                ],
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)
            return result.get("suggested_rules", [])

        except Exception as e:
            self.logger.error(f"Error suggesting rules with LLM: {e}")
            return []

    def get_tag_summary(self) -> Dict[str, Any]:
        """Get summary of tag usage and repository state."""
        return {
            "total_tags": len(self.tag_repository.tags),
            "popular_tags": self.tag_repository.get_popular_tags(),
            "tag_categories": {
                category: len(
                    [
                        t
                        for t in self.tag_repository.tags.values()
                        if t.get("category") == category
                    ]
                )
                for category in set(
                    t.get("category", "general")
                    for t in self.tag_repository.tags.values()
                )
            },
        }
