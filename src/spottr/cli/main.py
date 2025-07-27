import argparse
import json
import logging
import os
from typing import Any, Dict

from dotenv import load_dotenv

from spottr.analysis.analyzer import EnhancedLogAnalyzer
from spottr.analysis.temporal import TemporalInsight
from spottr.core.models import Insight, LogFormat
from spottr.utils.json_utils import NumpyEncoder

load_dotenv()


def main():
    """Enhanced CLI interface for the log analyzer."""
    parser = argparse.ArgumentParser(
        description="Enhanced Spottr - Log Analysis with Temporal Patterns and Multi-File Correlation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file analysis
  spottr logs/application.log
  
  # Multiple files with correlation
  spottr logs/user-service.log logs/payment-service.log logs/database.log
  
  # Directory analysis
  spottr --directory logs/ --pattern "*.log"
  
  # Enhanced analysis with temporal patterns
  spottr logs/app.log --include-temporal --temporal-window 10
  
  # Custom target statements
  spottr logs/*.log -t "authentication failures" "payment errors" "database timeouts"
  
  # Service names for better correlation
  spottr logs/user.log logs/pay.log --service-names user-service payment-service
        """,
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("log_files", nargs="*", help="Log files to analyze")
    input_group.add_argument(
        "--directory", help="Directory containing log files to analyze"
    )

    # Input configuration
    parser.add_argument(
        "--pattern",
        default="*.log",
        help="File pattern for directory analysis (default: *.log)",
    )
    parser.add_argument(
        "--service-names",
        nargs="*",
        help="Service names corresponding to log files (for correlation)",
    )
    parser.add_argument(
        "--format",
        choices=["auto", "json", "line_delimited"],
        default="auto",
        help="Log format (default: auto-detect)",
    )

    # Analysis options
    parser.add_argument(
        "-t",
        "--targets",
        nargs="+",
        default=[
            "system performance issues",
            "authentication and security problems",
            "memory and resource exhaustion",
            "database performance problems",
            "network connectivity issues",
        ],
        help="Target statements for entailment scoring",
    )
    parser.add_argument(
        "-w",
        "--window",
        type=int,
        default=5,
        help="Time window in minutes for rule analysis (default: 5)",
    )
    parser.add_argument(
        "--temporal-window",
        type=int,
        default=5,
        help="Time window in minutes for temporal pattern analysis (default: 5)",
    )
    parser.add_argument(
        "--correlation-window",
        type=int,
        default=10,
        help="Time window in minutes for correlation analysis (default: 10)",
    )

    # Feature toggles
    parser.add_argument(
        "--no-entailment",
        action="store_true",
        help="Disable entailment scoring (rules only)",
    )
    parser.add_argument(
        "--include-temporal",
        action="store_true",
        default=True,
        help="Include temporal pattern analysis (default: enabled)",
    )
    parser.add_argument(
        "--no-temporal", action="store_true", help="Disable temporal pattern analysis"
    )
    parser.add_argument(
        "--include-correlation",
        action="store_true",
        default=True,
        help="Include multi-file correlation analysis (default: enabled for multiple files)",
    )
    parser.add_argument(
        "--no-correlation", action="store_true", help="Disable correlation analysis"
    )

    # Output options
    parser.add_argument("-o", "--output", help="Output file for JSON report")
    parser.add_argument(
        "--output-format",
        choices=["human", "json", "both"],
        default="human",
        help="Output format (default: human-readable)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--debug", action="store_true", help="Debug level logging")
    llm_group = parser.add_argument_group("LLM Analysis Options")
    llm_group.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable LLM-powered analysis (requires OpenAI API key)",
    )
    llm_group.add_argument(
        "--openai-api-key",
        help="OpenAI API key (can also use OPENAI_API_KEY environment variable)",
    )
    llm_group.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    llm_group.add_argument(
        "--llm-insights",
        action="store_true",
        default=True,
        help="Extract LLM insights (default: enabled when --use-llm is set)",
    )
    llm_group.add_argument(
        "--no-llm-insights", action="store_true", help="Disable LLM insight extraction"
    )
    llm_group.add_argument(
        "--quality-assessment",
        action="store_true",
        default=True,
        help="Include log quality assessment (default: enabled when --use-llm is set)",
    )
    llm_group.add_argument(
        "--root-cause-analysis",
        action="store_true",
        default=True,
        help="Include LLM root cause analysis for multi-file analysis (default: enabled when --use-llm is set)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.log_files and not args.directory:
        parser.error("Must specify either log files or --directory")

    # Set up logging
    if args.debug:
        level = logging.DEBUG
    elif args.verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")

    # Determine log format
    log_format = None
    if args.format != "auto":
        format_mapping = {
            "json": LogFormat.JSON,
            "line_delimited": LogFormat.LINE_DELIMITED,
        }
        log_format = format_mapping.get(args.format)

    # Process feature flags
    use_entailment = not args.no_entailment
    include_temporal = args.include_temporal and not args.no_temporal
    include_correlation = args.include_correlation and not args.no_correlation

    # LLM feature flags
    include_llm_insights = (
        args.llm_insights and not args.no_llm_insights and args.use_llm
    )
    include_quality_assessment = args.quality_assessment and args.use_llm
    include_root_cause_analysis = args.root_cause_analysis and args.use_llm

    # Validate LLM configuration
    if args.use_llm:
        api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            parser.error(
                "OpenAI API key required when using --use-llm. Set OPENAI_API_KEY environment variable or use --openai-api-key"
            )

    # Initialize enhanced analyzer
    analyzer = EnhancedLogAnalyzer(
        use_entailment=use_entailment,
        log_format=log_format,
        temporal_window_minutes=args.temporal_window,
        correlation_window_minutes=args.correlation_window,
        # LLM configuration
        use_llm=args.use_llm,
        openai_api_key=args.openai_api_key or os.getenv("OPENAI_API_KEY"),
        llm_model=args.llm_model,
    )

    try:
        # Determine file list
        if args.directory:
            print(f"Analyzing directory: {args.directory}")
            # Note: For directory analysis, we'd need to add LLM support to analyze_directory method
            result = analyzer.analyze_directory(
                args.directory,
                file_pattern=args.pattern,
                target_statements=args.targets if use_entailment else None,
                include_temporal=include_temporal,
                include_correlation=include_correlation,
                time_window_minutes=args.window,
            )
            analysis_type = "directory"

        elif len(args.log_files) > 1:
            print(f"Analyzing {len(args.log_files)} log files with correlation")
            if args.use_llm:
                result = analyzer.analyze_multiple_files_with_llm(
                    args.log_files,
                    target_statements=args.targets if use_entailment else None,
                    service_names=args.service_names,
                    include_temporal=include_temporal,
                    include_correlation=include_correlation,
                    include_llm_insights=include_llm_insights,
                    include_root_cause_analysis=include_root_cause_analysis,
                    time_window_minutes=args.window,
                )
            else:
                result = analyzer.analyze_multiple_files(
                    args.log_files,
                    target_statements=args.targets if use_entailment else None,
                    service_names=args.service_names,
                    include_temporal=include_temporal,
                    include_correlation=include_correlation,
                    time_window_minutes=args.window,
                )
            analysis_type = "multi_file"

        else:
            # Single file analysis
            print(f"Analyzing single file: {args.log_files[0]}")
            if args.use_llm:
                result = analyzer.analyze_file_with_llm(
                    args.log_files[0],
                    target_statements=args.targets if use_entailment else None,
                    include_temporal=include_temporal,
                    include_llm_insights=include_llm_insights,
                    include_quality_assessment=include_quality_assessment,
                    time_window_minutes=args.window,
                )
            else:
                result = analyzer.analyze_file_enhanced(
                    args.log_files[0],
                    target_statements=args.targets if use_entailment else None,
                    include_temporal=include_temporal,
                    time_window_minutes=args.window,
                )
            analysis_type = "single_file"

        print(
            f"DEBUG: Analysis completed, result keys: {list(result.keys()) if result else 'None'}"
        )

        # Generate output
        if args.output_format in ["json", "both"]:
            json_output = format_json_output(result, analysis_type)

            if args.output:
                with open(args.output, "w") as f:
                    json.dump(json_output, f, indent=2, cls=NumpyEncoder)
                print(f"JSON report saved to: {args.output}")

            if args.output_format == "json":
                print(json.dumps(json_output, indent=2, cls=NumpyEncoder))

        if args.output_format in ["human", "both"]:
            print_human_readable_report(
                result,
                analysis_type,
                include_temporal,
                include_correlation,
                args.use_llm,
            )

    except Exception as e:
        import traceback

        print(f"Error during analysis: {e}")
        if args.debug:
            print("Full traceback:")
            traceback.print_exc()
        return 1

    return 0


def format_json_output(result, analysis_type: str):
    """Format result for JSON output."""
    if analysis_type == "single_file":
        # Convert single file result to JSON-serializable format
        json_result = {
            "analysis_type": "single_file",
            "summary": result["summary"],
            "insights": [
                insight.__dict__ if hasattr(insight, "__dict__") else insight
                for insight in result["standard_insights"]
            ],
            "temporal_insights": [
                insight.__dict__ if hasattr(insight, "__dict__") else insight
                for insight in result["temporal_insights"]
            ],
        }
    else:
        # Multi-file or directory analysis
        json_result = result.copy()
        json_result["analysis_type"] = analysis_type

    return json_result


def print_human_readable_report(
    result,
    analysis_type: str,
    include_temporal: bool,
    include_correlation: bool,
    include_llm: bool = False,
):
    """Print human-readable analysis report."""

    print("\n" + "=" * 80)
    print("SPOTTR ENHANCED LOG ANALYSIS REPORT")
    if include_llm:
        print("(with LLM-Powered Insights)")
    print("=" * 80)

    if analysis_type == "single_file":
        print_single_file_report(result, include_temporal, include_llm)
    else:
        print_multi_file_report(
            result, include_temporal, include_correlation, include_llm
        )


def print_single_file_report(result, include_temporal: bool, include_llm: bool = False):
    """Print single file analysis report."""

    summary = result["summary"]
    standard_insights: list[Insight] = result["standard_insights"]
    temporal_insights: list[TemporalInsight] = result["temporal_insights"]

    print(f"Log Entries Processed: {summary['total_entries']}")
    print(f"Standard Insights Found: {summary['total_insights']}")
    if include_temporal:
        print(f"Temporal Patterns Found: {summary['temporal_patterns']}")

    # Standard insights summary
    if standard_insights:
        avg_conf = sum(i.confidence for i in standard_insights) / len(standard_insights)
        print(f"Average Confidence: {avg_conf:.3f}")

        # Category breakdown
        categories = {}
        severities = {}
        for insight in standard_insights:
            categories[insight.category] = categories.get(insight.category, 0) + 1
            severities[insight.severity] = severities.get(insight.severity, 0) + 1

        print(f"\nInsight Categories: {dict(categories)}")
        print(f"Severity Distribution: {dict(severities)}")

    # Top insights
    print("\nTop Standard Insights:")
    top_insights = sorted(standard_insights, key=lambda x: x.confidence, reverse=True)[
        :5
    ]
    for i, insight in enumerate(top_insights, 1):
        print(f"{i}. [{insight.severity}] {insight.description}")
        print(f"   Confidence: {insight.confidence:.3f} | Category: {insight.category}")
        if insight.evidence:
            print(f"   Example: {insight.evidence[0].raw_line[:100]}...")
        print()

    # Temporal patterns
    if include_temporal and temporal_insights:
        print("\nTemporal Patterns Detected:")
        for i, pattern in enumerate(temporal_insights[:5], 1):
            print(f"{i}. [{pattern.pattern_type.value.upper()}] {pattern.description}")
            print(
                f"   Confidence: {pattern.confidence:.3f} | Trend: {pattern.trend_direction}"
            )
            if hasattr(pattern, "velocity"):
                print(f"   Velocity: {pattern.velocity:.3f}")
            print()

    # LLM Analysis
    if include_llm and "llm_analysis" in result:
        print_llm_analysis_section(result["llm_analysis"])


def print_multi_file_report(
    result, include_temporal: bool, include_correlation: bool, include_llm: bool = False
):
    """Print multi-file analysis report."""

    summary = result["summary"]
    service_analysis = result["service_analysis"]
    correlation_analysis = result["correlation_analysis"]
    risk_assessment = result["risk_assessment"]

    print(f"Files Analyzed: {summary['files_analyzed']}")
    print(f"Services: {', '.join(summary['services_involved'])}")
    print(f"Total Log Entries: {summary['total_log_entries']}")
    print(f"Total Insights: {summary['total_insights']}")

    if include_correlation:
        print(f"Correlations Found: {summary['correlations_found']}")

    if include_temporal:
        print(f"Temporal Patterns: {summary['temporal_patterns']}")

    # Risk assessment
    print("\nRisk Assessment:")
    print(f"Overall Risk Level: {risk_assessment['overall_risk_level']}")
    print(f"Risk Score: {risk_assessment['risk_score']:.1f}")

    risk_factors: dict = risk_assessment["risk_factors"]
    if any(risk_factors.values()):
        print("Risk Factors:")
        for factor, count in risk_factors.items():
            if count > 0:
                print(f"  - {factor.replace('_', ' ').title()}: {count}")

    # Service breakdown
    print("\nService Analysis:")
    for service, stats in service_analysis.items():
        print(f"  {service}:")
        print(
            f"    Entries: {stats['log_entries']}, Insights: {stats['insights_found']}"
        )
        print(f"    Avg Confidence: {stats['avg_confidence']:.3f}")
        if stats["categories"]:
            print(f"    Categories: {stats['categories']}")

    # Correlations
    if include_correlation and correlation_analysis["total"] > 0:
        print("\nCorrelation Analysis:")
        print(f"Total Correlations: {correlation_analysis['total']}")
        print(f"High Confidence (>0.7): {correlation_analysis['high_confidence']}")
        print(f"Average Confidence: {correlation_analysis['avg_confidence']:.3f}")
        print(f"Correlation Types: {correlation_analysis['types']}")

        if "most_affected_services" in correlation_analysis:
            print("Most Affected Services:")
            for service, count in correlation_analysis["most_affected_services"][:3]:
                print(f"  - {service}: {count} correlations")

    # Top correlations
    if include_correlation and "detailed_correlations" in result:
        print("\nTop Correlations:")
        for i, corr in enumerate(result["detailed_correlations"][:3], 1):
            print(f"{i}. [{corr['type'].upper()}] {corr['description']}")
            print(f"   Confidence: {corr['confidence']:.3f}")
            print(f"   Services: {', '.join(corr['affected_services'])}")
            if corr["root_cause_analysis"]:
                print(f"   Analysis: {corr['root_cause_analysis']}")
            print()

    # Recommendations
    if risk_assessment["recommendations"]:
        print("Recommendations:")
        for i, rec in enumerate(risk_assessment["recommendations"], 1):
            print(f"  {i}. {rec}")

    # LLM Analysis
    if include_llm and "llm_analysis" in result:
        print_llm_analysis_section(result["llm_analysis"])


def print_llm_analysis_section(llm_analysis: Dict[str, Any]):
    """Print LLM analysis results."""
    if not llm_analysis or "error" in llm_analysis:
        if "error" in llm_analysis:
            print(f"LLM Analysis Error: {llm_analysis['error']}")
        return

    print("\nLLM-Powered Analysis:")
    print("-" * 30)

    # LLM Insights
    if "llm_insights" in llm_analysis and llm_analysis["llm_insights"]:
        print(f"LLM Insights Found: {len(llm_analysis['llm_insights'])}")
        for i, insight in enumerate(llm_analysis["llm_insights"][:3], 1):
            print(f"{i}. [{insight['severity']}] {insight['description']}")
            print(f"   Confidence: {insight['confidence']:.3f}")
            if insight["tags"]:
                print(f"   Tags: {', '.join(insight['tags'])}")
            if insight["business_impact"]:
                print(f"   Business Impact: {insight['business_impact']}")
            if insight["recommended_actions"]:
                print(
                    f"   Recommendations: {', '.join(insight['recommended_actions'][:2])}"
                )
            print()

    # Quality Assessment
    if "quality_assessment" in llm_analysis and llm_analysis["quality_assessment"]:
        qa = llm_analysis["quality_assessment"]
        print("\nLog Quality Assessment:")
        print(f"  Completeness Score: {qa['completeness_score']:.2f}/1.0")
        print(f"  Message Clarity Score: {qa['message_clarity_score']:.2f}/1.0")
        print(f"  Timestamp Consistency: {'✓' if qa['timestamp_consistency'] else '✗'}")

        if qa["missing_context"]:
            print("  Missing Context:")
            for context in qa["missing_context"][:3]:
                print(f"    - {context}")

        if qa["suggestions"]:
            print("  Improvement Suggestions:")
            for suggestion in qa["suggestions"][:3]:
                print(f"    - {suggestion}")

    # Root Cause Analysis
    if "root_cause_analysis" in llm_analysis and llm_analysis["root_cause_analysis"]:
        rca = llm_analysis["root_cause_analysis"]
        print(f"\nRoot Cause Analysis (Confidence: {rca['confidence']:.2f}):")
        print(f"  Primary Cause: {rca['primary_cause']}")

        if rca["contributing_factors"]:
            print("  Contributing Factors:")
            for factor in rca["contributing_factors"][:3]:
                print(f"    - {factor}")

        if rca["impact_scope"]:
            print(f"  Impact Scope: {', '.join(rca['impact_scope'])}")

        if rca["timeline_analysis"]:
            print(f"  Timeline: {rca['timeline_analysis']}")

    # Suggested Rules
    if "suggested_rules" in llm_analysis and llm_analysis["suggested_rules"]:
        print(f"\nSuggested Rules ({len(llm_analysis['suggested_rules'])}):")
        for i, rule in enumerate(llm_analysis["suggested_rules"][:3], 1):
            print(f"{i}. {rule['rule_name']} [{rule['severity']}]")
            print(f"   Pattern: {rule['pattern']}")
            print(f"   Description: {rule['description']}")

    # Tag Summary
    if "tag_summary" in llm_analysis and llm_analysis["tag_summary"]:
        tag_summary = llm_analysis["tag_summary"]
        print(f"\nTag Repository: {tag_summary['total_tags']} tags")
        if tag_summary.get("popular_tags"):
            print("  Most Used Tags:")
            for tag, count in tag_summary["popular_tags"][:5]:
                print(f"    - {tag} ({count} uses)")


if __name__ == "__main__":
    exit(main())
