import argparse
import json
import logging

from spottr.analysis.analyzer import EnhancedLogAnalyzer
from spottr.analysis.temporal import TemporalInsight
from spottr.core.models import Insight, LogFormat
from spottr.utils.json_utils import NumpyEncoder


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

    # Initialize enhanced analyzer
    analyzer = EnhancedLogAnalyzer(
        use_entailment=use_entailment,
        log_format=log_format,
        temporal_window_minutes=args.temporal_window,
        correlation_window_minutes=args.correlation_window,
    )

    try:
        # Determine file list
        if args.directory:
            print(f"Analyzing directory: {args.directory}")
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
            result = analyzer.analyze_file_enhanced(
                args.log_files[0],
                target_statements=args.targets if use_entailment else None,
                include_temporal=include_temporal,
                time_window_minutes=args.window,
            )
            analysis_type = "single_file"

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
                result, analysis_type, include_temporal, include_correlation
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
    result, analysis_type: str, include_temporal: bool, include_correlation: bool
):
    """Print human-readable analysis report."""

    print("\n" + "=" * 80)
    print("SPOTTR ENHANCED LOG ANALYSIS REPORT")
    print("=" * 80)

    if analysis_type == "single_file":
        print_single_file_report(result, include_temporal)
    else:
        print_multi_file_report(result, include_temporal, include_correlation)


def print_single_file_report(result, include_temporal: bool):
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


def print_multi_file_report(result, include_temporal: bool, include_correlation: bool):
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


if __name__ == "__main__":
    exit(main())
