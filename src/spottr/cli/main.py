import argparse
import json
import logging

from spottr.analysis.analyzer import LogAnalyzer
from spottr.core.models import LogFormat
from spottr.utils.json_utils import NumpyEncoder


def main():
    """CLI interface for the log analyzer."""
    parser = argparse.ArgumentParser(
        description="Log Analysis POC - Rule-Based + Entailment Extraction"
    )
    parser.add_argument("log_file", help="Path to the log file to analyze")
    parser.add_argument(
        "-w",
        "--window",
        type=int,
        default=5,
        help="Time window in minutes for analysis (default: 5)",
    )
    parser.add_argument("-o", "--output", help="Output file for JSON report")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--no-entailment",
        action="store_true",
        help="Disable entailment scoring (rules only)",
    )
    parser.add_argument(
        "-t",
        "--targets",
        nargs="+",
        default=[
            "logs related to system performance issues",
            "authentication and security problems",
            "memory and resource exhaustion",
            "database performance problems",
        ],
        help="Target statements for entailment scoring",
    )
    parser.add_argument(
        "--format",
        choices=["auto", "json", "line_delimited"],
        default="auto",
        help="Log format (default: auto-detect)",
    )

    args = parser.parse_args()

    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")

    # Determine log format
    log_format = None
    if args.format != "auto":
        format_mapping = {
            "json": LogFormat.JSON,
            "line_delimited": LogFormat.LINE_DELIMITED,
        }
        log_format = format_mapping.get(args.format)

    # Initialize analyzer
    use_entailment = not args.no_entailment
    analyzer = LogAnalyzer(use_entailment=use_entailment, log_format=log_format)

    try:
        # Analyze the log file
        print(f"Analyzing log file: {args.log_file}")
        if use_entailment:
            print(f"Target statements: {args.targets}")
            entries, insights = analyzer.analyze_file_with_targets(
                args.log_file, args.targets, args.window
            )
        else:
            entries, insights = analyzer.analyze_file(args.log_file, args.window)

        # Generate report
        report = analyzer.generate_report(insights)

        # Output results
        if args.output:
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2, cls=NumpyEncoder)
            print(f"Report saved to: {args.output}")
        else:
            print("\n" + "=" * 60)
            print("LOG ANALYSIS REPORT")
            print("=" * 60)

            summary = report["summary"]
            print(f"Total Insights: {summary['total_insights']}")
            print(f"Average Confidence: {summary['average_confidence']}")

            print("\nBy Category:")
            for category, count in summary["categories"].items():
                print(f"  {category}: {count}")

            print("\nBy Severity:")
            for severity, count in summary["severities"].items():
                print(f"  {severity}: {count}")

            print("\nTop Insights:")
            for insight in sorted(
                report["insights"], key=lambda x: x["confidence"], reverse=True
            )[:5]:
                print(f"\n- {insight['description']}")
                print(
                    f"  Severity: {insight['severity']}, Confidence: {insight['confidence']:.3f}"
                )
                print(f"  Rule: {insight['rule_name']}")
                if insight["evidence_sample"]:
                    print(f"  Example: {insight['evidence_sample'][0][:100]}...")

    except Exception as e:
        import traceback

        print(f"Error during analysis: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
