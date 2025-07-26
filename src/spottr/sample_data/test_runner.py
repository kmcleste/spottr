"""
Test Runner for Log Analysis POC
Generates sample logs and runs analysis to demonstrate capabilities.
"""

import json
import subprocess
from pathlib import Path


def run_command(cmd):
    """Run a command and return success status."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {cmd}")
            return True, result.stdout
        else:
            print(f"✗ {cmd}")
            print(f"Error: {result.stderr}")
            return False, result.stderr
    except Exception as e:
        print(f"✗ {cmd}")
        print(f"Exception: {e}")
        return False, str(e)


def main():
    """Run the complete test suite."""
    print("=" * 60)
    print("LOG ANALYSIS POC - TEST RUNNER")
    print("=" * 60)

    # Create sample_logs directory
    logs_dir = Path("sample_logs")
    logs_dir.mkdir(exist_ok=True)

    print("\n1. Generating Sample Logs...")
    print("-" * 30)

    # Generate different types of logs
    success, _ = run_command("uv run spottr-generate --all")
    if not success:
        print("Failed to generate sample logs. Exiting.")
        return 1

    print("\n2. Running Log Analysis...")
    print("-" * 30)

    # Test cases with different log files
    test_cases = [
        {
            "name": "Application Log Analysis",
            "file": "sample_logs/application.log",
            "description": "General application logs with mixed scenarios",
        },
        {
            "name": "High Error Rate Analysis",
            "file": "sample_logs/high_error_rate.log",
            "description": "Logs demonstrating error rate threshold detection",
        },
        {
            "name": "User Service Analysis",
            "file": "sample_logs/microservice_user-service.log",
            "description": "Authentication and user management patterns",
        },
        {
            "name": "Payment Service Analysis",
            "file": "sample_logs/microservice_payment-service.log",
            "description": "Payment processing with memory issues",
        },
    ]

    results = {}

    for test_case in test_cases:
        print(f"\n{test_case['name']}:")
        print(f"Description: {test_case['description']}")

        # Check if file exists
        if not Path(test_case["file"]).exists():
            print(f"✗ Log file not found: {test_case['file']}")
            continue

        # Run analysis with JSON output
        output_file = f"{test_case['file']}.analysis.json"
        cmd = f"uv run spottr '{test_case['file']}' -o '{output_file}'"

        success, output = run_command(cmd)
        if success:
            # Load and display results
            try:
                with open(output_file, "r") as f:
                    analysis_result = json.load(f)
                results[test_case["name"]] = analysis_result

                summary = analysis_result["summary"]
                print(f"  Insights Found: {summary['total_insights']}")
                print(f"  Avg Confidence: {summary['average_confidence']:.3f}")
                print(
                    f"  Categories: {', '.join(f'{k}({v})' for k, v in summary['categories'].items())}"
                )
                print(
                    f"  Severities: {', '.join(f'{k}({v})' for k, v in summary['severities'].items())}"
                )

                # Show top insight
                if analysis_result["insights"]:
                    top_insight = max(
                        analysis_result["insights"], key=lambda x: x["confidence"]
                    )
                    print(
                        f"  Top Insight: {top_insight['description']} (confidence: {top_insight['confidence']:.3f})"
                    )

            except Exception as e:
                print(f"  ✗ Failed to parse analysis results: {e}")
        else:
            print(f"  ✗ Analysis failed for {test_case['file']}")

    print("\n3. Summary Report")
    print("-" * 30)

    if results:
        total_insights = sum(r["summary"]["total_insights"] for r in results.values())
        avg_confidence = sum(
            r["summary"]["average_confidence"] for r in results.values()
        ) / len(results)

        # Aggregate categories and severities
        all_categories = {}
        all_severities = {}

        for result in results.values():
            for cat, count in result["summary"]["categories"].items():
                all_categories[cat] = all_categories.get(cat, 0) + count
            for sev, count in result["summary"]["severities"].items():
                all_severities[sev] = all_severities.get(sev, 0) + count

        print(f"Total Insights Across All Logs: {total_insights}")
        print(f"Average Confidence: {avg_confidence:.3f}")
        print(f"Category Distribution: {dict(all_categories)}")
        print(f"Severity Distribution: {dict(all_severities)}")

        # Show some interesting insights
        print("\nNotable Insights Found:")
        insight_count = 0
        for test_name, result in results.items():
            for insight in sorted(
                result["insights"], key=lambda x: x["confidence"], reverse=True
            )[:2]:
                insight_count += 1
                print(f"{insight_count}. [{test_name}] {insight['description']}")
                print(
                    f"   Confidence: {insight['confidence']:.3f}, Severity: {insight['severity']}"
                )
                if insight["evidence_sample"]:
                    print(f"   Example: {insight['evidence_sample'][0][:80]}...")
                if insight_count >= 5:
                    break
            if insight_count >= 5:
                break
    else:
        print("No analysis results to summarize.")

    print("\n4. Test Files Generated:")
    print("-" * 30)
    for log_file in logs_dir.glob("*.log"):
        size_kb = log_file.stat().st_size // 1024
        print(f"  {log_file.name} ({size_kb} KB)")

    print("\nAnalysis reports saved as *.analysis.json files")
    print("\nTest completed successfully! 🎉")

    print("\nNext Steps:")
    print("- Examine the generated log files in sample_logs/")
    print("- Review the analysis JSON files for detailed insights")
    print("- Try running: python analyzer.py sample_logs/application.log -v")
    print("- Modify rules in the LogAnalyzer to test custom patterns")

    return 0


if __name__ == "__main__":
    exit(main())
