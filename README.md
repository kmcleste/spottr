# Spottr 🔭

**Intelligent Log Analysis with Hybrid Rule-Based and Semantic Detection**

Spottr is an advanced log analysis system that combines deterministic rule-based extraction with modern Natural Language Inference (NLI) to automatically detect and extract actionable insights from application logs. The hybrid approach provides both high-precision pattern matching and semantic understanding capabilities, enabling organizations to proactively identify operational issues, security threats, and performance bottlenecks.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![UV](https://img.shields.io/badge/managed%20by-uv-blue)](https://github.com/astral-sh/uv)

## ✨ Features

### 🧠 **Hybrid Analysis Engine**
- **Rule-Based Detection**: Regex-driven detection for known issue patterns
- **Semantic Analysis**: NorGLM-powered entailment scoring for contextual understanding
- **Confidence Scoring**: Probability-based insight ranking with hybrid confidence
- **Custom Targets**: User-defined statements for domain-specific insight detection

### 📋 **Multi-Format Log Support**
- **Plain Text Logs**: Traditional application logs with flexible parsing
- **JSON Logs**: Structured log parsing with intelligent field mapping
- **Auto-Detection**: Automatically identifies and handles different log formats

### ⏰ **Temporal Pattern Analysis**
- **Escalation Detection**: Identify error rates increasing over time
- **Burst Analysis**: Detect sudden activity spikes and anomalies
- **Periodic Patterns**: Discover recurring issues using autocorrelation
- **Trend Analysis**: Track performance degradation and system health trends
- **Configurable Windows**: Customizable time windows for pattern detection

### 🔗 **Multi-File Correlation**
- **Cross-Service Analysis**: Correlate insights across multiple log files
- **Dependency Modeling**: Track failure propagation through service chains
- **Root Cause Detection**: Identify common infrastructure issues affecting multiple services
- **Synchronized Events**: Detect coordinated activities across distributed systems
- **Risk Assessment**: Comprehensive system health scoring and recommendations

### 💡 **Intelligent Insights**
- **Timestamped Evidence**: Maintains chain of evidence for each insight
- **Severity Classification**: CRITICAL, HIGH, MEDIUM, LOW priority levels
- **Category Organization**: PERFORMANCE, ERROR, SECURITY, RESOURCE
- **Actionable Descriptions**: Human-readable insight summaries

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Log Sources   │───▶│  Parser Factory  │───▶│ Enhanced Analyzer│
│                 │    │                  │    │                 │
│ • JSON Logs     │    │ • Auto-detection │    │ • Rule Engine   │
│ • Line Logs     │    │ • Format-specific│    │ • Entailment    │
│ • Multi-files   │    │   parsers        │    │ • Temporal      │
└─────────────────┘    └──────────────────┘    │ • Correlation   │
                                               └─────────────────┘
                                                        │
                       ┌─────────────────────────────────┼─────────────────────────────────┐
                       ▼                                 ▼                                 ▼
            ┌─────────────────┐              ┌─────────────────┐              ┌─────────────────┐
            │ Rule-Based      │              │ Temporal        │              │ Correlation     │
            │ Insights        │              │ Patterns        │              │ Analysis        │
            │                 │              │                 │              │                 │
            │ • Error patterns│              │ • Escalations   │              │ • Cascade fails │
            │ • Performance   │              │ • Bursts        │              │ • Root causes   │
            │ • Security      │              │ • Periodic      │              │ • Synchronized  │
            │ • Resources     │              │ • Degradation   │              │ • Dependencies  │
            └─────────────────┘              └─────────────────┘              └─────────────────┘
                       │                                 │                                 │
                       └─────────────────────────────────┼─────────────────────────────────┘
                                                        ▼
                                              ┌─────────────────┐
                                              │ Comprehensive   │
                                              │ Report          │
                                              │                 │
                                              │ • Risk scoring  │
                                              │ • Recommendations│
                                              │ • Evidence      │
                                              │ • Visualizations│
                                              └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- [UV](https://github.com/astral-sh/uv) for dependency management

### Installation

```bash
# Clone the repository
git clone https://github.com/kmcleste/spottr.git
cd spottr

# Install dependencies with UV
uv sync

# Activate the virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

### Basic Usage

```bash
# Analyze a log file with default settings
uv run spottr /path/to/your/app.log

# Specify custom entailment targets
uv run spottr app.log -t "database performance issues" "memory leaks"

# Export results to JSON
uv run spottr app.log -o insights_report.json

# Rule-based analysis only
uv run spottr app.log --no-entailment --no-temporal

# Specify log format explicitly
uv run spottr app.log --format json

# Verbose output with debugging
uv run spottr app.log -v

# Custom target statements
uv spottr /path/to/your/app.log -t "database errors" "authentication failures"
```

### Multi-File Analysis with Correlation

```bash
# Analyze multiple service logs
uv run spottr logs/user-service.log logs/payment-service.log logs/database.log

# Specify service names for better correlation
uv run spottr logs/*.log --service-names user-service payment-service database api-gateway

# Directory analysis with auto-detection
uv run spottr --directory logs/ --pattern "*.log"

# Custom correlation and temporal windows
uv run spottr logs/*.log --correlation-window 15 --temporal-window 10
```

### Advanced Options
```bash
# Full-featured analysis with custom settings
uv run spottr --directory logs/ \
    --correlation-window 20 \
    --temporal-window 5 \
    -t "memory issues" "performance degradation" "security events" \
    --output analysis-report.json \
    --output-format both

# Focus on specific analysis types
uv run spottr logs/*.log --no-correlation  # Disable correlation analysis
uv run spottr logs/*.log --no-temporal     # Disable temporal patterns
uv run spottr logs/*.log --no-entailment   # Disable semantic analysis
```

## 📋 Supported Rule Categories

| Category | Detection Patterns | Severity |
|----------|-------------------|----------|
| **Memory Exhaustion** | OutOfMemoryError, heap full, allocation failures | CRITICAL |
| **High Error Rates** | Configurable error rate thresholds over time windows | HIGH |
| **Performance Issues** | Slow queries, high response times, timeouts | MEDIUM |
| **Authentication Failures** | Login failures, invalid credentials, access denied | HIGH |
| **Network Issues** | Connection timeouts, network unreachable | HIGH |
| **Resource Warnings** | Disk full, CPU high, thread pool exhaustion | MEDIUM |

## 🕒 Temporal Pattern Types

| Pattern | Description | Use Case |
|---------|-------------|----------|
| **Escalation** | Error rates increasing over time | Early warning system |
| **Burst** | Sudden activity spikes | Capacity planning |
| **Periodic** | Recurring patterns | Maintenance scheduling |
| **Degradation** | Performance declining trends | Proactive optimization |

## 🔗 Correlation Types

| Type | Description | Business Value |
|------|-------------|----------------|
| **Temporal** | Events occurring simultaneously | System coordination issues |
| **Cascade** | Failure propagation chains | Impact assessment |
| **Synchronized** | Coordinated events across services | Attack detection |
| **Root Cause** | Multiple symptoms from shared issues | Infrastructure problems |

## 📊 Output Formats

### Human-Readable Report
```
SPOTTR ENHANCED LOG ANALYSIS REPORT
================================================================================
Files Analyzed: 3
Services: user-service, payment-service, database
Total Log Entries: 15,247
Total Insights: 23
Correlations Found: 5
Temporal Patterns: 3

Risk Assessment:
Overall Risk Level: MEDIUM
Risk Score: 3.2

Service Analysis:
  user-service: 5,234 entries, 8 insights (avg conf: 0.842)
  payment-service: 7,891 entries, 12 insights (avg conf: 0.756)
  database: 2,122 entries, 3 insights (avg conf: 0.923)

Top Correlations:
1. [CASCADE] Payment service failures following database timeouts
   Confidence: 0.891 | Services: database, payment-service
   Analysis: Database connection pool exhaustion causing downstream failures

Recommendations:
1. Review database connection pooling and query performance
2. Implement circuit breakers to prevent cascade failures
3. Set up automated alerting for error rate increases
```

### JSON Report
```json
{
  "summary": {
    "files_analyzed": 3,
    "services_analyzed": ["user-service", "payment-service", "database"],
    "total_log_entries": 15247,
    "total_insights": 23,
    "correlations_found": 5,
    "temporal_patterns": 3,
    "analysis_timestamp": "2024-01-15T14:30:25.123Z"
  },
  "risk_assessment": {
    "overall_risk_level": "MEDIUM",
    "risk_score": 3.2,
    "risk_factors": {
      "cascade_failures": 2,
      "escalating_errors": 1,
      "critical_insights": 0,
      "multi_service_impacts": 2
    },
    "recommendations": [
      "Review database connection pooling and query performance",
      "Implement circuit breakers to prevent cascade failures"
    ]
  },
  "correlation_analysis": {
    "total": 5,
    "types": {
      "cascade": 2,
      "temporal": 2,
      "root_cause": 1
    },
    "high_confidence": 3,
    "avg_confidence": 0.782
  },
  "temporal_analysis": {
    "total": 3,
    "patterns": {
      "escalation": 1,
      "burst": 1,
      "periodic": 1
    },
    "avg_confidence": 0.756
  }
}
```

## 🧪 Sample Data Generation

Generate realistic test logs for development and testing:

```bash
# Generate various log types
uv run spottr-generate --all

# Generate specific scenarios
uv run spottr-generate --app-log --error-log --microservices

# Custom generation
uv run spottr-generate --output-dir test_logs --duration 4
```

This creates:
- `application.log` - Mixed operational scenarios
- `high_error_rate.log` - Error threshold testing
- `microservice_*.log` - Service-specific patterns
- `security_incidents.log` - Security testing scenarios

## 🔧 Configuration

### Service Dependencies
Configure service dependency relationships for better correlation:

```python
from spottr.analysis.analyzer import EnhancedLogAnalyzer

analyzer = EnhancedLogAnalyzer()

# Add custom dependencies
analyzer.correlation_engine.dependency_graph.add_dependency('api-gateway', 'user-service')
analyzer.correlation_engine.dependency_graph.add_dependency('payment-service', 'database')
analyzer.correlation_engine.dependency_graph.add_dependency('order-service', 'payment-service')
```

### Custom Rules
Add domain-specific detection rules:

```python
# Add custom rule to rule engine
analyzer.rule_engine.add_rule("custom_pattern", {
    "patterns": [r"CUSTOM_ERROR.*timeout", r"business logic failure"],
    "severity": "HIGH",
    "category": "BUSINESS",
    "description": "Custom business logic errors"
})
```

### Temporal Analysis Tuning
```python
# Customize temporal analysis
analyzer = EnhancedLogAnalyzer(
    temporal_window_minutes=10,     # 10-minute analysis windows
    correlation_window_minutes=20   # 20-minute correlation windows
)
```

## 🚀 Advanced Usage

### Programmatic API
```python
from spottr.analysis.analyzer import EnhancedLogAnalyzer

# Initialize with custom settings
analyzer = EnhancedLogAnalyzer(
    use_entailment=True,
    temporal_window_minutes=5,
    correlation_window_minutes=10
)

# Single file analysis
result = analyzer.analyze_file_enhanced(
    "logs/application.log",
    target_statements=["database errors", "memory issues"],
    include_temporal=True
)

# Multi-file correlation analysis
result = analyzer.analyze_multiple_files(
    ["logs/service1.log", "logs/service2.log"],
    service_names=["user-service", "payment-service"],
    include_correlation=True,
    include_temporal=True
)

# Directory batch analysis
result = analyzer.analyze_directory(
    "logs/",
    file_pattern="*.log",
    include_correlation=True
)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup
git clone https://github.com/kmcleste/spottr.git
cd spottr
uv sync --dev
```

### Adding New Log Formats

1. Create a new parser class inheriting from `BaseLogParser`
2. Implement required methods: `can_parse()`, `parse_line()`, `get_format_info()`
3. Add to `LogParserFactory` parser list
4. Add tests for the new format

### Adding Detection Rules

1. Define patterns and thresholds in `RuleEngine._load_default_rules()`
2. Specify severity and category classifications
3. Add corresponding test cases

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [NorGLM](https://huggingface.co/NorGLM/Entailment) for the entailment model
- [Hugging Face Transformers](https://github.com/huggingface/transformers) for model infrastructure
- [UV](https://github.com/astral-sh/uv) for excellent Python project management

## 🚨 Support

- **Issues**: [GitHub Issues](https://github.com/kmcleste/spottr/issues)
- **Discussions**: [GitHub Discussions](https://github.com/kmcleste/spottr/discussions)
- **Documentation**: [Wiki](https://github.com/kmcleste/spottr/wiki)

---

**Made with ❤️ for DevOps teams who are tired of manual log analysis**

*Spottr automatically spots issues so you don't have to.*