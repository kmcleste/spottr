# Spottr 🔭

**Intelligent Log Analysis with Hybrid Rule-Based and Semantic Detection**

Spottr automatically detects and extracts actionable insights from application logs using a powerful combination of deterministic rule-based extraction and modern Natural Language Inference (NLI). Stop manually sifting through logs – let Spottr spot the issues for you.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![UV](https://img.shields.io/badge/managed%20by-uv-blue)](https://github.com/astral-sh/uv)

## ✨ Features

### 🔍 **Hybrid Analysis Engine**
- **Rule-Based Detection**: High-precision pattern matching for known issues
- **Semantic Understanding**: NorGLM-powered entailment scoring for nuanced insights
- **Confidence Fusion**: Combines deterministic and probabilistic confidence scores

### 📋 **Multi-Format Log Support**
- **Plain Text Logs**: Traditional application logs with flexible parsing
- **JSON Logs**: Structured log parsing with intelligent field mapping
- **Auto-Detection**: Automatically identifies and handles different log formats

### 🎛️ **Built-in Detection Rules**
| Category | Examples | Severity |
|----------|----------|----------|
| **Memory Issues** | OutOfMemoryError, heap exhaustion | CRITICAL |
| **Performance** | Slow queries, high response times | MEDIUM-HIGH |
| **Authentication** | Login failures, access denied | HIGH |
| **Network** | Connection timeouts, unreachable hosts | HIGH |
| **Resource Warnings** | Disk full, CPU high, thread pool exhaustion | MEDIUM |

### 💡 **Intelligent Insights**
- **Timestamped Evidence**: Maintains chain of evidence for each insight
- **Severity Classification**: CRITICAL, HIGH, MEDIUM, LOW priority levels
- **Category Organization**: PERFORMANCE, ERROR, SECURITY, RESOURCE
- **Actionable Descriptions**: Human-readable insight summaries


## 📁 Directory Structure

```bash
spottr/
├── cli/
│   └── main.py                    # CLI interface and argument parsing
├── core/
│   └── models.py                  # Data models (LogEntry, Insight)
├── parsers/
│   ├── base.py                    # BaseLogParser, LogFormat enum
│   ├── line_delimited.py          # LineDelimitedParser
│   ├── json_parser.py             # JSONLogParser
│   └── factory.py                 # LogParserFactory
├── analysis/
│   ├── rule_engine.py             # RuleEngine
│   ├── entailment.py              # EntailmentScorer
│   └── analyzer.py                # LogAnalyzer
├── utils/
│   ├── validation.py              # validate_log_entries and helpers
│   └── json_utils.py              # NumpyEncoder and JSON utilities
└── sample_data/
    ├── generator.py               # LogGenerator
    └── test_runner.py             # Test runner script
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

# Disable entailment scoring (rules only)
uv run spottr app.log --no-entailment

# Specify log format explicitly
uv run spottr app.log --format json

# Verbose output with debugging
uv run spottr app.log -v
```

### Generate Sample Logs for Testing

```bash
# Generate all sample log types
uv run spottr-generate --all

# Generate specific log types
uv run spottr-generate --app-log --microservices

# Run complete test suite
uv run spottr-test
```

## 📊 Example Output

```json
{
  "summary": {
    "total_insights": 12,
    "average_confidence": 0.847,
    "categories": {
      "ERROR": 5,
      "PERFORMANCE": 4,
      "SECURITY": 2,
      "RESOURCE": 1
    },
    "severities": {
      "CRITICAL": 2,
      "HIGH": 4,
      "MEDIUM": 6
    }
  },
  "insights": [
    {
      "rule_name": "memory_exhaustion",
      "confidence": 0.95,
      "description": "Memory exhaustion detected (3 occurrences)",
      "severity": "CRITICAL",
      "category": "RESOURCE",
      "avg_entailment_score": 0.89,
      "target_statement": "memory and resource exhaustion",
      "evidence_sample": [
        "2024-01-15 14:30:25 [PaymentProcessor] FATAL: OutOfMemoryError: Java heap space"
      ]
    }
  ]
}
```

## 🏗️ Architecture

### Core Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Log Parser     │    │   Rule Engine    │    │ Entailment      │
│  Factory        │────▶   (Pattern       │────▶ Scorer         │
│                 │    │   Detection)     │    │ (NorGLM)        │
│ • Auto-detect   │    │                  │    │                 │
│ • JSON          │    │ • Memory issues  │    │ • Semantic      │
│ • Plain text    │    │ • Auth failures  │    │   matching      │
│ • Extensible    │    │ • Performance    │    │ • Confidence    │
└─────────────────┘    │ • Custom rules   │    │   scoring       │
                       └──────────────────┘    └─────────────────┘
                                ▼
                       ┌──────────────────┐
                       │ Insight          │
                       │ Generator        │
                       │                  │
                       │ • Evidence       │
                       │ • Confidence     │
                       │ • Categorization │
                       │ • JSON export    │
                       └──────────────────┘
```

### Parser Architecture

Spottr uses a flexible parser architecture that automatically detects log formats:

- **BaseLogParser**: Abstract interface for all parsers
- **LineDelimitedParser**: Traditional application logs
- **JSONLogParser**: Structured JSON logs with field mapping
- **LogParserFactory**: Automatic format detection and parser selection

## 🔧 Configuration

### Custom Target Statements

Define domain-specific insights using natural language:

```bash
uv run python spottr/analyzer.py app.log -t \
    "authentication and authorization failures" \
    "database connection pool exhaustion" \
    "payment processing errors" \
    "memory leak indicators"
```

### Custom Rules

Add custom detection rules by extending the `RuleEngine`:

```python
analyzer.rule_engine.add_rule('custom_pattern', {
    'patterns': [r'CUSTOM_ERROR', r'specific.*pattern'],
    'severity': 'HIGH',
    'category': 'CUSTOM',
    'description': 'Custom issue detected'
})
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