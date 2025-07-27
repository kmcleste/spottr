# LLM-Powered Analysis

Spottr now includes optional LLM-powered analysis using OpenAI's API for enhanced log insights, quality assessment, and root cause analysis.

## Features

### 🤖 Structured Insight Extraction
- Automatically categorizes and tags log patterns
- Provides business impact assessment
- Suggests actionable remediation steps
- Generates root cause hypotheses

### 📊 Log Quality Assessment
- Evaluates completeness and clarity of log entries
- Identifies missing context and information gaps
- Provides specific improvement suggestions
- Assesses timestamp consistency and message quality

### 🔍 Root Cause Analysis
- Analyzes correlated insights across multiple services
- Identifies primary causes and contributing factors
- Provides timeline analysis of issue development
- Assesses impact scope across system components

### 🏷️ Smart Tagging System
- Maintains a repository of categorical tags
- Automatically matches similar patterns to existing tags
- Tracks tag usage and popularity
- Prevents tag proliferation through similarity matching

### 🛠️ Rule Suggestion Engine
- Analyzes log patterns to suggest new detection rules
- Provides regex patterns for automated monitoring
- Categorizes suggested rules by severity and type

## Setup

### API Key Configuration

Set your OpenAI API key using one of the following methods:

```bash
# Environment variable (recommended)
export OPENAI_API_KEY="your-api-key-here"

# Or via command line
spottr logs/app.log --use-llm --openai-api-key "your-api-key"
```

Or, create an `.env` file containing:

```txt
OPENAI_API_KEY=your-api-key-here
```

### Model Selection

Choose your preferred OpenAI model:

```bash
# Default: gpt-4o-mini (cost-effective)
spottr logs/app.log --use-llm

# Use GPT-4 for enhanced analysis
spottr logs/app.log --use-llm --llm-model "gpt-4"
```

## Usage Examples

### Basic LLM Analysis

```bash
# Single file with LLM insights
spottr logs/application.log --use-llm

# Include quality assessment
spottr logs/application.log --use-llm --quality-assessment
```

### Multi-File Analysis with Root Cause

```bash
# Analyze multiple services with correlation and root cause analysis
spottr logs/user-service.log logs/payment-service.log logs/database.log \
  --use-llm --root-cause-analysis \
  --service-names user payment database
```

### Comprehensive Analysis

```bash
# All features enabled
spottr logs/*.log --use-llm \
  --include-temporal \
  --include-correlation \
  --quality-assessment \
  --root-cause-analysis \
  --output comprehensive_report.json
```

### Feature Control

```bash
# Disable specific LLM features
spottr logs/app.log --use-llm --no-llm-insights  # Quality assessment only

# Enable only rule suggestions
spottr logs/app.log --use-llm --no-llm-insights --no-quality-assessment
```

## LLM Output Examples

### Extracted Insights

```json
{
  "category": "performance",
  "confidence": 0.87,
  "description": "Database query performance degradation detected",
  "severity": "HIGH",
  "tags": ["database_performance", "query_optimization", "latency"],
  "business_impact": "User experience degraded, potential revenue impact",
  "recommended_actions": [
    "Review slow query log",
    "Consider database indexing",
    "Scale database resources"
  ],
  "root_cause_hypothesis": "Database connection pool exhaustion under high load"
}
```

### Quality Assessment

```json
{
  "completeness_score": 0.75,
  "missing_context": [
    "User identifiers for error tracking",
    "Request correlation IDs",
    "Service version information"
  ],
  "suggestions": [
    "Add structured logging with consistent fields",
    "Include request tracing information",
    "Add more detailed error context"
  ],
  "timestamp_consistency": true,
  "message_clarity_score": 0.82
}
```

### Root Cause Analysis

```json
{
  "primary_cause": "Database connection pool exhaustion during peak traffic",
  "contributing_factors": [
    "Insufficient connection pool sizing",
    "Long-running queries not properly optimized",
    "Lack of connection timeout configuration"
  ],
  "evidence_summary": "Multiple timeout errors across user and payment services starting at 14:30",
  "confidence": 0.91,
  "timeline_analysis": "Issues escalated from warnings to critical errors over 15-minute period",
  "impact_scope": ["user-service", "payment-service", "database"]
}
```

### Suggested Rules

```json
{
  "suggested_rules": [
    {
      "rule_name": "database_connection_timeout",
      "pattern": "connection.*timeout.*database",
      "severity": "HIGH",
      "category": "PERFORMANCE",
      "description": "Detects database connection timeouts"
    }
  ]
}
```

### Optimization Tips
```bash
# Analyze smaller log samples for cost efficiency
spottr logs/large-file.log --use-llm --max-entries 100

# Use LLM only for critical analysis
spottr logs/*.log --use-llm --root-cause-analysis --no-llm-insights
```