from spottr import LogAnalyzer


def test_log_analyzer_initialization():
    """Test the initialization of LogAnalyzer."""
    analyzer = LogAnalyzer()
    assert isinstance(analyzer, LogAnalyzer), (
        "LogAnalyzer should be initialized correctly."
    )
