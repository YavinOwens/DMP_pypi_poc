"""
Data Management GenAI Package

A comprehensive Python package for Snowflake Cortex AI model benchmarking,
data quality management, and RAG-enhanced reporting.
"""

__version__ = "0.1.4"
__author__ = "Yavin.O"

# Configure logging early to suppress third-party verbose messages
# This must happen before any imports that might trigger logging
try:
    from .logging_config import set_verbosity, configure_logging, is_jupyter_environment
    # Configure logging early to suppress great_expectations and other verbose logs
    import logging
    if not logging.getLogger().handlers:
        configure_logging(verbose=False)
    # Ensure great_expectations logs are suppressed even if imported later
    logging.getLogger('great_expectations').setLevel(logging.WARNING)
    logging.getLogger('great_expectations._docs_decorators').setLevel(logging.WARNING)
    logging.getLogger('great_expectations.expectations.registry').setLevel(logging.WARNING)
except (ImportError, AttributeError):
    # If import fails, create stub functions and configure basic logging
    import logging
    logging.getLogger('great_expectations').setLevel(logging.WARNING)
    logging.getLogger('great_expectations._docs_decorators').setLevel(logging.WARNING)
    logging.getLogger('great_expectations.expectations.registry').setLevel(logging.WARNING)
    
    def set_verbosity(verbose: bool = True) -> None:
        """Stub function if logging_config not available"""
        pass
    
    def configure_logging(level=None, verbose=False, log_file=None) -> None:
        """Stub function if logging_config not available"""
        pass
    
    def is_jupyter_environment() -> bool:
        """Stub function if logging_config not available"""
        return False

# Core imports
from .helpers import get_snowflake_session, get_table_columns
from .config import load_test_config, get_models, get_test_prompts
from .benchmark import (
    run_model_benchmarks,
    check_model_availability,
    generate_llm_report,
    create_word_document,
    analyze_benchmark_results,
    execute_cortex_query,
)
from .reporting import generate_full_report

# Subpackage imports
from .rag.system import RAGSystem
from .data_quality.rules_manager import DataQualityRulesManager

# Jupyter helpers (optional, for notebook environments)
try:
    from .jupyter_helpers import (
        get_package_root,
        display_results_as_dataframe,
        get_config_summary,
        quick_benchmark,
    )
except ImportError:
    # pandas might not be available
    get_package_root = None
    display_results_as_dataframe = None
    get_config_summary = None
    quick_benchmark = None

__all__ = [
    "__version__",
    "get_snowflake_session",
    "get_table_columns",
    "load_test_config",
    "get_models",
    "get_test_prompts",
    "run_model_benchmarks",
    "check_model_availability",
    "generate_llm_report",
    "create_word_document",
    "analyze_benchmark_results",
    "execute_cortex_query",
    "generate_full_report",
    "RAGSystem",
    "DataQualityRulesManager",
    "get_package_root",
    "display_results_as_dataframe",
    "get_config_summary",
    "quick_benchmark",
    "set_verbosity",
    "configure_logging",
    "is_jupyter_environment",
]
