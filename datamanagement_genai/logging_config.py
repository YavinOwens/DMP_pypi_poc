"""
Logging configuration for datamanagement_genai package

Provides utilities to configure logging with appropriate verbosity levels
for different environments (Jupyter notebooks, scripts, etc.)
"""

import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

# Detect if running in Jupyter/IPython
def is_jupyter_environment() -> bool:
    """Check if code is running in a Jupyter notebook or IPython"""
    try:
        # Check for IPython
        if 'ipykernel' in sys.modules:
            return True
        # Check for Jupyter
        if 'IPython' in sys.modules:
            from IPython import get_ipython
            ipython = get_ipython()
            if ipython and ipython.__class__.__name__ == 'ZMQInteractiveShell':
                return True
    except:
        pass
    return False


def configure_logging(
    level: Optional[str] = None,
    verbose: bool = False,
    log_file: Optional[Path] = None
) -> None:
    """
    Configure logging for the package with appropriate verbosity
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR). If None, auto-detects based on environment
        verbose: If True, use INFO level. If False, use WARNING level in Jupyter
        log_file: Optional path to log file. If None, uses package .tmp directory
    """
    # Auto-detect if in Jupyter
    in_jupyter = is_jupyter_environment()
    
    # Determine log level
    if level is None:
        if verbose:
            log_level = logging.INFO
        elif in_jupyter:
            # Less verbose in Jupyter by default
            log_level = logging.WARNING
        else:
            log_level = logging.INFO
    else:
        log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Set up log file path
    if log_file is None:
        package_root = Path(__file__).parent.parent
        log_file = package_root / ".tmp" / "datamanagement_genai.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # File handler (always INFO for debugging)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler (respects log_level)
    if in_jupyter and not verbose:
        # In Jupyter, use a simpler format and higher threshold
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
    else:
        # In scripts, use full format
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
    
    root_logger.addHandler(console_handler)
    
    # Suppress verbose third-party logs
    logging.getLogger('snowflake.connector').setLevel(logging.WARNING)
    logging.getLogger('snowflake.snowpark').setLevel(logging.WARNING)
    logging.getLogger('snowflake.connector.connection').setLevel(logging.WARNING)
    logging.getLogger('snowflake.snowpark.session').setLevel(logging.WARNING)
    
    # Suppress boto3 deprecation warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='boto3')
    
    # Suppress urllib3 warnings
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    # Package-specific loggers - can be more verbose if needed
    package_logger = logging.getLogger('datamanagement_genai')
    if verbose:
        package_logger.setLevel(logging.INFO)
    else:
        package_logger.setLevel(log_level)


def set_verbosity(verbose: bool = True) -> None:
    """
    Quick function to toggle verbosity in Jupyter notebooks
    
    Args:
        verbose: If True, show INFO logs. If False, only show WARNING and above
    """
    configure_logging(verbose=verbose)


# Auto-configure on import if in Jupyter
if is_jupyter_environment():
    configure_logging(verbose=False)
