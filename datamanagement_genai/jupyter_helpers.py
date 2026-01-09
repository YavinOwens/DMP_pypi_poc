"""
Jupyter Notebook Helper Functions

Utility functions to make the package easier to use in Jupyter notebooks.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Union

# Optional pandas import (for Jupyter environments)
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None


def get_package_root() -> Path:
    """
    Get the package root directory, works in both script and Jupyter environments.
    
    Returns:
        Path to package root directory
    """
    # Try to find package root by looking for setup.py or pyproject.toml
    current = Path.cwd()
    
    # Check current directory
    if (current / "setup.py").exists() or (current / "pyproject.toml").exists():
        return current
    
    # Check parent directories (up to 3 levels)
    for parent in [current.parent, current.parent.parent, current.parent.parent.parent]:
        if (parent / "setup.py").exists() or (parent / "pyproject.toml").exists():
            return parent
    
    # Fallback to current directory
    return current


def display_results_as_dataframe(results: list):
    """
    Convert benchmark results to a pandas DataFrame for easy display in Jupyter.
    
    Args:
        results: List of benchmark result dictionaries
        
    Returns:
        pandas DataFrame with results (or list if pandas not available)
    """
    if not PANDAS_AVAILABLE:
        return results
    
    if not results:
        return pd.DataFrame()
    
    # Flatten nested dictionaries for better display
    flattened = []
    for result in results:
        flat_result = result.copy()
        # Flatten nested dicts
        if "metadata" in flat_result:
            for key, value in flat_result["metadata"].items():
                flat_result[f"metadata_{key}"] = value
            del flat_result["metadata"]
        flattened.append(flat_result)
    
    return pd.DataFrame(flattened)


def get_config_summary(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get a summary of the configuration for display in Jupyter.
    
    Args:
        config: Optional config dict (if None, loads from file)
        
    Returns:
        Dictionary with configuration summary
    """
    from .config import load_test_config, get_models, get_test_prompts
    
    if config is None:
        config = load_test_config()
    
    models = get_models(config)
    test_prompts = get_test_prompts(config)
    
    return {
        "models": {
            name: {
                "display": info.get("display", name),
                "enabled": info.get("enabled", True),
            }
            for name, info in models.items()
        },
        "test_prompts": {
            name: {
                "test_type": info.get("test_type", "unknown"),
                "enabled": info.get("enabled", True),
            }
            for name, info in test_prompts.items()
        },
        "settings": config.get("settings", {}),
    }


def quick_benchmark(
    session,
    models: Optional[list] = None,
    max_tests: Optional[int] = None,
    display_results: bool = True,
):
    """
    Quick benchmark function for Jupyter notebooks.
    
    Args:
        session: Snowflake session
        models: Optional list of models to test (default: all available)
        max_tests: Optional maximum number of tests per model (for quick testing)
        display_results: Whether to display results as DataFrame (if pandas available)
        
    Returns:
        pandas DataFrame with results (or list if pandas not available or display_results=False)
    """
    from .benchmark import run_model_benchmarks, check_model_availability
    from .config import load_test_config, get_models
    
    # Load config
    config = load_test_config()
    available_models = get_models(config)
    
    # Check availability
    if models is None:
        availability = check_model_availability(session, list(available_models.keys()))
        models = [m for m, avail in availability.items() if avail]
    
    # Run benchmarks
    results = run_model_benchmarks(session, models_to_test=models)
    
    # Limit results if requested
    if max_tests and results:
        # Group by model and limit
        model_results = {}
        for result in results:
            model = result["model"]
            if model not in model_results:
                model_results[model] = []
            if len(model_results[model]) < max_tests:
                model_results[model].append(result)
        
        results = []
        for model_results_list in model_results.values():
            results.extend(model_results_list)
    
    if display_results:
        return display_results_as_dataframe(results)
    else:
        return results


__all__ = [
    "get_package_root",
    "display_results_as_dataframe",
    "get_config_summary",
    "quick_benchmark",
]
