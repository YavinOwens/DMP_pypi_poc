"""
Configuration management for Cortex AI benchmarking
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Default models (fallback if config file not found)
DEFAULT_MODELS = {
    "claude-3-5-sonnet": {
        "display": "Claude 3.5 Sonnet",
        "cost_per_1M_input": 3.00,
        "cost_per_1M_output": 15.00,
    },
    "mistral-7b": {
        "display": "Mistral 7B",
        "cost_per_1M_input": 0.14,
        "cost_per_1M_output": 0.14,
    },
    "mixtral-8x7b": {
        "display": "Mixtral 8x7B",
        "cost_per_1M_input": 0.24,
        "cost_per_1M_output": 0.24,
    },
}

# Default test prompts
DEFAULT_TEST_PROMPTS = {
    "data_quality_gx": {
        "natural_language": "The L_ORDERKEY column should not be null and should be unique",
        "table": "LINEITEM",
        "schema": "SNOWFLAKE_SAMPLE_DATA.TPCH_SF1",
        "expected_elements": ["expect_column_values_to_not_be_null", "expect_column_values_to_be_unique", "L_ORDERKEY", "LINEITEM"],
        "test_type": "code_generation"
    },
    "sql_query_generation": {
        "prompt": "Write a SQL query to find all customers from the CUSTOMER table in SNOWFLAKE_SAMPLE_DATA.TPCH_SF1 who have a C_ACCTBAL greater than 5000, ordered by C_ACCTBAL descending. Return C_CUSTKEY, C_NAME, and C_ACCTBAL.",
        "expected_elements": ["SELECT", "FROM", "CUSTOMER", "C_ACCTBAL", "WHERE", "ORDER BY"],
        "test_type": "sql_generation"
    },
}

# Default token limits
DEFAULT_TOKEN_LIMITS = {
    "claude-3-5-sonnet": 200000,
    "mistral-7b": 8192,
    "mixtral-8x7b": 32768,
}


def load_test_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load test configuration from JSON file
    
    Args:
        config_path: Path to config file (default: test_config.json in current directory)
    
    Returns:
        Dictionary with 'models', 'test_prompts', 'token_limits', and 'settings'
    """
    if config_path is None:
        # Try to find config in current directory or package directory
        current_dir = Path.cwd()
        package_dir = Path(__file__).parent.parent
        config_path = current_dir / "test_config.json"
        if not config_path.exists():
            config_path = package_dir / "test_config.json"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {
            "models": DEFAULT_MODELS,
            "test_prompts": DEFAULT_TEST_PROMPTS,
            "token_limits": DEFAULT_TOKEN_LIMITS,
            "settings": {
                "parallel_execution": True,
                "max_workers": 3,
                "output_directory": "tests",
                "log_level": "INFO"
            }
        }
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        logger.info(f"Loaded configuration from: {config_path}")
        
        # Validate and merge with defaults
        models = config.get("models", DEFAULT_MODELS)
        test_prompts = config.get("test_prompts", DEFAULT_TEST_PROMPTS)
        token_limits = config.get("token_limits", DEFAULT_TOKEN_LIMITS)
        settings = config.get("settings", {
            "parallel_execution": True,
            "max_workers": 3,
            "output_directory": "tests",
            "log_level": "INFO"
        })
        
        # Filter out disabled models and test prompts
        models = {k: v for k, v in models.items() if v.get("enabled", True)}
        test_prompts = {k: v for k, v in test_prompts.items() if v.get("enabled", True)}
        
        logger.info(f"Loaded {len(models)} enabled models and {len(test_prompts)} enabled test prompts")
        
        return {
            "models": models,
            "test_prompts": test_prompts,
            "token_limits": token_limits,
            "settings": settings
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing config file {config_path}: {e}, using defaults")
        return {
            "models": DEFAULT_MODELS,
            "test_prompts": DEFAULT_TEST_PROMPTS,
            "token_limits": DEFAULT_TOKEN_LIMITS,
            "settings": {
                "parallel_execution": True,
                "max_workers": 3,
                "output_directory": "tests",
                "log_level": "INFO"
            }
        }
    except Exception as e:
        logger.error(f"Error loading config file {config_path}: {e}, using defaults", exc_info=True)
        return {
            "models": DEFAULT_MODELS,
            "test_prompts": DEFAULT_TEST_PROMPTS,
            "token_limits": DEFAULT_TOKEN_LIMITS,
            "settings": {
                "parallel_execution": True,
                "max_workers": 3,
                "output_directory": "tests",
                "log_level": "INFO"
            }
        }


def get_models(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get models from config"""
    if config is None:
        config = load_test_config()
    return config.get("models", DEFAULT_MODELS)


def get_test_prompts(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get test prompts from config"""
    if config is None:
        config = load_test_config()
    return config.get("test_prompts", DEFAULT_TEST_PROMPTS)
