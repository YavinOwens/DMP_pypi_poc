# Data Management GenAI Package

A comprehensive Python package for Snowflake Cortex AI model benchmarking, data quality management, and RAG-enhanced reporting.

## Features

- **Model Benchmarking**: Test and compare Snowflake Cortex AI models
- **Data Quality Management**: LLM-powered data quality rules generation and management
- **RAG System**: Retrieval-Augmented Generation for knowledge base enhancement
- **Report Generation**: Automated Word document generation with citations

## Installation

```bash
pip install -e .
```

## Usage

### As a Package

```python
from datamanagement_genai import (
    get_snowflake_session,
    run_model_benchmarks,
    generate_llm_report,
    create_word_document,
    DataQualityRulesManager,
    RAGSystem,
)

# Connect to Snowflake
session = get_snowflake_session()

# Run benchmarks
results = run_model_benchmarks(session)

# Generate reports
report = generate_llm_report(session, results)
```

### In Jupyter Notebooks

The package works seamlessly in Jupyter notebooks with helper functions for easier use. See `examples/jupyter_example.ipynb` for a comprehensive example.

```python
# In a Jupyter notebook cell
from datamanagement_genai import (
    get_snowflake_session, 
    run_model_benchmarks,
    quick_benchmark,
    display_results_as_dataframe,
)

# Quick benchmark with automatic DataFrame display
session = get_snowflake_session()
results_df = quick_benchmark(session, max_tests=2)  # Quick test with 2 tests per model
results_df  # Automatically displays as DataFrame in Jupyter
```

The package includes Jupyter-specific helpers:
- `quick_benchmark()` - Run benchmarks with automatic DataFrame conversion
- `display_results_as_dataframe()` - Convert results to pandas DataFrame
- `get_config_summary()` - Get configuration summary for display
- `get_package_root()` - Get package root directory (works in any environment)

### As a CLI

```bash
datamanagement-genai
```

## Package Structure

```
datamanagement_genai/
├── datamanagement_genai/
│   ├── __init__.py
│   ├── benchmark.py      # Core benchmarking functionality
│   ├── config.py         # Configuration management
│   ├── helpers.py        # Snowflake helpers
│   ├── reporting.py      # Report generation
│   ├── rag/
│   │   ├── __init__.py
│   │   └── system.py     # RAG system
│   ├── data_quality/
│   │   ├── __init__.py
│   │   └── rules_manager.py  # Data quality rules management
│   └── cli/
│       └── __init__.py   # CLI entry point
├── setup.py
├── pyproject.toml
├── requirements.txt
├── test_config.json
└── data_quality_rules.csv
```

## Requirements

- Python 3.9+
- Snowflake account with Cortex AI access
- See `requirements.txt` for dependencies

## License

MIT
