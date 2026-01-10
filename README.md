# Data Management GenAI Package

A Python package for Snowflake Cortex AI model benchmarking, data quality management, with RAG reporting.

Full documentation is available at: https://dmp-pypi-poc.readthedocs.io/en/latest/index.html

## Features

- **Model Benchmarking**: Test and compare Snowflake Cortex AI models with comprehensive metrics
- **Data Quality Management**: LLM-powered data quality rules generation and management
- **RAG System**: Retrieval-Augmented Generation for knowledge base enhancement
- **Multi-Backend Support**: Choose from Snowflake, Qdrant, ChromaDB, or FAISS vector stores
- **Report Generation**: Automated Word document generation with citations and data solution propositions
- **Jupyter Support**: Seamless integration with Jupyter notebooks
- **Token Counting**: Optional tiktoken support for precise token estimation
- **Data Quality Validation**: Optional great-expectations integration for code validation

## Installation

## Requirements

- Python 3.9 or higher
- Snowflake account with Cortex AI enabled
- Snowflake credentials (account, user, password, warehouse, database, schema)


### Optional Dependencies

See installation instructions above for optional dependency groups (rag, benchmark, jupyter, qdrant, chromadb, faiss, local-embeddings).


### From TestPyPI (Testing)

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ datamanagement-genai
```

### From GitHub (Development)

```bash
pip install git+https://github.com/YavinOwens/DMP_pypi_poc.git
```

### Local Development

```bash
pip install -e .
```

### Optional Dependencies

Install additional features as needed:

```bash
# RAG system support
pip install datamanagement-genai[rag]

# Benchmarking support (tiktoken, great-expectations)
pip install datamanagement-genai[benchmark]

# Jupyter notebook support
pip install datamanagement-genai[jupyter]

# Vector store backends
pip install datamanagement-genai[qdrant,chromadb,faiss]

# Local embeddings (sentence-transformers, torch)
pip install datamanagement-genai[local-embeddings]

# All features
pip install datamanagement-genai[rag,jupyter,qdrant,chromadb,faiss,local-embeddings,benchmark]
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

# Generate Word document with data solution proposition
doc = create_word_document(
    results=results,
    model_rankings=model_rankings,
    best_model=best_model,
    llm_report=report,
    session=session  # Required for data solution proposition generation
)
```

### Data Solution

- Data-driven insights from benchmark results
- Cost optimization recommendations
- Quality improvement metrics
- ROI and business value statements
- Professional consultative language suitable for executives

```python
from datamanagement_genai import create_word_document

# Data solution proposition is automatically generated when session is provided
doc = create_word_document(
    results=benchmark_results,
    model_rankings=rankings,
    best_model=best_model,
    session=database_session  # Enables data solution proposition generation
)
```

### In Jupyter Notebooks

The package works in Jupyter notebooks with helper functions for easier use. See `examples/jupyter_example.ipynb` for examples.

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

