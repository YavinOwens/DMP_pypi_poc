Data Management GenAI Documentation
====================================

Welcome to the Data Management GenAI package documentation!

This package provides comprehensive tools for Snowflake Cortex AI model benchmarking, data quality management, and RAG-enhanced reporting.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index
   examples
   contributing

Features
--------

* **Model Benchmarking**: Test and compare Snowflake Cortex AI models
* **Data Quality Management**: LLM-powered data quality rules generation and management
* **RAG System**: Retrieval-Augmented Generation for knowledge base enhancement
* **Multi-Backend Support**: Choose from Snowflake, Qdrant, ChromaDB, or FAISS vector stores
* **Report Generation**: Automated Word document generation with citations
* **Jupyter Support**: Seamless integration with Jupyter notebooks

Installation
------------

Install the package using pip:

.. code-block:: bash

   pip install datamanagement-genai

For optional features:

.. code-block:: bash

   # With RAG support
   pip install datamanagement-genai[rag]

   # With Jupyter support
   pip install datamanagement-genai[jupyter]

   # With vector store backends
   pip install datamanagement-genai[qdrant,chromadb,faiss]

   # With all features
   pip install datamanagement-genai[rag,jupyter,qdrant,chromadb,faiss]

Quick Start
-----------

.. code-block:: python

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

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
