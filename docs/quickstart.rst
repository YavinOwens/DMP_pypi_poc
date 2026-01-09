Quick Start Guide
=================

This guide will help you get started with the Data Management GenAI package.

Basic Usage
-----------

Connect to Snowflake
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from datamanagement_genai import get_snowflake_session

   session = get_snowflake_session()
   if session:
       print("Connected successfully!")
   else:
       print("Connection failed - check your configuration")

Run Model Benchmarks
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from datamanagement_genai import run_model_benchmarks

   results = run_model_benchmarks(session)
   
   # View results
   for result in results:
       print(f"Model: {result['model']}")
       print(f"Test: {result['test_name']}")
       print(f"Quality: {result['quality_score']}/10")
       print(f"Cost: ${result['estimated_cost']:.6f}")
       print()

Generate Reports
~~~~~~~~~~~~~~~~

.. code-block:: python

   from datamanagement_genai import generate_llm_report, create_word_document

   # Generate LLM report
   report = generate_llm_report(
       session=session,
       results=results,
       use_rag=True  # Enable RAG enhancement
   )

   # Create Word document
   doc_path = create_word_document(
       session=session,
       report=report,
       output_path="report.docx"
   )

Data Quality Rules Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from datamanagement_genai import DataQualityRulesManager

   # Initialize manager
   dq_manager = DataQualityRulesManager("data_quality_rules.csv")

   # Analyze data and generate rules
   stats = dq_manager.analyze_and_update_with_llm(
       session=session,
       model="claude-3-5-sonnet",
       data_context={
           "table_name": "LINEITEM",
           "schema_name": "SNOWFLAKE_SAMPLE_DATA.TPCH_SF1"
       }
   )

   print(f"Created {stats['created']} rules")
   print(f"Updated {stats['updated']} rules")

RAG System Usage
~~~~~~~~~~~~~~~~

.. code-block:: python

   from datamanagement_genai import RAGSystem

   # Initialize RAG system
   rag = RAGSystem(
       session=session,
       vector_store_database="RAG_KNOWLEDGE_BASE",
       vector_store_schema="PUBLIC"
   )

   # Store PDFs in vector store
   rag.store_pdf_in_vector_store("knowledgebase_docs/")

   # Enhance text with knowledge base
   enhanced = rag.enhance_report_section(
       section_text="Data quality is important...",
       model="claude-3-5-sonnet"
   )

Jupyter Notebook Usage
~~~~~~~~~~~~~~~~~~~~~~

The package works seamlessly in Jupyter notebooks:

.. code-block:: python

   # Configure verbosity for cleaner output
   from datamanagement_genai import set_verbosity
   set_verbosity(verbose=False)

   # Use helper functions
   from datamanagement_genai import quick_benchmark, display_results_as_dataframe

   # Run quick benchmark
   results_df = quick_benchmark(session)
   display_results_as_dataframe(results_df)

For more examples, see the `examples/jupyter_example.ipynb` notebook.
