Examples
========

This section provides practical examples of using the Data Management GenAI package.

Jupyter Notebook Example
-------------------------

A comprehensive Jupyter notebook example is available in the repository:

`examples/jupyter_example.ipynb <https://github.com/YavinOwens/DMP_pypi_poc/blob/main/examples/jupyter_example.ipynb>`_

This notebook demonstrates:

* Package installation and setup
* Snowflake connection configuration
* Model availability checking
* Running benchmarks
* Analyzing results
* Generating reports with RAG
* Data quality rules management
* Word document generation

Running the Example
~~~~~~~~~~~~~~~~~~~

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/YavinOwens/DMP_pypi_poc.git
      cd DMP_pypi_poc

2. Install the package:

   .. code-block:: bash

      pip install -e .

3. Open the notebook:

   .. code-block:: bash

      jupyter notebook examples/jupyter_example.ipynb

4. Configure your Snowflake credentials (see Installation section)

5. Run the cells sequentially

Code Examples
-------------

Complete Benchmark Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from datamanagement_genai import (
       get_snowflake_session,
       run_model_benchmarks,
       analyze_benchmark_results,
       generate_llm_report,
       create_word_document,
   )

   # Connect
   session = get_snowflake_session()

   # Run benchmarks
   results = run_model_benchmarks(session)

   # Analyze results
   analysis = analyze_benchmark_results(results)

   # Generate report
   report = generate_llm_report(
       session=session,
       results=results,
       use_rag=True
   )

   # Create Word document
   doc_path = create_word_document(
       session=session,
       report=report,
       output_path="benchmark_report.docx"
   )

Data Quality Rules Example
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from datamanagement_genai import DataQualityRulesManager

   # Initialize
   dq_manager = DataQualityRulesManager("data_quality_rules.csv")

   # Add a rule manually
   dq_manager.add_rule(
       business_rule="Customer IDs must be unique",
       data_quality_rule="validator.expect_column_values_to_be_unique(column='customer_id')",
       data_quality_dimension="Uniqueness",
       raci="Responsible"
   )

   # Use LLM to analyze and generate rules
   stats = dq_manager.analyze_and_update_with_llm(
       session=session,
       model="claude-3-5-sonnet",
       data_context={
           "table_name": "CUSTOMER",
           "schema_name": "SNOWFLAKE_SAMPLE_DATA.TPCH_SF1"
       },
       include_sample_data=True,
       include_statistics=True
   )

   # Get all rules
   rules = dq_manager.get_all_rules()
   for rule in rules:
       print(f"{rule['validation_number']}: {rule['business_rule']}")

RAG System Example
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from datamanagement_genai import RAGSystem

   # Initialize
   rag = RAGSystem(
       session=session,
       vector_store_database="RAG_KNOWLEDGE_BASE",
       vector_store_schema="PUBLIC"
   )

   # Store PDF documents
   rag.store_pdf_in_vector_store("knowledgebase_docs/")

   # Enhance a report section
   original_text = """
   Data quality is critical for business success. Organizations need to ensure
   their data is accurate, complete, and consistent.
   """

   enhanced = rag.enhance_report_section(
       section_text=original_text,
       model="claude-3-5-sonnet",
       top_k=5
   )

   print("Enhanced text with citations:")
   print(enhanced)
