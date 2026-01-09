Installation
============

The Data Management GenAI package can be installed from PyPI or directly from GitHub.

From PyPI
---------

.. code-block:: bash

   pip install datamanagement-genai

From GitHub
-----------

.. code-block:: bash

   pip install git+https://github.com/YavinOwens/DMP_pypi_poc.git

Optional Dependencies
---------------------

The package supports optional dependencies for extended functionality:

RAG System Support
~~~~~~~~~~~~~~~~~~

For RAG (Retrieval-Augmented Generation) features:

.. code-block:: bash

   pip install datamanagement-genai[rag]

This includes:
* langchain
* langchain-community
* pypdf
* PyPDF2
* pdfplumber

Jupyter Notebook Support
~~~~~~~~~~~~~~~~~~~~~~~~

For enhanced Jupyter notebook integration:

.. code-block:: bash

   pip install datamanagement-genai[jupyter]

This includes:
* jupyter
* ipykernel

All Features
~~~~~~~~~~~~

To install with all optional dependencies:

.. code-block:: bash

   pip install datamanagement-genai[rag,jupyter]

Requirements
------------

* Python 3.9 or higher
* Snowflake account with Cortex AI enabled
* Snowflake credentials (account, user, password, warehouse, database, schema)

Configuration
-------------

Configure your Snowflake connection using one of these methods:

1. **Config file** (recommended): Create `config.toml` or `.streamlit/secrets.toml`:

   .. code-block:: toml

      [snowflake]
      account = "your-account"
      user = "your-user"
      password = "your-password"
      warehouse = "COMPUTE_WH"
      database = "SNOWFLAKE_SAMPLE_DATA"
      schema = "TPCH_SF1"

2. **Environment variables**:

   .. code-block:: bash

      export SNOWFLAKE_ACCOUNT="your-account"
      export SNOWFLAKE_USER="your-user"
      export SNOWFLAKE_PASSWORD="your-password"
      export SNOWFLAKE_WAREHOUSE="COMPUTE_WH"
      export SNOWFLAKE_DATABASE="SNOWFLAKE_SAMPLE_DATA"
      export SNOWFLAKE_SCHEMA="TPCH_SF1"

Verification
-----------

Test your installation:

.. code-block:: python

   from datamanagement_genai import get_snowflake_session

   session = get_snowflake_session()
   if session:
       print("✓ Installation successful!")
   else:
       print("✗ Connection failed - check your configuration")
