# Examples

This directory contains example usage of the `datamanagement_genai` package.

## Jupyter Notebook Example

`jupyter_example.ipynb` - A comprehensive Jupyter notebook demonstrating:

1. Package imports and setup
2. Snowflake connection
3. Model availability checking
4. Running model benchmarks
5. Analyzing results
6. Generating LLM reports with RAG
7. Data quality rules management
8. RAG system usage
9. Word document generation

### Usage

1. Open the notebook in Jupyter:
   ```bash
   jupyter notebook examples/jupyter_example.ipynb
   ```

2. Or use JupyterLab:
   ```bash
   jupyter lab examples/jupyter_example.ipynb
   ```

3. Make sure the package is installed in editable mode:
   ```bash
   # From the datamanagement_genai directory
   pip install -e .
   
   # Or from a Jupyter notebook cell:
   !pip install -e /Users/ymo/python_projects/datamanagement_genai
   ```
   
   **Important:** This is a local package (not on PyPI), so you must use `pip install -e .` from the package directory, or provide the full path.

4. Configure your Snowflake connection in `.streamlit/secrets.toml` or environment variables

5. Run the cells sequentially

## Notes

- The notebook is designed to work in any Jupyter environment (Jupyter Notebook, JupyterLab, VS Code, etc.)
- All paths are handled automatically by the package
- The notebook can be run interactively or converted to a script
- Some operations (like processing knowledge base PDFs) may take several minutes
