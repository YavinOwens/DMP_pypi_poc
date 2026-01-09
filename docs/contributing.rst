Contributing
============

We welcome contributions to the Data Management GenAI package!

Getting Started
---------------

1. Fork the repository on GitHub
2. Clone your fork locally:

   .. code-block:: bash

      git clone https://github.com/YOUR_USERNAME/DMP_pypi_poc.git
      cd DMP_pypi_poc

3. Create a virtual environment:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

4. Install the package in development mode:

   .. code-block:: bash

      pip install -e ".[rag,jupyter]"

5. Install development dependencies:

   .. code-block:: bash

      pip install pytest black flake8 mypy

Development Guidelines
-----------------------

Code Style
~~~~~~~~~~

* Follow PEP 8 style guidelines
* Use type hints where possible
* Write docstrings for all public functions and classes
* Use Google-style docstrings

Testing
~~~~~~~

* Write tests for new features
* Ensure all tests pass before submitting
* Aim for good test coverage

Documentation
~~~~~~~~~~~~~

* Update documentation for new features
* Add examples where appropriate
* Keep docstrings up to date

Submitting Changes
------------------

1. Create a feature branch:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

2. Make your changes and commit:

   .. code-block:: bash

      git add .
      git commit -m "Add feature: description of changes"

3. Push to your fork:

   .. code-block:: bash

      git push origin feature/your-feature-name

4. Create a Pull Request on GitHub

Reporting Issues
---------------

If you find a bug or have a feature request, please open an issue on GitHub:

https://github.com/YavinOwens/DMP_pypi_poc/issues

Include:
* Description of the issue
* Steps to reproduce
* Expected vs actual behavior
* Environment details (Python version, OS, etc.)

License
-------

By contributing, you agree that your contributions will be licensed under the MIT License.
