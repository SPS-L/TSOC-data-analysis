Installation
===========

Requirements
------------

* Python 3.7 or higher
* pip (Python package installer)

Dependencies
-----------

The following Python packages are required:

* pandas>=1.3.0
* numpy>=1.20.0
* matplotlib>=3.3.0
* seaborn>=0.11.0
* openpyxl>=3.0.0
* scikit-learn>=1.0.0
* scipy>=1.7.0
* psutil>=5.8.0
* joblib>=1.1.0

Installation Methods
-------------------

From PyPI (when available)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install tsoc-data-analysis

From Source
~~~~~~~~~~~

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/sps-lab/tsoc-data-analysis.git
      cd tsoc-data-analysis

2. Install in development mode:

   .. code-block:: bash

      pip install -e .

   Or install with all development dependencies:

   .. code-block:: bash

      pip install -e ".[dev]"

3. Install with documentation dependencies:

   .. code-block:: bash

      pip install -e ".[docs]"

Using Requirements File
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install -r requirements.txt

Verification
-----------

After installation, you can verify that the package is working correctly:

.. code-block:: python

   import tsoc_data_analysis
   print(tsoc_data_analysis.__version__)

Or test the command-line interface:

.. code-block:: bash

   tsoc-analyze --help

Development Setup
----------------

For development work, install with development dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

This includes:

* pytest - for running tests
* pytest-cov - for test coverage
* black - for code formatting
* flake8 - for linting
* mypy - for type checking

Running Tests
~~~~~~~~~~~~

.. code-block:: bash

   pytest

With coverage:

.. code-block:: bash

   pytest --cov=tsoc_data_analysis

Code Formatting
~~~~~~~~~~~~~~

.. code-block:: bash

   black src/tsoc_data_analysis/

Linting
~~~~~~~

.. code-block:: bash

   flake8 src/tsoc_data_analysis/

Type Checking
~~~~~~~~~~~~~

.. code-block:: bash

   mypy src/tsoc_data_analysis/

Building Documentation
---------------------

1. Install documentation dependencies:

   .. code-block:: bash

      pip install -e ".[docs]"

2. Build the documentation:

   .. code-block:: bash

      cd docs
      make html

3. View the documentation by opening `docs/build/html/index.html` in your browser.

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~

**Import Error: No module named 'tsoc_data_analysis'**

Make sure you've installed the package in development mode:

.. code-block:: bash

   pip install -e .

**Missing Dependencies**

If you encounter missing dependency errors, install them manually:

.. code-block:: bash

   pip install pandas numpy matplotlib seaborn openpyxl scikit-learn scipy psutil joblib

**Permission Errors**

On some systems, you may need to use:

.. code-block:: bash

   pip install --user -e .

Or use a virtual environment:

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e . 