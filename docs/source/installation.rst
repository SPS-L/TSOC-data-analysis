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

.. code-block:: bash

   pip install git+https://github.com/SPS-L/TSOC-data-analysis.git

Verification
-----------

After installation, you can verify that the package is working correctly:

.. code-block:: python

   import tsoc_data_analysis
   print(tsoc_data_analysis.__version__)

Or test the command-line interface:

.. code-block:: bash

   tsoc-analyze --help