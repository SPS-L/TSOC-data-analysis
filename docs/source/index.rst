.. TSOC Data Analysis documentation master file, created by
   sphinx-quickstart on Mon Jan 01 00:00:00 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TSOC Data Analysis's documentation!
==============================================

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
   :alt: License

.. image:: https://img.shields.io/badge/python-3.7+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python

.. image:: https://img.shields.io/badge/docs-sphinx-blue.svg
   :target: https://tsoc-data-analysis.sps-lab.org/
   :alt: Documentation

.. image:: https://img.shields.io/badge/GitHub-SPS--L%2FTSOC--data--analysis-blue.svg
   :target: https://github.com/SPS-L/TSOC-data-analysis
   :alt: GitHub Repository

**Author:** Sustainable Power Systems Lab (SPSL), `https://sps-lab.org <https://sps-lab.org>`_, contact: info@sps-lab.org

A comprehensive Python tool for analyzing the TSOC power system operational data from Excel files. The tool provides a powerful command-line interface (CLI) and modular Python API for load analysis, generator categorization, wind power analysis, reactive power calculations, and representative operating point extraction.

**Pure Python Implementation**: This tool is implemented entirely in Python. It can be used from the command line, imported as Python modules, or integrated into automated analysis pipelines.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   user_guide
   configuration
   examples
   troubleshooting
   license
   license_dependency

Features
--------

* **Month-based data filtering** for efficient processing of large datasets
* **Load calculations** (Total Load, Net Load) with comprehensive statistics
* **Wind power analysis** with generation statistics and profiles
* **Generator categorization** (Voltage Control vs PQ Control)
* **Reactive power analysis** with comprehensive calculations
* **Data validation** with type checking, limit validation, and gap filling
* **Representative operating points extraction** using K-means clustering with performance optimizations
* **Comprehensive logging** and error handling

Quick Start
-----------

.. code-block:: python

   from tsoc_data_analysis import execute, extract_representative_ops
   
   # Load and analyze data
   success, df = execute(month='2024-01', data_dir='raw_data', output_dir='results')
   if success:
       # Extract representative points
       rep_df, diagnostics = extract_representative_ops(
           df, max_power=850, MAPGL=200, output_dir='results'
       )

Command Line Usage
------------------

.. code-block:: bash

   # Run full analysis with all outputs for January 2024
   tsoc-analyze 2024-01 --output-dir results --save-plots --save-csv
   
   # Run analysis with specific data directory for March 2024
   tsoc-analyze 2024-03 --data-dir "raw_data" --verbose

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 