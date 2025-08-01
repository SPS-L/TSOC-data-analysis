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

A Python tool for analyzing the TSOC power system operational data from Excel files. The tool provides a modular Python API for load analysis, generator categorization, wind power analysis, reactive power calculations, and representative operating point extraction.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   user_guide
   configuration
   examples
   troubleshooting
   license

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

   from tsoc_data_analysis import execute, extract_representative_ops, extract_representative_ops_enhanced
   
   # Load and analyze data
   success, df = execute(month='2024-01', data_dir='raw_data', output_dir='results')
   if success:
       # Extract representative points (standard method)
       rep_df, diagnostics = extract_representative_ops(
           df, max_power=450, MAPGL=200, output_dir='results'
       )
       
       # Or use enhanced clustering for better quality
       enh_rep_df, enh_diagnostics = extract_representative_ops_enhanced(
           df, max_power=450, MAPGL=200, output_dir='results_enhanced',
           use_enhanced_preprocessing=True, try_alternative_algorithms=True
       )

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 