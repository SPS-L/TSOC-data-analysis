User Guide
==========

This guide provides comprehensive instructions for using the TSOC Data Analysis package, from basic installation to advanced analysis workflows.

Getting Started
--------------

The TSOC Data Analysis package is designed to be user-friendly while providing powerful analysis capabilities. This guide will walk you through the essential workflows and best practices.

Package Overview
---------------

The package consists of several interconnected modules:

**Core Analysis Modules**
- **Power System Analytics**: Load calculations, generator categorization, reactive power analysis
- **Operating Point Extractor**: Representative operating points extraction using K-means clustering

**Data Processing Modules**
- **Excel Data Processor**: Excel file loading and data preprocessing
- **Power Data Validator**: Comprehensive data validation and quality assurance

**Configuration and Utilities**
- **System Configuration**: Central configuration hub and shared utilities
- **Power System Visualizer**: Visualization and plotting functions

**User Interface**
- **Power Analysis CLI**: Command-line interface and main execution functions

Key Features
-----------

- **Month-based data filtering** for efficient processing of large datasets
- **Load calculations** (Total Load, Net Load) with comprehensive statistics
- **Wind power analysis** with generation statistics and profiles
- **Generator categorization** (Voltage Control vs PQ Control)
- **Reactive power analysis** with comprehensive calculations
- **Data validation** with type checking, limit validation, and gap filling
- **Representative operating points extraction** using K-means clustering with performance optimizations
- **Comprehensive logging** and error handling

Data Requirements
----------------

Input Data Format
~~~~~~~~~~~~~~~~

The package expects Excel files with the following structure:

**Substation Data**

* **Active Power (MW)**: ``substation_active_power.xlsx``
  * Column naming: ``ss_mw_[substation_name]``
* **Reactive Power (MVAR)**: ``substation_reactive_power.xlsx``
  * Column naming: ``ss_mvar_[substation_name]``
* **Structure**: Timestamps in column C (row 6+), substation names in row 2, data in row 6+

**Generator Data**

* **Voltage Setpoints (KV)**: ``generator_voltage_setpoints.xlsx``
  * Column naming: ``gen_v_[generator_name]``
* **Reactive Power (MVAR)**: ``generator_reactive_power.xlsx``
  * Column naming: ``gen_mvar_[generator_name]``
* **Structure**: Timestamps in column C (row 6+), generator names in row 3, data in row 6+

**Wind Farm Data**

* **Active Power (MW)**: ``wind_farm_active_power.xlsx``
  * Column naming: ``wind_mw_[wind_farm_name]``
  * **Structure**: Timestamps in column C (row 6+), wind farm names in row 3, data in row 6+

**Shunt Elements**

* **Reactive Power (MVAR)**: ``shunt_element_reactive_power.xlsx``
  * Column naming: ``shunt_mvar_[shunt_name]`` and ``shunt_tap_[shunt_name]``
  * **Structure**: Timestamps in column C (row 6+), shunt element names in row 3, data in row 6+

Data Quality Requirements
~~~~~~~~~~~~~~~~~~~~~~~~

* **Time Series Continuity**: Data should be continuous with regular time intervals
* **Unit Consistency**: All power values in MW/MVAR, voltages in KV
* **Sign Conventions**: 
  * Positive load values indicate consumption
  * Positive generation values indicate production
  * Generator reactive power is subtracted (negative contribution)
* **Missing Data**: Gaps up to 3 time steps are interpolated linearly

Basic Workflow
-------------

#. **Data Preparation**: Organize Excel files in the correct format
#. **Data Loading**: Use the package to load and merge data
#. **Data Validation**: Perform quality checks and gap filling
#. **Analysis**: Calculate loads, categorize generators, analyze wind power
#. **Representative Points**: Extract representative operating points using clustering
#. **Visualization**: Create plots and analysis dashboards
#. **Results**: Save analysis results and reports

Command Line Interface
---------------------

The package provides a powerful command-line interface for quick analysis:

**Basic Usage:**

.. code-block:: bash

   # Analyze January 2024 data
   tsoc-analyze 2024-01
   
   # Specify custom data and output directories
   tsoc-analyze 2024-01 --data-dir "2024-2025 data" --output-dir "results"
   
   # Enable verbose output and save plots
   tsoc-analyze 2024-01 --verbose --save-plots --save-csv

**Advanced Options:**

.. code-block:: bash

   # Extract representative points with custom parameters
   tsoc-analyze 2024-01 --max-power 850 --mapgl 200 --k-max 15
   
   # Use specific configuration file
   tsoc-analyze 2024-01 --config-file custom_config.json
   
   # Parallel processing for large datasets
   tsoc-analyze 2024-01 --n-jobs 4

**Output Options:**

.. code-block:: bash

   # Save all outputs
   tsoc-analyze 2024-01 --save-plots --save-csv --save-json
   
   # Specify output format
   tsoc-analyze 2024-01 --plot-format png --csv-separator ";"
   
   # Create comprehensive report
   tsoc-analyze 2024-01 --create-report

Python API Usage
---------------

For programmatic access and custom workflows:

**Basic Analysis:**

.. code-block:: python

   from tsoc_data_analysis import execute
   
   # Execute full analysis pipeline
   success, df = execute(
       month='2024-01',
       data_dir='raw_data',
       output_dir='results',
       save_plots=True,
       save_csv=True,
       verbose=True
   )
   
   if success:
       print(f"Analysis completed successfully")
       print(f"Data shape: {df.shape}")
   else:
       print("Analysis failed")

**Custom Analysis Workflow:**

.. code-block:: python

   from tsoc_data_analysis import (
       loadallpowerdf,
       calculate_total_load,
       calculate_net_load,
       categorize_generators,
       extract_representative_ops
   )
   
   # Step 1: Load data
   df = loadallpowerdf('2024-01', data_dir='raw_data')
   
   # Step 2: Calculate loads
   total_load = calculate_total_load(df)
   net_load = calculate_net_load(df)
   
   # Step 3: Categorize generators
   voltage_control, pq_control = categorize_generators(df)
   
   # Step 4: Extract representative points
   rep_df, diagnostics = extract_representative_ops(
       df,
       max_power=850,
       MAPGL=200,
       output_dir='results'
   )
   
   print(f"Analysis Results:")
   print(f"  Total load range: {total_load.min():.1f} - {total_load.max():.1f} MW")
   print(f"  Voltage control generators: {len(voltage_control)}")
   print(f"  Representative clusters: {len(rep_df)}")

**Data Validation:**

.. code-block:: python

   from tsoc_data_analysis import DataValidator
   
   # Create validator instance
   validator = DataValidator(df)
   
   # Perform validation
   validation_results = validator.validate_data()
   
   print(f"Validation Results:")
   print(f"  Valid records: {validation_results['valid_records']}")
   print(f"  Invalid records: {validation_results['invalid_records']}")
   print(f"  Missing values: {validation_results['missing_values']}")

Output Files
-----------

The package generates various output files depending on the options selected:

**Analysis Results:**
- `analysis_summary.txt` - Summary statistics and key metrics
- `load_statistics.csv` - Detailed load analysis results
- `generator_analysis.csv` - Generator categorization and statistics
- `wind_power_analysis.csv` - Wind farm analysis results

**Representative Points:**
- `representative_operating_points.csv` - Extracted representative points
- `clustering_summary.txt` - Clustering analysis summary
- `clustering_info.json` - Detailed clustering diagnostics

**Visualization:**
- `total_load_timeseries.png` - Total load time series plot
- `net_load_timeseries.png` - Net load time series plot
- `daily_load_profiles.png` - Daily load profile analysis
- `comprehensive_analysis.png` - Multi-panel analysis dashboard

**Configuration:**
- `analysis_config.json` - Configuration used for analysis
- `validation_report.json` - Data validation results

Best Practices
-------------

**Data Preparation:**
1. **File Organization**: Keep all Excel files in a dedicated data directory
2. **Naming Conventions**: Use consistent file names and column prefixes
3. **Data Quality**: Ensure data is clean and follows expected format
4. **Backup**: Always keep backup copies of original data files

**Analysis Workflow:**
1. **Start Simple**: Begin with basic analysis before advanced features
2. **Validate Data**: Always run data validation before analysis
3. **Parameter Tuning**: Adjust clustering parameters based on your system
4. **Documentation**: Keep records of analysis parameters and results

**Performance Optimization:**
1. **Data Filtering**: Use month filtering for large datasets
2. **Memory Management**: Process data in chunks if memory is limited
3. **Parallel Processing**: Enable parallel processing for faster analysis
4. **Caching**: Reuse validated data for multiple analyses

**Quality Assurance:**
1. **Cross-Validation**: Compare results across different time periods
2. **Sensitivity Analysis**: Test different parameter combinations
3. **Error Handling**: Monitor and address validation errors
4. **Result Verification**: Verify results against known system characteristics

Troubleshooting
--------------

**Common Issues:**

1. **File Not Found Errors**: Check file paths and naming conventions
2. **Data Quality Issues**: Run validation and address missing/invalid data
3. **Performance Problems**: Reduce dataset size or adjust parameters
4. **Memory Issues**: Process data in smaller chunks

**Getting Help:**

1. **Check Documentation**: Review this guide and API reference
2. **Error Messages**: Read error messages carefully for specific issues
3. **Sample Data**: Test with sample data to isolate problems
4. **Contact Support**: Reach out to info@sps-lab.org for persistent issues

**Debugging Tips:**

.. code-block:: python

   # Enable verbose mode for detailed output
   success, df = execute(month='2024-01', verbose=True)
   
   # Check data structure
   print(f"Data shape: {df.shape}")
   print(f"Columns: {list(df.columns)}")
   print(f"Data types: {df.dtypes}")
   
   # Validate data step by step
   from tsoc_data_analysis import DataValidator
   validator = DataValidator(df)
   results = validator.validate_data()
   print(f"Validation results: {results}")

Getting Help
-----------

**Additional Resources:**

1. **Documentation**: Complete API reference and examples
2. **Examples**: Step-by-step examples for common workflows
3. **Configuration Guide**: Detailed configuration options
4. **Troubleshooting**: Solutions to common problems

**Support Channels:**

- **Email**: info@sps-lab.org
- **Documentation**: https://sps-lab.org/tsoc-data-analysis
- **Issues**: GitHub issue tracker (if available)

**Before Seeking Help:**

1. **Check Documentation**: Review relevant sections of this guide
2. **Test with Sample Data**: Verify the issue with known good data
3. **Provide Details**: Include error messages, data structure, and configuration
4. **Reproduce Issue**: Ensure the problem can be reproduced consistently 