Power Analysis CLI Module
=========================

.. automodule:: tsoc_data_analysis.power_analysis_cli
   :members:
   :undoc-members:
   :show-inheritance:

Module Overview
--------------

The ``power_analysis_cli`` module provides the command-line interface (CLI) for the TSOC Data Analysis package. It serves as the main entry point for users who prefer command-line operation and provides a comprehensive analysis pipeline that orchestrates all other modules.

The module includes the main execution function and command-line argument parsing, making it easy to run complete power system analysis workflows from the command line.

Core Functions
-------------

execute()
~~~~~~~~~

Main execution function that orchestrates the complete analysis pipeline.

.. code-block:: python

   def execute(month=None, data_dir='raw_data', output_dir='results', 
               save_csv=False, save_plots=False, summary_only=False, 
               verbose=False, input_df=None):
       """
       Execute complete power system analysis pipeline.
       
       Args:
           month: Month filter (format: 'YYYY-MM') or None for all data
           data_dir: Directory containing Excel data files
           output_dir: Directory to save output files
           save_csv: Whether to save analysis results to CSV files
           save_plots: Whether to generate and save plots
           summary_only: Whether to generate only summary report
           verbose: Whether to print detailed progress information
           input_df: Pre-loaded DataFrame (optional)
           
       Returns:
           Tuple of (success_boolean, result_dataframe)
       """

**Example:**
.. code-block:: python

   from tsoc_data_analysis import execute
   
   # Run complete analysis
   success, df = execute(
       month='2024-01',
       data_dir='raw_data',
       output_dir='results',
       save_csv=True,
       save_plots=True,
       verbose=True
   )
   
   if success:
       print(f"Analysis completed successfully. Processed {len(df)} records.")

main()
~~~~~~

Command-line interface main function.

.. code-block:: python

   def main():
       """
       Main command-line interface function.
       
       Parses command-line arguments and executes the analysis pipeline.
       """

**Command Line Usage:**
.. code-block:: bash

   # Basic usage
   tsoc-analyze 2024-01
   
   # With options
   tsoc-analyze 2024-01 --output-dir results --save-plots --save-csv --verbose
   
   # All data (no month filter)
   tsoc-analyze --output-dir results --save-plots --save-csv

Command Line Interface
---------------------

The CLI provides a powerful interface for running power system analysis:

**Basic Syntax:**
.. code-block:: bash

   tsoc-analyze [MONTH] [OPTIONS]
   tsoc-analyze --help
   tsoc-analyze --version
```

**Arguments:**
- **MONTH**: Month to filter data for (format: "YYYY-MM") or None for all data

**Options:**
- **--data-dir**: Directory containing Excel data files (default: `raw_data`)
- **--output-dir**: Directory to save output files (default: `results`)
- **--save-csv**: Save analysis results to CSV files
- **--save-plots**: Generate and save plots as PNG files
- **--summary-only**: Generate only summary report
- **--verbose**: Print detailed progress information
- **--version**: Show version information
- **-h, --help**: Show help message

**Usage Examples:**

Basic Analysis
~~~~~~~~~~~~~

.. code-block:: bash

   # Run full analysis with all outputs for January 2024
   tsoc-analyze 2024-01 --output-dir results --save-plots --save-csv

   # Run analysis with specific data directory for March 2024
   tsoc-analyze 2024-03 --data-dir "2024-2025 data" --verbose

   # Run analysis and save only summary report for December 2024
   tsoc-analyze 2024-12 --output-dir results --summary-only

Advanced Options
~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run analysis for all data (no month filter)
   tsoc-analyze --output-dir results --save-plots --save-csv

   # Quick analysis for a specific month with summary only
   tsoc-analyze 2024-06 --summary-only

   # Verbose analysis with custom directories
   tsoc-analyze 2024-09 --data-dir "custom_data" --output-dir "custom_results" --verbose

Analysis Pipeline
----------------

The CLI orchestrates a comprehensive analysis pipeline:

**Step 1: Data Loading**
- Load Excel files from specified directory
- Apply month filtering if specified
- Merge data from multiple sources
- Validate data structure and quality

**Step 2: Data Validation**
- Perform comprehensive data validation
- Apply gap filling and outlier detection
- Ensure data quality standards
- Generate validation reports

**Step 3: Power System Analysis**
- Calculate total and net load
- Analyze wind power generation
- Categorize generators (Voltage Control vs PQ Control)
- Calculate reactive power balance
- Generate load statistics

**Step 4: Representative Points Extraction**
- Extract representative operating points using clustering
- Apply MAPGL belt analysis
- Generate clustering diagnostics
- Create comprehensive reports

**Step 5: Visualization**
- Create time series plots
- Generate daily and monthly profiles
- Produce comprehensive analysis dashboards
- Save publication-quality figures

**Step 6: Output Generation**
- Save analysis results to CSV files
- Generate summary reports
- Create detailed logging
- Organize output files

Output Files
-----------

The CLI generates comprehensive output files:

**Analysis Results:**
- `analysis_summary.json` - Complete analysis results in JSON format
- `analysis_summary.txt` - Human-readable analysis summary
- `total_load.csv` - Total load time series data
- `net_load.csv` - Net load time series data
- `load_statistics.csv` - Load statistics (min, max, mean, std)
- `generator_categories.csv` - Generator categorization results
- `comprehensive_power_data.csv` - All power system data in one file

**Representative Operating Points:**
- `representative_operating_points.csv` - Clean representative points with readable column names
- `clustering_summary.txt` - User-friendly clustering analysis report with quality assessment
- `clustering_info.json` - Detailed clustering metrics for programmatic access
- `clustering_visualization.png` - Comprehensive 9-panel visualization dashboard

**Plot Files** (with `--save-plots`):
- `load_timeseries.png` - Total and net load time series
- `daily_profile.png` - Average daily load profiles
- `monthly_profile.png` - Monthly load profiles
- `comprehensive_analysis.png` - Combined analysis plots

**Log Files:**
- `logs/analysis_YYYY-MM.log` - Detailed analysis logs with timestamps

Configuration Integration
------------------------

The CLI uses configuration from `system_configuration.py`:

**Default Settings:**
.. code-block:: python

   # Default settings
   DEFAULT_OUTPUT_DIR = 'results'
   DEFAULT_VERBOSE = False
   DATA_DIR = 'raw_data/'

**File Mappings:**
.. code-block:: python

   FILES = {
       'substation_mw': 'substation_active_power.xlsx',
       'substation_mvar': 'substation_reactive_power.xlsx', 
       'wind_power': 'wind_farm_active_power.xlsx',
       'shunt_elements': 'shunt_element_reactive_power.xlsx',
       'gen_voltage': 'generator_voltage_setpoints.xlsx',
       'gen_mvar': 'generator_reactive_power.xlsx'
   }

**Customization:**
Users can customize behavior by modifying configuration values:

.. code-block:: python

   # Custom default directories
   DEFAULT_OUTPUT_DIR = 'custom_results'
   DATA_DIR = 'custom_data/'
   
   # Custom file mappings
   FILES['substation_mw'] = 'custom_substation_power.xlsx'

Error Handling
-------------

**Comprehensive Error Handling:**
- **File Not Found**: Graceful handling of missing data files
- **Data Quality Issues**: Validation and reporting of data problems
- **Configuration Errors**: Clear error messages for configuration issues
- **Memory Issues**: Handling of large datasets and memory constraints
- **Permission Errors**: File access and permission handling

**User-Friendly Messages:**
- **Progress Information**: Detailed progress reporting with `--verbose`
- **Error Descriptions**: Clear explanations of what went wrong
- **Suggestions**: Helpful suggestions for fixing common issues
- **Logging**: Comprehensive logging for debugging

**Example Error Handling:**
.. code-block:: bash

   # Missing data directory
   tsoc-analyze 2024-01 --data-dir "nonexistent"
   # Error: Data directory 'nonexistent' not found
   # Suggestion: Check the directory path and ensure it exists

   # Invalid month format
   tsoc-analyze 2024-13 --data-dir "raw_data"
   # Error: Invalid month format '2024-13'. Expected format: YYYY-MM
   # Suggestion: Use a valid month (01-12)

Performance Considerations
-------------------------

**Optimization Features:**
- **Month Filtering**: Efficient processing of large datasets
- **Parallel Processing**: Uses parallel processing for clustering analysis
- **Memory Management**: Efficient handling of large datasets
- **Progress Tracking**: Detailed progress information for long-running analyses

**Scalability:**
- **Large Datasets**: Handles datasets with thousands of time points
- **Many Variables**: Efficiently processes hundreds of power system variables
- **Multiple Files**: Scales to process dozens of Excel files
- **Memory Management**: Optimized for memory-constrained environments

**Performance Tips:**
.. code-block:: bash

   # Use month filtering for large datasets
   tsoc-analyze 2024-01 --data-dir "large_dataset"
   
   # Use summary-only for quick analysis
   tsoc-analyze 2024-01 --summary-only
   
   # Use verbose mode to monitor progress
   tsoc-analyze 2024-01 --verbose

Integration with Python API
--------------------------

The CLI can be used in combination with the Python API:

**CLI for Batch Processing:**
.. code-block:: bash

   # Process multiple months
   for month in 2024-01 2024-02 2024-03; do
       tsoc-analyze $month --output-dir "results_$month" --summary-only
   done

**Python API for Custom Analysis:**
.. code-block:: python

   from tsoc_data_analysis import execute, extract_representative_ops
   
   # Use CLI results in Python
   success, df = execute(month='2024-01', summary_only=True)
   if success:
       # Custom analysis
       rep_df, diagnostics = extract_representative_ops(
           df, max_power=850, MAPGL=200
       )

**Combined Workflow:**
.. code-block:: python

   # Load data using CLI
   success, df = execute(month='2024-01', data_dir='raw_data')
   
   if success:
       # Custom analysis using Python API
       from tsoc_data_analysis import calculate_total_load, calculate_net_load
       total_load = calculate_total_load(df)
       net_load = calculate_net_load(df, total_load)
       
       # Custom visualization
       import matplotlib.pyplot as plt
       plt.figure(figsize=(12, 8))
       plt.plot(total_load.index, total_load.values, label='Total Load')
       plt.plot(net_load.index, net_load.values, label='Net Load')
       plt.legend()
       plt.savefig('custom_analysis.png')

The CLI module is designed to be robust and user-friendly, providing a powerful command-line interface while maintaining full compatibility with the Python API for advanced users. 