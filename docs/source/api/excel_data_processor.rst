Excel Data Processor Module
===========================

.. automodule:: tsoc_data_analysis.excel_data_processor
   :members:
   :undoc-members:
   :show-inheritance:

Module Overview
--------------

The ``excel_data_processor`` module provides functions for loading and preprocessing Excel files containing power system operational data. It handles the complex structure of Excel files with timestamps, variable names, and data organized in specific formats.

The module is designed to work with Excel files that follow a standardized structure for power system data, including substation data, generator data, wind farm data, and shunt element data.

Core Functions
-------------

load_single_excel_file()
~~~~~~~~~~~~~~~~~~~~~~~

Load a single Excel file with power system data.

.. code-block:: python

   def load_single_excel_file(file_path: str, file_type: str) -> pd.DataFrame:
       """
       Load a single Excel file with power system data.
       
       Args:
           file_path: Path to the Excel file
           file_type: Type of data file (e.g., 'substation_mw', 'wind_power')
           
       Returns:
           DataFrame with loaded data
       """

**Example:**
.. code-block:: python

   from tsoc_data_analysis.excel_data_processor import load_single_excel_file
   
   # Load substation active power data
   substation_df = load_single_excel_file(
       'raw_data/substation_active_power.xlsx', 
       'substation_mw'
   )
   print(f"Loaded {len(substation_df)} records from substation data")

load_all_excel_files()
~~~~~~~~~~~~~~~~~~~~~~

Load all Excel files from a directory and merge them into a single DataFrame.

.. code-block:: python

   def load_all_excel_files(data_dir: str) -> pd.DataFrame:
       """
       Load all Excel files from directory and merge into single DataFrame.
       
       Args:
           data_dir: Directory containing Excel files
           
       Returns:
           Merged DataFrame with all data
       """

**Example:**
.. code-block:: python

   from tsoc_data_analysis.excel_data_processor import load_all_excel_files
   
   # Load all data from directory
   merged_df = load_all_excel_files('raw_data')
   print(f"Loaded {len(merged_df)} records with {len(merged_df.columns)} columns")

Excel File Structure
-------------------

The module expects Excel files to follow a specific structure:

**File Organization:**
- **Timestamps**: Located in column C (0-indexed column 2)
- **Data Start Row**: Row 6 (0-indexed row 5)
- **Variable Names**: Located in specific rows depending on file type
- **Data Columns**: Start from column G (0-indexed column 6) with 2-column spacing

**File Types and Structures:**

Substation Data
~~~~~~~~~~~~~~

**Active Power (MW)**: `substation_active_power.xlsx`
- **Variable Names**: Row 2 (0-indexed row 1)
- **Column Prefix**: `ss_mw_`
- **Data Type**: Active power consumption (MW)

**Reactive Power (MVAR)**: `substation_reactive_power.xlsx`
- **Variable Names**: Row 2 (0-indexed row 1)
- **Column Prefix**: `ss_mvar_`
- **Data Type**: Reactive power consumption (MVAR)

Generator Data
~~~~~~~~~~~~~

**Voltage Setpoints (KV)**: `generator_voltage_setpoints.xlsx`
- **Variable Names**: Row 3 (0-indexed row 2)
- **Column Prefix**: `gen_v_`
- **Data Type**: Voltage setpoints (KV)

**Reactive Power (MVAR)**: `generator_reactive_power.xlsx`
- **Variable Names**: Row 3 (0-indexed row 2)
- **Column Prefix**: `gen_mvar_`
- **Data Type**: Reactive power (MVAR)

Wind Farm Data
~~~~~~~~~~~~~

**Active Power (MW)**: `wind_farm_active_power.xlsx`
- **Variable Names**: Row 3 (0-indexed row 2)
- **Column Prefix**: `wind_mw_`
- **Data Type**: Active power generation (MW)

Shunt Elements
~~~~~~~~~~~~~

**Reactive Power (MVAR)**: `shunt_element_reactive_power.xlsx`
- **Variable Names**: Row 3 (0-indexed row 2)
- **Column Prefixes**: `shunt_mvar_` and `shunt_tap_`
- **Data Types**: Reactive power (MVAR) and tap positions (integer)

Data Processing Steps
--------------------

The module performs several processing steps when loading Excel files:

**1. File Structure Detection**
- Identifies timestamp column and data start row
- Locates variable names based on file type
- Determines data column range

**2. Data Extraction**
- Extracts timestamps from the specified column
- Reads variable names from the appropriate row
- Extracts data values from the data range

**3. Column Naming**
- Applies appropriate prefixes based on file type
- Creates standardized column names
- Handles special cases for different data types

**4. Data Validation**
- Checks for missing or invalid timestamps
- Validates data types and ranges
- Handles empty or corrupted data

**5. Merging**
- Combines data from multiple files
- Aligns timestamps across all data sources
- Handles missing data gracefully

Configuration Integration
------------------------

The module uses configuration from `system_configuration.py`:

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

**Column Prefixes:**
.. code-block:: python

   COLUMN_PREFIXES = {
       'substation_mw': 'ss_mw_',
       'substation_mvar': 'ss_mvar_',
       'wind_power': 'wind_mw_',
       'shunt_elements': 'shunt_',
       'gen_voltage': 'gen_v_',
       'gen_mvar': 'gen_mvar_'
   }

**Excel Structure Constants:**
.. code-block:: python

   TIMESTAMP_COLUMN = 2      # Column C (0-indexed)
   DATA_START_ROW = 5        # Row 6 (0-indexed)
   SUBSTATION_NAME_ROW = 1   # Row 2 (0-indexed)
   GENERATOR_NAME_ROW = 2    # Row 3 (0-indexed)
   DATA_COLUMN_START = 6     # Column G (0-indexed)
   DATA_COLUMN_STEP = 2      # Skip every other column

Usage Examples
-------------

Basic File Loading
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from tsoc_data_analysis.excel_data_processor import load_single_excel_file
   
   # Load individual file types
   substation_mw = load_single_excel_file(
       'raw_data/substation_active_power.xlsx', 
       'substation_mw'
   )
   wind_power = load_single_excel_file(
       'raw_data/wind_farm_active_power.xlsx', 
       'wind_power'
   )
   
   print(f"Substation data: {len(substation_mw)} records")
   print(f"Wind data: {len(wind_power)} records")

Complete Data Loading
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from tsoc_data_analysis.excel_data_processor import load_all_excel_files
   
   # Load all data at once
   merged_df = load_all_excel_files('raw_data')
   
   print(f"Complete dataset: {len(merged_df)} records")
   print(f"Variables: {len(merged_df.columns)}")
   print(f"Time range: {merged_df.index.min()} to {merged_df.index.max()}")

Integration with Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from tsoc_data_analysis.excel_data_processor import load_all_excel_files
   from tsoc_data_analysis import calculate_total_load, calculate_net_load
   
   # Load data
   df = load_all_excel_files('raw_data')
   
   # Perform analysis
   total_load = calculate_total_load(df)
   net_load = calculate_net_load(df, total_load)
   
   print(f"Average total load: {total_load.mean():.2f} MW")
   print(f"Average net load: {net_load.mean():.2f} MW")

Error Handling
-------------

**File Not Found:**
- Graceful handling of missing files
- Informative error messages
- Continues processing with available files

**Data Structure Issues:**
- Validates Excel file structure
- Handles variations in data organization
- Provides detailed error reporting

**Data Quality Issues:**
- Handles missing or corrupted data
- Validates data types and ranges
- Continues processing with valid data

**Memory Management:**
- Efficient memory usage for large files
- Handles out-of-memory conditions
- Provides progress information for large datasets

Performance Considerations
-------------------------

**Optimization Features:**
- **Selective Loading**: Loads only required columns
- **Memory Efficiency**: Processes data in chunks for large files
- **Parallel Processing**: Can process multiple files in parallel
- **Caching**: Caches processed data for repeated access

**Scalability:**
- **Large Files**: Handles Excel files with thousands of rows
- **Many Variables**: Efficiently processes hundreds of columns
- **Multiple Files**: Scales to process dozens of Excel files
- **Memory Management**: Optimized for memory-constrained environments

Integration with Other Modules
-----------------------------

This module integrates seamlessly with other package modules:

- **Data Validation**: Output can be validated using `power_data_validator`
- **Analysis**: Loaded data can be analyzed using `power_system_analytics`
- **Representative Points**: Data can be used for clustering in `operating_point_extractor`
- **Visualization**: Results can be plotted using `power_system_visualizer`
- **Configuration**: Uses centralized configuration from `system_configuration`

The module is designed to be robust and efficient, providing reliable data loading capabilities while maintaining high performance and user-friendly error handling. 