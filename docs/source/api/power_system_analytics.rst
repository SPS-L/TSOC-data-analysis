Power System Analytics Module
============================

.. automodule:: tsoc_data_analysis.power_system_analytics
   :members:
   :undoc-members:
   :show-inheritance:

Module Overview
--------------

The ``power_system_analytics`` module provides functions for analyzing power system operational data, including load calculations, generator categorization, and statistical analysis.

The module works with merged DataFrames that contain data from multiple Excel files with standardized column naming conventions:

- ``ss_mw_*``: Substation active power (MW)
- ``ss_mvar_*``: Substation reactive power (MVAR)
- ``wind_mw_*``: Wind farm active power (MW)
- ``shunt_mvar_*``: Shunt element reactive power (MVAR)
- ``shunt_tap_*``: Shunt element tap position
- ``gen_v_*``: Generator voltage setpoints (KV)
- ``gen_mvar_*``: Generator reactive power (MVAR)

Core Functions
-------------

Load Calculations
~~~~~~~~~~~~~~~~

calculate_total_load()
^^^^^^^^^^^^^^^^^^^^^

Calculate total load by summing all substation active power columns.

.. code-block:: python

   def calculate_total_load(merged_df):
       """
       Calculate total load by summing all substation active power columns.
       
       Args:
           merged_df (pandas.DataFrame): Merged DataFrame containing all data
           
       Returns:
           pandas.Series: Time series of total load values
       """

**Example:**
.. code-block:: python

   from tsoc_data_analysis import calculate_total_load
   
   # Calculate total load from merged data
   total_load = calculate_total_load(merged_df)
   print(f"Total load range: {total_load.min():.2f} - {total_load.max():.2f} MW")

calculate_net_load()
^^^^^^^^^^^^^^^^^^^

Calculate net load by subtracting total wind generation from total load.

.. code-block:: python

   def calculate_net_load(merged_df, total_load=None):
       """
       Calculate net load by subtracting total wind generation from total load.
       
       Args:
           merged_df (pandas.DataFrame): Merged DataFrame containing all data
           total_load (pandas.Series, optional): Pre-calculated total load. If None, will calculate it.
           
       Returns:
           pandas.Series: Time series of net load values
       """

**Example:**
.. code-block:: python

   from tsoc_data_analysis import calculate_net_load
   
   # Calculate net load (total load minus wind generation)
   net_load = calculate_net_load(merged_df)
   print(f"Net load range: {net_load.min():.2f} - {net_load.max():.2f} MW")

get_load_statistics()
^^^^^^^^^^^^^^^^^^^^

Compute and return statistical measures of the total load.

.. code-block:: python

   def get_load_statistics(total_load):
       """
       Compute and return the max, min, and mean of the total load.
       
       Args:
           total_load (pandas.Series): Time series of total load values
           
       Returns:
           dict: Dictionary containing max, min, mean, and std load values
       """

**Returns:**
- ``max_load``: Maximum load value
- ``min_load``: Minimum load value
- ``mean_load``: Average load value
- ``std_load``: Standard deviation of load values

**Example:**
.. code-block:: python

   from tsoc_data_analysis import get_load_statistics
   
   # Get load statistics
   stats = get_load_statistics(total_load)
   print(f"Load Statistics:")
   print(f"  Maximum: {stats['max_load']:.2f} MW")
   print(f"  Minimum: {stats['min_load']:.2f} MW")
   print(f"  Average: {stats['mean_load']:.2f} MW")
   print(f"  Std Dev: {stats['std_load']:.2f} MW")

Generator Analysis
~~~~~~~~~~~~~~~~~

categorize_generators()
^^^^^^^^^^^^^^^^^^^^^^

Categorize generators into Voltage Control and PQ Control types based on available data.

.. code-block:: python

   def categorize_generators(merged_df):
       """
       Categorize generators into Voltage Control and PQ Control types.
       
       Args:
           merged_df (pandas.DataFrame): Merged DataFrame containing all data
           
       Returns:
           dict: Dictionary with 'voltage_control' and 'pq_control' generator lists
       """

**Returns:**
- ``voltage_control``: List of voltage control generator names
- ``pq_control``: List of PQ control generator names

**Example:**
.. code-block:: python

   from tsoc_data_analysis import categorize_generators
   
   # Categorize generators
   generator_categories = categorize_generators(merged_df)
   print(f"Voltage Control Generators: {len(generator_categories['voltage_control'])}")
   print(f"PQ Control Generators: {len(generator_categories['pq_control'])}")

Wind Power Analysis
~~~~~~~~~~~~~~~~~~

calculate_total_wind()
^^^^^^^^^^^^^^^^^^^^^

Calculate total wind generation by summing all wind farm active power columns.

.. code-block:: python

   def calculate_total_wind(merged_df):
       """
       Calculate total wind generation by summing all wind farm active power columns.
       
       Args:
           merged_df (pandas.DataFrame): Merged DataFrame containing all data
           
       Returns:
           pandas.Series: Time series of total wind generation values
       """

**Example:**
.. code-block:: python

   from tsoc_data_analysis import calculate_total_wind
   
   # Calculate total wind generation
   total_wind = calculate_total_wind(merged_df)
   print(f"Wind generation range: {total_wind.min():.2f} - {total_wind.max():.2f} MW")

Reactive Power Analysis
~~~~~~~~~~~~~~~~~~~~~~

calculate_total_reactive_power()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calculate total reactive power in the system.

.. code-block:: python

   def calculate_total_reactive_power(merged_df):
       """
       Calculate total reactive power in the system.
       
       Args:
           merged_df (pandas.DataFrame): Merged DataFrame containing all data
           
       Returns:
           pandas.Series: Time series of total reactive power values
       """

**Calculation Method:**
Total reactive power is calculated as:
- Substation reactive power (positive contribution)
- Shunt element reactive power (positive contribution)
- Generator reactive power (negative contribution)

**Example:**
.. code-block:: python

   from tsoc_data_analysis import calculate_total_reactive_power
   
   # Calculate total reactive power
   total_reactive = calculate_total_reactive_power(merged_df)
   print(f"Reactive power range: {total_reactive.min():.2f} - {total_reactive.max():.2f} MVAR")

Usage Examples
-------------

Complete Analysis Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from tsoc_data_analysis import (
       calculate_total_load,
       calculate_net_load,
       get_load_statistics,
       categorize_generators,
       calculate_total_wind,
       calculate_total_reactive_power
   )
   
   # Load analysis
   total_load = calculate_total_load(merged_df)
   net_load = calculate_net_load(merged_df, total_load)
   load_stats = get_load_statistics(total_load)
   
   # Generator analysis
   generator_categories = categorize_generators(merged_df)
   
   # Wind analysis
   total_wind = calculate_total_wind(merged_df)
   
   # Reactive power analysis
   total_reactive = calculate_total_reactive_power(merged_df)
   
   # Print comprehensive summary
   print("=== POWER SYSTEM ANALYSIS SUMMARY ===")
   print(f"Total Load: {load_stats['mean_load']:.2f} ± {load_stats['std_load']:.2f} MW")
   print(f"Net Load: {net_load.mean():.2f} ± {net_load.std():.2f} MW")
   print(f"Wind Generation: {total_wind.mean():.2f} ± {total_wind.std():.2f} MW")
   print(f"Reactive Power: {total_reactive.mean():.2f} ± {total_reactive.std():.2f} MVAR")
   print(f"Voltage Control Generators: {len(generator_categories['voltage_control'])}")
   print(f"PQ Control Generators: {len(generator_categories['pq_control'])}")

Data Requirements
----------------

Input DataFrame Structure
~~~~~~~~~~~~~~~~~~~~~~~~

The module expects a merged DataFrame with the following column naming conventions:

**Substation Data:**
- ``ss_mw_[substation_name]``: Active power consumption (MW)
- ``ss_mvar_[substation_name]``: Reactive power consumption (MVAR)

**Wind Farm Data:**
- ``wind_mw_[wind_farm_name]``: Active power generation (MW)

**Generator Data:**
- ``gen_v_[generator_name]``: Voltage setpoints (KV) - for voltage control generators
- ``gen_mvar_[generator_name]``: Reactive power (MVAR) - for PQ control generators

**Shunt Elements:**
- ``shunt_mvar_[shunt_name]``: Reactive power (MVAR)
- ``shunt_tap_[shunt_name]``: Tap positions (integer)

Data Quality Requirements
~~~~~~~~~~~~~~~~~~~~~~~~

- All power values should be numeric
- Timestamps should be in chronological order
- Missing values should be handled before analysis
- Column names should follow the specified naming conventions

Error Handling
-------------

The module includes comprehensive error handling:

**Missing Columns:**
- Functions will raise ``ValueError`` if required columns are not found
- Warnings are issued for missing optional data (e.g., wind generation)

**Data Type Issues:**
- Functions expect numeric data for power calculations
- Non-numeric data will cause errors during calculations

**Empty DataFrames:**
- Functions will raise appropriate errors for empty or invalid DataFrames

Performance Considerations
-------------------------

- Functions are optimized for large time series data
- Vectorized operations are used for efficient calculations
- Memory usage is minimized by avoiding unnecessary data copies
- Functions can handle DataFrames with thousands of time points efficiently

Integration with Other Modules
-----------------------------

This module is designed to work seamlessly with other modules in the package:

- **Data Loading**: Works with data from ``excel_data_processor``
- **Validation**: Compatible with validated data from ``power_data_validator``
- **Visualization**: Results can be plotted using ``power_system_visualizer``
- **Representative Points**: Analysis results can be used in ``operating_point_extractor`` 