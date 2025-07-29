Power System Visualizer Module
==============================

.. automodule:: tsoc_data_analysis.power_system_visualizer
   :members:
   :undoc-members:
   :show-inheritance:

Module Overview
--------------

The ``power_system_visualizer`` module provides comprehensive visualization capabilities for power system operational data. It includes functions for creating time series plots, daily and monthly profiles, and comprehensive analysis dashboards.

The module is designed to work with the analysis results from other modules and provides publication-quality visualizations with consistent styling and formatting.

Core Functions
-------------

Time Series Visualization
~~~~~~~~~~~~~~~~~~~~~~~~

plot_load_timeseries()
^^^^^^^^^^^^^^^^^^^^^

Create time series plots for total and net load.

.. code-block:: python

   def plot_load_timeseries(total_load, net_load, save_path=None):
       """
       Create time series plots for total and net load.
       
       Args:
           total_load: Time series of total load values
           net_load: Time series of net load values
           save_path: Optional path to save the plot
       """

**Example:**
.. code-block:: python

   from tsoc_data_analysis import plot_load_timeseries, calculate_total_load, calculate_net_load
   
   # Calculate loads
   total_load = calculate_total_load(df)
   net_load = calculate_net_load(df, total_load)
   
   # Create time series plot
   plot_load_timeseries(total_load, net_load, save_path='load_timeseries.png')

plot_monthly_profile()
^^^^^^^^^^^^^^^^^^^^^

Create monthly load profile plots.

.. code-block:: python

   def plot_monthly_profile(total_load, net_load, save_path=None):
       """
       Create monthly load profile plots.
       
       Args:
           total_load: Time series of total load values
           net_load: Time series of net load values
           save_path: Optional path to save the plot
       """

Daily Profile Functions
~~~~~~~~~~~~~~~~~~~~~~

plot_total_load_daily_profile()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create average daily load profile for total load.

.. code-block:: python

   def plot_total_load_daily_profile(total_load, save_path=None):
       """
       Create average daily load profile for total load.
       
       Args:
           total_load: Time series of total load values
           save_path: Optional path to save the plot
       """

plot_net_load_daily_profile()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create average daily load profile for net load.

.. code-block:: python

   def plot_net_load_daily_profile(net_load, save_path=None):
       """
       Create average daily load profile for net load.
       
       Args:
           net_load: Time series of net load values
           save_path: Optional path to save the plot
       """

Comprehensive Analysis
~~~~~~~~~~~~~~~~~~~~~

create_comprehensive_plots()
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create comprehensive analysis plots including multiple visualizations.

.. code-block:: python

   def create_comprehensive_plots(df, total_load, net_load, save_path=None):
       """
       Create comprehensive analysis plots.
       
       Args:
           df: DataFrame with power system data
           total_load: Time series of total load values
           net_load: Time series of net load values
           save_path: Optional path to save the plot
       """

**Example:**
.. code-block:: python

   from tsoc_data_analysis import create_comprehensive_plots
   
   # Create comprehensive analysis
   create_comprehensive_plots(df, total_load, net_load, save_path='comprehensive_analysis.png')

Visualization Features
---------------------

Plot Styling
~~~~~~~~~~~

The module uses consistent styling with configurable parameters:

**Style Settings:**
.. code-block:: python

   PLOT_STYLE = 'seaborn-v0_8'
   PLOT_PALETTE = 'husl'
   
   FIGURE_SIZES = {
       'timeseries': (12, 8),
       'daily_profile': (10, 6),
       'monthly_profile': (10, 6),
       'comprehensive': (15, 10),
       'clustering': (16, 12)
   }
   
   FONT_SIZES = {
       'title': 16,
       'axis_label': 14,
       'tick_label': 12,
       'legend': 12,
       'annotation': 10
   }

**Customization:**
Users can customize plot appearance by modifying configuration values:

.. code-block:: python

   # Customize plot styling
   import matplotlib.pyplot as plt
   plt.style.use('seaborn-v0_8')
   plt.rcParams['figure.figsize'] = (12, 8)
   plt.rcParams['font.size'] = 12

Plot Types
----------

Time Series Plots
~~~~~~~~~~~~~~~~

**Features:**
- Dual-axis plots for total and net load
- Automatic date formatting
- Grid lines for readability
- Legend with clear labels
- Configurable color schemes

**Example Output:**
- Shows load variations over time
- Highlights peak and minimum loads
- Displays load patterns and trends

Daily Profile Plots
~~~~~~~~~~~~~~~~~~

**Features:**
- 24-hour average profiles
- Error bars showing variability
- Hour-of-day x-axis
- Multiple load types on same plot
- Statistical summaries

**Example Output:**
- Shows typical daily load patterns
- Highlights peak hours
- Displays load variability

Monthly Profile Plots
~~~~~~~~~~~~~~~~~~~~

**Features:**
- Monthly average patterns
- Seasonal variations
- Multiple years comparison
- Statistical analysis
- Trend identification

**Example Output:**
- Shows seasonal load patterns
- Highlights monthly variations
- Displays long-term trends

Comprehensive Dashboards
~~~~~~~~~~~~~~~~~~~~~~~

**Features:**
- Multi-panel layouts
- Multiple visualization types
- Statistical summaries
- Interactive elements (if supported)
- Publication-quality output

**Example Output:**
- Complete system overview
- Multiple analysis perspectives
- Professional presentation quality

Configuration Integration
------------------------

The module uses configuration from `system_configuration.py`:

**Plot Configuration:**
.. code-block:: python

   # Plotting settings
   PLOT_STYLE = 'seaborn-v0_8'
   PLOT_PALETTE = 'husl'
   
   # Figure sizes for different plot types
   FIGURE_SIZES = {
       'timeseries': (12, 8),
       'daily_profile': (10, 6),
       'monthly_profile': (10, 6),
       'comprehensive': (15, 10),
       'clustering': (16, 12)
   }
   
   # Font sizes for different elements
   FONT_SIZES = {
       'title': 16,
       'axis_label': 14,
       'tick_label': 12,
       'legend': 12,
       'annotation': 10
   }

**Customization Examples:**
.. code-block:: python

   # For larger plots
   FIGURE_SIZES['timeseries'] = (16, 10)
   
   # For different color schemes
   PLOT_PALETTE = 'viridis'
   
   # For larger fonts
   FONT_SIZES['title'] = 18
   FONT_SIZES['axis_label'] = 16

Usage Examples
-------------

Basic Time Series Plotting
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from tsoc_data_analysis import (
       plot_load_timeseries,
       calculate_total_load,
       calculate_net_load
   )
   
   # Calculate loads
   total_load = calculate_total_load(df)
   net_load = calculate_net_load(df, total_load)
   
   # Create time series plot
   plot_load_timeseries(total_load, net_load, save_path='load_timeseries.png')

Daily Profile Analysis
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from tsoc_data_analysis import (
       plot_total_load_daily_profile,
       plot_net_load_daily_profile
   )
   
   # Create daily profiles
   plot_total_load_daily_profile(total_load, save_path='total_load_daily.png')
   plot_net_load_daily_profile(net_load, save_path='net_load_daily.png')

Comprehensive Analysis
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from tsoc_data_analysis import create_comprehensive_plots
   
   # Create comprehensive analysis
   create_comprehensive_plots(
       df, total_load, net_load, 
       save_path='comprehensive_analysis.png'
   )

Custom Plotting
~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   import seaborn as sns
   
   # Custom styling
   plt.style.use('seaborn-v0_8')
   sns.set_palette('husl')
   
   # Create custom plot
   fig, ax = plt.subplots(figsize=(12, 8))
   ax.plot(total_load.index, total_load.values, label='Total Load')
   ax.plot(net_load.index, net_load.values, label='Net Load')
   ax.set_xlabel('Time')
   ax.set_ylabel('Load (MW)')
   ax.set_title('Power System Load Analysis')
   ax.legend()
   ax.grid(True)
   plt.tight_layout()
   plt.savefig('custom_plot.png', dpi=300, bbox_inches='tight')
   plt.show()

Integration with Analysis Pipeline
--------------------------------

The visualizer integrates seamlessly with the analysis workflow:

.. code-block:: python

   from tsoc_data_analysis import (
       execute,
       plot_load_timeseries,
       create_comprehensive_plots
   )
   
   # Step 1: Perform analysis
   success, df = execute(month='2024-01', data_dir='raw_data', output_dir='results')
   
   if success:
       # Step 2: Calculate loads
       total_load = calculate_total_load(df)
       net_load = calculate_net_load(df, total_load)
       
       # Step 3: Create visualizations
       plot_load_timeseries(total_load, net_load, save_path='results/load_timeseries.png')
       create_comprehensive_plots(df, total_load, net_load, save_path='results/comprehensive.png')

Output Formats
-------------

**Supported Formats:**
- **PNG**: High-quality raster images
- **PDF**: Vector graphics for publications
- **SVG**: Scalable vector graphics
- **JPG**: Compressed images for web

**Quality Settings:**
- **DPI**: Configurable resolution (default: 300 DPI)
- **Compression**: Optimized file sizes
- **Transparency**: Support for transparent backgrounds
- **Metadata**: Embedded plot information

**Example:**
.. code-block:: python

   # High-quality output
   plt.savefig('plot.png', dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
   
   # Vector output for publications
   plt.savefig('plot.pdf', bbox_inches='tight', 
               facecolor='white', edgecolor='none')

Performance Considerations
-------------------------

**Optimization Features:**
- **Efficient Plotting**: Optimized for large datasets
- **Memory Management**: Efficient handling of large time series
- **Batch Processing**: Can create multiple plots efficiently
- **Caching**: Caches processed data for repeated plotting

**Scalability:**
- **Large Datasets**: Handles time series with thousands of points
- **Multiple Plots**: Efficiently creates multiple visualizations
- **High Resolution**: Supports high-DPI output without memory issues
- **Batch Operations**: Can process multiple datasets simultaneously

Error Handling
-------------

**Comprehensive Error Handling:**
- **Data Validation**: Checks for valid input data
- **Missing Data**: Handles missing or invalid values gracefully
- **File Operations**: Graceful handling of file I/O errors
- **Memory Issues**: Handles out-of-memory conditions

**User-Friendly Messages:**
- **Clear Error Messages**: Informative error descriptions
- **Suggestions**: Provides helpful suggestions for fixing issues
- **Fallback Options**: Offers alternative approaches when possible

Integration with Other Modules
-----------------------------

This module integrates seamlessly with other package modules:

- **Data Loading**: Works with data from `excel_data_processor`
- **Analysis**: Visualizes results from `power_system_analytics`
- **Representative Points**: Can visualize clustering results from `operating_point_extractor`
- **Validation**: Can show validation results from `power_data_validator`
- **Configuration**: Uses centralized configuration from `system_configuration`

The module is designed to be flexible and user-friendly, providing high-quality visualizations while maintaining good performance and comprehensive error handling. 