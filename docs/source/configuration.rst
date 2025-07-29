Configuration Guide
==================

This guide provides detailed information about configuring the TSOC Data Analysis package for different power systems and use cases.

The TSOC Data Analysis package uses a centralized configuration system that allows users to customize all aspects of the analysis without modifying the source code. All configuration parameters are stored in `system_configuration.py` and organized into logical sections.

Configuration Areas
------------------

1. **Data File Mappings** - Excel file names and column prefixes
2. **Data Validation Settings** - Limits, thresholds, and validation rules
3. **Representative Operations Parameters** - Clustering and analysis settings
4. **Plotting and Visualization Settings** - Plot styles and formatting
5. **Shared Utility Functions** - Common utilities and helpers

System Configuration
-------------------

The main configuration file `system_configuration.py` contains all configurable parameters:

**File Structure:**

.. code-block:: python

   # Data directory and file mappings
   DATA_DIR = 'raw_data/'
   FILES = {...}
   COLUMN_PREFIXES = {...}
   
   # Data validation settings
   DATA_VALIDATION = {...}
   ENHANCED_DATA_VALIDATION = {...}
   
   # Representative operations settings
   REPRESENTATIVE_OPS = {...}
   
   # Plotting settings
   PLOT_STYLE = 'seaborn-v0_8'
   PLOT_PALETTE = 'husl'
   FIGURE_SIZES = {...}
   FONT_SIZES = {...}

Data File Configuration
----------------------

**File Mappings:**

Configure the Excel file names for your power system:

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

Define the column naming conventions:

.. code-block:: python

   COLUMN_PREFIXES = {
       'substation_mw': 'ss_mw_',
       'substation_mvar': 'ss_mvar_',
       'wind_power': 'wind_mw_',
       'shunt_elements': 'shunt_',
       'gen_voltage': 'gen_v_',
       'gen_mvar': 'gen_mvar_'
   }

**Customization Examples:**

.. code-block:: python

   # For different file naming conventions
   FILES['substation_mw'] = 'load_active_power.xlsx'
   FILES['wind_power'] = 'renewable_generation.xlsx'
   
   # For different column prefixes
   COLUMN_PREFIXES['substation_mw'] = 'load_mw_'
   COLUMN_PREFIXES['wind_power'] = 'renewable_mw_'

Data Validation Configuration
---------------------------

**Basic Validation Settings:**

.. code-block:: python

   DATA_VALIDATION = {
       'type_checks': {
           'real_numbers': ['ss_mw_', 'ss_mvar_', 'wind_mw_'],
           'integers': ['shunt_tap_']
       },
       'limit_checks': {
           'power_limits': {
               'wind': {'min_mw': 0, 'max_mw': 100},
               'substation': {'min_mw': -100, 'max_mw': 100}
           }
       },
       'gap_filling': {
           'max_gap_steps': 3,
           'advanced_max_gap_steps': 12,
           'remove_large_gaps_threshold': 24,
           'enable_advanced_gap_filling': True
       }
   }

**Enhanced Validation Settings:**

.. code-block:: python

   ENHANCED_DATA_VALIDATION = {
       'advanced_gap_filling': {
           'enable_advanced_gap_filling': True,
           'default_method': 'adaptive',
           'context_size_ratio': 0.25,
           'min_context_points': 10,
           'adaptive_thresholds': {
               'small_gap_size': 3,
               'medium_gap_size': 6,
               'large_gap_size': 12,
           }
       },
       'outlier_detection': {
           'default_methods': ['iqr', 'isolation_forest'],
           'contamination': 0.1,
           'zscore_threshold': 3.0,
           'modified_zscore_threshold': 3.5,
           'iqr_multiplier': 1.5,
       },
       'variable_groups': {
           'generators': ['gen_mvar_'],
           'substations': ['ss_mw_', 'ss_mvar_'],
           'wind': ['wind_mw_'],
           'shunts': ['shunt_mvar_', 'shunt_tap_'],
           'voltages': ['gen_v_']
       }
   }

**Customization Examples:**

.. code-block:: python

   # Adjust power limits for different systems
   DATA_VALIDATION['limit_checks']['power_limits']['wind']['max_mw'] = 200
   DATA_VALIDATION['limit_checks']['power_limits']['substation']['max_mw'] = 500
   
   # Enable more aggressive gap filling
   DATA_VALIDATION['gap_filling']['max_gap_steps'] = 6
   DATA_VALIDATION['gap_filling']['advanced_max_gap_steps'] = 24

Representative Operations Configuration
-------------------------------------

**Clustering Parameters:**

.. code-block:: python

   REPRESENTATIVE_OPS = {
       'defaults': {
           'k_max': 10,                    # Maximum clusters to test
           'random_state': 42,             # Reproducibility seed
           'mapgl_belt_multiplier': 1.1,   # MAPGL belt definition
           'fallback_clusters': 2          # Fallback if no quality clusters
       },
       'quality_thresholds': {
           'min_silhouette': 0.25,         # Minimum clustering quality
           'silhouette_excellent': 0.7,    # Excellent quality threshold
           'silhouette_good': 0.5,         # Good quality threshold
       },
       'ranking_weights': {
           'silhouette_weight': 1000,      # Multi-objective ranking weights
           'calinski_harabasz_weight': 1,
           'davies_bouldin_weight': 10
       },
       'output_files': {
           'representative_points': 'representative_operating_points.csv',
           'clustering_summary': 'clustering_summary.txt',
           'clustering_info': 'clustering_info.json'
       }
   }

**Customization Examples:**

.. code-block:: python

   # For larger power systems (more clusters needed)
   REPRESENTATIVE_OPS['defaults']['k_max'] = 20
   
   # For higher quality clustering requirements
   REPRESENTATIVE_OPS['quality_thresholds']['min_silhouette'] = 0.4
   
   # For different MAPGL belt definition
   REPRESENTATIVE_OPS['defaults']['mapgl_belt_multiplier'] = 1.15

Visualization Configuration
-------------------------

**Plot Style Settings:**

.. code-block:: python

   PLOT_STYLE = 'seaborn-v0_8'
   PLOT_PALETTE = 'husl'

**Figure Sizes:**

.. code-block:: python

   FIGURE_SIZES = {
       'timeseries': (12, 8),
       'daily_profile': (10, 6),
       'monthly_profile': (10, 6),
       'comprehensive': (15, 10),
       'clustering': (16, 12)
   }

**Font Sizes:**

.. code-block:: python

   FONT_SIZES = {
       'title': 16,
       'axis_label': 14,
       'tick_label': 12,
       'legend': 12,
       'annotation': 10
   }

**Customization Examples:**

.. code-block:: python

   # For different plot styles
   PLOT_STYLE = 'default'
   PLOT_PALETTE = 'viridis'
   
   # For larger plots
   FIGURE_SIZES['comprehensive'] = (20, 15)
   FIGURE_SIZES['clustering'] = (24, 18)
   
   # For different font sizes
   FONT_SIZES['title'] = 18
   FONT_SIZES['axis_label'] = 16