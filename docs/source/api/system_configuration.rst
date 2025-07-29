System Configuration Module
===========================

.. automodule:: tsoc_data_analysis.system_configuration
   :members:
   :undoc-members:
   :show-inheritance:

Configuration Overview
---------------------

The ``system_configuration`` module serves as the central configuration hub for the entire power system analysis toolkit. It provides a single source of truth for all configurable parameters, file mappings, validation settings, and shared utilities.

Configuration Areas
------------------

1. **Data File Mappings** (``FILES``, ``COLUMN_PREFIXES``)
2. **Data Validation Settings** (``DATA_VALIDATION``, ``ENHANCED_DATA_VALIDATION``)
3. **Representative Operations Parameters** (``REPRESENTATIVE_OPS``)
4. **Plotting and Visualization Settings** (``PLOT_STYLE``, ``FIGURE_SIZES``)
5. **Shared Utility Functions** (``clean_column_name``, ``convert_numpy_types``)

Data File Configuration
----------------------

File Mappings
~~~~~~~~~~~~

.. code-block:: python

   FILES = {
       'substation_mw': 'substation_active_power.xlsx',
       'substation_mvar': 'substation_reactive_power.xlsx', 
       'wind_power': 'wind_farm_active_power.xlsx',
       'shunt_elements': 'shunt_element_reactive_power.xlsx',
       'gen_voltage': 'generator_voltage_setpoints.xlsx',
       'gen_mvar': 'generator_reactive_power.xlsx'
   }

Column Prefixes
~~~~~~~~~~~~~~

.. code-block:: python

   COLUMN_PREFIXES = {
       'substation_mw': 'ss_mw_',
       'substation_mvar': 'ss_mvar_',
       'wind_power': 'wind_mw_',
       'shunt_elements': 'shunt_',
       'gen_voltage': 'gen_v_',
       'gen_mvar': 'gen_mvar_'
   }

Data Type Descriptions
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   DATA_TYPE_DESCRIPTIONS = {
       'substation_mw': 'Substation active power data (MW)',
       'substation_mvar': 'Substation reactive power data (MVAR)',
       'wind_power': 'Wind farm active power data (MW)',
       'shunt_elements': 'Shunt element reactive power data (MVAR)',
       'gen_voltage': 'Generator voltage setpoints data (KV)',
       'gen_mvar': 'Generator reactive power data (MVAR)'
   }

Excel File Structure Constants
-----------------------------

.. code-block:: python

   TIMESTAMP_COLUMN = 2      # Column C (0-indexed)
   DATA_START_ROW = 5        # Row 6 (0-indexed)
   SUBSTATION_NAME_ROW = 1   # Row 2 (0-indexed)
   GENERATOR_NAME_ROW = 2    # Row 3 (0-indexed)
   DATA_COLUMN_START = 6     # Column G (0-indexed)
   DATA_COLUMN_STEP = 2      # Skip every other column

Data Validation Configuration
---------------------------

Basic Validation Settings
~~~~~~~~~~~~~~~~~~~~~~~~

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

Enhanced Validation Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Representative Operations Configuration
-------------------------------------

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

Plotting Configuration
---------------------

Style Settings
~~~~~~~~~~~~~

.. code-block:: python

   PLOT_STYLE = 'seaborn-v0_8'
   PLOT_PALETTE = 'husl'

Figure Sizes
~~~~~~~~~~~

.. code-block:: python

   FIGURE_SIZES = {
       'timeseries': (12, 8),
       'daily_profile': (10, 6),
       'monthly_profile': (10, 6),
       'comprehensive': (15, 10),
       'clustering': (16, 12)
   }

Font Sizes
~~~~~~~~~~

.. code-block:: python

   FONT_SIZES = {
       'title': 16,
       'axis_label': 14,
       'tick_label': 12,
       'legend': 12,
       'annotation': 10
   }

Utility Functions
----------------

Column Name Cleaning
~~~~~~~~~~~~~~~~~~~

The ``clean_column_name`` function removes verbose suffixes from column names to create cleaner, more readable names.

.. code-block:: python

   def clean_column_name(col_name):
       """
       Clean column names by removing verbose suffixes.
       
       Args:
           col_name (str): Original column name
           
       Returns:
           str: Cleaned column name
       """
       
   # Example usage:
   clean_name = clean_column_name('ss_mw_STATION1_132REACTOR_REACTIVE_POWER')
   # Result: 'ss_mw_STATION1'

NumPy Type Conversion
~~~~~~~~~~~~~~~~~~~~

The ``convert_numpy_types`` function converts NumPy types to native Python types for JSON serialization.

.. code-block:: python

   def convert_numpy_types(obj):
       """
       Convert NumPy types to native Python types for JSON serialization.
       
       Args:
           obj: Object containing NumPy types
           
       Returns:
           Object with native Python types
       """
       
   # Example usage:
   import numpy as np
   data = {'value': np.int64(42), 'array': np.array([1, 2, 3])}
   json_safe_data = convert_numpy_types(data)
   # Result: {'value': 42, 'array': [1, 2, 3]}

Usage Examples
-------------

Basic Configuration Access
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from tsoc_data_analysis.system_configuration import FILES, REPRESENTATIVE_OPS
   
   # Use default clustering parameters
   k_max = REPRESENTATIVE_OPS['defaults']['k_max']
   random_state = REPRESENTATIVE_OPS['defaults']['random_state']
   
   # Access validation settings
   power_limits = DATA_VALIDATION['limit_checks']['power_limits']

Utility Functions
~~~~~~~~~~~~~~~~

.. code-block:: python

   from tsoc_data_analysis.system_configuration import clean_column_name, convert_numpy_types
   
   # Clean column names
   clean_name = clean_column_name('ss_mw_STATION1_132REACTOR_REACTIVE_POWER')
   
   # Convert numpy types for JSON serialization
   import numpy as np
   data = {'value': np.int64(42), 'array': np.array([1, 2, 3])}
   json_safe_data = convert_numpy_types(data)

Configuration Customization
--------------------------

Users can customize the configuration by modifying the values in ``system_configuration.py``:

Representative Operations
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For larger power systems (more clusters needed)
   REPRESENTATIVE_OPS['defaults']['k_max'] = 20
   
   # For higher quality clustering requirements
   REPRESENTATIVE_OPS['quality_thresholds']['min_silhouette'] = 0.4
   
   # For different MAPGL belt definition
   REPRESENTATIVE_OPS['defaults']['mapgl_belt_multiplier'] = 1.15

Data Validation
~~~~~~~~~~~~~~

.. code-block:: python

   # For different power system limits
   DATA_VALIDATION['limit_checks']['power_limits']['wind']['max_mw'] = 200
   DATA_VALIDATION['limit_checks']['power_limits']['substation']['max_mw'] = 500
   
   # For more aggressive gap filling
   DATA_VALIDATION['gap_filling']['max_gap_steps'] = 6
   DATA_VALIDATION['gap_filling']['advanced_max_gap_steps'] = 24

Output Files
~~~~~~~~~~~~

.. code-block:: python

   # Custom output file names
   REPRESENTATIVE_OPS['output_files'] = {
       'representative_points': 'selected_operating_points.csv',
       'clustering_summary': 'clustering_analysis.txt',
       'clustering_info': 'clustering_data.json'
   } 