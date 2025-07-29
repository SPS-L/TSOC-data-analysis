Configuration Guide
==================

This guide provides detailed information about configuring the TSOC Data Analysis package for different power systems and use cases.

Overview
--------

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

Configuration Best Practices
---------------------------

**General Guidelines:**

1. **Backup Original Configuration**: Always keep a backup of the original configuration before making changes.

2. **Test Incrementally**: Make small changes and test them before proceeding with larger modifications.

3. **Document Changes**: Keep a record of configuration changes for future reference.

4. **Validate Settings**: Ensure that all configuration values are within reasonable ranges.

**Power System Specific Guidelines:**

1. **File Naming**: Use descriptive file names that clearly indicate the data type and source.

2. **Column Prefixes**: Choose prefixes that are unique and meaningful for your system.

3. **Validation Limits**: Set limits based on your power system's actual operating ranges.

4. **Clustering Parameters**: Adjust clustering parameters based on the size and complexity of your system.

**Performance Guidelines:**

1. **Gap Filling**: Balance between data quality and processing time when setting gap filling parameters.

2. **Clustering**: Use appropriate `k_max` values to avoid excessive computation time.

3. **Memory Usage**: Consider memory constraints when processing large datasets.

Configuration Validation
-----------------------

**Validation Functions:**

The package includes built-in validation for configuration settings:

.. code-block:: python

   from tsoc_data_analysis.system_configuration import validate_configuration
   
   # Validate all configuration settings
   errors = validate_configuration()
   if errors:
       print("Configuration errors found:")
       for error in errors:
           print(f"  - {error}")
   else:
       print("Configuration is valid")

**Common Validation Checks:**

1. **File Existence**: Verify that all referenced Excel files exist in the data directory.

2. **Column Consistency**: Ensure that column prefixes are consistent across all files.

3. **Limit Validation**: Check that all limits are within reasonable ranges.

4. **Parameter Types**: Verify that all parameters have the correct data types.

Configuration Migration
----------------------

**Upgrading Configuration:**

When upgrading to new versions, you may need to update your configuration:

.. code-block:: python

   # Example: Updating from version 1.0 to 1.1
   # Old configuration
   OLD_CONFIG = {
       'max_clusters': 10,
       'quality_threshold': 0.25
   }
   
   # New configuration
   NEW_CONFIG = {
       'defaults': {
           'k_max': 10,  # renamed from max_clusters
           'fallback_clusters': 2  # new parameter
       },
       'quality_thresholds': {
           'min_silhouette': 0.25  # moved to nested structure
       }
   }

**Backward Compatibility:**

The package maintains backward compatibility where possible, but some changes may require configuration updates.

Advanced Configuration
---------------------

**Custom Validation Rules:**

You can define custom validation rules for your specific power system:

.. code-block:: python

   # Custom validation function
   def custom_validation(data):
       """Custom validation for specific power system requirements."""
       errors = []
       
       # Check for specific substation requirements
       if 'ss_mw_CRITICAL_SUBSTATION' not in data.columns:
           errors.append("Critical substation data missing")
       
       # Check for minimum data quality
       if data.isnull().sum().sum() > len(data) * 0.1:
           errors.append("Too much missing data")
       
       return errors

**Dynamic Configuration:**

For advanced users, you can implement dynamic configuration loading:

.. code-block:: python

   import json
   from tsoc_data_analysis.system_configuration import load_configuration
   
   # Load configuration from JSON file
   def load_custom_config(config_file):
       with open(config_file, 'r') as f:
           config_data = json.load(f)
       
       # Apply configuration
       load_configuration(config_data)
       
       return True

**Environment-Specific Configuration:**

You can create different configurations for different environments:

.. code-block:: python

   import os
   
   # Load environment-specific configuration
   environment = os.getenv('TSOC_ENV', 'development')
   
   if environment == 'production':
       # Production settings
       REPRESENTATIVE_OPS['defaults']['k_max'] = 15
       DATA_VALIDATION['gap_filling']['max_gap_steps'] = 2
   elif environment == 'development':
       # Development settings
       REPRESENTATIVE_OPS['defaults']['k_max'] = 5
       DATA_VALIDATION['gap_filling']['max_gap_steps'] = 6 