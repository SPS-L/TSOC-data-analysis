Power Data Validator Module
===========================

.. automodule:: tsoc_data_analysis.power_data_validator
   :members:
   :undoc-members:
   :show-inheritance:

Module Overview
===============

The ``power_data_validator`` module provides comprehensive data quality assurance for power system operational data. It implements a sophisticated three-step validation workflow that includes advanced gap filling techniques and anomaly detection capabilities.

The module implements a complete power systems data validation pipeline with intelligent gap processing and enhanced anomaly detection.

Validation Workflow
===================

The enhanced data validation system provides a comprehensive three-step validation workflow:

**Step 1: Simple Validation**
- Basic gap filling (≤3 time steps using linear interpolation)
- Data type validation and correction
- Limit checking and constraint enforcement
- Shunt tap integer conversion

**Step 2: Comprehensive Validation** *(if enabled)*
- Advanced outlier detection using multiple statistical and ML methods
- Rate of change violation detection with gap-aware sensitivity
- Correlation anomaly identification between related variables
- Power balance validation for energy conservation
- Clustering-based anomaly detection for operational patterns

**Step 3: Advanced Gap Filling** *(if enabled)*
- **Intelligent method selection** based on gap characteristics and data patterns
- **Multiple sophisticated techniques**: spline, polynomial, KNN, ML-based imputation
- **Large gap removal**: gaps ≥24 time steps completely removed
- **Adaptive algorithms**: automatically choose optimal method per gap

Core Classes
============

DataValidator
~~~~~~~~~~~~~

Main validation class that orchestrates the complete validation workflow.

.. code-block:: python

   class DataValidator:
       """
       Comprehensive data validator for power system operational data.
       
       Implements a three-step validation workflow with advanced gap filling
       and anomaly detection capabilities.
       """

**Key Methods:**

validate_data()
^^^^^^^^^^^^^^^

Main validation method that performs the complete validation workflow.

.. code-block:: python

   def validate_data(self, df: pd.DataFrame, enable_enhanced: bool = True) -> pd.DataFrame:
       """
       Perform comprehensive data validation and cleaning.
       
       Args:
           df: Input DataFrame with power system data
           enable_enhanced: Whether to enable enhanced validation features
           
       Returns:
           Cleaned and validated DataFrame
       """

**Example:**
.. code-block:: python

   from tsoc_data_analysis import DataValidator
   
   # Create validator instance
   validator = DataValidator()
   
   # Perform validation
   clean_df = validator.validate_data(df, enable_enhanced=True)
   print(f"Validation completed. Cleaned {len(clean_df)} records.")

get_validation_summary()
^^^^^^^^^^^^^^^^^^^^^^^^

Get a comprehensive summary of validation results.

.. code-block:: python

   def get_validation_summary(self) -> dict:
       """
       Get comprehensive validation summary.
       
       Returns:
           Dictionary with validation statistics and error details
       """

**Example:**
.. code-block:: python

   # Get validation summary
   summary = validator.get_validation_summary()
   print(f"Records processed: {summary['total_records']}")
   print(f"Records with errors: {summary['records_with_errors']}")
   print(f"Gaps filled: {summary['gaps_filled']}")

Advanced Gap Filling
====================

The module provides sophisticated gap filling capabilities with multiple algorithms:

**Intelligent Method Selection:**
- **Small gaps (≤3 steps)**: Linear interpolation for simple continuity
- **Medium gaps (4-6 steps)**: Cubic spline interpolation for smooth curves  
- **Large gaps (7-12 steps)**: KNN or polynomial based on data variance and trends
- **Very large gaps (≥24 steps)**: Complete removal to maintain data integrity

**Available Algorithms:**
- **`'adaptive'`**: Automatically selects best method based on gap size and data characteristics
- **`'spline'`**: Cubic spline interpolation with configurable smoothing
- **`'polynomial'`**: Polynomial interpolation for trend-following behavior
- **`'knn'`**: K-Nearest Neighbors using temporal features (hour, day, seasonality)
- **`'ml'`**: Random Forest-based with lagged features and cross-variable relationships

**Example:**
.. code-block:: python

   # Configure advanced gap filling
   validator = DataValidator()
   validator.set_gap_filling_method('adaptive')
   
   # Perform validation with advanced gap filling
   clean_df = validator.validate_data(df, enable_enhanced=True)

Anomaly Detection
=================

The module implements multiple anomaly detection methods:

**Statistical Outlier Detection**
- **IQR (Interquartile Range)**: Detects outliers using configurable IQR multiplier
- **Z-Score**: Identifies values beyond configurable threshold
- **Modified Z-Score**: Uses median absolute deviation for robust detection
- **Isolation Forest**: ML-based anomaly detection with configurable contamination rate
- **Local Outlier Factor (LOF)**: Density-based outlier detection for local anomalies

**Advanced Anomaly Detection**
- **Rate of Change Violations**: Detects unrealistic temporal changes with adaptive thresholds
- **Correlation Anomaly Detection**: Identifies breaks in expected correlation patterns
- **Power Balance Validation**: Validates energy conservation principles
- **Clustering-Based Anomalies**: Uses DBSCAN to identify abnormal operational patterns

**Example:**
.. code-block:: python

   # Configure anomaly detection
   validator = DataValidator()
   validator.set_outlier_detection_methods(['iqr', 'isolation_forest'])
   validator.set_outlier_contamination(0.1)
   
   # Perform validation with anomaly detection
   clean_df = validator.validate_data(df, enable_enhanced=True)

Power System Specific Validation
================================

The module includes specialized validation for power system variables:

**Variable Grouping**
The system intelligently groups related power system variables:

- **Generators**: ``gen_mvar_*`` (reactive power of Q control generators)
- **Substations**: ``ss_mw_*``, ``ss_mvar_*`` (active/reactive power consumption)
- **Wind Farms**: ``wind_mw_*`` (active power generation of wind parcs)
- **Shunt Elements**: ``shunt_mvar_*``, ``shunt_tap_*`` (reactive power, tap positions)
- **Voltages**: ``gen_v_*`` (voltage setpoints of Constant V generators)

**Power Balance Validation**
Validates energy conservation principles with configurable tolerance:

.. code-block:: python

   # Configure power balance validation
   validator = DataValidator()
   validator.set_power_balance_tolerance(0.05)  # 5% tolerance
   
   # Perform validation with power balance checking
   clean_df = validator.validate_data(df, enable_enhanced=True)

Configuration
=============

All validation parameters are centrally managed in ``system_configuration.py``:

**Basic Validation Settings**

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

**Enhanced Validation Settings**

.. code-block:: python

   ENHANCED_DATA_VALIDATION = {
       'advanced_gap_filling': {
           'enable_advanced_gap_filling': True,
           'default_method': 'adaptive',
           'context_size_ratio': 0.25,
           'min_context_points': 10,
       },
       'outlier_detection': {
           'default_methods': ['iqr', 'isolation_forest'],
           'contamination': 0.1,
           'zscore_threshold': 3.0,
           'iqr_multiplier': 1.5,
       },
       'power_balance_validation': {
           'tolerance': 0.05,
           'epsilon': 1e-6,
       }
   }

Usage Examples
==============

Basic Validation
~~~~~~~~~~~~~~~

.. code-block:: python

   from tsoc_data_analysis import DataValidator
   
   # Create validator and perform basic validation
   validator = DataValidator()
   clean_df = validator.validate_data(df)
   
   # Get validation summary
   summary = validator.get_validation_summary()
   print(f"Validation completed: {summary['total_records']} records processed")

Enhanced Validation
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Perform enhanced validation with all features
   validator = DataValidator()
   
   # Configure enhanced features
   validator.set_gap_filling_method('adaptive')
   validator.set_outlier_detection_methods(['iqr', 'isolation_forest', 'lof'])
   validator.set_power_balance_tolerance(0.02)
   
   # Perform validation
   clean_df = validator.validate_data(df, enable_enhanced=True)
   
   # Get detailed summary
   summary = validator.get_enhanced_validation_summary()
   print(f"Statistical outliers: {summary['statistical_outliers_count']}")
   print(f"Rate violations: {summary['rate_violations_count']}")
   print(f"Power balance violations: {summary['power_balance_violations_count']}")

Custom Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create validator with custom configuration
   validator = DataValidator()
   
   # Custom gap filling settings
   validator.set_gap_filling_config({
       'max_gap_steps': 6,
       'advanced_max_gap_steps': 18,
       'remove_large_gaps_threshold': 30
   })
   
   # Custom outlier detection
   validator.set_outlier_detection_config({
       'contamination': 0.05,
       'zscore_threshold': 2.5,
       'iqr_multiplier': 2.0
   })
   
   # Perform validation
   clean_df = validator.validate_data(df, enable_enhanced=True)

Integration with Analysis Pipeline
===================================

The validator seamlessly integrates with the existing analysis workflow:

.. code-block:: python

   from tsoc_data_analysis import DataValidator, execute, extract_representative_ops
   
   # Step 1: Validate data
   validator = DataValidator()
   clean_df = validator.validate_data(df, enable_enhanced=True)
   
   # Step 2: Perform analysis
   success, analysis_df = execute(
       month='2024-01', 
       data_dir='raw_data', 
       output_dir='results',
       input_df=clean_df  # Use validated data
   )
   
   # Step 3: Extract representative points
   if success:
       rep_df, diagnostics = extract_representative_ops(
           analysis_df, max_power=850, MAPGL=200, output_dir='results'
       )

Performance Considerations
==========================

**Optimization Features:**
- **Parallel Processing**: Uses joblib for parallel validation tasks
- **Memory Efficiency**: Processes data in chunks for large datasets
- **Adaptive Algorithms**: Automatically adjusts to data characteristics
- **Early Termination**: Stops validation when critical errors are detected

**Scalability:**
- **Large Datasets**: Handles datasets with thousands of time points
- **Many Variables**: Efficiently processes hundreds of power system variables
- **Memory Management**: Optimized for memory-constrained environments
- **Progress Tracking**: Provides detailed progress information for long-running validations

Error Handling
==============

**Comprehensive Error Handling:**
- **Data Type Issues**: Automatic conversion and validation of data types
- **Missing Data**: Intelligent handling of missing values and gaps
- **Limit Violations**: Configurable handling of out-of-range values
- **File Operations**: Graceful handling of file I/O errors
- **Memory Issues**: Handles out-of-memory conditions gracefully

**Validation Results:**
The validator provides detailed information about all validation issues:

.. code-block:: python

   # Get detailed validation results
   summary = validator.get_enhanced_validation_summary()
   
   # Access specific error types
   print("Validation Issues:")
   print(f"  Type errors: {len(summary['type_errors'])}")
   print(f"  Limit violations: {len(summary['limit_violations'])}")
   print(f"  Statistical outliers: {summary['statistical_outliers_count']}")
   print(f"  Rate violations: {summary['rate_violations_count']}")
   print(f"  Correlation anomalies: {summary['correlation_anomalies_count']}")
   print(f"  Power balance violations: {summary['power_balance_violations_count']}")
   
   # Access detailed error messages
   for error in summary['statistical_outliers']:
       print(f"  Outlier: {error}")

The module is designed to be robust and informative, providing comprehensive validation capabilities while maintaining high performance and user-friendly error reporting. 