User Guide
==========

Getting Started
---------------

The TSOC Data Analysis package is designed to be user-friendly while providing powerful analysis capabilities. This guide will walk you through the essential workflows and best practices.

The package consists of several interconnected modules:

**Core Analysis Modules**

- **Power System Analytics**: Load calculations, generator categorization, reactive power analysis
- **Operating Point Extractor**: Representative operating points extraction using K-means clustering and advanced clustering algorithms

**Data Processing Modules**

- **Excel Data Processor**: Excel file loading and data preprocessing
- **Power Data Validator**: Comprehensive data validation and quality assurance

**Configuration and Utilities**

- **System Configuration**: Central configuration hub and shared utilities
- **Power System Visualizer**: Visualization and plotting functions

Key Features
------------

- **Month-based data filtering** for efficient processing of large datasets
- **Load calculations** (Total Load, Net Load) with comprehensive statistics
- **Wind power analysis** with generation statistics and profiles
- **Generator categorization** (Voltage Control vs PQ Control)
- **Reactive power analysis** with comprehensive calculations
- **Data validation** with type checking, limit validation, and gap filling
- **Representative operating points extraction** using K-means clustering and enhanced algorithms with automatic quality optimization
- **Comprehensive logging** and error handling

Data Requirements
-----------------

Input Data Format
~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~

* **Time Series Continuity**: Data should be continuous with regular time intervals
* **Unit Consistency**: All power values in MW/MVAR, voltages in KV
* **Sign Conventions**: 

  * Positive load values indicate consumption
  * Positive generation values indicate production
  * Generator reactive power is subtracted (negative contribution)

* **Missing Data**: Gaps up to 3 time steps are interpolated linearly

Basic Workflow
--------------

#. **Data Preparation**: Organize Excel files in the correct format
#. **Data Loading**: Use the package to load and merge data
#. **Data Validation**: Perform quality checks and gap filling
#. **Analysis**: Calculate loads, categorize generators, analyze wind power
#. **Representative Points**: Extract representative operating points using clustering
#. **Visualization**: Create plots and analysis dashboards
#. **Results**: Save analysis results and reports

Clustering Methods
------------------

The package provides two main approaches for extracting representative operating points:

Standard Clustering
~~~~~~~~~~~~~~~~~~~

The standard approach uses K-means clustering with automatic cluster number selection:

.. code-block:: python

   from tsoc_data_analysis import extract_representative_ops
   
   rep_df, diagnostics = extract_representative_ops(
       df,
       max_power=850,
       MAPGL=200,
       output_dir='results'
   )

**Features:**

- Fast K-means clustering
- Automatic cluster number selection (k=2 to k_max)
- MAPGL belt inclusion for critical low-load points
- Quality metrics (silhouette score, Calinski-Harabasz, Davies-Bouldin)

Enhanced Clustering
~~~~~~~~~~~~~~~~~~~

The enhanced approach provides advanced clustering with multiple optimization techniques:

.. code-block:: python

   from tsoc_data_analysis import extract_representative_ops_enhanced
   
   rep_df, diagnostics = extract_representative_ops_enhanced(
       df,
       max_power=850,
       MAPGL=200,
       output_dir='results_enhanced',
       use_enhanced_preprocessing=True,
       try_alternative_algorithms=True,
       use_dimensionality_reduction=True
   )

**Advanced Features:**

- **Data Preprocessing**: Outlier removal, zero-variance feature elimination, correlation analysis
- **Feature Engineering**: Power factors, load diversity, wind penetration, temporal patterns
- **Alternative Algorithms**: DBSCAN, Agglomerative Clustering, Gaussian Mixture Models
- **Dimensionality Reduction**: Principal Component Analysis (PCA) before clustering
- **Automatic Method Selection**: Tests multiple approaches and selects the best performing one

**When to Use Enhanced Clustering:**

- Standard clustering gives poor quality scores (< 0.4)
- Dataset has complex operational patterns
- High-dimensional data with many features
- Need maximum clustering quality for critical analysis

**Performance Considerations:**

- Enhanced clustering is 3-5x slower than standard
- Recommended for final analysis rather than exploratory work
- Provides detailed comparison reports for method evaluation

Python API Usage
----------------

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
   df = loadallpowerdf('results')
   
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
   print(f"  Representative points: {len(rep_df)}")

**Enhanced Clustering:**

.. code-block:: python

   from tsoc_data_analysis import extract_representative_ops_enhanced
   
   # Enhanced clustering with advanced algorithms
   rep_df, diagnostics = extract_representative_ops_enhanced(
       df,
       max_power=850,
       MAPGL=200,
       output_dir='results_enhanced',
       use_enhanced_preprocessing=True,
       try_alternative_algorithms=True,
       use_dimensionality_reduction=True
   )
   
   print(f"Enhanced Clustering Results:")
   print(f"  Best method: {diagnostics['best_method']}")
   print(f"  Quality improvement: {diagnostics['best_silhouette']:.3f}")
   print(f"  Representative points: {len(rep_df)}")

**Data Validation:**

.. code-block:: python

   from tsoc_data_analysis import DataValidator
   
   # Create validator instance
   validator = DataValidator()
   
   # Perform validation
   validated_df = validator.validate_dataframe(df)
   validation_summary = validator.get_validation_summary()
   
   print(f"Validation Results:")
   print(f"  Total records processed: {validation_summary['total_records_processed']}")
   print(f"  Records with errors: {validation_summary['records_with_errors']}")
   print(f"  Type errors: {len(validation_summary['type_errors'])}")
   print(f"  Limit errors: {len(validation_summary['limit_errors'])}")
   print(f"  Gaps filled: {validation_summary['gaps_filled']}")

Output Files
------------  

The package generates various output files depending on the options selected:

**Analysis Results:**

- `analysis_summary.txt` - Summary statistics and key metrics
- `load_statistics.csv` - Detailed load analysis results
- `generator_analysis.csv` - Generator categorization and statistics
- `wind_power_analysis.csv` - Wind farm analysis results

**Representative Points:**

- `representative_operating_points.csv` - Extracted representative points
- `clustering_summary.txt` - Clustering analysis summary
- `enhanced_clustering_summary.txt` - Enhanced clustering comparison and details (when using enhanced method)

**Visualization:**

- `total_load_timeseries.png` - Total load time series plot
- `net_load_timeseries.png` - Net load time series plot
- `daily_load_profiles.png` - Daily load profile analysis
- `comprehensive_analysis.png` - Multi-panel analysis dashboard
