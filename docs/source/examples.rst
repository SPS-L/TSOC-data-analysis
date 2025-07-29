Examples
========

This section provides comprehensive examples of using the TSOC Data Analysis package for various power system analysis scenarios.

Example Categories
-----------------

The examples are organized into the following categories:

1. **Basic Analysis Examples** - Simple load analysis and generator categorization
2. **Representative Points Examples** - Clustering and representative operating points
3. **Data Validation Examples** - Data quality checks and validation workflows
4. **Visualization Examples** - Plotting and visualization techniques
5. **Advanced Workflow Examples** - Complete analysis pipelines and custom workflows

Basic Analysis Examples
----------------------

Simple Load Analysis
~~~~~~~~~~~~~~~~~~~

**Objective:** Calculate total and net load from power system data.

**Data Requirements:**
- Substation active power data (``ss_mw_*`` columns)
- Wind farm active power data (``wind_mw_*`` columns)

**Example Code:**

.. code-block:: python

   from tsoc_data_analysis import (
       loadallpowerdf,
       calculate_total_load,
       calculate_net_load,
       get_load_statistics
   )
   
   # Load data for January 2024
   df = loadallpowerdf('2024-01', data_dir='raw_data')
   
   # Calculate total load
   total_load = calculate_total_load(df)
   
   # Calculate net load (total load minus wind generation)
   net_load = calculate_net_load(df)
   
   # Get load statistics
   load_stats = get_load_statistics(total_load)
   
   print(f"Total Load Statistics:")
   print(f"  Maximum: {load_stats['max']:.2f} MW")
   print(f"  Minimum: {load_stats['min']:.2f} MW")
   print(f"  Average: {load_stats['mean']:.2f} MW")
   print(f"  Standard Deviation: {load_stats['std']:.2f} MW")

**Expected Output:**

.. code-block:: text

   Total Load Statistics:
     Maximum: 1250.45 MW
     Minimum: 450.23 MW
     Average: 850.67 MW
     Standard Deviation: 180.34 MW

Generator Categorization
~~~~~~~~~~~~~~~~~~~~~~~

**Objective:** Categorize generators as voltage control or PQ control based on their reactive power characteristics.

**Data Requirements:**
- Generator reactive power data (``gen_mvar_*`` columns)
- Generator voltage setpoints data (``gen_v_*`` columns)

**Example Code:**

.. code-block:: python

   from tsoc_data_analysis import categorize_generators
   
   # Categorize generators
   voltage_control, pq_control = categorize_generators(df)
   
   print(f"Voltage Control Generators: {len(voltage_control)}")
   for gen in voltage_control:
       print(f"  - {gen}")
   
   print(f"\nPQ Control Generators: {len(pq_control)}")
   for gen in pq_control:
       print(f"  - {gen}")

**Expected Output:**

.. code-block:: text

   Voltage Control Generators: 3
     - gen_mvar_GEN1
     - gen_mvar_GEN2
     - gen_mvar_GEN3
   
   PQ Control Generators: 2
     - gen_mvar_GEN4
     - gen_mvar_GEN5

Representative Points Examples
----------------------------

Basic Clustering Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

**Objective:** Extract representative operating points using K-means clustering.

**Data Requirements:**
- All power system data (substations, generators, wind farms)

**Example Code:**

.. code-block:: python

   from tsoc_data_analysis import extract_representative_ops
   
   # Extract representative operating points
   rep_df, diagnostics = extract_representative_ops(
       df,
       max_power=850,
       MAPGL=200,
       output_dir='results'
   )
   
   print(f"Clustering Results:")
   print(f"  Number of clusters: {diagnostics['n_clusters']}")
   print(f"  Silhouette score: {diagnostics['silhouette']:.3f}")
   print(f"  Calinski-Harabasz score: {diagnostics['calinski_harabasz']:.2f}")
   print(f"  Davies-Bouldin score: {diagnostics['davies_bouldin']:.3f}")
   
   print(f"\nRepresentative points saved to: {diagnostics['output_files']['representative_points']}")

**Expected Output:**

.. code-block:: text

   Clustering Results:
     Number of clusters: 5
     Silhouette score: 0.623
     Calinski-Harabasz score: 1250.45
     Davies-Bouldin score: 0.456
   
   Representative points saved to: results/representative_operating_points.csv

Advanced Clustering with Custom Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective:** Perform clustering with custom parameters for specific analysis requirements.

**Example Code:**

.. code-block:: python

   from tsoc_data_analysis import extract_representative_ops
   
   # Custom clustering parameters
   custom_params = {
       'k_max': 15,                    # Test up to 15 clusters
       'random_state': 123,            # Different seed for reproducibility
       'mapgl_belt_multiplier': 1.15,  # Wider MAPGL belt
       'quality_thresholds': {
           'min_silhouette': 0.3,      # Higher quality requirement
           'silhouette_excellent': 0.75,
           'silhouette_good': 0.55
       }
   }
   
   # Extract representative points with custom parameters
   rep_df, diagnostics = extract_representative_ops(
       df,
       max_power=850,
       MAPGL=200,
       output_dir='results',
       **custom_params
   )
   
   print(f"Custom Clustering Results:")
   print(f"  Selected clusters: {diagnostics['n_clusters']}")
   print(f"  Quality score: {diagnostics['silhouette']:.3f}")
   print(f"  Quality rating: {diagnostics['quality_rating']}")

**Expected Output:**

.. code-block:: text

   Custom Clustering Results:
     Selected clusters: 8
     Quality score: 0.712
     Quality rating: Good

Data Validation Examples
-----------------------

Basic Data Validation
~~~~~~~~~~~~~~~~~~~~

**Objective:** Perform comprehensive data validation to ensure data quality.

**Example Code:**

.. code-block:: python

   from tsoc_data_analysis import DataValidator
   
   # Create validator instance
   validator = DataValidator(df)
   
   # Perform basic validation
   validation_results = validator.validate_data()
   
   print(f"Data Validation Results:")
   print(f"  Total records: {validation_results['total_records']}")
   print(f"  Valid records: {validation_results['valid_records']}")
   print(f"  Invalid records: {validation_results['invalid_records']}")
   print(f"  Missing values: {validation_results['missing_values']}")
   
   if validation_results['errors']:
       print(f"\nValidation Errors:")
       for error in validation_results['errors'][:5]:  # Show first 5 errors
           print(f"  - {error}")

**Expected Output:**

.. code-block:: text

   Data Validation Results:
     Total records: 744
     Valid records: 738
     Invalid records: 6
     Missing values: 12
   
   Validation Errors:
     - Column ss_mw_SUBSTATION1: Value 1500.5 exceeds maximum limit (1000.0)
     - Column wind_mw_FARM1: Negative value (-5.2) found

Enhanced Validation with Anomaly Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective:** Perform advanced validation with anomaly detection and gap filling.

**Example Code:**

.. code-block:: python

   from tsoc_data_analysis import DataValidator
   
   # Create validator with enhanced settings
   validator = DataValidator(
       df,
       enable_advanced_gap_filling=True,
       enable_anomaly_detection=True
   )
   
   # Perform enhanced validation
   enhanced_results = validator.validate_data()
   
   print(f"Enhanced Validation Results:")
   print(f"  Anomalies detected: {enhanced_results['anomalies_detected']}")
   print(f"  Gaps filled: {enhanced_results['gaps_filled']}")
   print(f"  Outliers removed: {enhanced_results['outliers_removed']}")
   
   if enhanced_results['anomaly_details']:
       print(f"\nAnomaly Details:")
       for anomaly in enhanced_results['anomaly_details'][:3]:
           print(f"  - {anomaly['column']}: {anomaly['type']} at index {anomaly['index']}")

**Expected Output:**

.. code-block:: text

   Enhanced Validation Results:
     Anomalies detected: 8
     Gaps filled: 15
     Outliers removed: 3
   
   Anomaly Details:
     - ss_mw_SUBSTATION1: Statistical outlier at index 245
     - wind_mw_FARM1: Rate of change anomaly at index 312
     - gen_mvar_GEN1: Correlation anomaly at index 189

Visualization Examples
---------------------

Time Series Plotting
~~~~~~~~~~~~~~~~~~~

**Objective:** Create time series plots for power system variables.

**Example Code:**

.. code-block:: python

   from tsoc_data_analysis import plot_timeseries
   
   # Plot total load time series
   plot_timeseries(
       df,
       columns=['total_load'],
       title='Total Load Time Series - January 2024',
       output_file='results/total_load_timeseries.png'
   )
   
   # Plot multiple variables
   plot_timeseries(
       df,
       columns=['total_load', 'net_load', 'total_wind'],
       title='Power System Overview - January 2024',
       output_file='results/power_overview.png'
   )

**Expected Output:** Time series plots saved as PNG files in the results directory.

Comprehensive Analysis Dashboard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective:** Create a comprehensive analysis dashboard with multiple plots.

**Example Code:**

.. code-block:: python

   from tsoc_data_analysis import create_comprehensive_plots
   
   # Create comprehensive analysis dashboard
   create_comprehensive_plots(
       df,
       output_dir='results',
       save_plots=True,
       plot_format='png'
   )
   
   print("Comprehensive analysis dashboard created with:")
   print("  - Time series plots")
   print("  - Daily load profiles")
   print("  - Monthly statistics")
   print("  - Generator analysis")
   print("  - Wind power analysis")

**Expected Output:** Multiple analysis plots and summary files in the results directory.

Advanced Workflow Examples
-------------------------

Complete Analysis Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective:** Perform a complete analysis pipeline from data loading to results generation.

**Example Code:**

.. code-block:: python

   from tsoc_data_analysis import execute, extract_representative_ops
   
   # Step 1: Execute full analysis
   success, df = execute(
       month='2024-01',
       data_dir='raw_data',
       output_dir='results',
       save_plots=True,
       save_csv=True,
       verbose=True
   )
   
   if success:
       print("Basic analysis completed successfully")
       
       # Step 2: Extract representative points
       rep_df, diagnostics = extract_representative_ops(
           df,
           max_power=850,
           MAPGL=200,
           output_dir='results'
       )
       
       print(f"Representative points extracted: {len(rep_df)} clusters")
       
       # Step 3: Generate summary report
       print("\nAnalysis Summary:")
       print(f"  Data period: January 2024")
       print(f"  Total records: {len(df)}")
       print(f"  Representative clusters: {len(rep_df)}")
       print(f"  Clustering quality: {diagnostics['silhouette']:.3f}")

**Expected Output:**

.. code-block:: text

   Basic analysis completed successfully
   Representative points extracted: 5 clusters
   
   Analysis Summary:
     Data period: January 2024
     Total records: 744
     Representative clusters: 5
     Clustering quality: 0.623

Multi-Month Analysis
~~~~~~~~~~~~~~~~~~~

**Objective:** Perform analysis across multiple months and compare results.

**Example Code:**

.. code-block:: python

   from tsoc_data_analysis import execute, extract_representative_ops
   import pandas as pd
   
   # Analyze multiple months
   months = ['2024-01', '2024-02', '2024-03']
   results = {}
   
   for month in months:
       print(f"\nAnalyzing {month}...")
       
       # Execute analysis for each month
       success, df = execute(
           month=month,
           data_dir='raw_data',
           output_dir=f'results/{month}',
           save_plots=True
       )
       
       if success:
           # Extract representative points
           rep_df, diagnostics = extract_representative_ops(
               df,
               max_power=850,
               MAPGL=200,
               output_dir=f'results/{month}'
           )
           
           results[month] = {
               'data': df,
               'representative_points': rep_df,
               'diagnostics': diagnostics
           }
   
   # Compare results across months
   print(f"\nMulti-Month Comparison:")
   for month, result in results.items():
       print(f"  {month}: {len(result['representative_points'])} clusters, "
             f"quality: {result['diagnostics']['silhouette']:.3f}")

**Expected Output:**

.. code-block:: text

   Analyzing 2024-01...
   Analyzing 2024-02...
   Analyzing 2024-03...
   
   Multi-Month Comparison:
     2024-01: 5 clusters, quality: 0.623
     2024-02: 6 clusters, quality: 0.589
     2024-03: 5 clusters, quality: 0.647

Custom Analysis Workflow
~~~~~~~~~~~~~~~~~~~~~~~

**Objective:** Create a custom analysis workflow for specific requirements.

**Example Code:**

.. code-block:: python

   from tsoc_data_analysis import (
       loadallpowerdf,
       calculate_total_load,
       calculate_net_load,
       categorize_generators,
       extract_representative_ops
   )
   import pandas as pd
   import matplotlib.pyplot as plt
   
   # Custom analysis workflow
   def custom_analysis(month, data_dir, output_dir):
       """Custom analysis workflow for specific requirements."""
       
       # Step 1: Load and preprocess data
       print(f"Loading data for {month}...")
       df = loadallpowerdf(month, data_dir=data_dir)
       
       # Step 2: Calculate key metrics
       print("Calculating key metrics...")
       total_load = calculate_total_load(df)
       net_load = calculate_net_load(df)
       
       # Step 3: Generator analysis
       print("Analyzing generators...")
       voltage_control, pq_control = categorize_generators(df)
       
       # Step 4: Representative points with custom parameters
       print("Extracting representative points...")
       rep_df, diagnostics = extract_representative_ops(
           df,
           max_power=850,
           MAPGL=200,
           k_max=8,
           random_state=42
       )
       
       # Step 5: Generate custom report
       report = {
           'month': month,
           'total_records': len(df),
           'total_load_stats': {
               'max': total_load.max(),
               'min': total_load.min(),
               'mean': total_load.mean()
           },
           'generators': {
               'voltage_control': len(voltage_control),
               'pq_control': len(pq_control)
           },
           'clustering': {
               'n_clusters': len(rep_df),
               'quality': diagnostics['silhouette']
           }
       }
       
       # Step 6: Save results
       pd.DataFrame([report]).to_csv(f'{output_dir}/custom_report.csv', index=False)
       
       return report
   
   # Execute custom analysis
   report = custom_analysis('2024-01', 'raw_data', 'results')
   print(f"\nCustom Analysis Report:")
   print(f"  Month: {report['month']}")
   print(f"  Records: {report['total_records']}")
   print(f"  Load range: {report['total_load_stats']['min']:.1f} - {report['total_load_stats']['max']:.1f} MW")
   print(f"  Generators: {report['generators']['voltage_control']} voltage control, {report['generators']['pq_control']} PQ control")
   print(f"  Clusters: {report['clustering']['n_clusters']} (quality: {report['clustering']['quality']:.3f})")

**Expected Output:**

.. code-block:: text

   Loading data for 2024-01...
   Calculating key metrics...
   Analyzing generators...
   Extracting representative points...
   
   Custom Analysis Report:
     Month: 2024-01
     Records: 744
     Load range: 450.2 - 1250.5 MW
     Generators: 3 voltage control, 2 PQ control
     Clusters: 5 (quality: 0.623) 