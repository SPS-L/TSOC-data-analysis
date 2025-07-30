Examples
========

This section provides comprehensive examples of using the TSOC Data Analysis package for various power system analysis scenarios.

Example Categories
------------------

The examples are organized into the following categories:

1. **Basic Analysis Examples** - Simple load analysis and generator categorization
2. **Representative Points Examples** - Clustering and representative operating points
3. **Data Validation Examples** - Data quality checks and validation workflows
4. **Visualization Examples** - Plotting and visualization techniques
5. **Advanced Workflow Examples** - Complete analysis pipelines and custom workflows

Basic Analysis Examples
-----------------------

Simple Load Analysis
~~~~~~~~~~~~~~~~~~~~

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
   
   # Load data from results directory
   df = loadallpowerdf('results')
   
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
~~~~~~~~~~~~~~~~~~~~~~~~

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
------------------------------

Basic Clustering Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

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
   print(f"  Number of clusters: {diagnostics['k']}")
   print(f"  Silhouette score: {diagnostics['silhouette']:.3f}")
   print(f"  Calinski-Harabasz score: {diagnostics['ch']:.2f}")
   print(f"  Davies-Bouldin score: {diagnostics['db']:.3f}")
   
   print(f"\nRepresentative points saved to: results/representative_operating_points.csv")

**Expected Output:**

.. code-block:: text

   Clustering Results:
     Number of clusters: 5
     Silhouette score: 0.623
     Calinski-Harabasz score: 1250.45
     Davies-Bouldin score: 0.456
   
   Representative points saved to: results/representative_operating_points.csv

Advanced Clustering with Custom Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective:** Perform clustering with custom parameters for specific analysis requirements.

**Example Code:**

.. code-block:: python

   from tsoc_data_analysis import extract_representative_ops, REPRESENTATIVE_OPS

   # Temporarily modify the configuration
   original_config = REPRESENTATIVE_OPS.copy()

   # Apply your custom parameters
   REPRESENTATIVE_OPS['defaults']['k_max'] = 15
   REPRESENTATIVE_OPS['defaults']['random_state'] = 123
   REPRESENTATIVE_OPS['defaults']['mapgl_belt_multiplier'] = 1.15
   REPRESENTATIVE_OPS['quality_thresholds']['min_silhouette'] = 0.3
   REPRESENTATIVE_OPS['quality_thresholds']['silhouette_excellent'] = 0.75
   REPRESENTATIVE_OPS['quality_thresholds']['silhouette_good'] = 0.55
   
   # Extract representative points with custom parameters
   rep_df, diagnostics = extract_representative_ops(
       df,
       max_power=850,
       MAPGL=200,
       output_dir='results'
   )
   
   print(f"Custom Clustering Results:")
   print(f"  Selected clusters: {diagnostics['k']}")
   print(f"  Quality score: {diagnostics['silhouette']:.3f}")
   
   # Determine quality rating based on silhouette score
   if diagnostics['silhouette'] > 0.7:
       quality_rating = "Excellent"
   elif diagnostics['silhouette'] > 0.5:
       quality_rating = "Good"
   elif diagnostics['silhouette'] > 0.25:
       quality_rating = "Acceptable"
   else:
       quality_rating = "Poor"
   
   print(f"  Quality rating: {quality_rating}")
   
   # Restore original configuration (optional)
   REPRESENTATIVE_OPS.update(original_config)

**Expected Output:**

.. code-block:: text

   Custom Clustering Results:
     Selected clusters: 8
     Quality score: 0.712
     Quality rating: Good

Data Validation Examples
------------------------

Basic Data Validation
~~~~~~~~~~~~~~~~~~~~~

**Objective:** Perform comprehensive data validation to ensure data quality.

**Example Code:**

.. code-block:: python

   from tsoc_data_analysis import DataValidator
   
   # Create validator instance
   validator = DataValidator()
   
   # Perform basic validation
   validated_df = validator.validate_dataframe(df)
   validation_summary = validator.get_validation_summary()
   
   print(f"Data Validation Results:")
   print(f"  Total records processed: {validation_summary['total_records_processed']}")
   print(f"  Records with errors: {validation_summary['records_with_errors']}")
   print(f"  Type errors: {len(validation_summary['type_errors'])}")
   print(f"  Limit errors: {len(validation_summary['limit_errors'])}")
   print(f"  Gaps filled: {validation_summary['gaps_filled']}")
   
   if validation_summary['limit_errors']:
       print(f"\nLimit Validation Errors:")
       for error in validation_summary['limit_errors'][:5]:  # Show first 5 errors
           print(f"  - {error}")

**Expected Output:**

.. code-block:: text

   Data Validation Results:
     Total records processed: 744
     Records with errors: 6
     Type errors: 0
     Limit errors: 6
     Gaps filled: 12
   
   Limit Validation Errors:
     - Column ss_mw_SUBSTATION1: Value 1500.5 exceeds maximum limit (1000.0)
     - Column wind_mw_FARM1: Negative value (-5.2) found

Enhanced Validation with Anomaly Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective:** Perform advanced validation with anomaly detection and gap filling.

**Example Code:**

.. code-block:: python

   from tsoc_data_analysis import EnhancedDataValidator
   
   # Create enhanced validator
   validator = EnhancedDataValidator()
   
   # Perform enhanced validation with anomaly detection
   validated_df = validator.validate_dataframe_enhanced(
       df,
       use_comprehensive_anomaly_detection=True,
       use_advanced_gap_filling=True
   )
   
   enhanced_summary = validator.get_enhanced_validation_summary()
   
   print(f"Enhanced Validation Results:")
   print(f"  Anomalies detected: {enhanced_summary.get('anomalies_detected', 0)}")
   print(f"  Gaps filled: {enhanced_summary.get('gaps_filled', 0)}")
   print(f"  Outliers removed: {enhanced_summary.get('outliers_removed', 0)}")
   
   if enhanced_summary.get('anomaly_details'):
       print(f"\nAnomaly Details:")
       for anomaly in enhanced_summary['anomaly_details'][:3]:
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
----------------------

Time Series Plotting
~~~~~~~~~~~~~~~~~~~~

**Objective:** Create time series plots for power system variables.

**Example Code:**

.. code-block:: python

   from tsoc_data_analysis.power_system_visualizer import plot_load_timeseries
   import matplotlib.pyplot as plt
   
   # Plot total load and net load time series
   fig, ax = plt.subplots(figsize=(12, 6))
   plot_load_timeseries(total_load, net_load, ax=ax)
   plt.savefig('results/load_timeseries.png', dpi=300, bbox_inches='tight')
   plt.close()

**Expected Output:** Time series plot saved as PNG file in the results directory.

Comprehensive Analysis Dashboard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective:** Create a comprehensive analysis dashboard with multiple plots.

**Example Code:**

.. code-block:: python

   from tsoc_data_analysis.power_system_visualizer import create_comprehensive_plots
   import matplotlib.pyplot as plt
   
   # Create comprehensive analysis dashboard
   fig = create_comprehensive_plots(total_load, net_load)
   fig.savefig('results/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
   plt.close()
   
   print("Comprehensive analysis dashboard created with:")
   print("  - Time series plots")
   print("  - Daily load profiles")
   print("  - Monthly statistics")
   print("  - Load statistics summary")

**Expected Output:** Comprehensive analysis plot saved as PNG file in the results directory.

Advanced Workflow Examples
--------------------------

Complete Analysis Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~

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
       
       print(f"Representative points extracted: {len(rep_df)} points")
       
       # Step 3: Generate summary report
       print("\nAnalysis Summary:")
       print(f"  Data period: January 2024")
       print(f"  Total records: {len(df)}")
       print(f"  Representative points: {len(rep_df)}")
       print(f"  Clustering quality: {diagnostics['silhouette']:.3f}")

**Expected Output:**

.. code-block:: text

   Basic analysis completed successfully
   Representative points extracted: 5 points
   
   Analysis Summary:
     Data period: January 2024
     Total records: 744
     Representative points: 5
     Clustering quality: 0.623

Multi-Month Analysis
~~~~~~~~~~~~~~~~~~~~

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
       print(f"  {month}: {len(result['representative_points'])} points, "
             f"quality: {result['diagnostics']['silhouette']:.3f}")

**Expected Output:**

.. code-block:: text

   Analyzing 2024-01...
   Analyzing 2024-02...
   Analyzing 2024-03...
   
   Multi-Month Comparison:
     2024-01: 5 points, quality: 0.623
     2024-02: 6 points, quality: 0.589
     2024-03: 5 points, quality: 0.647

Custom Analysis Workflow
~~~~~~~~~~~~~~~~~~~~~~~~

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
   def custom_analysis(data_dir, output_dir):
       """Custom analysis workflow for specific requirements."""
       
       # Step 1: Load and preprocess data
       print("Loading data...")
       df = loadallpowerdf(data_dir)
       
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
               'n_points': len(rep_df),
               'quality': diagnostics['silhouette']
           }
       }
       
       # Step 6: Save results
       pd.DataFrame([report]).to_csv(f'{output_dir}/custom_report.csv', index=False)
       
       return report
   
   # Execute custom analysis
   report = custom_analysis('results', 'results')
   print(f"\nCustom Analysis Report:")
   print(f"  Records: {report['total_records']}")
   print(f"  Load range: {report['total_load_stats']['min']:.1f} - {report['total_load_stats']['max']:.1f} MW")
   print(f"  Generators: {report['generators']['voltage_control']} voltage control, {report['generators']['pq_control']} PQ control")
   print(f"  Representative points: {report['clustering']['n_points']} (quality: {report['clustering']['quality']:.3f})") 