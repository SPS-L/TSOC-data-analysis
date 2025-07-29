Operating Point Extractor Module
================================

.. automodule:: tsoc_data_analysis.operating_point_extractor
   :members:
   :undoc-members:
   :show-inheritance:

Module Overview
--------------

The ``operating_point_extractor`` module provides functions to extract representative operating points from power system data using K-means clustering with automatic cluster count selection.

This module implements the methodology described in "Automated Extraction of Representative Operating Points for a 132 kV Transmission System" and provides a comprehensive solution for identifying key operating states in power system data.

Clustering Methodology
--------------------

The module follows a systematic approach to extract representative operating points:

1. **Data Filtering**: Based on power limits and MAPGL constraints
2. **Feature Extraction**: Uses power injection features (``ss_mw_*``, ``ss_mvar_*``, ``wind_mw_*``)
3. **Standardization**: Applies StandardScaler for feature normalization
4. **Clustering**: K-means with automatic cluster count selection using multiple metrics
5. **Medoid Selection**: Returns actual snapshots closest to cluster centers
6. **MAPGL Belt**: Includes critical low-load operating points near MAPGL threshold
7. **Output Generation**: Saves results with clean column names and comprehensive reports

Core Functions
-------------

extract_representative_ops()
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Main function to extract representative operating points from power system data.

.. code-block:: python

   def extract_representative_ops(
       all_power: pd.DataFrame,
       max_power: float,
       MAPGL: float,
       k_max: int = REPRESENTATIVE_OPS['defaults']['k_max'],
       random_state: int = REPRESENTATIVE_OPS['defaults']['random_state'],
       output_dir: Optional[str] = None,
   ) -> Tuple[pd.DataFrame, dict]:
       """
       Extract representative operating points using K-means clustering.
       
       Args:
           all_power: DataFrame containing all power system data
           max_power: Maximum power limit for filtering
           MAPGL: Maximum Available Power Generation Limit
           k_max: Maximum number of clusters to test
           random_state: Random seed for reproducibility
           output_dir: Directory to save output files
           
       Returns:
           Tuple of (representative_points_df, diagnostics_dict)
       """

**Parameters:**
- ``all_power``: DataFrame with power system data (columns: ``ss_mw_*``, ``ss_mvar_*``, ``wind_mw_*``, etc.)
- ``max_power``: Maximum power limit for data filtering (MW)
- ``MAPGL``: Maximum Available Power Generation Limit (MW)
- ``k_max``: Maximum number of clusters to test (default: 10)
- ``random_state``: Random seed for reproducibility (default: 42)
- ``output_dir``: Directory to save output files (optional)

**Returns:**
- ``representative_points_df``: DataFrame with selected representative operating points
- ``diagnostics_dict``: Dictionary containing clustering diagnostics and metrics

**Example:**
.. code-block:: python

   from tsoc_data_analysis import extract_representative_ops
   
   # Extract representative points
   rep_df, diagnostics = extract_representative_ops(
       all_power=df, 
       max_power=850, 
       MAPGL=200,
       output_dir="results"
   )
   
   print(f"Selected {len(rep_df)} representative points")
   print(f"Clustering quality: {diagnostics['silhouette']:.3f}")

loadallpowerdf()
~~~~~~~~~~~~~~~

Load all_power*.csv files from a directory.

.. code-block:: python

   def loadallpowerdf(directory: str) -> pd.DataFrame:
       """
       Load all_power*.csv files from directory.
       
       Args:
           directory: Directory containing CSV files
           
       Returns:
           DataFrame with merged data from all CSV files
       """

**Example:**
.. code-block:: python

   from tsoc_data_analysis import loadallpowerdf
   
   # Load data from results directory
   df = loadallpowerdf('results')
   print(f"Loaded {len(df)} snapshots with {len(df.columns)} columns")

Configuration Integration
------------------------

All parameters are imported from ``system_configuration.REPRESENTATIVE_OPS``:

**Default Settings:**
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
       }
   }

**Customization:**
Users can override configuration values by passing parameters directly to functions:

.. code-block:: python

   # Override configuration defaults
   rep_df, diagnostics = extract_representative_ops(
       all_power=df, 
       max_power=850, 
       MAPGL=200,
       k_max=15,           # Override default k_max
       random_state=123    # Override default random_state
   )

Clustering Algorithm
-------------------

Automatic K-Means Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~

The module uses an automatic cluster count selection algorithm that evaluates multiple metrics:

**Quality Metrics:**
- **Silhouette Score**: Measures cluster cohesion and separation
- **Calinski-Harabasz Score**: Ratio of between-cluster to within-cluster variance
- **Davies-Bouldin Score**: Average similarity measure of clusters

**Selection Process:**
1. Test cluster counts from 2 to ``k_max``
2. Calculate quality metrics for each cluster count
3. Apply multi-objective ranking with configurable weights
4. Select optimal cluster count based on quality thresholds
5. Fall back to default if no clusters meet quality requirements

**Quality Thresholds:**
- **Excellent**: Silhouette score ≥ 0.7
- **Good**: Silhouette score ≥ 0.5
- **Acceptable**: Silhouette score ≥ 0.25
- **Minimum**: Silhouette score ≥ 0.1

Feature Selection
----------------

The module automatically identifies clustering features based on column prefixes:

**Power Injection Features:**
- ``ss_mw_*``: Substation active power (MW)
- ``ss_mvar_*``: Substation reactive power (MVAR)
- ``wind_mw_*``: Wind farm active power (MW)

**Feature Processing:**
1. **Identification**: Automatically finds columns with specified prefixes
2. **Filtering**: Removes columns with insufficient data
3. **Standardization**: Applies StandardScaler for normalization
4. **Validation**: Ensures minimum feature requirements are met

MAPGL Belt Analysis
------------------

The module includes special handling for critical low-load operating points near the MAPGL threshold:

**MAPGL Belt Definition:**
- **Belt Range**: MAPGL ± (MAPGL × ``mapgl_belt_multiplier``)
- **Critical Points**: Operating points within the MAPGL belt
- **Inclusion**: Ensures representation of critical low-load conditions

**Belt Processing:**
1. **Identification**: Finds snapshots within MAPGL belt
2. **Clustering**: Includes belt points in clustering analysis
3. **Selection**: Ensures belt points are represented in final selection

Output Generation
----------------

The module generates comprehensive output files with clean, readable formats:

**Output Files:**
- **Representative Points**: CSV file with selected operating points
- **Clustering Summary**: Text report with analysis details
- **Clustering Info**: JSON file with technical metrics
- **Visualizations**: Comprehensive 9-panel analysis dashboard

**File Naming:**
Output files use standardized names from configuration:
.. code-block:: python

   REPRESENTATIVE_OPS['output_files'] = {
       'representative_points': 'representative_operating_points.csv',
       'clustering_summary': 'clustering_summary.txt',
       'clustering_info': 'clustering_info.json'
   }

**Column Name Cleaning:**
Output files use clean column names via ``clean_column_name()`` function:
- Removes verbose suffixes (e.g., ``_132REACTOR_REACTIVE_POWER``)
- Creates readable, consistent naming
- Maintains data integrity and traceability

Diagnostics and Reporting
------------------------

The module provides comprehensive diagnostics and reporting capabilities:

**Diagnostics Dictionary:**
.. code-block:: python

   diagnostics = {
       'n_total': len(rep_df),                    # Number of representative points
       'original_size': len(all_power),           # Original dataset size
       'working_size': len(working),              # Filtered dataset size
       'silhouette': silhouette_score,            # Clustering quality
       'n_clusters': n_clusters,                  # Selected cluster count
       'compression_ratio': compression_ratio,    # Data compression achieved
       'mapgl_belt_points': mapgl_belt_count,     # MAPGL belt points included
       'feature_columns': feature_columns,        # Features used for clustering
       'clustering_metrics': clustering_metrics   # Detailed quality metrics
   }

**Clustering Summary Report:**
- **Data Overview**: Original and filtered dataset statistics
- **Clustering Results**: Selected cluster count and quality metrics
- **Representative Points**: Summary of selected operating points
- **MAPGL Analysis**: Belt point inclusion and analysis
- **Quality Assessment**: Detailed clustering quality evaluation

Usage Examples
-------------

Basic Usage
~~~~~~~~~~

.. code-block:: python

   from tsoc_data_analysis import extract_representative_ops, loadallpowerdf
   
   # Load data and extract representative points
   df = loadallpowerdf('results')
   rep_df, diagnostics = extract_representative_ops(
       all_power=df, 
       max_power=850, 
       MAPGL=200,
       output_dir='results'
   )
   
   # Display results
   print(f"Selected {len(rep_df)} representative points")
   print(f"Compression ratio: {diagnostics['original_size']/diagnostics['n_total']:.1f}:1")
   print(f"Clustering quality: {diagnostics['silhouette']:.3f}")

Advanced Usage
~~~~~~~~~~~~~

.. code-block:: python

   # Custom clustering parameters
   rep_df, diagnostics = extract_representative_ops(
       all_power=df, 
       max_power=850, 
       MAPGL=200,
       k_max=15,           # Test more clusters
       random_state=123,   # Custom random seed
       output_dir='custom_results'
   )
   
   # Analyze diagnostics
   print("Clustering Diagnostics:")
   print(f"  Original dataset: {diagnostics['original_size']} points")
   print(f"  Filtered dataset: {diagnostics['working_size']} points")
   print(f"  Representative points: {diagnostics['n_total']} points")
   print(f"  Selected clusters: {diagnostics['n_clusters']}")
   print(f"  Silhouette score: {diagnostics['silhouette']:.3f}")
   print(f"  MAPGL belt points: {diagnostics['mapgl_belt_points']}")

Complete Workflow
~~~~~~~~~~~~~~~~

.. code-block:: python

   from tsoc_data_analysis import (
       loadallpowerdf, 
       extract_representative_ops,
       calculate_total_load,
       calculate_net_load
   )
   
   # Step 1: Load data
   df = loadallpowerdf('results')
   
   # Step 2: Basic analysis
   total_load = calculate_total_load(df)
   net_load = calculate_net_load(df, total_load)
   
   # Step 3: Extract representative points
   rep_df, diagnostics = extract_representative_ops(
       all_power=df, 
       max_power=850, 
       MAPGL=200,
       output_dir='results'
   )
   
   # Step 4: Validate representative points
   rep_total_load = calculate_total_load(rep_df)
   rep_net_load = calculate_net_load(rep_df, rep_total_load)
   
   # Step 5: Compare statistics
   print("Original vs Representative Points:")
   print(f"  Original average net load: {net_load.mean():.2f} MW")
   print(f"  Representative average net load: {rep_net_load.mean():.2f} MW")

Performance Considerations
-------------------------

**Optimization Features:**
- **Parallel Processing**: Uses joblib for parallel cluster evaluation
- **Memory Efficiency**: Processes data in chunks for large datasets
- **Adaptive Algorithms**: Automatically adjusts to data characteristics
- **Early Termination**: Stops evaluation when quality thresholds are met

**Scalability:**
- **Large Datasets**: Handles datasets with thousands of time points
- **Many Features**: Efficiently processes hundreds of power system variables
- **Multiple Clusters**: Scales to test up to 20+ cluster counts
- **Memory Management**: Optimized for memory-constrained environments

Error Handling
-------------

**Comprehensive Error Handling:**
- **Data Validation**: Checks for required columns and data quality
- **Feature Requirements**: Ensures minimum feature count for clustering
- **Quality Thresholds**: Handles cases where no clusters meet quality requirements
- **File Operations**: Graceful handling of file I/O errors
- **Memory Issues**: Handles out-of-memory conditions gracefully

**Fallback Mechanisms:**
- **Default Clusters**: Falls back to 2 clusters if quality requirements not met
- **Feature Selection**: Adapts to available data columns
- **Output Directory**: Creates directories if they don't exist
- **File Naming**: Handles file naming conflicts gracefully

Integration with Other Modules
-----------------------------

This module integrates seamlessly with other package modules:

- **Data Loading**: Works with data from ``excel_data_processor``
- **Validation**: Compatible with validated data from ``power_data_validator``
- **Analysis**: Results can be analyzed using ``power_system_analytics``
- **Visualization**: Includes comprehensive visualization capabilities
- **Configuration**: Uses centralized configuration from ``system_configuration`` 