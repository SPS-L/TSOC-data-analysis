Troubleshooting
===============

This guide provides solutions to common issues and problems that may arise when using the TSOC Data Analysis package.

Common Error Messages
---------------------


File Not Found Errors
~~~~~~~~~~~~~~~~~~~~

**Error:** `FileNotFoundError: [Errno 2] No such file or directory: 'raw_data/substation_active_power.xlsx'`

**Cause:** Excel files are missing or in the wrong location.

**Solution:**

1. **Check file structure:**

   .. code-block:: bash

      ls -la raw_data/
      # Should show:
      # substation_active_power.xlsx
      # substation_reactive_power.xlsx
      # wind_farm_active_power.xlsx
      # shunt_element_reactive_power.xlsx
      # generator_voltage_setpoints.xlsx
      # generator_reactive_power.xlsx

2. **Verify file names match configuration:**

   .. code-block:: python

      from tsoc_data_analysis.system_configuration import FILES
      
      for data_type, filename in FILES.items():
          print(f"{data_type}: {filename}")

3. **Check data directory path:**

   .. code-block:: python

      # Use absolute path or correct relative path
      success, df = execute(month='2024-01', data_dir='/full/path/to/raw_data')

Data Quality Issues
~~~~~~~~~~~~~~~~~~

**Error:** `ValueError: Data contains too many missing values`

**Cause:** Excel files have excessive missing data or incorrect structure.

**Solution:**

1. **Check Excel file structure:**

   .. code-block:: python

      import pandas as pd
      
      # Load Excel file and check structure
      df = pd.read_excel('raw_data/substation_active_power.xlsx')
      print(f"Shape: {df.shape}")
      print(f"Missing values: {df.isnull().sum().sum()}")
      print(f"Columns: {list(df.columns)}")

2. **Verify data starts at correct row:**

   .. code-block:: python

      # Check if data starts at row 6 (0-indexed)
      df = pd.read_excel('raw_data/substation_active_power.xlsx', header=None)
      print(f"Row 5 (should be timestamps): {df.iloc[4, :5]}")
      print(f"Row 6 (should be data): {df.iloc[5, :5]}")

3. **Check column naming:**

   .. code-block:: python

      # Verify column names follow expected pattern
      expected_prefix = 'ss_mw_'
      matching_cols = [col for col in df.columns if col.startswith(expected_prefix)]
      print(f"Found {len(matching_cols)} columns with prefix '{expected_prefix}'")

Performance Issues
------------------

Slow Clustering for Large Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Clustering takes too long for large datasets.

**Solutions:**

1. **Reduce dataset size:**

   .. code-block:: python

      # Use sampling for large datasets
      from tsoc_data_analysis import extract_representative_ops
      
      # Sample data for faster clustering
      sample_df = df.sample(n=10000, random_state=42)
      
      rep_df, diagnostics = extract_representative_ops(
          sample_df,
          max_power=850,
          MAPGL=200
      )

2. **Adjust clustering parameters:**

   .. code-block:: python

      # Use fewer clusters for faster processing
      rep_df, diagnostics = extract_representative_ops(
          df,
          max_power=850,
          MAPGL=200,
          k_max=5,  # Reduce from default 10
          random_state=42
      )

3. **Use parallel processing:**

   .. code-block:: python

      # Enable parallel processing if available
      from joblib import parallel_backend
      
      with parallel_backend('threading', n_jobs=4):
          rep_df, diagnostics = extract_representative_ops(
              df,
              max_power=850,
              MAPGL=200
          )

Memory Issues
~~~~~~~~~~~~

**Problem:** Out of memory errors when processing large datasets.

**Solutions:**

1. **Process data in chunks:**

   .. code-block:: python

      # Process data month by month
      months = ['2024-01', '2024-02', '2024-03']
      results = {}
      
      for month in months:
          print(f"Processing {month}...")
          success, df = execute(month=month, data_dir='raw_data')
          if success:
              results[month] = df
              # Clear memory
              del df

2. **Reduce memory usage:**

   .. code-block:: python

      # Use smaller data types
      import pandas as pd
      
      # Convert to smaller data types
      df = df.astype({
          'ss_mw_SUBSTATION1': 'float32',
          'wind_mw_FARM1': 'float32'
      })

3. **Monitor memory usage:**

   .. code-block:: python

      import psutil
      
      def check_memory():
          memory = psutil.virtual_memory()
          print(f"Memory usage: {memory.percent}%")
          return memory.percent < 90  # Warning if > 90%
      
      # Check before processing
      if check_memory():
          # Proceed with processing
          pass

Configuration Problems
----------------------

Invalid Configuration Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Configuration errors or invalid parameter values.

**Solutions:**

1. **Validate configuration:**

   .. code-block:: python

      from tsoc_data_analysis.system_configuration import (
          FILES, COLUMN_PREFIXES, DATA_VALIDATION, REPRESENTATIVE_OPS
      )
      
      # Check file mappings
      for data_type, filename in FILES.items():
          if not filename.endswith('.xlsx'):
              print(f"Warning: {data_type} file should end with .xlsx")
      
      # Check column prefixes
      for data_type, prefix in COLUMN_PREFIXES.items():
          if not prefix.endswith('_'):
              print(f"Warning: {data_type} prefix should end with '_'")

2. **Reset to defaults:**

   .. code-block:: python

      # Reset clustering parameters to defaults
      REPRESENTATIVE_OPS['defaults']['k_max'] = 10
      REPRESENTATIVE_OPS['defaults']['random_state'] = 42
      REPRESENTATIVE_OPS['quality_thresholds']['min_silhouette'] = 0.25

3. **Check parameter ranges:**

   .. code-block:: python

      # Validate parameter ranges
      if REPRESENTATIVE_OPS['defaults']['k_max'] < 2:
          print("Error: k_max must be at least 2")
      
      if DATA_VALIDATION['gap_filling']['max_gap_steps'] < 1:
          print("Error: max_gap_steps must be at least 1")

Missing Dependencies
--------------------

**Problem:** Import errors or missing packages.

**Solutions:**

1. **Install missing dependencies:**

   .. code-block:: bash

      pip install pandas numpy matplotlib seaborn openpyxl scikit-learn scipy psutil joblib

2. **Check package versions:**

   .. code-block:: python

      import pandas as pd
      import numpy as np
      import matplotlib
      import seaborn
      import openpyxl
      import sklearn
      
      print(f"pandas: {pd.__version__}")
      print(f"numpy: {np.__version__}")
      print(f"matplotlib: {matplotlib.__version__}")
      print(f"seaborn: {seaborn.__version__}")
      print(f"openpyxl: {openpyxl.__version__}")
      print(f"scikit-learn: {sklearn.__version__}")

3. **Install development dependencies:**

   .. code-block:: bash

      pip install -e ".[dev]"

Visualization Issues
~~~~~~~~~~~~~~~~~~~~

**Problem:** Plotting errors or missing plots.

**Solutions:**

1. **Check matplotlib backend:**

   .. code-block:: python

      import matplotlib
      print(f"Backend: {matplotlib.get_backend()}")
      
      # Set backend if needed
      matplotlib.use('Agg')  # For non-interactive environments

2. **Create output directory:**

   .. code-block:: python

      import os
      
      # Ensure output directory exists
      output_dir = 'results'
      os.makedirs(output_dir, exist_ok=True)

3. **Check file permissions:**

   .. code-block:: python

      # Check if directory is writable
      import os
      
      if os.access('results', os.W_OK):
          print("Directory is writable")
      else:
          print("Directory is not writable")

Parallel Processing Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Parallel processing errors or performance issues.

**Solutions:**

1. **Disable parallel processing:**

   .. code-block:: python

      # Use single-threaded processing
      from joblib import parallel_backend
      
      with parallel_backend('sequential'):
          rep_df, diagnostics = extract_representative_ops(
              df,
              max_power=850,
              MAPGL=200
          )

2. **Adjust number of jobs:**

   .. code-block:: python

      # Use fewer parallel jobs
      from joblib import parallel_backend
      
      with parallel_backend('threading', n_jobs=2):
          rep_df, diagnostics = extract_representative_ops(
              df,
              max_power=850,
              MAPGL=200
          )

Data Format Issues
------------------

Excel File Structure Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Excel files have incorrect structure or format.

**Solutions:**

1. **Check Excel file format:**

   .. code-block:: python

      import pandas as pd
      
      # Check if file can be read
      try:
          df = pd.read_excel('raw_data/substation_active_power.xlsx')
          print("File can be read successfully")
      except Exception as e:
          print(f"Error reading file: {e}")

2. **Verify data structure:**

   .. code-block:: python

      # Check expected structure
      df = pd.read_excel('raw_data/substation_active_power.xlsx', header=None)
      
      # Check timestamp column (column C, row 6+)
      timestamps = df.iloc[5:, 2]  # Column C (0-indexed = 2)
      print(f"Timestamp range: {timestamps.min()} to {timestamps.max()}")
      
      # Check substation names (row 2)
      substation_names = df.iloc[1, 6:]  # Row 2, starting from column G
      print(f"Substation names: {list(substation_names)}")

3. **Fix common structure issues:**

   .. code-block:: python

      # If timestamps are in wrong column
      if df.iloc[5, 2] is None:  # Column C is empty
          # Check other columns for timestamps
          for col in range(df.shape[1]):
              if df.iloc[5, col] is not None:
                  print(f"Timestamps found in column {col}")

Data Type Issues
~~~~~~~~~~~~~~~~

**Problem:** Data type conversion errors or incorrect data types.

**Solutions:**

1. **Check data types:**

   .. code-block:: python

      # Check column data types
      for col in df.columns:
          if col.startswith('ss_mw_'):
              print(f"{col}: {df[col].dtype}")
              print(f"  Sample values: {df[col].head()}")

2. **Convert data types:**

   .. code-block:: python

      # Convert to numeric types
      for col in df.columns:
          if col.startswith('ss_mw_'):
              df[col] = pd.to_numeric(df[col], errors='coerce')

3. **Handle non-numeric values:**

   .. code-block:: python

      # Find and handle non-numeric values
      for col in df.columns:
          if col.startswith('ss_mw_'):
              non_numeric = pd.to_numeric(df[col], errors='coerce').isna()
              if non_numeric.any():
                  print(f"Non-numeric values in {col}: {df[col][non_numeric].unique()}")

Debugging Techniques
--------------------

Enable Verbose Mode
~~~~~~~~~~~~~~~~~~~

**Solution:** Use verbose mode for detailed output.

.. code-block:: python

   # Enable verbose mode in CLI
   tsoc-analyze 2024-01 --verbose
   
   # Enable verbose mode in Python
   success, df = execute(
       month='2024-01',
       data_dir='raw_data',
       output_dir='results',
       verbose=True
   )

Log Analysis
~~~~~~~~~~~~

**Solution:** Check log files for detailed error information.

.. code-block:: python

   import logging
   
   # Set up logging
   logging.basicConfig(
       level=logging.DEBUG,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
       handlers=[
           logging.FileHandler('tsoc_analysis.log'),
           logging.StreamHandler()
       ]
   )
   
   # Run analysis with logging
   success, df = execute(month='2024-01', data_dir='raw_data')

Step-by-Step Debugging
~~~~~~~~~~~~~~~~~~~~~~

**Solution:** Debug each step individually.

.. code-block:: python

   # Step 1: Check data loading
   try:
       df = loadallpowerdf('2024-01', data_dir='raw_data')
       print(f"Data loaded: {df.shape}")
   except Exception as e:
       print(f"Data loading error: {e}")
       return
   
   # Step 2: Check data validation
   try:
       validator = DataValidator(df)
       validation_results = validator.validate_data()
       print(f"Validation completed: {validation_results['valid_records']} valid records")
   except Exception as e:
       print(f"Validation error: {e}")
       return
   
   # Step 3: Check clustering
   try:
       rep_df, diagnostics = extract_representative_ops(
           df,
           max_power=850,
           MAPGL=200
       )
       print(f"Clustering completed: {len(rep_df)} clusters")
   except Exception as e:
       print(f"Clustering error: {e}")

Getting Help
------------

**Additional Resources:**

1. **Check the documentation** for detailed API reference and examples
2. **Review error messages** carefully for specific issue details
3. **Test with sample data** to isolate the problem
4. **Check system requirements** and dependencies
5. **Contact support** at info@sps-lab.org for persistent issues

**Common Debugging Checklist:**

- [ ] All required Excel files are present in the data directory
- [ ] File names match the configuration in `system_configuration.py`
- [ ] Excel files have the correct structure (timestamps in column C, data starting at row 6)
- [ ] Column names follow the expected prefix patterns (``ss_mw_*``, ``wind_mw_*``, etc.)
- [ ] Data types are numeric (no text or mixed types)
- [ ] Sufficient memory is available for the dataset size
- [ ] All required Python packages are installed with compatible versions 