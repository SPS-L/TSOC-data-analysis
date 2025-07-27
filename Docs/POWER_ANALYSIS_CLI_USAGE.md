# Power Analysis CLI - Dual Usage Guide

**Author:** Sustainable Power Systems Lab (SPSL)  
**Web:** https://sps-lab.org  
**Contact:** info@sps-lab.org

## Overview

The `power_analysis_cli.py` tool has been enhanced to support both command-line usage and programmatic access from other Python scripts. This dual-mode functionality allows for:

- **Command Line Interface**: Traditional CLI usage for interactive analysis
- **Python Module**: Direct function calls for integration into larger workflows

## Usage Methods

### 1. Command Line Interface (CLI)

Use the tool from the command line exactly as before:

```bash
# Basic analysis for all data
python power_analysis_cli.py --save-plots --save-csv

# Month-specific analysis
python power_analysis_cli.py 2024-01 --output-dir results --verbose

# Summary only
python power_analysis_cli.py 2024-12 --summary-only
```

**Available CLI Arguments:**
- `month` (optional): Month filter in YYYY-MM format
- `--data-dir`: Data directory path
- `--output-dir`: Output directory path
- `--save-csv`: Save results to CSV files
- `--save-plots`: Generate and save plots
- `--summary-only`: Generate only summary report
- `--verbose`: Enable detailed progress output

### 2. Python Module Import

Import and use the `execute()` function from other Python scripts:

```python
from power_analysis_cli import execute

# Basic analysis
success, df = execute()
if success:
    print(f"Analysis completed with {len(df)} time points")

# Full analysis with options
success, df = execute(
    month="2024-01",
    data_dir="raw_data",
    output_dir="results",
    save_csv=True,
    save_plots=True,
    verbose=True
)
if success:
    # Use the dataframe for further analysis
    from representative_ops import extract_representative_ops
    rep_df, diag = extract_representative_ops(df, max_power=850, MAPGL=200)
```

## Function Signature

```python
def execute(month=None, data_dir=DATA_DIR, output_dir=DEFAULT_OUTPUT_DIR, 
           save_csv=False, save_plots=False, summary_only=False, verbose=DEFAULT_VERBOSE):
    """
    Execute power system analysis with specified parameters.
    
    Args:
        month (str, optional): Month filter (format: "YYYY-MM") or None for all data
        data_dir (str): Directory containing Excel data files
        output_dir (str): Directory to save output files
        save_csv (bool): Save analysis results to CSV files
        save_plots (bool): Generate and save plots as PNG files
        summary_only (bool): Generate only summary report
        verbose (bool): Print detailed progress information
        
    Returns:
        tuple: (success, all_power_df) where:
            - success (bool): True if analysis completed successfully, False otherwise
            - all_power_df (pandas.DataFrame or None): The merged power system dataframe 
              containing all loaded and processed data. None if analysis failed.
              This dataframe can be used for further analysis such as calling 
              extract_representative_ops().
        
    Raises:
        ValueError: If month format is invalid or data directory doesn't exist
    """
```

## Practical Examples

### Example 1: Simple Integration

```python
from power_analysis_cli import execute

def analyze_power_data():
    """Simple wrapper function for power analysis."""
    try:
        success, df = execute(month="2024-01", save_csv=True, verbose=True)
        if success:
            print("Power analysis completed successfully!")
            print(f"Processed {len(df)} time points with {len(df.columns)} variables")
            return True, df
        else:
            print("Power analysis failed!")
            return False, None
    except ValueError as e:
        print(f"Configuration error: {e}")
        return False, None
```

### Example 2: Batch Processing

```python
from power_analysis_cli import execute

def batch_monthly_analysis(months):
    """Analyze multiple months in batch."""
    results = {}
    dataframes = {}
    
    for month in months:
        try:
            success, df = execute(
                month=month,
                output_dir=f"results_{month.replace('-', '_')}",
                summary_only=True,
                verbose=False
            )
            results[month] = success
            if success:
                dataframes[month] = df
                print(f"✓ {month}: {len(df)} time points processed")
        except Exception as e:
            print(f"Error processing {month}: {e}")
            results[month] = False
    
    return results, dataframes

# Usage
months = ["2024-01", "2024-02", "2024-03"]
results, dataframes = batch_monthly_analysis(months)

# Use the dataframes for further analysis
for month, df in dataframes.items():
    print(f"{month}: {len(df)} time points available for analysis")
```

### Example 3: Integration with Workflow

```python
from power_analysis_cli import execute
import os

def automated_power_analysis_workflow():
    """Complete automated workflow with multiple analysis steps."""
    
    # Step 1: Verify data availability
    if not os.path.exists("raw_data"):
        print("Error: Data directory not found")
        return False
    
    # Step 2: Run comprehensive analysis
    print("Starting comprehensive power analysis...")
    success, all_data_df = execute(
        data_dir="raw_data",
        output_dir="automated_results",
        save_csv=True,
        save_plots=True,
        verbose=True
    )
    
    if not success:
        print("Comprehensive analysis failed")
        return False
    
    print(f"Comprehensive analysis completed: {len(all_data_df)} time points")
    
    # Step 3: Extract representative operating points from comprehensive data
    try:
        from representative_ops import extract_representative_ops
        rep_df, diag = extract_representative_ops(all_data_df, max_power=850, MAPGL=200)
        print(f"Extracted {len(rep_df)} representative operating points")
        rep_df.to_csv("automated_results/representative_operating_points.csv")
    except Exception as e:
        print(f"Warning: Representative ops extraction failed: {e}")
    
    # Step 4: Run month-specific analyses for recent months
    recent_months = ["2024-01", "2024-02", "2024-03"]
    monthly_dataframes = {}
    
    for month in recent_months:
        print(f"Analyzing {month}...")
        month_success, month_df = execute(
            month=month,
            output_dir=f"automated_results/monthly/{month}",
            save_csv=True,
            summary_only=True
        )
        
        if month_success:
            monthly_dataframes[month] = month_df
            print(f"✓ {month} completed: {len(month_df)} time points")
        else:
            print(f"Warning: Analysis for {month} failed")
    
    print("Workflow completed successfully!")
    return True
```

## Error Handling

The `execute()` function provides robust error handling:

```python
from power_analysis_cli import execute

try:
    success, df = execute(month="2024-01", data_dir="my_data")
    if success:
        print("Analysis completed!")
        print(f"Dataframe contains {len(df)} time points")
        
        # Use the dataframe for further analysis
        from representative_ops import extract_representative_ops
        rep_df, diag = extract_representative_ops(df, max_power=850, MAPGL=200)
        print(f"Representative points extracted: {len(rep_df)}")
    else:
        print("Analysis failed - check logs for details")
        
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Key Benefits

### 1. **Backward Compatibility**
- All existing command-line usage continues to work unchanged
- No breaking changes to existing workflows

### 2. **Programmatic Access**
- Direct function calls for automation
- Easy integration into larger Python applications
- Proper error handling with exceptions

### 3. **Consistent Behavior**
- Same validation and processing logic for both usage methods
- Identical output formats and file structures
- Unified logging and error reporting

### 4. **Flexible Integration**
- Suitable for batch processing workflows
- Easy to wrap in higher-level automation scripts
- Compatible with Python scripts and interactive environments

## Migration Guide

### From CLI Scripts to Python Integration

**Before (CLI script):**
```bash
#!/bin/bash
python power_analysis_cli.py 2024-01 --save-csv --save-plots --verbose
python power_analysis_cli.py 2024-02 --save-csv --save-plots --verbose
```

**After (Python script):**
```python
#!/usr/bin/env python3
from power_analysis_cli import execute

months = ["2024-01", "2024-02"]
dataframes = {}

for month in months:
    success, df = execute(month=month, save_csv=True, save_plots=True, verbose=True)
    if success:
        dataframes[month] = df
        print(f"✓ Processed {month}: {len(df)} time points")
    else:
        print(f"✗ Failed to process {month}")

# Use the dataframes for further analysis
for month, df in dataframes.items():
    print(f"Available for analysis: {month} with {len(df)} time points")
```

## Testing the New Functionality

You can test the dual functionality using the provided example script:

```bash
# Test programmatic usage
python example_usage.py
```

This will run various examples demonstrating different ways to use the `execute()` function.

## Summary

The enhanced `power_analysis_cli.py` maintains full backward compatibility while adding powerful programmatic access capabilities. Whether you prefer command-line usage or need to integrate the analysis into larger Python workflows, the tool now provides flexible options to meet your needs. 