#!/usr/bin/env python3
"""
Example script demonstrating how to use the power_analysis_cli module from another Python script.

Author: Sustainable Power Systems Lab (SPSL)
Web: https://sps-lab.org
Contact: info@sps-lab.org

This script shows various ways to call the power analysis tool programmatically
instead of using the command line interface.
"""

from power_analysis_cli import execute
import os

def example_basic_analysis():
    """Example 1: Basic analysis for all data."""
    print("Running Example 1: Basic analysis for all data")
    print("-" * 50)
    
    try:
        success, df = execute()
        if success:
            print("✓ Basic analysis completed successfully!")
            print(f"  Processed {len(df)} time points with {len(df.columns)} variables")
        else:
            print("✗ Basic analysis failed.")
    except Exception as e:
        print(f"✗ Error during basic analysis: {e}")
    
    print()

def example_month_specific_analysis():
    """Example 2: Analysis for a specific month with full outputs."""
    print("Running Example 2: Month-specific analysis with full outputs")
    print("-" * 50)
    
    try:
        success, df = execute(
            month="2024-01",
            output_dir="results_jan_2024",
            save_csv=True,
            save_plots=True,
            verbose=True
        )
        if success:
            print("✓ Month-specific analysis completed successfully!")
            print("  Check 'results_jan_2024' directory for outputs.")
            print(f"  Dataframe contains {len(df)} time points for January 2024")
            
            # Example of using the dataframe for representative operations
            try:
                from representative_ops import extract_representative_ops
                rep_df, diag = extract_representative_ops(df, max_power=850, MAPGL=200)
                print(f"  ✓ Extracted {len(rep_df)} representative operating points")
                print(f"    Optimal clusters: {diag['k']}")
            except Exception as rep_error:
                print(f"  ⚠ Representative ops extraction failed: {rep_error}")
        else:
            print("✗ Month-specific analysis failed.")
    except Exception as e:
        print(f"✗ Error during month-specific analysis: {e}")
    
    print()

def example_summary_only():
    """Example 3: Summary-only analysis."""
    print("Running Example 3: Summary-only analysis")
    print("-" * 50)
    
    try:
        success, df = execute(
            month="2024-12",
            output_dir="results_summary_only",
            summary_only=True,
            verbose=False
        )
        if success:
            print("✓ Summary-only analysis completed successfully!")
            print("  Check 'results_summary_only' directory for summary files.")
            print(f"  Dataframe available with {len(df)} time points (December 2024)")
        else:
            print("✗ Summary-only analysis failed.")
    except Exception as e:
        print(f"✗ Error during summary-only analysis: {e}")
    
    print()

def example_custom_configuration():
    """Example 4: Custom data and output directories."""
    print("Running Example 4: Custom configuration")
    print("-" * 50)
    
    # Check if a custom data directory exists, otherwise use default
    custom_data_dir = "custom_data" if os.path.exists("custom_data") else "raw_data"
    
    try:
        success, df = execute(
            data_dir=custom_data_dir,
            output_dir="custom_results",
            save_csv=True,
            verbose=True
        )
        if success:
            print("✓ Custom configuration analysis completed successfully!")
            print("  Check 'custom_results' directory for outputs.")
            print(f"  Loaded data from '{custom_data_dir}' with {len(df)} time points")
        else:
            print("✗ Custom configuration analysis failed.")
    except Exception as e:
        print(f"✗ Error during custom configuration analysis: {e}")
    
    print()

def example_batch_analysis():
    """Example 5: Batch analysis for multiple months."""
    print("Running Example 5: Batch analysis for multiple months")
    print("-" * 50)
    
    months = ["2024-01", "2024-02", "2024-03"]
    successful_months = []
    failed_months = []
    batch_dataframes = {}  # Store dataframes for successful analyses
    
    for month in months:
        try:
            print(f"  Processing {month}...")
            success, df = execute(
                month=month,
                output_dir=f"batch_results_{month.replace('-', '_')}",
                summary_only=True,  # Keep it lightweight for batch processing
                verbose=False
            )
            if success:
                successful_months.append(month)
                batch_dataframes[month] = df
                print(f"  ✓ {month} completed successfully ({len(df)} time points)")
            else:
                failed_months.append(month)
                print(f"  ✗ {month} failed")
        except Exception as e:
            failed_months.append(month)
            print(f"  ✗ {month} error: {e}")
    
    print(f"\nBatch Analysis Results:")
    print(f"  Successful: {len(successful_months)} months {successful_months}")
    print(f"  Failed: {len(failed_months)} months {failed_months}")
    
    # Example of using batch dataframes for further analysis
    if batch_dataframes:
        print(f"\nAvailable dataframes for further analysis:")
        for month, df in batch_dataframes.items():
            print(f"  {month}: {len(df)} time points, {len(df.columns)} variables")
    
    print()

def example_error_handling():
    """Example 6: Error handling demonstration."""
    print("Running Example 6: Error handling demonstration")
    print("-" * 50)
    
    # Test invalid month format
    try:
        success, df = execute(month="invalid-month")
        print("This should not be reached")
    except ValueError as e:
        print(f"✓ Correctly caught invalid month format: {e}")
    
    # Test non-existent data directory
    try:
        success, df = execute(data_dir="non_existent_directory")
        print("This should not be reached")
    except ValueError as e:
        print(f"✓ Correctly caught non-existent directory: {e}")
    
    print()

def main():
    """Run all examples."""
    print("="*60)
    print("POWER ANALYSIS CLI - PYTHON SCRIPT USAGE EXAMPLES")
    print("="*60)
    print()
    
    # Note: Uncomment the examples you want to run
    # Some examples might fail if the required data files are not available
    
    print("Note: Some examples might fail if data files are not available in 'raw_data' directory.")
    print("This is expected behavior and demonstrates proper error handling.")
    print()
    
    # Run examples (uncomment as needed)
    example_basic_analysis()
    example_month_specific_analysis()
    example_summary_only()
    example_custom_configuration()
    example_batch_analysis()
    example_error_handling()
    
    print("="*60)
    print("ALL EXAMPLES COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main() 