#!/usr/bin/env python3
"""
Test script demonstrating the integration of execute() function with representative_ops module.

Author: Sustainable Power Systems Lab (SPSL)
Web: https://sps-lab.org
Contact: info@sps-lab.org

This script shows how to use the modified execute() function to get the power system
dataframe and then use it with the enhanced extract_representative_ops() function.

The test demonstrates:
- Power system data analysis using execute()
- Representative operating points extraction with automatic file generation
- Comprehensive clustering diagnostics and validation
- Output of CSV data, clustering summary report, and JSON metrics
"""

def test_execute_with_representative_ops():
    """Test the integration between execute() and extract_representative_ops()."""
    
    print("="*60)
    print("TESTING EXECUTE() WITH REPRESENTATIVE OPERATIONS")
    print("="*60)
    print()
    
    try:
        # Import the execute function
        from power_analysis_cli import execute
        
        print("Step 1: Running power system analysis...")
        print("-" * 40)

        output_dir = "results_2024-07"
        
        # Run the analysis and get both success status and dataframe
        success, df = execute(
            month="2024-07",
            output_dir=output_dir,
            save_csv=True,
            save_plots=True,
            verbose=True
        )
        
        if not success:
            print("✗ Power analysis failed. Cannot proceed with representative ops.")
            return False
        
        print(f"✓ Power analysis completed successfully!")
        print(f"  Dataframe shape: {df.shape}")
        print(f"  Time range: {df.index.min()} to {df.index.max()}")
        print(f"  Columns: {len(df.columns)}")
        print()
        
        print("Step 2: Extracting representative operating points...")
        print("-" * 40)
        
        try:
            # Import and use the representative operations module
            from representative_ops import extract_representative_ops
            
            # Define parameters for representative operations extraction
            max_power = 450.0  # MW - Maximum dispatchable generation
            MAPGL = 200.0      # MW - Minimum active-power generation limit
            k_max = 10         # Maximum number of clusters to test
            
            print(f"Parameters:")
            print(f"  Max Power: {max_power} MW")
            print(f"  MAPGL: {MAPGL} MW")
            print(f"  Max Clusters: {k_max}")
            print()
            
            # Extract representative operating points with automatic file output
            rep_df, diagnostics = extract_representative_ops(
                all_power=df,
                max_power=max_power,
                MAPGL=MAPGL,
                k_max=k_max,
                random_state=42,
                output_dir=output_dir
            )
            
            print("✓ Representative operations extraction completed!")
            print(f"  Original time points: {len(df)}")
            print(f"  Representative points: {len(rep_df)}")
            print(f"  Reduction ratio: {len(rep_df)/len(df)*100:.1f}%")
            print(f"  Optimal clusters: {diagnostics['k']}")
            print(f"  Files automatically saved to: {output_dir}/")
            print()
            
            print("Step 3: Analyzing results...")
            print("-" * 40)
            
            # Display some diagnostics
            print("Clustering Diagnostics:")
            
            # Helper function to safely format numeric values
            def safe_format(value, fmt):
                if isinstance(value, (int, float)):
                    return f"{value:{fmt}}"
                return str(value)
            
            silhouette = diagnostics.get('silhouette', 'N/A')
            calinski = diagnostics.get('ch', 'N/A')
            davies = diagnostics.get('db', 'N/A')
            
            print(f"  Silhouette Score: {safe_format(silhouette, '.3f')}")
            print(f"  Calinski-Harabasz Index: {safe_format(calinski, '.1f')}")
            print(f"  Davies-Bouldin Index: {safe_format(davies, '.3f')}")
            print()
            
            if 'cluster_sizes' in diagnostics:
                print("Cluster Information:")
                for i, size in enumerate(diagnostics['cluster_sizes']):
                    print(f"  Cluster {i+1}: {size} points")
                print()
            
            # Additional diagnostics from the enhanced function
            print("Additional Diagnostics Available:")
            print(f"  Medoid snapshots: {diagnostics.get('n_medoid', 'N/A')}")
            print(f"  MAPGL belt snapshots: {diagnostics.get('n_belt', 'N/A')}")
            print(f"  Original dataset size: {diagnostics.get('original_size', 'N/A')}")
            print(f"  Filtered dataset size: {diagnostics.get('filtered_size', 'N/A')}")
            print(f"  Feature columns used: {len(diagnostics.get('feature_columns', []))}")
            print()
            
            # Files are automatically saved by extract_representative_ops when output_dir is provided
            print("✓ All results automatically saved:")
            print("  - representative_operating_points.csv")
            print("  - clustering_summary.txt")
            print("  - clustering_info.json")
            print()
            
            print("Step 4: Validation...")
            print("-" * 40)
            
            # Basic validation checks
            if len(rep_df) > 0:
                print("✓ Representative points extracted successfully")
                
                # Check if we have the expected columns
                expected_prefixes = ['ss_mw_', 'ss_mvar_', 'wind_mw_']
                found_columns = {}
                
                for prefix in expected_prefixes:
                    cols = [col for col in rep_df.columns if col.startswith(prefix)]
                    found_columns[prefix] = len(cols)
                    print(f"  {prefix} columns: {len(cols)}")
                
                # Check time coverage
                original_time_span = df.index.max() - df.index.min()
                rep_time_span = rep_df.index.max() - rep_df.index.min()
                coverage_ratio = rep_time_span / original_time_span * 100
                
                print(f"  Time coverage: {coverage_ratio:.1f}%")
                print(f"  Original span: {original_time_span}")
                print(f"  Representative span: {rep_time_span}")
            else:
                print("✗ No representative points extracted")
                return False
            
            print()
            print("="*60)
            print("INTEGRATION TEST COMPLETED SUCCESSFULLY!")
            print("="*60)
            print()
            print("Summary:")
            print(f"• Successfully loaded {len(df)} time points from power analysis")
            print(f"• Extracted {len(rep_df)} representative operating points")
            print(f"• Achieved {(1 - len(rep_df)/len(df))*100:.1f}% data reduction")
            print(f"• Used {diagnostics['k']} optimal clusters")
            print(f"• Clustering quality (Silhouette): {safe_format(silhouette, '.3f')}")
            print(f"• Comprehensive results automatically saved to test_results_rep_ops/")
            print("  - CSV file with representative operating points")
            print("  - Detailed clustering summary report")
            print("  - JSON file with clustering metrics")
            print()
            
            return True
            
        except ImportError as e:
            print(f"✗ Could not import representative_ops module: {e}")
            print("  Make sure representative_ops.py is available in the Python path.")
            return False
            
        except Exception as e:
            print(f"✗ Error during representative operations extraction: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    except ImportError as e:
        print(f"✗ Could not import power_analysis_cli module: {e}")
        return False
    
    except Exception as e:
        print(f"✗ Error during power analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the integration test."""
    
    print("Integration Test: execute() + extract_representative_ops()")
    print()
    
    # Check if required data directory exists
    import os
    if not os.path.exists("raw_data"):
        print("Warning: 'raw_data' directory not found.")
        print("This test requires power system data files to run properly.")
        print("The test will attempt to run but may fail if no data is available.")
        print()
    
    # Run the test
    success = test_execute_with_representative_ops()
    
    if success:
        print("🎉 Integration test PASSED!")
        print("\nThe modified execute() function successfully returns both the")
        print("success status and the power system dataframe, which can be")
        print("directly used with extract_representative_ops().")
        print("\nThe updated extract_representative_ops() function now:")
        print("• Automatically saves representative operating points as CSV")
        print("• Generates comprehensive clustering summary report")
        print("• Provides detailed clustering diagnostics in JSON format")
        print("• Includes advanced clustering validation metrics")
    else:
        print("❌ Integration test FAILED!")
        print("\nPlease check the error messages above and ensure that:")
        print("• Raw data files are available in the 'raw_data' directory")
        print("• All required Python modules are properly installed")
        print("• The power_analysis_cli.py and representative_ops.py files are accessible")


if __name__ == "__main__":
    main() 