#!/usr/bin/env python3
"""
Clustering Methods Comparison Script

Author: Sustainable Power Systems Lab (SPSL)
Web: https://sps-lab.org
Contact: info@sps-lab.org

This script compares the standard and enhanced clustering methods for 
representative operating points extraction, providing detailed analysis
of performance improvements and method effectiveness.
"""

import pandas as pd
import numpy as np
import time
from typing import Dict, Any

def compare_clustering_methods():
    """
    Compare standard vs enhanced clustering methods for representative 
    operating points extraction.
    """
    
    print("🚀 CLUSTERING METHODS COMPARISON")
    print("="*70)
    
    # Step 1: Load data
    print("\n📊 Loading data...")
    try:
        from tsoc_data_analysis import (
            loadallpowerdf, 
            extract_representative_ops, 
            extract_representative_ops_enhanced
        )
        
        # Load the all_power data
        df = loadallpowerdf('results')
        
        print(f"✅ Data loaded successfully:")
        print(f"   📈 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"   📅 Date range: {df.index.min()} to {df.index.max()}")
        print(f"   🔍 Sample columns: {list(df.columns[:5])}...")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    # Set analysis parameters
    max_power = 850
    MAPGL = 200
    output_dir = 'results_comparison'
    
    print(f"\n⚙️ Analysis parameters:")
    print(f"   🔋 Max Power: {max_power} MW")
    print(f"   ⚡ MAPGL: {MAPGL} MW")
    print(f"   📁 Output directory: {output_dir}")
    
    # Step 2: Run standard clustering
    print(f"\n🎯 STANDARD CLUSTERING METHOD")
    print("-"*50)
    
    try:
        start_time = time.time()
        
        std_rep_df, std_diagnostics = extract_representative_ops(
            df, 
            max_power=max_power, 
            MAPGL=MAPGL, 
            output_dir=f"{output_dir}/standard"
        )
        
        std_time = time.time() - start_time
        
        print(f"✅ Standard clustering completed in {std_time:.2f} seconds")
        print(f"   📊 Clusters found: {std_diagnostics['k']}")
        print(f"   📈 Silhouette score: {std_diagnostics['silhouette']:.3f}")
        print(f"   📉 Davies-Bouldin: {std_diagnostics['db']:.3f}")
        print(f"   📊 Calinski-Harabasz: {std_diagnostics['ch']:.2f}")
        print(f"   ⚡ Representative points: {len(std_rep_df)}")
        
    except Exception as e:
        print(f"❌ Error in standard clustering: {e}")
        return None
    
    # Step 3: Run enhanced clustering
    print(f"\n🚀 ENHANCED CLUSTERING METHOD")
    print("-"*50)
    
    try:
        start_time = time.time()
        
        enh_rep_df, enh_diagnostics = extract_representative_ops_enhanced(
            df,
            max_power=max_power,
            MAPGL=MAPGL,
            output_dir=f"{output_dir}/enhanced",
            use_enhanced_preprocessing=True,
            try_alternative_algorithms=True,
            use_dimensionality_reduction=True
        )
        
        enh_time = time.time() - start_time
        
        print(f"✅ Enhanced clustering completed in {enh_time:.2f} seconds")
        print(f"   🏆 Best method: {enh_diagnostics['best_method']}")
        print(f"   📈 Best silhouette score: {enh_diagnostics['best_silhouette']:.3f}")
        print(f"   ⚡ Representative points: {len(enh_rep_df)}")
        
    except Exception as e:
        print(f"❌ Error in enhanced clustering: {e}")
        return None
    
    # Step 4: Detailed comparison
    print(f"\n📋 DETAILED COMPARISON RESULTS")
    print("="*70)
    
    # Quality comparison
    std_quality = std_diagnostics['silhouette']
    enh_quality = enh_diagnostics['best_silhouette']
    quality_improvement = ((enh_quality - std_quality) / std_quality) * 100 if std_quality > 0 else 0
    
    print(f"\n🎯 CLUSTERING QUALITY:")
    print(f"   Standard Silhouette:     {std_quality:.3f}")
    print(f"   Enhanced Silhouette:     {enh_quality:.3f}")
    if quality_improvement > 0:
        print(f"   📈 Improvement:          +{quality_improvement:.1f}% 🎉")
    elif quality_improvement < 0:
        print(f"   📉 Change:               {quality_improvement:.1f}%")
    else:
        print(f"   ➖ Change:               No improvement")
    
    # Method details
    print(f"\n🔧 METHOD DETAILS:")
    print(f"   Standard Method:         K-means")
    print(f"   Enhanced Best Method:    {enh_diagnostics['best_method']}")
    print(f"   Standard Clusters:       {std_diagnostics['k']}")
    print(f"   Enhanced Clusters:       {enh_diagnostics.get('k', 'Variable')}")
    
    # Performance comparison
    print(f"\n⏱️ PERFORMANCE:")
    print(f"   Standard Time:           {std_time:.2f} seconds")
    print(f"   Enhanced Time:           {enh_time:.2f} seconds")
    print(f"   Time Ratio:              {enh_time/std_time:.1f}x")
    
    # Data reduction comparison
    std_compression = std_diagnostics['original_size'] / std_diagnostics['n_total']
    enh_compression = enh_diagnostics['original_size'] / enh_diagnostics['n_total']
    
    print(f"\n📊 DATA REDUCTION:")
    print(f"   Original Data Points:    {std_diagnostics['original_size']:,}")
    print(f"   Standard Representatives: {std_diagnostics['n_total']} (ratio: {std_compression:.1f}:1)")
    print(f"   Enhanced Representatives: {enh_diagnostics['n_total']} (ratio: {enh_compression:.1f}:1)")
    
    # Feature engineering summary
    if 'feature_engineering_summary' in enh_diagnostics:
        fe_summary = enh_diagnostics['feature_engineering_summary']
        print(f"\n⚙️ FEATURE ENGINEERING:")
        print(f"   Original Features:       {fe_summary.get('original_features', 'N/A')}")
        print(f"   Final Features:          {fe_summary.get('final_features', 'N/A')}")
        if 'engineered_features' in fe_summary:
            print(f"   Engineered Features:     {fe_summary['engineered_features']}")
    
    # Method comparison details
    if 'method_comparison' in enh_diagnostics:
        print(f"\n🔄 METHOD COMPARISON DETAILS:")
        for method_name, method_result in enh_diagnostics['method_comparison'].items():
            if isinstance(method_result, dict) and 'silhouette' in method_result:
                status = "🏆" if method_name == enh_diagnostics['best_method'] else "  "
                print(f"   {status} {method_name:<25}: {method_result['silhouette']:.3f}")
    
    # Step 5: Quality assessment and recommendations
    print(f"\n💡 QUALITY ASSESSMENT & RECOMMENDATIONS")
    print("="*70)
    
    # Determine quality categories
    def get_quality_category(score):
        if score > 0.7:
            return "EXCELLENT", "🟢"
        elif score > 0.5:
            return "GOOD", "🟡"
        elif score > 0.25:
            return "ACCEPTABLE", "🟠"
        else:
            return "POOR", "🔴"
    
    std_category, std_emoji = get_quality_category(std_quality)
    enh_category, enh_emoji = get_quality_category(enh_quality)
    
    print(f"\n📊 QUALITY CATEGORIES:")
    print(f"   Standard Method:         {std_emoji} {std_category}")
    print(f"   Enhanced Method:         {enh_emoji} {enh_category}")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    
    if enh_quality > std_quality + 0.05:  # Significant improvement
        print(f"   ✅ RECOMMENDATION: Use Enhanced Method")
        print(f"      • Significant quality improvement achieved")
        print(f"      • Better clustering structure found")
        print(f"      • More reliable representative points")
        
    elif enh_quality > std_quality:  # Minor improvement
        print(f"   ✅ RECOMMENDATION: Consider Enhanced Method")
        print(f"      • Moderate quality improvement")
        print(f"      • Enhanced method provides better insights")
        
    elif abs(enh_quality - std_quality) < 0.02:  # Similar performance
        print(f"   ⚖️ RECOMMENDATION: Either Method Acceptable")
        print(f"      • Similar performance between methods")
        print(f"      • Choose based on computational requirements")
        print(f"      • Standard method is faster")
        
    else:  # Standard performed better
        print(f"   ⚠️ RECOMMENDATION: Use Standard Method")
        print(f"      • Standard method performed better")
        print(f"      • Enhanced features may not suit this dataset")
    
    # Data quality insights
    if enh_quality < 0.3:
        print(f"\n⚠️ DATA QUALITY INSIGHTS:")
        print(f"   • Low clustering quality suggests limited operational diversity")
        print(f"   • Consider longer data collection periods")
        print(f"   • Review system operating patterns")
        print(f"   • Validate data preprocessing steps")
    
    # Step 6: Save comparison summary
    print(f"\n💾 SAVING COMPARISON SUMMARY")
    print("-"*50)
    
    try:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Create comparison summary file
        summary_file = f"{output_dir}/clustering_comparison_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("CLUSTERING METHODS COMPARISON SUMMARY\n")
            f.write("="*60 + "\n")
            f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("QUALITY COMPARISON:\n")
            f.write(f"  Standard Silhouette:     {std_quality:.3f} ({std_category})\n")
            f.write(f"  Enhanced Silhouette:     {enh_quality:.3f} ({enh_category})\n")
            f.write(f"  Quality Improvement:     {quality_improvement:.1f}%\n\n")
            
            f.write("METHOD DETAILS:\n")
            f.write(f"  Standard Method:         K-means ({std_diagnostics['k']} clusters)\n")
            f.write(f"  Enhanced Best Method:    {enh_diagnostics['best_method']}\n\n")
            
            f.write("PERFORMANCE:\n")
            f.write(f"  Standard Time:           {std_time:.2f} seconds\n")
            f.write(f"  Enhanced Time:           {enh_time:.2f} seconds\n")
            f.write(f"  Time Ratio:              {enh_time/std_time:.1f}x\n\n")
            
            f.write("DATA REDUCTION:\n")
            f.write(f"  Original Points:         {std_diagnostics['original_size']:,}\n")
            f.write(f"  Standard Representatives: {std_diagnostics['n_total']} ({std_compression:.1f}:1)\n")
            f.write(f"  Enhanced Representatives: {enh_diagnostics['n_total']} ({enh_compression:.1f}:1)\n\n")
            
            if enh_quality > std_quality + 0.05:
                f.write("RECOMMENDATION: Use Enhanced Method (Significant Improvement)\n")
            elif enh_quality > std_quality:
                f.write("RECOMMENDATION: Consider Enhanced Method (Moderate Improvement)\n")
            else:
                f.write("RECOMMENDATION: Use Standard Method\n")
        
        print(f"✅ Comparison summary saved to: {summary_file}")
        
        # Create CSV comparison
        comparison_data = {
            'Method': ['Standard', 'Enhanced'],
            'Silhouette_Score': [std_quality, enh_quality],
            'Quality_Category': [std_category, enh_category],
            'Clusters': [std_diagnostics['k'], enh_diagnostics.get('k', 'Variable')],
            'Representative_Points': [std_diagnostics['n_total'], enh_diagnostics['n_total']],
            'Compression_Ratio': [std_compression, enh_compression],
            'Processing_Time_Seconds': [std_time, enh_time],
            'Best_Method': ['K-means', enh_diagnostics['best_method']]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        csv_file = f"{output_dir}/clustering_comparison_data.csv"
        comparison_df.to_csv(csv_file, index=False)
        
        print(f"✅ Comparison data saved to: {csv_file}")
        
    except Exception as e:
        print(f"⚠️ Warning: Could not save comparison summary: {e}")
    
    # Step 7: Final summary
    print(f"\n🎉 COMPARISON COMPLETE!")
    print("="*70)
    
    if enh_quality > std_quality + 0.05:
        print(f"🏆 WINNER: Enhanced Method (+{quality_improvement:.1f}% improvement)")
        print(f"   🎯 Best approach: {enh_diagnostics['best_method']}")
        print(f"   📈 Quality: {enh_quality:.3f} vs {std_quality:.3f}")
    elif enh_quality > std_quality:
        print(f"🥈 SLIGHT WINNER: Enhanced Method (+{quality_improvement:.1f}% improvement)")
    else:
        print(f"🥇 WINNER: Standard Method (no improvement from enhancement)")
    
    print(f"\n📁 Results saved in: {output_dir}/")
    print(f"   📊 Standard results: {output_dir}/standard/")
    print(f"   🚀 Enhanced results: {output_dir}/enhanced/")
    print(f"   📋 Comparison summary: {output_dir}/clustering_comparison_summary.txt")
    
    return {
        'standard': {'rep_df': std_rep_df, 'diagnostics': std_diagnostics, 'time': std_time},
        'enhanced': {'rep_df': enh_rep_df, 'diagnostics': enh_diagnostics, 'time': enh_time},
        'quality_improvement': quality_improvement,
        'winner': 'enhanced' if enh_quality > std_quality else 'standard'
    }


if __name__ == "__main__":
    """Run the comparison when script is executed directly."""
    
    print("Starting clustering methods comparison...")
    
    try:
        results = compare_clustering_methods()
        
        if results:
            print(f"\n✅ Comparison completed successfully!")
            print(f"🏆 Winner: {results['winner'].title()} method")
            if results['quality_improvement'] > 0:
                print(f"📈 Quality improvement: +{results['quality_improvement']:.1f}%")
        else:
            print(f"\n❌ Comparison failed. Check error messages above.")
            
    except KeyboardInterrupt:
        print(f"\n⏹️ Comparison interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}") 