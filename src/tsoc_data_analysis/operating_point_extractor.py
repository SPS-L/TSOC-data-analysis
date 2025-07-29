"""
Representative Operating Points Extraction Module

Author: Sustainable Power Systems Lab (SPSL)
Web: https://sps-lab.org
Contact: info@sps-lab.org

This module provides functions to extract representative operating points 
from power system data using K-means clustering with automatic cluster count selection.

CLUSTERING METHODOLOGY:
======================

The module implements the methodology described in "Automated Extraction of 
Representative Operating Points for a 132 kV Transmission System":

1. DATA FILTERING: Based on power limits and MAPGL constraints
2. FEATURE EXTRACTION: Uses power injection features (ss_mw_*, ss_mvar_*, wind_mw_*)
3. STANDARDIZATION: Applies StandardScaler for feature normalization
4. CLUSTERING: K-means with automatic cluster count selection using multiple metrics
5. MEDOID SELECTION: Returns actual snapshots closest to cluster centers
6. MAPGL BELT: Includes critical low-load operating points near MAPGL threshold
7. OUTPUT GENERATION: Saves results with clean column names and comprehensive reports

CONFIGURATION INTEGRATION:
=========================

All parameters are imported from config.REPRESENTATIVE_OPS:

- defaults: k_max, random_state, mapgl_belt_multiplier, fallback_clusters
- kmeans: n_init, algorithm settings
- quality_thresholds: min_silhouette, excellent/good/acceptable thresholds  
- ranking_weights: Multi-objective ranking weights for cluster selection
- feature_columns: Column prefixes for clustering features
- output_files: Standardized output file names
- validation: Display limits and data requirements

ENHANCED FEATURES:
=================

- CLEAN OUTPUT: Uses centralized clean_column_name() for readable CSV files
- QUALITY ASSESSMENT: Multi-objective cluster quality evaluation
- COMPREHENSIVE REPORTING: Detailed clustering summary with diagnostics
- ADAPTIVE ALGORITHMS: Automatically adjusts to data characteristics
- POWER SYSTEM FOCUS: Specialized for electrical power system analysis

Functions:
- loadallpowerdf(): Load all_power*.csv files from directory
- extract_representative_ops(): Main function to extract representative points
- _select_feature_columns(): Helper to identify clustering features (config-driven)
- _auto_kmeans(): Helper for automatic K-means cluster selection (config-driven)
- _create_clustering_summary(): Generates comprehensive analysis reports

USAGE EXAMPLES:
==============

# Load all_power data from directory
df = loadallpowerdf('results')

# Basic usage with default configuration
rep_df, diagnostics = extract_representative_ops(
    all_power=df, max_power=850, MAPGL=200
)

# Save results with automatic file naming
rep_df, diagnostics = extract_representative_ops(
    all_power=df, max_power=850, MAPGL=200, 
    output_dir="results"  # Uses config file names
)

# Combined workflow: load data and extract representative points
rep_df, diagnostics = extract_representative_ops(
    loadallpowerdf('results'), max_power=850, MAPGL=200, output_dir='results'
)

# Custom parameters (overrides config defaults)
rep_df, diagnostics = extract_representative_ops(
    all_power=df, max_power=850, MAPGL=200,
    k_max=15, random_state=123  # Override config values
)
"""

from __future__ import annotations
from typing import Tuple, Dict, Optional

import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from .system_configuration import clean_column_name, REPRESENTATIVE_OPS, convert_numpy_types

__all__ = ["extract_representative_ops", "loadallpowerdf"]


def loadallpowerdf(directory: str) -> pd.DataFrame:
    """
    Load all_power*.csv file from the specified directory into a DataFrame.
    
    This function searches for files matching the pattern 'all_power*.csv' in the
    specified directory and loads the first matching file. It's designed to work
    with the power system analysis workflow where all_power data files are
    generated with timestamps or version suffixes.
    
    Parameters
    ----------
    directory : str
        Directory path to search for all_power*.csv files.
        
    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with power system data. The index is preserved as
        timestamps if present in the original CSV file.
        
    Raises
    ------
    FileNotFoundError
        If no all_power*.csv file is found in the specified directory.
    ValueError
        If the directory doesn't exist or is not accessible.
        
    Examples
    --------
    >>> # Load all_power data from results directory
    >>> df = loadallpowerdf('results')
    >>> print(f"Loaded {len(df)} snapshots with {len(df.columns)} columns")
    
    >>> # Use in representative operating points extraction
    >>> rep_df, diagnostics = extract_representative_ops(
    ...     loadallpowerdf('results'), max_power=850, MAPGL=200, output_dir='results'
    ... )
    
    Notes
    -----
    - Searches for files matching pattern 'all_power*.csv'
    - Uses pandas.read_csv() with automatic index parsing
    - Preserves original column names and data types
    - Handles common CSV formats with comma or semicolon separators
    """
    import glob
    
    # Check if directory exists
    if not os.path.exists(directory):
        raise ValueError(f"Directory '{directory}' does not exist")
    
    if not os.path.isdir(directory):
        raise ValueError(f"'{directory}' is not a directory")
    
    # Search for all_power*.csv files
    pattern = os.path.join(directory, "all_power*.csv")
    matching_files = glob.glob(pattern)
    
    if not matching_files:
        raise FileNotFoundError(
            f"No all_power*.csv files found in directory '{directory}'. "
            f"Searched pattern: {pattern}"
        )
    
    # Use the first matching file (most recent if sorted)
    file_path = sorted(matching_files)[0]
    
    try:
        # Try to read with automatic index parsing (for timestamps)
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    except (ValueError, TypeError):
        # Fallback: read without index parsing if it fails
        try:
            df = pd.read_csv(file_path, index_col=0)
        except (ValueError, TypeError):
            # Final fallback: read without index
            df = pd.read_csv(file_path)
    
    print(f"Loaded all_power data from: {file_path}")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
    
    return df


def _select_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return columns starting with ss_mw_, ss_mvar_ or wind_mw_."""
    keep_prefix = tuple(REPRESENTATIVE_OPS['feature_columns']['clustering_prefixes'])
    return [c for c in df.columns if c.startswith(keep_prefix)]


def _auto_kmeans(
    x: np.ndarray,
    k_max: int = REPRESENTATIVE_OPS['defaults']['k_max'],
    random_state: int | None = REPRESENTATIVE_OPS['defaults']['random_state'],
) -> Tuple[KMeans, Dict[str, float]]:
    """Fit k-means while automatically selecting k with performance optimizations."""
    best_model: KMeans | None = None
    best_score: float = -np.inf
    best_k: int = 0
    best_metrics: dict[str, float] = {}

    # Optimize k range based on data size
    n_samples = len(x)
    k_max = min(k_max, n_samples - 1, 20)  # Cap at 20 for performance
    k_min = max(2, min(5, n_samples // 100))  # Adaptive minimum k
    
    # Use parallel processing for multiple k values
    from joblib import Parallel, delayed
    
    def evaluate_k(k):
        try:
            km = KMeans(
                n_clusters=k, 
                random_state=random_state, 
                n_init=REPRESENTATIVE_OPS['kmeans']['n_init']
            )
            labels = km.fit_predict(x)

            sil = silhouette_score(x, labels)
            ch = calinski_harabasz_score(x, labels)
            db = davies_bouldin_score(x, labels)

            # Multi-objective ranking: maximise CH & Sil, minimise DB
            weights = REPRESENTATIVE_OPS['ranking_weights']
            combo = (ch * weights['calinski_harabasz_weight'] + 
                    sil * weights['silhouette_weight'] - 
                    db * weights['davies_bouldin_weight'])
            
            min_silhouette = REPRESENTATIVE_OPS['quality_thresholds']['min_silhouette']
            if sil > min_silhouette:
                return k, combo, km, {"silhouette": sil, "ch": ch, "db": db}
            return k, -np.inf, None, {}
        except Exception:
            return k, -np.inf, None, {}

    # Evaluate k values in parallel
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(evaluate_k)(k) for k in range(k_min, k_max + 1)
    )
    
    # Find best result
    for k, score, model, metrics in results:
        if score > best_score and model is not None:
            best_score = score
            best_model = model
            best_k = k
            best_metrics = metrics

    if best_model is None:  # fall-back to default clusters
        fallback_k = REPRESENTATIVE_OPS['defaults']['fallback_clusters']
        best_model = KMeans(
            n_clusters=fallback_k, 
            random_state=random_state, 
            n_init=REPRESENTATIVE_OPS['kmeans']['n_init']
        ).fit(x)
        best_k = fallback_k
        labels = best_model.labels_
        best_metrics = {
            "silhouette": silhouette_score(x, labels),
            "ch": calinski_harabasz_score(x, labels),
            "db": davies_bouldin_score(x, labels),
        }

    best_metrics["k"] = best_k
    return best_model, best_metrics


def _create_visualizations(
    output_dir: str,
    working: pd.DataFrame,
    rep_df: pd.DataFrame,
    info: dict,
    model: KMeans,
    scaler: StandardScaler,
    feat_cols: list,
    max_power: float,
    MAPGL: float,
) -> None:
    """Create comprehensive visualizations for clustering analysis."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.patches import Rectangle
        import warnings
        warnings.filterwarnings('ignore')
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Clustering Quality Metrics Dashboard
        ax1 = plt.subplot(3, 3, 1)
        metrics = ['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin']
        values = [info['silhouette'], info['ch'], info['db']]
        colors = ['green' if info['silhouette'] > 0.5 else 'orange' if info['silhouette'] > 0.25 else 'red',
                 'green' if info['ch'] > 100 else 'orange' if info['ch'] > 50 else 'red',
                 'green' if info['db'] < 0.5 else 'orange' if info['db'] < 1.0 else 'red']
        
        bars = ax1.bar(metrics, values, color=colors, alpha=0.7)
        ax1.set_title('Clustering Quality Metrics', fontweight='bold')
        ax1.set_ylabel('Score')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Cluster Size Distribution
        ax2 = plt.subplot(3, 3, 2)
        cluster_sizes = info['cluster_sizes']
        cluster_labels = [f'C{i+1}' for i in range(len(cluster_sizes))]
        colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_sizes)))
        
        wedges, texts, autotexts = ax2.pie(cluster_sizes, labels=cluster_labels, 
                                          colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Cluster Size Distribution', fontweight='bold')
        
        # 3. Net Load Distribution
        ax3 = plt.subplot(3, 3, 3)
        if 'net_load' in working.columns:
            # Original vs Representative
            ax3.hist(working['net_load'], bins=30, alpha=0.6, label='Original', density=True)
            ax3.hist(rep_df['net_load'], bins=15, alpha=0.8, label='Representative', density=True)
            ax3.axvline(MAPGL, color='red', linestyle='--', label=f'MAPGL ({MAPGL} MW)')
            ax3.axvline(max_power, color='red', linestyle='--', label=f'Max Power ({max_power} MW)')
            ax3.set_xlabel('Net Load (MW)')
            ax3.set_ylabel('Density')
            ax3.set_title('Net Load Distribution', fontweight='bold')
            ax3.legend()
        
        # 4. Feature Importance (Variance)
        ax4 = plt.subplot(3, 3, 4)
        if len(feat_cols) > 0:
            feature_vars = working[feat_cols].var().sort_values(ascending=False)
            top_features = feature_vars.head(10)
            feature_names = [col.replace('ss_mw_', '').replace('ss_mvar_', '').replace('wind_mw_', '') 
                           for col in top_features.index]
            
            bars = ax4.barh(range(len(top_features)), top_features.values, alpha=0.7)
            ax4.set_yticks(range(len(top_features)))
            ax4.set_yticklabels(feature_names, fontsize=8)
            ax4.set_xlabel('Variance')
            ax4.set_title('Top 10 Features by Variance', fontweight='bold')
            ax4.invert_yaxis()
        
        # 5. Compression Analysis
        ax5 = plt.subplot(3, 3, 5)
        categories = ['Original', 'Filtered', 'Representative']
        sizes = [info['original_size'], info['filtered_size'], info['n_total']]
        colors = ['lightblue', 'lightgreen', 'orange']
        
        bars = ax5.bar(categories, sizes, color=colors, alpha=0.7)
        ax5.set_title('Data Reduction Analysis', fontweight='bold')
        ax5.set_ylabel('Number of Snapshots')
        
        # Add percentage labels
        for bar, size in zip(bars, sizes):
            height = bar.get_height()
            percentage = (size / info['original_size']) * 100
            ax5.text(bar.get_x() + bar.get_width()/2., height + max(sizes)*0.01,
                    f'{percentage:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 6. MAPGL Belt Analysis
        ax6 = plt.subplot(3, 3, 6)
        if 'net_load' in working.columns:
            mapgl_multiplier = REPRESENTATIVE_OPS['defaults']['mapgl_belt_multiplier']
            belt_mask = (working["net_load"] > MAPGL) & (working["net_load"] < mapgl_multiplier * MAPGL)
            
            # Create histogram with MAPGL belt highlighted
            n, bins, patches = ax6.hist(working['net_load'], bins=50, alpha=0.6, color='lightblue')
            
            # Highlight MAPGL belt
            belt_indices = np.where((bins[:-1] >= MAPGL) & (bins[1:] <= mapgl_multiplier * MAPGL))[0]
            for idx in belt_indices:
                patches[idx].set_facecolor('red')
                patches[idx].set_alpha(0.8)
            
            ax6.axvline(MAPGL, color='red', linestyle='--', linewidth=2, label=f'MAPGL ({MAPGL} MW)')
            ax6.axvline(mapgl_multiplier * MAPGL, color='red', linestyle='--', linewidth=2, 
                       label=f'MAPGL Belt Upper ({mapgl_multiplier * MAPGL:.1f} MW)')
            ax6.set_xlabel('Net Load (MW)')
            ax6.set_ylabel('Frequency')
            ax6.set_title('MAPGL Belt Analysis', fontweight='bold')
            ax6.legend()
        
        # 7. Cluster Centers Heatmap (if not too many features)
        ax7 = plt.subplot(3, 3, 7)
        if len(feat_cols) <= 20:
            cluster_centers_orig = scaler.inverse_transform(model.cluster_centers_)
            feature_names_short = [col.replace('ss_mw_', '').replace('ss_mvar_', '').replace('wind_mw_', '') 
                                 for col in feat_cols]
            
            im = ax7.imshow(cluster_centers_orig.T, cmap='RdYlBu_r', aspect='auto')
            ax7.set_xticks(range(model.n_clusters))
            ax7.set_xticklabels([f'C{i+1}' for i in range(model.n_clusters)])
            ax7.set_yticks(range(len(feature_names_short)))
            ax7.set_yticklabels(feature_names_short, fontsize=8)
            ax7.set_title('Cluster Centers Heatmap', fontweight='bold')
            plt.colorbar(im, ax=ax7, label='Feature Value')
        
        # 8. Quality Assessment Summary
        ax8 = plt.subplot(3, 3, 8)
        ax8.axis('off')
        
        # Determine overall quality
        silhouette_val = info.get('silhouette', 0)
        compression_ratio = info['original_size'] / info['n_total']
        
        if silhouette_val > 0.7 and compression_ratio > 20:
            overall_quality = "EXCELLENT"
            quality_color = "green"
            quality_emoji = "🟢"
        elif silhouette_val > 0.5 and compression_ratio > 10:
            overall_quality = "GOOD"
            quality_color = "orange"
            quality_emoji = "🟡"
        elif silhouette_val > 0.25:
            overall_quality = "ACCEPTABLE"
            quality_color = "red"
            quality_emoji = "🟠"
        else:
            overall_quality = "POOR"
            quality_color = "darkred"
            quality_emoji = "🔴"
        
        summary_text = f"""
{quality_emoji} OVERALL QUALITY: {overall_quality}

📊 CLUSTERING METRICS:
• Silhouette Score: {silhouette_val:.3f}
• Calinski-Harabasz: {info['ch']:.1f}
• Davies-Bouldin: {info['db']:.3f}
• Optimal Clusters: {info['k']}

📈 DATA REDUCTION:
• Original: {info['original_size']:,} snapshots
• Representative: {info['n_total']} snapshots
• Compression: {compression_ratio:.1f}:1
• Retention: {(info['n_total']/info['original_size'])*100:.1f}%

⚡ REPRESENTATIVE POINTS:
• Medoids: {info['n_medoid']}
• MAPGL Belt: {info['n_belt']}
• Total: {info['n_total']}
        """
        
        ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=quality_color, alpha=0.1))
        
        # 9. Recommendations
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        recommendations = []
        if silhouette_val < 0.25:
            recommendations.append("⚠️ Consider increasing dataset size")
            recommendations.append("⚠️ Review feature selection")
            recommendations.append("⚠️ Check data quality")
        elif silhouette_val < 0.5:
            recommendations.append("⚠️ Validate results carefully")
            recommendations.append("⚠️ Consider parameter adjustment")
        
        if compression_ratio < 5:
            recommendations.append("ℹ️ Low data reduction - high diversity")
        
        if info['n_belt'] == 0:
            recommendations.append("ℹ️ No MAPGL belt snapshots found")
        
        if not recommendations:
            recommendations.append("✅ Results look good for analysis")
            recommendations.append("✅ Proceed with power system studies")
        
        rec_text = "RECOMMENDATIONS:\n\n" + "\n".join(recommendations)
        ax9.text(0.05, 0.95, rec_text, transform=ax9.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        
        # Save the visualization
        output_files = REPRESENTATIVE_OPS['output_files']
        viz_filename = os.path.join(output_dir, 'clustering_visualization.png')
        plt.savefig(viz_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Visualization: {viz_filename}")
        
    except ImportError:
        print("Warning: matplotlib/seaborn not available. Skipping visualizations.")
    except Exception as e:
        print(f"Warning: Could not create visualizations: {e}")


def _create_clustering_summary(
    filename: str,
    all_power: pd.DataFrame,
    working: pd.DataFrame,
    rep_df: pd.DataFrame,
    info: dict,
    max_power: float,
    MAPGL: float,
    k_max: int,
    random_state: int,
    feat_cols: list,
    model: KMeans,
    scaler: StandardScaler,
) -> None:
    """Create a comprehensive clustering summary report with improved formatting."""
    with open(filename, "w", encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("REPRESENTATIVE OPERATING POINTS CLUSTERING SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Author: Sustainable Power Systems Lab (SPSL)\n")
        f.write(f"Web: https://sps-lab.org\n")
        f.write(f"Contact: info@sps-lab.org\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # EXECUTIVE SUMMARY
        f.write("📋 EXECUTIVE SUMMARY\n")
        f.write("="*50 + "\n")
        
        # Determine overall quality
        silhouette_val = info.get('silhouette', 0)
        compression_ratio = info['original_size'] / info['n_total']
        
        if silhouette_val > 0.7 and compression_ratio > 20:
            overall_quality = "EXCELLENT"
            quality_emoji = "🟢"
        elif silhouette_val > 0.5 and compression_ratio > 10:
            overall_quality = "GOOD"
            quality_emoji = "🟡"
        elif silhouette_val > 0.25:
            overall_quality = "ACCEPTABLE"
            quality_emoji = "🟠"
        else:
            overall_quality = "POOR"
            quality_emoji = "🔴"
        
        f.write(f"{quality_emoji} Overall Quality: {overall_quality}\n")
        f.write(f"📊 Clustering Score: {silhouette_val:.3f} (Silhouette)\n")
        f.write(f"📈 Data Reduction: {compression_ratio:.1f}:1 ({((compression_ratio-1)/compression_ratio)*100:.1f}% reduction)\n")
        f.write(f"⚡ Representative Points: {info['n_total']} from {info['original_size']:,} original\n")
        f.write(f"🎯 Optimal Clusters: {info['k']}\n\n")
        
        # 1. METHODOLOGY OVERVIEW
        f.write("1. 📚 METHODOLOGY OVERVIEW\n")
        f.write("-"*30 + "\n")
        f.write("This analysis implements the methodology described in:\n")
        f.write("'Automated Extraction of Representative Operating Points for a 132 kV Transmission System'\n\n")
        f.write("🔄 Process Steps:\n")
        f.write("   1️⃣ Data filtering based on power limits and MAPGL constraints\n")
        f.write("   2️⃣ Feature extraction from power injection variables\n")
        f.write("   3️⃣ Data standardization using StandardScaler\n")
        f.write("   4️⃣ K-means clustering with automatic cluster count selection\n")
        f.write("   5️⃣ Medoid identification (actual snapshots closest to cluster centers)\n")
        f.write("   6️⃣ Addition of MAPGL-belt snapshots for critical low-load conditions\n\n")
        
        # 2. INPUT PARAMETERS
        f.write("2. ⚙️ INPUT PARAMETERS\n")
        f.write("-"*30 + "\n")
        f.write(f"🔋 Maximum Power Limit:        {max_power:.2f} MW\n")
        f.write(f"⚡ MAPGL (Min Generation):     {MAPGL:.2f} MW\n")
        f.write(f"🎯 Maximum Clusters Tested:    {k_max}\n")
        f.write(f"🎲 Random State:               {random_state}\n")
        mapgl_multiplier = REPRESENTATIVE_OPS['defaults']['mapgl_belt_multiplier']
        f.write(f"📏 MAPGL Belt Range:           {MAPGL:.2f} - {mapgl_multiplier*MAPGL:.2f} MW\n\n")
        
        # 3. DATA PROCESSING SUMMARY
        f.write("3. 📊 DATA PROCESSING SUMMARY\n")
        f.write("-"*30 + "\n")
        f.write(f"📁 Original Dataset Size:      {info['original_size']:,} snapshots\n")
        f.write(f"🔍 After Power Filtering:      {info['filtered_size']:,} snapshots\n")
        f.write(f"📉 Reduction Factor:           {info['original_size']/info['filtered_size']:.2f}x\n\n")
        
        # Data quality information
        if 'data_quality' in info:
            f.write("🔍 Data Quality Assessment:\n")
            f.write(f"   • Missing Values: {info['data_quality']['missing_values']}\n")
            f.write(f"   • Infinite Values: {info['data_quality']['infinite_values']}\n")
            f.write(f"   • Zero Variance Features Excluded: {info['data_quality']['zero_variance_features_excluded']}\n\n")
        
        f.write("🎯 Feature Columns Used for Clustering:\n")
        for i, col in enumerate(feat_cols, 1):
            f.write(f"   {i:2d}. {col}\n")
        f.write(f"\n📊 Total Features:             {len(feat_cols)}\n\n")
        
        # 4. CLUSTERING RESULTS
        f.write("4. 🎯 CLUSTERING RESULTS\n")
        f.write("-"*30 + "\n")
        f.write(f"🏆 Optimal Number of Clusters: {info['k']}\n")
        f.write(f"📈 Silhouette Score:           {info['silhouette']:.4f}\n")
        f.write(f"📊 Calinski-Harabasz Index:    {info['ch']:.2f}\n")
        f.write(f"📉 Davies-Bouldin Index:       {info['db']:.4f}\n\n")
        
        f.write("📊 Cluster Composition:\n")
        for i, size in enumerate(info['cluster_sizes']):
            percentage = (size / info['filtered_size']) * 100
            f.write(f"   Cluster {i+1:2d}: {size:6,} snapshots ({percentage:5.1f}%)\n")
        f.write(f"   Total:       {sum(info['cluster_sizes']):6,} snapshots (100.0%)\n\n")
        
        # 5. REPRESENTATIVE POINTS SELECTION
        f.write("5. ⚡ REPRESENTATIVE POINTS SELECTION\n")
        f.write("-"*30 + "\n")
        f.write(f"🎯 Medoids from Clusters:      {info['n_medoid']}\n")
        f.write(f"📏 MAPGL Belt Snapshots:       {info['n_belt']}\n")
        f.write(f"📊 Total Representative Points: {info['n_total']}\n")
        f.write(f"📉 Compression Ratio:          {info['original_size']/info['n_total']:.1f}:1\n")
        f.write(f"📈 Retention Rate:             {(info['n_total']/info['original_size'])*100:.2f}%\n\n")
        
        # 6. QUALITY ASSESSMENT
        f.write("6. ✅ QUALITY ASSESSMENT\n")
        f.write("-"*30 + "\n")
        
        f.write("📊 Silhouette Score Analysis:\n")
        if silhouette_val > 0.7:
            f.write("   ✅ Excellent separation: Clusters are well-defined and distinct\n")
            f.write("   ✅ Representative points are highly reliable\n")
            f.write("   ✅ Strong confidence in operating point selection\n")
        elif silhouette_val > 0.5:
            f.write("   ✅ Good separation: Clusters are reasonably well-defined\n")
            f.write("   ✅ Representative points are reliable\n")
            f.write("   ⚠️ Minor overlap between some clusters is acceptable\n")
        elif silhouette_val > 0.25:
            f.write("   ⚠️ Moderate separation: Some cluster overlap present\n")
            f.write("   ⚠️ Representative points should be validated carefully\n")
            f.write("   ⚠️ Consider additional validation of selected points\n")
        else:
            f.write("   ❌ Poor separation: Significant cluster overlap\n")
            f.write("   ❌ Representative points may not be reliable\n")
            f.write("   ❌ Strong recommendation to revise approach\n")
        
        f.write(f"\n📈 Calinski-Harabasz Index Analysis (Current: {info['ch']:.1f}):\n")
        if info['ch'] > 100:
            f.write("   ✅ High index: Well-separated, compact clusters\n")
            f.write("   ✅ Strong internal cluster cohesion\n")
        elif info['ch'] > 50:
            f.write("   ✅ Moderate index: Reasonably good cluster structure\n")
        else:
            f.write("   ⚠️ Low index: Clusters may be poorly separated or too diffuse\n")
        
        f.write(f"\n📉 Davies-Bouldin Index Analysis (Current: {info['db']:.3f}):\n")
        if info['db'] < 0.5:
            f.write("   ✅ Excellent: Very low similarity between clusters\n")
        elif info['db'] < 1.0:
            f.write("   ✅ Good: Low similarity between clusters\n")
        elif info['db'] < 1.5:
            f.write("   ⚠️ Acceptable: Moderate similarity between clusters\n")
        else:
            f.write("   ❌ Poor: High similarity between clusters indicates overlap\n")
        f.write("\n")
        
        # 7. RECOMMENDATIONS
        f.write("7. 💡 RECOMMENDATIONS AND NEXT STEPS\n")
        f.write("-"*30 + "\n")
        
        if silhouette_val < 0.25:
            f.write("⚠️ LOW CLUSTERING QUALITY WARNING:\n")
            f.write("   Consider:\n")
            f.write("   • Increasing the dataset size\n")
            f.write("   • Reviewing feature selection\n")
            f.write("   • Adjusting power limits or MAPGL\n\n")
        
        if info['n_total'] < 10:
            f.write("⚠️ FEW REPRESENTATIVE POINTS:\n")
            f.write("   Consider lowering k_max or adjusting clustering parameters\n\n")
        
        f.write("🔧 For power system analysis:\n")
        f.write("   1. Validate representative points against operational constraints\n")
        f.write("   2. Verify load flow convergence for all selected snapshots\n")
        f.write("   3. Check that critical operating conditions are captured\n")
        f.write("   4. Consider seasonal or temporal patterns if relevant\n\n")
        
        # 8. USAGE RECOMMENDATIONS
        f.write("8. 🎯 RECOMMENDED USAGE BASED ON RESULTS\n")
        f.write("-"*30 + "\n")
        
        if overall_quality in ["EXCELLENT", "GOOD"]:
            f.write("✅ HIGH CONFIDENCE APPLICATIONS:\n")
            f.write("   • Long-term transmission planning studies\n")
            f.write("   • Investment analysis and capacity expansion\n")
            f.write("   • Security assessment and contingency analysis\n")
            f.write("   • Renewable integration studies\n")
            f.write("   • Grid code compliance verification\n\n")
        
        if overall_quality in ["ACCEPTABLE"]:
            f.write("⚠️ MODERATE CONFIDENCE APPLICATIONS:\n")
            f.write("   • Preliminary planning studies (with validation)\n")
            f.write("   • Scenario development for detailed analysis\n")
            f.write("   • Identification of critical operating conditions\n")
            f.write("   ⚠️ Recommend additional validation before final decisions\n\n")
        
        if overall_quality in ["POOR"]:
            f.write("❌ LIMITED CONFIDENCE APPLICATIONS:\n")
            f.write("   • Initial screening studies only\n")
            f.write("   • Pattern identification in operational data\n")
            f.write("   • Troubleshooting and methodology development\n")
            f.write("   ❌ NOT recommended for critical planning decisions\n\n")
        
        f.write("="*80 + "\n")
        f.write("END OF CLUSTERING SUMMARY\n")
        f.write("="*80 + "\n")


def extract_representative_ops(
    all_power: pd.DataFrame,
    max_power: float,
    MAPGL: float,
    k_max: int = REPRESENTATIVE_OPS['defaults']['k_max'],
    random_state: int = REPRESENTATIVE_OPS['defaults']['random_state'],
    output_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Extract representative operating points from power system data using clustering.

    This function implements the methodology described in "Automated Extraction of 
    Representative Operating Points for a 132 kV Transmission System" with a 
    configuration-driven approach for enhanced maintainability and customization.

    CONFIGURATION-DRIVEN FEATURES:
    ==============================
    - All clustering parameters imported from config.REPRESENTATIVE_OPS
    - Default values managed centrally for consistency across analyses
    - Feature column selection based on config.REPRESENTATIVE_OPS['feature_columns']
    - Output file names standardized via config.REPRESENTATIVE_OPS['output_files']
    - Clean column names in CSV output using config.clean_column_name()

    CLUSTERING WORKFLOW:
    ===================
    1. Data filtering based on power limits and MAPGL constraints
    2. Feature extraction using power injection variables (config-driven prefixes)
    3. Data standardization using StandardScaler
    4. K-means clustering with automatic cluster count selection
    5. Multi-objective quality assessment (silhouette, Calinski-Harabasz, Davies-Bouldin)
    6. Medoid identification (actual snapshots closest to cluster centers)
    7. MAPGL belt analysis for critical low-load operating points
    8. Results saved with clean column names and comprehensive diagnostics

    NET LOAD HANDLING:
    ==================
    - If 'net_load' column exists in input data, it will be used directly
    - If 'net_load' is missing, it will be calculated from power system data
    - This avoids redundant calculations when data already contains computed values

    Parameters
    ----------
    all_power : pandas.DataFrame
        Input time-series DataFrame with power system data including columns:
        - ss_mw_*: Substation active power (MW)
        - ss_mvar_*: Substation reactive power (MVAR) 
        - wind_mw_*: Wind farm active power (MW)
        - net_load: Net load (optional, will be calculated if not present)
        - total_load: Total load (optional, will be calculated if net_load not present)
    max_power : float
        Maximum dispatchable generation for the horizon under study [MW].
    MAPGL : float
        Minimum active-power generation limit [MW].
    k_max : int, optional
        Upper bound for clusters to test. 
        Default from config.REPRESENTATIVE_OPS['defaults']['k_max'].
    random_state : int or None, optional
        Reproducibility parameter for k-means.
        Default from config.REPRESENTATIVE_OPS['defaults']['random_state'].
    output_dir : str or None, optional
        Directory to save results. If provided, saves files with standardized names:
        - representative_operating_points.csv: Clean column names, timestamps as index
        - clustering_summary.txt: Comprehensive clustering analysis report
        - clustering_info.json: Detailed clustering metrics for programmatic access

    Returns
    -------
    rep_df : pandas.DataFrame
        Subset of input DataFrame containing representative operating points
        (medoids from clusters plus MAPGL-belt snapshots). Column names retain
        original structure for compatibility with analysis functions.
    info : dict
        Comprehensive diagnostics including:
        - 'k': Optimal number of clusters selected
        - 'silhouette': Silhouette score for clustering quality
        - 'ch': Calinski-Harabasz index
        - 'db': Davies-Bouldin index  
        - 'cluster_sizes': Number of points in each cluster
        - 'n_medoid': Number of medoid representatives
        - 'n_belt': Number of MAPGL belt snapshots
        - 'n_total': Total representative points
        - 'original_size': Size of input dataset
        - 'filtered_size': Size after filtering
        - 'feature_columns': Columns used for clustering

    Raises
    ------
    ValueError
        If any surviving snapshot violates net_load < MAPGL, or if no suitable
        feature columns are found for clustering.

    Configuration Dependencies
    -------------------------
    This function relies on config.REPRESENTATIVE_OPS for:
    - Default parameter values (k_max, random_state, etc.)
    - Clustering quality thresholds
    - Feature column prefixes for automatic selection
    - Multi-objective ranking weights
    - Output file naming conventions

    Examples
    --------
    >>> from operating_point_extractor import extract_representative_ops
    >>> # Basic usage with config defaults
    >>> rep_df, diag = extract_representative_ops(df, max_power=850, MAPGL=200)
    >>> print(f"Selected {len(rep_df)} representative points from {len(df)} total")
    >>> print(f"Optimal clusters: {diag['k']} (quality: {diag['silhouette']:.3f})")
    
    >>> # Save results with clean column names and comprehensive reports
    >>> rep_df, diag = extract_representative_ops(
    ...     df, max_power=850, MAPGL=200, output_dir="./results"
    ... )
    >>> # Files saved: representative_operating_points.csv (clean names),
    >>> #               clustering_summary.txt, clustering_info.json
    
    >>> # Override config defaults for custom analysis
    >>> rep_df, diag = extract_representative_ops(
    ...     df, max_power=850, MAPGL=200, k_max=15, random_state=123
    ... )
    
    >>> # Access comprehensive diagnostics
    >>> print(f"Compression ratio: {diag['original_size']/diag['n_total']:.1f}:1")
    >>> print(f"Feature columns: {len(diag['feature_columns'])}")
    >>> print(f"Cluster sizes: {diag['cluster_sizes']}")
    """
    
    # Input validation
    if all_power.empty:
        raise ValueError("Input DataFrame is empty")
    
    if max_power <= 0:
        raise ValueError(f"max_power must be positive, got {max_power}")
    
    if MAPGL <= 0:
        raise ValueError(f"MAPGL must be positive, got {MAPGL}")
    
    if MAPGL >= max_power:
        raise ValueError(f"MAPGL ({MAPGL}) must be less than max_power ({max_power})")
    
    if k_max < 2:
        raise ValueError(f"k_max must be at least 2, got {k_max}")
    
    # Data quality checks
    missing_data = all_power.isnull().sum().sum()
    if missing_data > 0:
        print(f"Warning: {missing_data} missing values detected in input data")
    
    # Check for infinite values
    inf_count = np.isinf(all_power.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        raise ValueError(f"Input data contains {inf_count} infinite values")
    
    # Create working copy
    working = all_power.copy()
    
    # Check if net_load and total_load columns already exist
    if 'net_load' not in working.columns:
        # Import analysis functions for net load calculation only if needed
        from .power_system_analytics import calculate_total_load, calculate_net_load
        
        # Calculate net load for filtering
        total_load = calculate_total_load(working)
        net_load = calculate_net_load(working, total_load)
        working['net_load'] = net_load
        print("Calculated net_load from power system data")
    else:
        print("Using existing net_load column from input data")

    # 1 ───────────────── Data integrity checks
    working = working[working["net_load"] <= max_power]

    if (working["net_load"] < MAPGL).any():
        bad = working[working["net_load"] < MAPGL]
        raise ValueError(
            f"{len(bad)} snapshots violate MAPGL ({MAPGL} MW). "
            "Aborting; please correct input."
        )

    # 2 ───────────────── Feature extraction & scaling
    feat_cols = _select_feature_columns(working)
    if len(feat_cols) == 0:
        raise ValueError("No suitable feature columns found (ss_mw_*, ss_mvar_*, wind_mw_*)")
    
    # Check feature data quality
    feature_data = working[feat_cols]
    zero_variance_features = feature_data.var() == 0
    if zero_variance_features.any():
        print(f"Warning: {zero_variance_features.sum()} features have zero variance and will be excluded")
        feat_cols = [col for col in feat_cols if not zero_variance_features[col]]
        if len(feat_cols) == 0:
            raise ValueError("No features with non-zero variance found")
    
    x_raw = working[feat_cols].to_numpy(float)
    scaler = StandardScaler()
    x = scaler.fit_transform(x_raw)

    # 3 ───────────────── K-means with automatic k
    model, metrics = _auto_kmeans(x, k_max=k_max, random_state=random_state)
    labels = model.labels_
    centres = model.cluster_centers_

    # 4 ───────────────── Medoid identification
    medoid_ids: list = []
    for k in range(model.n_clusters):
        members = np.where(labels == k)[0]
        if members.size == 0:
            continue
        centre = centres[k]
        member_vecs = x[members]
        dist2 = ((member_vecs - centre) ** 2).sum(axis=1)
        # Get the position in the members array, then map to DataFrame index
        medoid_pos = members[int(dist2.argmin())]
        medoid_id = working.index[medoid_pos]  # Map array position to DataFrame index
        medoid_ids.append(medoid_id)

    # 5 ───────────────── Append MAPGL belt snapshots
    mapgl_multiplier = REPRESENTATIVE_OPS['defaults']['mapgl_belt_multiplier']
    belt_mask = (working["net_load"] > MAPGL) & (working["net_load"] < mapgl_multiplier * MAPGL)
    belt_ids = working.index[belt_mask].tolist()

    all_ids = sorted(set(medoid_ids).union(belt_ids))
    rep_df = working.loc[all_ids].copy()

    # 6 ───────────────── Return with diagnostics
    info = {
        **metrics,
        "cluster_sizes": np.bincount(labels, minlength=model.n_clusters).tolist(),
        "n_medoid": len(medoid_ids),
        "n_belt": len(belt_ids),
        "n_total": len(rep_df),
        "original_size": len(all_power),
        "filtered_size": len(working),
        "feature_columns": feat_cols,
        "data_quality": {
            "missing_values": missing_data,
            "infinite_values": inf_count,
            "zero_variance_features_excluded": zero_variance_features.sum() if 'zero_variance_features' in locals() else 0
        }
    }

    # 7 ───────────────── Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save representative operating points as CSV (include timestamps in index)
        output_files = REPRESENTATIVE_OPS['output_files']
        filename_rep = os.path.join(output_dir, output_files['representative_points'])
        
        # Create a copy with cleaned column names for better readability
        rep_df_clean = rep_df.copy()
        rep_df_clean.columns = [clean_column_name(col) for col in rep_df_clean.columns]
        
        rep_df_clean.to_csv(filename_rep, index=True)
        
        # Create comprehensive clustering summary
        summary_filename = os.path.join(output_dir, output_files['clustering_summary'])
        _create_clustering_summary(
            summary_filename, all_power, working, rep_df, info, 
            max_power, MAPGL, k_max, random_state, feat_cols, model, scaler
        )
        
        # Also save detailed info as JSON for programmatic access
        json_filename = os.path.join(output_dir, output_files['clustering_info'])
        
        # Convert the info dictionary to JSON-safe format
        json_safe_info = convert_numpy_types(info)
        
        try:
            with open(json_filename, "w", encoding='utf-8') as f:
                json.dump(json_safe_info, f, indent=4)
        except (TypeError, ValueError) as e:
            print(f"Warning: Could not save JSON file due to serialization error: {e}")
            print("Saving simplified JSON without problematic data types...")
            
            # Fallback: save only basic metrics
            basic_info = {
                'k': int(info.get('k', 0)),
                'silhouette': float(info.get('silhouette', 0.0)),
                'ch': float(info.get('ch', 0.0)),
                'db': float(info.get('db', 0.0)),
                'cluster_sizes': [int(x) for x in info.get('cluster_sizes', [])],
                'n_medoid': int(info.get('n_medoid', 0)),
                'n_belt': int(info.get('n_belt', 0)),
                'n_total': int(info.get('n_total', 0)),
                'original_size': int(info.get('original_size', 0)),
                'filtered_size': int(info.get('filtered_size', 0)),
                'feature_columns': list(info.get('feature_columns', [])),
            }
            
            with open(json_filename, "w", encoding='utf-8') as f:
                json.dump(basic_info, f, indent=4)
        
        # Create visualizations
        _create_visualizations(
            output_dir, working, rep_df, info, model, scaler, feat_cols, max_power, MAPGL
        )

        print(f"Results saved to:")
        print(f"  Representative points: {filename_rep}")
        print(f"  Clustering summary: {summary_filename}")
        print(f"  Detailed info (JSON): {json_filename}")

    return rep_df, info 