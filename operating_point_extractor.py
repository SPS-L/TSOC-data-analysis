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
- extract_representative_ops(): Main function to extract representative points
- _select_feature_columns(): Helper to identify clustering features (config-driven)
- _auto_kmeans(): Helper for automatic K-means cluster selection (config-driven)
- _create_clustering_summary(): Generates comprehensive analysis reports

USAGE EXAMPLES:
==============

# Basic usage with default configuration
rep_df, diagnostics = extract_representative_ops(
    all_power=df, max_power=850, MAPGL=200
)

# Save results with automatic file naming
rep_df, diagnostics = extract_representative_ops(
    all_power=df, max_power=850, MAPGL=200, 
    output_dir="results"  # Uses config file names
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
from system_configuration import clean_column_name, REPRESENTATIVE_OPS

__all__ = ["extract_representative_ops"]


def _select_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return columns starting with ss_mw_, ss_mvar_ or wind_mw_."""
    keep_prefix = tuple(REPRESENTATIVE_OPS['feature_columns']['clustering_prefixes'])
    return [c for c in df.columns if c.startswith(keep_prefix)]


def _auto_kmeans(
    x: np.ndarray,
    k_max: int = REPRESENTATIVE_OPS['defaults']['k_max'],
    random_state: int | None = REPRESENTATIVE_OPS['defaults']['random_state'],
) -> Tuple[KMeans, Dict[str, float]]:
    """Fit k-means while automatically selecting k."""
    best_model: KMeans | None = None
    best_score: float = -np.inf
    best_k: int = 0
    best_metrics: dict[str, float] = {}

    for k in range(2, min(k_max, len(x) - 1) + 1):
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
        if combo > best_score and sil > min_silhouette:
            best_score = combo
            best_model = km
            best_k = k
            best_metrics = {"silhouette": sil, "ch": ch, "db": db}

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
    """Create a comprehensive clustering summary report."""
    with open(filename, "w", encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("REPRESENTATIVE OPERATING POINTS CLUSTERING SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Author: Sustainable Power Systems Lab (SPSL)\n")
        f.write(f"Web: https://sps-lab.org\n")
        f.write(f"Contact: info@sps-lab.org\n\n")
        
        # 1. METHODOLOGY OVERVIEW
        f.write("1. METHODOLOGY OVERVIEW\n")
        f.write("-"*30 + "\n")
        f.write("This analysis implements the methodology described in:\n")
        f.write("'Automated Extraction of Representative Operating Points for a 132 kV Transmission System'\n\n")
        f.write("Process Steps:\n")
        f.write("  1. Data filtering based on power limits and MAPGL constraints\n")
        f.write("  2. Feature extraction from power injection variables\n")
        f.write("  3. Data standardization using StandardScaler\n")
        f.write("  4. K-means clustering with automatic cluster count selection\n")
        f.write("  5. Medoid identification (actual snapshots closest to cluster centers)\n")
        f.write("  6. Addition of MAPGL-belt snapshots for critical low-load conditions\n\n")
        
        # 2. INPUT PARAMETERS
        f.write("2. INPUT PARAMETERS\n")
        f.write("-"*30 + "\n")
        f.write(f"Maximum Power Limit:        {max_power:.2f} MW\n")
        f.write(f"MAPGL (Min Generation):     {MAPGL:.2f} MW\n")
        f.write(f"Maximum Clusters Tested:    {k_max}\n")
        f.write(f"Random State:               {random_state}\n")
        mapgl_multiplier = REPRESENTATIVE_OPS['defaults']['mapgl_belt_multiplier']
        f.write(f"MAPGL Belt Range:           {MAPGL:.2f} - {mapgl_multiplier*MAPGL:.2f} MW\n\n")
        
        # 3. DATA PROCESSING SUMMARY
        f.write("3. DATA PROCESSING SUMMARY\n")
        f.write("-"*30 + "\n")
        f.write(f"Original Dataset Size:      {info['original_size']:,} snapshots\n")
        f.write(f"After Power Filtering:      {info['filtered_size']:,} snapshots\n")
        f.write(f"Reduction Factor:           {info['original_size']/info['filtered_size']:.2f}x\n\n")
        
        f.write("Feature Columns Used for Clustering:\n")
        for i, col in enumerate(feat_cols, 1):
            f.write(f"  {i:2d}. {col}\n")
        f.write(f"\nTotal Features:             {len(feat_cols)}\n\n")
        
        # 4. CLUSTERING RESULTS
        f.write("4. CLUSTERING RESULTS\n")
        f.write("-"*30 + "\n")
        f.write(f"Optimal Number of Clusters: {info['k']}\n")
        f.write(f"Silhouette Score:           {info['silhouette']:.4f}\n")
        f.write(f"Calinski-Harabasz Index:    {info['ch']:.2f}\n")
        f.write(f"Davies-Bouldin Index:       {info['db']:.4f}\n\n")
        
        f.write("Cluster Composition:\n")
        for i, size in enumerate(info['cluster_sizes']):
            percentage = (size / info['filtered_size']) * 100
            f.write(f"  Cluster {i+1:2d}: {size:6,} snapshots ({percentage:5.1f}%)\n")
        f.write(f"  Total:       {sum(info['cluster_sizes']):6,} snapshots (100.0%)\n\n")
        
        # Add detailed cluster analysis
        f.write("Cluster Centers (Original Feature Space):\n")
        # Transform cluster centers back to original space
        cluster_centers_orig = scaler.inverse_transform(model.cluster_centers_)
        for i, center in enumerate(cluster_centers_orig):
            f.write(f"  Cluster {i+1}:\n")
            max_features_display = min(5, REPRESENTATIVE_OPS['validation']['max_features_to_display'])
            display_limit = min(max_features_display, len(feat_cols))
            for j, feature in enumerate(feat_cols[:display_limit]):
                f.write(f"    {feature}: {center[j]:.3f}\n")
            if len(feat_cols) > display_limit:
                f.write(f"    ... and {len(feat_cols)-display_limit} more features\n")
        f.write("\n")
        
        # Within-cluster sum of squares (inertia)
        f.write(f"Within-Cluster Sum of Squares (WCSS): {model.inertia_:.2f}\n")
        f.write(f"Average Distance to Centroid:         {model.inertia_/info['filtered_size']:.4f}\n\n")
        
        # 5. REPRESENTATIVE POINTS SELECTION
        f.write("5. REPRESENTATIVE POINTS SELECTION\n")
        f.write("-"*30 + "\n")
        f.write(f"Medoids from Clusters:      {info['n_medoid']}\n")
        f.write(f"MAPGL Belt Snapshots:       {info['n_belt']}\n")
        f.write(f"Total Representative Points: {info['n_total']}\n")
        f.write(f"Compression Ratio:          {info['original_size']/info['n_total']:.1f}:1\n")
        f.write(f"Retention Rate:             {(info['n_total']/info['original_size'])*100:.2f}%\n\n")
        
        # Detailed medoid analysis
        f.write("Medoid Details (Selected Representative Snapshots):\n")
        # Get medoid information
        x_raw = working[feat_cols].to_numpy(float)
        x_scaled = scaler.transform(x_raw)
        labels = model.labels_
        
        for k in range(model.n_clusters):
            members = np.where(labels == k)[0]
            if members.size > 0:
                centre = model.cluster_centers_[k]
                member_vecs = x_scaled[members]
                dist2 = ((member_vecs - centre) ** 2).sum(axis=1)
                medoid_pos = members[int(dist2.argmin())]
                medoid_id = working.index[medoid_pos]
                
                f.write(f"  Cluster {k+1}: Snapshot ID {medoid_id}\n")
                if 'net_load' in working.columns:
                    net_load_val = working.loc[medoid_id, 'net_load']
                    f.write(f"    Net Load: {net_load_val:.2f} MW\n")
                f.write(f"    Distance to Centroid: {np.sqrt(dist2.min()):.4f}\n")
        f.write("\n")
        
        # Inter-cluster distances
        f.write("Inter-Cluster Distances (Euclidean):\n")
        from scipy.spatial.distance import pdist, squareform
        if model.n_clusters > 1:
            cluster_distances = pdist(model.cluster_centers_)
            dist_matrix = squareform(cluster_distances)
            f.write("  Distance Matrix (scaled feature space):\n")
            f.write("  " + "".join([f"Clust{i+1:2d}" for i in range(model.n_clusters)]) + "\n")
            for i in range(model.n_clusters):
                f.write(f"C{i+1} ")
                for j in range(model.n_clusters):
                    f.write(f"{dist_matrix[i,j]:6.3f} ")
                f.write("\n")
            f.write(f"  Average Inter-Cluster Distance: {cluster_distances.mean():.4f}\n")
            f.write(f"  Minimum Inter-Cluster Distance: {cluster_distances.min():.4f}\n")
            f.write(f"  Maximum Inter-Cluster Distance: {cluster_distances.max():.4f}\n")
        f.write("\n")
        
        # 6. VALIDATION METRICS EXPLANATION
        f.write("6. CLUSTERING QUALITY METRICS\n")
        f.write("-"*30 + "\n")
        f.write("Silhouette Score (-1 to 1):\n")
        f.write("  Measures how similar points are to their own cluster vs other clusters.\n")
        f.write("  Values > 0.5 indicate good clustering, > 0.7 excellent.\n")
        
        # Determine clustering quality
        if info['silhouette'] > 0.7:
            quality = "Excellent clustering"
        elif info['silhouette'] > 0.5:
            quality = "Good clustering"
        elif info['silhouette'] > 0.25:
            quality = "Acceptable clustering"
        else:
            quality = "Poor clustering"
        
        f.write(f"  Current score: {info['silhouette']:.4f} - {quality}\n")
        
        f.write("\nCalinski-Harabasz Index (higher is better):\n")
        f.write("  Ratio of between-cluster to within-cluster variance.\n")
        f.write("  Higher values indicate more compact and well-separated clusters.\n")
        f.write(f"  Current score: {info['ch']:.2f}\n")
        
        f.write("\nDavies-Bouldin Index (lower is better):\n")
        f.write("  Average similarity ratio of each cluster with most similar cluster.\n")
        f.write("  Lower values indicate better clustering with distinct clusters.\n")
        f.write(f"  Current score: {info['db']:.4f}\n\n")
        
        # 7. REPRESENTATIVE POINTS DETAILS
        f.write("7. REPRESENTATIVE POINTS DETAILS\n")
        f.write("-"*30 + "\n")
        if 'net_load' in rep_df.columns:
            f.write("Net Load Statistics for Representative Points:\n")
            f.write(f"  Minimum:  {rep_df['net_load'].min():.2f} MW\n")
            f.write(f"  Maximum:  {rep_df['net_load'].max():.2f} MW\n")
            f.write(f"  Mean:     {rep_df['net_load'].mean():.2f} MW\n")
            f.write(f"  Median:   {rep_df['net_load'].median():.2f} MW\n")
            f.write(f"  Std Dev:  {rep_df['net_load'].std():.2f} MW\n\n")
            
            # MAPGL belt analysis
            if info['n_belt'] > 0:
                mapgl_multiplier = REPRESENTATIVE_OPS['defaults']['mapgl_belt_multiplier']
                belt_mask = (working["net_load"] > MAPGL) & (working["net_load"] < mapgl_multiplier * MAPGL)
                belt_snapshots = working[belt_mask]
                f.write("MAPGL Belt Snapshots Analysis:\n")
                f.write(f"  Belt Range: {MAPGL:.2f} - {mapgl_multiplier*MAPGL:.2f} MW\n")
                f.write(f"  Belt Snapshots: {len(belt_snapshots)}\n")
                f.write(f"  Belt Net Load Mean: {belt_snapshots['net_load'].mean():.2f} MW\n")
                f.write(f"  Belt Net Load Std:  {belt_snapshots['net_load'].std():.2f} MW\n\n")
        
        # Feature statistics comparison
        f.write("Feature Statistics (Original vs Representative Points):\n")
        f.write("  Feature Name               Original        Representative    Ratio\n")
        f.write("  " + "-"*65 + "\n")
        max_features_display = REPRESENTATIVE_OPS['validation']['max_features_to_display']
        display_limit = min(max_features_display, len(feat_cols))
        for feature in feat_cols[:display_limit]:
            orig_mean = working[feature].mean()
            rep_mean = rep_df[feature].mean()
            ratio = rep_mean / orig_mean if orig_mean != 0 else float('nan')
            f.write(f"  {feature:<25} {orig_mean:10.3f}    {rep_mean:10.3f}     {ratio:6.3f}\n")
        if len(feat_cols) > display_limit:
            f.write(f"  ... and {len(feat_cols)-display_limit} more features\n")
        f.write("\n")
        
        # 7b. ADVANCED CLUSTERING DIAGNOSTICS
        f.write("7b. ADVANCED CLUSTERING DIAGNOSTICS\n")
        f.write("-"*30 + "\n")
        
        # Individual cluster statistics
        f.write("Individual Cluster Statistics:\n")
        for k in range(model.n_clusters):
            cluster_mask = labels == k
            cluster_data = working[cluster_mask]
            f.write(f"  Cluster {k+1} ({info['cluster_sizes'][k]} snapshots):\n")
            if 'net_load' in cluster_data.columns:
                f.write(f"    Net Load Range: {cluster_data['net_load'].min():.2f} - {cluster_data['net_load'].max():.2f} MW\n")
                f.write(f"    Net Load Mean:  {cluster_data['net_load'].mean():.2f} +/- {cluster_data['net_load'].std():.2f} MW\n")
            
            # Cluster compactness (average distance to centroid)
            cluster_vectors = x_scaled[cluster_mask]
            centroid = model.cluster_centers_[k]
            distances = np.sqrt(((cluster_vectors - centroid) ** 2).sum(axis=1))
            f.write(f"    Avg Distance to Centroid: {distances.mean():.4f} +/- {distances.std():.4f}\n")
            f.write(f"    Max Distance to Centroid: {distances.max():.4f}\n")
        f.write("\n")
        
        # Feature scaling diagnostics
        f.write("Feature Scaling Diagnostics:\n")
        f.write("  Original Feature Statistics (before scaling):\n")
        original_means = np.mean(x_raw, axis=0)
        original_stds = np.std(x_raw, axis=0)
        f.write(f"    Mean feature magnitude: {np.mean(np.abs(original_means)):.3f}\n")
        f.write(f"    Mean feature std dev:   {np.mean(original_stds):.3f}\n")
        f.write(f"    Feature scale ratio:    {np.max(original_stds)/np.min(original_stds):.2f}:1\n")
        f.write("  Post-scaling verification:\n")
        f.write(f"    Scaled means near zero: {np.allclose(np.mean(x_scaled, axis=0), 0, atol=1e-10)}\n")
        f.write(f"    Scaled std devs = 1:    {np.allclose(np.std(x_scaled, axis=0), 1, atol=1e-10)}\n\n")
        
        # 8. TECHNICAL DETAILS
        f.write("8. TECHNICAL IMPLEMENTATION DETAILS\n")
        f.write("-"*30 + "\n")
        f.write("Clustering Algorithm:       K-means with k-means++\n")
        f.write("Feature Scaling:            StandardScaler (z-score normalization)\n")
        f.write("Cluster Selection Method:   Multi-objective optimization\n")
        f.write("  - Maximize: Silhouette Score + Calinski-Harabasz Index\n")
        f.write("  - Minimize: Davies-Bouldin Index\n")
        f.write("  - Quality Threshold: Silhouette > 0.25\n")
        f.write("Medoid Selection:           Euclidean distance to cluster centroid\n")
        f.write("MAPGL Belt Definition:      1.0 < net_load/MAPGL < 1.1\n\n")
        
        # 9. RECOMMENDATIONS
        f.write("9. RECOMMENDATIONS AND NEXT STEPS\n")
        f.write("-"*30 + "\n")
        if info['silhouette'] < 0.25:
            f.write("⚠️  LOW CLUSTERING QUALITY WARNING:\n")
            f.write("   Consider:\n")
            f.write("   - Increasing the dataset size\n")
            f.write("   - Reviewing feature selection\n")
            f.write("   - Adjusting power limits or MAPGL\n\n")
        
        if info['n_total'] < 10:
            f.write("⚠️  FEW REPRESENTATIVE POINTS:\n")
            f.write("   Consider lowering k_max or adjusting clustering parameters\n\n")
        
        f.write("For power system analysis:\n")
        f.write("1. Validate representative points against operational constraints\n")
        f.write("2. Verify load flow convergence for all selected snapshots\n")
        f.write("3. Check that critical operating conditions are captured\n")
        f.write("4. Consider seasonal or temporal patterns if relevant\n\n")
        
        # 10. RESULTS INTERPRETATION GUIDE
        f.write("10. RESULTS INTERPRETATION GUIDE\n")
        f.write("-"*30 + "\n")
        
        # Overall clustering quality assessment
        f.write("OVERALL CLUSTERING QUALITY ASSESSMENT:\n")
        silhouette_val = info.get('silhouette', 0)
        compression_ratio = info['original_size'] / info['n_total']
        
        if silhouette_val > 0.7 and compression_ratio > 20:
            overall_quality = "EXCELLENT"
            f.write("🟢 EXCELLENT: High-quality clustering with significant data reduction.\n")
        elif silhouette_val > 0.5 and compression_ratio > 10:
            overall_quality = "GOOD"
            f.write("🟡 GOOD: Acceptable clustering quality with reasonable data reduction.\n")
        elif silhouette_val > 0.25:
            overall_quality = "ACCEPTABLE"
            f.write("🟠 ACCEPTABLE: Basic clustering quality, results usable with caution.\n")
        else:
            overall_quality = "POOR"
            f.write("🔴 POOR: Low clustering quality, consider data preprocessing or parameter adjustment.\n")
        
        f.write(f"Quality Score: {overall_quality}\n")
        f.write(f"Data Reduction: {compression_ratio:.1f}:1 ({((compression_ratio-1)/compression_ratio)*100:.1f}% reduction)\n\n")
        
        # Interpretation of clustering metrics
        f.write("CLUSTERING METRICS INTERPRETATION:\n")
        
        f.write("Silhouette Score Analysis:\n")
        if silhouette_val > 0.7:
            f.write("  ✓ Excellent separation: Clusters are well-defined and distinct\n")
            f.write("  ✓ Representative points are highly reliable\n")
            f.write("  ✓ Strong confidence in operating point selection\n")
        elif silhouette_val > 0.5:
            f.write("  ✓ Good separation: Clusters are reasonably well-defined\n")
            f.write("  ✓ Representative points are reliable\n")
            f.write("  ~ Minor overlap between some clusters is acceptable\n")
        elif silhouette_val > 0.25:
            f.write("  ~ Moderate separation: Some cluster overlap present\n")
            f.write("  ~ Representative points should be validated carefully\n")
            f.write("  ⚠ Consider additional validation of selected points\n")
        else:
            f.write("  ✗ Poor separation: Significant cluster overlap\n")
            f.write("  ✗ Representative points may not be reliable\n")
            f.write("  ⚠ Strong recommendation to revise approach\n")
        
        ch_val = info.get('ch', 0)
        f.write(f"\nCalinski-Harabasz Index Analysis (Current: {ch_val:.1f}):\n")
        if ch_val > 100:
            f.write("  ✓ High index: Well-separated, compact clusters\n")
            f.write("  ✓ Strong internal cluster cohesion\n")
        elif ch_val > 50:
            f.write("  ✓ Moderate index: Reasonably good cluster structure\n")
        else:
            f.write("  ~ Low index: Clusters may be poorly separated or too diffuse\n")
        
        db_val = info.get('db', float('inf'))
        f.write(f"\nDavies-Bouldin Index Analysis (Current: {db_val:.3f}):\n")
        if db_val < 0.5:
            f.write("  ✓ Excellent: Very low similarity between clusters\n")
        elif db_val < 1.0:
            f.write("  ✓ Good: Low similarity between clusters\n")
        elif db_val < 1.5:
            f.write("  ~ Acceptable: Moderate similarity between clusters\n")
        else:
            f.write("  ✗ Poor: High similarity between clusters indicates overlap\n")
        f.write("\n")
        
        # Practical interpretation for power systems
        f.write("POWER SYSTEM ANALYSIS INTERPRETATION:\n")
        
        f.write("Representative Point Validation Checklist:\n")
        f.write("□ Load flow convergence: Run power flow for each representative point\n")
        f.write("□ Voltage limits: Check bus voltages within acceptable ranges\n")
        f.write("□ Thermal limits: Verify line/transformer loadings are feasible\n")
        f.write("□ Generation limits: Confirm generator outputs respect P/Q limits\n")
        f.write("□ N-1 security: Test contingency analysis for critical scenarios\n")
        f.write("□ Operational feasibility: Verify points represent realistic conditions\n\n")
        
        # Cluster interpretation
        f.write("Cluster Interpretation Guidelines:\n")
        if info['k'] <= 3:
            f.write("  • Small number of clusters suggests limited operating diversity\n")
            f.write("  • May indicate stable operating conditions or limited renewable variation\n")
        elif info['k'] <= 6:
            f.write("  • Moderate number of clusters indicates typical operating diversity\n")
            f.write("  • Good balance between detail and simplification\n")
        else:
            f.write("  • Large number of clusters suggests high operating diversity\n")
            f.write("  • May indicate significant renewable variability or complex system behavior\n")
        
        f.write(f"  • Data retention: {(info['n_total']/info['original_size'])*100:.1f}% of original time points\n")
        f.write(f"  • Time coverage: Representative points span the analysis period\n\n")
        
        # MAPGL belt analysis interpretation
        if info['n_belt'] > 0:
            belt_percentage = (info['n_belt'] / info['n_total']) * 100
            f.write("MAPGL Belt Analysis:\n")
            f.write(f"  • {info['n_belt']} snapshots in MAPGL belt ({belt_percentage:.1f}% of representatives)\n")
            if belt_percentage > 20:
                f.write("  • High proportion of low-load conditions captured\n")
                f.write("  • Good representation of minimum generation scenarios\n")
            else:
                f.write("  • Limited low-load representation - typical for high-demand periods\n")
            f.write("  • These points are critical for minimum generation studies\n\n")
        
        # Usage recommendations based on results
        f.write("RECOMMENDED USAGE BASED ON RESULTS:\n")
        
        if overall_quality in ["EXCELLENT", "GOOD"]:
            f.write("✓ HIGH CONFIDENCE APPLICATIONS:\n")
            f.write("  • Long-term transmission planning studies\n")
            f.write("  • Investment analysis and capacity expansion\n")
            f.write("  • Security assessment and contingency analysis\n")
            f.write("  • Renewable integration studies\n")
            f.write("  • Grid code compliance verification\n\n")
        
        if overall_quality in ["ACCEPTABLE"]:
            f.write("~ MODERATE CONFIDENCE APPLICATIONS:\n")
            f.write("  • Preliminary planning studies (with validation)\n")
            f.write("  • Scenario development for detailed analysis\n")
            f.write("  • Identification of critical operating conditions\n")
            f.write("  ⚠ Recommend additional validation before final decisions\n\n")
        
        if overall_quality in ["POOR"]:
            f.write("⚠ LIMITED CONFIDENCE APPLICATIONS:\n")
            f.write("  • Initial screening studies only\n")
            f.write("  • Pattern identification in operational data\n")
            f.write("  • Troubleshooting and methodology development\n")
            f.write("  ✗ NOT recommended for critical planning decisions\n\n")
        
        # Troubleshooting guide
        f.write("TROUBLESHOOTING COMMON ISSUES:\n")
        
        if silhouette_val < 0.25:
            f.write("Poor Clustering Quality:\n")
            f.write("  • Increase dataset size (more time periods)\n")
            f.write("  • Review feature selection (check power injection columns)\n")
            f.write("  • Verify data quality (missing values, outliers)\n")
            f.write("  • Consider different power limits or MAPGL values\n")
            f.write("  • Check for sufficient operating diversity in data\n\n")
        
        if compression_ratio < 5:
            f.write("Low Data Reduction:\n")
            f.write("  • Dataset may have high natural diversity\n")
            f.write("  • Consider increasing k_max parameter\n")
            f.write("  • Review clustering features for redundancy\n")
            f.write("  • May indicate complex system with many distinct operating states\n\n")
        
        if info['n_belt'] == 0 and info['filtered_size'] > 100:
            f.write("No MAPGL Belt Snapshots:\n")
            f.write("  • All snapshots may be well above MAPGL\n")
            f.write("  • Consider lowering MAPGL or checking minimum generation scenarios\n")
            f.write("  • May indicate high-demand period analysis\n\n")
        
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

    Parameters
    ----------
    all_power : pandas.DataFrame
        Input time-series DataFrame with power system data including columns:
        - ss_mw_*: Substation active power (MW)
        - ss_mvar_*: Substation reactive power (MVAR) 
        - wind_mw_*: Wind farm active power (MW)
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
    
    # Import analysis functions for net load calculation
    from power_system_analytics import calculate_total_load, calculate_net_load

    # Create working copy and calculate net_load
    working = all_power.copy()
    
    # Calculate net load for filtering
    total_load = calculate_total_load(working)
    net_load = calculate_net_load(working, total_load)
    working['net_load'] = net_load

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
        with open(json_filename, "w", encoding='utf-8') as f:
            json.dump(info, f, indent=4)
        
        print(f"Results saved to:")
        print(f"  Representative points: {filename_rep}")
        print(f"  Clustering summary: {summary_filename}")
        print(f"  Detailed info (JSON): {json_filename}")

    return rep_df, info 