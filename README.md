# Power System Data Analysis Tool

**Author:** Sustainable Power Systems Lab (SPSL), [https://sps-lab.org](https://sps-lab.org), contact: info@sps-lab.org

A comprehensive Python tool for analyzing power system operational data from Excel files. The tool provides a powerful command-line interface (CLI) and modular Python API for load analysis, generator categorization, wind power analysis, reactive power calculations, and representative operating point extraction.

**Pure Python Implementation**: This tool is implemented entirely in Python with no external dependencies on notebook environments. It can be used from the command line, imported as Python modules, or integrated into automated analysis pipelines.

## Features

- **Month-based data filtering** for efficient processing of large datasets
- **Load calculations** (Total Load, Net Load) with comprehensive statistics
- **Wind power analysis** with generation statistics and profiles
- **Generator categorization** (Voltage Control vs PQ Control)
- **Reactive power analysis** with comprehensive calculations
- **Representative operating points extraction** using K-means clustering
- **Data validation** with type checking, limit validation, and gap filling
- **Centralized configuration management** for easy customization and maintenance
- **Comprehensive logging** and error handling
- **Multiple output formats** (CSV, JSON, PNG plots, text summaries)
- **Clean column naming** for improved readability of output files

## Configuration Management

### Design Philosophy and Rationale

The tool implements a **centralized configuration architecture** designed to address common challenges in power system analysis software:

#### **Problem: Scattered Parameters**
Traditional power system analysis tools often suffer from:
- Hardcoded parameters scattered throughout multiple files
- Difficulty in customizing analysis behavior
- Inconsistent parameter values across different modules
- Poor maintainability when parameters need updates
- Code duplication for common utilities

#### **Solution: Centralized Configuration**
This tool centralizes all configuration in `system_configuration.py`, providing:

**🎯 Single Source of Truth**
- All parameters defined in one location (`system_configuration.py`)
- Consistent values across all modules
- Easy discovery of configurable options
- Comprehensive documentation for each parameter

**🔧 Easy Customization**
- Modify analysis behavior by changing config values
- No need to hunt through multiple files
- Immediate effect across all modules using the parameter
- Support for different analysis scenarios through configuration

**📚 Better Maintainability**
- Parameter updates require changes in only one file
- Reduced risk of inconsistent values
- Clear dependency tracking
- Simplified testing and validation

**♻️ Code Reusability**
- Shared utilities like `clean_column_name()` avoid duplication
- Consistent behavior across different analysis workflows
- Modular design enables component reuse

### Key Configuration Areas

#### **1. Representative Operations Configuration**
```python
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
```

#### **2. Data Validation Configuration**
```python
DATA_VALIDATION = {
    'type_checks': {
        'real_numbers': ['ss_mw_', 'ss_mvar_', 'wind_mw_'],
        'integers': ['shunt_tap_']
    },
    'limit_checks': {
        'power_limits': {
            'wind': {'min_mw': 0, 'max_mw': 100},
            'substation': {'min_mw': -100, 'max_mw': 100}
        }
    }
}
```

#### **3. File Structure Configuration**
```python
FILES = {
    'substation_mw': 'substation_active_power.xlsx',
    'wind_power': 'wind_farm_active_power.xlsx',
    # ... other file mappings
}

COLUMN_PREFIXES = {
    'substation_mw': 'ss_mw_',
    'wind_power': 'wind_mw_',
    # ... other prefixes
}
```

### Configuration Benefits in Practice

#### **For Researchers**
- Quickly adjust clustering parameters for different studies
- Modify validation thresholds for different power system types
- Easy comparison of different parameter sets
- Reproducible research through documented configurations

#### **For Operators**
- Customize limits based on actual system constraints
- Adjust analysis sensitivity for different operational conditions
- Configure file locations for different data sources
- Maintain consistent analysis across different time periods

#### **For Developers**
- Add new parameters without modifying multiple files
- Clear understanding of configurable vs. fixed behavior
- Easier testing with different parameter combinations
- Simplified maintenance and updates

### Utility Functions

#### **Clean Column Naming**
The `clean_column_name()` function in `system_configuration.py` provides consistent column name cleaning:

```python
def clean_column_name(col_name):
    """Remove verbose suffixes for cleaner output files."""
    # Removes suffixes like '_132REACTOR_REACTIVE_POWER'
    # Result: 'ss_mw_STATION1' instead of 'ss_mw_STATION1_132REACTOR_REACTIVE_POWER'
```

**Benefits:**
- **Reusable**: Used across multiple modules (CLI and representative ops)
- **Consistent**: Same cleaning logic everywhere
- **Maintainable**: Single location for suffix definitions
- **Readable**: Cleaner column names in output CSV files

### Configuration Customization Examples

#### **Customizing Representative Operations**
```python
# In system_configuration.py, modify REPRESENTATIVE_OPS for your system:

# For larger power systems (more clusters needed)
REPRESENTATIVE_OPS['defaults']['k_max'] = 20

# For higher quality clustering requirements
REPRESENTATIVE_OPS['quality_thresholds']['min_silhouette'] = 0.4

# For different MAPGL belt definition
REPRESENTATIVE_OPS['defaults']['mapgl_belt_multiplier'] = 1.15

# For custom ranking weights (emphasize different metrics)
REPRESENTATIVE_OPS['ranking_weights'] = {
    'silhouette_weight': 2000,        # Double importance
    'calinski_harabasz_weight': 2,    # Higher weight
    'davies_bouldin_weight': 5        # Lower penalty
}
```

#### **Customizing Data Validation**
```python
# In system_configuration.py, modify DATA_VALIDATION for your data:

# For different power system limits
DATA_VALIDATION['limit_checks']['power_limits']['wind']['max_mw'] = 200
DATA_VALIDATION['limit_checks']['power_limits']['substation']['max_mw'] = 500

# For more aggressive gap filling
DATA_VALIDATION['gap_filling']['max_gap_steps'] = 6
DATA_VALIDATION['gap_filling']['advanced_max_gap_steps'] = 24

# For stricter data quality requirements
DATA_VALIDATION['quality_thresholds']['max_missing_percentage'] = 5.0
```

#### **Customizing Output Files**
```python
# In system_configuration.py, modify file naming:

# Custom suffixes to remove
def clean_column_name(col_name):
    suffixes_to_remove = [
        '_YOUR_CUSTOM_SUFFIX',
        '_ANOTHER_VERBOSE_ENDING',
        # ... add your system's specific suffixes
    ]
    # ... rest of function

# Custom output file names
REPRESENTATIVE_OPS['output_files'] = {
    'representative_points': 'selected_operating_points.csv',
    'clustering_summary': 'clustering_analysis.txt',
    'clustering_info': 'clustering_data.json'
}
```

## Quick Start

### Basic Analysis
```bash
# Run analysis for September 2024
python power_analysis_cli.py 2024-09 --save-csv --save-plots

# Run analysis for all data
python power_analysis_cli.py --save-csv --save-plots --verbose
```

### Representative Operating Points
```bash
# Extract representative points with default parameters
python -c "
from power_analysis_cli import execute
from operating_point_extractor import extract_representative_ops

# Load and analyze data
success, df = execute(month='2024-01', save_csv=True)
if success:
    # Extract representative points using config parameters
    rep_df, diagnostics = extract_representative_ops(
        df, max_power=850, MAPGL=200, output_dir='results'
    )
    print(f'Selected {len(rep_df)} representative points')
"
```

## Usage

### Basic Analysis

Run power system analysis with various options:

```bash
# Run full analysis with all outputs for January 2024
python power_analysis_cli.py 2024-01 --output-dir results --save-plots --save-csv

# Run analysis with specific data directory for March 2024
python power_analysis_cli.py 2024-03 --data-dir "2024-2025 data" --verbose

# Run analysis and save only summary report for December 2024
python power_analysis_cli.py 2024-12 --output-dir results --summary-only

# Run analysis for all data (no month filter)
python power_analysis_cli.py --output-dir results --save-plots --save-csv

# Quick analysis for a specific month with summary only
python power_analysis_cli.py 2024-06 --summary-only
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `MONTH` | Month to filter data for (format: "YYYY-MM") or None for all data | None |
| `--data-dir` | Directory containing Excel data files | `raw_data` |
| `--output-dir` | Directory to save output files | `results` |
| `--save-csv` | Save analysis results to CSV files | False |
| `--save-plots` | Generate and save plots as PNG files | False |
| `--summary-only` | Generate only summary report | False |
| `--verbose` | Print detailed progress information | False |
| `--version` | Show version information | - |
| `-h, --help` | Show help message | - |

## Output Files

### Standard Analysis Files
- `analysis_summary.json` - Complete analysis results in JSON format
- `analysis_summary.txt` - Human-readable analysis summary
- `total_load.csv` - Total load time series data
- `net_load.csv` - Net load time series data
- `load_statistics.csv` - Load statistics (min, max, mean, std)
- `generator_categories.csv` - Generator categorization results
- `comprehensive_power_data.csv` - All power system data in one file

### Plot Files (with `--save-plots`)
- `load_timeseries.png` - Total and net load time series
- `daily_profile.png` - Average daily load profiles
- `monthly_profile.png` - Monthly load profiles
- `comprehensive_analysis.png` - Combined analysis plots

### Log Files
- `logs/analysis_YYYY-MM.log` - Detailed analysis logs with timestamps

## Analysis Features

### Load Calculations
- **Total Load**: Sum of all substation active power consumption (`ss_mw_*` columns)
- **Net Load**: Total load minus wind farm generation (`wind_mw_*` columns)
- **Statistics**: Minimum, maximum, average, and standard deviation
- **Time-based Analysis**: Daily and monthly load profiles

### Wind Power Analysis
- **Total Wind Generation**: Sum of all wind farm active power (`wind_mw_*` columns)
- **Wind Statistics**: Maximum, minimum, average, and standard deviation
- **Time-based Analysis**: Wind generation profiles and patterns
- **Integration with Load Analysis**: Wind generation subtracted from total load

### Generator Categorization
- **Voltage Control Generators**: Generators with voltage setpoint data (`gen_v_*` columns)
- **PQ Control Generators**: Generators with reactive power data (`gen_mvar_*` columns)
- **Automatic Detection**: Based on data availability in voltage setpoint files

### Reactive Power Analysis
- **Total Reactive Power**: Comprehensive calculation including:
  - Substation reactive power (`ss_mvar_*` columns) - positive contribution
  - Generator reactive power (`gen_mvar_*` columns) - negative contribution
  - Shunt element reactive power (`shunt_mvar_*` columns) - positive contribution
- **Reactive Power Balance**: Analysis of reactive power flows in the system

## Data Validation

### Type Validation
- **Real Numbers**: All power and voltage measurements must be real numbers
- **Integers**: Tap positions must be integer values
- **Non-negative Values**: Wind power generation cannot be negative

### Limit Validation
- **Wind Power Limits**: 0-100 MW per wind farm
- **Substation Power Limits**: -100 to +100 MW/MVAR
- **Shunt Element Limits**: -100 to +100 MVAR
- **Generator Reactive Power Limits**: -100 to +100 MVAR
- **Voltage Limits**: 0-20 KV
- **Tap Position Limits**: 1-20

### Gap Filling
- **Linear Interpolation**: Fills gaps up to 3 time steps
- **Maximum Gap Size**: Gaps larger than 3 steps are marked as NaN
- **Data Quality**: Rows with >50% missing data are removed

### Data Quality Checks
- **Missing Data Threshold**: Maximum 10% missing data allowed
- **Minimum Records**: At least 100 valid records required
- **Timestamp Validation**: Ensures proper chronological order

## Enhanced Data Validation with Advanced Gap Filling

The enhanced data validation system provides a comprehensive three-step validation workflow that includes sophisticated gap filling techniques and advanced anomaly detection capabilities. The system implements a complete power systems data validation pipeline with intelligent gap processing.

### 📋 **Complete Validation Workflow**

**Step 1: Simple Validation**
- Basic gap filling (≤3 time steps using linear interpolation)
- Data type validation and correction
- Limit checking and constraint enforcement
- Shunt tap integer conversion

**Step 2: Comprehensive Validation** *(if enabled)*
- Advanced outlier detection using multiple statistical and ML methods
- Rate of change violation detection with gap-aware sensitivity
- Correlation anomaly identification between related variables
- Power balance validation for energy conservation
- Clustering-based anomaly detection for operational patterns

**Step 3: Advanced Gap Filling** *(if enabled)*
- **Intelligent method selection** based on gap characteristics and data patterns
- **Multiple sophisticated techniques**: spline, polynomial, KNN, ML-based imputation
- **Large gap removal**: gaps ≥24 time steps completely removed
- **Adaptive algorithms**: automatically choose optimal method per gap

### 🔬 **Advanced Gap Filling Methods**

**Intelligent Method Selection:**
- **Small gaps (≤3 steps)**: Linear interpolation for simple continuity
- **Medium gaps (4-6 steps)**: Cubic spline interpolation for smooth curves  
- **Large gaps (7-12 steps)**: KNN or polynomial based on data variance and trends
- **Very large gaps (≥24 steps)**: Complete removal to maintain data integrity

**Available Algorithms:**
- **`'adaptive'`**: Automatically selects best method based on gap size and data characteristics
- **`'spline'`**: Cubic spline interpolation with configurable smoothing
- **`'polynomial'`**: Polynomial interpolation for trend-following behavior
- **`'knn'`**: K-Nearest Neighbors using temporal features (hour, day, seasonality)
- **`'ml'`**: Random Forest-based with lagged features and cross-variable relationships

### ⚙️ **Configuration Architecture**

All validation parameters are centrally managed in `system_configuration.py` with separate sections for different validation aspects and automatic adaptation to actual data structure.

### Enhanced Validation Features

#### Statistical Outlier Detection
- **IQR (Interquartile Range)**: Detects outliers using configurable IQR multiplier (default: 1.5×IQR)
- **Z-Score**: Identifies values beyond configurable threshold (default: 3 standard deviations)
- **Modified Z-Score**: Uses median absolute deviation for robust detection (default: 3.5 threshold)
- **Isolation Forest**: ML-based anomaly detection with configurable contamination rate
- **Local Outlier Factor (LOF)**: Density-based outlier detection for local anomalies

#### Advanced Anomaly Detection
- **Rate of Change Violations**: Detects unrealistic temporal changes with adaptive thresholds
- **Correlation Anomaly Detection**: Identifies breaks in expected correlation patterns between related variables
- **Power Balance Validation**: Validates energy conservation principles with configurable tolerance
- **Clustering-Based Anomalies**: Uses DBSCAN to identify abnormal operational patterns

#### Power System Specific Validation
- **Variable Grouping**: Intelligent grouping of related power system variables
  - Generators: `gen_mvar_*` (reactive power)
  - Substations: `ss_mw_*`, `ss_mvar_*` (active/reactive power)
  - Wind Farms: `wind_mw_*` (active power)
  - Shunt Elements: `shunt_mvar_*`, `shunt_tap_*` (reactive power, tap positions)
  - Voltages: `gen_v_*` (voltage setpoints)

### Configuration Management

**Enhanced Gap Filling Settings (`system_configuration.py`):**
```python
ENHANCED_DATA_VALIDATION = {
    'advanced_gap_filling': {
        'enable_advanced_gap_filling': True,
        'default_method': 'adaptive',
        'context_size_ratio': 0.25,        # Context window size
        'min_context_points': 10,          # Minimum context needed
        'adaptive_thresholds': {
            'small_gap_size': 3,           # Linear interpolation limit
            'medium_gap_size': 6,          # Spline interpolation limit  
            'large_gap_size': 12,          # Advanced method limit
        }
    },
    'variable_groups': {
        'generators': ['gen_mvar_'],              # Generator reactive power
        'substations': ['ss_mw_', 'ss_mvar_'],    # Substation active/reactive power
        'wind': ['wind_mw_'],                     # Wind generation active power
        'shunts': ['shunt_mvar_', 'shunt_tap_'],  # Shunt compensation & tap positions
        'voltages': ['gen_v_']                    # Generator voltage setpoints
    }
}

DATA_VALIDATION = {
    'gap_filling': {
        'max_gap_steps': 3,                    # Basic method limit
        'advanced_max_gap_steps': 12,          # Advanced method limit
        'remove_large_gaps_threshold': 24,     # Complete removal threshold
        'enable_advanced_gap_filling': True    # Enable advanced processing
    }
}
```

**Statistical Validation Settings:**
```python
ENHANCED_DATA_VALIDATION = {
    'outlier_detection': {
        'default_methods': ['iqr', 'isolation_forest'],
        'contamination': 0.1,
        'zscore_threshold': 3.0,
        'modified_zscore_threshold': 3.5,
        'iqr_multiplier': 1.5,
    },
    'rate_validation': {
        'enable_rate_check': True,
        'adaptive_threshold_multiplier': 3.0,
    },
    'power_balance_validation': {
        'tolerance': 0.05,
        'epsilon': 1e-6,
    }
}
```

### Usage Examples

**Complete Enhanced Validation (Recommended):**
```python
from power_data_validator import EnhancedDataValidator

# Create enhanced validator with all features
validator = EnhancedDataValidator()

# Execute complete 3-step validation workflow
validated_data = validator.validate_dataframe_enhanced(raw_data)

# Review comprehensive results
summary = validator.get_enhanced_validation_summary()
print(f"Simple validation - Gaps filled: {summary['gaps_filled']}")
print(f"Advanced detection - Anomalies: {summary['total_anomalies']}")
```

**Advanced Gap Filling Focus:**
```python
# Apply advanced gap filling with specific method
filled_data = validator.advanced_gap_filling(data, method='adaptive')

# Test different gap filling approaches
methods = ['adaptive', 'spline', 'polynomial', 'knn']
for method in methods:
    test_data = validator.advanced_gap_filling(data.copy(), method=method)
    filled_count = data.isna().sum().sum() - test_data.isna().sum().sum()
    print(f"Method '{method}': Filled {filled_count} gaps")
```

**Step-by-Step Workflow Control:**
```python
# Manual control over each validation step
step1_data = validator.validate_dataframe(raw_data)              # Simple validation
step2_data = validator.comprehensive_anomaly_detection(step1_data)  # Anomaly detection
step3_data = validator.advanced_gap_filling(step2_data)          # Advanced gap filling

# Or with custom settings
validated_data = validator.validate_dataframe_enhanced(
    raw_data,
    use_comprehensive_anomaly_detection=True,   # Enable advanced detection
    use_advanced_gap_filling=True               # Enable sophisticated gap filling
)
```

**Gap-Aware Anomaly Detection:**
```python
# Automatic gap analysis and conservative processing
gap_analysis = validator.detect_large_gaps(data)

if gap_analysis['large_gaps_found']:
    print(f"Large gaps detected in {len(gap_analysis['gap_columns'])} columns")
    # System automatically applies conservative settings:
    # - Higher rate violation thresholds (+50%)
    # - Reduced correlation sensitivity
    # - Selective method application based on gap percentage
    
validated_data = validator.comprehensive_anomaly_detection_gap_aware(data, gap_analysis)
```

### Validation Results and Reporting

#### Enhanced Validation Summary
```python
summary = validator.get_enhanced_validation_summary()

# Available summary information:
print(f"Statistical outliers: {summary['statistical_outliers_count']}")
print(f"Rate violations: {summary['rate_violations_count']}")
print(f"Correlation anomalies: {summary['correlation_anomalies_count']}")
print(f"Power balance violations: {summary['power_balance_violations_count']}")
print(f"Clustering anomalies: {summary['clustering_anomalies_count']}")

# Detailed error messages
for error in summary['statistical_outliers']:
    print(f"Outlier detected: {error}")
```

#### Performance Metrics
The enhanced validation provides comprehensive performance reporting:
- Total records processed
- Records with errors/anomalies
- Number of gaps filled
- Processing time for each validation method
- Data quality metrics

### Integration with Analysis Pipeline

The enhanced validation seamlessly integrates with the existing analysis workflow:

1. **Standard Validation**: Type checking, limit validation, gap filling
2. **Enhanced Validation**: Advanced anomaly detection and outlier removal
3. **Analysis Processing**: Clean data ready for power system analysis
4. **Quality Reporting**: Comprehensive validation summary with actionable insights

### Configuration Customization

Users can easily customize validation behavior by modifying `system_configuration.py`:

```python
# Example: More aggressive outlier detection
ENHANCED_DATA_VALIDATION['outlier_detection']['contamination'] = 0.05
ENHANCED_DATA_VALIDATION['outlier_detection']['zscore_threshold'] = 2.5

# Example: Stricter power balance checking
ENHANCED_DATA_VALIDATION['power_balance_validation']['tolerance'] = 0.02

# Example: Custom rate validation
ENHANCED_DATA_VALIDATION['rate_validation']['adaptive_threshold_multiplier'] = 2.0
```

### Data Structure Compatibility

The enhanced validation system automatically adapts to the actual power system data structure:

- ✅ **Generators**: `gen_mvar_*` (reactive power), `gen_v_*` (voltage setpoints)
- ✅ **Substations**: `ss_mw_*` (active power), `ss_mvar_*` (reactive power)
- ✅ **Wind Farms**: `wind_mw_*` (active power)
- ✅ **Shunt Elements**: `shunt_mvar_*` (reactive power), `shunt_tap_*` (tap positions as integers)

### Advanced Gap Filling Workflow

The complete validation workflow follows this intelligent three-step process:

**Workflow Sequence:**
1. **Simple Validation**: Basic gap filling (≤3 steps), type checking, constraint validation
2. **Comprehensive Validation**: Advanced anomaly detection with gap-aware sensitivity adjustment
3. **Advanced Gap Filling**: Sophisticated interpolation methods with large gap removal

**Gap Processing Intelligence:**
- **Gap Analysis**: Pre-processing assessment of data quality and gap distribution
- **Adaptive Methods**: Automatic selection based on gap size, data variance, and trends
- **Quality Preservation**: Large gaps (≥24 steps) completely removed to maintain data integrity
- **Method Fallbacks**: Graceful degradation from advanced to simpler methods when needed

**Supported Gap Filling Methods:**
- **Linear**: Simple interpolation for small gaps (≤3 steps)
- **Spline**: Smooth cubic spline interpolation for medium gaps (4-6 steps)
- **Polynomial**: Trend-following polynomial interpolation
- **KNN**: Time-aware K-Nearest Neighbors using temporal features
- **ML**: Random Forest with lagged features and cross-variable relationships
- **Adaptive**: Intelligent method selection based on gap characteristics

**Configuration Examples:**
```python
# Enable advanced gap filling
DATA_VALIDATION['gap_filling']['enable_advanced_gap_filling'] = True
ENHANCED_DATA_VALIDATION['advanced_gap_filling']['default_method'] = 'adaptive'

# Customize gap size thresholds
ENHANCED_DATA_VALIDATION['advanced_gap_filling']['adaptive_thresholds'] = {
    'small_gap_size': 3,      # Linear interpolation limit
    'medium_gap_size': 6,     # Spline interpolation limit  
    'large_gap_size': 12,     # Advanced method limit
}

# Set removal threshold for very large gaps
DATA_VALIDATION['gap_filling']['remove_large_gaps_threshold'] = 24
```

## Detailed Calculations

### Load Calculations

#### Total Load
```python
total_load = ss_mw_columns.sum(axis=1)  # Sum across all substations
```

#### Net Load
```python
net_load = total_load - wind_mw_columns.sum(axis=1)  # Subtract wind generation
```

#### Statistics
```python
load_stats = {
    'max': net_load.max(),
    'min': net_load.min(),
    'mean': net_load.mean(),
    'std': net_load.std()
}
```

### Wind Power Calculations

#### Total Wind Generation
```python
total_wind = wind_mw_columns.sum(axis=1)  # Sum across all wind farms
```

#### Wind Statistics
```python
wind_stats = {
    'max_total_wind_mw': total_wind.max(),
    'min_total_wind_mw': total_wind.min(),
    'mean_total_wind_mw': total_wind.mean(),
    'std_total_wind_mw': total_wind.std()
}
```

### Generator Categorization

#### Voltage Control Generators
```python
voltage_control = [col.replace('gen_v_', '') for col in gen_v_columns 
                  if merged_df[col].notna().any() and (merged_df[col] != 0).any()]
```

#### PQ Control Generators
```python
pq_control = [col.replace('gen_mvar_', '') for col in gen_mvar_columns]
```

### Reactive Power Calculations

#### Total Reactive Power
```python
total_reactive = (ss_mvar_columns.sum(axis=1) + 
                 shunt_mvar_columns.sum(axis=1) - 
                 gen_mvar_columns.sum(axis=1))
```

## Data File Structure

The tool expects Excel files with the following structure:

### Substation Data
- **Active Power (MW)**: `substation_active_power.xlsx`
  - Column naming: `ss_mw_[substation_name]`
- **Reactive Power (MVAR)**: `substation_reactive_power.xlsx`
  - Column naming: `ss_mvar_[substation_name]`
- **Structure**: Timestamps in column C (row 6+), substation names in row 2, data in row 6+

### Generator Data
- **Voltage Setpoints (KV)**: `generator_voltage_setpoints.xlsx`
  - Column naming: `gen_v_[generator_name]`
- **Reactive Power (MVAR)**: `generator_reactive_power.xlsx`
  - Column naming: `gen_mvar_[generator_name]`
- **Structure**: Timestamps in column C (row 6+), generator names in row 3, data in row 6+

### Wind Farm Data
- **Active Power (MW)**: `wind_farm_active_power.xlsx`
  - Column naming: `wind_mw_[wind_farm_name]`
- **Structure**: Timestamps in column C (row 6+), wind farm names in row 3, data in row 6+

### Shunt Elements
- **Reactive Power (MVAR)**: `shunt_element_reactive_power.xlsx`
  - Column naming: `shunt_mvar_[shunt_name]` and `shunt_tap_[shunt_name]`
- **Structure**: Timestamps in column C (row 6+), shunt element names in row 3, data in row 6+

## Assumptions

### Data Processing Assumptions
1. **Time Series Continuity**: Data is assumed to be continuous with regular time intervals
2. **Unit Consistency**: All power values are in MW/MVAR, voltages in KV
3. **Sign Conventions**: 
   - Positive load values indicate consumption
   - Positive generation values indicate production
   - Generator reactive power is subtracted (negative contribution)
4. **Missing Data**: Gaps up to 3 time steps are interpolated linearly

### System Assumptions
1. **Load Balance**: Total generation equals total load plus losses
2. **Reactive Power Balance**: System maintains reactive power balance
3. **Generator Control**: Generators are either voltage-controlled or PQ-controlled
4. **Wind Integration**: Wind generation is directly subtracted from total load

### Validation Assumptions
1. **Physical Limits**: Power values are within reasonable physical limits
2. **Data Quality**: Sufficient data quality for meaningful analysis
3. **Temporal Consistency**: Data timestamps are properly ordered

## Examples

### Example 1: Basic Analysis
```bash
# Run analysis for September 2024 with all outputs
python power_analysis_cli.py 2024-09 --save-csv --save-plots --verbose

# Run analysis for all data with summary only
python power_analysis_cli.py --summary-only
```

### Example 2: Custom Data Location
```bash
# Use custom data directory
python power_analysis_cli.py 2024-09 --data-dir "my_data_folder" --output-dir "my_results"

# Use custom output directory
python power_analysis_cli.py 2024-09 --output-dir "analysis_results_2024"
```

## Sample Results

### Analysis Summary
```
==================================================
POWER SYSTEM ANALYSIS SUMMARY
==================================================

Month Filter: 2024-09

Data Overview:
  Total time points: 1393
  Time range: 2024-09-01 to 2024-09-30
  Total variables: 156

Load Analysis:
  Maximum Total Load: 834.85 MW
  Minimum Total Load: 255.68 MW
  Average Total Load: 523.81 MW
  Load Standard Deviation: 125.30 MW

Net Load Analysis:
  Maximum Net Load: 814.67 MW
  Minimum Net Load: 255.68 MW
  Average Net Load: 498.63 MW
  Net Load Standard Deviation: 123.11 MW

Wind Power Analysis:
  Maximum Total Wind: 45.23 MW
  Minimum Total Wind: 0.00 MW
  Average Total Wind: 25.18 MW
  Wind Generation Standard Deviation: 12.45 MW

Generator Analysis:
  Voltage Control Generators: 24
  PQ Control Generators: 9

Reactive Power Analysis:
  Maximum Total Reactive: 156.78 MVAR
  Minimum Total Reactive: -89.45 MVAR
  Average Total Reactive: 23.67 MVAR
```

## Development

### Architecture Overview

The tool follows a **modular, configuration-driven architecture** designed for maintainability, extensibility, and ease of use:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Layer    │    │  Configuration  │    │    Utilities   │
│                 │    │     Layer       │    │                │
│ CLI Interface   │◄──►│system_configuration│◄──►│ clean_column_   │
│ Python Scripts  │    │                 │    │ name()          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Analysis Core  │    │  Validation     │    │ Representative  │
│                 │    │     Layer       │    │   Operations    │
│power_system_    │    │power_data_      │    │                │
│analytics.py     │    │validator.py     │    │operating_point_ │
│power_system_    │    │                 │    │extractor.py     │
│visualizer.py    │    │                 │    │                │
│excel_data_      │    │                 │    │                │
│processor.py     │    │                 │    │                │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Modules

#### **Configuration Layer**
- **`system_configuration.py`** - **Centralized configuration hub** containing:
  - File mappings and column prefixes
  - Data validation parameters and limits
  - Representative operations clustering parameters
  - Output file naming conventions
  - Shared utility functions (`clean_column_name`)
  - **Rationale**: Single source of truth for all configurable parameters

#### **User Interface Layer**  
- **`power_analysis_cli.py`** - Main command-line interface with:
  - Month filtering and optimized data loading
  - Comprehensive analysis pipeline orchestration
  - Configuration-driven parameter usage
  - **Enhanced**: Now imports `clean_column_name` from system_configuration for consistency

#### **Analysis Core**
- **`power_system_analytics.py`** - Core analysis functions:
  - Load calculations (total, net load)
  - Generator categorization (voltage control vs PQ control)
  - Reactive power calculations with proper sign conventions
  - Wind power integration and statistics

- **`power_system_visualizer.py`** - Visualization functions:
  - Time series plots (load, wind, reactive power)
  - Daily and monthly profiles
  - Comprehensive analysis dashboards

- **`excel_data_processor.py`** - Data loading and preprocessing:
  - Excel file structure handling
  - Modular data loading functions for reusable analysis workflows

#### **Validation Layer**
- **`power_data_validator.py`** - Comprehensive data quality assurance:
  - Type checking and limit validation
  - Advanced gap filling with multiple interpolation methods
  - Enhanced anomaly detection (statistical, ML-based)
  - Power system specific validation rules

#### **Representative Operations**
- **`operating_point_extractor.py`** - Advanced clustering analysis:
  - K-means clustering for operating point extraction
  - **Configuration-driven**: All parameters imported from `system_configuration.py`
  - Automatic cluster quality assessment
  - MAPGL belt analysis for critical low-load conditions
  - **Enhanced**: Uses centralized configuration for all parameters

### Architectural Improvements

#### **Before: Scattered Configuration**
```python
# operating_point_extractor.py (old approach)
def extract_representative_ops(k_max=10, random_state=42):
    # Hardcoded parameters scattered in functions
    if sil > 0.25:  # Quality threshold hardcoded
        # ... clustering logic

# power_analysis_cli.py (old approach)  
def clean_column_name(col_name):
    # Duplicate utility function
    suffixes = ['_132REACTOR_REACTIVE_POWER']  # Hardcoded list
```

#### **After: Centralized Configuration**
```python
# system_configuration.py (new approach)
REPRESENTATIVE_OPS = {
    'defaults': {'k_max': 10, 'random_state': 42},
    'quality_thresholds': {'min_silhouette': 0.25},
    # ... all parameters documented and centralized
}

def clean_column_name(col_name):
    # Single, reusable utility function
    
# operating_point_extractor.py (new approach)
from system_configuration import REPRESENTATIVE_OPS
def extract_representative_ops(
    k_max=REPRESENTATIVE_OPS['defaults']['k_max']
):
    # Configuration-driven parameters
```

### Key Architectural Benefits

#### **1. Maintainability** 
- ✅ **Single Point of Change**: Modify parameters in one location
- ✅ **Consistent Values**: No risk of parameter drift across modules  
- ✅ **Clear Dependencies**: Easy to understand what each module configures
- ✅ **Reduced Code Duplication**: Shared utilities prevent inconsistencies

#### **2. Extensibility**
- ✅ **Easy Parameter Addition**: Add new config without touching multiple files
- ✅ **Module Independence**: Modules depend on config, not each other
- ✅ **Testing Flexibility**: Easy to test with different parameter sets
- ✅ **Documentation Co-location**: Parameters documented where defined

#### **3. User Experience**
- ✅ **Discoverable Options**: All customizable parameters in one place
- ✅ **Consistent Behavior**: Same parameter values across all analysis workflows
- ✅ **Easy Customization**: Modify behavior without code changes
- ✅ **Reproducible Research**: Configuration snapshots enable reproducibility

### Development Features
- **Month-based filtering** for efficient data processing
- **Complex Excel structure handling** with proper column and row mapping
- **Comprehensive data validation** with type checking, limit validation, and gap filling
- **Representative operating point extraction** using advanced clustering algorithms
- **Reactive power analysis** with proper sign conventions
- **Wind power integration** with load calculations
- **Centralized configuration management** for easy maintenance and customization
- **Comprehensive logging** with detailed progress tracking
- **Error handling** with graceful failure and recovery
- **Clean column naming** for improved output readability

### Requirements
- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- openpyxl
- scikit-learn
- scipy
- psutil

### Installation
```bash
pip install pandas numpy matplotlib seaborn openpyxl scikit-learn scipy psutil
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

## License

This tool is developed for power system analysis and research purposes by the Sustainable Power Systems Lab (SPSL).