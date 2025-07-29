API Reference
=============

This section provides detailed documentation for all modules, classes, and functions in the TSOC Data Analysis package.

.. toctree::
   :maxdepth: 2

   api/system_configuration
   api/power_system_analytics
   api/power_system_visualizer
   api/power_data_validator
   api/operating_point_extractor
   api/excel_data_processor
   api/power_analysis_cli

Package Overview
---------------

The TSOC Data Analysis package is organized into the following main modules:

**Core Analysis Modules**
- :doc:`api/power_system_analytics` - Load calculations, generator categorization, and reactive power analysis
- :doc:`api/operating_point_extractor` - Representative operating points extraction using K-means clustering

**Data Processing Modules**
- :doc:`api/excel_data_processor` - Excel file loading and data preprocessing
- :doc:`api/power_data_validator` - Comprehensive data validation and quality assurance

**Configuration and Utilities**
- :doc:`api/system_configuration` - Central configuration hub and shared utilities
- :doc:`api/power_system_visualizer` - Visualization and plotting functions

**User Interface**
- :doc:`api/power_analysis_cli` - Command-line interface and main execution functions

Quick Import Guide
-----------------

For most use cases, you can import directly from the main package:

.. code-block:: python

   from tsoc_data_analysis import (
       # Core analysis functions
       calculate_total_load,
       calculate_net_load,
       extract_representative_ops,
       
       # Data loading
       loadallpowerdf,
       
       # Configuration
       FILES,
       REPRESENTATIVE_OPS,
       
       # Validation
       DataValidator,
       
       # Main execution
       execute
   )

For advanced usage or specific module access:

.. code-block:: python

   # Import specific modules
   from tsoc_data_analysis import (
       power_system_analytics,
       operating_point_extractor,
       system_configuration
   )
   
   # Access module-specific functions
   from tsoc_data_analysis.power_system_analytics import get_load_statistics
   from tsoc_data_analysis.operating_point_extractor import loadallpowerdf 