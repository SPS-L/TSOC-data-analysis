# Product Requirements Document: Power System Data Analysis Script

## 1. Introduction/Overview

This document outlines the requirements for a Python script designed to be a reusable tool for processing power system operational data. The script will load timeseries data from multiple Excel files, combine them into a single coherent dataset, and perform various analyses. The target user is a power systems engineer or data analyst who needs to regularly process and analyze this data for studies and reporting. The final output will be a Jupyter Notebook containing the code, analysis, and visualizations.

## 2. Goals

- **G1:** To automate the loading and merging of power system data from various Excel files into a single, clean pandas DataFrame.
- **G2:** To implement a set of standard analyses, including total load, net load, and load statistics (min, max, mean).
- **G3:** To create clear and informative visualizations of the data, such as daily and monthly load profiles.
- **G4:** To provide a well-documented and reusable Jupyter Notebook that can be easily adapted for future analyses.

## 3. User Stories

- **As a power systems engineer,** I want to combine various Excel data sources into one dataset so that I can perform a comprehensive analysis of system behavior without manual data wrangling.
- **As a data analyst,** I want to automatically calculate and visualize daily and monthly load profiles to quickly identify and report on demand patterns.
- **As a grid planner,** I want to assess the impact of renewable energy on the net load to better inform our grid planning and investment decisions.

## 4. Functional Requirements

- **FR1: Data Loading:** The script must load all specified Excel (`.xlsx`) files from the `/raw_data` directory.
- **FR2: Header and Index:**
    - The script must use the second row of the Excel files to determine the names of substations, generators, etc., and use them as column headers.
    - It must identify the timestamp column and set it as the index for each DataFrame.
- **FR3: Data Merging:** All individual DataFrames must be merged into a single timeseries DataFrame, using the timestamp as the common key.
- **FR4: Handling Missing Data:** Any missing numerical values in the final merged DataFrame must be filled using linear interpolation.
- **FR5: Analysis Calculations:** The script must compute the following:
    - Total Load (sum of all substation active power loads from `SS_MW_30MIN.xlsx`).
    - Total Net Load (Total Load - sum of active power from `Wind_Farms_Active_Power.xlsx`).
    - Maximum, minimum, and mean values for the Total Load.
- **FR6: Visualizations:** The script must generate and display:
    - A time series plot of Total Load and Total Net Load over the entire period.
    - Plots showing the average load profile for a typical day.
    - Plots showing the average load profile for a typical month.
- **FR7: Generator Control Logic:**
    - Generators listed in `Gen_Voltage.xlsx` with non-zero voltage setpoints are to be considered in 'Voltage Control'.
    - Generators listed in `Q_Steam_Vass_&_ICE.xlsx` are to be considered in 'PQ Control'.
- **FR8: Output Format:** The entire process, including data loading, analysis, and visualizations, must be encapsulated in a single Jupyter Notebook (`.ipynb`) file.
- **FR9: Error Handling:** The script must fail fast. If any file is not found, or a required column/row for headers is missing, the script must stop execution and report a clear error message.

## 5. Non-Goals (Out of Scope)

- The script will not perform real-time data processing.
- The script will not have a graphical user interface (GUI).
- The script will not save the final processed data into a new file format unless explicitly added as a new requirement.

## 6. Design Considerations

- The Jupyter Notebook should be well-structured with Markdown cells to clearly explain each step of the process (e.g., "Loading Data", "Cleaning and Merging", "Performing Analysis").
- All plots must have clear titles, axis labels, and legends to be easily understandable.

## 7. Technical Considerations

- **Libraries:** The script will primarily use `pandas` for data manipulation and `matplotlib` and/or `seaborn` for plotting. `openpyxl` will be required for reading `.xlsx` files.
- **Modularity:** The code should be organized into functions within the notebook to improve readability and reusability (e.g., `load_data()`, `calculate_metrics()`, `plot_profiles()`).

## 8. Success Metrics

- The script successfully runs from start to finish without errors, processing all specified input files.
- The final Jupyter Notebook is generated and contains the combined dataset along with all required analyses and visualizations.
- The calculated metrics (total load, net load, etc.) are accurate and correctly reflect the input data.

## 9. Open Questions

- For now, we assume the timestamp column is named consistently across all files. If not, this might require a more robust column-finding logic. 