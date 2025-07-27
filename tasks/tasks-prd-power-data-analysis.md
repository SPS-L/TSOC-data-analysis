## Relevant Files

- `power_analysis_notebook.ipynb` - The final Jupyter Notebook containing the entire analysis with structured sections for data loading, analysis, and visualization.
- `data_loader.py` - A utility script for loading Excel files, cleaning data, and merging DataFrames with proper error handling and interpolation.
- `analysis.py` - A utility script for core analysis functions including total load calculation, net load calculation, load statistics, and generator categorization.
- `plotting.py` - A utility script for visualization functions including time series plots, daily/monthly profiles, and comprehensive plotting capabilities.

### Notes

- The utility scripts (`data_loader.py`, `analysis.py`, `plotting.py`) will be developed to keep the final notebook clean and modular. They will be imported into the main notebook.
- Running the `power_analysis_notebook.ipynb` notebook will execute the entire analysis from start to finish.

## Tasks

- [x] 1.0 Setup Environment and Initial Data Loading
  - [x] 1.1 Define file paths and constants for data loading.
  - [x] 1.2 In `data_loader.py`, create a function to load a single Excel file, using the second row as the header (`header=1`) and the first column as the index (`index_col=0`).
  - [x] 1.3 In `data_loader.py`, create a main function that orchestrates loading all specified Excel files into a dictionary of DataFrames.
- [x] 2.0 Data Cleaning, Preprocessing, and Merging
  - [x] 2.1 In `data_loader.py`, create a function to clean column names (e.g., strip whitespace, remove special characters).
  - [x] 2.2 Convert the index of each DataFrame to datetime objects and ensure they are consistent.
  - [x] 2.3 In `data_loader.py`, create a function to merge all DataFrames into a single time-series DataFrame based on their timestamp index.
  - [x] 2.4 Add unique prefixes to columns from each file to prevent name collisions during the merge (e.g., `ss_mw_`, `wind_`).
  - [x] 2.5 In the merge function, apply linear interpolation to fill any missing numerical data.
- [x] 3.0 Implement Core Analysis Functions
  - [x] 3.1 In `analysis.py`, create a function `calculate_total_load` that takes the merged DataFrame and returns a Series representing the sum of all substation active power columns.
  - [x] 3.2 In `analysis.py`, create a function `calculate_net_load` that subtracts the total wind generation from the total load.
  - [x] 3.3 In `analysis.py`, create a function `get_load_statistics` to compute and return the max, min, and mean of the total load.
  - [x] 3.4 In `analysis.py`, implement the logic to categorize generators into "Voltage Control" and "PQ Control" based on the rules in FR7.
- [x] 4.0 Develop Visualization Functions
  - [x] 4.1 In `plotting.py`, create a function `plot_load_timeseries` to generate a time series plot of total load and net load.
  - [x] 4.2 In `plotting.py`, create a function `plot_daily_profile` to calculate and plot the average load profile for a typical day (24-hour cycle).
  - [x] 4.3 In `plotting.py`, create a function `plot_monthly_profile` to calculate and plot the average load profile across months.
  - [x] 4.4 Ensure all plotting functions accept a Matplotlib axes object for flexibility and include clear titles, labels, and legends.
- [x] 5.0 Finalize and Structure the Jupyter Notebook
  - [x] 5.1 Create the main Jupyter Notebook: `power_analysis_notebook.ipynb`.
  - [x] 5.2 Add Markdown cells to structure the notebook with clear headings for each analysis step (e.g., Data Loading, Analysis, Visualization).
  - [x] 5.3 Import the necessary functions from `data_loader.py`, `analysis.py`, and `plotting.py`.
  - [x] 5.4 Call the imported functions in sequence to execute the full data processing and analysis pipeline.
  - [x] 5.5 Display the key results (e.g., statistics table, plots) in the notebook output cells.
  - [x] 5.6 Review and ensure the entire notebook runs smoothly from top to bottom. 