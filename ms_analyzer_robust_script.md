# Robust MS Time Series Analysis Execution Script

## Setup

1. Save the robust analyzer script as `Script2_analyzer.py`
2. Save the configuration as `ms_analyzer_robust_config.yaml`
3. Ensure all required packages are installed:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn statsmodels click pyyaml tqdm
```

## Usage

### Create Sample Configuration

```bash
python Script2_analyzer.py --create-config "C:\dummy\path"
```

### Run Analysis with Configuration

```bash
# For your specific directory
python Script2_analyzer.py "YOUR_DATA_DIRECTORY" --config ms_analyzer_robust_config.yaml

# For any directory
python Script2_analyzer.py "YOUR_DATA_DIRECTORY" --config ms_analyzer_robust_config.yaml
```

### Specify Output Directory
 
```bash
python Script2_analyzer.py "YOUR_DATA_DIRECTORY" --output "custom_results"
```

## Key Improvements in Robust Version

1. **Clear Peak Tracking**: Each peak is represented as a `Peak` object with unique ID and clear source tracking
2. **Simplified Architecture**: Separate data structures for each scan type, combined only for analysis
3. **Robust Indexing**: No confusion between m/z indices and positions in peak lists
4. **Better Error Handling**: Comprehensive validation and informative error messages
5. **Cleaner Visualizations**: Properly handles multiple scan types in all plots

## Configuration Options

### Peak Detection
- `peak_height_threshold`: Minimum peak height
- `peak_prominence`: Minimum peak prominence
- `min_peak_intensity_across_timepoints`: Minimum intensity across time series
- `min_timepoints`: Minimum number of non-zero timepoints
- `snr_threshold`: Signal-to-noise ratio threshold

### Analysis
- `correlation_method`: Method for correlation analysis (pearson/spearman/kendall)
- `clustering_method`: Method for clustering (hierarchical/kmeans/dbscan)
- `n_clusters`: Number of clusters to find
- `trend_significance`: P-value threshold for trend detection

### Output
- `output_dir`: Directory for results
- `create_single_pdf`: Whether to create a PDF report
- `pdf_in_data_directory`: Save PDF in data directory vs output directory

## Expected Output

The analysis will generate:
1. A comprehensive PDF report with all visualizations
2. `peak_statistics.csv` - Detailed information about each detected peak
3. `correlation_matrix.csv` - Peak-to-peak correlations
4. `analysis_config.yaml` - Configuration used for the analysis

## Troubleshooting

If you encounter issues:
1. Check that all .txt files have 'cyc' followed by numbers in their names
2. Ensure files contain two columns (m/z and intensity)
3. Verify the configuration parameters are appropriate for your data
4. Check the log output for specific error messages
