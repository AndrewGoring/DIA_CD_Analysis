#!/usr/bin/env python3
"""
Mass Spectrometry Time Series Analysis Suite - ROBUST VERSION
============================================================

A robust implementation for analyzing mass spectrometry time series data
with proper handling of multiple scan types and peak indexing.

Author: MS Analysis Suite
Version: 2.0 - Robust Multi-Scan Implementation
"""

import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import json

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from statsmodels.stats.multitest import multipletests
import click
import yaml
from tqdm import tqdm

# Configure matplotlib for publication-quality figures
mplstyle.use(['seaborn-v0_8', 'seaborn-v0_8-whitegrid'])
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'png',
    'savefig.bbox': 'tight'
})

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrendType(Enum):
    """Enumeration of temporal trend types"""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    OSCILLATING = "oscillating"
    UNKNOWN = "unknown"


@dataclass
class Peak:
    """Represents a single peak with all its metadata"""
    mz_index: int  # Index in the m/z array
    mz_value: float  # Actual m/z value
    scan_type: str  # Which scan type this peak belongs to
    position_in_scan: int  # Position in the scan type's peak list
    intensities: np.ndarray  # Time series intensities
    cluster_id: Optional[int] = None
    trend_type: Optional[TrendType] = None

    @property
    def mean_intensity(self) -> float:
        return np.mean(self.intensities)

    @property
    def std_intensity(self) -> float:
        return np.std(self.intensities)

    @property
    def cv_intensity(self) -> float:
        mean_val = self.mean_intensity
        return self.std_intensity / mean_val if mean_val > 0 else np.inf

    @property
    def unique_id(self) -> str:
        """Unique identifier for this peak"""
        return f"{self.scan_type}_{self.mz_index}"


@dataclass
class ScanTypeData:
    """Container for data from a single scan type"""
    scan_type: str
    mz_values: np.ndarray
    intensities: np.ndarray  # Shape: (n_mz, n_timepoints)
    timepoints: np.ndarray
    detected_peaks: List[Peak] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CombinedAnalysisData:
    """Container for combined analysis results"""
    all_peaks: List[Peak]
    correlation_matrix: np.ndarray
    correlation_pvalues: np.ndarray
    cluster_labels: np.ndarray
    trend_classifications: List[TrendType]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisConfig:
    """Configuration parameters for MS time series analysis"""
    # Peak detection parameters
    peak_height_threshold: float = 1000.0
    peak_prominence: float = 500.0
    peak_prominence: float = 500.0
    min_peak_intensity_across_timepoints: float = 100.0
    min_timepoints: int = 5

    #naming
    scan_string: str = "scan"
    cycle_string: str = "cyc"

    # Predefined peaks input
    use_predefined_peaks: bool = False  # Set True to use predefined peak list instead of automatic detection
    predefined_peak_list: str = ""  # Path to Excel/CSV/TSV file with columns: Index, Scan, Peak

    # Normalization method
    use_reference_cycle_normalization: bool = False  # Set True to normalize to reference cycle range

    # Peak width configuration
    use_mz_width_range: bool = True
    mz_width_range: Tuple[float, float] = (0.1, 5.0)

    # S/N parameters
    snr_threshold: float = 3.0
    snr_window_size: int = 50
    baseline_correction: bool = True

    # Detection spectrum selection
    peak_detection_spectrum: str = "first"
    cycle_range_start: Optional[int] = None
    cycle_range_end: Optional[int] = None

    # Statistical parameters
    correlation_method: str = "pearson"
    significance_threshold: float = 0.05
    multiple_testing_method: str = "fdr_bh"
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95

    # Clustering parameters
    clustering_method: str = "hierarchical"
    n_clusters: int = 5
    db_eps: float = 0.2
    db_min_samples: int = 2

    # Trend classification
    trend_significance: float = 0.05
    oscillation_threshold: float = 2.0

    # Output parameters
    output_dir: str = "results"
    create_single_pdf: bool = True
    pdf_in_data_directory: bool = True

    # Sub-clustering parameters
    enable_subclustering: bool = True
    subcluster_large_clusters: bool = True
    subcluster_threshold: int = 50  # Subcluster if cluster has more than this many peaks
    n_subclusters: int = 10  # Number of subclusters to create
    subclustering_method: str = "hierarchical"  # "kmeans", "hierarchical", or "dbscan"

    # Peak overlap analysis
    analyze_overlaps: bool = True
    overlap_threshold_mz: float = 1.0  # m/z units to consider peaks overlapping
    remove_overlapping_peaks: bool = False  # Whether to remove overlapping peaks
    overlap_correlation_threshold: float = 0.8  # Correlation threshold for overlap removal

    # Alternative clustering approaches
    use_trend_clustering: bool = False  # Cluster by slope/R² instead of time series
    trend_features: List[str] = field(default_factory=lambda: ["slope", "intercept", "r_squared", "cv"])
    n_trend_clusters: int = 8

    # Advanced visualization options
    create_diagnostic_plots: bool = True
    max_peaks_to_plot: int = 50  # Maximum peaks to show in diagnostic plots
    plot_top_n_variable_peaks: int = 20  # Number of most variable peaks to highlight

    # PCA analysis parameters
    perform_pca_analysis: bool = True
    n_pca_components: int = 3
    plot_pca_variance: bool = True

    # Time series preprocessing
    detrend_before_clustering: bool = False  # Remove linear trend before clustering
    smooth_before_clustering: bool = False  # Apply smoothing before clustering
    smoothing_window_size: int = 3

    # Quality control parameters
    min_peak_quality_score: float = 0.0  # Minimum quality score (e.g., based on S/N)
    remove_low_quality_peaks: bool = False
    quality_metrics: List[str] = field(default_factory=lambda: ["snr", "cv", "presence"])

    # Correlation-based filtering
    min_correlation_with_cluster: float = 0.3  # Minimum correlation to belong to a cluster
    reassign_uncorrelated_peaks: bool = True  # Reassign peaks with low cluster correlation

    # Export options
    export_subclusters: bool = True  # Export subcluster assignments
    export_peak_features: bool = True  # Export extracted features (slope, R², etc.)
    export_format: str = "csv"  # "csv" or "excel"

    @classmethod
    def from_yaml(cls, config_path: str) -> 'AnalysisConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Flatten nested structure
        flat_config = {}
        for section, params in config_dict.items():
            if isinstance(params, dict):
                flat_config.update(params)
            else:
                flat_config[section] = params

        # Create config with only valid parameters
        valid_params = {k: v for k, v in flat_config.items()
                       if k in cls.__annotations__}

        return cls(**valid_params)


class RobustMSAnalyzer:
    """Robust mass spectrometry time series analyzer"""

    def __init__(self, config: Optional[AnalysisConfig] = None):
        """Initialize analyzer with configuration"""
        self.config = config or AnalysisConfig()
        self.scan_data: Dict[str, ScanTypeData] = {}
        self.combined_analysis: Optional[CombinedAnalysisData] = None
        self._data_directory: Optional[str] = None

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def _normalize_intensities(self, peak: Peak) -> np.ndarray:
        """Normalize peak intensities based on configuration

        If use_reference_cycle_normalization = True:
            Divides by mean of reference cycle range (cycle_range_start:cycle_range_end)
        Else:
            Divides by sum of all intensities (original behavior)

        Args:
            peak: Peak object with intensities to normalize

        Returns:
            Normalized intensity array
        """
        if self.config.use_reference_cycle_normalization:
            # NEW METHOD: Normalize to reference range mean
            start = self.config.cycle_range_start
            end = self.config.cycle_range_end

            ref_intensity = np.mean(peak.intensities[start:end])

            if ref_intensity > 0:
                return peak.intensities / ref_intensity
            else:
                logger.warning(
                    f"Peak at m/z {peak.mz_value:.2f} in {peak.scan_type} has "
                    f"zero/low intensity in reference range [{start}:{end}]. "
                    f"Falling back to sum normalization."
                )
                # Fall back to sum normalization
                peak_sum = np.sum(peak.intensities)
                return peak.intensities / peak_sum if peak_sum > 0 else peak.intensities
        else:
            # ORIGINAL METHOD: Normalize by sum
            peak_sum = np.sum(peak.intensities)
            if peak_sum > 0:
                return peak.intensities / peak_sum
            else:
                return peak.intensities

    def load_data(self, data_directory: str, file_pattern: str = f"*scan*.txt") -> None:
        """Load MS data files from directory"""
        logger.info(f"Loading data from {data_directory}")
        self._data_directory = data_directory

        data_path = Path(data_directory)
        if not data_path.exists():
            raise FileNotFoundError(f"Directory not found: {data_directory}")

        # Find all matching files
        file_pattern = f"*{self.config.scan_string}*.txt"
        file_paths = list(data_path.glob(file_pattern))
        if not file_paths:
            raise FileNotFoundError(f"No files matching pattern: {file_pattern}")

        logger.info(f"Found {len(file_paths)} files")

        # Group files by scan type and cycle
        scan_files = {}
        for file_path in file_paths:
            # Extract scan type and cycle
            cycle_match = re.search(r'cyc(\d+)', file_path.name)
            scan_match = re.search(r'scan(\d+)', file_path.name)

            if not cycle_match:
                logger.warning(f"No cycle number in: {file_path.name}")
                continue

            cycle_num = int(cycle_match.group(1))
            scan_type = f"scan{scan_match.group(1)}" if scan_match else 'main'

            if scan_type not in scan_files:
                scan_files[scan_type] = []
            scan_files[scan_type].append((cycle_num, file_path))

        logger.info(f"Found scan types: {list(scan_files.keys())}")

        # Process each scan type
        for scan_type, files in scan_files.items():
            self._load_scan_type(scan_type, files)

    def _load_scan_type(self, scan_type: str, files: List[Tuple[int, Path]]) -> None:
        """Load data for a single scan type"""
        # Sort by cycle number
        files.sort(key=lambda x: x[0])

        logger.info(f"Loading {scan_type} with {len(files)} cycles")

        # Load all spectra
        all_spectra = []
        cycle_numbers = []

        for cycle_num, file_path in tqdm(files, desc=f"Loading {scan_type}"):
            try:
                mz, intensity = self._load_spectrum_file(file_path)
                all_spectra.append((mz, intensity))
                cycle_numbers.append(cycle_num)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        if not all_spectra:
            logger.warning(f"No valid files for {scan_type}")
            return

        # Create common m/z grid
        all_mz = np.concatenate([spec[0] for spec in all_spectra])
        mz_min, mz_max = np.min(all_mz), np.max(all_mz)
        mz_resolution = np.median(np.diff(all_spectra[0][0]))
        common_mz = np.arange(mz_min, mz_max + mz_resolution, mz_resolution)

        # Interpolate to common grid
        intensity_matrix = []
        for mz, intensity in all_spectra:
            interp_intensity = np.interp(common_mz, mz, intensity, left=0, right=0)
            intensity_matrix.append(interp_intensity)

        intensity_matrix = np.array(intensity_matrix).T  # Shape: (n_mz, n_timepoints)

        # Store scan data
        self.scan_data[scan_type] = ScanTypeData(
            scan_type=scan_type,
            mz_values=common_mz,
            intensities=intensity_matrix,
            timepoints=np.array(cycle_numbers),
            metadata={
                'n_timepoints': len(cycle_numbers),
                'n_mz_points': len(common_mz),
                'mz_range': (mz_min, mz_max),
                'cycle_range': (min(cycle_numbers), max(cycle_numbers))
            }
        )

        logger.info(f"Loaded {scan_type}: {len(common_mz)} m/z points, {len(cycle_numbers)} timepoints")

    def _load_spectrum_file(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load a single spectrum file"""
        # Try different delimiters
        for delimiter in ['\t', ',', ' ', ';']:
            try:
                data = np.loadtxt(file_path, delimiter=delimiter)
                if data.shape[1] >= 2:
                    return data[:, 0], data[:, 1]
            except:
                continue

        # Try pandas if numpy fails
        try:
            df = pd.read_csv(file_path, sep=None, engine='python')
            if df.shape[1] >= 2:
                return df.iloc[:, 0].values, df.iloc[:, 1].values
        except:
            pass

        raise ValueError(f"Could not parse file: {file_path}")

    def _load_predefined_peaks(self) -> None:
        """Load peaks from predefined list (Excel/CSV/TSV file)"""
        logger.info(f"Loading predefined peaks from {self.config.predefined_peak_list}")
        
        # Determine file format and load
        file_path = Path(self.config.predefined_peak_list)
        if not file_path.exists():
            raise FileNotFoundError(f"Predefined peak list not found: {file_path}")
        
        # Load the file
        file_ext = file_path.suffix.lower()
        if file_ext in ['.xlsx', '.xls']:
            peak_df = pd.read_excel(file_path)
        elif file_ext == '.csv':
            peak_df = pd.read_csv(file_path)
        elif file_ext in ['.tsv', '.txt']:
            peak_df = pd.read_csv(file_path, sep='\t')
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Use .xlsx, .csv, or .tsv")
        
        # Validate columns
        required_cols = ['Index', 'Scan', 'Peak']
        if not all(col in peak_df.columns for col in required_cols):
            raise ValueError(f"Predefined peak list must have columns: {required_cols}. Found: {peak_df.columns.tolist()}")
        
        logger.info(f"Loaded {len(peak_df)} predefined peaks")
        
        # Group peaks by scan type
        peak_groups = {}
        for _, row in peak_df.iterrows():
            scan_num = int(row['Scan'])
            scan_type = f"scan{scan_num}"
            mz_value = float(row['Peak'])
            
            if scan_type not in peak_groups:
                peak_groups[scan_type] = []
            peak_groups[scan_type].append(mz_value)
        
        logger.info(f"Found peaks for scan types: {list(peak_groups.keys())}")
        
        # Create Peak objects for each predefined peak
        for scan_type, mz_values in peak_groups.items():
            if scan_type not in self.scan_data:
                logger.warning(f"Scan type {scan_type} not found in loaded data, skipping")
                continue
            
            scan_data = self.scan_data[scan_type]
            peaks_created = []
            
            for mz_value in mz_values:
                # Find closest m/z index in the data
                mz_idx = np.argmin(np.abs(scan_data.mz_values - mz_value))
                actual_mz = scan_data.mz_values[mz_idx]
                
                # Extract intensities for this peak across all timepoints
                peak_intensities = scan_data.intensities[mz_idx, :].copy()
                
                # Create Peak object
                peak = Peak(
                    mz_index=mz_idx,
                    mz_value=actual_mz,
                    scan_type=scan_type,
                    position_in_scan=len(peaks_created),
                    intensities=peak_intensities
                )
                peaks_created.append(peak)
            
            scan_data.detected_peaks = peaks_created
            logger.info(f"Created {len(peaks_created)} peaks for {scan_type}")

    def detect_peaks_all_scans(self) -> None:
        """Detect peaks in all scan types (or load from predefined list)"""
        if self.config.use_predefined_peaks:
            logger.info("Using predefined peak list instead of automatic detection")
            self._load_predefined_peaks()
        else:
            logger.info("Performing automatic peak detection")
            for scan_type, scan_data in self.scan_data.items():
                logger.info(f"Detecting peaks in {scan_type}")
                self._detect_peaks_for_scan(scan_data)
                logger.info(f"Detected {len(scan_data.detected_peaks)} peaks in {scan_type}")

    def _detect_peaks_for_scan(self, scan_data: ScanTypeData) -> None:
        """Detect peaks for a single scan type"""
        # Select reference spectrum
        if self.config.peak_detection_spectrum == "first":
            ref_spectrum = scan_data.intensities[:, 0]
        elif self.config.peak_detection_spectrum == "mean":
            ref_spectrum = np.mean(scan_data.intensities, axis=1)
        elif self.config.peak_detection_spectrum == "median":
            ref_spectrum = np.median(scan_data.intensities, axis=1)
        elif self.config.peak_detection_spectrum == "range":
            ref_spectrum = np.mean(scan_data.intensities[:,self.config.cycle_range_start:self.config.cycle_range_end], axis=1)
        else:
            ref_spectrum = scan_data.intensities[:, 0]

        # Apply baseline correction if needed
        if self.config.baseline_correction:
            ref_spectrum = self._apply_baseline_correction(ref_spectrum)

        # Calculate peak width range
        mz_spacing = np.median(np.diff(scan_data.mz_values))
        min_width = max(1, int(self.config.mz_width_range[0] / mz_spacing))
        max_width = int(self.config.mz_width_range[1] / mz_spacing)

        # Find peaks
        peaks, properties = signal.find_peaks(
            ref_spectrum,
            height=self.config.peak_height_threshold,
            prominence=self.config.peak_prominence,
            width=(min_width, max_width)
        )

        # Filter peaks
        filtered_peaks = []
        for i, peak_idx in enumerate(peaks):
            # Check temporal consistency
            peak_intensities = scan_data.intensities[peak_idx, :]
            non_zero_count = np.sum(peak_intensities > 0)

            if non_zero_count < self.config.min_timepoints:
                continue

            # Check minimum intensity
            if np.max(peak_intensities) < self.config.min_peak_intensity_across_timepoints:
                continue

            # Check S/N ratio
            noise = self._estimate_local_noise(ref_spectrum, peak_idx)
            if ref_spectrum[peak_idx] / noise < self.config.snr_threshold:
                continue

            # Create Peak object
            peak = Peak(
                mz_index=peak_idx,
                mz_value=scan_data.mz_values[peak_idx],
                scan_type=scan_data.scan_type,
                position_in_scan=len(filtered_peaks),
                intensities=peak_intensities.copy()
            )
            filtered_peaks.append(peak)

        scan_data.detected_peaks = filtered_peaks

    def _apply_baseline_correction(self, spectrum: np.ndarray) -> np.ndarray:
        """Apply baseline correction"""
        window_size = min(100, len(spectrum) // 10)
        baseline = np.zeros_like(spectrum)

        for i in range(len(spectrum)):
            start = max(0, i - window_size // 2)
            end = min(len(spectrum), i + window_size // 2)
            baseline[i] = np.percentile(spectrum[start:end], 10)

        from scipy.ndimage import gaussian_filter1d
        baseline = gaussian_filter1d(baseline, sigma=2)

        corrected = spectrum - baseline
        corrected[corrected < 0] = 0
        return corrected

    def _estimate_local_noise(self, spectrum: np.ndarray, peak_idx: int) -> float:
        """Estimate local noise around a peak"""
        window = self.config.snr_window_size
        start = max(0, peak_idx - window // 2)
        end = min(len(spectrum), peak_idx + window // 2)

        local_data = spectrum[start:end]
        # Use bottom 60% of data as noise
        threshold = np.percentile(local_data, 60)
        noise_data = local_data[local_data <= threshold]

        if len(noise_data) > 3:
            return np.sqrt(np.mean(noise_data ** 2))
        else:
            return np.std(local_data)

    def combine_and_analyze(self) -> None:
        """Combine peaks from all scan types and perform analysis"""
        # Collect all peaks
        all_peaks = []
        for scan_type, scan_data in self.scan_data.items():
            all_peaks.extend(scan_data.detected_peaks)

        if not all_peaks:
            raise ValueError("No peaks detected in any scan type")

        logger.info(f"Analyzing {len(all_peaks)} peaks from {len(self.scan_data)} scan types")

        # Calculate correlations
        correlation_matrix, pvalues = self._calculate_correlations(all_peaks)

        # Perform clustering
        cluster_labels = self._perform_clustering(all_peaks, correlation_matrix)

        # Classify trends
        trend_classifications = self._classify_trends(all_peaks)

        # Update peak objects
        for i, peak in enumerate(all_peaks):
            peak.cluster_id = cluster_labels[i]
            peak.trend_type = trend_classifications[i]

        # Store results
        self.combined_analysis = CombinedAnalysisData(
            all_peaks=all_peaks,
            correlation_matrix=correlation_matrix,
            correlation_pvalues=pvalues,
            cluster_labels=cluster_labels,
            trend_classifications=trend_classifications,
            metadata={
                'n_peaks': len(all_peaks),
                'n_scan_types': len(self.scan_data),
                'analysis_timestamp': datetime.now().isoformat()
            }
        )

        # Calculate bootstrap confidence intervals
        logger.info("Calculating bootstrap confidence intervals")
        if self.config.bootstrap_samples >=5:
            bootstrap_ci = self.calculate_bootstrap_confidence_intervals()
        else:
            logger.warning("Bootstrap confidence intervals will not be calculated")
            bootstrap_ci = None
        # Update metadata with bootstrap confidence intervals
        self.combined_analysis.metadata['bootstrap_ci'] = bootstrap_ci

    def _calculate_correlations(self, peaks: List[Peak]) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate correlation matrix between peaks"""
        n_peaks = len(peaks)
        correlation_matrix = np.zeros((n_peaks, n_peaks))
        pvalue_matrix = np.zeros((n_peaks, n_peaks))

        # Normalize intensities
        normalized_intensities = []
        # Normalize intensities
        normalized_intensities = []
        for peak in peaks:
            # Normalize (method depends on config)
            norm_intensities = self._normalize_intensities(peak)
            # Log transform
            log_intensities = np.log1p(norm_intensities) / np.log(10)
            normalized_intensities.append(log_intensities)

        # Calculate correlations
        for i in range(n_peaks):
            for j in range(n_peaks):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                    pvalue_matrix[i, j] = 0.0
                else:
                    if self.config.correlation_method == "pearson":
                        corr, pval = stats.pearsonr(normalized_intensities[i], 
                                                   normalized_intensities[j])
                    elif self.config.correlation_method == "spearman":
                        corr, pval = stats.spearmanr(normalized_intensities[i], 
                                                    normalized_intensities[j])
                    else:
                        corr, pval = stats.kendalltau(normalized_intensities[i], 
                                                     normalized_intensities[j])

                    correlation_matrix[i, j] = corr
                    pvalue_matrix[i, j] = pval

        return correlation_matrix, pvalue_matrix

    def calculate_bootstrap_confidence_intervals(self) -> Dict[str, np.ndarray]:
        """Calculate bootstrap confidence intervals for correlations"""
        if not self.combined_analysis:
            raise ValueError("No analysis results available")

        logger.info("Calculating bootstrap confidence intervals")

        peaks = self.combined_analysis.all_peaks
        n_peaks = len(peaks)

        # Group peaks by scan type to get proper timepoints
        peaks_by_scan = {}
        for peak in peaks:
            if peak.scan_type not in peaks_by_scan:
                peaks_by_scan[peak.scan_type] = []
            peaks_by_scan[peak.scan_type].append(peak)

        # Process each scan type separately
        bootstrap_correlations = np.zeros((self.config.bootstrap_samples, n_peaks, n_peaks))

        for boot_idx in tqdm(range(self.config.bootstrap_samples), desc="Bootstrap sampling"):
            # Create resampled data for each scan type
            resampled_intensities = []

            for scan_type, scan_peaks in peaks_by_scan.items():
                scan_data = self.scan_data[scan_type]
                n_timepoints = len(scan_data.timepoints)

                # Resample timepoints with replacement
                boot_indices = np.random.choice(n_timepoints, n_timepoints, replace=True)

                # Extract resampled intensities for each peak in this scan type
                for peak in scan_peaks:
                    # Get original intensities
                    intensities = peak.intensities

                    # Resample
                    boot_intensities = intensities[boot_indices]

                    # Normalize and log transform
                    # Create temporary peak for normalization
                    temp_peak = Peak(
                        mz_index=peak.mz_index,
                        mz_value=peak.mz_value,
                        scan_type=peak.scan_type,
                        position_in_scan=peak.position_in_scan,
                        intensities=boot_intensities
                    )
                    norm_intensities = self._normalize_intensities(temp_peak)
                    log_intensities = np.log1p(norm_intensities) / np.log(10)

                    resampled_intensities.append(log_intensities)

            # Calculate correlation matrix for bootstrap sample
            for i in range(n_peaks):
                for j in range(n_peaks):
                    if i == j:
                        bootstrap_correlations[boot_idx, i, j] = 1.0
                    else:
                        if self.config.correlation_method == "pearson":
                            corr, _ = stats.pearsonr(resampled_intensities[i], resampled_intensities[j])
                        elif self.config.correlation_method == "spearman":
                            corr, _ = stats.spearmanr(resampled_intensities[i], resampled_intensities[j])
                        else:
                            corr, _ = stats.kendalltau(resampled_intensities[i], resampled_intensities[j])

                        bootstrap_correlations[boot_idx, i, j] = corr

        # Calculate confidence intervals
        alpha = 1 - self.config.confidence_level
        lower_percentile = 100 * (alpha / 2)
        upper_percentile = 100 * (1 - alpha / 2)

        correlation_ci_lower = np.percentile(bootstrap_correlations, lower_percentile, axis=0)
        correlation_ci_upper = np.percentile(bootstrap_correlations, upper_percentile, axis=0)

        return {
            'lower': correlation_ci_lower,
            'upper': correlation_ci_upper,
            'bootstrap_samples': bootstrap_correlations
        }

    def _perform_clustering(self, peaks: List[Peak], correlation_matrix: np.ndarray) -> np.ndarray:
        """Perform clustering on peaks"""
        if self.config.clustering_method == "hierarchical":
            # Use correlation distance
            distance_matrix = 1 - np.abs(correlation_matrix)
            linkage_matrix = linkage(squareform(distance_matrix), method='average')
            cluster_labels = fcluster(linkage_matrix, self.config.n_clusters, 
                                    criterion='maxclust') - 1
        else:
            # Use normalized intensities
            normalized = []
            for peak in peaks:
                norm = self._normalize_intensities(peak)
                norm = np.log1p(norm) / np.log(10)

                # Smooth if enabled
                if self.config.smooth_before_clustering:
                    from scipy.signal import savgol_filter
                    if len(norm) > self.config.smoothing_window_size:
                        norm = savgol_filter(norm,
                                                    window_length=self.config.smoothing_window_size,
                                                    polyorder=2)

                normalized.append(norm)

            normalized = np.array(normalized)

            if self.config.clustering_method == "kmeans":

                inertias = []
                k_range = range(1, min(len(peaks),30))  # Test k from 1 to 30 or # peaks

                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42)  # n_init for robustness
                    kmeans.fit(normalized)
                    inertias.append(kmeans.inertia_)
                plt.figure(figsize=(8, 6))
                plt.plot(k_range, inertias, marker='o')
                plt.title('Elbow Method for Optimal K')
                plt.xlabel('Number of Clusters (K)')
                plt.ylabel('Inertia (Within-cluster Sum of Squares)')
                plt.grid(True)
                plt.show()

                plt.figure(figsize=(8, 6))
                plt.plot(k_range, inertias, marker='o')
                plt.title('Elbow Method for Optimal K')
                plt.xlabel('Number of Clusters (K)')
                plt.ylabel('Inertia (Within-cluster Sum of Squares)')
                plt.grid(True)
                plt.show()

                clusterer = KMeans(n_clusters=self.config.n_clusters, random_state=42)
                cluster_labels = clusterer.fit_predict(normalized)
            else:
                clusterer = DBSCAN(eps=self.config.db_eps, min_samples=self.config.db_min_samples)
                cluster_labels = clusterer.fit_predict(normalized)

        return cluster_labels

    def diagnose_clustering(self) -> None:
        """Diagnose why peaks are clustering the way they are"""
        if not self.combined_analysis:
            raise ValueError("No analysis results available")

        logger.info("Diagnosing clustering results")

        # Get cluster 4 peaks (the large cluster)
        large_cluster_id = 0  # Adjust if needed
        large_cluster_peaks = [p for p in self.combined_analysis.all_peaks
                               if p.cluster_id == large_cluster_id]

        logger.info(f"Cluster {large_cluster_id} contains {len(large_cluster_peaks)} peaks")

        # Analyze normalized intensities
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Plot all normalized time series in the large cluster
        for peak in large_cluster_peaks[:50]:  # First 50 to avoid overcrowding
            norm = self._normalize_intensities(peak)
            log_int = np.log1p(norm) / np.log(10)
            timepoints = self.scan_data[peak.scan_type].timepoints
            ax1.plot(timepoints, log_int, alpha=0.3, linewidth=0.5)

        ax1.set_xlabel('Time Point')
        ax1.set_ylabel('Log(Normalized Intensity)')
        ax1.set_title(f'First 50 Time Series in Cluster {large_cluster_id}')
        ax1.grid(True, alpha=0.3)

        # 2. Distribution of correlation coefficients within large cluster
        within_cluster_corrs = []
        for i in range(min(100, len(large_cluster_peaks))):
            for j in range(i + 1, min(100, len(large_cluster_peaks))):
                peak_i = large_cluster_peaks[i]
                peak_j = large_cluster_peaks[j]

                # Get positions in correlation matrix
                idx_i = self.combined_analysis.all_peaks.index(peak_i)
                idx_j = self.combined_analysis.all_peaks.index(peak_j)

                corr = self.combined_analysis.correlation_matrix[idx_i, idx_j]
                within_cluster_corrs.append(corr)

        ax2.hist(within_cluster_corrs, bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(within_cluster_corrs), color='red', linestyle='--',
                    label=f'Mean: {np.mean(within_cluster_corrs):.3f}')
        ax2.set_xlabel('Correlation Coefficient')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Within-Cluster Correlations (Cluster {large_cluster_id})')
        ax2.legend()

        # 3. PCA of large cluster
        large_cluster_data = []
        for peak in large_cluster_peaks:
            norm = self._normalize_intensities(peak)
            log_int = np.log1p(norm) / np.log(10)
            large_cluster_data.append(log_int)

        large_cluster_data = np.array(large_cluster_data)

        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(large_cluster_data)

        scatter = ax3.scatter(pca_result[:, 0], pca_result[:, 1],
                              c=pca_result[:, 2], cmap='viridis',
                              s=30, alpha=0.6)
        ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax3.set_title(f'PCA of Cluster {large_cluster_id} (colored by PC3)')
        plt.colorbar(scatter, ax=ax3, label='PC3')

        # 4. Coefficient of variation distribution
        cv_values = [peak.cv_intensity for peak in large_cluster_peaks]
        ax4.hist(cv_values, bins=50, alpha=0.7, edgecolor='black')
        ax4.axvline(np.median(cv_values), color='red', linestyle='--',
                    label=f'Median CV: {np.median(cv_values):.3f}')
        ax4.set_xlabel('Coefficient of Variation')
        ax4.set_ylabel('Frequency')
        ax4.set_title(f'CV Distribution in Cluster {large_cluster_id}')
        ax4.legend()

        plt.tight_layout()
        plt.savefig(self.config.output_dir + '/cluster_diagnostics.png', dpi=300)
        plt.show()

        # Print summary statistics
        logger.info(f"Cluster {large_cluster_id} statistics:")
        logger.info(f"  Mean within-cluster correlation: {np.mean(within_cluster_corrs):.3f}")
        logger.info(f"  PCA variance explained: {pca.explained_variance_ratio_[:3]}")
        logger.info(f"  CV range: {np.min(cv_values):.3f} - {np.max(cv_values):.3f}")

    def _visualize_subclusters(self, peaks: List, labels: np.ndarray,
                               parent_cluster: int, method: str) -> None:
        """Visualize sub-clustering results"""
        unique_labels = np.unique(labels)
        n_subclusters = len(unique_labels[unique_labels >= 0])  # Exclude noise (-1) for DBSCAN

        # Create figure with subplots for each subcluster
        n_cols = min(4, n_subclusters)
        n_rows = (n_subclusters + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()

        fig.suptitle(f'Sub-clusters of Cluster {parent_cluster} ({method} clustering)',
                     fontsize=16)

        # Plot each subcluster
        for idx, subcluster_id in enumerate(unique_labels):
            if subcluster_id < 0:  # Skip noise points for DBSCAN
                continue

            if idx >= len(axes):
                break

            ax = axes[idx]

            # Get peaks in this subcluster
            subcluster_peaks = [peaks[i] for i, label in enumerate(labels)
                                if label == subcluster_id]

            # Plot individual time series
            for peak in subcluster_peaks:
                # Get proper timepoints
                timepoints = self.scan_data[peak.scan_type].timepoints

                # Normalize
                norm = self._normalize_intensities(peak)
                log_int = np.log1p(norm) / np.log(10)

                ax.plot(timepoints, log_int, alpha=0.3, linewidth=1)

            # Plot mean
            if subcluster_peaks:
                # Group by scan type for proper alignment
                by_scan = {}
                for peak in subcluster_peaks:
                    if peak.scan_type not in by_scan:
                        by_scan[peak.scan_type] = []
                    by_scan[peak.scan_type].append(peak)

                # Use first scan type for mean (simplified)
                if by_scan:
                    scan_type = list(by_scan.keys())[0]
                    scan_peaks = by_scan[scan_type]
                    timepoints = self.scan_data[scan_type].timepoints

                    mean_series = np.zeros(len(timepoints))
                    for peak in scan_peaks:
                        norm = self._normalize_intensities(peak)
                        mean_series += np.log1p(norm) / np.log(10)
                    mean_series /= len(scan_peaks)

                    ax.plot(timepoints, mean_series, 'k-', linewidth=3,
                            label=f'Mean (n={len(scan_peaks)})')

                    # Add trend line
                    z = np.polyfit(timepoints, mean_series, 1)
                    p = np.poly1d(z)
                    ax.plot(timepoints, p(timepoints), 'r--', linewidth=2,
                            label=f'Slope: {z[0]:.4f}')

            ax.set_xlabel('Time Point')
            ax.set_ylabel('Log(Normalized Intensity)')
            ax.set_title(f'Sub-cluster {subcluster_id} ({len(subcluster_peaks)} peaks)')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for j in range(idx + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"{self.config.output_dir}/subclusters_{parent_cluster}_{method}.png", dpi=300)
        plt.show()

    def analyze_peak_overlap(self) -> None:
        """Analyze potential peak overlaps using configuration parameters"""
        if not self.combined_analysis:
            raise ValueError("No analysis results available")

        logger.info("Analyzing potential peak overlaps")

        # Use configured overlap threshold
        overlap_threshold = self.config.overlap_threshold_mz

        overlapping_peaks = []
        peaks_to_remove = set()

        for scan_type, scan_data in self.scan_data.items():
            peaks_in_scan = [p for p in self.combined_analysis.all_peaks
                             if p.scan_type == scan_type]

            # Sort by m/z
            peaks_in_scan.sort(key=lambda p: p.mz_value)

            # Find overlaps
            for i in range(len(peaks_in_scan) - 1):
                if peaks_in_scan[i + 1].mz_value - peaks_in_scan[i].mz_value < overlap_threshold:
                    overlapping_peaks.append((peaks_in_scan[i], peaks_in_scan[i + 1]))

                    # Check if we should remove one based on correlation
                    if self.config.remove_overlapping_peaks:
                        corr = np.corrcoef(peaks_in_scan[i].intensities,
                                           peaks_in_scan[i + 1].intensities)[0, 1]

                        if abs(corr) > self.config.overlap_correlation_threshold:
                            # Remove the peak with lower mean intensity
                            if peaks_in_scan[i].mean_intensity < peaks_in_scan[i + 1].mean_intensity:
                                peaks_to_remove.add(peaks_in_scan[i].unique_id)
                            else:
                                peaks_to_remove.add(peaks_in_scan[i + 1].unique_id)

        logger.info(f"Found {len(overlapping_peaks)} potentially overlapping peak pairs")
        if self.config.remove_overlapping_peaks:
            logger.info(f"Marked {len(peaks_to_remove)} peaks for removal")

        # Visualize if diagnostic plots are enabled
        if self.config.create_diagnostic_plots and overlapping_peaks:
            self._visualize_overlapping_peaks(overlapping_peaks[:self.config.max_peaks_to_plot])

        return overlapping_peaks, peaks_to_remove

    def alternative_clustering_approaches(self) -> Dict[str, np.ndarray]:
        """Try alternative clustering approaches using configuration"""
        logger.info("Trying alternative clustering approaches")

        results = {}

        # 1. Clustering based on trend parameters
        if 'slope' in self.config.trend_features or 'r_squared' in self.config.trend_features:
            trend_features = []
            feature_names = []

            for peak in self.combined_analysis.all_peaks:
                timepoints = self.scan_data[peak.scan_type].timepoints
                norm = self._normalize_intensities(peak)
                log_int = np.log1p(norm) / np.log(10)

                features = []

                # Fit linear trend
                z = np.polyfit(timepoints, log_int, 1)
                slope, intercept = z

                if 'slope' in self.config.trend_features:
                    features.append(slope)
                    if len(trend_features) == 0:
                        feature_names.append('slope')

                if 'intercept' in self.config.trend_features:
                    features.append(intercept)
                    if len(trend_features) == 0:
                        feature_names.append('intercept')

                if 'r_squared' in self.config.trend_features:
                    # Calculate R²
                    p = np.poly1d(z)
                    y_pred = p(timepoints)
                    ss_res = np.sum((log_int - y_pred) ** 2)
                    ss_tot = np.sum((log_int - np.mean(log_int)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    features.append(r_squared)
                    if len(trend_features) == 0:
                        feature_names.append('r_squared')

                if 'cv' in self.config.trend_features:
                    features.append(peak.cv_intensity)
                    if len(trend_features) == 0:
                        feature_names.append('cv')

                trend_features.append(features)

            trend_features = np.array(trend_features)

            # Standardize features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            trend_features_scaled = scaler.fit_transform(trend_features)

            # Cluster based on trend features
            if self.config.clustering_method == 'kmeans':
                from sklearn.cluster import KMeans
                clusterer = KMeans(n_clusters=self.config.n_trend_clusters, random_state=42)
            elif self.config.clustering_method == 'hierarchical':
                from sklearn.cluster import AgglomerativeClustering
                clusterer = AgglomerativeClustering(n_clusters=self.config.n_trend_clusters)
            else:  # DBSCAN
                from sklearn.cluster import DBSCAN
                clusterer = DBSCAN(eps=self.config.dbscan_eps,
                                   min_samples=self.config.dbscan_min_samples)

            trend_clusters = clusterer.fit_predict(trend_features_scaled)
            results['trend_based'] = trend_clusters

            # Export features if enabled
            if self.config.export_peak_features:
                self._export_peak_features(trend_features, feature_names, trend_clusters)

            # Visualize if enabled
            if self.config.create_diagnostic_plots:
                self._visualize_alternative_clustering(results, trend_features)

        return results

    def _visualize_alternative_clustering(self, clustering_results: Dict[str, np.ndarray],
                                          trend_features: np.ndarray) -> None:
        """Visualize alternative clustering results"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Trend-based clustering in feature space
        ax1 = axes[0, 0]
        scatter = ax1.scatter(trend_features[:, 0], trend_features[:, 2],
                              c=clustering_results['trend_based'],
                              cmap='tab10', s=50, alpha=0.7)
        ax1.set_xlabel('Slope')
        ax1.set_ylabel('R²')
        ax1.set_title('Trend-based Clustering (Slope vs R²)')
        plt.colorbar(scatter, ax=ax1, label='Cluster')

        # 2. Compare with original clustering
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(trend_features[:, 0], trend_features[:, 2],
                               c=self.combined_analysis.cluster_labels,
                               cmap='tab10', s=50, alpha=0.7)
        ax2.set_xlabel('Slope')
        ax2.set_ylabel('R²')
        ax2.set_title('Original Clustering (Slope vs R²)')
        plt.colorbar(scatter2, ax=ax2, label='Cluster')

        # 3. Cluster size comparison
        ax3 = axes[1, 0]
        methods = ['Original', 'Trend-based']
        cluster_sizes = []

        for method, labels in [('Original', self.combined_analysis.cluster_labels),
                               ('Trend-based', clustering_results['trend_based'])]:
            unique, counts = np.unique(labels, return_counts=True)
            # Sort by size
            sorted_idx = np.argsort(counts)[::-1]
            cluster_sizes.append(counts[sorted_idx])

        x = np.arange(max(len(s) for s in cluster_sizes))
        width = 0.35

        for i, (method, sizes) in enumerate(zip(methods, cluster_sizes)):
            ax3.bar(x[:len(sizes)] + i * width, sizes, width, label=method)

        ax3.set_xlabel('Cluster Rank (by size)')
        ax3.set_ylabel('Number of Peaks')
        ax3.set_title('Cluster Size Distribution Comparison')
        ax3.legend()

        # 4. Slope distribution by cluster
        ax4 = axes[1, 1]
        for cluster_id in np.unique(clustering_results['trend_based']):
            mask = clustering_results['trend_based'] == cluster_id
            slopes = trend_features[mask, 0]
            ax4.hist(slopes, bins=20, alpha=0.5, label=f'Cluster {cluster_id}')

        ax4.set_xlabel('Slope')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Slope Distribution by Trend-based Clusters')
        ax4.legend()

        plt.tight_layout()
        plt.savefig(f"{self.config.output_dir}/alternative_clustering.png", dpi=300)
        plt.show()

    # Updated methods that use configuration parameters instead of hardcoded values

    def advanced_clustering_analysis(self):
        """Run advanced clustering analysis using configuration parameters"""

        # Check if advanced analysis is enabled
        if hasattr(self.config, 'enable_subclustering') and not self.config.enable_subclustering:
            logger.info("Advanced clustering analysis is disabled in configuration")
            return None, None

        # Create diagnostic plots if enabled
        if self.config.create_diagnostic_plots:
            self.diagnose_clustering()

        # Sub-cluster large clusters if enabled
        subcluster_results = None
        if self.config.subcluster_large_clusters:
            # Find clusters larger than threshold
            unique_clusters, counts = np.unique(self.combined_analysis.cluster_labels,
                                                return_counts=True)

            for cluster_id, count in zip(unique_clusters, counts):
                if count > self.config.subcluster_threshold:
                    logger.info(f"Cluster {cluster_id} has {count} peaks, performing sub-clustering")
                    subcluster_results = self.subcluster_large_cluster(
                        large_cluster_id=cluster_id,
                        n_subclusters=self.config.n_subclusters,
                        method=self.config.subclustering_method
                    )

                    # Export subclustering results to Excel if enabled
                    if hasattr(self.config, 'export_subclusters') and self.config.export_subclusters:
                        self.export_subclustering_results_to_excel(
                            peaks=subcluster_results['peaks'],
                            subcluster_labels=subcluster_results['subcluster_labels'],
                            parent_cluster=cluster_id
                        )

        # Analyze peak overlaps if enabled
        if self.config.analyze_overlaps:
            self.analyze_peak_overlap()

        # Try alternative clustering if enabled
        alt_clustering = None
        if self.config.use_trend_clustering:
            alt_clustering = self.alternative_clustering_approaches()

        return subcluster_results, alt_clustering

    def subcluster_large_cluster(self, large_cluster_id: int,
                                 n_subclusters: Optional[int] = None,
                                 method: Optional[str] = None) -> Dict[str, Any]:
        """Sub-cluster a large cluster using configuration parameters"""

        # Use config values if not specified
        if n_subclusters is None:
            n_subclusters = self.config.n_subclusters
        if method is None:
            method = self.config.subclustering_method

        logger.info(f"Sub-clustering cluster {large_cluster_id} into {n_subclusters} groups using {method}")

        # Get peaks in the large cluster
        large_cluster_peaks = [p for p in self.combined_analysis.all_peaks
                               if p.cluster_id == large_cluster_id]

        if len(large_cluster_peaks) < n_subclusters:
            logger.warning(f"Cluster {large_cluster_id} has fewer peaks than requested subclusters")
            n_subclusters = max(2, len(large_cluster_peaks) // 5)

        # Prepare data for clustering
        cluster_data = []
        for peak in large_cluster_peaks:
            # Apply preprocessing if configured
            intensities = peak.intensities.copy()

            # Detrend if enabled
            if self.config.detrend_before_clustering:
                timepoints = self.scan_data[peak.scan_type].timepoints
                z = np.polyfit(timepoints, intensities, 1)
                p = np.poly1d(z)
                intensities = intensities - p(timepoints) + np.mean(intensities)

            # Smooth if enabled
            if self.config.smooth_before_clustering:
                from scipy.signal import savgol_filter
                if len(intensities) > self.config.smoothing_window_size:
                    intensities = savgol_filter(intensities,
                                                window_length=self.config.smoothing_window_size,
                                                polyorder=2)

            # Normalize
            temp_peak = Peak(
                mz_index=peak.mz_index,
                mz_value=peak.mz_value,
                scan_type=peak.scan_type,
                position_in_scan=peak.position_in_scan,
                intensities=intensities
            )
            norm = self._normalize_intensities(temp_peak)
            log_int = np.log1p(norm) / np.log(10)
            cluster_data.append(log_int)

        cluster_data = np.array(cluster_data)

        # Perform sub-clustering
        if method == 'kmeans':
            from sklearn.cluster import KMeans
            clusterer = KMeans(n_clusters=n_subclusters, random_state=42, n_init=20)
            subcluster_labels = clusterer.fit_predict(cluster_data)

        elif method == 'hierarchical':
            # Calculate correlation distance matrix
            n_peaks = len(large_cluster_peaks)
            corr_matrix = np.zeros((n_peaks, n_peaks))

            for i in range(n_peaks):
                for j in range(i + 1):  # Only compute lower triangle
                    if i == j:
                        corr_matrix[i, j] = 1.0
                    else:
                        corr = np.corrcoef(cluster_data[i], cluster_data[j])[0, 1]
                        corr_matrix[i, j] = corr
                        corr_matrix[j, i] = corr  # Make matrix symmetric

            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform

            # Convert to distance matrix
            distance_matrix = 1 - np.abs(corr_matrix)

            # Verify symmetry
            if not np.allclose(distance_matrix, distance_matrix.T):
                logger.warning("Fixing asymmetric distance matrix")
                distance_matrix = (distance_matrix + distance_matrix.T) / 2

            condensed_dist = squareform(distance_matrix)
            linkage_matrix = linkage(condensed_dist, method='average')
            subcluster_labels = fcluster(linkage_matrix, n_subclusters, criterion='maxclust') - 1

        elif method == 'dbscan':
            from sklearn.cluster import DBSCAN
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)

            clusterer = DBSCAN(eps=self.config.dbscan_eps,
                               min_samples=self.config.dbscan_min_samples)
            subcluster_labels = clusterer.fit_predict(scaled_data)

        return {
            'peaks': large_cluster_peaks,
            'subcluster_labels': subcluster_labels,
            'n_subclusters': len(np.unique(subcluster_labels[subcluster_labels >= 0])),
            'method': method,
            'parent_cluster': large_cluster_id
        }

    def _reassign_uncorrelated_peaks(self, peaks: List[Peak],
                                     cluster_data: np.ndarray,
                                     labels: np.ndarray) -> np.ndarray:
        """Reassign peaks with low correlation to their assigned cluster"""
        new_labels = labels.copy()

        for i, (peak, label) in enumerate(zip(peaks, labels)):
            if label < 0:  # Skip noise points
                continue

            # Calculate mean correlation with cluster members
            cluster_members = [j for j, l in enumerate(labels) if l == label and j != i]

            if len(cluster_members) > 0:
                correlations = []
                for j in cluster_members:
                    corr = np.corrcoef(cluster_data[i], cluster_data[j])[0, 1]
                    correlations.append(abs(corr))

                mean_corr = np.mean(correlations)

                # Reassign if correlation is too low
                if mean_corr < self.config.min_correlation_with_cluster:
                    # Find best cluster
                    best_cluster = -1
                    best_corr = self.config.min_correlation_with_cluster

                    for new_label in np.unique(labels):
                        if new_label < 0 or new_label == label:
                            continue

                        cluster_members = [j for j, l in enumerate(labels) if l == new_label]
                        if cluster_members:
                            corrs = [abs(np.corrcoef(cluster_data[i], cluster_data[j])[0, 1])
                                     for j in cluster_members]
                            mean_corr = np.mean(corrs)

                            if mean_corr > best_corr:
                                best_corr = mean_corr
                                best_cluster = new_label

                    if best_cluster >= 0:
                        new_labels[i] = best_cluster
                        logger.debug(f"Reassigned peak {i} from cluster {label} to {best_cluster}")

        return new_labels

    def _export_subcluster_results(self, peaks: List[Peak],
                                   subcluster_labels: np.ndarray,
                                   parent_cluster: int) -> None:
        """Export subcluster assignments to file"""
        output_data = []

        for peak, sublabel in zip(peaks, subcluster_labels):
            output_data.append({
                'mz_value': peak.mz_value,
                'scan_type': peak.scan_type,
                'parent_cluster': parent_cluster,
                'subcluster': sublabel,
                'mean_intensity': peak.mean_intensity,
                'cv': peak.cv_intensity,
                'trend': peak.trend_type.value if peak.trend_type else 'unknown'
            })

        df = pd.DataFrame(output_data)

        if self.config.export_format == 'excel':
            output_path = f"{self.config.output_dir}/subclusters_cluster{parent_cluster}.xlsx"
            df.to_excel(output_path, index=False)
        else:  # CSV
            output_path = f"{self.config.output_dir}/subclusters_cluster{parent_cluster}.csv"
            df.to_csv(output_path, index=False)

        logger.info(f"Exported subcluster results to {output_path}")

    def _export_peak_features(self, features: np.ndarray,
                              feature_names: List[str],
                              cluster_labels: np.ndarray) -> None:
        """Export extracted peak features"""
        df = pd.DataFrame(features, columns=feature_names)
        df['cluster'] = cluster_labels

        # Add peak info
        peak_info = []
        for peak in self.combined_analysis.all_peaks:
            peak_info.append({
                'mz_value': peak.mz_value,
                'scan_type': peak.scan_type,
                'mean_intensity': peak.mean_intensity
            })

        peak_df = pd.DataFrame(peak_info)
        df = pd.concat([peak_df, df], axis=1)

        if self.config.export_format == 'excel':
            output_path = f"{self.config.output_dir}/peak_features.xlsx"
            df.to_excel(output_path, index=False)
        else:
            output_path = f"{self.config.output_dir}/peak_features.csv"
            df.to_csv(output_path, index=False)

        logger.info(f"Exported peak features to {output_path}")

    def export_clustering_results_to_excel(self) -> None:
        """Export clustering results to Excel with all required columns

        Creates an Excel file with the following columns:
        - peak m/z or mass value
        - scan group
        - reference spectrum intensity
        - max intensity
        - min intensity
        - clustering statistics (cluster group, confidence, etc.)
        - extracted intensities across cycles (one column per cycle)
        """
        if not self.combined_analysis:
            raise ValueError("No analysis results to save")

        logger.info("Exporting clustering results to Excel")

        output_dir = Path(self.config.output_dir)

        # Create data for Excel export
        data = []

        for peak in self.combined_analysis.all_peaks:
            # Get timepoints for this peak's scan type
            timepoints = self.scan_data[peak.scan_type].timepoints

            # Basic peak info             start = self.config.cycle_range_start
            #             end = self.config.cycle_range_end
            #
            #             ref_intensity =
            peak_info = {
                'mz_value': peak.mz_value,
                'scan_group': peak.scan_type,
                'reference_intensity': np.mean(peak.intensities[self.config.cycle_range_start:self.config.cycle_range_end]) if len(peak.intensities) > 0 else 0,
                'max_intensity': np.max(peak.intensities) if len(peak.intensities) > 0 else 0,
                'min_intensity': np.min(peak.intensities) if len(peak.intensities) > 0 else 0,
                'mean_intensity': peak.mean_intensity,
                'std_intensity': peak.std_intensity,
                'cv_intensity': peak.cv_intensity,
                'cluster_id': peak.cluster_id,
                'trend_type': peak.trend_type.value if peak.trend_type else 'unknown'
            }

            # Add intensities for each cycle
            for i, intensity in enumerate(peak.intensities):
                cycle_name = f"cycle_{i}"
                peak_info[cycle_name] = intensity

            data.append(peak_info)

        # Create DataFrame
        df = pd.DataFrame(data)

        # Save to Excel
        excel_path = output_dir / "clustering_results.xlsx"
        df.to_excel(excel_path, index=False)

        logger.info(f"Clustering results exported to {excel_path}")

    def export_subclustering_results_to_excel(self, peaks: List[Peak],
                                           subcluster_labels: np.ndarray,
                                           parent_cluster: int) -> None:
        """Export subclustering results to Excel with all required columns

        Creates an Excel file with the following columns:
        - peak m/z or mass value
        - scan group
        - reference spectrum intensity
        - max intensity
        - min intensity
        - clustering statistics (parent cluster, subcluster, etc.)
        - extracted intensities across cycles (one column per cycle)
        """
        if not peaks or len(peaks) == 0:
            logger.warning(f"No peaks to export for subcluster of parent cluster {parent_cluster}")
            return

        logger.info(f"Exporting subclustering results for parent cluster {parent_cluster} to Excel")

        output_dir = Path(self.config.output_dir)

        # Create data for Excel export
        data = []

        for peak, sublabel in zip(peaks, subcluster_labels):
            # Get timepoints for this peak's scan type
            timepoints = self.scan_data[peak.scan_type].timepoints

            # Basic peak info
            peak_info = {
                'mz_value': peak.mz_value,
                'scan_group': peak.scan_type,
                'reference_intensity': np.mean(peak.intensities[self.config.cycle_range_start:self.config.cycle_range_end]) if len(peak.intensities) > 0 else 0,
                'max_intensity': np.max(peak.intensities) if len(peak.intensities) > 0 else 0,
                'min_intensity': np.min(peak.intensities) if len(peak.intensities) > 0 else 0,
                'mean_intensity': peak.mean_intensity,
                'std_intensity': peak.std_intensity,
                'cv_intensity': peak.cv_intensity,
                'parent_cluster': parent_cluster,
                'subcluster': sublabel,
                'trend_type': peak.trend_type.value if peak.trend_type else 'unknown'
            }

            # Add intensities for each cycle
            for i, intensity in enumerate(peak.intensities):
                cycle_name = f"cycle_{i}"
                peak_info[cycle_name] = intensity

            data.append(peak_info)

        # Create DataFrame
        df = pd.DataFrame(data)

        # Save to Excel
        excel_path = output_dir / f"subclustering_results_cluster{parent_cluster}.xlsx"
        df.to_excel(excel_path, index=False)

        logger.info(f"Subclustering results for parent cluster {parent_cluster} exported to {excel_path}")

    def _classify_trends(self, peaks: List[Peak]) -> List[TrendType]:
        """Classify temporal trends for peaks"""
        trends = []

        for peak in peaks:
            # Get timepoints from the peak's scan type
            scan_data = self.scan_data[peak.scan_type]
            timepoints = scan_data.timepoints

            # Normalize intensities
            norm_intensities = self._normalize_intensities(peak)
            log_intensities = np.log1p(norm_intensities) / np.log(10)

            # Mann-Kendall test
            n = len(log_intensities)
            s = 0
            for i in range(n-1):
                for j in range(i+1, n):
                    if log_intensities[j] > log_intensities[i]:
                        s += 1
                    elif log_intensities[j] < log_intensities[i]:
                        s -= 1

            # Calculate variance
            var_s = n * (n - 1) * (2 * n + 5) / 18

            # Z-score
            if s > 0:
                z = (s - 1) / np.sqrt(var_s)
            elif s < 0:
                z = (s + 1) / np.sqrt(var_s)
            else:
                z = 0

            # P-value
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))

            # Check for oscillation
            fft = np.fft.fft(log_intensities - np.mean(log_intensities))
            power = np.abs(fft) ** 2
            max_power = np.max(power[1:len(power)//2])
            mean_power = np.mean(power[1:len(power)//2])

            is_oscillating = max_power > self.config.oscillation_threshold * mean_power

            # Classify
            if is_oscillating:
                trend = TrendType.OSCILLATING
            elif p_value < self.config.trend_significance:
                if z > 0:
                    trend = TrendType.INCREASING
                else:
                    trend = TrendType.DECREASING
            else:
                trend = TrendType.STABLE

            trends.append(trend)

        return trends

    def create_visualizations(self) -> None:
        """Create all visualizations"""
        if not self.combined_analysis:
            raise ValueError("No analysis results available")

        logger.info("Creating visualizations")

        # Generate PDF path
        if self._data_directory and self.config.pdf_in_data_directory:
            pdf_dir = Path(self._data_directory)
        else:
            pdf_dir = Path(self.config.output_dir)

        timestamp = datetime.now().strftime("%y%m%d_%H%M")
        pdf_filename = f"MS_analysis_{timestamp}.pdf"
        pdf_path = pdf_dir / pdf_filename

        with PdfPages(pdf_path) as pdf:
            # Summary page
            self._create_summary_page(pdf)

            # Correlation heatmap
            self._plot_correlations(pdf)

            # Clustering results
            self._plot_clustering(pdf)

            # Cluster timeseries with individual peaks and confidence intervals
            self._plot_cluster_timeseries(pdf)

            # Trend analysis
            self._plot_trends(pdf)

            # Peak zoom views
            self._plot_peak_zooms(pdf)

            # Stacked spectra by scan type
            self._plot_stacked_spectra(pdf)

        logger.info(f"PDF report saved: {pdf_path}")

        # Create a second PDF for subclustering results if enabled
        if hasattr(self.config, 'create_subcluster_pdf') and self.config.create_subcluster_pdf:
            self.create_subcluster_visualizations()

    def create_subcluster_visualizations(self) -> None:
        """Create visualizations for subclustering results"""
        if not self.combined_analysis:
            raise ValueError("No analysis results available")

        # Check if subclustering is enabled
        if not hasattr(self.config, 'enable_subclustering') or not self.config.enable_subclustering:
            logger.info("Subclustering is disabled, skipping subcluster PDF generation")
            return

        # Check if there are any large clusters to subcluster
        unique_clusters, counts = np.unique(self.combined_analysis.cluster_labels, return_counts=True)
        large_clusters = [cluster_id for cluster_id, count in zip(unique_clusters, counts) 
                         if count > self.config.subcluster_threshold]

        if not large_clusters:
            logger.info("No large clusters found for subclustering, skipping subcluster PDF generation")
            return

        logger.info("Creating subcluster visualizations")

        # Generate PDF path
        if self._data_directory and self.config.pdf_in_data_directory:
            pdf_dir = Path(self._data_directory)
        else:
            pdf_dir = Path(self.config.output_dir)

        timestamp = datetime.now().strftime("%y%m%d_%H%M")
        pdf_filename = f"MS_subcluster_analysis_{timestamp}.pdf"
        pdf_path = pdf_dir / pdf_filename

        with PdfPages(pdf_path) as pdf:
            # Summary page
            self._create_subcluster_summary_page(pdf, large_clusters)

            # Process each large cluster
            for cluster_id in large_clusters:
                # Perform subclustering
                subcluster_results = self.subcluster_large_cluster(
                    large_cluster_id=cluster_id,
                    n_subclusters=self.config.n_subclusters,
                    method=self.config.subclustering_method
                )

                # Get peaks and labels
                peaks = subcluster_results['peaks']
                subcluster_labels = subcluster_results['subcluster_labels']

                # Create visualizations for this subcluster
                self._plot_subcluster_results(pdf, peaks, subcluster_labels, cluster_id)

        logger.info(f"Subcluster PDF report saved: {pdf_path}")

    def _create_subcluster_summary_page(self, pdf: PdfPages, large_clusters: List[int]) -> None:
        """Create summary page for subcluster PDF"""
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("MS Subclustering Analysis Report", fontsize=16, fontweight='bold')

        # Create text summary
        summary_text = f"""
Subclustering Analysis Summary
=============================
Data Directory: {Path(self._data_directory).name if self._data_directory else 'Unknown'}
Analysis Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Large Clusters Analyzed: {len(large_clusters)}
Cluster IDs: {', '.join(map(str, large_clusters))}

Configuration:
  Subcluster threshold: {self.config.subcluster_threshold}
  Number of subclusters: {self.config.n_subclusters}
  Subclustering method: {self.config.subclustering_method}
"""

        # Add text to figure
        ax = fig.add_subplot(111)
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
                fontsize=10, fontfamily='monospace',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax.axis('off')

        pdf.savefig(fig)
        plt.close(fig)

    def _plot_subcluster_results(self, pdf: PdfPages, peaks: List, subcluster_labels: np.ndarray, parent_cluster: int) -> None:
        """Plot subclustering results for a specific parent cluster"""
        # Plot subcluster time series
        unique_labels = np.unique(subcluster_labels)
        n_subclusters = len(unique_labels[unique_labels >= 0])  # Exclude noise (-1) for DBSCAN

        # Create figure with subplots for each subcluster
        n_cols = min(2, n_subclusters)
        n_rows = (n_subclusters + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        elif n_rows == 1 or n_cols == 1:
            axes = np.array(axes).flatten()
        else:
            axes = axes.flatten()

        fig.suptitle(f'Sub-clusters of Cluster {parent_cluster} ({self.config.subclustering_method} clustering)',
                     fontsize=16)

        # Plot each subcluster
        for idx, subcluster_id in enumerate(unique_labels):
            if subcluster_id < 0:  # Skip noise points for DBSCAN
                continue

            if idx >= len(axes):
                break

            ax = axes[idx]

            # Get peaks in this subcluster
            subcluster_peaks = [peaks[i] for i, label in enumerate(subcluster_labels)
                              if label == subcluster_id]

            # Plot individual time series
            for peak in subcluster_peaks:
                # Get proper timepoints
                timepoints = self.scan_data[peak.scan_type].timepoints

                # Normalize
                norm = self._normalize_intensities(peak)
                log_int = np.log1p(norm) / np.log(10)

                ax.plot(timepoints, log_int, alpha=0.3, linewidth=1)

            # Plot mean
            if subcluster_peaks:
                # Group by scan type for proper alignment
                by_scan = {}
                for peak in subcluster_peaks:
                    if peak.scan_type not in by_scan:
                        by_scan[peak.scan_type] = []
                    by_scan[peak.scan_type].append(peak)

                # Use first scan type for mean (simplified)
                if by_scan:
                    scan_type = list(by_scan.keys())[0]
                    scan_peaks = by_scan[scan_type]
                    timepoints = self.scan_data[scan_type].timepoints

                    mean_series = np.zeros(len(timepoints))
                    for peak in scan_peaks:
                        norm = self._normalize_intensities(peak)
                        mean_series += np.log1p(norm) / np.log(10)
                    mean_series /= len(scan_peaks)

                    ax.plot(timepoints, mean_series, 'k-', linewidth=3,
                           label=f'Mean (n={len(scan_peaks)})')

                    # Add trend line
                    z = np.polyfit(timepoints, mean_series, 1)
                    p = np.poly1d(z)
                    ax.plot(timepoints, p(timepoints), 'r--', linewidth=2,
                           label=f'Slope: {z[0]:.4f}')

            ax.set_xlabel('Time Point')
            ax.set_ylabel('Log(Normalized Intensity)')
            ax.set_title(f'Sub-cluster {subcluster_id} ({len(subcluster_peaks)} peaks)')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for j in range(idx + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Add additional plots for each subcluster
        for subcluster_id in unique_labels:
            if subcluster_id < 0:  # Skip noise points for DBSCAN
                continue

            # Get peaks in this subcluster
            subcluster_peaks = [peaks[i] for i, label in enumerate(subcluster_labels)
                              if label == subcluster_id]

            if not subcluster_peaks:
                continue

            # Plot mass spectra for this subcluster
            self._plot_subcluster_mass_spectra(pdf, subcluster_peaks, parent_cluster, subcluster_id)

            # Plot peak zoom views for this subcluster
            self._plot_subcluster_peak_zooms(pdf, subcluster_peaks, parent_cluster, subcluster_id)

    def _plot_subcluster_mass_spectra(self, pdf: PdfPages, peaks: List, parent_cluster: int, subcluster_id: int) -> None:
        """Plot mass spectra for a specific subcluster"""
        # Group peaks by scan type
        by_scan = {}
        for peak in peaks:
            if peak.scan_type not in by_scan:
                by_scan[peak.scan_type] = []
            by_scan[peak.scan_type].append(peak)

        if not by_scan:
            return

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle(f'Mass Spectra for Cluster {parent_cluster}, Sub-cluster {subcluster_id}',
                    fontsize=16)

        # Plot mass spectra for each scan type
        colors = plt.cm.tab10.colors
        for i, (scan_type, scan_peaks) in enumerate(by_scan.items()):
            # Get mass data for this scan type
            scan_data = self.scan_data[scan_type]
            mass_data = scan_data.mass_data

            if mass_data is None or len(mass_data) == 0:
                continue

            # Plot mass spectrum
            ax.plot(mass_data[:, 0], mass_data[:, 1], alpha=0.5, color=colors[i % len(colors)],
                   label=f'{scan_type} (n={len(scan_peaks)})')

            # Mark peak positions
            for peak in scan_peaks:
                ax.axvline(peak.mz_value, color=colors[i % len(colors)], linestyle='--', alpha=0.5)

        ax.set_xlabel('m/z')
        ax.set_ylabel('Intensity')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def _plot_subcluster_peak_zooms(self, pdf: PdfPages, peaks: List, parent_cluster: int, subcluster_id: int) -> None:
        """Plot zoomed views of peaks in a specific subcluster"""
        if not peaks:
            return

        # Sort peaks by m/z
        sorted_peaks = sorted(peaks, key=lambda p: p.mz_value)

        # Create figure with subplots
        max_peaks = min(len(sorted_peaks), 16)  # Limit to 16 peaks per page
        n_cols = 4
        n_rows = (max_peaks + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        elif n_rows == 1 or n_cols == 1:
            axes = np.array(axes).flatten()
        else:
            axes = axes.flatten()

        fig.suptitle(f'Peak Zoom Views for Cluster {parent_cluster}, Sub-cluster {subcluster_id}',
                    fontsize=16)

        # Plot each peak
        for i, peak in enumerate(sorted_peaks[:max_peaks]):
            ax = axes[i]

            # Get timepoints
            timepoints = self.scan_data[peak.scan_type].timepoints

            # Plot raw intensities
            ax.plot(timepoints, peak.intensities, 'o-', markersize=4)

            # Add trend line
            z = np.polyfit(timepoints, peak.intensities, 1)
            p = np.poly1d(z)
            ax.plot(timepoints, p(timepoints), 'r--', linewidth=2)

            ax.set_title(f'm/z {peak.mz_value:.2f}')
            ax.set_xlabel('Time Point')
            ax.set_ylabel('Intensity')
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for j in range(max_peaks, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def _create_summary_page(self, pdf: PdfPages) -> None:
        """Create summary page"""
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("MS Time Series Analysis Report", fontsize=16, fontweight='bold')

        # Create text summary
        summary_text = f"""
Analysis Summary
================
Data Directory: {Path(self._data_directory).name if self._data_directory else 'Unknown'}
Analysis Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Scan Types Analyzed:
"""
        for scan_type, scan_data in self.scan_data.items():
            summary_text += f"\n  {scan_type}:"
            summary_text += f"\n    - Cycles: {scan_data.metadata['cycle_range'][0]} - {scan_data.metadata['cycle_range'][1]}"
            summary_text += f"\n    - Detected peaks: {len(scan_data.detected_peaks)}"
            summary_text += f"\n    - M/z range: {scan_data.metadata['mz_range'][0]:.1f} - {scan_data.metadata['mz_range'][1]:.1f}"

        summary_text += f"\n\nTotal Peaks Analyzed: {len(self.combined_analysis.all_peaks)}"
        summary_text += f"\nNumber of Clusters: {self.config.n_clusters}"

        # Count trends
        trend_counts = {}
        for trend in self.combined_analysis.trend_classifications:
            trend_counts[trend.value] = trend_counts.get(trend.value, 0) + 1

        summary_text += "\n\nTrend Distribution:"
        for trend, count in trend_counts.items():
            summary_text += f"\n  {trend}: {count} ({100*count/len(self.combined_analysis.all_peaks):.1f}%)"

        # Add configuration info
        summary_text += f"\n\nConfiguration:"
        summary_text += f"\n  Peak height threshold: {self.config.peak_height_threshold}"
        summary_text += f"\n  Peak prominence: {self.config.peak_prominence}"
        summary_text += f"\n  Min timepoints: {self.config.min_timepoints}"
        summary_text += f"\n  Correlation method: {self.config.correlation_method}"
        summary_text += f"\n  Clustering method: {self.config.clustering_method}"

        # Add text to figure
        ax = fig.add_subplot(111)
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
                fontsize=10, fontfamily='monospace',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax.axis('off')

        pdf.savefig(fig)
        plt.close(fig)

    def _plot_correlations(self, pdf: PdfPages) -> None:
        """Plot correlation heatmap"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Correlation matrix
        sns.heatmap(self.combined_analysis.correlation_matrix,
                    cmap='RdBu_r', center=0,
                    ax=ax1, cbar_kws={'label': 'Correlation'})
        ax1.set_title('Peak Correlation Matrix')
        ax1.set_xlabel('Peak Index')
        ax1.set_ylabel('Peak Index')

        # Distribution of correlations
        upper_tri = np.triu_indices_from(self.combined_analysis.correlation_matrix, k=1)
        correlations = self.combined_analysis.correlation_matrix[upper_tri]

        ax2.hist(correlations, bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(correlations), color='red', linestyle='--',
                    label=f'Mean: {np.mean(correlations):.3f}')
        ax2.set_xlabel('Correlation Coefficient')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Peak Correlations')
        ax2.legend()

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def _plot_clustering(self, pdf: PdfPages) -> None:
        """Plot clustering results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Prepare data for PCA
        all_intensities = []
        for peak in self.combined_analysis.all_peaks:
            # Normalize and log transform
            norm = self._normalize_intensities(peak)
            all_intensities.append(np.log1p(norm) / np.log(10))

        all_intensities = np.array(all_intensities)

        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(all_intensities)

        # Plot PCA with clusters
        scatter = ax1.scatter(pca_result[:, 0], pca_result[:, 1],
                            c=self.combined_analysis.cluster_labels,
                            cmap='tab10', s=50, alpha=0.7)
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax1.set_title('Peaks in PCA Space')
        plt.colorbar(scatter, ax=ax1, label='Cluster')

        # Cluster sizes
        unique_clusters = np.unique(self.combined_analysis.cluster_labels)
        cluster_sizes = [np.sum(self.combined_analysis.cluster_labels == c) 
                        for c in unique_clusters]

        ax2.bar(unique_clusters, cluster_sizes)
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Number of Peaks')
        ax2.set_title('Cluster Size Distribution')

        # Average time series per cluster
        for cluster_id in unique_clusters:
            cluster_peaks = [p for p in self.combined_analysis.all_peaks 
                           if p.cluster_id == cluster_id]

            # Group by scan type to get proper timepoints
            by_scan = {}
            for peak in cluster_peaks:
                if peak.scan_type not in by_scan:
                    by_scan[peak.scan_type] = []
                by_scan[peak.scan_type].append(peak)

            # Plot average for first scan type (for simplicity)
            if by_scan:
                scan_type = list(by_scan.keys())[0]
                scan_peaks = by_scan[scan_type]
                timepoints = self.scan_data[scan_type].timepoints

                # Calculate mean intensities
                mean_intensities = np.zeros(len(timepoints))
                for peak in scan_peaks:
                    norm = self._normalize_intensities(peak)
                    mean_intensities += np.log1p(norm) / np.log(10)
                mean_intensities /= len(scan_peaks)

                ax3.plot(timepoints, mean_intensities, 
                        label=f'Cluster {cluster_id}', linewidth=2)

        ax3.set_xlabel('Cycle Number')
        ax3.set_ylabel('Log Intensity')
        ax3.set_title('Average Time Series by Cluster')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Peak distribution by scan type and cluster
        scan_cluster_matrix = np.zeros((len(unique_clusters), len(self.scan_data)))
        scan_types = list(self.scan_data.keys())

        for i, cluster_id in enumerate(unique_clusters):
            for j, scan_type in enumerate(scan_types):
                count = sum(1 for p in self.combined_analysis.all_peaks
                           if p.cluster_id == cluster_id and p.scan_type == scan_type)
                scan_cluster_matrix[i, j] = count

        im = ax4.imshow(scan_cluster_matrix, aspect='auto', cmap='Blues')
        ax4.set_xticks(range(len(scan_types)))
        ax4.set_xticklabels(scan_types, rotation=45)
        ax4.set_yticks(range(len(unique_clusters)))
        ax4.set_yticklabels([f'Cluster {c}' for c in unique_clusters])
        ax4.set_xlabel('Scan Type')
        ax4.set_ylabel('Cluster')
        ax4.set_title('Peak Distribution: Clusters vs Scan Types')
        plt.colorbar(im, ax=ax4, label='Number of Peaks')

        # Add text annotations
        for i in range(len(unique_clusters)):
            for j in range(len(scan_types)):
                text = ax4.text(j, i, f'{int(scan_cluster_matrix[i, j])}',
                               ha='center', va='center', color='black')

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def _plot_cluster_timeseries(self, pdf: PdfPages) -> None:
        """Plot time series grouped by clusters with individual peaks and confidence intervals"""
        # Get unique clusters
        unique_clusters = np.unique(self.combined_analysis.cluster_labels)


        # Calculate grid size
        ncols = min(3, len(unique_clusters))
        nrows = (len(unique_clusters) + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6 * nrows))
        if nrows == 1:
            axes = [axes] if ncols == 1 else axes
        else:
            axes = axes.flatten()

        # Color palette for clusters
        cluster_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

        for i, cluster_id in enumerate(unique_clusters):
            if i >= len(axes):
                break

            ax = axes[i]

            # Get peaks in this cluster
            cluster_peaks = [p for p in self.combined_analysis.all_peaks 
                           if p.cluster_id == cluster_id]

            # Group by scan type to get proper timepoints
            by_scan = {}
            for peak in cluster_peaks:
                if peak.scan_type not in by_scan:
                    by_scan[peak.scan_type] = []
                by_scan[peak.scan_type].append(peak)

            # Collect all normalized peaks from all scan types
            all_normalized_peaks = []
            all_timepoints = []
            all_scan_types = []

            # Plot individual peaks by scan type (for proper timepoints)
            for scan_type, scan_peaks in by_scan.items():
                timepoints = self.scan_data[scan_type].timepoints
                cluster_color = cluster_colors[i]

                # Normalize and log transform each peak's intensities
                normalized_peaks = []
                for peak in scan_peaks:
                    norm = self._normalize_intensities(peak)
                    log_intensities = np.log1p(norm) / np.log(10)
                    normalized_peaks.append(log_intensities)
                    all_normalized_peaks.append(log_intensities)
                    all_timepoints.append(timepoints)
                    all_scan_types.append(scan_type)

                # Plot individual peaks with transparency
                for peak_series in normalized_peaks:
                    ax.plot(timepoints, peak_series,
                            color=cluster_color,
                            alpha=0.3,  # Transparency for individual peaks
                            linewidth=1)

            # Calculate and plot cluster mean for all peaks combined
            if all_normalized_peaks:
                # For combining data from different scan types, we'll use a common relative time scale
                # We'll normalize each scan type's timepoints to [0, 1] range

                # First, collect all normalized peaks by scan type
                normalized_by_scan = {}
                for scan_type in by_scan.keys():
                    normalized_by_scan[scan_type] = []

                for i, peak in enumerate(all_normalized_peaks):
                    scan_type = all_scan_types[i]
                    normalized_by_scan[scan_type].append(peak)

                # Calculate mean for each scan type
                mean_by_scan = {}
                for scan_type, peaks in normalized_by_scan.items():
                    if peaks:
                        mean_by_scan[scan_type] = np.mean(peaks, axis=0)

                # Choose a reference scan type for plotting
                reference_scan_type = list(by_scan.keys())[0]
                reference_timepoints = self.scan_data[reference_scan_type].timepoints

                # Combine all normalized peaks for statistics
                # We'll use all peaks from all scan types
                all_peaks_combined = []
                for scan_type, peaks in normalized_by_scan.items():
                    all_peaks_combined.extend(peaks)

                if all_peaks_combined:
                    # Calculate overall mean across all scan types
                    # This is the average of the means from each scan type
                    overall_mean = np.mean(list(mean_by_scan.values()), axis=0)

                    # Calculate overall standard deviation
                    # This is more complex - we need to account for different sample sizes
                    # We'll use a weighted average of variances
                    total_peaks = sum(len(peaks) for peaks in normalized_by_scan.values())
                    overall_var = np.zeros_like(overall_mean)

                    for scan_type, peaks in normalized_by_scan.items():
                        if peaks:
                            n_peaks = len(peaks)
                            weight = n_peaks / total_peaks
                            scan_var = np.var(peaks, axis=0)
                            overall_var += weight * scan_var

                    overall_std = np.sqrt(overall_var)

                    # Plot mean using reference timepoints
                    ax.plot(reference_timepoints, mean_by_scan[reference_scan_type],
                            color=cluster_color, linewidth=3, 
                            label=f'Cluster {cluster_id} Mean')

                    # Add 95% confidence interval
                    n_peaks = len(normalized_by_scan[reference_scan_type])
                    sem = overall_std / np.sqrt(total_peaks)  # Use total number of peaks for SEM
                    ci_multiplier = 1.96  # 95% CI

                    # Get bootstrap CIs for cluster mean
                    if 'bootstrap_ci' in self.combined_analysis.metadata:
                        bootstrap_ci = self.combined_analysis.metadata['bootstrap_ci']
                        # Use bootstrap_ci['lower'] and bootstrap_ci['upper'] for the fill_between

                    ax.fill_between(reference_timepoints,
                                    mean_by_scan[reference_scan_type] - (ci_multiplier * sem if self.config.bootstrap_samples < 5 else bootstrap_ci['lower'] ),
                                    mean_by_scan[reference_scan_type] + (ci_multiplier * sem if self.config.bootstrap_samples < 5 else bootstrap_ci['upper'] ),
                                    color=cluster_color, alpha=0.2)

                    # Add trend line for cluster mean
                    z = np.polyfit(reference_timepoints, mean_by_scan[reference_scan_type], 1)
                    p = np.poly1d(z)
                    ax.plot(reference_timepoints, p(reference_timepoints),
                            '--', color='black', linewidth=2, alpha=0.8)

                    # Calculate trend statistics
                    slope = z[0]
                    r_squared = 1 - (np.sum((mean_by_scan[reference_scan_type] - p(reference_timepoints)) ** 2) /
                                     np.sum((mean_by_scan[reference_scan_type] - np.mean(mean_by_scan[reference_scan_type])) ** 2))

                    ax.set_title(f'Cluster {cluster_id} ({len(cluster_peaks)} peaks)\n'
                                 f'Slope: {slope:.4f}, R²: {r_squared:.3f}')

            ax.set_xlabel('Time Point (Cycle)')
            ax.set_ylabel('Log(Intensity + 1)')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for j in range(len(unique_clusters), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def _plot_trends(self, pdf: PdfPages) -> None:
        """Plot trend analysis results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Trend distribution
        trend_counts = {}
        for trend in self.combined_analysis.trend_classifications:
            trend_counts[trend.value] = trend_counts.get(trend.value, 0) + 1

        # Pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(trend_counts)))
        wedges, texts, autotexts = ax1.pie(trend_counts.values(),
                                          labels=trend_counts.keys(),
                                          autopct='%1.1f%%',
                                          colors=colors,
                                          startangle=90)
        ax1.set_title('Distribution of Trend Types')

        # Example trends
        trend_examples = {trend: [] for trend in TrendType}
        for i, peak in enumerate(self.combined_analysis.all_peaks):
            if len(trend_examples[peak.trend_type]) < 3:
                trend_examples[peak.trend_type].append(peak)

        # Plot examples for each trend type
        axes = [ax2, ax3, ax4]
        for ax, (trend_type, examples) in zip(axes, list(trend_examples.items())[:3]):
            for peak in examples:
                timepoints = self.scan_data[peak.scan_type].timepoints
                norm = self._normalize_intensities(peak)
                log_int = np.log1p(norm) / np.log(10)

                ax.plot(timepoints, log_int,
                       label=f'm/z {peak.mz_value:.2f}',
                       marker='o', markersize=4)

            ax.set_xlabel('Cycle Number')
            ax.set_ylabel('Log Intensity')
            ax.set_title(f'{trend_type.value.title()} Trend Examples')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def _plot_peak_zooms(self, pdf: PdfPages) -> None:
        """Plot zoomed views of individual peaks"""
        # Sort peaks by cluster and m/z
        sorted_peaks = sorted(self.combined_analysis.all_peaks,
                            key=lambda p: (p.cluster_id, p.mz_value))

        peaks_per_page = 12
        n_pages = (len(sorted_peaks) + peaks_per_page - 1) // peaks_per_page

        for page in range(n_pages):
            start_idx = page * peaks_per_page
            end_idx = min(start_idx + peaks_per_page, len(sorted_peaks))
            page_peaks = sorted_peaks[start_idx:end_idx]

            n_peaks = len(page_peaks)
            ncols = 4
            nrows = (n_peaks + ncols - 1) // ncols

            fig, axes = plt.subplots(nrows, ncols, figsize=(16, 12))
            if nrows == 1:
                axes = [axes] if ncols == 1 else axes
            else:
                axes = axes.flatten()

            fig.suptitle(f'Peak Zoom Views (Page {page+1}/{n_pages})',
                        fontsize=14, fontweight='bold')

            for i, peak in enumerate(page_peaks):
                if i >= len(axes):
                    break

                ax = axes[i]

                # Get scan data
                scan_data = self.scan_data[peak.scan_type]

                # Define zoom range
                mz_spacing = np.median(np.diff(scan_data.mz_values))
                zoom_width = max(0.5, 50 * mz_spacing)  # At least 0.5 m/z
                mz_min = peak.mz_value - zoom_width
                mz_max = peak.mz_value + zoom_width

                # Find indices
                mz_mask = (scan_data.mz_values >= mz_min) & (scan_data.mz_values <= mz_max)

                if not np.any(mz_mask):
                    ax.text(0.5, 0.5, 'No data in range',
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'Peak at m/z {peak.mz_value:.2f}')
                    continue

                zoom_mz = scan_data.mz_values[mz_mask]
                zoom_intensities = scan_data.intensities[mz_mask, :]

                # Plot spectra
                jet_cmap = plt.cm.jet
                n_timepoints = len(scan_data.timepoints)

                for j in range(n_timepoints):
                    color = jet_cmap(j / (n_timepoints - 1))
                    ax.plot(zoom_mz, zoom_intensities[:, j],
                           color=color, alpha=0.6, linewidth=1)

                # Mark peak
                ax.axvline(peak.mz_value, color='red', linestyle='--', alpha=0.8)
                ax.scatter([peak.mz_value], [peak.mean_intensity],
                          color='red', s=100, marker='*', zorder=10,
                          edgecolors='black', linewidth=1)

                # Add info
                info_text = f"m/z: {peak.mz_value:.2f}\n"
                info_text += f"Scan: {peak.scan_type}\n"
                info_text += f"Cluster: {peak.cluster_id}\n"
                info_text += f"Trend: {peak.trend_type.value}\n"
                info_text += f"Mean Int: {peak.mean_intensity:.1e}\n"
                info_text += f"CV: {peak.cv_intensity:.2f}"

                ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                       verticalalignment='top', fontsize=8,
                       fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                ax.set_xlabel('m/z')
                ax.set_ylabel('Intensity')
                ax.set_title(f'Peak {start_idx + i + 1}')
                ax.grid(True, alpha=0.3)

            # Hide unused subplots
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    def _plot_stacked_spectra(self, pdf: PdfPages) -> None:
        """Plot stacked spectra for each scan type"""
        for scan_type, scan_data in self.scan_data.items():
            fig, ax = plt.subplots(figsize=(16, 10))

            jet_cmap = plt.cm.jet
            n_timepoints = len(scan_data.timepoints)
            colors = [jet_cmap(i / (n_timepoints - 1)) for i in range(n_timepoints)]

            # Normalize and stack
            offset = 0.1
            for i, cycle in enumerate(scan_data.timepoints):
                spectrum = scan_data.intensities[:, i]
                # Normalize
                max_int = np.max(spectrum)
                if max_int > 0:
                    norm_spectrum = spectrum #/ max_int
                else:
                    norm_spectrum = spectrum

                # Plot with offset (cycle 0 at top)
                y_offset = (n_timepoints - 1 - i) * offset
                ax.plot(scan_data.mz_values, norm_spectrum + y_offset,
                       color=colors[i], alpha=0.7, linewidth=0.8)

            # Mark detected peaks
            peak_mz_values = [p.mz_value for p in scan_data.detected_peaks]
            peak_positions = []
            for mz in peak_mz_values:
                idx = np.argmin(np.abs(scan_data.mz_values - mz))
                peak_positions.append(scan_data.mz_values[idx])

            # Get peak intensities from first spectrum
            first_spectrum = scan_data.intensities[:, 0]
            max_int = np.max(first_spectrum)
            if max_int > 0:
                norm_first = first_spectrum / max_int
            else:
                norm_first = first_spectrum

            peak_heights = []
            for mz in peak_mz_values:
                idx = np.argmin(np.abs(scan_data.mz_values - mz))
                peak_heights.append(norm_first[idx] + (n_timepoints - 1) * offset)

            ax.scatter(peak_positions, peak_heights,
                      color='red', s=50, marker='v',
                      label=f'{len(peak_mz_values)} Detected Peaks', zorder=10)

            ax.set_xlabel('m/z')
            ax.set_ylabel('Normalized Intensity + Offset')
            ax.set_title(f'Stacked Spectra - {scan_type}\n' + 
                        f'Cycles {scan_data.timepoints[0]} to {scan_data.timepoints[-1]}')

            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=jet_cmap,
                                      norm=plt.Normalize(vmin=scan_data.timepoints[0],
                                                        vmax=scan_data.timepoints[-1]))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
            cbar.set_label('Cycle Number')

            ax.legend()
            ax.grid(True, alpha=0.3)

            pdf.savefig(fig)
            plt.close(fig)

    def save_results(self) -> None:
        """Save analysis results to files"""
        if not self.combined_analysis:
            raise ValueError("No analysis results to save")

        logger.info("Saving results")

        output_dir = Path(self.config.output_dir)

        # Create peak statistics DataFrame
        peak_data = []
        for peak in self.combined_analysis.all_peaks:
            peak_data.append({
                'mz_value': peak.mz_value,
                'mz_index': peak.mz_index,
                'scan_type': peak.scan_type,
                'position_in_scan': peak.position_in_scan,
                'cluster_id': peak.cluster_id,
                'trend_type': peak.trend_type.value,
                'mean_intensity': peak.mean_intensity,
                'std_intensity': peak.std_intensity,
                'cv_intensity': peak.cv_intensity
            })

        peak_df = pd.DataFrame(peak_data)
        peak_df.to_csv(output_dir / "peak_statistics.csv", index=False)

        # Save correlation matrix
        corr_df = pd.DataFrame(self.combined_analysis.correlation_matrix)
        corr_df.to_csv(output_dir / "correlation_matrix.csv", index=False)

        # Save bootstrap confidence intervals
        if 'bootstrap_ci' in self.combined_analysis.metadata and self.config.bootstrap_samples >=5:
            bootstrap_ci = self.combined_analysis.metadata['bootstrap_ci']
            np.savez(output_dir / "bootstrap_confidence_intervals.npz",
                    lower=bootstrap_ci['lower'],
                    upper=bootstrap_ci['upper'])

        # Export clustering results to Excel
        self.export_clustering_results_to_excel()

        # Save configuration
        config_dict = {
            'peak_detection': {
                'peak_height_threshold': self.config.peak_height_threshold,
                'peak_prominence': self.config.peak_prominence,
                'min_peak_intensity': self.config.min_peak_intensity_across_timepoints,
                'min_timepoints': self.config.min_timepoints
            },
            'analysis': {
                'correlation_method': self.config.correlation_method,
                'clustering_method': self.config.clustering_method,
                'n_clusters': self.config.n_clusters
            }
        }

        with open(output_dir / "analysis_config.yaml", 'w') as f:
            yaml.dump(config_dict, f)

        logger.info(f"Results saved to {output_dir}")

    def run_complete_analysis(self, data_directory: str, 
                            file_pattern: str = "*scan*.txt") -> None:
        """Run complete analysis pipeline"""
        logger.info("Starting robust MS time series analysis")
        file_pattern = f"*{self.config.scan_string}*.txt"

        # Load data
        self.load_data(data_directory, file_pattern)

        # Detect peaks
        self.detect_peaks_all_scans()

        # Combine and analyze
        self.combine_and_analyze()

        # Create visualizations
        self.create_visualizations()

        # Save results
        self.save_results()

        logger.info("Analysis complete!")


# Command-line interface
@click.command()
@click.argument('data_directory', type=click.Path(exists=True))
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file')
@click.option('--output', '-o', default='results', help='Output directory')
@click.option('--file-pattern', default='*scan*.txt', help='File pattern')
@click.option('--create-config', is_flag=True, help='Create sample config and exit')
def main(data_directory, config, output, file_pattern, create_config):
    """Robust MS Time Series Analysis Tool"""

    if create_config:
        sample_config = AnalysisConfig()
        config_path = "sample_config.yaml"

        config_dict = {
            'peak_detection': {
                'peak_height_threshold': 1000.0,
                'peak_prominence': 500.0,
                'min_peak_intensity_across_timepoints': 100.0,
                'min_timepoints': 5,
                'use_mz_width_range': True,
                'mz_width_range': [0.1, 5.0],
                'snr_threshold': 3.0,
                'baseline_correction': True
            },
            'analysis': {
                'correlation_method': 'pearson',
                'clustering_method': 'hierarchical',
                'n_clusters': 5,
                'significance_threshold': 0.05,
                'bootstrap_samples': 1000,
                'confidence_level': 0.95
            },
            'output': {
                'output_dir': 'results',
                'create_single_pdf': True,
                'pdf_in_data_directory': True
            }
        }

        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        click.echo(f"Created sample config: {config_path}")
        return

    # Load configuration
    if config:
        analysis_config = AnalysisConfig.from_yaml(config)
    else:
        analysis_config = AnalysisConfig()

    # Override output directory if specified
    if output != 'results':
        analysis_config.output_dir = output

    if output == 'DATADIR':
        analysis_config.output_dir = data_directory
        print(f"Using data directory as output directory: {data_directory}")
        #output_dir.mkdir(exist_ok=True, parents=True)

    # Create analyzer and run
    analyzer = RobustMSAnalyzer(analysis_config)

    try:
        analyzer.run_complete_analysis(data_directory, file_pattern)
        click.echo("Analysis complete!")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise

    analyzer.advanced_clustering_analysis()
    '''
    # Add this after your main analysis
    subcluster_results = analyzer.subcluster_large_cluster(
        large_cluster_id=4,
        n_subclusters=10,  # Start with 10, adjust based on results
        method='hierarchical'  # Best for correlation-based patterns
    )
    '''
if __name__ == "__main__":
    main()
