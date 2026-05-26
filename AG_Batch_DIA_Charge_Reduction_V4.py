"""
SCRIPT FOR ANALYZING DIA-CHARGE REDUCTION DATA
-Made for single file AND batch processing (analyzes all files in specified directory)
-Single files (e.g. SLOMO time-courses): relevant output in directory with same name
-Batch processing (e.g. 1 file = 1 datapoint in titration, etc): all other output is relevant.

SETUP:
-Download "Unidec-master" unidec source code, and put this script in the Unidec-master directory
-Make sure Unidec is working (download packages, etc)

INSTRUCTIONS:
-Adjust params and input (directory with .raw's) in "main" section near end of script
    -Line ~1300
-Run script
    -errors not to worry about:
        "Unable to import SciexImporter"
        "pyimzML not found. Imaging features won't work."

FILE SPECIFICS:
-Made to analyze low-res DIA-charge reduction datasets in .raw format, with 1 or multiple duty cycles
    -params section defines:
        -length of duty cycle (in scans)
        -which scans to process
        -# of cycles to average identical scans
        -# cycles (e.g. time-points) to analyze
-Can be used on non-charge reduced data. Set unidec rounds to 1 or 2 in conf file to speed up script and ignore mass outputs
-Double check averaging results for new data (e.g. different resolutions)

PROCESSING & OUTPUT OVERVIEW:
-For each .raw file...
    -identical scans from consecutive cycles are averaged, deconvolved, and masses summed
    -output files for each step are written in the directory named <raw file name>
        -peaks picked at a scan-level and scored using unidec if specified, filtered by DSCORE, then written to output csv.
        -deconvolved masses from each cycle are summed and written as single files and aggregated dataset (rows=mass, columns=cycles)
        -makes unidec HDF5 files for each scan type and resulting deconvolved masses, which can be opened in unidec
-Batch processing is performed automatically on all .raw in directory
    -Aggregate information from each file is grouped with like information from the other files
        -e.g. aggregated masses file is (rows=mass, cols=file).
    -If multiple cycles are processed in batch mode, I think the first cycle is used to compare across files

"""
from os.path import exists

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from unidec.modules.hdf5_tools import replace_dataset
import os
import io
# import unidec.tools.linearize as ln
import unidec.UniDecImporter.Thermo.RawFileReader as dr
#fromtopdown
from pyopenms import *
import unidec.engine as unidec
import time
import numpy as np
import unidec.tools as ud
import unidec.modules.unidecstructure as ustructure
import time
import pandas as pd
import matplotlib.cm as cm
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.signal import convolve
from unidec.metaunidec.mudeng import MetaUniDec

start_time = time.time()


################################################################################
# DIA-CYCLE PLOTTING FUNCTIONS - Added for comprehensive cycle visualization
################################################################################

def plot_cycle_analysis_inline(
    saved_peaks, scan_av_dict, mass_av_dict, cycle_num, output_dir,
    max_scan_intensity_dict=None, mass_sum_data=None
):
    """Generate comprehensive plots for a single cycle."""
    print(f"\n{'=' * 70}")
    print(f"Generating Cycle {cycle_num} Analysis Plots")
    print(f"{'=' * 70}")

    if max_scan_intensity_dict is None:
        max_scan_intensity_dict = {}
        for scan_idx, scan_data in scan_av_dict.items():
            max_scan_intensity_dict[scan_idx] = np.max(scan_data[:, 1])
        print(f"Computed max scan intensities for {len(max_scan_intensity_dict)} scans")

    output_pdf = os.path.join(output_dir, f"cycle_{cycle_num}_analysis.pdf")

    with PdfPages(output_pdf) as pdf:
        # Plot A: Composite mass spectrum
        print("\nGenerating Plot A: Composite Mass Spectrum...")
        fig_composite = _plot_composite_mass_spectrum(
            saved_peaks, mass_sum_data, mass_av_dict, cycle_num
        )
        pdf.savefig(fig_composite, bbox_inches='tight', dpi=300)
        plt.close(fig_composite)
        print("  ✓ Composite mass spectrum complete")

        # Plot B: Individual scans (sequential, no threading)
        print("\nGenerating Plot B: Individual Raw Scans...")
        scan_indices = sorted(scan_av_dict.keys())

        for i, scan_idx in enumerate(scan_indices):
            fig = _plot_individual_scan(
                scan_idx, saved_peaks.get(scan_idx, []),
                scan_av_dict[scan_idx], cycle_num
            )
            pdf.savefig(fig, bbox_inches='tight', dpi=300)
            plt.close(fig)

            if (i + 1) % 5 == 0 or (i + 1) == len(scan_indices):
                print(f"  Progress: {i + 1}/{len(scan_indices)} scans")

        print(f"  ✓ Completed {len(scan_indices)} individual scan plots")

        d = pdf.infodict()
        d['Title'] = f'DIA-Charge Reduction Cycle {cycle_num} Analysis'
        d['Author'] = 'AG_Batch_DIA_Charge_Reduction'

    print(f"\n{'=' * 70}")
    print(f"✓ Analysis complete! Saved to:\n  {output_pdf}")
    print(f"{'=' * 70}\n")


def _plot_composite_mass_spectrumX(saved_peaks, mass_av_dict, max_scan_intensity_dict, cycle_num):
    """Plot composite sum mass spectrum with overlaid peak mdist colored by dscore."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Compute composite sum spectrum
    mass_grid = None
    composite_intensity = None

    for scan_idx, mass_data in mass_av_dict.items():
        if mass_grid is None:
            mass_grid = mass_data[:, 0]
            composite_intensity = np.zeros_like(mass_grid)
        composite_intensity += mass_data[:, 1]

    # Setup colormap for dscore (0 to 1)
    scan_indices = sorted(saved_peaks.keys())
    n_scans = len(scan_indices)

    # Collect all dscores and plot peaks FIRST (before composite)
    all_dscores = []
    n_peaks_plotted = 0

    if n_scans > 0:
        cmap = matplotlib.colormaps['jet']
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)  # Dscore range 0-1

        # Overlay each peak colored by its dscore
        for scan_idx in scan_indices:
            peaks = saved_peaks[scan_idx]
            max_scan_int = max_scan_intensity_dict.get(scan_idx, 1.0)

            for peak in peaks:
                try:
                    # Get dscore for this peak
                    dscore = getattr(peak, 'dscore', 0.5)  # Default to 0.5 if missing
                    all_dscores.append(dscore)
                    color = cmap(norm(dscore))

                    mass_values = peak.mdist[:, 0]
                    # Scale by peak height and max scan intensity
                    intensity_values = peak.mdist[:, 1] * peak.height * max_scan_int

                    # Plot the peak overlay colored by dscore
                    ax.plot(mass_values, intensity_values, '-',
                            color=color, linewidth=2.0, alpha=0.7, zorder=5)
                    n_peaks_plotted += 1
                except (AttributeError, IndexError) as e:
                    print(f"Error: {e} (peak: {peak}, dscore: {dscore}, max_scan_int: {max_scan_int})")

        print(f"  Plotted {n_peaks_plotted} peak overlays")

        # Colorbar showing dscore range
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='D-score')
        cbar.set_ticks(np.linspace(0, 1, 11))  # 0, 0.1, 0.2, ... 1.0

        if len(all_dscores) > 0:
            min_dscore = min(all_dscores)
            max_dscore = max(all_dscores)
            print(f"  D-score range: {min_dscore:.3f} - {max_dscore:.3f}")

    # Plot composite spectrum LAST (on top but transparent)
    ax.plot(mass_grid, composite_intensity, 'k-', linewidth=1.5,
            label='Composite Sum', zorder=10, alpha=0.5)

    ax.set_xlabel('Mass (Da)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Intensity (AU)', fontsize=14, fontweight='bold')
    ax.set_title(f'Cycle {cycle_num}: Composite Mass Spectrum with Peak Overlays\n'
                 f'Peak mdist colored by D-score ({n_peaks_plotted} peaks from {n_scans} scans)',
                 fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=11)

    if mass_grid is not None:
        ax.set_xlim(np.min(mass_grid), np.max(mass_grid))
        max_composite = np.max(composite_intensity)
        ax.set_ylim(0, max_composite * 1.1)

    plt.tight_layout()
    return fig


def _plot_composite_mass_spectrumY(saved_peaks, mass_av_dict, max_scan_intensity_dict, cycle_num):
    """Plot composite sum mass spectrum with overlaid peak mdist colored by dscore."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Compute composite sum spectrum
    mass_grid = None
    composite_intensity = None

    for scan_idx, mass_data in mass_av_dict.items():
        if mass_grid is None:
            mass_grid = mass_data[:, 0]
            composite_intensity = np.zeros_like(mass_grid)
        composite_intensity += mass_data[:, 1]

    max_composite = np.max(composite_intensity)

    # Setup colormap for dscore (0 to 1)
    scan_indices = sorted(saved_peaks.keys())
    n_scans = len(scan_indices)

    # Collect all dscores and plot peaks FIRST
    all_dscores = []
    n_peaks_plotted = 0

    if n_scans > 0:
        cmap = matplotlib.colormaps['jet']
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)  # Dscore range 0-1

        # Overlay each peak colored by its dscore
        for scan_idx in scan_indices:
            peaks = saved_peaks[scan_idx]

            for peak in peaks:
                try:
                    # Get dscore for this peak
                    dscore = getattr(peak, 'dscore', 0.5)
                    all_dscores.append(dscore)
                    color = cmap(norm(dscore))

                    mass_values = peak.mdist[:, 0]
                    # FIXED SCALING: Normalize mdist then scale to composite height
                    mdist_intensity = peak.mdist[:, 1]

                    # Normalize the mdist to 0-1
                    if np.max(mdist_intensity) > 0:
                        mdist_normalized = mdist_intensity / np.max(mdist_intensity)
                        # Scale to a visible fraction of composite max (using peak.height as multiplier)
                        intensity_values = mdist_normalized * peak.height * 0.8
                    else:
                        intensity_values = mdist_intensity

                    # Plot the peak overlay colored by dscore
                    ax.plot(mass_values, intensity_values, '-',
                            color=color, linewidth=2.0, alpha=0.7, zorder=5)
                    n_peaks_plotted += 1
                except (AttributeError, IndexError) as e:
                    continue

        print(f"  Plotted {n_peaks_plotted} peak overlays")

        # Colorbar showing dscore range
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='D-score')
        cbar.set_ticks(np.linspace(0, 1, 11))

        if len(all_dscores) > 0:
            min_dscore = min(all_dscores)
            max_dscore = max(all_dscores)
            print(f"  D-score range: {min_dscore:.3f} - {max_dscore:.3f}")

    # Plot composite spectrum LAST (on top but semi-transparent)
    ax.plot(mass_grid, composite_intensity, 'k-', linewidth=1.5,
            label='Composite Sum', zorder=10, alpha=0.4)

    ax.set_xlabel('Mass (Da)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Intensity (AU)', fontsize=14, fontweight='bold')
    ax.set_title(f'Cycle {cycle_num}: Composite Mass Spectrum with Peak Overlays\n'
                 f'Peak mdist colored by D-score ({n_peaks_plotted} peaks from {n_scans} scans)',
                 fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=11)

    if mass_grid is not None:
        ax.set_xlim(np.min(mass_grid), np.max(mass_grid))
        ax.set_ylim(0, max_composite * 1.1)

    plt.tight_layout()
    return fig


def _plot_composite_mass_spectrum(saved_peaks, mass_sum_data, max_scan_intensity_dict, cycle_num):
    """Plot composite sum mass spectrum with overlaid peak mdist colored by dscore."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Safety check
    if mass_sum_data is None:
        print("ERROR: mass_sum_data is None! Cannot create composite plot.")
        return plt.figure()

    # Use the pre-computed mass sum data
    mass_grid_orig = mass_sum_data[:, 0]
    composite_intensity_orig = mass_sum_data[:, 1]

    # FIX #2: Interpolate onto a finer grid for better resolution
    # Create high-resolution mass grid (10x more points)
    mass_grid_fine = np.linspace(np.min(mass_grid_orig), np.max(mass_grid_orig),
                                 len(mass_grid_orig) * 10)
    # Interpolate composite intensity onto fine grid
    from scipy.interpolate import interp1d
    interpolator = interp1d(mass_grid_orig, composite_intensity_orig,
                            kind='linear', bounds_error=False, fill_value=0)
    composite_intensity = interpolator(mass_grid_fine)
    mass_grid = mass_grid_fine

    # Setup colormap for dscore
    scan_indices = sorted(saved_peaks.keys())
    n_scans = len(scan_indices)

    # Collect all dscores and plot peaks FIRST
    all_dscores = []
    n_peaks_plotted = 0
    max_peak_intensity = 0  # Track max peak intensity for y-axis

    if n_scans > 0:
        cmap = matplotlib.colormaps['jet']
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

        # Overlay each peak colored by its dscore
        for scan_idx in scan_indices:
            peaks = saved_peaks[scan_idx]

            for peak in peaks:
                try:
                    # Get dscore for this peak
                    dscore = getattr(peak, 'dscore', 0.5)
                    all_dscores.append(dscore)
                    color = cmap(norm(dscore))

                    mass_values = peak.mdist[:, 0]
                    mdist_intensity = peak.mdist[:, 1]

                    if np.max(mdist_intensity) > 0:
                        mdist_normalized = mdist_intensity #/ np.max(mdist_intensity)
                        intensity_values = mdist_normalized * peak.height #* 0.8
                    else:
                        intensity_values = mdist_intensity

                    # Track max peak intensity
                    max_peak_intensity = max(max_peak_intensity, np.max(intensity_values))

                    # Plot the peak overlay colored by dscore
                    ax.plot(mass_values, intensity_values, '-',
                            color=color, linewidth=2.0, alpha=0.7, zorder=5)
                    n_peaks_plotted += 1
                except (AttributeError, IndexError):
                    continue

        print(f"  Plotted {n_peaks_plotted} peak overlays")

        # Colorbar showing dscore range
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='D-score')
        cbar.set_ticks(np.linspace(0, 1, 11))

        if len(all_dscores) > 0:
            min_dscore = min(all_dscores)
            max_dscore = max(all_dscores)
            print(f"  D-score range: {min_dscore:.3f} - {max_dscore:.3f}")

    # Plot composite spectrum LAST
    ax.plot(mass_grid_orig, composite_intensity_orig, 'k-', linewidth=1.5,
            label='Composite Sum', zorder=10, alpha=0.4)

    ax.set_xlabel('Mass (Da)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Intensity (AU)', fontsize=14, fontweight='bold')
    ax.set_title(f'Cycle {cycle_num}: Composite Mass Spectrum with Peak Overlays\n'
                 f'Peak mdist colored by D-score ({n_peaks_plotted} peaks from {n_scans} scans)',
                 fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=11)

    # FIX #1: Set y-axis limit to accommodate both composite AND peak overlays
    max_composite = np.max(composite_intensity)
    max_y = max(max_composite, max_peak_intensity)  # Use the larger of the two
    ax.set_xlim(np.min(mass_grid), np.max(mass_grid))
    ax.set_ylim(0, max_y * 1.15)  # 15% headroom above tallest feature

    plt.tight_layout()
    return fig

def _plot_individual_scan(scan_idx, peaks, scan_data, cycle_num):
    """Plot individual averaged raw scan with peak m/z extracts overlaid as profiles."""
    fig, ax = plt.subplots(figsize=(14, 6))

    mz_values = scan_data[:, 0]
    intensity_values = scan_data[:, 1]
    max_intensity = np.max(intensity_values)

    # Plot raw averaged scan
    ax.plot(mz_values, intensity_values, 'k-', linewidth=1.5,
            label='Averaged Raw Scan', zorder=10)

    if peaks:
        n_peaks = len(peaks)
        cmap = matplotlib.colormaps['rainbow']

        for peak_idx, peak in enumerate(peaks):
            color = cmap(peak_idx / max(n_peaks - 1, 1))
            peak_plotted = False

            # Try to plot from mzstack
            if hasattr(peak, 'mzstack') and len(peak.mzstack):
                for z_idx, mz_array in enumerate(peak.mzstack):
                    try:
                        if mz_array is not None and len(mz_array) > 0:
                            if mz_array.ndim == 2 and mz_array.shape[1] >= 2:
                                peak_mz = mz_array[:, 0]
                                # Scale peak intensity to match scan scale
                                peak_int_raw = mz_array[:, 1]
                                if np.max(peak_int_raw) > 0:
                                    peak_int = peak_int_raw  #* peak.height)#/ np.max(peak_int_raw)) * peak.height
                                else:
                                    peak_int = peak_int_raw #* peak.height
                            elif mz_array.ndim == 1:
                                peak_mz = mz_array
                                peak_int = np.interp(peak_mz, mz_values, intensity_values) * 0.8
                            else:
                                continue

                            # Only plot if within scan m/z range
                            mask = (peak_mz >= np.min(mz_values)) & (peak_mz <= np.max(mz_values))
                            if np.any(mask):
                                label = (f'Peak {peak_idx + 1} (z={z_idx + 1}, M={peak.mass:.1f} Da)'
                                         if z_idx == 0 and not peak_plotted else None)
                                ax.plot(peak_mz[mask], peak_int[mask], '-', color=color,
                                        linewidth=2.0, alpha=0.7, zorder=5, label=label)
                                peak_plotted = True
                    except (AttributeError, IndexError, TypeError):
                        continue

            # Fallback to mztab if mzstack didn't work
            if not peak_plotted and hasattr(peak, 'mztab') and peak.mztab is not None:
                try:
                    if peak.mztab.ndim == 2 and peak.mztab.shape[1] >= 2:
                        peak_mz = peak.mztab[:, 0]
                        peak_int_raw = peak.mztab[:, 1]
                        if np.max(peak_int_raw) > 0:
                            peak_int = (peak_int_raw / np.max(peak_int_raw)) * peak.height
                        else:
                            peak_int = peak_int_raw * peak.height

                        mask = (peak_mz >= np.min(mz_values)) & (peak_mz <= np.max(mz_values))
                        if np.any(mask):
                            label = f'Peak {peak_idx + 1} (M={peak.mass:.1f} Da)'
                            ax.plot(peak_mz[mask], peak_int[mask], '-', color=color,
                                    linewidth=2.0, alpha=0.7, zorder=5, label=label)
                except (AttributeError, IndexError, TypeError):
                    continue

    ax.set_xlabel('m/z', fontsize=14, fontweight='bold')
    ax.set_ylabel('Intensity (AU)', fontsize=14, fontweight='bold')
    ax.set_title(f'Cycle {cycle_num}, Scan {scan_idx}: Raw Spectrum with Peak Overlays\n'
                 f'Peaks extracted from mzstack (n={len(peaks)} peaks)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')

    handles, labels = ax.get_legend_handles_labels()
    if len(labels) > 10:
        ax.legend(handles[:10], labels[:10], loc='upper right',
                  fontsize=9, title=f'First 10/{len(labels)} peaks')
    elif len(labels) > 0:
        ax.legend(loc='upper right', fontsize=9)

    ax.set_xlim(np.min(mz_values), np.max(mz_values))
    ax.set_ylim(0, max_intensity * 1.15)

    plt.tight_layout()
    return fig


################################################################################
# END OF PLOTTING FUNCTIONS
################################################################################

def modified_sinc_kernel(m, degree=4):
	"""
	Create a modified sinc kernel for smoothing, based on the algorithm described in
	Schmid and Diebold, "Why and how Savitzky-Golay filters should be replaced"
	(ACS Measurement Science Au, 2022).

	Args:
		m: Half-width of the kernel (kernel size will be 2*m+1)
		degree: Degree of the filter (2, 4, 6, 8, or 10)

	Returns:
		Kernel array of size 2*m+1
	"""
	kernel = np.zeros(2 * m + 1)
	kernel[m] = 1.0
	for i in range(1, m + 1):
		x = i / (m + 1)
		sinc_arg = np.pi * 0.5 * (degree + 4) * x
		k = np.sin(sinc_arg) / sinc_arg if sinc_arg != 0 else 1.0
		k *= np.exp(-x * x * 4.0)
		kernel[m + i] = kernel[m - i] = k
	return kernel / np.sum(kernel)

def apply_modified_sinc_smooth(data, kernel_halfwidth=10, degree=4):
	"""
	Apply modified sinc kernel smoothing to data.

	Args:
		data: 1D array of intensity values
		kernel_halfwidth: Half-width of the kernel
		degree: Degree of the filter (2, 4, 6, 8, or 10)

	Returns:
		Smoothed data array
	"""
	kernel = modified_sinc_kernel(kernel_halfwidth, degree)
	smoothed = convolve(data, kernel, mode='same')
	edge = kernel_halfwidth
	if len(data) > 2 * edge:
		smoothed[:edge] = data[:edge]
		smoothed[-edge:] = data[-edge:]
	return np.maximum(smoothed, 0)

def find_peak_near(mz, intensity, target_mz, window=0.1):
	mask = (mz > target_mz - window) & (mz < target_mz + window)
	if not np.any(mask):
		return target_mz  # fallback
	local_mz = mz[mask]
	local_intensity = intensity[mask]
	return local_mz[np.argmax(local_intensity)]

def align_spectrum_by_peak(spec, observed_peak, target_peak):
	shift = target_peak - observed_peak
	return np.column_stack((spec[:, 0] + shift, spec[:, 1]))

def estimate_adaptive_bin_gridOLD(
	spectra_list,
	region_width: float = 100.0,
	mz_min_cutoff: float = 350.0,
	points_per_peak: float = 3.0,
	min_spacing: float = 0.005,
	required_regions: int = 3
) -> np.ndarray:
	"""
	Estimate adaptive nonlinear m/z grid from native MS spectra based on local point spacing in real peaks.

	Args:
		spectra_list: list of (mz, intensity) arrays
		region_width: m/z window to define local peak regions (default: 100 Th)
		mz_min_cutoff: discard data below this m/z value (default: 300 Th)
		points_per_peak: average number of points per peak
		min_spacing: minimum allowed bin width
		required_regions: minimum number of regions required to fit (default: 3)

	Returns:
		Nonlinear adaptive m/z binning grid
	"""
	spacing_data = []

	for spec in spectra_list:
		mz, intensity = spec[:, 0], spec[:, 1]

		# Filter to region above 300 Th
		valid = mz > mz_min_cutoff
		mz, intensity = mz[valid], intensity[valid]

		if len(mz) < 10:
			continue  # skip empty spectrum

		mz_range = (np.min(mz), np.max(mz))
		num_regions = int(np.ceil((mz_range[1] - mz_range[0]) / region_width))

		for i in range(num_regions):
			r_start = mz_range[0] + i * region_width
			r_end = r_start + region_width
			region_mask = (mz >= r_start) & (mz < r_end)

			if not np.any(region_mask):
				continue

			mz_r, int_r = mz[region_mask], intensity[region_mask]
			if len(mz_r) < 3:
				continue

			# Find local max
			peak_idx = np.argmax(int_r)
			# Take ~5 points left and right (as available)
			window = slice(max(peak_idx - 5, 0), min(peak_idx + 6, len(mz_r)))
			mz_peak_region = mz_r[window]

			# Compute spacing and midpoints
			delta = np.diff(mz_peak_region)
			mid_mz = (mz_peak_region[1:] + mz_peak_region[:-1]) / 2
			if len(delta) >= 1:
				spacing_data.append((mid_mz, delta))

	if len(spacing_data) < required_regions:
		raise ValueError(f"Not enough valid regions for spacing fit. Found {len(spacing_data)}, need {required_regions}.")

	# Concatenate all mid_mz and spacing pairs
	all_mid_mz = np.concatenate([d[0] for d in spacing_data])
	all_delta = np.concatenate([d[1] for d in spacing_data])
	root_mz = np.sqrt(all_mid_mz)

	# Try both models: delta = k * sqrt(m/z) and delta = k * m/z
	# Fit 1: delta = k * sqrt(m/z)
	slope_root, residuals_root, _, _ = np.linalg.lstsq(root_mz[:, None], all_delta, rcond=None)
	k_root = slope_root[0]
	predicted_delta_root = k_root * root_mz
	rsse_root = np.sum((all_delta - predicted_delta_root) ** 2)

	# Fit 2: delta = k * m/z
	slope_linear, residuals_linear, _, _ = np.linalg.lstsq(all_mid_mz[:, None], all_delta, rcond=None)
	k_linear = slope_linear[0]
	predicted_delta_linear = k_linear * all_mid_mz
	rsse_linear = np.sum((all_delta - predicted_delta_linear) ** 2)

	# Select the better fit based on RSSE
	if rsse_root <= rsse_linear:
		k = k_root
		use_sqrt = True
		print(f"Using sqrt(m/z) model: RSSE = {rsse_root:.3e} (linear RSSE = {rsse_linear:.3e})")
		print(f"Estimated resolution @400 m/z: {400/(k*np.sqrt(400)):.0f}, @200 m/z: {200/(k*np.sqrt(200)):.0f}")
	else:
		k = k_linear
		use_sqrt = False
		print(f"Using linear m/z model: RSSE = {rsse_linear:.3e} (sqrt model RSSE = {rsse_root:.3e})")
		print(f"Estimated resolution: {200/(200*k):.0f}")

	# Generate nonlinear m/z grid
	# Generate nonlinear m/z grid
	#print(f"Estimated resolution @400 m/z: {rsse_linear:.6e} (sqrt RSSE = {rsse_root:.6e})")

	mz_start = np.min([np.min(spec[:, 0]) for spec in spectra_list])
	mz_end = np.max([np.max(spec[:, 0]) for spec in spectra_list])

	grid = [mz_start]
	current_mz = mz_start
	while current_mz < mz_end:
		if use_sqrt:
			spacing = max(k * np.sqrt(current_mz) / points_per_peak, min_spacing)
		else:
			spacing = max(k * current_mz / points_per_peak, min_spacing)
		current_mz += spacing
		grid.append(current_mz)

	return np.array(grid)

# --- 3-equation instrument-aware adaptive grid (inlined from adaptive_binning.py) ---
def estimate_adaptive_bin_grid(
    spectra_list,
    region_width: float = 100.0,
    mz_min_cutoff: float = 350.0,
    points_per_peak: float = 1.0,
    min_spacing: float = 0.0001,
    required_regions: int = 3,
    verbose: bool = True,
    snr_threshold: float = 5.0,
):
    """Estimate adaptive nonlinear m/z grid from native MS spectra
    based on local point spacing around real peaks.

    Args:
        spectra_list: list of (mz, intensity) arrays — each shape (N, 2)
                      OR list of (mz_array, int_array) tuples
        region_width: m/z window to define local peak regions
        mz_min_cutoff: discard data below this m/z value
        points_per_peak: average number of points per peak
        min_spacing: minimum allowed bin width
        required_regions: minimum regions required to fit

    Returns:
        (grid, detection_info) where detection_info is a dict with
        model parameters, instrument type, etc.
    """
    spacing_data = []
    n_pts_per_scan = []

    for spec in spectra_list:
        # Handle both (N,2) arrays and (mz, int) tuples
        if isinstance(spec, tuple) and len(spec) == 2:
            mz, intensity = spec[0], spec[1]
        elif hasattr(spec, 'shape') and spec.ndim == 2:
            mz, intensity = spec[:, 0], spec[:, 1]
        else:
            continue

        mz = np.asarray(mz, dtype=np.float64)
        intensity = np.asarray(intensity, dtype=np.float64)

        # Sort by m/z within this single scan
        sort_idx = np.argsort(mz)
        mz = mz[sort_idx]
        intensity = intensity[sort_idx]

        # Filter to region above cutoff
        valid = mz > mz_min_cutoff
        mz, intensity = mz[valid], intensity[valid]

        if len(mz) < 10:
            continue

        n_pts_per_scan.append(len(mz))

        # Compute noise floor for THIS scan only
        nonzero_int = intensity[intensity > 0]
        noise = float(np.median(nonzero_int)) if len(nonzero_int) > 0 else 0

        mz_range = (float(np.min(mz)), float(np.max(mz)))
        num_regions = int(np.ceil((mz_range[1] - mz_range[0]) / region_width))

        for i in range(num_regions):
            r_start = mz_range[0] + i * region_width
            r_end = r_start + region_width
            region_mask = (mz >= r_start) & (mz < r_end)

            if not np.any(region_mask):
                continue

            mz_r, int_r = mz[region_mask], intensity[region_mask]
            if len(mz_r) < 3:
                continue

            # Skip regions where the tallest peak is below noise floor
            max_int = float(np.max(int_r))
            if noise > 0 and max_int < snr_threshold * noise:
                continue

            # Find local max
            peak_idx = np.argmax(int_r)
            # Take ~5 points left and right (as available)
            window = slice(max(peak_idx - 4, 0), min(peak_idx + 5, len(mz_r)))
            mz_peak_region = mz_r[window]

            # Compute spacing and midpoints WITHIN this single scan
            delta = np.diff(mz_peak_region)
            #print(delta)
            mid_mz = (mz_peak_region[1:] + mz_peak_region[:-1]) / 2
            #print(mid_mz)
            #mid_mz = np.array(np.median(mid_mz), dtype=np.float64)
            #print(mid_mz)
            if len(delta) >= 1:
                spacing_data.append((mid_mz, delta))

    if len(spacing_data) < required_regions:
        if verbose:
            print(f"  Not enough valid regions for spacing fit. "
                  f"Found {len(spacing_data)}, need {required_regions}.")
            print(f"  Falling back to uniform grid.")
        # Fallback: uniform grid
        mz_start = min(float(np.min(s[0] if isinstance(s, tuple) else s[:, 0]))
                       for s in spectra_list)
        mz_end = max(float(np.max(s[0] if isinstance(s, tuple) else s[:, 0]))
                     for s in spectra_list)
        grid = np.arange(mz_start, mz_end, 0.01)
        return grid, {"instrument": "unknown", "use_sqrt": True,
                       "k": 0.01, "rsse_root": 0, "rsse_linear": 0}

    # Concatenate all mid_mz and spacing pairs
    all_mid_mz = np.concatenate([d[0] for d in spacing_data])
    all_delta = np.concatenate([d[1] for d in spacing_data])

    # Three models based on instrument physics:
    #   TOF:      delta = k * (m/z)^0.5    (uniform time bins, m/z ~ t^2)
    #   Orbitrap: delta = k * (m/z)^1.5    (uniform freq FFT, m/z ~ 1/w^2)
    #   Linear:   delta = k * (m/z)^1.0    (fallback / vendor resampled)
    models = {
        "tof": {"exp": 0.5, "basis": all_mid_mz ** 0.5},
        "orbitrap": {"exp": 1.5, "basis": all_mid_mz ** 1.5},
        "linear": {"exp": 1.0, "basis": all_mid_mz ** 1.0},
    }

    best_name = None
    best_rsse = np.inf
    fit_results = {}

    for name, m in models.items():
        # Fit delta = k * (m/z)^exp + b  (with intercept)
        A = np.column_stack([m["basis"], np.ones(len(m["basis"]))])
        coeffs, _, _, _ = np.linalg.lstsq(A, all_delta, rcond=None)
        k_fit = float(coeffs[0])
        b_fit = float(coeffs[1])
        predicted = k_fit * m["basis"] + b_fit
        rsse = float(np.sum((all_delta - predicted) ** 2))
        fit_results[name] = {"k": k_fit, "b": b_fit, "rsse": rsse, "exp": m["exp"]}
        if rsse < best_rsse:
            best_rsse = rsse
            best_name = name

    k = fit_results[best_name]["k"]
    b = fit_results[best_name]["b"]
    best_exp = fit_results[best_name]["exp"]
    instrument = best_name

    # Use the intercept as the natural min_spacing floor
    resolution_enhancement = 3.0
    #fitted_min_spacing = max(b, min_spacing)
    fitted_min_spacing = max(k * (350 ** best_exp) + b / resolution_enhancement, min_spacing) #resolution enhanced

    if verbose:
        print(f"\n  ── Adaptive Grid Estimation ──")
        print(f"  Spectra analyzed: {len(spectra_list)}")
        if n_pts_per_scan:
            print(f"  Median pts/scan: {int(np.median(n_pts_per_scan))}")
        print(f"  Valid spacing regions: {len(spacing_data)}")
        print(f"  Model comparison (RSSE):")
        for name in ("tof", "orbitrap", "linear"):
            fr = fit_results[name]
            winner = " <-- best" if name == best_name else ""
            print(f"    {name:10s}: delta = {fr['k']:.3e} * (m/z)^{fr['exp']}, "
                  f"RSSE={fr['rsse']:.3e}{winner}")
        print(f"  Selected: {instrument.upper()} "
              f"(delta = {k:.3e} * (m/z)^{best_exp} + {b:.4f})")
        print(f"  Fitted intercept (min floor): {b:.6f} Th")
        # Resolution at key m/z values
        print(f"  Resolution estimates:")
        for mz_check in [200, 400, 1000, 4000]:
            sp = k * (mz_check ** best_exp)
            res = mz_check / sp if sp > 0 else 0
            print(f"    R @ {mz_check:5d} m/z = {res:>10.0f}  "
                  f"(spacing = {sp:.6f} Th)")
        for name in ("tof", "orbitrap", "linear"):
            fr = fit_results[name]
            winner = " <-- best" if name == best_name else ""
            print(f"    {name:10s}: delta = {fr['k']:.3e} * (m/z)^{fr['exp']}, "
                  f"RSSE={fr['rsse']:.3e}{winner}")
        print(f"  Selected: {instrument.upper()}")
        for mz_check in [400, 1000, 2000, 4000]:
            sp = k * (mz_check ** best_exp) + b
            res = mz_check / sp if sp > 0 else 0
            print(f"    @ {mz_check} m/z: spacing={sp:.5f}, R={res:.0f}")

    # Generate nonlinear m/z grid using best model
    mz_start = min(float(np.min(s[0] if isinstance(s, tuple) else s[:, 0]))
                   for s in spectra_list)
    mz_end = max(float(np.max(s[0] if isinstance(s, tuple) else s[:, 0]))
                 for s in spectra_list)

    grid = [mz_start]
    current_mz = mz_start

    while current_mz < mz_end:
        #spacing = max(k * (current_mz ** best_exp) / 1, min_spacing)#max(k * (current_mz ** best_exp) / points_per_peak, min_spacing)
        spacing = max((k * (current_mz ** best_exp) + b ) / resolution_enhancement, min_spacing) #resolution enhanced
        current_mz += spacing
        grid.append(current_mz)
        #if spacing < min_spacing:
        #    spacing = min_spacing
        #    current_mz += spacing
        #    grid.append(current_mz)

    grid = np.array(grid)

    detection_info = {
        "instrument": instrument,
        "exponent": best_exp,
        "k": k,
        "b": b,
        "fitted_min_spacing": fitted_min_spacing,
        "fit_results": fit_results,
        "points_per_peak": points_per_peak,
        "min_spacing": min_spacing,
        "n_regions": len(spacing_data),
        "n_spectra": len(spectra_list),
        "median_pts_per_scan": int(np.median(n_pts_per_scan)) if n_pts_per_scan else 0,
        "all_mid_mz": all_mid_mz,
        "all_delta": all_delta,
    }

    if verbose:
        print(f"  Grid: {len(grid)} points over {mz_start:.0f}-{mz_end:.0f} m/z")

    return grid, detection_info

# --- QC plotting for adaptive-grid fitting (validation of the 3-model fit) ---
def plot_grid_fit_qc(grid, info, save_path, label=""):
    """Save a QC figure for one adaptive-grid fit. Never raises.

    Panels: (A) measured point spacing vs m/z with the 3 model fits,
    (B) residuals of the chosen model, (C) resulting grid bin widths.
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        fit_results = info.get("fit_results", {})
        all_mid_mz = np.asarray(info.get("all_mid_mz", []), dtype=float)
        all_delta = np.asarray(info.get("all_delta", []), dtype=float)
        instrument = info.get("instrument", "unknown")
        best_exp = info.get("exponent", None)
        k = info.get("k", 0.0)
        b = info.get("b", 0.0)
        model_colors = {"tof": "green", "orbitrap": "purple", "linear": "orange"}

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Panel A: spacing vs m/z + 3 model fits
        ax = axes[0]
        if all_mid_mz.size > 0:
            ax.scatter(all_mid_mz, all_delta, s=6, alpha=0.35, c="steelblue",
                       label="Measured spacing")
            mz_fit = np.linspace(max(float(all_mid_mz.min()), 1.0),
                                 float(all_mid_mz.max()), 200)
            for name in ("tof", "orbitrap", "linear"):
                if name not in fit_results:
                    continue
                fr = fit_results[name]
                pred = fr["k"] * (mz_fit ** fr["exp"]) + fr.get("b", 0.0)
                is_best = (name == instrument)
                ax.plot(mz_fit, pred, color=model_colors.get(name, "gray"),
                        linewidth=2 if is_best else 1.2,
                        alpha=0.4 if is_best else 0.4,
                        label="%s^%s: RSSE=%.2e%s" % (
                            name, fr["exp"], fr["rsse"],
                            "  <-- best" if is_best else ""))
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "No spacing data\n(uniform-grid fallback)",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("m/z")
        ax.set_ylabel("Point spacing (Th)")
        ax.set_title("Spacing fit - 3 models")

        # Panel B: residuals of the chosen model
        ax = axes[1]
        if all_mid_mz.size > 0 and best_exp is not None:
            resid = all_delta - (k * (all_mid_mz ** best_exp) + b)
            ax.scatter(all_mid_mz, resid, s=6, alpha=0.35,
                       c=model_colors.get(instrument, "gray"))
            ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        else:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_xlabel("m/z")
        ax.set_ylabel("Residual (Th)")
        ax.set_title("Residuals - %s" % instrument.upper())

        # Panel C: resulting grid bin widths
        ax = axes[2]
        if grid is not None and len(grid) > 1:
            g = np.asarray(grid, dtype=float)
            bw = np.diff(g)
            bc = (g[:-1] + g[1:]) / 2.0
            ax.plot(bc, bw, linewidth=0.8, color="steelblue",
                    label="Grid bin width (%d pts)" % len(g))
            if best_exp is not None:
                ax.plot(bc, k * (bc ** best_exp) + b, ":", color="purple",
                        linewidth=1.2, alpha=0.7, label="Raw instrument spacing")
            ax.legend(fontsize=7)
        ax.set_xlabel("m/z")
        ax.set_ylabel("Bin width (Th)")
        ax.set_title("Resulting adaptive grid")

        res_strs = []
        if best_exp is not None:
            for mz_c in (400, 1000, 2000, 4000):
                sp = k * (mz_c ** best_exp) + b
                if sp > 0:
                    res_strs.append("R@%d=%.0f" % (mz_c, mz_c / sp))
        eqn = ("delta = %.3e*(m/z)^%s + %.4f" % (k, best_exp, b)
               if best_exp is not None else "uniform-grid fallback")
        fig.suptitle("Adaptive grid QC%s  |  %s  |  %s\n%s" % (
            (" - " + label) if label else "", instrument.upper(),
            eqn, "   ".join(res_strs)), fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.93])

        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved adaptive-grid QC plot -> %s" % save_path)
    except Exception as _qc_err:
        print("  Adaptive-grid QC plot skipped: %s" % _qc_err)

#GETTING A NORMAL AVERAGING FUNCTION WAS VERY HARD. PyCharm's a.i. made this and it is perfect (I think...).... at least for UHMR 1k res ECCR data.
def improved_average_spectra(
		spectra_list,
		bin_width=None,
		apply_smoothing=True,
		smoothing_halfwidth=5,
		smoothing_degree=4,
		alignment_window=0.2,
		use_multiple_alignment_peaks=True,
		preserve_peak_shape=True,
        interpolation_method='linear',
        qc_plot_path=None,
        qc_label=""
):
	"""
	Improved method for averaging native MS spectra that addresses issues with jaggedness,
	valleys in peaks, and information loss. Uses a combination of multiple alignment points,
	adaptive binning, and gentle smoothing to preserve peak shapes.

	Args:
		spectra_list: List of (mz, intensity) arrays (profile-mode)
		bin_width: Optional m/z bin width. If None, auto-determines based on data
		apply_smoothing: Apply modified sinc smoothing after averaging
		smoothing_halfwidth: Half-width of sinc smoothing kernel
		smoothing_degree: Degree of smoothing kernel (lower is gentler)
		alignment_window: Window to search for peak shift alignment (in Th)
		use_multiple_alignment_peaks: Use multiple peaks for alignment instead of just one
		preserve_peak_shape: Use techniques to better preserve peak shapes
		interpolation_method: Method for interpolation ('linear', 'cubic', or 'quadratic')

	Returns:
		Averaged spectrum as (mz, intensity) array
	"""
	# Concatenate all m/z values to find global range
	all_mz = np.concatenate([spec[:, 0] for spec in spectra_list])
	mz_min, mz_max = np.min(all_mz), np.max(all_mz)

	# Create adaptive bin grid based on data characteristics
	if bin_width is None:
		# Use adaptive binning that accounts for peak density
		bins, _grid_info = estimate_adaptive_bin_grid(spectra_list, points_per_peak=1.0, min_spacing=0.0001, verbose=True)

		if qc_plot_path is not None: plot_grid_fit_qc(bins, _grid_info, qc_plot_path, qc_label)

	else:
		# Use fixed bin width if specified
		bins = np.arange(mz_min, mz_max + bin_width, bin_width)

	# Initialize array for summed intensities
	binned_intensities = np.zeros(len(bins) - 1)
	mz_centers = (bins[:-1] + bins[1:]) / 2

	# Find reference peaks for alignment
	if use_multiple_alignment_peaks:
		# Find multiple reference peaks across the m/z range for better alignment
		ref_spec = spectra_list[0]
		ref_mz, ref_int = ref_spec[:, 0], ref_spec[:, 1]

		# Normalize intensities for peak finding
		norm_int = ref_int / np.max(ref_int)

		# Find peaks above threshold (adjust threshold as needed)
		peak_indices = np.where((norm_int > 0.3) &
							   (np.r_[True, norm_int[1:] > norm_int[:-1]] &
								np.r_[norm_int[:-1] > norm_int[1:], True]))[0]

		# Select a subset of well-distributed peaks (up to 5)
		if len(peak_indices) > 5:
			# Divide m/z range into sections and pick strongest peak in each
			sections = 5
			section_size = len(ref_mz) // sections
			selected_peaks = []

			for i in range(sections):
				start_idx = i * section_size
				end_idx = (i + 1) * section_size if i < sections - 1 else len(ref_mz)
				section_peaks = [idx for idx in peak_indices if start_idx <= idx < end_idx]

				if section_peaks:
					# Get strongest peak in this section
					strongest = max(section_peaks, key=lambda idx: ref_int[idx])
					selected_peaks.append(strongest)

			peak_indices = selected_peaks

		# Get m/z values for reference peaks
		ref_peaks = [ref_mz[idx] for idx in peak_indices]

		# If no peaks found, use the strongest peak
		if not ref_peaks:
			ref_peaks = [ref_mz[np.argmax(ref_int)]]
	else:
		# Use single strongest peak as reference (original method)
		ref_mz, ref_int = spectra_list[0][:, 0], spectra_list[0][:, 1]
		ref_peaks = [ref_mz[np.argmax(ref_int)]]

	# Process each spectrum
	for spec in spectra_list:
		mz, intensity = spec[:, 0], spec[:, 1]

		if use_multiple_alignment_peaks and len(ref_peaks) > 1:
			# Multi-point alignment
			# Find corresponding peaks in this spectrum
			observed_peaks = []
			for ref_peak in ref_peaks:
				observed_peak = find_peak_near(mz, intensity, ref_peak, window=alignment_window)
				observed_peaks.append(observed_peak)

			# Calculate alignment shifts at each reference point
			shifts = [ref - obs for ref, obs in zip(ref_peaks, observed_peaks)]

			# Create alignment function that varies with m/z
			# Use linear interpolation between alignment points
			alignment_points = np.array(observed_peaks)
			shift_values = np.array(shifts)

			# Create interpolation function for the shift
			if len(observed_peaks) > 2:
				shift_interpolator = interp1d(
					alignment_points,
					shift_values,
					kind=min(3, len(observed_peaks)-1),  # Use cubic if possible
					bounds_error=False,
					fill_value=(shift_values[0], shift_values[-1])
				)

				# Apply varying shift across m/z range
				aligned_mz = mz + shift_interpolator(mz)
			else:
				# If only one or two points, use average shift
				avg_shift = np.mean(shifts)
				aligned_mz = mz + avg_shift
		else:
			# Single-point alignment (original method)
			observed_peak = find_peak_near(mz, intensity, ref_peaks[0], window=alignment_window)
			shift = ref_peaks[0] - observed_peak
			aligned_mz = mz + shift

		# Create aligned spectrum
		aligned_spec = np.column_stack((aligned_mz, intensity))

		if preserve_peak_shape:
			# Use interpolation to maintain peak shapes when binning
			interpolator = interp1d(
				aligned_mz,
				intensity,
				kind=interpolation_method,
				bounds_error=False,
				fill_value=0
			)

			# Interpolate onto bin centers for more accurate representation
			interpolated_intensity = interpolator(mz_centers)
			binned_intensities += interpolated_intensity
		else:
			# Use traditional binning (original method)
			bin_idx = np.digitize(aligned_mz, bins) - 1
			valid = (bin_idx >= 0) & (bin_idx < len(binned_intensities))
			np.add.at(binned_intensities, bin_idx[valid], intensity[valid])

	# Calculate average
	avg_intensity = binned_intensities / len(spectra_list)

	# Apply smoothing if requested
	if apply_smoothing:
		# Use gentler smoothing to preserve peak shapes
		if preserve_peak_shape:
			# Adjust smoothing parameters to be gentler
			effective_halfwidth = max(3, smoothing_halfwidth - 1)
			effective_degree = max(2, smoothing_degree - 1)
			avg_intensity = apply_modified_sinc_smooth(
				avg_intensity,
				effective_halfwidth,
				effective_degree
			)
		else:
			# Use original smoothing parameters
			avg_intensity = apply_modified_sinc_smooth(
				avg_intensity,
				smoothing_halfwidth,
				smoothing_degree
			)

	# Ensure no negative values
	avg_intensity = np.maximum(avg_intensity, 0)

	return np.column_stack((mz_centers, avg_intensity))

def average_native_ms_autoalign(
		spectra_list,
		bin_width=None,
		apply_smoothing=True,
		smoothing_halfwidth=5,
		smoothing_degree=4,
		alignment_window=0.2
):
	"""
	Averages native MS spectra using direct binning with auto-alignment to strongest internal peak.

	Note: This is the original implementation. For improved results with better peak shape
	preservation and reduced artifacts, use improved_average_spectra() instead.

	Args:
		spectra_list: List of (mz, intensity) arrays (profile-mode)
		bin_width: Optional m/z bin width. If None, auto-determines based on 1 ppm
		apply_smoothing: Apply modified sinc smoothing after averaging
		smoothing_halfwidth: Half-width of sinc smoothing kernel
		smoothing_degree: Degree of smoothing kernel
		alignment_window: Window to search for peak shift alignment (in Th)

	Returns:
		Averaged spectrum as (mz, intensity) array
	"""
	all_mz = np.concatenate([spec[:, 0] for spec in spectra_list])
	mz_min, mz_max = np.min(all_mz), np.max(all_mz)

	if bin_width is None:
		bins, _grid_info = estimate_adaptive_bin_grid(spectra_list, points_per_peak=1.0, min_spacing=0.0001, verbose=True)
	else:
		bins = np.arange(mz_min, mz_max + bin_width, bin_width)

	#bins = np.arange(mz_min, mz_max + bin_width, bin_width)
	binned_intensities = np.zeros(len(bins) - 1)

	# Reference peak: strongest peak in the first spectrum
	ref_mz, ref_int = spectra_list[0][:, 0], spectra_list[0][:, 1]
	ref_peak = ref_mz[np.argmax(ref_int)]

	for spec in spectra_list:
		mz, intensity = spec[:, 0], spec[:, 1]
		observed_peak = find_peak_near(mz, intensity, ref_peak, window=alignment_window)
		aligned_spec = align_spectrum_by_peak(spec, observed_peak, ref_peak)

		bin_idx = np.digitize(aligned_spec[:, 0], bins) - 1
		valid = (bin_idx >= 0) & (bin_idx < len(binned_intensities))
		np.add.at(binned_intensities, bin_idx[valid], aligned_spec[:, 1][valid])

	avg_intensity = binned_intensities / len(spectra_list)
	mz_centers = (bins[:-1] + bins[1:]) / 2

	if apply_smoothing:
		avg_intensity = apply_modified_sinc_smooth(avg_intensity, smoothing_halfwidth, smoothing_degree)

	return np.column_stack((mz_centers, avg_intensity))

def average_spectra(spectra_to_average, auto_optimal_mz_grid):
	"""
	Average multiple spectra using interpolation on an optimal m/z grid,
	with advanced smoothing based on modified sinc kernel principles.

	This implementation is based on the ModifiedSincSmoother algorithm
	described in Schmid and Diebold, "Why and how Savitzky-Golay filters
	should be replaced" (ACS Measurement Science Au, 2022).

	Args:
		spectra_to_average: List of (mz, intensity) arrays
		auto_optimal_mz_grid: Integer flag for grid creation method

	Returns:
		averaged spectrum as (mz, intensity) array
	"""
	# Create optimal m/z grid
	if auto_optimal_mz_grid == 3:
		optimal_mz_grid = create_optimal_mz_grid_auto(spectra_to_average)
	elif auto_optimal_mz_grid == 1:
		optimal_mz_grid = create_optimal_mz_grid_elementary(spectra_to_average)
	elif auto_optimal_mz_grid == 0:
		optimal_mz_grid = create_full_mz_grid_except_redundant(spectra_to_average)
	elif auto_optimal_mz_grid == 2:
		optimal_mz_grid = create_optimal_mz_grid(spectra_to_average)

	# Initialize array for summed intensities
	summed_intensity = np.zeros_like(optimal_mz_grid)

	# Step 1: Linearize each spectrum to a nonlinear grid with constant resolution
	# This ensures consistent representation across the m/z range
	linearized_spectra = []
	for spec in spectra_to_average:
		mz, intensity = spec[:, 0], spec[:, 1]
		# Use nonlinear axis with constant resolution (similar to MS instruments)
		# This helps preserve peak shapes across the m/z range
		min_mz = np.min(optimal_mz_grid)
		max_mz = np.max(optimal_mz_grid)
		resolution = 3000  # Typical resolution for high-res MS

		# Import linearize function from unidec.tools if not already imported
		import unidec.tools as ud

		# Create a temporary 2D array for linearization
		temp_data = np.column_stack((mz, intensity))

		# Use linearize with nonlinear axis (flag=1)
		# This maintains constant resolution across m/z range
		binsize = min_mz / resolution
		linearized = ud.linearize(temp_data, binsize, 1)
		linearized_spectra.append(linearized)

	# Step 2: Interpolate each linearized spectrum onto the optimal grid
	for spec in linearized_spectra:
		mz, intensity = spec[:, 0], spec[:, 1]
		interpolator = interp1d(mz, intensity, bounds_error=False, fill_value=0)
		summed_intensity += interpolator(optimal_mz_grid)

	# Calculate average
	averaged_intensity = summed_intensity / len(spectra_to_average)

	# Step 3: Apply modified sinc kernel smoothing to reduce noise while preserving peak shapes
	# Determine appropriate kernel parameters based on data characteristics
	data_length = len(averaged_intensity)

	# Calculate typical peak width to determine appropriate kernel size
	# For MS data, a good rule of thumb is to use a kernel half-width of about 1/20 of the data length
	# but not less than 5 or more than 25 points
	kernel_halfwidth = max(5, min(25, data_length // 20))

	# Use degree 4 for a good balance between smoothing and peak preservation
	# Higher degrees (6, 8, 10) provide sharper cutoff but may introduce ringing artifacts
	degree = 4

	# Apply modified sinc kernel smoothing
	smoothed_intensity = apply_modified_sinc_smooth(averaged_intensity, kernel_halfwidth, degree)

	# Ensure no negative values
	smoothed_intensity = np.maximum(smoothed_intensity, 0)

	# Return the smoothed averaged spectrum
	return np.column_stack((optimal_mz_grid, smoothed_intensity))

def iterate_files_by_type(directory, file_type):
    """
    Iterates through files of a specified type in a directory.

    Args:
    directory (str): The path to the directory.
    file_type (str): The file extension (e.g., ".txt", ".pdf").
    """
    file_df = pd.DataFrame(columns=["fname", "directory", "fpath"])
    for filename in os.listdir(directory):
        filename = filename.lower()
        if filename.endswith(file_type.lower()):
            file_path = os.path.join(directory, filename)
            # Perform operations on the file here
            print(f"Processing file: {file_path}")
            file_df.loc[len(file_df)] = [filename, directory, file_path]
    return file_df


def process_file(fname, file_dir_path, conf_file, cycle_length, scans_to_skip, n_to_av, tot_avs,
                 auto_optimal_mz_grid, extract_mass_list, extract_method, extract_window, mass_grid,
                 extract_from_raw=False, extract_mz_from_scans=None, extract_mzs_by_scan=None,
                 extract_mass_from_scans=False, extract_mass_scans=None):
    """
    Process a single file and return extracted values.

    This function has been optimized to only import and store scans that will be processed,
    which reduces memory usage and improves performance for large files where only a subset
    of spectra will be processed.

    Args:
        fname: File name
        file_dir_path: Path to directory with files
        conf_file: Configuration file path
        cycle_length: Length of each cycle
        scans_to_skip: Scans to skip in each cycle
        n_to_av: Number of scans to average
        tot_avs: Number of times for forward averaging
        auto_optimal_mz_grid: Flag for using optimal m/z grid
        extract_mass_list: List of masses to extract
        extract_method: Method for extraction
        extract_window: Window for extraction
        mass_grid: Parameters for mass grid [min, max, step]
        extract_from_raw: If True, extract values from averaged mz spectra using extract_mz_from_scans and extract_mzs_by_scan
        extract_mz_from_scans: Nested list of scan numbers to extract values from, corresponding to extract_mass_list
        extract_mzs_by_scan: Nested list of m/z values to extract from each scan in extract_mz_from_scans
        extract_mass_from_scans: If True, extract mass values from specific scans using extract_mass_scans
        extract_mass_scans: Nested list of scan numbers to extract mass values from, corresponding to extract_mass_list

    Returns:
        Dictionary with extracted values for each mass and cycle, and the mass array. AND NOW PEAK PICKING DICT
    """
    path = os.path.join(file_dir_path, fname)
    output_dir = os.path.join(file_dir_path, fname[0:-4])

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a single directory for all files from all processed files
    all_spectra_dir = os.path.join(file_dir_path, "all_spectra")
    if not os.path.exists(all_spectra_dir):
        os.makedirs(all_spectra_dir)

    # Import & store raw data
    im = dr.RawFileReader(path)
    num_scans = im.NumSpectra  # Get the total number of scans in the file
    print(f"{fname}\nTotal scans in file: {num_scans}")

    # OPTIMIZATION: Only import and store scans that will be processed
    # This reduces memory usage and improves performance for large files
    # where only a subset of spectra will be processed

    # Determine which scans will be needed based on parameters
    needed_scans = set()
    for timeCycle in range(tot_avs):
        for scan_index_in_cycle in range(cycle_length):
            if scan_index_in_cycle in scans_to_skip:
                continue

            scanStartNumber = (timeCycle) * cycle_length + 1
            scanStopNumber = scanStartNumber + n_to_av * cycle_length

            for scan in range(scanStartNumber, scanStopNumber):
                scan_num_temp = (scan - scanStartNumber) % cycle_length
                if scan_num_temp == scan_index_in_cycle:
                    needed_scans.add(scan)

    # Convert to sorted list for more predictable iteration (helpful for debugging)
    needed_scans_list = sorted(list(needed_scans))
    print(
        f"Total scans needed: {len(needed_scans_list)} out of {num_scans} ({(len(needed_scans_list) / num_scans) * 100:.2f}%)")

    # Dictionary to store only the needed spectra
    scans_dict = {}
    # Extract and store only the needed scans
    for scan in needed_scans_list:
        scans_dict[scan] = im.GetSpectrum(scanNumber=scan)

    # Dictionary to store averaged spectra
    scan_av_dict = {}
    mass_av_dict = {}
    mass_sum_dict = {}
    saved_peaks = {}
    filtered_peaks_by_mz = {}
    # Dictionary to store peak picking data
    picked_peak_df = pd.DataFrame(columns=["scan", "mass", "peak_int", "peak_std", "peak_std_std"])  # PLACEHOLDER

    # Initialize mass grid for averaging
    standard_mass_grid = np.arange(mass_grid[0], mass_grid[1], mass_grid[2])
    mass_data_zeros = np.column_stack(
        (standard_mass_grid, np.zeros_like(standard_mass_grid, dtype='float64')))
    mass_sum_array_all = np.column_stack(
        (standard_mass_grid, np.zeros([len(standard_mass_grid), tot_avs])))
    print(f"Standard mass grid: {standard_mass_grid}")

    # loop counter for dscore peaks
    peak_picking_counter = 0
    # Extract averages and store average scans
    for timeCycle in range(tot_avs):
        for scan_index_in_cycle in range(cycle_length):
            if scan_index_in_cycle in scans_to_skip:
                continue

            # Begin collecting scans to average
            spectra_to_average = []
            scanStartNumber = (timeCycle) * cycle_length + 1
            scanStopNumber = scanStartNumber + n_to_av * cycle_length

            for scan in range(scanStartNumber, scanStopNumber):
                scan_num_temp = (scan - scanStartNumber) % cycle_length
                if scan_num_temp == scan_index_in_cycle:
                    spectra_to_average.append(scans_dict[scan])

            if spectra_to_average:
                # Use the improved averaging method for better peak shape preservation
                averaged_spectrum = improved_average_spectra(
                    spectra_to_average,
                    apply_smoothing=False,
                    smoothing_halfwidth=5,
                    smoothing_degree=4,
                    use_multiple_alignment_peaks=True,
                    preserve_peak_shape=True,
                    interpolation_method='linear',
                    qc_plot_path=os.path.join(output_dir, f"gridfit_QC_scan{scan_index_in_cycle}_cyc{timeCycle}.png"),
                    qc_label=f"scan {scan_index_in_cycle}, cycle {timeCycle}"
                )
                scan_av_dict[scan_index_in_cycle] = averaged_spectrum

        # Write averages for specified cycle
        picked_peaks_mass_v_intensity = []
        for scani, dat in scan_av_dict.items():
            # Define output file path for the averaged scan (directly in output directory)
            output_file = os.path.join(output_dir, f"averaged_scan{scani}_cyc{timeCycle}.txt")
            # Write the two-column data (m/z and intensity) to the text file
            np.savetxt(output_file, dat, fmt="%.18e")
            print(f"Saved averaged scan {scani} to {output_file}")

            # Save the same file to the all_spectra_dir with modified filename
            # Use first 15 characters of the original filename
            short_fname = fname  # [33:55] if len(fname) > 22 else fname[:-4]
            all_spectra_file = os.path.join(all_spectra_dir, f"averaged_scan{scani}_cyc{timeCycle}_{short_fname}.txt")
            np.savetxt(all_spectra_file, dat, fmt="%.18e")
            print(f"Saved averaged scan {scani} to {all_spectra_file}")

            # Now unidec to get masses
            mz = dat[:, 0]
            intensity = dat[:, 1]
            # Create a UniDec engine
            eng = unidec.UniDec()
            eng.load_config(conf_file)
            inputdata1 = np.transpose([mz, intensity])
            inputdata = ud.dataprep(inputdata1, eng.config, peaks=False, intthresh=False, silent=True)
            # Run UniDec
            eng.pass_data_in(inputdata, silent=True, refresh=True)
            eng.process_data(silent=True)

            # peak picking, filtering and summing logic below
            if pick_peaks:
                eng.run_unidec(silent=True, efficiency=False)
                deconv_masses_max_int = max(eng.data.massdat[:, 1])
                raw_peak_picking_int = picked_peak_raw_int_THRESH
                eng.config.peakthresh = raw_peak_picking_int / deconv_masses_max_int
                if picked_peak_raw_int_THRESH < 1:
                    eng.config.peakthresh = picked_peak_raw_int_THRESH
                if picked_peak_raw_int_THRESH == 0:
                    eng.config.peakthresh = 0.0001
                eng.config.peakwindow = 25
                eng.pick_peaks(calc_dscore=True)
                calculated_scan_center = picked_peak_scan_baseMZ + int(scani)*picked_peak_scan_spacing + picked_peak_scan_spacing/2.0
                calculated_scan_low_mz = calculated_scan_center - picked_peak_scan_spacing/2.0
                calculated_scan_high_mz = calculated_scan_center + picked_peak_scan_spacing / 2.0
                #filtering
                if filter_peaks_by_score:
                    eng.filter_peaks(minscore=filter_peaks_by_score)
                ignore_peaks_indices = []
                if filter_peaks_by_window_mz:
                    for i, peak in enumerate(eng.pks.peaks):
                        z_close_temp = 0
                        mz_close_temp = 0
                        for z_temp in range(1, 100):
                            mz_calc = (peak.mass + z_temp) / float(z_temp)
                            if abs(mz_calc - calculated_scan_center) < abs(mz_close_temp - calculated_scan_center):
                                z_close_temp = z_temp
                                mz_close_temp = mz_calc
                            if (mz_calc >= calculated_scan_low_mz - abs(filter_peaks_by_window_mz_thresh[0])) * (
                                    mz_calc <= calculated_scan_high_mz + abs(filter_peaks_by_window_mz_thresh[1])):
                                #print(f"Keeping peak: {peak.mass} (Dscore {peak.dscore:.2f}) from window: {calculated_scan_center}mz, {z_temp}+ @ {mz_calc:.2f}mz")
                                break
                            if z_temp == 99:

                                try:
                                    filtered_peaks_by_mz[scani].append(peak)
                                except:
                                    filtered_peaks_by_mz[scani] = peak
                                print(
                                    f"DIFF {mz_close_temp - calculated_scan_center:.2f}! Filtered {peak.mass} Da, {peak.height:.1e} (Dscore {peak.dscore:.2f}),  from window: {calculated_scan_center}mz. Closest: {z_close_temp}+ @ {mz_close_temp:.2f}mz")
                                eng.pks.peaks.remove(peak)

                # Back to peak logic. Making DF
                df_peaks_TEMP = eng.pks.to_df(type="Full", drop_zeros=False)
                df_peaks_TEMP.insert(loc=1, column='File', value=fname)#['File'] = fname
                df_peaks_TEMP.insert(loc=2, column = 'Cyc', value = timeCycle)
                df_peaks_TEMP.insert(loc=3, column='Scan', value=scani)#['Scan'] = scani df.insert(loc=1, column='C', value=10)
                df_peaks_TEMP.insert(loc=4, column='Scan Center', value=(calculated_scan_center))#['Scan'] = scani df.insert(loc=1, column='C', value=10)picked_peak_scan_baseMZ
                df_peaks_TEMP.insert(loc=5, column='Estimated Charge', value=round(df_peaks_TEMP['Mass'].astype(float)/df_peaks_TEMP['Scan Center'], 0)  )
                #df_peaks_TEMP['File'] = fname
                if peak_picking_counter == 0: picked_peak_df = df_peaks_TEMP
                if peak_picking_counter != 0: picked_peak_df = pd.concat([picked_peak_df, df_peaks_TEMP], ignore_index=True)
                #if peak_picking_counter == 0: rolling_peaks = eng.pks
                #if peak_picking_counter != 0: rolling_peaks.merge_in_peaks(eng.pks,filename=timeCycle,filenumber=scani)
                #picked_peak_df = rolling_peaks.to_df(type="Full", drop_zeros=False)
                saved_peaks[scani] = eng.pks.peaks #TODO add peak picking/filtering step where peaks are filtered if they appear across different scans (done above... could be improved)
                if extract_picked_peak_spectrum:
                    try:
                        mass, intensity_mass = eng.data.massdat[:, 0], eng.data.massdat[:, 1]*0 #intensity vals = 0
                        for peak in eng.pks.peaks:
                            #Add intensities only for picked peaks
                            intensity_mass[(mass <= peak.mdist[-1,0])*(mass >= peak.mdist[0,0])] += peak.mdist[:, 1] * peak.height
                    except:
                        mass, intensity_mass = [], []                    #mass_index = mass_grid*0

                #else:
                #    continue
                peak_picking_counter += 1
            else:
                eng.run_unidec(silent=True, efficiency=True)

            if not extract_picked_peak_spectrum or not pick_peaks:
                try:
                    mass, intensity_mass = eng.data.massdat[:, 0], eng.data.massdat[:, 1]
                except:
                    mass, intensity_mass = [], []

            print(len(mass))
            if len(mass) != 0:
                interpolator_mass = interp1d(mass, intensity_mass, bounds_error=False,
                                             fill_value=0)
                interpolated_mass_intensity = interpolator_mass(standard_mass_grid) / n_to_av
                mass_av_dict[scani] = np.transpose([standard_mass_grid, interpolated_mass_intensity])
            else:
                zeros_array = np.zeros_like(standard_mass_grid, dtype='float64')
                mass_av_dict[scani] = np.transpose([standard_mass_grid, zeros_array])

        mass_sum_temp = np.copy(mass_data_zeros)
        for massi, dat_mass in mass_av_dict.items():


            # Define output file path for the averaged mass scan (directly in output directory)
            output_file_mass = os.path.join(output_dir, f"av_mass_scan{massi}_cyc{timeCycle}.txt")
            # Write the two-column data (m/z and intensity) to the text file
            np.savetxt(output_file_mass, dat_mass, fmt="%.18e")
            print(f"Saved averaged mass scan {massi} to {output_file_mass}")

            # Save the same file to the all_spectra_dir with modified filename
            # Use first 15 characters of the original filename
            short_fname = fname  # [33:55] if len(fname) > 22 else fname[:-4]
            all_spectra_file_mass = os.path.join(all_spectra_dir,
                                                 f"av_mass_scan{massi}_cyc{timeCycle}_{short_fname}.txt")
            np.savetxt(all_spectra_file_mass, dat_mass, fmt="%.18e")
            print(f"Saved averaged mass scan {massi} to {all_spectra_file_mass}")

            # NOW SUM MASSES
            mass_sum_temp[:, 1] += dat_mass[:, 1]

        # Define output file path for the summed mass spectrum (directly in output directory)
        output_file_mass_sum = os.path.join(output_dir, f"summed_mass_cyc{timeCycle}.txt")
        # Write the two-column data (m/z and intensity) to the text file
        np.savetxt(output_file_mass_sum, mass_sum_temp, fmt="%.18e")
        print(f"Saved summed masses cycle {timeCycle} to {output_file_mass_sum}")

        # Save the same file to the all_spectra_dir with modified filename
        # Use first 15 characters of the original filename
        short_fname = fname  # [:-4] #short_fname = fname[33:55] if len(fname) > 22 else fname[:-4]
        all_spectra_file_sum = os.path.join(all_spectra_dir, f"summed_mass_cyc{timeCycle}_{short_fname}.txt")
        np.savetxt(all_spectra_file_sum, mass_sum_temp, fmt="%.18e")
        print(f"Saved summed masses cycle {timeCycle} to {all_spectra_file_sum}")

        # Save cycle sums
        mass_sum_dict[timeCycle] = mass_sum_temp
        mass_sum_array_all[:, timeCycle + 1] = mass_sum_temp[:, 1]

        # Generate comprehensive cycle plots
        # Extract the correct cycle data from mass_sum_array_all
        # Column 0 = mass grid, Column (timeCycle+1) = this cycle's intensities
        cycle_mass_data = np.column_stack([
            mass_sum_array_all[:, 0],  # Mass grid
            mass_sum_array_all[:, timeCycle + 1]  # This cycle's intensities
        ])

        plot_cycle_analysis_inline(
            saved_peaks=saved_peaks,
            scan_av_dict=scan_av_dict,
            mass_av_dict=mass_sum_array_all,
            cycle_num=timeCycle,
            output_dir=output_dir,
            max_scan_intensity_dict=None,
            mass_sum_data=cycle_mass_data
        )

        # Reset dictionaries if averaging more than one cycle
        if timeCycle < tot_avs - 1:
            scan_av_dict = {}
            mass_av_dict = {}

    #PRINT PICKED PEAKS
    if pick_peaks:
        picked_peak_df.to_csv(os.path.join(output_dir, fname[:-4] + "_ppeaks.csv"), index=False)
        print(f"Saved picked peaks to {os.path.join(output_dir, fname[:-4] + '_ppeaks.csv')}")

    # Define output file path for the summed masses vs cycle (directly in output directory)
    output_file_full_mass_array = os.path.join(output_dir, fname[:-4] + "_array.csv")
    # Write the two-column data (m/z and intensity) to the text file
    np.savetxt(output_file_full_mass_array, mass_sum_array_all, delimiter=",", fmt="%.18e")
    print(f"Saved mass vs cycle dataset to {output_file_full_mass_array}")

    # Save the same file to the all_spectra_dir with modified filename
    # Use first 15 characters of the original filename
    short_fname = fname  # [33:55] if len(fname) > 22 else fname[:-4]
    all_spectra_file_array = os.path.join(all_spectra_dir, f"array_{short_fname}.csv")
    np.savetxt(all_spectra_file_array, mass_sum_array_all, delimiter=",", fmt="%.18e")
    print(f"Saved mass vs cycle dataset to {all_spectra_file_array}")

    # Extract masses and store results
    extracted_values = {}
    print(fname)

    # Standard extraction method
    if not extract_from_raw and not extract_mass_from_scans:
        for w, mass_to_extract in enumerate(extract_mass_list):
            extracted_values[mass_to_extract] = []
            print("extracting mass:", str(mass_to_extract))
            for i in range(tot_avs):
                data = np.transpose([mass_sum_array_all[:, 0], mass_sum_array_all[:, i + 1]])

                window = None
                if extract_window:
                    window = extract_window[w]

                extracted_data = ud.data_extract(data, mass_to_extract, extract_method, window=window)
                extracted_values[mass_to_extract].append(extracted_data)
                print(str(extracted_data))

    # Extract from raw m/z spectra
    elif extract_from_raw and extract_mz_from_scans is not None and extract_mzs_by_scan is not None:
        for w, mass_placeholder in enumerate(extract_mass_list):
            extracted_values[mass_placeholder] = []
            print(f"extracting from raw spectra for placeholder mass: {mass_placeholder}")

            # Get the scan numbers and m/z values for this mass
            if w < len(extract_mz_from_scans):
                scan_numbers = extract_mz_from_scans[w]
                mz_values_by_scan = extract_mzs_by_scan[w]

                # Check if the lists have the same length
                if len(scan_numbers) != len(mz_values_by_scan):
                    print(
                        f"Warning: scan_numbers and mz_values_by_scan have different lengths for mass {mass_placeholder}")
                    continue

                # Extract values from each scan
                total_value = 0
                for j, (scan_num, mz_values) in enumerate(zip(scan_numbers, mz_values_by_scan)):
                    if scan_num in scan_av_dict:
                        scan_data = scan_av_dict[scan_num]

                        # Extract values for each m/z in this scan
                        scan_value = 0
                        for mz in mz_values:
                            # Use the data_extract function to extract the value at this m/z
                            value = ud.data_extract(scan_data, mz, extract_method,
                                                    window=5)  # Use a small window for m/z extraction
                            scan_value += value

                        total_value += scan_value
                        print(f"  Scan {scan_num}, m/z values {mz_values}, extracted value: {scan_value}")
                    else:
                        print(f"  Warning: Scan {scan_num} not found in scan_av_dict")

                # Store the total extracted value for this mass
                for i in range(tot_avs):
                    extracted_values[mass_placeholder].append(total_value)
                print(f"  Total extracted value: {total_value}")
            else:
                print(f"  Warning: No scan numbers defined for mass {mass_placeholder}")
                for i in range(tot_avs):
                    extracted_values[mass_placeholder].append(0)

    # Extract masses from specific scans
    elif extract_mass_from_scans and extract_mass_scans is not None:
        for w, mass_to_extract in enumerate(extract_mass_list):
            extracted_values[mass_to_extract] = []
            print(f"extracting mass {mass_to_extract} from specific scans")

            # Get the scan numbers for this mass
            if w < len(extract_mass_scans):
                scan_numbers = extract_mass_scans[w]

                # Extract values from each scan
                total_value = 0
                for scan_num in scan_numbers:
                    if scan_num in mass_av_dict:
                        mass_data = mass_av_dict[scan_num]

                        window = None
                        if extract_window:
                            window = extract_window[w]

                        # Use the data_extract function to extract the value at this mass
                        value = ud.data_extract(mass_data, mass_to_extract, extract_method, window=window)
                        total_value += value
                        print(f"  Scan {scan_num}, extracted value: {value}")
                    else:
                        print(f"  Warning: Scan {scan_num} not found in mass_av_dict")

                # Store the total extracted value for this mass
                for i in range(tot_avs):
                    extracted_values[mass_to_extract].append(total_value)
                print(f"  Total extracted value: {total_value}")
            else:
                print(f"  Warning: No scan numbers defined for mass {mass_to_extract}")
                for i in range(tot_avs):
                    extracted_values[mass_to_extract].append(0)

    # Create HDF5 files for each scan number, containing spectra from all cycles
    # We need to read the saved files for all cycles since scan_av_dict and mass_av_dict only contain data for the last cycle

    # First, identify all valid scan indices
    valid_scan_indices = [i for i in range(cycle_length) if i not in scans_to_skip]

    # For each scan index, create an HDF5 file containing data from all cycles
    for scan_index_in_cycle in valid_scan_indices:
        # Create HDF5 file for the m/z spectra of this scan from all cycles
        mz_hdf5_file = os.path.join(output_dir, f"mz_scan{scan_index_in_cycle}.hdf5")
        mz_engine = MetaUniDec()
        mz_engine.data.new_file(mz_hdf5_file)

        # Create HDF5 file for the mass spectra of this scan from all cycles
        mass_hdf5_file = os.path.join(output_dir, f"mass_scan{scan_index_in_cycle}.hdf5")
        mass_engine = MetaUniDec()
        mass_engine.data.new_file(mass_hdf5_file)

        # Add data from all cycles for this scan index
        for timeCycle in range(tot_avs):
            # Add m/z spectrum for this cycle
            mz_file_path = os.path.join(output_dir, f"averaged_scan{scan_index_in_cycle}_cyc{timeCycle}.txt")
            if os.path.exists(mz_file_path):
                # Add the m/z spectrum file to the mz_engine
                mz_engine.data.add_file(path=mz_file_path)

                # Set metadata for the spectrum
                mz_engine.data.spectra[-1].var1 = timeCycle
                mz_engine.data.spectra[-1].var2 = scan_index_in_cycle
                mz_engine.data.spectra[-1].name = f"cycle{timeCycle}_scan{scan_index_in_cycle}"
                mz_engine.data.spectra[-1].attrs["cycle"] = timeCycle
                mz_engine.data.spectra[-1].attrs["scan"] = scan_index_in_cycle

            # Add mass spectrum for this cycle
            mass_file_path = os.path.join(output_dir, f"av_mass_scan{scan_index_in_cycle}_cyc{timeCycle}.txt")
            if os.path.exists(mass_file_path):
                # Add the mass spectrum file to the mass_engine
                mass_engine.data.add_file(path=mass_file_path)

                # Set metadata for the spectrum
                mass_engine.data.spectra[-1].var1 = timeCycle
                mass_engine.data.spectra[-1].var2 = scan_index_in_cycle
                mass_engine.data.spectra[-1].name = f"cycle{timeCycle}_scan{scan_index_in_cycle}"
                mass_engine.data.spectra[-1].attrs["cycle"] = timeCycle
                mass_engine.data.spectra[-1].attrs["scan"] = scan_index_in_cycle

        # Export the m/z spectra to HDF5
        mz_engine.data.export_hdf5()
        print(f"Saved m/z spectra for scan {scan_index_in_cycle} from all cycles to {mz_hdf5_file}")

        # Export the mass spectra to HDF5
        mass_engine.data.export_hdf5()
        print(f"Saved mass spectra for scan {scan_index_in_cycle} from all cycles to {mass_hdf5_file}")

    # Return the extracted values, mass array, and the scan_av_dict for the last cycle
    return extracted_values, mass_sum_array_all, scan_av_dict, mass_av_dict, picked_peak_df

def plot_overlayed_spectra(mass_arrays, file_names, output_dir, title="Overlayed Mass Spectra", save_png=True):
	"""
	Plot overlayed spectra of mass arrays with no zoom.

	Args:
		mass_arrays: List of mass arrays, each with shape (n, 2+) where column 0 is mass and column 1 is intensity
		file_names: List of file names corresponding to the mass arrays
		output_dir: Directory to save the plot
		title: Plot title
		save_png: Whether to save the figure as a PNG file

	Returns:
		The figure object
	"""
	fig = plt.figure(figsize=(12, 8))

	# Get the jet colormap
	cmap = matplotlib.colormaps['jet']

	# Sort file names and corresponding mass arrays
	sorted_indices = np.argsort([os.path.basename(f) for f in file_names])
	sorted_file_names = [file_names[i] for i in sorted_indices]
	sorted_mass_arrays = [mass_arrays[i] for i in sorted_indices]

	# Calculate max intensity for each spectrum
	max_intensities = []
	for mass_array in sorted_mass_arrays:
		max_intensities.append(np.max(mass_array[:, 1]))

	# Plot each spectrum with offset
	offset_step = 0.05
	for i, (mass_array, fname) in enumerate(zip(sorted_mass_arrays, sorted_file_names)):
		# Normalize to max intensity of this spectrum
		normalized_intensity = mass_array[:, 1] / max_intensities[i]

		# Apply offset
		offset = i * offset_step

		# Get color from jet colormap
		color = cmap(i / len(sorted_mass_arrays))

		# Plot the spectrum
		plt.plot(mass_array[:, 0], normalized_intensity + offset, color=color, label=os.path.basename(fname))

		# Add file name label on right side
		plt.text(mass_array[-1, 0] * 1.01, offset, os.path.basename(fname), color=color, va='center', fontsize=7)

	plt.xlabel('Mass (Da)')
	plt.ylabel('Normalized Intensity')
	plt.title(title)
	plt.grid(True, alpha=0.3)

	# Save the figure
	plt.tight_layout()
	if save_png:
		plt.savefig(os.path.join(output_dir, "overlayed_mass_spectra.png"), dpi=300)
	plt.close(fig)
	return fig

def plot_zoomed_spectra(mass_arrays, file_names, extract_mass_list, extract_window, output_dir, save_png=True):
	"""
	Plot zoomed-in overlayed spectra of mass arrays for each extraction window.

	Args:
		mass_arrays: List of mass arrays, each with shape (n, 2+) where column 0 is mass and column 1 is intensity
		file_names: List of file names corresponding to the mass arrays
		extract_mass_list: List of masses to extract
		extract_window: List of windows for extraction
		output_dir: Directory to save the plots
		save_png: Whether to save the figures as PNG files

	Returns:
		List of figure objects
	"""
	# Sort file names and corresponding mass arrays
	sorted_indices = np.argsort([os.path.basename(f) for f in file_names])
	sorted_file_names = [file_names[i] for i in sorted_indices]
	sorted_mass_arrays = [mass_arrays[i] for i in sorted_indices]

	# Get the jet colormap
	cmap = matplotlib.colormaps['jet']

	# List to store figure objects
	figures = []

	# Create a plot for each extraction window
	for w, (mass_to_plot, window) in enumerate(zip(extract_mass_list, extract_window)):
		print("strating: "+str(mass_to_plot))
		fig = plt.figure(figsize=(12, 8))
		figures.append(fig)

		# Define the extraction window
		x_min = mass_to_plot - 3 * window
		x_max = mass_to_plot + 3 * window
		if window < 300:
			x_min = mass_to_plot - 900
			x_max = mass_to_plot + 900

		# Find the maximum intensity within each window for each file
		max_intensities = [[0] * len(extract_mass_list) for _ in range(len(sorted_mass_arrays))]
		for f, mass_array in enumerate(sorted_mass_arrays):
			for i, (mass_to_plot, window) in enumerate(zip(extract_mass_list, extract_window)):
				# Filter data within the window
				mask = (mass_array[:, 0] >= (mass_to_plot - window)) & (mass_array[:, 0] <= (mass_to_plot + window))
				if np.any(mask):
					max_intensities[f][i] = np.max(mass_array[mask, 1])

		# Plot each spectrum with offset
		offset_step = 0.025
		limit_y_max = 0
		for i, (mass_array, fname) in enumerate(zip(sorted_mass_arrays, sorted_file_names)):
			# Filter data within the window
			mask = (mass_array[:, 0] >= x_min) & (mass_array[:, 0] <= x_max)

			if np.any(mask):
				# Normalize to max intensity within the window for this file
				normalized_intensity = mass_array[mask, 1] / max_intensities[i][w]
				if max(normalized_intensity) > 5: limit_y_max = 1

				# Apply offset
				offset = i * offset_step

				# Get color from jet colormap
				color = cmap(i / len(sorted_mass_arrays))

				# Plot the spectrum
				plt.plot(mass_array[mask, 0], normalized_intensity + offset, color=color, label=os.path.basename(fname))

				# Add file name label on right side
				plt.text(x_max * 1.01, offset, os.path.basename(fname), color=color, va='center', fontsize=7)

		plt.xlabel('Mass (Da)')
		plt.ylabel('Normalized Intensity')
		plt.title(f'Zoomed Mass Spectra - Mass: {mass_to_plot}, Window: {window}')
		print(str(mass_to_plot))
		plt.xlim(x_min, x_max)
		if limit_y_max: plt.ylim(0, 5)
		plt.grid(True, alpha=0.3)

		# Save the figure
		plt.tight_layout()
		if save_png:
			plt.savefig(os.path.join(output_dir, f"zoomed_mass_spectra_{mass_to_plot}.png"), dpi=300)
		plt.close(fig)

	return figures

def save_figures_to_pdf(figures, output_dir, filename="all_plots.pdf"):
	"""
	Save a list of figures to a single PDF file.

	Args:
		figures: List of figure objects to save
		output_dir: Directory to save the PDF file
		filename: Name of the PDF file
	"""
	pdf_path = os.path.join(output_dir, filename)
	with PdfPages(pdf_path) as pdf:
		for fig in figures:
			pdf.savefig(fig)
	print(f"Saved all plots to {pdf_path}")

def plot_raw_ms_spectra(scan_av_dicts, file_names, output_dir, extract_mass_list, extract_window, conf_file, mass_av_dicts, title="Overlayed Raw MS Spectra", save_png=True):
	"""
	Plot raw MS spectra for each extracted mass, showing the scan that contributed most to each mass.

	Args:
		scan_av_dicts: List of dictionaries containing averaged spectra for each scan
		file_names: List of file names corresponding to the scan dictionaries
		output_dir: Directory to save the plots
		extract_mass_list: List of masses to extract
		extract_window: List of windows for extraction
		title: Plot title
		save_png: Whether to save the figures as PNG files

	Returns:
		List of figure objects
	"""
	# Sort file names and corresponding scan dictionaries
	sorted_indices = np.argsort([os.path.basename(f) for f in file_names])
	sorted_file_names = [file_names[i] for i in sorted_indices]
	sorted_scan_av_dicts = [scan_av_dicts[i] for i in sorted_indices]
	sorted_mass_av_dicts = [mass_av_dicts[i] for i in sorted_indices]

	# Get the jet colormap
	cmap = matplotlib.colormaps['jet']

	# List to store figure objects
	figures = []

	# For each extracted mass
	for w, (mass_to_extract, window) in enumerate(zip(extract_mass_list, extract_window)):
		print(f"Processing mass: {mass_to_extract}")

		# Dictionary to store the best scan for each file
		best_scans = {}
		best_scan_contributions = {}



		# For each file, find which scan contributed most to the extracted mass
		for f, (scan_av_dict, mass_av_dict, fname) in enumerate(
				zip(sorted_scan_av_dicts, sorted_mass_av_dicts, sorted_file_names)):
			max_contribution = 0
			total_contribution = 0
			best_scan = None

			# Check each scan's contribution to the extracted mass
			# Fix: Iterate over the dictionary items properly
			for scan_num, scan_data_mass in mass_av_dict.items():  # Changed mass_av_dicts to mass_av_dict
				try:
					mass, intensity_mass = scan_data_mass[:, 0], scan_data_mass[:, 1]
					if len(mass) > 0:
						# Extract the contribution of this scan to the mass
						mass_data = np.column_stack((mass, intensity_mass))
						contribution = ud.data_extract(mass_data, mass_to_extract, 2, window=window)

						if contribution > max_contribution:
							max_contribution = contribution
							best_scan = scan_num
					else:
						continue
				except:
					continue

			best_scans[fname] = best_scan
			best_scan_contributions[fname] = max_contribution
			percent_contribution = 100.0 * max_contribution / total_contribution if total_contribution > 0 else 0
			print(
				f"File: {os.path.basename(fname)}, Best scan: {best_scan}, Contribution: {percent_contribution:.2f}% ({max_contribution} of {total_contribution})")

		# Create figure for overlayed mass spectra from best scans
		fig_mass = plt.figure(figsize=(12, 8))
		figures.append(fig_mass)

		# Plot overlayed mass spectra from best scans
		offset_step = 0.025
		for i, (fname, mass_av_dict) in enumerate(zip(sorted_file_names, sorted_mass_av_dicts)):
			best_scan = best_scans[fname]
			if best_scan is None:
				continue

			# Get the mass spectrum for the best scan
			try:
				# Process the scan data through UniDec
				mass_data = mass_av_dict[best_scan]
				mass, intensity_mass = mass_data[:, 0], mass_data[:, 1]
				print("len mass data:"+str( len(mass_data)))
				if len(mass) == 0:
					continue

				mass_data = np.column_stack((mass, intensity_mass))

				# Filter data within the window
				x_min = mass_to_extract - 3 * window
				x_max = mass_to_extract + 3 * window
				if window < 300:
					x_min = mass_to_extract - 900
					x_max = mass_to_extract + 900

				mask = (mass_data[:, 0] >= x_min) & (mass_data[:, 0] <= x_max)

				if np.any(mask):
					# Normalize to max intensity within the window
					max_intensity = np.max(mass_data[mask, 1])
					normalized_intensity = mass_data[mask, 1] / max_intensity

					# Apply offset
					offset = i * offset_step

					# Get color from jet colormap
					color = cmap(i / len(sorted_file_names))

					# Plot the spectrum
					plt.plot(mass_data[mask, 0], normalized_intensity + offset, color=color, label=os.path.basename(fname))

					# Add file name label on right side
					plt.text(x_max * 1.01, offset, f"{os.path.basename(fname)} (Scan {best_scan})", color=color, va='center', fontsize=7)
			except Exception as e:
				print(f"Error processing {fname}, scan {best_scan}: {e}")
				continue

	#for fname, mass_av_dict in zip(sorted_file_names, sorted_mass_av_dicts):
	#        best_scan = best_scans[fname]
	#        if best_scan is None:

		plt.xlabel('Mass (Da)')
		plt.ylabel('Normalized Intensity')
		plt.title(f'Best Mass Spectra - Mass: {mass_to_extract}, Window: {window}' )
				  #f'")#/nFile: {os.path.basename(fname)}, Best scan: {best_scan}, Contribution: {percent_contribution:.2f}% ({max_contribution} of {total_contribution})")
		plt.xlim(mass_to_extract-3*window, mass_to_extract-3+window)
		plt.grid(True, alpha=0.3)

		# Save the figure
		plt.tight_layout()
		if save_png:
			plt.savefig(os.path.join(output_dir, f"best_mass_spectra_{mass_to_extract}.png"), dpi=300)

		# Create figure for raw MS spectra from best scans
		fig_raw = plt.figure(figsize=(12, 8))
		figures.append(fig_raw)

		# Plot raw MS spectra from best scans
		for i, (fname, scan_av_dict) in enumerate(zip(sorted_file_names, sorted_scan_av_dicts)):
			best_scan = best_scans[fname]
			if best_scan is None:
				continue

			# Get the raw MS spectrum for the best scan
			try:
				scan_data = scan_av_dict[best_scan]

				# Normalize to max intensity
				max_intensity = np.max(scan_data[:, 1])
				normalized_intensity = scan_data[:, 1] / max_intensity

				# Apply offset
				offset = i * offset_step

				# Get color from jet colormap
				color = cmap(i / len(sorted_file_names))

				# Plot the spectrum
				plt.plot(scan_data[:, 0], normalized_intensity + offset, color=color, label=os.path.basename(fname))

				# Add file name label on right side
				plt.text(scan_data[-1, 0] * 1.01, offset, f"{os.path.basename(fname)} (Scan {best_scan})", color=color, va='center', fontsize=7)
			except Exception as e:
				print(f"Error processing raw MS for {fname}, scan {best_scan}: {e}")
				continue

		plt.xlabel('m/z')
		plt.ylabel('Normalized Intensity')
		plt.title(f'Raw MS Spectra - Mass: {mass_to_extract}, Window: {window}')
		plt.grid(True, alpha=0.3)

		# Save the figure
		plt.tight_layout()
		if save_png:
			plt.savefig(os.path.join(output_dir, f"raw_ms_spectra_{mass_to_extract}.png"), dpi=300)

	# Save all figures to PDF
	save_figures_to_pdf(figures, output_dir, filename="raw_ms_spectra.pdf")

	return figures


# Main execution
if __name__ == "__main__":
    ############################################################################
    ############################################################################
    #          ADJUST PARAMETERS HERE       ####################################
    ############################################################################
    ############################################################################

    ### Define parameters for scan processing
    # Define cycle length (in # scans/cycle) for your files
    cycle_length=140
    # this value should be actual value - 1, due to python indexing. Cycle length above is normal
    scans_to_skip = [i for i in range(140) if (i >= 70)]#[32]#[i for i in range(140) if (i >= 70)]#[i for i in range(140) if (i >= 70)]#AscendPTCR[52]#[i for i in range(140) if (i >= 70)]
    # Number of scans to average
    n_to_av = 5
    # Number of cycle rounds (e.g. time-points)
    tot_avs = 1

    # File path of directory with raw files
    file_dir_path = 'C:/Users/andre/OneDrive/Desktop/25PC/25Cd/25Cd19_PTCR_Ascend_revisit25/260524_anal'#'C:/Users/andre/OneDrive/Desktop/25PC/25Cd/25Cd16_Protease_KOs_1/250513_CdKOs1/16-DIP0964_SLOMO/DATA/SLOMO'
    # Path to unidec configuration file (from normal unidec output, as "<filename>//<filename>_conf.dat")
    conf_file = "Data_and_configs/SLOMO_ECCR_z20_conf.dat"

    ### Other parameters
    mz_grid_choice = 1  #I dont change this. Used for scan averaging
    mass_grid = [10000, 60000, 5] #Used for script output. Must be within range of Unidec search mass and spacing

    # Unidec peak picking extraction/filtering parameters
    pick_peaks = True
    extract_picked_peak_spectrum = False #True uses intensities of filtered picked peaks for aggregated mass spectrum
    picked_peak_raw_int_THRESH = 0.03 #If < 1, uses relative scan intensity rather than absolute #. If 0, uses 0.0001
    #Filtering stuff
    filter_peaks_by_score = 0.4
    filter_peaks_by_window_mz = False
    filter_peaks_by_window_mz_thresh = [5,5] #in units of mz. Allowed [lower, upper] window bleed through
    filter_peaks_by_numscans = 2 #TODO implement this filter
    #filter_peaks_by_num_scans = 0 #TODO if >0 or 1, must be found in that many other scans
    #filter_by_num_scans_delta = 5 #Da difference for peaks to count as same
    #define window mz by scan # in cycle (requires evenly-spaced windows)
    picked_peak_scan_baseMZ = 2500 #Used to estimate m/z of isolation window and peak charge state
    picked_peak_scan_spacing = 50 #See above. Assumes windows start @ scan 0. Scan center = baseMZ + spacing/2 + (spacing * scan #)


    # Manual Extraction parameters (from mass grids, or averaged scan m/z grids)
    extract_mass_list = [30912, 30912+616] #[8564, 15740, 16480, 32230, 53240]#[8564, 15740, 16480, 32230, 53240, 64460, 157400]
    extract_method = 0  # 0=simple, 1=local max, 2=area
    extract_window =  [10, 10]

    # Set to True to extract from raw m/z spectra
    extract_from_raw = False
    # List of lists. Scan numbers within defined for each extract
    extract_mz_from_scans = [[i for i in range(140) if (i >= 70)],
                             [i for i in range(140) if (i >= 70)],
                             [i for i in range(140) if (i >= 70)],
                             [i for i in range(140) if (i >= 70)]] # Example: [[22,23,24],[23,24]] - scan numbers to extract from for each mass
    # List in list in list. Defined m/z values to sum for the scans specified above, for each extract
    extract_mzs_by_scan = [[[614.]],
                           [[616.]],
                           [[555.]],
                           [[557.]]]
    # Set to True to extract masses from specific scans
    extract_mass_from_scans = False
    extract_mass_scans =[[7], [7]]

    # Example of how to use the new extraction methods (uncomment and modify as needed):
    """
    # Example 1: Extract from raw m/z spectra
    extract_from_raw = True
    extract_mass_list = [32223, 64446]  # These act as placeholders, actual values not used for extraction
    extract_mz_from_scans = [[22,23,24], [23,24]]  # Scan numbers to extract from for each mass
    extract_mzs_by_scan = [[[2930,32223],[2930],[2930,32223]], [[4030],[4030]]]  # m/z values to extract for each scan
    
    # Example 2: Extract masses from specific scans
    extract_mass_from_scans = True
    extract_mass_list = [32223, 64446]  # Actual mass values to extract
    extract_mass_scans = [[22,23,24], [23,24]]  # Scan numbers to extract masses from for each mass
    """

    ############################################################################
    #   PROCESSING STARTS HERE
    ############################################################################

    # Get list of files to process
    file_type = "RAW"
    files_df = iterate_files_by_type(file_dir_path, file_type)
    print(f"Found {len(files_df)} files to process")

    # Create summary dataframe
    columns = ["file_name", "extracted_mass", "extract_window", "extract_method"]
    for i in range(tot_avs):
        columns.append(f"cycle_{i+1}")

    summary_df = pd.DataFrame(columns=columns)

    # Lists to store mass arrays and scan dictionaries
    all_mass_arrays = []
    all_file_names = []
    all_scan_av_dicts = []
    all_mass_av_dicts = []

    # Process each file
    for index, row in files_df.iterrows():
        fname = row["fname"]
        file_path = row["directory"]

        print(f"Processing file {index+1}/{len(files_df)}: {fname}")

        # Process the file
        extracted_values, mass_array, scan_av_dict, mass_av_dict, picked_peaks_df = process_file(
            fname, file_path, conf_file, cycle_length, scans_to_skip, n_to_av, tot_avs,
            mz_grid_choice, extract_mass_list, extract_method, extract_window,
            mass_grid, extract_from_raw, extract_mz_from_scans, extract_mzs_by_scan,
            extract_mass_from_scans, extract_mass_scans
        )

        # Store the scan_av_dict
        all_scan_av_dicts.append(scan_av_dict)
        all_mass_av_dicts.append(mass_av_dict)
        # Store the mass array and file name
        all_mass_arrays.append(mass_array)
        all_file_names.append(fname)

        # Add results to summary dataframe
        for w, mass in enumerate(extract_mass_list):
            row_data = {
                "file_name": fname,
                "extracted_mass": mass,
                "extract_window": extract_window[w],
                "extract_method": extract_method
            }

            for i in range(tot_avs):
                row_data[f"cycle_{i+1}"] = extracted_values[mass][i]

            summary_df = pd.concat([summary_df, pd.DataFrame([row_data])], ignore_index=True) #TODO: causes warning "The behavior of DataFrame concatenation with empty or all-NA entries is deprecated...

    # Save summary dataframe
    summary_file = os.path.join(file_dir_path, "batch_processing_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved summary to {summary_file}")

    # Create output directory for plots
    plots_dir = os.path.join(file_dir_path, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Create and save aggregated mass array CSV
    # First, ensure all mass arrays have the same mass grid
    mass_grid_values = all_mass_arrays[0][:, 0]  # Get mass grid from first file

    # Create a DataFrame with the mass grid as the first column
    aggregated_df = pd.DataFrame({"Mass": mass_grid_values})

    # Add intensity columns for each file
    for i, (mass_array, fname) in enumerate(zip(all_mass_arrays, all_file_names)):
        # Use the second column (index 1) which contains the intensity values
        aggregated_df[os.path.basename(fname)] = mass_array[:, 1]

    # Save to CSV
    aggregated_file = os.path.join(file_dir_path, "aggregated_mass_array.csv")
    aggregated_df.to_csv(aggregated_file, index=False)
    print(f"Saved aggregated mass array to {aggregated_file}")

    # Save to Excel
    aggregated_excel = os.path.join(file_dir_path, "aggregated_mass_array.xlsx")
    aggregated_df.to_excel(aggregated_excel, index=False)
    print(f"Saved aggregated mass array to {aggregated_excel}")

    # Generate plots
    print("Generating overlayed mass spectra plot...")
    overlayed_fig = plot_overlayed_spectra(all_mass_arrays, all_file_names, plots_dir)

    print("Generating zoomed mass spectra plots...")
    zoomed_figs = plot_zoomed_spectra(all_mass_arrays, all_file_names, extract_mass_list, extract_window, plots_dir)

    print("Generating overlayed raw MS spectra plot...")
    raw_ms_fig = plot_raw_ms_spectra(all_scan_av_dicts, all_file_names, plots_dir, extract_mass_list, extract_window, conf_file, all_mass_av_dicts)

    # Save all plots to a single PDF file
    print("Saving all plots to a single PDF file...")
    all_figures = [overlayed_fig] + zoomed_figs + raw_ms_fig
    save_figures_to_pdf(all_figures, plots_dir)

    # Code to be timed here
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
