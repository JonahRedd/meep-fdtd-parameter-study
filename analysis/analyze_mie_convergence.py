"""
Mie Scattering Analysis - Resolution Convergence
=================================================
Analyzes results from resolution sweep simulations and generates
convergence plots comparing subpixel smoothing ON vs OFF.

Author: Saïd ECH-CHADI & Youness ECHCHADI
Date: 2026-01-19
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

# Handle both Windows and WSL paths
import os
SCRIPT_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = SCRIPT_DIR.parent / "results" / "mie_scattering"
FIGURES_DIR = SCRIPT_DIR.parent / "paper" / "figures"

# Debug: print paths
print(f"Script directory: {SCRIPT_DIR}")
print(f"Results directory: {RESULTS_DIR}")
print(f"Results exists: {RESULTS_DIR.exists()}")

# Ensure figures directory exists
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATA LOADING
# =============================================================================

def load_resolution_sweep(subpixel: bool) -> List[Dict]:
    """Load all results from a resolution sweep."""
    folder = "resolution_sweep_subpixel_on" if subpixel else "resolution_sweep_subpixel_off"
    sweep_dir = RESULTS_DIR / folder
    
    results = []
    for json_file in sorted(sweep_dir.glob("*.json")):
        with open(json_file, 'r') as f:
            results.append(json.load(f))
    
    return results


def extract_convergence_data(results: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract resolution, scattered flux, and runtime from results.
    
    Returns:
        resolutions: Array of resolution values
        fluxes: Array of total scattered flux values
        runtimes: Array of runtime values
    """
    resolutions = np.array([r['config']['resolution'] for r in results])
    fluxes = np.array([r['scattered_flux']['total'] for r in results])
    runtimes = np.array([r['runtime_seconds'] for r in results])
    analytical = results[0]['analytical']['sigma_scattering']
    
    return resolutions, fluxes, runtimes, analytical


# =============================================================================
# ANALYSIS
# =============================================================================

def compute_convergence_order(resolutions: np.ndarray, errors: np.ndarray) -> float:
    """
    Compute convergence order from log-log fit.
    
    Order p is defined by: error ∝ (Δx)^p = (1/resolution)^p
    So: log(error) = -p * log(resolution) + const
    """
    # Filter out zero or negative errors
    mask = errors > 0
    if np.sum(mask) < 2:
        return np.nan
    
    log_res = np.log(resolutions[mask])
    log_err = np.log(errors[mask])
    
    # Linear fit
    coeffs = np.polyfit(log_res, log_err, 1)
    order = -coeffs[0]  # Negative because error decreases with resolution
    
    return order


# =============================================================================
# PLOTTING
# =============================================================================

def plot_resolution_convergence(save: bool = True):
    """
    Generate resolution convergence plot comparing subpixel ON vs OFF.
    """
    # Load data
    results_on = load_resolution_sweep(subpixel=True)
    results_off = load_resolution_sweep(subpixel=False)
    
    res_on, flux_on, time_on, analytical = extract_convergence_data(results_on)
    res_off, flux_off, time_off, _ = extract_convergence_data(results_off)
    
    print("="*60)
    print("RESOLUTION CONVERGENCE ANALYSIS")
    print("="*60)
    print(f"\nAnalytical σ_sca = {analytical:.6f}")
    
    print("\n--- Subpixel Smoothing ON ---")
    print(f"{'Resolution':>10} {'Flux':>12} {'Runtime (s)':>12}")
    for r, f, t in zip(res_on, flux_on, time_on):
        print(f"{r:>10} {f:>12.6f} {t:>12.2f}")
    
    print("\n--- Subpixel Smoothing OFF ---")
    print(f"{'Resolution':>10} {'Flux':>12} {'Runtime (s)':>12}")
    for r, f, t in zip(res_off, flux_off, time_off):
        print(f"{r:>10} {f:>12.6f} {t:>12.2f}")
    
    # Use highest resolution as reference (since flux-based cross-section needs normalization)
    # For now, we show the raw flux values and their convergence
    ref_on = flux_on[-1]  # Highest resolution with subpixel
    ref_off = flux_off[-1]
    
    # Compute relative differences from highest resolution
    rel_diff_on = np.abs(flux_on - ref_on) / np.abs(ref_on)
    rel_diff_off = np.abs(flux_off - ref_off) / np.abs(ref_off)
    
    # Replace zeros with small value for log plot
    rel_diff_on[rel_diff_on == 0] = 1e-10
    rel_diff_off[rel_diff_off == 0] = 1e-10
    
    # Compute convergence order (excluding the reference point)
    order_on = compute_convergence_order(res_on[:-1], rel_diff_on[:-1])
    order_off = compute_convergence_order(res_off[:-1], rel_diff_off[:-1])
    
    print(f"\nConvergence order (subpixel ON):  {order_on:.2f}")
    print(f"Convergence order (subpixel OFF): {order_off:.2f}")
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- Subplot 1: Convergence ---
    ax1 = axes[0]
    ax1.loglog(res_on[:-1], rel_diff_on[:-1], 'bo-', markersize=8, linewidth=2,
               label=f'Subpixel ON (order ≈ {order_on:.1f})')
    ax1.loglog(res_off[:-1], rel_diff_off[:-1], 'rs--', markersize=8, linewidth=2,
               label=f'Subpixel OFF (order ≈ {order_off:.1f})')
    
    # Reference lines for 1st and 2nd order convergence
    res_ref = np.array([10, 80])
    ax1.loglog(res_ref, 0.5 * (10/res_ref)**1, 'k:', alpha=0.5, label='1st order')
    ax1.loglog(res_ref, 0.5 * (10/res_ref)**2, 'k--', alpha=0.5, label='2nd order')
    
    ax1.set_xlabel('Resolution (pixels/μm)', fontsize=12)
    ax1.set_ylabel('Relative difference from high-res reference', fontsize=12)
    ax1.set_title('Resolution Convergence', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_xlim([8, 100])
    
    # --- Subplot 2: Cost-Accuracy Trade-off ---
    ax2 = axes[1]
    ax2.loglog(time_on[:-1], rel_diff_on[:-1], 'bo-', markersize=8, linewidth=2,
               label='Subpixel ON')
    ax2.loglog(time_off[:-1], rel_diff_off[:-1], 'rs--', markersize=8, linewidth=2,
               label='Subpixel OFF')
    
    ax2.set_xlabel('Runtime (seconds)', fontsize=12)
    ax2.set_ylabel('Relative difference from high-res reference', fontsize=12)
    ax2.set_title('Cost-Accuracy Trade-off', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    if save:
        output_path = FIGURES_DIR / "mie_resolution_convergence.pdf"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {output_path}")
        
        # Also save as PNG for quick viewing
        plt.savefig(FIGURES_DIR / "mie_resolution_convergence.png", dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return fig


def plot_flux_values(save: bool = True):
    """
    Plot raw flux values vs resolution to show convergence behavior.
    """
    results_on = load_resolution_sweep(subpixel=True)
    results_off = load_resolution_sweep(subpixel=False)
    
    res_on, flux_on, _, analytical = extract_convergence_data(results_on)
    res_off, flux_off, _, _ = extract_convergence_data(results_off)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.semilogx(res_on, flux_on, 'bo-', markersize=10, linewidth=2, label='Subpixel ON')
    ax.semilogx(res_off, flux_off, 'rs--', markersize=10, linewidth=2, label='Subpixel OFF')
    
    # Mark convergence
    ax.axhline(flux_on[-1], color='blue', linestyle=':', alpha=0.5)
    ax.axhline(flux_off[-1], color='red', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Resolution (pixels/μm)', fontsize=12)
    ax.set_ylabel('Total Scattered Flux (arb. units)', fontsize=12)
    ax.set_title('Scattered Flux Convergence with Resolution', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(FIGURES_DIR / "mie_flux_convergence.png", dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return fig


def generate_summary_table():
    """Generate a summary table of all results."""
    results_on = load_resolution_sweep(subpixel=True)
    results_off = load_resolution_sweep(subpixel=False)
    
    print("\n" + "="*80)
    print("SUMMARY TABLE: Resolution Sweep Results")
    print("="*80)
    print(f"\n{'Resolution':>10} | {'Subpixel ON':>15} | {'Subpixel OFF':>15} | {'Δ Runtime':>12}")
    print(f"{'(px/μm)':>10} | {'Flux':>15} | {'Flux':>15} | {'(ON - OFF)':>12}")
    print("-"*80)
    
    for r_on, r_off in zip(results_on, results_off):
        res = r_on['config']['resolution']
        flux_on = r_on['scattered_flux']['total']
        flux_off = r_off['scattered_flux']['total']
        time_on = r_on['runtime_seconds']
        time_off = r_off['runtime_seconds']
        
        print(f"{res:>10} | {flux_on:>15.6f} | {flux_off:>15.6f} | {time_on - time_off:>+12.2f}s")
    
    print("="*80)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("MIE SCATTERING ANALYSIS")
    print("="*60)
    
    # Generate summary
    generate_summary_table()
    
    # Generate plots
    print("\nGenerating convergence plots...")
    plot_resolution_convergence(save=True)
    plot_flux_values(save=True)
    
    print("\nAnalysis complete!")
