"""
Mie Transmission Analysis - Convergence Study
==============================================
Analyzes transmission-based scattering results.

Author: [Your Name]
Date: 2026-01-19
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = SCRIPT_DIR.parent / "results" / "mie_transmission"
FIGURES_DIR = SCRIPT_DIR.parent / "paper" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Plotting style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'figure.figsize': (10, 6),
})


# =============================================================================
# DATA LOADING
# =============================================================================

def load_transmission_results(subdir: str = None) -> List[Dict]:
    """Load all transmission results from a directory."""
    if subdir:
        search_dir = RESULTS_DIR / subdir
    else:
        search_dir = RESULTS_DIR
    
    results = []
    for json_file in sorted(search_dir.glob("*.json")):
        with open(json_file, 'r') as f:
            data = json.load(f)
            data['filename'] = json_file.name
            results.append(data)
    
    # Sort by resolution
    results.sort(key=lambda x: x['config']['resolution'])
    return results


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_convergence(results: List[Dict]) -> Dict:
    """Analyze convergence behavior from transmission results."""
    
    resolutions = []
    extinctions = []  # 1 - T at center wavelength
    runtimes = []
    
    for r in results:
        res = r['config']['resolution']
        wavelengths = np.array(r['wavelengths'])
        transmission = np.array(r['transmission'])
        
        # Find value at center wavelength (λ = 1.0 μm)
        center_idx = np.argmin(np.abs(wavelengths - 1.0))
        T_center = transmission[center_idx]
        
        resolutions.append(res)
        extinctions.append(1 - T_center)
        runtimes.append(r['runtime_total'])
    
    return {
        'resolutions': np.array(resolutions),
        'extinctions': np.array(extinctions),
        'runtimes': np.array(runtimes),
    }


def compute_convergence_order(resolutions: np.ndarray, values: np.ndarray) -> float:
    """Compute convergence order from log-log fit."""
    # Use difference from highest resolution as error proxy
    ref_value = values[-1]
    errors = np.abs(values - ref_value)
    
    # Filter valid points (exclude reference and zeros)
    mask = (errors > 1e-10) & (np.arange(len(errors)) < len(errors) - 1)
    
    if np.sum(mask) < 2:
        return np.nan
    
    log_res = np.log(resolutions[mask])
    log_err = np.log(errors[mask])
    
    coeffs = np.polyfit(log_res, log_err, 1)
    return -coeffs[0]


# =============================================================================
# PLOTTING
# =============================================================================

def plot_transmission_spectra(results: List[Dict], save: bool = True):
    """Plot transmission spectra at different resolutions."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results)))
    
    for r, color in zip(results, colors):
        res = r['config']['resolution']
        wavelengths = np.array(r['wavelengths'])
        transmission = np.array(r['transmission'])
        
        # Sort by wavelength for clean plot
        sort_idx = np.argsort(wavelengths)
        
        ax.plot(wavelengths[sort_idx], transmission[sort_idx], 
                '-', linewidth=2, color=color, label=f'{res} px/μm')
    
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel('Transmission')
    ax.set_title('Mie Scattering: Transmission Spectrum Convergence')
    ax.legend(title='Resolution')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.5, 2.0])
    ax.set_ylim([0.5, 1.05])
    
    plt.tight_layout()
    
    if save:
        plt.savefig(FIGURES_DIR / 'mie_transmission_spectra.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(FIGURES_DIR / 'mie_transmission_spectra.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {FIGURES_DIR / 'mie_transmission_spectra.png'}")
    
    plt.show()
    return fig


def plot_extinction_convergence(results: List[Dict], save: bool = True):
    """Plot extinction (1-T) convergence with resolution."""
    
    data = analyze_convergence(results)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- Left: Extinction vs Resolution ---
    ax1 = axes[0]
    ax1.semilogx(data['resolutions'], data['extinctions'], 'bo-', 
                  markersize=10, linewidth=2, label='FDTD (Meep)')
    
    # Mark convergence trend
    ax1.axhline(data['extinctions'][-1], color='gray', linestyle='--', alpha=0.7,
                label=f'High-res value: {data["extinctions"][-1]:.4f}')
    
    ax1.set_xlabel('Resolution (pixels/μm)')
    ax1.set_ylabel('Extinction (1 - T) at λ = 1 μm')
    ax1.set_title('Extinction Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # --- Right: Runtime vs Resolution ---
    ax2 = axes[1]
    ax2.loglog(data['resolutions'], data['runtimes'], 'rs-', 
               markersize=10, linewidth=2)
    
    # Fit power law for scaling
    log_res = np.log(data['resolutions'])
    log_time = np.log(data['runtimes'])
    coeffs = np.polyfit(log_res, log_time, 1)
    scaling = coeffs[0]
    
    # Plot fit
    res_fit = np.linspace(data['resolutions'].min(), data['resolutions'].max() * 2, 100)
    time_fit = np.exp(coeffs[1]) * res_fit**scaling
    ax2.loglog(res_fit, time_fit, 'k--', alpha=0.5, 
               label=f'Scaling: O(N^{{{scaling:.1f}}})')
    
    ax2.set_xlabel('Resolution (pixels/μm)')
    ax2.set_ylabel('Runtime (seconds)')
    ax2.set_title('Computational Cost Scaling')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    if save:
        plt.savefig(FIGURES_DIR / 'mie_extinction_convergence.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(FIGURES_DIR / 'mie_extinction_convergence.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {FIGURES_DIR / 'mie_extinction_convergence.png'}")
    
    plt.show()
    return fig


def generate_summary_table(results: List[Dict]):
    """Print a summary table of results."""
    
    print("\n" + "="*70)
    print("MIE SCATTERING - TRANSMISSION CONVERGENCE SUMMARY")
    print("="*70)
    print(f"\n{'Resolution':>12} {'Extinction':>12} {'ΔExtinction':>12} {'Runtime':>10}")
    print(f"{'(px/μm)':>12} {'(1-T)':>12} {'(vs prev)':>12} {'(s)':>10}")
    print("-"*70)
    
    prev_ext = None
    for r in results:
        res = r['config']['resolution']
        wavelengths = np.array(r['wavelengths'])
        transmission = np.array(r['transmission'])
        
        center_idx = np.argmin(np.abs(wavelengths - 1.0))
        extinction = 1 - transmission[center_idx]
        runtime = r['runtime_total']
        
        delta = f"{extinction - prev_ext:+.6f}" if prev_ext is not None else "---"
        print(f"{res:>12} {extinction:>12.6f} {delta:>12} {runtime:>10.2f}")
        
        prev_ext = extinction
    
    print("="*70)
    print(f"\nAnalytical Q_ext = {results[0]['analytical']['Q_ext']:.4f}")
    print("Note: Extinction here is normalized transmission loss, not cross-section")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("MIE TRANSMISSION ANALYSIS")
    print("="*60)
    
    # Try to load subpixel_on results first
    try:
        results = load_transmission_results("subpixel_on")
        if not results:
            print("No results in subpixel_on/, trying root directory...")
            results = load_transmission_results()
    except:
        results = load_transmission_results()
    
    if not results:
        print("ERROR: No transmission results found!")
        print(f"Expected location: {RESULTS_DIR}")
        exit(1)
    
    print(f"\nLoaded {len(results)} result files")
    
    # Generate summary
    generate_summary_table(results)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_transmission_spectra(results, save=True)
    plot_extinction_convergence(results, save=True)
    
    print("\n✓ Analysis complete!")
