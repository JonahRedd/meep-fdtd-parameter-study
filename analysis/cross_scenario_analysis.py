"""
Cross-Scenario Analysis for FDTD Parameter Sensitivity Study
=============================================================
Generates unified convergence plots, PML effectiveness analysis,
and cost-accuracy trade-off curves across all three scenarios.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10


def load_mie_results(base_dir: Path) -> Dict:
    """Load Mie scattering results."""
    results = {
        'resolution_on': {},
        'resolution_off': {},
        'pml': {}
    }
    
    # Resolution sweep - subpixel ON
    on_dir = base_dir / 'mie_transmission' / 'subpixel_on'
    for f in on_dir.glob('*.json'):
        with open(f) as fp:
            d = json.load(fp)
        res = d['config']['resolution']
        freqs = np.array(d['frequencies'])
        trans = np.array(d['transmission'])
        idx = np.argmin(np.abs(freqs - 1.0))
        results['resolution_on'][res] = {
            'transmission': trans[idx],
            'extinction': 1 - trans[idx]
        }
    
    # Resolution sweep - subpixel OFF
    off_dir = base_dir / 'mie_transmission' / 'subpixel_off'
    for f in off_dir.glob('*.json'):
        with open(f) as fp:
            d = json.load(fp)
        res = d['config']['resolution']
        freqs = np.array(d['frequencies'])
        trans = np.array(d['transmission'])
        idx = np.argmin(np.abs(freqs - 1.0))
        results['resolution_off'][res] = {
            'transmission': trans[idx],
            'extinction': 1 - trans[idx]
        }
    
    # PML sweep
    pml_dir = base_dir / 'mie_transmission' / 'pml_sweep'
    for f in pml_dir.glob('transmission_res40_pml*.json'):
        with open(f) as fp:
            d = json.load(fp)
        pml = d['config']['pml_thickness']
        freqs = np.array(d['frequencies'])
        trans = np.array(d['transmission'])
        idx = np.argmin(np.abs(freqs - 1.0))
        results['pml'][pml] = {
            'transmission': trans[idx],
            'extinction': 1 - trans[idx]
        }
    
    return results


def load_cavity_results(base_dir: Path) -> Dict:
    """Load photonic cavity results."""
    results = {
        'resolution': {},
        'pml': {}
    }
    
    # Resolution sweep
    res_dir = base_dir / 'photonic_cavity' / 'resolution_sweep'
    for f in res_dir.glob('*.json'):
        with open(f) as fp:
            d = json.load(fp)
        res = d['config']['resolution']
        if d['dominant_mode']:
            results['resolution'][res] = {
                'frequency': d['dominant_mode']['frequency'],
                'wavelength': d['dominant_mode']['wavelength'],
                'Q_factor': d['dominant_mode']['Q_factor'],
                'runtime': d['simulation']['runtime_seconds']
            }
    
    # PML sweep
    pml_dir = base_dir / 'photonic_cavity' / 'pml_sweep'
    for f in pml_dir.glob('*.json'):
        with open(f) as fp:
            d = json.load(fp)
        pml = d['config']['pml_thickness']
        if d['dominant_mode']:
            results['pml'][pml] = {
                'frequency': d['dominant_mode']['frequency'],
                'Q_factor': d['dominant_mode']['Q_factor'],
                'runtime': d['simulation']['runtime_seconds']
            }
    
    return results


def load_waveguide_results(base_dir: Path) -> Dict:
    """Load waveguide results."""
    results = {
        'resolution': {},
        'pml': {}
    }
    
    # Resolution sweep
    res_dir = base_dir / 'waveguide' / 'resolution_sweep'
    for f in res_dir.glob('*.json'):
        with open(f) as fp:
            d = json.load(fp)
        res = d['config']['resolution']
        results['resolution'][res] = {
            'avg_transmission': d['summary']['average_transmission'],
            'peak_transmission': d['summary']['peak_transmission'],
            'runtime': d['simulation']['runtime_seconds']
        }
    
    # PML sweep
    pml_dir = base_dir / 'waveguide' / 'pml_sweep'
    for f in pml_dir.glob('*.json'):
        with open(f) as fp:
            d = json.load(fp)
        pml = d['config']['pml_thickness']
        results['pml'][pml] = {
            'avg_transmission': d['summary']['average_transmission'],
            'peak_transmission': d['summary']['peak_transmission'],
            'runtime': d['simulation']['runtime_seconds']
        }
    
    return results


def plot_resolution_convergence(mie: Dict, cavity: Dict, waveguide: Dict, output_dir: Path):
    """Generate resolution convergence comparison plot."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # === Mie Scattering ===
    ax = axes[0]
    resolutions = sorted(mie['resolution_on'].keys())
    ext_on = [mie['resolution_on'][r]['extinction'] for r in resolutions]
    ext_off = [mie['resolution_off'][r]['extinction'] for r in resolutions]
    
    ax.plot(resolutions, ext_on, 'o-', linewidth=2, markersize=8, 
            label='Subpixel ON', color='#2196F3')
    ax.plot(resolutions, ext_off, 's--', linewidth=2, markersize=8, 
            label='Subpixel OFF', color='#FF5722')
    
    # Reference line (converged value)
    converged = ext_off[-1]
    ax.axhline(y=converged, color='gray', linestyle=':', alpha=0.7)
    
    ax.set_xlabel('Resolution (px/μm)')
    ax.set_ylabel('Extinction (1 - T)')
    ax.set_title('(a) Mie Scattering')
    ax.legend(loc='lower right')
    ax.set_xscale('log', base=2)
    ax.set_xticks(resolutions)
    ax.set_xticklabels(resolutions)
    
    # === Photonic Cavity ===
    ax = axes[1]
    resolutions = sorted(cavity['resolution'].keys())
    freqs = [cavity['resolution'][r]['frequency'] for r in resolutions]
    Qs = [cavity['resolution'][r]['Q_factor'] for r in resolutions]
    
    ax.plot(resolutions, freqs, 'o-', linewidth=2, markersize=8, 
            color='#4CAF50', label='Frequency')
    ax.set_xlabel('Resolution (px/μm)')
    ax.set_ylabel('Resonance Frequency (f)', color='#4CAF50')
    ax.tick_params(axis='y', labelcolor='#4CAF50')
    
    ax2 = ax.twinx()
    ax2.plot(resolutions, Qs, 's--', linewidth=2, markersize=8, 
             color='#9C27B0', label='Q-factor')
    ax2.set_ylabel('Q-factor', color='#9C27B0')
    ax2.tick_params(axis='y', labelcolor='#9C27B0')
    
    # Annotate Q-factor non-monotonicity (peak at intermediate resolution)
    Q_max_idx = np.argmax(Qs)
    if 0 < Q_max_idx < len(Qs) - 1:  # Non-monotonic peak exists
        ax2.annotate('Q peak\n(non-monotonic)', 
                     xy=(resolutions[Q_max_idx], Qs[Q_max_idx]),
                     xytext=(resolutions[Q_max_idx] * 1.3, Qs[Q_max_idx] * 0.85),
                     fontsize=9, color='#9C27B0',
                     arrowprops=dict(arrowstyle='->', color='#9C27B0', lw=1.5))
    
    ax.set_title('(b) Photonic Cavity')
    ax.set_xscale('log', base=2)
    ax.set_xticks(resolutions)
    ax.set_xticklabels(resolutions)
    
    # === Waveguide ===
    ax = axes[2]
    resolutions = sorted(waveguide['resolution'].keys())
    trans = [waveguide['resolution'][r]['avg_transmission'] for r in resolutions]
    
    ax.plot(resolutions, trans, 'o-', linewidth=2, markersize=8, color='#FF9800')
    ax.set_xlabel('Resolution (px/μm)')
    ax.set_ylabel('Average Transmission')
    ax.set_title('(c) Waveguide')
    ax.set_xscale('log', base=2)
    ax.set_xticks(resolutions)
    ax.set_xticklabels(resolutions)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'resolution_convergence_all.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'resolution_convergence_all.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'resolution_convergence_all.png'}")
    plt.close()


def plot_pml_effectiveness(mie: Dict, cavity: Dict, waveguide: Dict, output_dir: Path):
    """Generate PML thickness effectiveness comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # === Mie Scattering ===
    ax = axes[0]
    pmls = sorted(mie['pml'].keys())
    ext = [mie['pml'][p]['extinction'] for p in pmls]
    
    bars = ax.bar(range(len(pmls)), ext, color='#2196F3', edgecolor='black', alpha=0.8)
    ax.set_xticks(range(len(pmls)))
    ax.set_xticklabels([f'{p}' for p in pmls])
    ax.set_xlabel('PML Thickness (μm)')
    ax.set_ylabel('Extinction')
    ax.set_title('(a) Mie Scattering')
    ax.set_ylim([min(ext) * 0.95, max(ext) * 1.05])
    
    for i, (p, e) in enumerate(zip(pmls, ext)):
        ax.annotate(f'{e:.4f}', (i, e + 0.001), ha='center', fontsize=9)
    
    # === Photonic Cavity ===
    ax = axes[1]
    pmls = sorted(cavity['pml'].keys())
    Qs = [cavity['pml'][p]['Q_factor'] for p in pmls]
    
    bars = ax.bar(range(len(pmls)), Qs, color='#4CAF50', edgecolor='black', alpha=0.8)
    ax.set_xticks(range(len(pmls)))
    ax.set_xticklabels([f'{p}' for p in pmls])
    ax.set_xlabel('PML Thickness (μm)')
    ax.set_ylabel('Q-factor')
    ax.set_title('(b) Photonic Cavity')
    ax.set_ylim([min(Qs) * 0.95, max(Qs) * 1.05])
    
    for i, (p, q) in enumerate(zip(pmls, Qs)):
        ax.annotate(f'{q:.1f}', (i, q + 1), ha='center', fontsize=9)
    
    # === Waveguide ===
    ax = axes[2]
    pmls = sorted(waveguide['pml'].keys())
    trans = [waveguide['pml'][p]['avg_transmission'] for p in pmls]
    
    bars = ax.bar(range(len(pmls)), trans, color='#FF9800', edgecolor='black', alpha=0.8)
    ax.set_xticks(range(len(pmls)))
    ax.set_xticklabels([f'{p}' for p in pmls])
    ax.set_xlabel('PML Thickness (μm)')
    ax.set_ylabel('Average Transmission')
    ax.set_title('(c) Waveguide')
    ax.set_ylim([min(trans) * 0.95, max(trans) * 1.05])
    
    for i, (p, t) in enumerate(zip(pmls, trans)):
        ax.annotate(f'{t:.4f}', (i, t + 0.005), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pml_effectiveness_all.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'pml_effectiveness_all.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'pml_effectiveness_all.png'}")
    plt.close()


def plot_convergence_error(mie: Dict, cavity: Dict, waveguide: Dict, output_dir: Path):
    """Generate log-log convergence error plot."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # === Mie Scattering ===
    ax = axes[0]
    resolutions = sorted(mie['resolution_off'].keys())
    ext = np.array([mie['resolution_off'][r]['extinction'] for r in resolutions])
    converged = ext[-1]
    error = np.abs(ext - converged)
    error[error == 0] = 1e-10  # Avoid log(0)
    
    ax.loglog(resolutions, error, 'o-', linewidth=2, markersize=8, 
              color='#FF5722', label='Measured error')
    
    # Reference lines
    res_arr = np.array(resolutions)
    ax.loglog(res_arr, 0.3/res_arr, ':', color='gray', alpha=0.7, label='O(1/N)')
    ax.loglog(res_arr, 3.0/res_arr**2, '--', color='gray', alpha=0.7, label='O(1/N²)')
    
    ax.set_xlabel('Resolution (px/μm)')
    ax.set_ylabel('|Error| from converged')
    ax.set_title('(a) Mie Scattering')
    ax.legend(loc='upper right')
    ax.grid(True, which='both', alpha=0.3)
    
    # === Photonic Cavity ===
    ax = axes[1]
    resolutions = sorted(cavity['resolution'].keys())
    freqs = np.array([cavity['resolution'][r]['frequency'] for r in resolutions])
    converged = freqs[-1]
    error = np.abs(freqs - converged)
    error[error == 0] = 1e-10
    
    ax.loglog(resolutions, error, 'o-', linewidth=2, markersize=8, 
              color='#4CAF50', label='Frequency error')
    
    res_arr = np.array(resolutions)
    ax.loglog(res_arr, 0.5/res_arr, ':', color='gray', alpha=0.7, label='O(1/N)')
    ax.loglog(res_arr, 5.0/res_arr**2, '--', color='gray', alpha=0.7, label='O(1/N²)')
    
    ax.set_xlabel('Resolution (px/μm)')
    ax.set_ylabel('|Frequency error|')
    ax.set_title('(b) Photonic Cavity')
    ax.legend(loc='upper right')
    ax.grid(True, which='both', alpha=0.3)
    
    # === Waveguide ===
    ax = axes[2]
    resolutions = sorted(waveguide['resolution'].keys())
    trans = np.array([waveguide['resolution'][r]['avg_transmission'] for r in resolutions])
    converged = trans[-1]
    error = np.abs(trans - converged)
    error[error == 0] = 1e-10
    
    ax.loglog(resolutions, error, 'o-', linewidth=2, markersize=8, 
              color='#FF9800', label='Transmission error')
    
    res_arr = np.array(resolutions)
    ax.loglog(res_arr, 0.5/res_arr, ':', color='gray', alpha=0.7, label='O(1/N)')
    ax.loglog(res_arr, 5.0/res_arr**2, '--', color='gray', alpha=0.7, label='O(1/N²)')
    
    ax.set_xlabel('Resolution (px/μm)')
    ax.set_ylabel('|Transmission error|')
    ax.set_title('(c) Waveguide')
    ax.legend(loc='upper right')
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_error_loglog.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'convergence_error_loglog.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'convergence_error_loglog.png'}")
    plt.close()


def plot_cost_accuracy_tradeoff(cavity: Dict, waveguide: Dict, output_dir: Path):
    """Generate cost (runtime) vs accuracy trade-off plot."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    
    # === Photonic Cavity ===
    ax = axes[0]
    resolutions = sorted(cavity['resolution'].keys())
    freqs = np.array([cavity['resolution'][r]['frequency'] for r in resolutions])
    runtimes = np.array([cavity['resolution'][r]['runtime'] for r in resolutions])
    
    converged = freqs[-1]
    error = np.abs(freqs - converged)
    error[error == 0] = 1e-10
    
    ax.loglog(runtimes, error, 'o-', linewidth=2, markersize=10, color='#4CAF50')
    
    for i, (r, rt, err) in enumerate(zip(resolutions, runtimes, error)):
        ax.annotate(f'{r} px/μm', (rt * 1.1, err), fontsize=9)
    
    ax.set_xlabel('Runtime (seconds)')
    ax.set_ylabel('Frequency Error')
    ax.set_title('(a) Photonic Cavity: Cost vs Accuracy')
    ax.grid(True, which='both', alpha=0.3)
    
    # === Waveguide ===
    ax = axes[1]
    resolutions = sorted(waveguide['resolution'].keys())
    trans = np.array([waveguide['resolution'][r]['avg_transmission'] for r in resolutions])
    runtimes = np.array([waveguide['resolution'][r]['runtime'] for r in resolutions])
    
    converged = trans[-1]
    error = np.abs(trans - converged)
    error[error == 0] = 1e-10
    
    ax.loglog(runtimes, error, 'o-', linewidth=2, markersize=10, color='#FF9800')
    
    # Annotate runtime anomaly at lowest resolution (highest runtime with high error)
    min_res_idx = 0  # Lowest resolution is first in sorted list
    if runtimes[min_res_idx] > runtimes[min_res_idx + 1]:  # Runtime anomaly exists
        ax.annotate('Runtime spike\\n(dispersion)', 
                    xy=(runtimes[min_res_idx], error[min_res_idx]),
                    xytext=(runtimes[min_res_idx] * 0.3, error[min_res_idx] * 0.5),
                    fontsize=9, color='#E65100',
                    arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.5))
    
    for i, (r, rt, err) in enumerate(zip(resolutions, runtimes, error)):
        ax.annotate(f'{r} px/μm', (rt * 1.1, err), fontsize=9)
    
    ax.set_xlabel('Runtime (seconds)')
    ax.set_ylabel('Transmission Error')
    ax.set_title('(b) Waveguide: Cost vs Accuracy')
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cost_accuracy_tradeoff.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'cost_accuracy_tradeoff.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'cost_accuracy_tradeoff.png'}")
    plt.close()


def generate_summary_table(mie: Dict, cavity: Dict, waveguide: Dict, output_dir: Path):
    """Generate LaTeX-ready summary tables."""
    
    # Resolution convergence table
    table_res = """
\\begin{table}[htbp]
\\centering
\\caption{Resolution Convergence Summary Across All Scenarios}
\\label{tab:resolution_convergence}
\\begin{tabular}{lccc}
\\toprule
Resolution & Mie Extinction & Cavity Q-factor & Waveguide Trans. \\\\
(px/$\\mu$m) & (1-T) & & \\\\
\\midrule
"""
    
    resolutions = [20, 40, 80]
    for res in resolutions:
        mie_ext = mie['resolution_off'].get(res, {}).get('extinction', 'N/A')
        cav_q = cavity['resolution'].get(res, {}).get('Q_factor', 'N/A')
        wg_t = waveguide['resolution'].get(res, {}).get('avg_transmission', 'N/A')
        
        mie_str = f"{mie_ext:.4f}" if isinstance(mie_ext, float) else mie_ext
        cav_str = f"{cav_q:.1f}" if isinstance(cav_q, float) else cav_q
        wg_str = f"{wg_t:.4f}" if isinstance(wg_t, float) else wg_t
        
        table_res += f"{res} & {mie_str} & {cav_str} & {wg_str} \\\\\n"
    
    table_res += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    # PML effectiveness table
    table_pml = """
\\begin{table}[htbp]
\\centering
\\caption{PML Thickness Sensitivity Analysis}
\\label{tab:pml_sensitivity}
\\begin{tabular}{lccc}
\\toprule
PML Thickness & Mie Extinction & Cavity Q-factor & Waveguide Trans. \\\\
($\\mu$m) & & & \\\\
\\midrule
"""
    
    pmls = [0.5, 1.0, 2.0]
    for pml in pmls:
        mie_ext = mie['pml'].get(pml, {}).get('extinction', 'N/A')
        cav_q = cavity['pml'].get(pml, {}).get('Q_factor', 'N/A')
        wg_t = waveguide['pml'].get(pml, {}).get('avg_transmission', 'N/A')
        
        mie_str = f"{mie_ext:.4f}" if isinstance(mie_ext, float) else mie_ext
        cav_str = f"{cav_q:.1f}" if isinstance(cav_q, float) else cav_q
        wg_str = f"{wg_t:.4f}" if isinstance(wg_t, float) else wg_t
        
        table_pml += f"{pml} & {mie_str} & {cav_str} & {wg_str} \\\\\n"
    
    table_pml += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    # Save tables
    with open(output_dir / 'tables_latex.tex', 'w') as f:
        f.write("% Auto-generated LaTeX tables for FDTD parameter study\n\n")
        f.write(table_res)
        f.write("\n")
        f.write(table_pml)
    
    print(f"Saved: {output_dir / 'tables_latex.tex'}")


def main():
    """Run complete cross-scenario analysis."""
    print("=" * 70)
    print("CROSS-SCENARIO ANALYSIS - FDTD PARAMETER SENSITIVITY STUDY")
    print("=" * 70)
    
    # Set paths
    base_dir = Path(__file__).parent.parent / 'results'
    output_dir = base_dir / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all results
    print("\n[1/6] Loading results...")
    mie = load_mie_results(base_dir)
    cavity = load_cavity_results(base_dir)
    waveguide = load_waveguide_results(base_dir)
    
    print(f"  Mie: {len(mie['resolution_on'])} resolution points, {len(mie['pml'])} PML points")
    print(f"  Cavity: {len(cavity['resolution'])} resolution points, {len(cavity['pml'])} PML points")
    print(f"  Waveguide: {len(waveguide['resolution'])} resolution points, {len(waveguide['pml'])} PML points")
    
    # Generate plots
    print("\n[2/6] Generating resolution convergence plot...")
    plot_resolution_convergence(mie, cavity, waveguide, output_dir)
    
    print("\n[3/6] Generating PML effectiveness plot...")
    plot_pml_effectiveness(mie, cavity, waveguide, output_dir)
    
    print("\n[4/6] Generating convergence error (log-log) plot...")
    plot_convergence_error(mie, cavity, waveguide, output_dir)
    
    print("\n[5/6] Generating cost-accuracy trade-off plot...")
    plot_cost_accuracy_tradeoff(cavity, waveguide, output_dir)
    
    print("\n[6/6] Generating LaTeX summary tables...")
    generate_summary_table(mie, cavity, waveguide, output_dir)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob('*')):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
