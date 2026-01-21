"""
Mie Scattering Simulation - Improved Version
=============================================
Uses scattered-field technique for accurate cross-section calculation.

The scattering cross-section is computed by:
1. First simulation: empty domain (no cylinder) to get incident flux
2. Second simulation: with cylinder, measuring total flux
3. Scattered power = difference in flux through monitors

Author: [Your Name]
Date: 2026-01-19
"""

import meep as mp
import numpy as np
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Optional

# Import analytical Mie theory
import sys
sys.path.insert(0, str(Path(__file__).parent))
from mie_theory import MieCylinder2D


@dataclass
class SimConfig:
    """Simulation configuration parameters."""
    resolution: int = 20           # pixels per μm
    wavelength: float = 1.0        # μm
    cylinder_radius: float = 0.5   # μm
    cylinder_n: float = 2.0        # refractive index
    pml_thickness: float = 1.0     # μm
    padding: float = 2.0           # space between cylinder and monitors
    source_offset: float = 1.5     # source distance from cylinder
    simulation_time: float = 100   # Meep time units
    use_subpixel: bool = True
    
    @property
    def frequency(self) -> float:
        return 1.0 / self.wavelength
    
    @property
    def cell_size(self) -> float:
        return 2 * (self.cylinder_radius + self.padding + self.pml_thickness + 1)


def run_scattering_simulation(config: SimConfig, with_cylinder: bool = True) -> Dict:
    """
    Run a scattering simulation.
    
    Parameters
    ----------
    config : SimConfig
        Simulation configuration
    with_cylinder : bool
        If True, include the cylinder. If False, run empty reference.
    
    Returns
    -------
    results : dict
        Flux data and runtime
    """
    cell_size = config.cell_size
    cell = mp.Vector3(cell_size, cell_size, 0)
    pml = [mp.PML(config.pml_thickness)]
    
    # Geometry
    geometry = []
    if with_cylinder:
        geometry = [
            mp.Cylinder(
                radius=config.cylinder_radius,
                height=mp.inf,
                axis=mp.Vector3(0, 0, 1),
                center=mp.Vector3(0, 0, 0),
                material=mp.Medium(epsilon=config.cylinder_n**2)
            )
        ]
    
    # Plane wave source (line source spanning the domain)
    src_x = -cell_size/2 + config.pml_thickness + 0.5
    sources = [
        mp.Source(
            src=mp.GaussianSource(config.frequency, fwidth=0.4*config.frequency),
            component=mp.Ez,
            center=mp.Vector3(src_x, 0),
            size=mp.Vector3(0, cell_size - 2*config.pml_thickness)
        )
    ]
    
    # Create simulation
    sim = mp.Simulation(
        cell_size=cell,
        resolution=config.resolution,
        geometry=geometry,
        sources=sources,
        boundary_layers=pml,
        eps_averaging=config.use_subpixel,
        force_complex_fields=False,
    )
    
    # Transmission monitor (downstream of cylinder)
    mon_x = config.cylinder_radius + config.padding
    nfreq = 50  # number of frequency points
    
    trans_fr = mp.FluxRegion(
        center=mp.Vector3(mon_x, 0),
        size=mp.Vector3(0, cell_size - 2*config.pml_thickness - 1)
    )
    trans = sim.add_flux(config.frequency, 0.4*config.frequency, nfreq, trans_fr)
    
    # Run simulation
    start_time = time.time()
    sim.run(until_after_sources=mp.stop_when_fields_decayed(20, mp.Ez, mp.Vector3(mon_x, 0), 1e-6))
    runtime = time.time() - start_time
    
    # Get flux data
    flux_freqs = mp.get_flux_freqs(trans)
    flux_data = mp.get_fluxes(trans)
    
    results = {
        'frequencies': flux_freqs,
        'flux': flux_data,
        'runtime': runtime,
        'with_cylinder': with_cylinder,
    }
    
    # Save flux data for normalization
    if not with_cylinder:
        results['flux_data_for_norm'] = sim.get_flux_data(trans)
    
    sim.reset_meep()
    
    return results


def compute_transmission_spectrum(config: SimConfig, output_dir: Path = None) -> Dict:
    """
    Compute transmission spectrum by comparing with/without cylinder.
    
    Returns normalized transmission T(ω) = P_trans / P_incident
    """
    print("="*60)
    print("MIE SCATTERING - TRANSMISSION SPECTRUM")
    print("="*60)
    print(f"Resolution: {config.resolution} px/μm")
    print(f"Wavelength: {config.wavelength} μm")
    print(f"Cylinder: r={config.cylinder_radius} μm, n={config.cylinder_n}")
    print(f"Subpixel: {config.use_subpixel}")
    print("="*60)
    
    # Run reference (no cylinder)
    print("\n[1/2] Running reference simulation (no cylinder)...")
    ref_results = run_scattering_simulation(config, with_cylinder=False)
    print(f"      Runtime: {ref_results['runtime']:.2f} s")
    
    # Run with cylinder
    print("\n[2/2] Running scattering simulation (with cylinder)...")
    scat_results = run_scattering_simulation(config, with_cylinder=True)
    print(f"      Runtime: {scat_results['runtime']:.2f} s")
    
    # Compute transmission
    freqs = np.array(ref_results['frequencies'])
    wavelengths = 1.0 / freqs
    
    incident_flux = np.array(ref_results['flux'])
    transmitted_flux = np.array(scat_results['flux'])
    
    # Transmission = transmitted / incident
    # Avoid division by zero
    transmission = np.zeros_like(transmitted_flux)
    mask = incident_flux > 0
    transmission[mask] = transmitted_flux[mask] / incident_flux[mask]
    
    # Extinction = 1 - T (for non-absorbing materials, this equals scattering)
    extinction = 1 - transmission
    
    # Get analytical reference at center frequency
    mie = MieCylinder2D(config.cylinder_radius, config.cylinder_n, n_background=1.0)
    analytical_Q_ext = mie.extinction_efficiency(config.wavelength)
    analytical_sigma_ext = mie.extinction_cross_section(config.wavelength)
    
    # Compile results
    results = {
        'config': asdict(config),
        'frequencies': freqs.tolist(),
        'wavelengths': wavelengths.tolist(),
        'incident_flux': incident_flux.tolist(),
        'transmitted_flux': transmitted_flux.tolist(),
        'transmission': transmission.tolist(),
        'extinction': extinction.tolist(),
        'runtime_reference': ref_results['runtime'],
        'runtime_scattering': scat_results['runtime'],
        'runtime_total': ref_results['runtime'] + scat_results['runtime'],
        'analytical': {
            'Q_ext': analytical_Q_ext,
            'sigma_ext': analytical_sigma_ext,
        }
    }
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        subpixel_str = "on" if config.use_subpixel else "off"
        filename = f"transmission_res{config.resolution}_pml{config.pml_thickness}_subpixel{subpixel_str}.json"
        
        with open(output_dir / filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_dir / filename}")
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    # Find transmission at center wavelength
    center_idx = np.argmin(np.abs(wavelengths - config.wavelength))
    T_center = transmission[center_idx]
    ext_center = extinction[center_idx]
    
    print(f"At λ = {config.wavelength} μm:")
    print(f"  Transmission T = {T_center:.4f}")
    print(f"  Extinction (1-T) = {ext_center:.4f}")
    print(f"  Analytical Q_ext = {analytical_Q_ext:.4f}")
    print(f"Total runtime: {results['runtime_total']:.2f} s")
    
    return results


def resolution_sweep(
    resolutions: list = [10, 20, 40, 80],
    subpixel: bool = True,
    output_dir: Path = None
) -> list:
    """Run transmission simulations at multiple resolutions."""
    
    all_results = []
    
    for res in resolutions:
        print("\n" + "#"*60)
        print(f"# RESOLUTION: {res} px/μm (subpixel={subpixel})")
        print("#"*60)
        
        config = SimConfig(
            resolution=res,
            use_subpixel=subpixel,
        )
        
        results = compute_transmission_spectrum(config, output_dir)
        all_results.append(results)
    
    return all_results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Mie Scattering Transmission Simulation")
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'sweep'],
                        help='Simulation mode')
    parser.add_argument('--resolution', type=int, default=20)
    parser.add_argument('--subpixel', action='store_true', default=True)
    parser.add_argument('--no-subpixel', dest='subpixel', action='store_false')
    parser.add_argument('--output', type=str, default='../../results/mie_transmission')
    
    args = parser.parse_args()
    
    output_dir = Path(__file__).parent / args.output
    
    if args.mode == 'single':
        config = SimConfig(
            resolution=args.resolution,
            use_subpixel=args.subpixel,
        )
        results = compute_transmission_spectrum(config, output_dir)
        
    elif args.mode == 'sweep':
        # Run sweep with subpixel ON
        print("\n" + "="*60)
        print("RESOLUTION SWEEP - SUBPIXEL ON")
        print("="*60)
        resolution_sweep(
            resolutions=[10, 20, 40, 80],
            subpixel=True,
            output_dir=output_dir / "subpixel_on"
        )
        
        # Run sweep with subpixel OFF
        print("\n" + "="*60)
        print("RESOLUTION SWEEP - SUBPIXEL OFF")
        print("="*60)
        resolution_sweep(
            resolutions=[10, 20, 40, 80],
            subpixel=False,
            output_dir=output_dir / "subpixel_off"
        )
        
        print("\n" + "="*60)
        print("ALL SWEEPS COMPLETE")
        print("="*60)
