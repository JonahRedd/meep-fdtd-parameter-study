"""
Mie Scattering Simulation - 2D Dielectric Cylinder
===================================================
FDTD Parameter Sensitivity Analysis - Scenario 1

This script simulates electromagnetic scattering from a 2D dielectric cylinder
using Meep FDTD. Results are compared against analytical Mie theory to quantify
numerical error as a function of simulation parameters.

Parameters under investigation:
- Spatial resolution (pixels per wavelength)
- Subpixel smoothing (on/off)
- PML thickness
- Simulation time

Author: [Your Name]
Date: 2026-01-19
"""

import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

# =============================================================================
# SIMULATION PARAMETERS (Configurable)
# =============================================================================

class SimulationConfig:
    """Configuration class for Mie scattering simulation parameters."""
    
    def __init__(
        self,
        resolution: int = 20,           # pixels per unit length (μm)
        wavelength: float = 1.0,        # wavelength in μm
        cylinder_radius: float = 0.5,   # cylinder radius in μm
        cylinder_n: float = 2.0,        # cylinder refractive index
        background_n: float = 1.0,      # background refractive index
        pml_thickness: float = 1.0,     # PML thickness in μm
        padding: float = 1.0,           # space between cylinder and PML
        simulation_time: float = 50,    # simulation time in Meep units
        use_subpixel: bool = True,      # enable subpixel smoothing
        courant: float = 0.5,           # Courant factor
    ):
        self.resolution = resolution
        self.wavelength = wavelength
        self.cylinder_radius = cylinder_radius
        self.cylinder_n = cylinder_n
        self.background_n = background_n
        self.pml_thickness = pml_thickness
        self.padding = padding
        self.simulation_time = simulation_time
        self.use_subpixel = use_subpixel
        self.courant = courant
        
        # Derived quantities
        self.frequency = 1.0 / wavelength
        self.cylinder_eps = cylinder_n ** 2
        self.cell_size = 2 * (cylinder_radius + padding + pml_thickness)
        self.pixels_per_wavelength = resolution * wavelength
        
    def to_dict(self):
        """Export configuration as dictionary for logging."""
        return {
            'resolution': self.resolution,
            'wavelength': self.wavelength,
            'cylinder_radius': self.cylinder_radius,
            'cylinder_n': self.cylinder_n,
            'pml_thickness': self.pml_thickness,
            'padding': self.padding,
            'simulation_time': self.simulation_time,
            'use_subpixel': self.use_subpixel,
            'courant': self.courant,
            'pixels_per_wavelength': self.pixels_per_wavelength,
            'cell_size': self.cell_size,
        }


# =============================================================================
# MIE THEORY REFERENCE (Analytical Solution)
# =============================================================================

def mie_coefficients_2d(n_cylinder, n_background, radius, wavelength, n_terms=20):
    """
    Calculate 2D Mie scattering coefficients for a dielectric cylinder.
    
    For TM polarization (E parallel to cylinder axis), the scattering
    coefficients are given by:
    
    a_n = (m*J_n(mx)*J_n'(x) - J_n(x)*J_n'(mx)) / 
          (m*J_n(mx)*H_n'(x) - H_n(x)*J_n'(mx))
    
    where m = n_cylinder/n_background, x = k*radius, k = 2π/λ
    
    Parameters
    ----------
    n_cylinder : float
        Refractive index of the cylinder
    n_background : float
        Refractive index of the background medium
    radius : float
        Cylinder radius
    wavelength : float
        Wavelength in the background medium
    n_terms : int
        Number of terms in the series expansion
        
    Returns
    -------
    a_n : array
        Scattering coefficients for n = 0, 1, 2, ..., n_terms-1
    """
    from scipy.special import jv, jvp, hankel1, h1vp
    
    m = n_cylinder / n_background
    k = 2 * np.pi / wavelength
    x = k * radius
    mx = m * x
    
    a_n = np.zeros(n_terms, dtype=complex)
    
    for n in range(n_terms):
        # Bessel functions and derivatives
        Jn_x = jv(n, x)
        Jn_mx = jv(n, mx)
        Jnp_x = jvp(n, x)
        Jnp_mx = jvp(n, mx)
        Hn_x = hankel1(n, x)
        Hnp_x = h1vp(n, x)
        
        # TM polarization scattering coefficient
        numerator = m * Jn_mx * Jnp_x - Jn_x * Jnp_mx
        denominator = m * Jn_mx * Hnp_x - Hn_x * Jnp_mx
        
        a_n[n] = numerator / denominator
    
    return a_n


def scattering_cross_section_2d(n_cylinder, n_background, radius, wavelength, n_terms=20):
    """
    Calculate the 2D scattering cross-section (scattering width) analytically.
    
    σ_sca = (4/k) * Σ |a_n|² * ε_n
    
    where ε_0 = 1, ε_n = 2 for n > 0 (Neumann factor)
    
    Returns
    -------
    sigma_sca : float
        Scattering cross-section (width) in units of length
    """
    a_n = mie_coefficients_2d(n_cylinder, n_background, radius, wavelength, n_terms)
    k = 2 * np.pi / wavelength
    
    # Neumann factor: ε_0 = 1, ε_n = 2 for n > 0
    epsilon_n = np.ones(n_terms)
    epsilon_n[1:] = 2
    
    sigma_sca = (4.0 / k) * np.sum(epsilon_n * np.abs(a_n) ** 2)
    
    return sigma_sca


def extinction_cross_section_2d(n_cylinder, n_background, radius, wavelength, n_terms=20):
    """
    Calculate the 2D extinction cross-section analytically.
    
    σ_ext = (4/k) * Re[Σ a_n * ε_n]
    
    Returns
    -------
    sigma_ext : float
        Extinction cross-section (width) in units of length
    """
    a_n = mie_coefficients_2d(n_cylinder, n_background, radius, wavelength, n_terms)
    k = 2 * np.pi / wavelength
    
    epsilon_n = np.ones(n_terms)
    epsilon_n[1:] = 2
    
    sigma_ext = (4.0 / k) * np.real(np.sum(epsilon_n * a_n))
    
    return sigma_ext


# =============================================================================
# MEEP SIMULATION
# =============================================================================

def run_mie_simulation(config: SimulationConfig, output_dir: Path = None):
    """
    Run a 2D Mie scattering simulation in Meep.
    
    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration parameters
    output_dir : Path, optional
        Directory to save output files
        
    Returns
    -------
    results : dict
        Dictionary containing simulation results and metrics
    """
    
    start_time = time.time()
    
    # Cell geometry
    cell_size = mp.Vector3(config.cell_size, config.cell_size, 0)
    
    # PML layers
    pml_layers = [mp.PML(thickness=config.pml_thickness)]
    
    # Dielectric cylinder at center
    geometry = [
        mp.Cylinder(
            radius=config.cylinder_radius,
            height=mp.inf,
            axis=mp.Vector3(0, 0, 1),
            center=mp.Vector3(0, 0, 0),
            material=mp.Medium(epsilon=config.cylinder_eps)
        )
    ]
    
    # Gaussian source (plane wave approximation via line source)
    # Source positioned to the left of the cylinder
    source_x = -config.cell_size / 2 + config.pml_thickness + 0.5
    sources = [
        mp.Source(
            src=mp.GaussianSource(frequency=config.frequency, fwidth=0.2 * config.frequency),
            component=mp.Ez,
            center=mp.Vector3(source_x, 0, 0),
            size=mp.Vector3(0, config.cell_size - 2 * config.pml_thickness, 0)
        )
    ]
    
    # Create simulation
    sim = mp.Simulation(
        cell_size=cell_size,
        resolution=config.resolution,
        geometry=geometry,
        sources=sources,
        boundary_layers=pml_layers,
        eps_averaging=config.use_subpixel,
        Courant=config.courant,
    )
    
    # Flux monitors for scattering cross-section
    # Near-field box around the cylinder
    nf_size = 2 * config.cylinder_radius + 0.5
    nfreq = 1  # single frequency for now
    
    # Four sides of the near-field box
    flux_box_top = sim.add_flux(
        config.frequency, 0, nfreq,
        mp.FluxRegion(center=mp.Vector3(0, nf_size/2, 0), size=mp.Vector3(nf_size, 0, 0))
    )
    flux_box_bottom = sim.add_flux(
        config.frequency, 0, nfreq,
        mp.FluxRegion(center=mp.Vector3(0, -nf_size/2, 0), size=mp.Vector3(nf_size, 0, 0), weight=-1)
    )
    flux_box_left = sim.add_flux(
        config.frequency, 0, nfreq,
        mp.FluxRegion(center=mp.Vector3(-nf_size/2, 0, 0), size=mp.Vector3(0, nf_size, 0), weight=-1)
    )
    flux_box_right = sim.add_flux(
        config.frequency, 0, nfreq,
        mp.FluxRegion(center=mp.Vector3(nf_size/2, 0, 0), size=mp.Vector3(0, nf_size, 0))
    )
    
    # Run simulation
    sim.run(until=config.simulation_time)
    
    # Get scattered flux (total flux through the box)
    flux_top = mp.get_fluxes(flux_box_top)[0]
    flux_bottom = mp.get_fluxes(flux_box_bottom)[0]
    flux_left = mp.get_fluxes(flux_box_left)[0]
    flux_right = mp.get_fluxes(flux_box_right)[0]
    
    total_scattered_flux = flux_top + flux_bottom + flux_left + flux_right
    
    # Calculate scattering cross-section
    # For a line source with height h, the incident intensity is P_inc / h
    # The scattered power divided by incident intensity gives cross-section
    # This is an approximation; more rigorous normalization may be needed
    
    end_time = time.time()
    runtime = end_time - start_time
    
    # Analytical reference
    sigma_sca_analytical = scattering_cross_section_2d(
        config.cylinder_n, config.background_n, 
        config.cylinder_radius, config.wavelength
    )
    sigma_ext_analytical = extinction_cross_section_2d(
        config.cylinder_n, config.background_n,
        config.cylinder_radius, config.wavelength
    )
    
    # Compile results
    results = {
        'config': config.to_dict(),
        'runtime_seconds': runtime,
        'scattered_flux': {
            'top': flux_top,
            'bottom': flux_bottom,
            'left': flux_left,
            'right': flux_right,
            'total': total_scattered_flux,
        },
        'analytical': {
            'sigma_scattering': sigma_sca_analytical,
            'sigma_extinction': sigma_ext_analytical,
        },
    }
    
    # Save results if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / f"mie_res{config.resolution}_pml{config.pml_thickness}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {results_file}")
    
    # Clean up
    sim.reset_meep()
    
    return results


# =============================================================================
# PARAMETER SWEEPS
# =============================================================================

def resolution_sweep(
    resolutions: list = [10, 20, 40, 80],
    use_subpixel: bool = True,
    output_dir: Path = None
):
    """
    Run resolution convergence study.
    
    Parameters
    ----------
    resolutions : list
        List of resolution values to test
    use_subpixel : bool
        Whether to enable subpixel smoothing
    output_dir : Path
        Directory to save results
        
    Returns
    -------
    results : list
        List of result dictionaries for each resolution
    """
    results = []
    
    for res in resolutions:
        print(f"\n{'='*60}")
        print(f"Running resolution = {res} (subpixel = {use_subpixel})")
        print(f"{'='*60}")
        
        config = SimulationConfig(
            resolution=res,
            use_subpixel=use_subpixel,
        )
        
        result = run_mie_simulation(config, output_dir)
        results.append(result)
        
        print(f"Runtime: {result['runtime_seconds']:.2f} s")
        print(f"Analytical σ_sca: {result['analytical']['sigma_scattering']:.6f}")
    
    return results


def pml_sweep(
    pml_thicknesses: list = [0.5, 1.0, 2.0, 4.0],
    resolution: int = 40,
    output_dir: Path = None
):
    """
    Run PML thickness convergence study.
    
    Parameters
    ----------
    pml_thicknesses : list
        List of PML thickness values to test (in wavelengths)
    resolution : int
        Fixed resolution for all runs
    output_dir : Path
        Directory to save results
        
    Returns
    -------
    results : list
        List of result dictionaries for each PML thickness
    """
    results = []
    
    for pml in pml_thicknesses:
        print(f"\n{'='*60}")
        print(f"Running PML thickness = {pml} λ")
        print(f"{'='*60}")
        
        config = SimulationConfig(
            resolution=resolution,
            pml_thickness=pml,  # thickness in wavelengths (λ = 1 μm)
        )
        
        result = run_mie_simulation(config, output_dir)
        results.append(result)
        
        print(f"Runtime: {result['runtime_seconds']:.2f} s")
    
    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Mie Scattering FDTD Simulation")
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'resolution_sweep', 'pml_sweep'],
                        help='Simulation mode')
    parser.add_argument('--resolution', type=int, default=20,
                        help='Grid resolution (pixels per μm)')
    parser.add_argument('--pml', type=float, default=1.0,
                        help='PML thickness (μm)')
    parser.add_argument('--subpixel', action='store_true', default=True,
                        help='Enable subpixel smoothing')
    parser.add_argument('--no-subpixel', dest='subpixel', action='store_false',
                        help='Disable subpixel smoothing')
    parser.add_argument('--output', type=str, default='../../results/mie_scattering',
                        help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(__file__).parent / args.output
    
    if args.mode == 'single':
        # Single simulation run
        config = SimulationConfig(
            resolution=args.resolution,
            pml_thickness=args.pml,
            use_subpixel=args.subpixel,
        )
        
        print("="*60)
        print("Mie Scattering Simulation - Single Run")
        print("="*60)
        print(f"Resolution: {config.resolution} px/μm")
        print(f"Pixels per wavelength: {config.pixels_per_wavelength}")
        print(f"PML thickness: {config.pml_thickness} μm")
        print(f"Subpixel smoothing: {config.use_subpixel}")
        print(f"Cell size: {config.cell_size} μm")
        print("="*60)
        
        results = run_mie_simulation(config, output_dir)
        
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Runtime: {results['runtime_seconds']:.2f} seconds")
        print(f"Analytical σ_sca: {results['analytical']['sigma_scattering']:.6f}")
        print(f"Analytical σ_ext: {results['analytical']['sigma_extinction']:.6f}")
        
    elif args.mode == 'resolution_sweep':
        # Resolution convergence study
        print("\n" + "="*60)
        print("RESOLUTION CONVERGENCE STUDY")
        print("="*60)
        
        # With subpixel smoothing
        results_with = resolution_sweep(
            resolutions=[10, 20, 40, 80],
            use_subpixel=True,
            output_dir=output_dir / "resolution_sweep_subpixel_on"
        )
        
        # Without subpixel smoothing
        results_without = resolution_sweep(
            resolutions=[10, 20, 40, 80],
            use_subpixel=False,
            output_dir=output_dir / "resolution_sweep_subpixel_off"
        )
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print("Resolution sweep complete. Results saved to:", output_dir)
        
    elif args.mode == 'pml_sweep':
        # PML thickness study
        print("\n" + "="*60)
        print("PML THICKNESS STUDY")
        print("="*60)
        
        results = pml_sweep(
            pml_thicknesses=[0.5, 1.0, 2.0, 4.0],
            resolution=40,
            output_dir=output_dir / "pml_sweep"
        )
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print("PML sweep complete. Results saved to:", output_dir)
