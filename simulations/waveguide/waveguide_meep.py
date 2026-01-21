"""
Meep FDTD Simulation for Dielectric Slab Waveguide
===================================================
Simulates transmission through a 2D dielectric slab waveguide.
Measures transmission spectrum and compares to analytical cutoff.

Structure: Air | Dielectric Core (n=3.5) | Air
"""

import meep as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
import json
from pathlib import Path


@dataclass
class WaveguideConfig:
    """Configuration for waveguide simulation."""
    # Material properties
    n_core: float = 3.5          # Core refractive index (e.g., silicon)
    n_clad: float = 1.0          # Cladding (air)
    
    # Geometry
    core_width: float = 0.5      # Waveguide core width (μm)
    waveguide_length: float = 5.0  # Propagation length (μm)
    
    # Simulation parameters
    resolution: int = 40         # pixels per μm
    pml_thickness: float = 1.0   # μm
    padding_y: float = 2.0       # Cladding thickness above/below core
    
    # Source parameters
    wavelength_min: float = 0.4  # μm
    wavelength_max: float = 2.0  # μm
    
    # Runtime
    simulation_time: Optional[float] = None
    decay_threshold: float = 1e-6
    
    @property
    def fcen(self) -> float:
        """Center frequency."""
        wl_center = (self.wavelength_min + self.wavelength_max) / 2
        return 1.0 / wl_center
    
    @property
    def df(self) -> float:
        """Frequency width."""
        f_min = 1.0 / self.wavelength_max
        f_max = 1.0 / self.wavelength_min
        return f_max - f_min
    
    @property
    def cell_x(self) -> float:
        """Cell size in x (propagation direction)."""
        return self.waveguide_length + 2 * self.pml_thickness + 2.0
    
    @property
    def cell_y(self) -> float:
        """Cell size in y (transverse direction)."""
        return self.core_width + 2 * self.padding_y + 2 * self.pml_thickness


def analytical_cutoff_frequencies(config: WaveguideConfig, num_modes: int = 5) -> List[dict]:
    """
    Calculate analytical cutoff frequencies for TE modes in a symmetric slab waveguide.
    
    For a symmetric slab waveguide with core index n1 and cladding index n2,
    the cutoff condition for TE_m mode is:
        f_cutoff = m * c / (2 * d * sqrt(n1^2 - n2^2))
    where d is the core width and m = 0, 1, 2, ...
    
    Note: m=0 (fundamental mode) has no cutoff - it's always guided.
    """
    n1 = config.n_core
    n2 = config.n_clad
    d = config.core_width
    
    NA = np.sqrt(n1**2 - n2**2)  # Numerical aperture
    
    modes = []
    for m in range(num_modes):
        if m == 0:
            f_cutoff = 0  # Fundamental mode always guided
            wl_cutoff = float('inf')
        else:
            # f_cutoff = m / (2 * d * NA)  in units where c=1
            f_cutoff = m / (2 * d * NA)
            wl_cutoff = 1.0 / f_cutoff if f_cutoff > 0 else float('inf')
        
        modes.append({
            "mode_number": m,
            "mode_name": f"TE{m}",
            "cutoff_frequency": f_cutoff,
            "cutoff_wavelength": wl_cutoff
        })
    
    return modes


def run_waveguide_simulation(
    config: WaveguideConfig,
    output_dir: Optional[Path] = None
) -> dict:
    """
    Run FDTD simulation of waveguide transmission.
    
    Uses two flux monitors:
    1. Input flux (near source)
    2. Output flux (after propagation)
    
    Transmission = output_flux / input_flux
    """
    print("=" * 60)
    print("MEEP FDTD - DIELECTRIC SLAB WAVEGUIDE")
    print("=" * 60)
    print(f"Resolution: {config.resolution} px/μm")
    print(f"Core: width={config.core_width} μm, n={config.n_core}")
    print(f"Cladding: n={config.n_clad}")
    print(f"Wavelength range: {config.wavelength_min} - {config.wavelength_max} μm")
    print(f"Cell size: {config.cell_x:.2f} x {config.cell_y:.2f} μm")
    print("=" * 60)
    
    # Cell size
    cell = mp.Vector3(config.cell_x, config.cell_y, 0)
    
    # Geometry: dielectric slab centered at y=0
    eps_core = config.n_core ** 2
    geometry = [
        mp.Block(
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(mp.inf, config.core_width, mp.inf),
            material=mp.Medium(epsilon=eps_core)
        )
    ]
    
    # PML
    pml_layers = [mp.PML(config.pml_thickness)]
    
    # Source: Gaussian pulse, eigenmode source for better coupling
    # Place source inside the waveguide, near the left edge
    source_x = -config.waveguide_length / 2 + 0.5
    
    sources = [mp.Source(
        src=mp.GaussianSource(config.fcen, fwidth=config.df),
        component=mp.Ez,  # TE polarization (Ez, Hx, Hy)
        center=mp.Vector3(source_x, 0, 0),
        size=mp.Vector3(0, config.core_width * 2, 0)  # Line source spanning core
    )]
    
    # Create simulation
    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        sources=sources,
        boundary_layers=pml_layers,
        resolution=config.resolution
    )
    
    # Flux monitors
    nfreq = 100  # Number of frequency points
    
    # Input flux (just after source)
    input_x = source_x + 0.5
    input_flux = sim.add_flux(
        config.fcen, config.df, nfreq,
        mp.FluxRegion(
            center=mp.Vector3(input_x, 0, 0),
            size=mp.Vector3(0, config.cell_y - 2 * config.pml_thickness, 0)
        )
    )
    
    # Output flux (near end of waveguide)
    output_x = config.waveguide_length / 2 - 0.5
    output_flux = sim.add_flux(
        config.fcen, config.df, nfreq,
        mp.FluxRegion(
            center=mp.Vector3(output_x, 0, 0),
            size=mp.Vector3(0, config.cell_y - 2 * config.pml_thickness, 0)
        )
    )
    
    print("\n[1/2] Running simulation...")
    
    import time
    start_time = time.time()
    
    if config.simulation_time is not None:
        sim.run(until=config.simulation_time)
    else:
        sim.run(until_after_sources=mp.stop_when_fields_decayed(
            50, mp.Ez, mp.Vector3(output_x, 0, 0), config.decay_threshold
        ))
    
    runtime = time.time() - start_time
    print(f"      Runtime: {runtime:.2f} s")
    print(f"      Simulation time: {sim.meep_time():.2f} Meep units")
    
    # Extract flux data
    print("\n[2/2] Extracting transmission spectrum...")
    
    freqs = np.array(mp.get_flux_freqs(input_flux))
    input_data = np.array(mp.get_fluxes(input_flux))
    output_data = np.array(mp.get_fluxes(output_flux))
    
    # Compute transmission (handle division carefully)
    with np.errstate(divide='ignore', invalid='ignore'):
        transmission = np.where(input_data > 0, output_data / input_data, 0)
    
    # Convert to wavelengths
    wavelengths = 1.0 / freqs
    
    # Get analytical cutoffs
    cutoff_modes = analytical_cutoff_frequencies(config)
    
    # Compile results
    results = {
        "config": {
            "n_core": config.n_core,
            "n_clad": config.n_clad,
            "core_width": config.core_width,
            "waveguide_length": config.waveguide_length,
            "resolution": config.resolution,
            "pml_thickness": config.pml_thickness,
            "wavelength_range": [config.wavelength_min, config.wavelength_max]
        },
        "simulation": {
            "runtime_seconds": runtime,
            "simulation_time": sim.meep_time(),
            "num_freq_points": nfreq
        },
        "frequencies": freqs.tolist(),
        "wavelengths": wavelengths.tolist(),
        "input_flux": input_data.tolist(),
        "output_flux": output_data.tolist(),
        "transmission": transmission.tolist(),
        "analytical_cutoffs": cutoff_modes
    }
    
    # Summary statistics
    valid_idx = (wavelengths >= config.wavelength_min) & (wavelengths <= config.wavelength_max)
    if np.any(valid_idx):
        avg_trans = np.mean(transmission[valid_idx])
        max_trans = np.max(transmission[valid_idx])
        print(f"  Average transmission: {avg_trans:.4f}")
        print(f"  Peak transmission: {max_trans:.4f}")
        results["summary"] = {
            "average_transmission": float(avg_trans),
            "peak_transmission": float(max_trans)
        }
    
    # Print cutoff info
    print(f"\n  Analytical cutoff frequencies:")
    for mode in cutoff_modes[:3]:
        if mode["cutoff_frequency"] > 0:
            print(f"    {mode['mode_name']}: f_c = {mode['cutoff_frequency']:.4f} (λ_c = {mode['cutoff_wavelength']:.4f} μm)")
        else:
            print(f"    {mode['mode_name']}: no cutoff (always guided)")
    
    # Save results
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"waveguide_res{config.resolution}_pml{config.pml_thickness}.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    if "summary" in results:
        print(f"  Average transmission: {results['summary']['average_transmission']:.4f}")
        print(f"  Peak transmission: {results['summary']['peak_transmission']:.4f}")
    print(f"  Total runtime: {runtime:.2f} s")
    
    return results


if __name__ == "__main__":
    # Default test run
    config = WaveguideConfig(
        resolution=40,
        pml_thickness=1.0
    )
    
    output_dir = Path("../../results/waveguide")
    results = run_waveguide_simulation(config, output_dir)
