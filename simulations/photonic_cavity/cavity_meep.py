"""
Meep FDTD Simulation for 1D Photonic Crystal Cavity
====================================================
Simulates a DBR (Distributed Bragg Reflector) cavity with a defect layer.
Extracts resonance frequency and Q-factor using Harminv.

Structure: [H L]^N  D  [L H]^N
"""

import meep as mp
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
import json
from pathlib import Path


@dataclass
class MeepCavityConfig:
    """Configuration for Meep cavity simulation."""
    # Material properties
    n_high: float = 3.5
    n_low: float = 1.5
    n_defect: float = 3.5
    
    # Design parameters
    center_wavelength: float = 1.0  # μm
    num_pairs: int = 5
    
    # Simulation parameters
    resolution: int = 40            # pixels per μm
    pml_thickness: float = 1.0      # μm
    padding: float = 1.0            # space between structure and PML
    
    # Harminv parameters
    harminv_dt: float = 0.1         # time step for Harminv output
    
    # Convergence - FIXED: use reasonable values for DBR cavities
    simulation_time: Optional[float] = None  # If None, run until decay
    decay_threshold: float = 1e-6   # More reasonable (was 1e-9)
    min_run_time: float = 200       # Reduced minimum for faster testing
    max_run_time: float = 2000      # Cap to prevent runaway simulations
    
    @property
    def d_high(self) -> float:
        return self.center_wavelength / (4 * self.n_high)
    
    @property
    def d_low(self) -> float:
        return self.center_wavelength / (4 * self.n_low)
    
    @property
    def d_defect(self) -> float:
        return self.center_wavelength / (2 * self.n_defect)
    
    @property
    def structure_length(self) -> float:
        """Total length of photonic crystal structure."""
        mirror = self.num_pairs * (self.d_high + self.d_low)
        return 2 * mirror + self.d_defect
    
    @property
    def cell_length(self) -> float:
        """Total simulation cell length."""
        return self.structure_length + 2 * self.padding + 2 * self.pml_thickness


def build_cavity_geometry(config: MeepCavityConfig) -> Tuple[List[mp.GeometricObject], float]:
    """
    Build the 1D cavity geometry.
    
    Returns
    -------
    geometry : list of Meep geometric objects
    center_x : x-coordinate of the defect center
    """
    geometry = []
    
    # Start position (left edge of structure)
    x = -config.structure_length / 2
    
    eps_high = config.n_high ** 2
    eps_low = config.n_low ** 2
    eps_defect = config.n_defect ** 2
    
    # Left mirror: [H L]^N
    for _ in range(config.num_pairs):
        # High-index layer
        geometry.append(mp.Block(
            center=mp.Vector3(x + config.d_high / 2, 0, 0),
            size=mp.Vector3(config.d_high, mp.inf, mp.inf),
            material=mp.Medium(epsilon=eps_high)
        ))
        x += config.d_high
        
        # Low-index layer
        geometry.append(mp.Block(
            center=mp.Vector3(x + config.d_low / 2, 0, 0),
            size=mp.Vector3(config.d_low, mp.inf, mp.inf),
            material=mp.Medium(epsilon=eps_low)
        ))
        x += config.d_low
    
    # Defect layer
    defect_center = x + config.d_defect / 2
    geometry.append(mp.Block(
        center=mp.Vector3(defect_center, 0, 0),
        size=mp.Vector3(config.d_defect, mp.inf, mp.inf),
        material=mp.Medium(epsilon=eps_defect)
    ))
    x += config.d_defect
    
    # Right mirror: [L H]^N
    for _ in range(config.num_pairs):
        # Low-index layer
        geometry.append(mp.Block(
            center=mp.Vector3(x + config.d_low / 2, 0, 0),
            size=mp.Vector3(config.d_low, mp.inf, mp.inf),
            material=mp.Medium(epsilon=eps_low)
        ))
        x += config.d_low
        
        # High-index layer
        geometry.append(mp.Block(
            center=mp.Vector3(x + config.d_high / 2, 0, 0),
            size=mp.Vector3(config.d_high, mp.inf, mp.inf),
            material=mp.Medium(epsilon=eps_high)
        ))
        x += config.d_high
    
    return geometry, defect_center


def run_cavity_simulation(
    config: MeepCavityConfig,
    output_dir: Optional[Path] = None
) -> dict:
    """
    Run FDTD simulation of the cavity and extract resonance via Harminv.
    
    Returns
    -------
    results : dict with resonance frequency, Q-factor, and field data
    """
    print("=" * 60)
    print("MEEP FDTD - 1D PHOTONIC CRYSTAL CAVITY")
    print("=" * 60)
    print(f"Resolution: {config.resolution} px/μm")
    print(f"Structure: [H L]^{config.num_pairs} D [L H]^{config.num_pairs}")
    print(f"  d_H = {config.d_high:.4f} μm, d_L = {config.d_low:.4f} μm")
    print(f"  d_defect = {config.d_defect:.4f} μm")
    print(f"  Total structure: {config.structure_length:.4f} μm")
    print(f"PML thickness: {config.pml_thickness} μm")
    print("=" * 60)
    
    # Build geometry
    geometry, defect_center = build_cavity_geometry(config)
    
    # Cell size (1D simulation in x)
    cell = mp.Vector3(config.cell_length, 0, 0)
    
    # PML layers
    pml_layers = [mp.PML(config.pml_thickness)]
    
    # Source: Gaussian pulse centered at design frequency
    fcen = 1.0 / config.center_wavelength  # center frequency
    df = 0.5 * fcen  # bandwidth (wide enough to excite resonance)
    
    # Place source at the defect location for efficient excitation
    sources = [mp.Source(
        src=mp.GaussianSource(fcen, fwidth=df),
        component=mp.Ez,
        center=mp.Vector3(defect_center, 0, 0)
    )]
    
    # Create simulation
    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        sources=sources,
        boundary_layers=pml_layers,
        resolution=config.resolution
    )
    
    # Harminv for resonance extraction
    # Monitor at defect center - use wider bandwidth to catch the resonance
    # Bandgap is roughly from f=0.2 to f=0.4 for this structure
    fcen_harminv = fcen  # Still center around design frequency
    df_harminv = 0.8 * fcen  # Very wide bandwidth for Harminv
    
    harminv_instance = mp.Harminv(mp.Ez, mp.Vector3(defect_center, 0, 0), fcen_harminv, df_harminv)
    
    # Field monitor for decay tracking
    field_data = []
    time_data = []
    
    def record_field(sim):
        field_data.append(abs(sim.get_field_point(mp.Ez, mp.Vector3(defect_center, 0, 0))))
        time_data.append(sim.meep_time())
    
    print("\n[1/2] Running simulation with Harminv...")
    print(f"      Harminv monitoring: fcen={fcen_harminv:.4f}, df={df_harminv:.4f}")
    
    import time
    start_time = time.time()
    
    # Run until field decays OR max time reached
    if config.simulation_time is not None:
        sim.run(
            mp.at_every(config.harminv_dt, record_field),
            harminv_instance,
            until=config.simulation_time
        )
    else:
        # First run minimum time for Harminv to accumulate data
        sim.run(
            mp.at_every(config.harminv_dt, record_field),
            harminv_instance,
            until=config.min_run_time
        )
        
        # Check if we already have good modes from Harminv
        print(f"      After min_run_time ({config.min_run_time}): {len(harminv_instance.modes)} modes found")
        
        # Continue with capped runtime to prevent runaway
        remaining_time = config.max_run_time - config.min_run_time
        if remaining_time > 0:
            sim.run(
                mp.at_every(config.harminv_dt, record_field),
                until=remaining_time
            )
    
    runtime = time.time() - start_time
    print(f"      Runtime: {runtime:.2f} s")
    print(f"      Simulation time: {sim.meep_time():.2f} Meep units")
    
    # Extract Harminv results
    print("\n[2/2] Extracting resonance modes...")
    
    harminv_results = []
    for mode in harminv_instance.modes:
        freq = mode.freq
        decay = mode.decay
        Q = -0.5 * freq / decay if decay != 0 else float('inf')
        amp = abs(mode.amp)
        err = abs(mode.err)  # Convert complex error to magnitude
        
        harminv_results.append({
            "frequency": freq,
            "wavelength": 1.0 / freq if freq > 0 else float('inf'),
            "decay_rate": decay,
            "Q_factor": Q,
            "amplitude": amp,
            "error": err
        })
        
        print(f"  Mode: f={freq:.6f}, λ={1/freq:.6f} μm, Q={Q:.1f}, err={err:.2e}")
    
    # Find the dominant mode (highest amplitude, reasonable Q)
    valid_modes = [m for m in harminv_results if m["Q_factor"] > 0 and m["error"] < 0.1]
    if valid_modes:
        dominant_mode = max(valid_modes, key=lambda m: m["amplitude"])
    else:
        dominant_mode = harminv_results[0] if harminv_results else None
    
    # Compile results
    results = {
        "config": {
            "n_high": config.n_high,
            "n_low": config.n_low,
            "n_defect": config.n_defect,
            "center_wavelength": config.center_wavelength,
            "num_pairs": config.num_pairs,
            "resolution": config.resolution,
            "pml_thickness": config.pml_thickness,
            "decay_threshold": config.decay_threshold
        },
        "simulation": {
            "runtime_seconds": runtime,
            "simulation_time": sim.meep_time(),
            "num_timesteps": sim.timestep()
        },
        "harminv_modes": harminv_results,
        "dominant_mode": dominant_mode,
        "field_decay": {
            "time": time_data,
            "amplitude": field_data
        }
    }
    
    # Save results
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"cavity_res{config.resolution}_pml{config.pml_thickness}.json"
        
        with open(output_dir / filename, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_dir / filename}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    if dominant_mode:
        print(f"  Resonance frequency: {dominant_mode['frequency']:.6f}")
        print(f"  Resonance wavelength: {dominant_mode['wavelength']:.6f} μm")
        print(f"  Q-factor: {dominant_mode['Q_factor']:.1f}")
    else:
        print("  No valid resonance mode found!")
    print(f"  Total runtime: {runtime:.2f} s")
    
    return results


def run_transmission_spectrum(
    config: MeepCavityConfig,
    freq_range: Tuple[float, float] = (0.8, 1.25),
    nfreq: int = 500,
    output_dir: Optional[Path] = None
) -> dict:
    """
    Run transmission simulation to get full spectrum.
    Uses flux monitors on either side of the cavity.
    """
    print("=" * 60)
    print("MEEP FDTD - CAVITY TRANSMISSION SPECTRUM")
    print("=" * 60)
    
    geometry, defect_center = build_cavity_geometry(config)
    cell = mp.Vector3(config.cell_length, 0, 0)
    pml_layers = [mp.PML(config.pml_thickness)]
    
    fcen = (freq_range[0] + freq_range[1]) / 2
    df = freq_range[1] - freq_range[0]
    
    # Source on left side
    src_x = -config.structure_length / 2 - config.padding / 2
    sources = [mp.Source(
        src=mp.GaussianSource(fcen, fwidth=df),
        component=mp.Ez,
        center=mp.Vector3(src_x, 0, 0)
    )]
    
    # --- Reference simulation (no structure) ---
    print("\n[1/2] Reference simulation (no structure)...")
    sim_ref = mp.Simulation(
        cell_size=cell,
        sources=sources,
        boundary_layers=pml_layers,
        resolution=config.resolution
    )
    
    # Flux monitor on right side
    flux_x = config.structure_length / 2 + config.padding / 2
    flux_ref = sim_ref.add_flux(fcen, df, nfreq, 
                                 mp.FluxRegion(center=mp.Vector3(flux_x, 0, 0), size=mp.Vector3(0, 0, 0)))
    
    import time
    t0 = time.time()
    sim_ref.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(flux_x, 0, 0), 1e-6))
    ref_time = time.time() - t0
    print(f"      Runtime: {ref_time:.2f} s")
    
    ref_flux = mp.get_fluxes(flux_ref)
    ref_freqs = mp.get_flux_freqs(flux_ref)
    
    # --- Structure simulation ---
    print("\n[2/2] Structure simulation...")
    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        sources=sources,
        boundary_layers=pml_layers,
        resolution=config.resolution
    )
    
    flux_trans = sim.add_flux(fcen, df, nfreq,
                               mp.FluxRegion(center=mp.Vector3(flux_x, 0, 0), size=mp.Vector3(0, 0, 0)))
    
    t0 = time.time()
    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(flux_x, 0, 0), 1e-6))
    struct_time = time.time() - t0
    print(f"      Runtime: {struct_time:.2f} s")
    
    trans_flux = mp.get_fluxes(flux_trans)
    
    # Compute transmission
    transmission = np.array(trans_flux) / np.array(ref_flux)
    wavelengths = 1.0 / np.array(ref_freqs)
    
    results = {
        "config": {
            "resolution": config.resolution,
            "pml_thickness": config.pml_thickness,
            "num_pairs": config.num_pairs
        },
        "spectrum": {
            "frequencies": list(ref_freqs),
            "wavelengths": wavelengths.tolist(),
            "transmission": transmission.tolist()
        },
        "runtime": {
            "reference": ref_time,
            "structure": struct_time,
            "total": ref_time + struct_time
        }
    }
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"cavity_spectrum_res{config.resolution}.json"
        with open(output_dir / filename, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {output_dir / filename}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run cavity simulation")
    parser.add_argument("--resolution", type=int, default=40)
    parser.add_argument("--pml", type=float, default=1.0)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--spectrum", action="store_true", help="Run transmission spectrum")
    args = parser.parse_args()
    
    config = MeepCavityConfig(
        resolution=args.resolution,
        pml_thickness=args.pml,
        num_pairs=args.pairs
    )
    
    output_dir = Path("../../results/photonic_cavity")
    
    if args.spectrum:
        run_transmission_spectrum(config, output_dir=output_dir)
    else:
        run_cavity_simulation(config, output_dir=output_dir)
