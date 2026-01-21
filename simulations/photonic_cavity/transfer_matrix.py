"""
Transfer Matrix Method (TMM) for 1D Photonic Crystal Cavity
============================================================
Calculates theoretical transmission spectrum and resonance frequencies
for a Bragg mirror cavity with a defect layer.

Structure: [H L]^N  D  [L H]^N
- H = high-index layer (n_H, thickness = λ₀/(4*n_H))
- L = low-index layer (n_L, thickness = λ₀/(4*n_L))
- D = defect layer (n_D, thickness = λ₀/(2*n_D)) for resonance at λ₀
- N = number of Bragg pairs per mirror
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
import json


@dataclass
class CavityConfig:
    """Configuration for 1D photonic crystal cavity."""
    n_high: float = 3.5       # High-index material (e.g., Si)
    n_low: float = 1.5        # Low-index material (e.g., SiO2)
    n_defect: float = 3.5     # Defect layer material
    n_ambient: float = 1.0    # Surrounding medium
    center_wavelength: float = 1.0  # Design wavelength (μm)
    num_pairs: int = 5        # Number of Bragg pairs per mirror
    
    @property
    def d_high(self) -> float:
        """Quarter-wave thickness for high-index layer."""
        return self.center_wavelength / (4 * self.n_high)
    
    @property
    def d_low(self) -> float:
        """Quarter-wave thickness for low-index layer."""
        return self.center_wavelength / (4 * self.n_low)
    
    @property
    def d_defect(self) -> float:
        """Half-wave thickness for defect (resonance at center wavelength)."""
        return self.center_wavelength / (2 * self.n_defect)
    
    @property
    def total_thickness(self) -> float:
        """Total structure thickness."""
        mirror_thickness = self.num_pairs * (self.d_high + self.d_low)
        return 2 * mirror_thickness + self.d_defect


def transfer_matrix_layer(n: float, d: float, wavelength: float) -> np.ndarray:
    """
    Transfer matrix for a single layer.
    
    Parameters
    ----------
    n : refractive index
    d : thickness (μm)
    wavelength : wavelength (μm)
    
    Returns
    -------
    2x2 transfer matrix
    """
    k = 2 * np.pi * n / wavelength
    phase = k * d
    
    M = np.array([
        [np.cos(phase), 1j * np.sin(phase) / n],
        [1j * n * np.sin(phase), np.cos(phase)]
    ], dtype=complex)
    
    return M


def transfer_matrix_interface(n1: float, n2: float) -> np.ndarray:
    """
    Transfer matrix for interface between two media (normal incidence).
    
    Using convention: [E_forward, E_backward] basis
    """
    # For the field continuity approach, we use the propagation matrix formulation
    # This is already included in the layer matrices above
    return np.eye(2, dtype=complex)


def compute_transmission_spectrum(
    config: CavityConfig,
    wavelengths: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute transmission and reflection spectra using transfer matrix method.
    
    Parameters
    ----------
    config : CavityConfig
    wavelengths : array of wavelengths (μm)
    
    Returns
    -------
    transmission, reflection : arrays of |t|², |r|²
    """
    transmission = np.zeros_like(wavelengths)
    reflection = np.zeros_like(wavelengths)
    
    for i, wl in enumerate(wavelengths):
        # Build total transfer matrix
        M_total = np.eye(2, dtype=complex)
        
        # Left mirror: [H L]^N (starting from ambient)
        for _ in range(config.num_pairs):
            M_total = M_total @ transfer_matrix_layer(config.n_high, config.d_high, wl)
            M_total = M_total @ transfer_matrix_layer(config.n_low, config.d_low, wl)
        
        # Defect layer
        M_total = M_total @ transfer_matrix_layer(config.n_defect, config.d_defect, wl)
        
        # Right mirror: [L H]^N
        for _ in range(config.num_pairs):
            M_total = M_total @ transfer_matrix_layer(config.n_low, config.d_low, wl)
            M_total = M_total @ transfer_matrix_layer(config.n_high, config.d_high, wl)
        
        # Extract transmission and reflection
        # For the transfer matrix M relating [E+, E-] at input to output:
        # t = 1/M[0,0], r = M[1,0]/M[0,0]
        # But we need to account for impedance matching at boundaries
        
        n_in = config.n_ambient
        n_out = config.n_ambient
        
        # Transmission coefficient (power)
        t_amplitude = 2 * n_in / (M_total[0, 0] * n_in + M_total[0, 1] * n_in * n_out + 
                                   M_total[1, 0] + M_total[1, 1] * n_out)
        transmission[i] = np.abs(t_amplitude)**2 * n_out / n_in
        
        # Reflection coefficient
        r_amplitude = (M_total[0, 0] * n_in + M_total[0, 1] * n_in * n_out - 
                      M_total[1, 0] - M_total[1, 1] * n_out) / \
                     (M_total[0, 0] * n_in + M_total[0, 1] * n_in * n_out + 
                      M_total[1, 0] + M_total[1, 1] * n_out)
        reflection[i] = np.abs(r_amplitude)**2
    
    return transmission, reflection


def find_resonance(
    config: CavityConfig,
    wavelength_range: Tuple[float, float] = (0.9, 1.1),
    num_points: int = 10001
) -> Tuple[float, float, float]:
    """
    Find resonance wavelength and estimate Q-factor from transmission peak.
    
    Returns
    -------
    resonance_wavelength, peak_transmission, estimated_Q
    """
    wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], num_points)
    transmission, _ = compute_transmission_spectrum(config, wavelengths)
    
    # Find peak
    peak_idx = np.argmax(transmission)
    wl_res = wavelengths[peak_idx]
    T_peak = transmission[peak_idx]
    
    # Estimate Q from FWHM
    half_max = T_peak / 2
    
    # Find left and right half-maximum points
    left_idx = peak_idx
    while left_idx > 0 and transmission[left_idx] > half_max:
        left_idx -= 1
    
    right_idx = peak_idx
    while right_idx < len(transmission) - 1 and transmission[right_idx] > half_max:
        right_idx += 1
    
    # Interpolate for better accuracy
    if left_idx > 0:
        wl_left = np.interp(half_max, 
                           [transmission[left_idx], transmission[left_idx + 1]],
                           [wavelengths[left_idx], wavelengths[left_idx + 1]])
    else:
        wl_left = wavelengths[0]
    
    if right_idx < len(transmission) - 1:
        wl_right = np.interp(half_max,
                            [transmission[right_idx], transmission[right_idx - 1]],
                            [wavelengths[right_idx], wavelengths[right_idx - 1]])
    else:
        wl_right = wavelengths[-1]
    
    fwhm = abs(wl_right - wl_left)
    Q_estimated = wl_res / fwhm if fwhm > 0 else np.inf
    
    return wl_res, T_peak, Q_estimated


def theoretical_Q_factor(config: CavityConfig) -> float:
    """
    Estimate theoretical Q-factor for DBR cavity.
    
    Q ≈ (π/2) * (n_H/n_L)^(2N) * n_D * L_eff / λ
    
    For a symmetric cavity with N pairs per mirror.
    """
    # Simplified estimate based on mirror reflectivity
    R_mirror = ((config.n_high - config.n_low) / (config.n_high + config.n_low))**(2 * config.num_pairs)
    
    # Effective cavity length (roughly the defect)
    L_eff = config.d_defect
    
    # Q from Fabry-Perot formula: Q = 2πn L / λ * F, where F = π√R / (1-R)
    finesse = np.pi * np.sqrt(R_mirror) / (1 - R_mirror)
    Q = 2 * np.pi * config.n_defect * L_eff / config.center_wavelength * finesse / np.pi
    
    return Q


if __name__ == "__main__":
    print("=" * 60)
    print("TRANSFER MATRIX METHOD - 1D PHOTONIC CRYSTAL CAVITY")
    print("=" * 60)
    
    # Default configuration
    config = CavityConfig(
        n_high=3.5,        # Silicon
        n_low=1.5,         # SiO2
        n_defect=3.5,      # Silicon defect
        center_wavelength=1.0,
        num_pairs=5
    )
    
    print(f"\nStructure: [H L]^{config.num_pairs} D [L H]^{config.num_pairs}")
    print(f"  n_H = {config.n_high}, d_H = {config.d_high:.4f} μm")
    print(f"  n_L = {config.n_low}, d_L = {config.d_low:.4f} μm")
    print(f"  n_D = {config.n_defect}, d_D = {config.d_defect:.4f} μm")
    print(f"  Total thickness: {config.total_thickness:.4f} μm")
    
    # Find resonance
    wl_res, T_peak, Q_est = find_resonance(config)
    Q_theory = theoretical_Q_factor(config)
    
    print(f"\n--- Resonance Analysis ---")
    print(f"  Resonance wavelength: {wl_res:.6f} μm")
    print(f"  Peak transmission: {T_peak:.6f}")
    print(f"  Estimated Q (FWHM): {Q_est:.1f}")
    print(f"  Theoretical Q estimate: {Q_theory:.1f}")
    
    # Compute and save spectrum
    wavelengths = np.linspace(0.8, 1.2, 1001)
    transmission, reflection = compute_transmission_spectrum(config, wavelengths)
    
    # Save results
    results = {
        "config": {
            "n_high": config.n_high,
            "n_low": config.n_low,
            "n_defect": config.n_defect,
            "center_wavelength": config.center_wavelength,
            "num_pairs": config.num_pairs,
            "d_high": config.d_high,
            "d_low": config.d_low,
            "d_defect": config.d_defect,
            "total_thickness": config.total_thickness
        },
        "resonance": {
            "wavelength": wl_res,
            "peak_transmission": T_peak,
            "Q_estimated": Q_est,
            "Q_theoretical": Q_theory
        },
        "spectrum": {
            "wavelengths": wavelengths.tolist(),
            "transmission": transmission.tolist(),
            "reflection": reflection.tolist()
        }
    }
    
    from pathlib import Path
    output_dir = Path("../../results/photonic_cavity")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "tmm_reference.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_dir / 'tmm_reference.json'}")
    
    # Plot if matplotlib available
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        ax1.semilogy(wavelengths, transmission, 'b-', linewidth=1.5, label='Transmission')
        ax1.axvline(wl_res, color='r', linestyle='--', alpha=0.7, label=f'Resonance: {wl_res:.4f} μm')
        ax1.set_ylabel('Transmission', fontsize=12)
        ax1.set_title(f'1D Photonic Crystal Cavity (N={config.num_pairs} pairs, Q≈{Q_est:.0f})', fontsize=13)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([1e-8, 2])
        
        ax2.plot(wavelengths, reflection, 'r-', linewidth=1.5, label='Reflection')
        ax2.set_xlabel('Wavelength (μm)', fontsize=12)
        ax2.set_ylabel('Reflection', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        fig_dir = Path("../../results/figures")
        fig_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_dir / "tmm_cavity_spectrum.png", dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {fig_dir / 'tmm_cavity_spectrum.png'}")
        
    except ImportError:
        print("Matplotlib not available, skipping plot.")
