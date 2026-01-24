"""
Mie Theory Reference Calculator
===============================
Analytical solutions for electromagnetic scattering by cylinders and spheres.

This module provides exact analytical solutions to validate FDTD simulations.

References:
- Bohren & Huffman, "Absorption and Scattering of Light by Small Particles" (1983)
- van de Hulst, "Light Scattering by Small Particles" (1957)
"""

import numpy as np
from scipy.special import jv, jvp, hankel1, h1vp
from scipy.special import spherical_jn, spherical_yn
from typing import Tuple, Optional
import matplotlib.pyplot as plt


# =============================================================================
# 2D MIE THEORY (Infinite Cylinder)
# =============================================================================

class MieCylinder2D:
    """
    Analytical Mie theory for 2D scattering by an infinite dielectric cylinder.
    
    For TM polarization (E || z-axis), the scattering coefficients are:
    
        a_n = [m·J_n(mx)·J_n'(x) - J_n(x)·J_n'(mx)] /
              [m·J_n(mx)·H_n^(1)'(x) - H_n^(1)(x)·J_n'(mx)]
    
    For TE polarization (H || z-axis):
    
        b_n = [J_n(mx)·J_n'(x) - m·J_n(x)·J_n'(mx)] /
              [J_n(mx)·H_n^(1)'(x) - m·H_n^(1)(x)·J_n'(mx)]
    
    where:
        m = n_cylinder / n_background (relative refractive index)
        x = k·a = 2π·a/λ (size parameter)
        a = cylinder radius
    """
    
    def __init__(
        self,
        radius: float,
        n_cylinder: float,
        n_background: float = 1.0,
        n_terms: int = 30
    ):
        """
        Initialize Mie cylinder calculator.
        
        Parameters
        ----------
        radius : float
            Cylinder radius (same units as wavelength)
        n_cylinder : float
            Refractive index of cylinder
        n_background : float
            Refractive index of background medium
        n_terms : int
            Number of terms in series expansion
        """
        self.radius = radius
        self.n_cylinder = n_cylinder
        self.n_background = n_background
        self.n_terms = n_terms
        self.m = n_cylinder / n_background
        
    def size_parameter(self, wavelength: float) -> float:
        """Calculate size parameter x = k·a = 2π·a/λ."""
        return 2 * np.pi * self.radius / wavelength
    
    def coefficients_tm(self, wavelength: float) -> np.ndarray:
        """
        Calculate TM scattering coefficients a_n.
        
        Parameters
        ----------
        wavelength : float
            Wavelength in background medium
            
        Returns
        -------
        a_n : ndarray
            Complex scattering coefficients for n = 0, 1, ..., n_terms-1
        """
        x = self.size_parameter(wavelength)
        mx = self.m * x
        
        a_n = np.zeros(self.n_terms, dtype=complex)
        
        for n in range(self.n_terms):
            Jn_x = jv(n, x)
            Jn_mx = jv(n, mx)
            Jnp_x = jvp(n, x)
            Jnp_mx = jvp(n, mx)
            Hn_x = hankel1(n, x)
            Hnp_x = h1vp(n, x)
            
            num = self.m * Jn_mx * Jnp_x - Jn_x * Jnp_mx
            den = self.m * Jn_mx * Hnp_x - Hn_x * Jnp_mx
            
            a_n[n] = num / den
            
        return a_n
    
    def coefficients_te(self, wavelength: float) -> np.ndarray:
        """
        Calculate TE scattering coefficients b_n.
        
        Parameters
        ----------
        wavelength : float
            Wavelength in background medium
            
        Returns
        -------
        b_n : ndarray
            Complex scattering coefficients for n = 0, 1, ..., n_terms-1
        """
        x = self.size_parameter(wavelength)
        mx = self.m * x
        
        b_n = np.zeros(self.n_terms, dtype=complex)
        
        for n in range(self.n_terms):
            Jn_x = jv(n, x)
            Jn_mx = jv(n, mx)
            Jnp_x = jvp(n, x)
            Jnp_mx = jvp(n, mx)
            Hn_x = hankel1(n, x)
            Hnp_x = h1vp(n, x)
            
            num = Jn_mx * Jnp_x - self.m * Jn_x * Jnp_mx
            den = Jn_mx * Hnp_x - self.m * Hn_x * Jnp_mx
            
            b_n[n] = num / den
            
        return b_n
    
    def scattering_efficiency(self, wavelength: float, polarization: str = 'TM') -> float:
        """
        Calculate scattering efficiency Q_sca = σ_sca / (2a).
        
        For 2D:
            Q_sca = (2/x) · Σ ε_n |a_n|²
        
        where ε_0 = 1, ε_n = 2 for n > 0.
        
        Parameters
        ----------
        wavelength : float
            Wavelength in background medium
        polarization : str
            'TM' or 'TE'
            
        Returns
        -------
        Q_sca : float
            Scattering efficiency (dimensionless)
        """
        x = self.size_parameter(wavelength)
        
        if polarization.upper() == 'TM':
            coeff = self.coefficients_tm(wavelength)
        else:
            coeff = self.coefficients_te(wavelength)
        
        # Neumann factor
        epsilon_n = np.ones(self.n_terms)
        epsilon_n[1:] = 2
        
        Q_sca = (2.0 / x) * np.sum(epsilon_n * np.abs(coeff)**2)
        
        return Q_sca
    
    def extinction_efficiency(self, wavelength: float, polarization: str = 'TM') -> float:
        """
        Calculate extinction efficiency Q_ext = σ_ext / (2a).
        
        For 2D:
            Q_ext = (2/x) · Re[Σ ε_n a_n]
        
        Parameters
        ----------
        wavelength : float
            Wavelength in background medium
        polarization : str
            'TM' or 'TE'
            
        Returns
        -------
        Q_ext : float
            Extinction efficiency (dimensionless)
        """
        x = self.size_parameter(wavelength)
        
        if polarization.upper() == 'TM':
            coeff = self.coefficients_tm(wavelength)
        else:
            coeff = self.coefficients_te(wavelength)
        
        epsilon_n = np.ones(self.n_terms)
        epsilon_n[1:] = 2
        
        Q_ext = (2.0 / x) * np.real(np.sum(epsilon_n * coeff))
        
        return Q_ext
    
    def scattering_cross_section(self, wavelength: float, polarization: str = 'TM') -> float:
        """
        Calculate scattering cross-section (width for 2D).
        
        σ_sca = Q_sca · (2a)
        
        Parameters
        ----------
        wavelength : float
            Wavelength in background medium
        polarization : str
            'TM' or 'TE'
            
        Returns
        -------
        sigma_sca : float
            Scattering cross-section (same units as radius)
        """
        Q_sca = self.scattering_efficiency(wavelength, polarization)
        return Q_sca * 2 * self.radius
    
    def extinction_cross_section(self, wavelength: float, polarization: str = 'TM') -> float:
        """
        Calculate extinction cross-section.
        
        σ_ext = Q_ext · (2a)
        
        Parameters
        ----------
        wavelength : float
            Wavelength in background medium
        polarization : str
            'TM' or 'TE'
            
        Returns
        -------
        sigma_ext : float
            Extinction cross-section (same units as radius)
        """
        Q_ext = self.extinction_efficiency(wavelength, polarization)
        return Q_ext * 2 * self.radius
    
    def differential_scattering(
        self,
        wavelength: float,
        angles: np.ndarray,
        polarization: str = 'TM'
    ) -> np.ndarray:
        """
        Calculate differential scattering cross-section dσ/dθ.
        
        Parameters
        ----------
        wavelength : float
            Wavelength in background medium
        angles : ndarray
            Scattering angles in radians
        polarization : str
            'TM' or 'TE'
            
        Returns
        -------
        dsigma : ndarray
            Differential scattering cross-section at each angle
        """
        x = self.size_parameter(wavelength)
        k = 2 * np.pi / wavelength
        
        if polarization.upper() == 'TM':
            coeff = self.coefficients_tm(wavelength)
        else:
            coeff = self.coefficients_te(wavelength)
        
        # Far-field amplitude
        f_theta = np.zeros_like(angles, dtype=complex)
        
        for n in range(self.n_terms):
            epsilon_n = 1 if n == 0 else 2
            f_theta += epsilon_n * coeff[n] * np.cos(n * angles)
        
        # Differential cross-section
        dsigma = (2.0 / (np.pi * k)) * np.abs(f_theta)**2
        
        return dsigma


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_spectrum(
    radius: float,
    n_cylinder: float,
    wavelengths: np.ndarray,
    n_background: float = 1.0,
    polarization: str = 'TM'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute scattering and extinction spectra over a wavelength range.
    
    Parameters
    ----------
    radius : float
        Cylinder radius
    n_cylinder : float
        Cylinder refractive index
    wavelengths : ndarray
        Array of wavelengths
    n_background : float
        Background refractive index
    polarization : str
        'TM' or 'TE'
        
    Returns
    -------
    Q_sca : ndarray
        Scattering efficiency spectrum
    Q_ext : ndarray
        Extinction efficiency spectrum
    """
    mie = MieCylinder2D(radius, n_cylinder, n_background)
    
    Q_sca = np.array([mie.scattering_efficiency(wl, polarization) for wl in wavelengths])
    Q_ext = np.array([mie.extinction_efficiency(wl, polarization) for wl in wavelengths])
    
    return Q_sca, Q_ext


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Test case: dielectric cylinder with n=2, radius=0.5 μm
    
    radius = 0.5  # μm
    n_cyl = 2.0
    n_bg = 1.0
    
    print("="*60)
    print("Mie Theory Reference Calculator - Test")
    print("="*60)
    print(f"Cylinder radius: {radius} μm")
    print(f"Cylinder index: {n_cyl}")
    print(f"Background index: {n_bg}")
    print("="*60)
    
    mie = MieCylinder2D(radius, n_cyl, n_bg)
    
    # Single wavelength test
    wavelength = 1.0  # μm
    x = mie.size_parameter(wavelength)
    
    print(f"\nWavelength: {wavelength} μm")
    print(f"Size parameter x = {x:.4f}")
    print(f"Q_sca (TM) = {mie.scattering_efficiency(wavelength, 'TM'):.6f}")
    print(f"Q_ext (TM) = {mie.extinction_efficiency(wavelength, 'TM'):.6f}")
    print(f"σ_sca (TM) = {mie.scattering_cross_section(wavelength, 'TM'):.6f} μm")
    print(f"σ_ext (TM) = {mie.extinction_cross_section(wavelength, 'TM'):.6f} μm")
    
    # Spectrum
    wavelengths = np.linspace(0.4, 2.0, 100)
    Q_sca, Q_ext = compute_spectrum(radius, n_cyl, wavelengths)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, Q_sca, 'b-', label='$Q_{sca}$', linewidth=2)
    plt.plot(wavelengths, Q_ext, 'r--', label='$Q_{ext}$', linewidth=2)
    plt.xlabel('Wavelength (μm)', fontsize=12)
    plt.ylabel('Efficiency', fontsize=12)
    plt.title(f'Mie Scattering: Cylinder r={radius} μm, n={n_cyl}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('mie_spectrum_test.png', dpi=150)
    plt.show()
    
    print("\nSpectrum plot saved to mie_spectrum_test.png")
