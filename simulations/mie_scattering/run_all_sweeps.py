"""
Batch runner for resolution and PML sweeps
==========================================
Run all remaining simulations for Scenario 1.
"""

import sys
sys.path.insert(0, '.')
from mie_transmission import SimConfig, compute_transmission_spectrum
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

output_base = Path('../../results/mie_transmission')

# ============================================================================
# RESOLUTION SWEEP - Complete missing runs
# ============================================================================

print("\n" + "="*70)
print("RESOLUTION SWEEP - SUBPIXEL ON")
print("="*70)

for res in [80]:  # Only missing one
    print(f"\n>>> Resolution {res} px/μm - Subpixel ON")
    config = SimConfig(resolution=res, use_subpixel=True)
    compute_transmission_spectrum(config, output_base / 'subpixel_on')

print("\n" + "="*70)
print("RESOLUTION SWEEP - SUBPIXEL OFF")
print("="*70)

for res in [10, 20, 40, 80]:
    print(f"\n>>> Resolution {res} px/μm - Subpixel OFF")
    config = SimConfig(resolution=res, use_subpixel=False)
    compute_transmission_spectrum(config, output_base / 'subpixel_off')

# ============================================================================
# PML SWEEP
# ============================================================================

print("\n" + "="*70)
print("PML THICKNESS SWEEP")
print("="*70)

pml_thicknesses = [0.5, 1.0, 2.0]  # in wavelengths (μm)

for pml in pml_thicknesses:
    print(f"\n>>> PML thickness = {pml} μm")
    config = SimConfig(
        resolution=40,  # Fixed resolution
        pml_thickness=pml,
        use_subpixel=True,
    )
    compute_transmission_spectrum(config, output_base / 'pml_sweep')

print("\n" + "="*70)
print("ALL SWEEPS COMPLETE!")
print("="*70)
