# FDTD Parameter Sensitivity Analysis in Meep

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Meep](https://img.shields.io/badge/Meep-1.25+-green.svg)](https://meep.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Systematic evaluation of FDTD simulation parameters for electromagnetic wave modeling**

## Authors

**Saïd ECH-CHADI** & **Youness ECHCHADI**  
*Cadi Ayyad University, Marrakech, Morocco*

---

## Abstract

This repository contains the simulation code, analysis scripts, and reproducible workflow for a systematic parameter sensitivity analysis of Finite-Difference Time-Domain (FDTD) simulations using the Meep software package. We investigate how spatial resolution, PML thickness, and subpixel smoothing affect simulation accuracy across three canonical electromagnetic problems.

### Key Findings

- **40 pixels/λ** provides sub-1% accuracy for most applications
- **PML thickness ≥ 0.5λ** is sufficient (thicker PML shows no improvement)
- **Subpixel smoothing** benefits are geometry-dependent

---

## Research Question

*"To what extent can systematic parameter sensitivity analysis in Meep FDTD simulations establish reliable computational guidelines for minimizing numerical dispersion and simulation error in electromagnetic wave modeling?"*

---

## Project Structure

```
meep-fdtd-parameter-study/
├── simulations/
│   ├── mie_scattering/       # Scenario 1: Mie theory validation
│   │   ├── mie_transmission.py
│   │   ├── mie_theory.py     # Analytical reference calculator
│   │   └── run_all_sweeps.py
│   ├── photonic_cavity/      # Scenario 2: 1D photonic crystal cavity
│   │   ├── cavity_meep.py
│   │   └── tmm_reference.py  # Transfer matrix method reference
│   └── waveguide/            # Scenario 3: Slab waveguide transmission
│       └── waveguide_meep.py
├── analysis/
│   └── cross_scenario_analysis.py  # Unified analysis & figure generation
├── results/
│   ├── mie_scattering/       # Raw simulation data
│   ├── photonic_cavity/
│   ├── waveguide/
│   └── figures/              # Generated publication figures
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.8+
- Conda (recommended) or pip
- Linux/macOS or WSL2 on Windows

### Quick Start

```bash
# Clone the repository
git clone https://github.com/[username]/meep-fdtd-parameter-study.git
cd meep-fdtd-parameter-study

# Create conda environment with Meep
conda create -n meep -c conda-forge pymeep
conda activate meep

# Install additional dependencies
pip install numpy scipy matplotlib pandas

# Verify installation
python -c "import meep; print(f'Meep version: {meep.__version__}')"
```

---

## Usage

### Running Individual Simulations

```bash
# Mie Scattering - Resolution sweep
cd simulations/mie_scattering
python mie_transmission.py --resolution 40 --pml 1.0

# Photonic Cavity
cd simulations/photonic_cavity
python cavity_meep.py --resolution 40

# Waveguide
cd simulations/waveguide
python waveguide_meep.py --resolution 40
```

### Running Full Parameter Sweeps

```bash
# Run all Mie scattering sweeps
cd simulations/mie_scattering
python run_all_sweeps.py

# Similar scripts available for other scenarios
```

### Generating Analysis Figures

```bash
cd analysis
python cross_scenario_analysis.py
# Outputs saved to results/figures/
```

---

## Test Scenarios

| Scenario | Description | Reference Solution |
|----------|-------------|-------------------|
| **Mie Scattering** | 2D dielectric cylinder (r=0.5μm, n=2.0) | Analytical Mie theory |
| **Photonic Cavity** | 1D DBR cavity ([HL]⁵ D [LH]⁵) | Transfer matrix method |
| **Waveguide** | 2D slab waveguide (w=0.5μm, n=3.5) | Analytical cutoff |

---

## Parameters Investigated

| Parameter | Values Tested | Key Finding |
|-----------|---------------|-------------|
| Spatial Resolution | 20, 40, 80 px/μm | 40 px/λ optimal for <1% error |
| PML Thickness | 0.5, 1.0, 2.0, 4.0 λ | No effect beyond 0.5λ |
| Subpixel Smoothing | ON / OFF | Geometry-dependent benefit |

---

## Results Summary

### Resolution Convergence

| Scenario | 20 px/μm | 40 px/μm | 80 px/μm |
|----------|----------|----------|----------|
| Mie Extinction | 0.093 | 0.140 | 0.144 |
| Cavity Q-factor | 116 | 136 | 131 |
| Waveguide Trans. | 0.725 | 0.663 | 0.653 |

### Computational Cost

- 40 px/μm achieves optimal accuracy-to-cost ratio
- Under-resolved simulations (20 px/μm) can paradoxically be *slower* due to numerical dispersion

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{echchadi2026fdtd,
  author       = {Ech-Chadi, Saïd and Echchadi, Youness},
  title        = {Systematic Parameter Sensitivity Analysis for FDTD Simulations in Meep},
  year         = {2026},
  institution  = {Cadi Ayyad University},
  howpublished = {\url{https://github.com/[username]/meep-fdtd-parameter-study}}
}
```

---

## References

- Oskooi, A. F., et al. (2010). *MEEP: A flexible free-software package for electromagnetic simulations by the FDTD method*. Computer Physics Communications.
- Taflove, A., & Hagness, S. C. (2005). *Computational Electrodynamics: The Finite-Difference Time-Domain Method*. Artech House.
- Farjadpour, A., et al. (2006). *Improving accuracy by subpixel smoothing in FDTD*. Optics Letters.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Meep Development Team](https://meep.readthedocs.io/) for the excellent FDTD software
- Cadi Ayyad University for institutional support
