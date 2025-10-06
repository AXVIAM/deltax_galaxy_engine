# DeltaX Galaxy Engine

This repository contains the full codebase for reproducing the results presented in the paper:

**"A Universal Geometric Equation for Galaxy Rotation Curves Derived from Modified Einstein Dynamics"**  
by Christopher P. B. Smolen

## Overview

The `deltax_galaxy_engine` implements the ΔX relation — a geometric mapping between luminous structure and galaxy rotation dynamics — derived from a modified Einstein field equation incorporating information-geometry structure.

This engine evaluates the ΔX equation across the SPARC sample (175 late-type galaxies) and includes extension logic for early-type galaxies via entropy-based inversion.

---

## Getting Started

### Installation

```bash
git clone https://github.com/AXVIAM/deltax_galaxy_engine.git
gh repo clone AXVIAM/deltax_galaxy_engine
cd deltax_galaxy_engine
pip install -r requirements.txt
```

Python 3.8+ is recommended.

---

### 🛠 Usage

You can run the engine over a SPARC-format input directory using:

```bash
python -m deltax_galaxy_engine.cli /path/to/Rotmod_LTG --verbose
```

The CLI will:
- Load each galaxy’s photometric and rotation data
- Apply the ΔX equation
- Output RMSE metrics, affine anchoring coefficients, and summary tables

---

## 📂 Repository Structure

```
deltax_galaxy_engine/
├── cli.py                # Command-line entry point
├── core/                 # Core logic for ΔX evaluation
├── utils/                # Helper methods for input/output
├── tests/                # Unit and regression tests
├── examples/             # Example galaxy files and usage
└── requirements.txt
```

---

## 📄 Reproducibility

A Zenodo snapshot of this repository is archived at:  
[DOI: https://doi.org/10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)  
This ensures all results from the publication can be reproduced exactly.

---

## License

This project is licensed under the MIT License — see [`LICENSE`](./LICENSE) for details.

---

## 🧾 Citation

If you use this code, please cite:

```bibtex
@article{Smolen2025,
  author = {Smolen, Christopher P. B.},
  title = {A Universal Geometric Equation for Galaxy Rotation Curves Derived from Modified Einstein Dynamics},
  journal = {Monthly Notices of the Royal Astronomical Society},
  year = {2025},
  note = {In review}
}
```# deltax_galaxy_engine
Code for running the DeltaX equation over ETG &amp; LTG galaxies using the SPARC datasets
