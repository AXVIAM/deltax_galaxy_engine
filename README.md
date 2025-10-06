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

### Usage

To evaluate ΔX for the SPARC LTG sample:

```bash
python deltax_galaxy_engine/cli.py /path/to/Rotmod_LTG --verbose
```

To run entropy inversion and prediction on ETG systems:

```bash
python deltax_galaxy_engine/predict_I_entropy.py /path/to/ETG_photometry_folder
```

Alternatively, you can invoke the module directly if using the optional `__main__.py` entry point:

```bash
python -m deltax_galaxy_engine /path/to/Rotmod_LTG --verbose
```


### 🔄 Additional Commands

To run the engine on a **single galaxy file** for inspection or debugging:

```bash
python -m deltax_galaxy_engine.cli /path/to/Rotmod_LTG/NGC3198_rotmod.dat --verbose
```

To run a **post-analysis velocity identity check** between raw ΔX predictions and affine-anchored outputs:

```bash
python dx_velocity_identity_check.py all_data_dx_Rotmod_LTG.csv dx_Rotmod_LTG_summary.csv
```

This script verifies that the scaled velocity profile $V_{\rm pred} = \Delta X \cdot V_{\rm lum}$ agrees with the affine-anchored profile $V_{\rm pred}^{\rm anc}$ for each galaxy, validating dynamic consistency of the ΔX mapping.

---

### 📤 Output Files

Running the engine on the SPARC LTG dataset (using `cli.py`) typically produces:

- `dx_Rotmod_LTG_summary.csv`  
  Summary of ΔX predictions, affine fit parameters, and success classification per galaxy.

- `all_data_dx_Rotmod_LTG.csv`  
  Per-radius ΔX values, velocity predictions, and curvature metrics across all galaxies.

Entropy modeling and inversion for ETGs using `predict_I_entropy.py` and `inverse_entropy.py` produce:

- `etg_I_predicted.csv`  
  Predicted $I(r)$ structure from photometric input.

- `etg_inversion_results.csv`  
  ΔX profiles derived from entropy inversion on ETG photometric fits.

After prediction and inversion, run:

```bash
python deltax_galaxy_engine/validate_entropy_reconstruction.py
```

This produces:

- `dx_Rotmod_ETG_outputs/dx_entropy_validation.csv`  
  Validation table comparing predicted vs reconstructed $I(r)$ and $\Delta X$.

- `dx_Rotmod_ETG_outputs/dx_entropy_summary.csv`  
  Summary statistics of entropy-based reconstruction accuracy.

All files are written to the working directory unless configured otherwise.

---

## 📂 Repository Structure

```
deltax_galaxy_engine/
├── __init__.py
├── __main__.py                  # Optional entry point
├── cli.py                       # Command-line entry point
├── anchoring.py                 # Affine anchoring methods
├── curvature.py                 # Curvature RMSE calculations
├── dx_math.py                   # Core ΔX equation
├── inverse_entropy.py           # Entropy inversion logic
├── io.py                        # Input/output handlers
├── ix_entropy.py                # Geometric invariant construction
├── metrics.py                   # Evaluation metrics
├── predict_I_entropy.py         # Predictive modeling of I(r)
├── sersic.py                    # Sersic profile support
├── utils.py                     # General utilities
├── validate_entropy_reconstruction.py  # Post-hoc inversion validation
```

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
```
