# DeltaX Galaxy Engine

This repository contains the complete analysis pipeline used to reproduce the results presented in:

**“A Fixed-Parameter Baryonic Mapping for Rotation-Curve Structure Across the SPARC Sample”**  
Christopher P. B. Smolen  
Submitted to *Monthly Notices of the Royal Astronomical Society*

---

## Overview

This repository implements the **ΔX (Delta-X) mapping**, a **purely empirical, fixed-parameter relation** that links directly observed baryonic structure to the *shape* of galaxy rotation curves.

The mapping predicts the dimensionless structural ratio

ΔX(r) = V_obs(r) / V_lum(r)

from three directly observed baryonic descriptors: mass buildup, geometric scale, and localized structural variation.

**No dynamical theory is assumed.**  
**No per-galaxy parameters are introduced.**  
**All quantities are evaluated in native SPARC units, without normalization or rescaling.**

When applied uniformly to all **175 late-type galaxies in the SPARC sample**, the ΔX mapping reproduces rotation-curve structure with high consistency using a **single global set of fixed exponents**.

This repository provides the *exact code, numerical constants, and procedures* used in the manuscript, enabling full independent reproduction of all reported results.

---

## Scientific Scope (Important)

- The ΔX relation is **empirical**, not a theory of gravity.
- The code does **not** implement dark-matter halos, MOND, or modified field equations.
- The mapping is intended as a **quantitative structural benchmark** for galaxy dynamics.
- Interpretation beyond empirical reproducibility lies outside the scope of the paper.

---

## Installation

```bash
git clone https://github.com/AXVIAM/deltax_galaxy_engine.git
cd deltax_galaxy_engine
pip install -r requirements.txt
```

Python 3.8 or later is recommended.

---

## Usage

### Run the ΔX mapping on the full SPARC LTG sample

```bash
python deltax_galaxy_engine/cli.py /path/to/Rotmod_LTG --verbose
```

This computes:
- ΔX(r) for each galaxy
- Predicted structural ratios
- Corresponding velocity predictions
- Curvature-based agreement metrics

---

### Run on a single galaxy (inspection/debugging)

```bash
python -m deltax_galaxy_engine.cli /path/to/Rotmod_LTG/NGC3198_rotmod.dat --verbose
```

---

### Velocity–structure identity check

```bash
python dx_velocity_identity_check.py all_data_dx_Rotmod_LTG.csv dx_Rotmod_LTG_summary.csv
```

This verifies the identity

V_pred(r) = ΔX(r) · V_lum(r),

confirming that velocity agreement follows directly from structural agreement and is **not an independent fit**.

---

## Early-Type Galaxy (ETG) Extension (Exploratory)

The repository includes exploratory scripts that invert the ΔX relation to reconstruct the structural invariant I(r) in **early-type galaxies**, where photometric information is sparse.

```bash
python deltax_galaxy_engine/predict_I_entropy.py /path/to/ETG_photometry_folder
```

and

```bash
python deltax_galaxy_engine/validate_entropy_reconstruction.py
```

These tools are provided as **proofs of concept only** and are explicitly identified as preliminary in the manuscript.

---

## Output Files

Typical outputs include:

- `dx_Rotmod_LTG_summary.csv`  
  Per-galaxy summary of ΔX performance and curvature RMSE.

- `all_data_dx_Rotmod_LTG.csv`  
  Per-radius ΔX values, predicted velocities, and curvature diagnostics.

- `dx_entropy_validation.csv`  
  Validation of entropy-based inversion for ETGs.

All outputs are written to the working directory unless configured otherwise.

---

## Repository Structure

```
deltax_galaxy_engine/
├── cli.py                       # Main analysis entry point
├── dx_math.py                   # Core ΔX equation
├── curvature.py                 # Curvature RMSE metrics
├── anchoring.py                 # Affine comparison frame
├── ix_entropy.py                # Structural invariant construction
├── inverse_entropy.py           # ETG inversion logic
├── predict_I_entropy.py         # ETG prediction pipeline
├── metrics.py                   # Evaluation metrics
├── io.py                        # Data I/O utilities
├── utils.py                     # Shared helpers
├── validate_entropy_reconstruction.py
```

---

## Reproducibility and Archiving

A frozen snapshot of this repository corresponding to the manuscript is archived on Zenodo:

https://doi.org/10.5281/zenodo.XXXXXXX

This snapshot contains the exact code state, numerical constants, and scripts used to generate all tables and figures in the paper.

---

## Citation

If you use this code, please cite:

```bibtex
@article{Smolen2025DeltaX,
  author  = {Smolen, Christopher P. B.},
  title   = {A Fixed-Parameter Baryonic Mapping for Rotation-Curve Structure Across the SPARC Sample},
  journal = {Monthly Notices of the Royal Astronomical Society},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.
