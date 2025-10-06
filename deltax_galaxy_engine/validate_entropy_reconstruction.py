import pandas as pd
import numpy as np
import os

# ΔX equation constants — fixed global exponents for the ΔX model
a = np.pi / 5
b = np.pi / 3
c = np.pi / 4
N = 7 * np.pi / 12

# Load the CSV file that contains I_predicted and dx_obs
CSV_PATH = "dx_Rotmod_ETG_outputs/all_data_dx_Rotmod_ETG.csv"
df = pd.read_csv("/Users/axviam/deltax_galaxy_engine/dx_Rotmod_ETG_outputs/all_data_dx_Rotmod_ETG.csv")

# Drop rows missing essential columns
df_valid = df.dropna(subset=["M", "D", "I_predicted", "dx_obs"])

# Recompute ΔX using predicted I(r)
df_valid["dx_reconstructed"] = a * df_valid["M"]**b * df_valid["D"]**c * df_valid["I_predicted"]**N

# Compute residuals
df_valid["dx_residual"] = df_valid["dx_obs"] - df_valid["dx_reconstructed"]
df_valid["dx_abs_error"] = np.abs(df_valid["dx_residual"])

# Summary error metrics per galaxy
summary = df_valid.groupby("filename").agg({
    "dx_residual": ["mean", "std"],
    "dx_abs_error": ["mean", "max"],
    "dx_obs": "count"
}).reset_index()

summary.columns = ["filename", "residual_mean", "residual_std", "mae", "max_abs_error", "n_points"]

# Save outputs
df_valid.to_csv("dx_Rotmod_ETG_outputs/dx_entropy_validation.csv", index=False)
summary.to_csv("dx_Rotmod_ETG_outputs/dx_entropy_summary.csv", index=False)

print("[✓] Validation complete. Outputs written to:")
print("    → dx_Rotmod_ETG_outputs/dx_entropy_validation.csv")
print("    → dx_Rotmod_ETG_outputs/dx_entropy_summary.csv")