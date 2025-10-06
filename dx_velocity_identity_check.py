import pandas as pd
import numpy as np

# Load per-point and per-galaxy outputs from your engine run
all_df   = pd.read_csv("all_data_dx_Rotmod_LTG.csv")     # point-by-point
sum_df   = pd.read_csv("dx_Rotmod_LTG_summary.csv")      # per-galaxy (includes RMSE_v_direct, RMSE_dx)

# Print available columns once for user awareness
print("all_df columns:", all_df.columns)

def per_gal_identity(df):
    Vobs   = df["Vobs"].values
    Vlum = df["v_lum"].values
    dXobs = df["dx_obs"].values
    dXpred = df["dx_pred_transformed"].values
    Vpred  = df["v_pred"].values  # direct, scale-only anchored variant

    # velocity RMSE (unweighted, to match paper)
    rmse_V = np.sqrt(np.mean((Vobs - Vpred)**2))

    # luminous RMS
    RMS_Vlum = np.sqrt(np.mean(Vlum**2))

    # V^2-weighted ΔX RMSE
    w = Vlum**2
    num = np.sum(w * (dXpred - dXobs)**2)
    den = np.sum(w) if np.sum(w) > 0 else np.nan
    WRMSE_dX_V2 = np.sqrt(num/den) if den == den else np.nan  # guard NaN

    # identity prediction
    rmse_V_hat = RMS_Vlum * WRMSE_dX_V2

    # bound
    Vmax = np.max(Vlum)
    # need unweighted RMSE_dx for the same radii subset
    RMSE_dX = np.sqrt(np.mean((dXpred - dXobs)**2))
    bound_rhs = Vmax * RMSE_dX

    return pd.Series(dict(
        rmse_V=rmse_V,
        rmse_V_hat=rmse_V_hat,
        diff=rmse_V - rmse_V_hat,
        ratio=(rmse_V_hat / rmse_V) if rmse_V>0 else np.nan,
        bound_rhs=bound_rhs
    ))

checks = (all_df
          .groupby("filename", group_keys=False)
          .apply(per_gal_identity)
          .reset_index())

print("sum_df columns:", sum_df.columns)

# Detect velocity RMSE column dynamically
possible_rmse_cols = ["RMSE_v_int_lsq", "RMSE_v_affine", "RMSE_v_int_beta"]
rmse_col_found = None
for col in possible_rmse_cols:
    if col in sum_df.columns:
        rmse_col_found = col
        break
if rmse_col_found is None:
    raise KeyError(f"None of the expected RMSE_v columns found in sum_df. Available columns: {list(sum_df.columns)}")

# Merge using detected column and rename to RMSE_v_summary for consistency
checks = checks.merge(sum_df[["filename", rmse_col_found, "RMSE_dx"]], on="filename", how="left")
checks = checks.rename(columns={rmse_col_found: "RMSE_v_summary"})

# Quick summary
print("Median |rmse_V - rmse_V_hat|:", np.median(np.abs(checks["diff"])))

# Additional reporting and exports
# Fraction satisfying velocity bound (rmse_V <= bound_rhs)
frac_satisfy_bound = np.mean(checks["rmse_V"] <= checks["bound_rhs"])
print(f"Fraction satisfying velocity bound (rmse_V <= bound_rhs): {frac_satisfy_bound:.3f}")

# Fraction within ±5% of the identity (|diff| <= 0.05 * rmse_V)
frac_within_5pct = np.mean(np.abs(checks["diff"]) <= 0.05 * checks["rmse_V"])
print(f"Fraction within ±5% of identity: {frac_within_5pct:.3f}")

# Extract outliers where |diff| > 0.10 * rmse_V
outliers = checks[np.abs(checks["diff"]) > 0.10 * checks["rmse_V"]]
outliers.to_csv("identity_outliers.csv", index=False)
print(f"Saved {len(outliers)} outliers to identity_outliers.csv")

# Save scatter data with selected columns
scatter_cols = ["filename", "rmse_V", "rmse_V_hat", "RMSE_v_summary", "RMSE_dx"]
checks[scatter_cols].to_csv("identity_scatter.csv", index=False)
print("Saved scatter data to identity_scatter.csv")