import numpy as np, pandas as pd

def build_lsq_summary(all_data: pd.DataFrame, sigma_floor_kms: float = 0.0, inner_trim_frac: float = 0.0) -> pd.DataFrame:
    if all_data is None or all_data.empty or ('v_pred_int' not in all_data.columns):
        return pd.DataFrame(columns=['filename','chi2_v_int_lsq_reduced','RMSE_v_int_lsq','n_vel_pts'])
    def _per_gal(grp: pd.DataFrame):
        r   = grp['r'].to_numpy(dtype=float)
        V   = grp['Vobs'].to_numpy(dtype=float)
        E   = grp['errV'].to_numpy(dtype=float)
        Vlsq= grp['v_pred_int'].to_numpy(dtype=float)
        rs  = float(grp['r_s'].iloc[0]) if 'r_s' in grp.columns else np.nan
        if np.any(np.isfinite(E)) and float(sigma_floor_kms) > 0:
            Eeff = np.sqrt(E**2 + float(sigma_floor_kms)**2)
        else:
            Eeff = E
        m = np.isfinite(r) & np.isfinite(V) & np.isfinite(Eeff) & (Eeff>0) & np.isfinite(Vlsq)
        if float(inner_trim_frac) > 0 and np.isfinite(rs) and rs > 0:
            m &= (r >= float(inner_trim_frac) * rs)
        if np.sum(m) < 3:
            return pd.Series({'chi2_v_int_lsq_reduced': np.nan, 'RMSE_v_int_lsq': np.nan, 'n_vel_pts': int(np.sum(m))})
        w = 1.0/(Eeff[m]**2)
        resid = V[m] - Vlsq[m]
        dof = max(int(np.sum(m)) - 1, 1)
        chi2_red = float(np.sum(w * resid**2) / dof)
        rmse = float(np.sqrt(np.mean(resid**2)))
        return pd.Series({'chi2_v_int_lsq_reduced': chi2_red, 'RMSE_v_int_lsq': rmse, 'n_vel_pts': int(np.sum(m))})
    return all_data.groupby('filename').apply(_per_gal).reset_index()

def build_direct_velocity_summary(all_data: pd.DataFrame, sigma_floor_kms: float = 0.0, inner_trim_frac: float = 0.0) -> pd.DataFrame:
    if all_data is None or all_data.empty or ('v_pred' not in all_data.columns):
        return pd.DataFrame(columns=['filename','chi2_v_direct_reduced','RMSE_v_direct','n_vel_pts'])
    def _per_gal(grp: pd.DataFrame):
        r   = grp['r'].to_numpy(dtype=float)
        V   = grp['Vobs'].to_numpy(dtype=float)
        E   = grp['errV'].to_numpy(dtype=float)
        Vdir= grp['v_pred'].to_numpy(dtype=float)
        rs  = float(grp['r_s'].iloc[0]) if 'r_s' in grp.columns else np.nan
        if np.any(np.isfinite(E)) and float(sigma_floor_kms) > 0:
            Eeff = np.sqrt(E**2 + float(sigma_floor_kms)**2)
        else:
            Eeff = E
        m = np.isfinite(r) & np.isfinite(V) & np.isfinite(Eeff) & (Eeff>0) & np.isfinite(Vdir)
        if float(inner_trim_frac) > 0 and np.isfinite(rs) and rs > 0:
            m &= (r >= float(inner_trim_frac) * rs)
        if np.sum(m) < 3:
            return pd.Series({'chi2_v_direct_reduced': np.nan, 'RMSE_v_direct': np.nan, 'n_vel_pts': int(np.sum(m))})
        w = 1.0/(Eeff[m]**2)
        resid = V[m] - Vdir[m]
        dof = max(int(np.sum(m)) - 1, 1)
        chi2_red = float(np.sum(w * resid**2) / dof)
        rmse = float(np.sqrt(np.mean(resid**2)))
        return pd.Series({'chi2_v_direct_reduced': chi2_red, 'RMSE_v_direct': rmse, 'n_vel_pts': int(np.sum(m))})
    return all_data.groupby('filename').apply(_per_gal).reset_index()

def build_beta_on_vint_summary(
    all_data: pd.DataFrame,
    sigma_floor_kms: float = 0.0,
    inner_trim_frac: float = 0.0,
    beta_bounds=(0.75, 1.1)  # loosen lower bound; set to None to disable
) -> pd.DataFrame:
    """
    Per-galaxy scalar β on LSQ-anchored integrated velocity (v_pred_int):
    minimize Σ w_i [Vobs_i − β * v_pred_int_i]^2 with w_i = 1/(errV_i^2 + sigma_floor^2),
    optional inner trim (r ≥ inner_trim_frac · r_s), and optional bounds on β.
    Returns: filename, beta_int, chi2_v_int_beta_reduced, RMSE_v_int_beta, n_beta_pts
    """
    if all_data is None or all_data.empty or ('v_pred_int' not in all_data.columns):
        return pd.DataFrame(columns=[
            'filename','beta_int','chi2_v_int_beta_reduced','RMSE_v_int_beta','n_beta_pts'
        ])

    def _per_gal(grp: pd.DataFrame):
        r    = grp['r'].to_numpy(float)
        V    = grp['Vobs'].to_numpy(float)
        E    = grp['errV'].to_numpy(float)
        Vint = grp['v_pred_int'].to_numpy(float)
        rs   = float(grp['r_s'].iloc[0]) if 'r_s' in grp.columns else np.nan

        # effective sigma with floor
        if np.any(np.isfinite(E)) and float(sigma_floor_kms) > 0:
            Eeff = np.sqrt(E**2 + float(sigma_floor_kms)**2)
        else:
            Eeff = E

        m = np.isfinite(r) & np.isfinite(V) & np.isfinite(Vint) & np.isfinite(Eeff) & (Eeff > 0)
        if float(inner_trim_frac) > 0 and np.isfinite(rs) and rs > 0:
            m &= (r >= float(inner_trim_frac) * rs)
        n = int(np.sum(m))
        if n < 3:
            return pd.Series({'beta_int': np.nan, 'chi2_v_int_beta_reduced': np.nan, 'RMSE_v_int_beta': np.nan, 'n_beta_pts': n})

        w = 1.0 / (Eeff[m]**2)
        # Closed-form weighted LSQ for scalar β
        num = float(np.sum(w * Vint[m] * V[m]))
        den = float(np.sum(w * Vint[m]**2))
        beta = (num / den) if den > 0 else np.nan

        # Optional clamp
        if np.isfinite(beta) and (beta_bounds is not None):
            lo, hi = beta_bounds
            beta = float(np.clip(beta, lo, hi))

        Vfit  = beta * Vint[m]
        resid = V[m] - Vfit
        dof = max(n - 1, 1)
        chi2_red = float(np.sum(w * resid**2) / dof)
        rmse = float(np.sqrt(np.mean(resid**2)))

        return pd.Series({
            'beta_int': beta,
            'chi2_v_int_beta_reduced': chi2_red,
            'RMSE_v_int_beta': rmse,
            'n_beta_pts': n
        })

    return all_data.groupby('filename').apply(_per_gal).reset_index()

def analyze_beta_vs_properties(all_data: pd.DataFrame, outdir: str):
    import os, matplotlib.pyplot as plt
    results = []
    for fname, g in all_data.groupby('filename'):
        Vobs = g['Vobs'].values
        v_pred = g['v_pred'].values if 'v_pred' in g.columns else None
        errV = g['errV'].values if 'errV' in g.columns else None
        if v_pred is None or errV is None:
            beta = np.nan
        else:
            m = np.isfinite(Vobs) & np.isfinite(v_pred) & np.isfinite(errV) & (errV > 0)
            if not np.any(m): beta = np.nan
            else:
                w = 1.0/(errV[m]**2)
                num = np.sum(w * v_pred[m] * Vobs[m]); den = np.sum(w * (v_pred[m]**2))
                beta = float(num/den) if den>0 else np.nan
        r_s = float(g['r_s'].iloc[0]) if 'r_s' in g.columns else np.nan
        med_d = float(g['median_damping'].iloc[0]) if 'median_damping' in g.columns else np.nan
        med_f = float(g['median_feedback'].iloc[0]) if 'median_feedback' in g.columns else np.nan
        lsf   = float(g['lum_scale_factor'].iloc[0]) if 'lum_scale_factor' in g.columns else np.nan
        results.append({'filename': fname, 'beta': beta, 'r_s': r_s, 'median_damping': med_d, 'median_feedback': med_f, 'lum_scale_factor': lsf})
    df = pd.DataFrame(results)
    csv_path = os.path.join(outdir, 'beta_vs_properties.csv')
    df.to_csv(csv_path, index=False)
    fig, axarr = plt.subplots(2,2, figsize=(12,10))
    ax = axarr.flat
    ax[0].scatter(df['r_s'], df['beta']); ax[0].set_xlabel('r_s'); ax[0].set_ylabel('beta'); ax[0].set_title('beta vs r_s')
    ax[1].scatter(df['median_damping'], df['beta']); ax[1].set_xlabel('median_damping'); ax[1].set_ylabel('beta'); ax[1].set_title('beta vs median_damping')
    ax[2].scatter(df['median_feedback'], df['beta']); ax[2].set_xlabel('median_feedback'); ax[2].set_ylabel('beta'); ax[2].set_title('beta vs median_feedback')
    ax[3].scatter(df['lum_scale_factor'], df['beta']); ax[3].set_xlabel('lum_scale_factor'); ax[3].set_ylabel('beta'); ax[3].set_title('beta vs lum_scale_factor')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'beta_vs_properties.png'))
    plt.close(fig)