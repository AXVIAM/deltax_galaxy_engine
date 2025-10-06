import numpy as np, pandas as pd
from .utils import moving_average

def compute_kappa_from_velocity(r, V, smooth_win=5):
    r = np.asarray(r, dtype=float); V = np.asarray(V, dtype=float)
    if r.size == 0 or V.size != r.size:
        return np.zeros_like(r, dtype=float)
    V_s = moving_average(V, smooth_win)
    with np.errstate(invalid='ignore'):
        dV  = np.gradient(V_s, r)
        d2V = np.gradient(dV,  r)
    kappa = np.abs(d2V)
    return np.where(np.isfinite(kappa), kappa, 0.0)

def write_curvature_benchmark(all_data: pd.DataFrame, outdir: str, cv_split: float = 0.0, seed: int = 0):
    import numpy as _np, os
    rows = []
    filenames = sorted(set(all_data['filename'].astype(str)))
    rng = _np.random.default_rng(seed)
    n_test = int(_np.floor(len(filenames) * max(0.0, min(1.0, float(cv_split)))))
    test_set = set(rng.choice(filenames, size=n_test, replace=False).tolist()) if n_test > 0 else set()

    for fname, grp in all_data.groupby('filename'):
        r = grp['r'].to_numpy(float)
        V = grp['Vobs'].to_numpy(float)
        dxo = grp['dx_obs'].to_numpy(float) if 'dx_obs' in grp.columns else None
        dxp = grp['dx_pred_transformed'].to_numpy(float) if 'dx_pred_transformed' in grp.columns else None
        rs  = float(grp['r_s'].iloc[0]) if 'r_s' in grp.columns else _np.nan
        if r.size == 0 or V.size != r.size or dxo is None or dxp is None or dxo.size != r.size or dxp.size != r.size:
            continue
        with _np.errstate(divide='ignore', invalid='ignore'):
            Vlum = _np.where(dxo > 0, V / dxo, _np.nan)
        k_obs = compute_kappa_from_velocity(r, V, smooth_win=5)
        rs_eff = rs if (_np.isfinite(rs) and rs>0) else max(0.5*float(_np.nanmax(r)), 1e-6)
        k_pred = dxp * (Vlum / (rs_eff**2 + 1e-12))
        m = _np.isfinite(k_obs) & _np.isfinite(k_pred)
        if not _np.any(m): continue
        rmse = float(_np.sqrt(_np.mean((k_obs[m] - k_pred[m])**2)))
        mae  = float(_np.mean(_np.abs(k_obs[m] - k_pred[m])))
        rows.append({'filename': str(fname), 'RMSE_kappa': rmse, 'MAE_kappa': mae, 'r_s': rs, 'n_points': int(_np.sum(m)),
                     'split': ('test' if str(fname) in test_set else 'train') if n_test > 0 else 'all'})
    dfk = pd.DataFrame(rows)
    if not dfk.empty:
        dfk = dfk.sort_values(['split','RMSE_kappa','MAE_kappa'], na_position='last')
    out_path = os.path.join(outdir, 'curvature_benchmark_summary.csv')
    dfk.to_csv(out_path, index=False)
    return out_path