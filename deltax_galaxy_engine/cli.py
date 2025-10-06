import os, sys, glob, json, numpy as np, pandas as pd
from datetime import datetime

from .utils import G, git_hash, luminous_velocity, valid_mask_from
from .io import parse_rotmod_file, parse_component_dat, find_etg_triplet, merge_etg_triplet
from .sersic import parse_sersic_table
from .dx_math import default_a, default_b, default_c, default_N, deltaX, deltaX_with_metrics, integrate_velocity_from_deltax
from .anchoring import anchor_velocity_profile, anchor_velocity_scale_only
from .curvature import write_curvature_benchmark
from .metrics import build_lsq_summary, analyze_beta_vs_properties, build_beta_on_vint_summary, build_direct_velocity_summary

# Import inverse entropy solver
from .inverse_entropy import inverse_entropy_solver

# Prefer modular ix_entropy; fall back to a local helper if not present
try:
    from .ix_entropy import derive_Ix_entropy_curvature as derive_Ix
except Exception:
    def derive_Ix_entropy_curvature(r, Vgas, Vdisk, Vbul, SBdisk, entropy_as_prob: bool = False):
        import numpy as _np
        r = _np.asarray(r, dtype=float)
        Vgas = _np.asarray(Vgas, dtype=float)
        Vdisk = _np.asarray(Vdisk, dtype=float)
        Vbul  = _np.asarray(Vbul,  dtype=float)
        SBdisk= _np.asarray(SBdisk, dtype=float)
        v_lum = _np.sqrt(_np.maximum(Vdisk,0)**2 + _np.maximum(Vbul,0)**2 + _np.maximum(Vgas,0)**2)
        dv   = _np.gradient(v_lum, r)
        d2v  = _np.gradient(dv, r)
        curvature = _np.abs(d2v)
        SBp = _np.where(SBdisk > 0, SBdisk, 1e-12)
        p = SBp / (_np.nanmax(SBp) + 1e-12)
        entropy_core = -p * _np.log(_np.clip(p, 1e-12, None))
        dS  = _np.gradient(entropy_core, r)
        d2S = _np.gradient(dS, r)
        entropy_memory = _np.abs(d2S)
        I = _np.sqrt(curvature * entropy_memory)
        I[_np.isnan(I)] = 0.0
        return I
    derive_Ix = derive_Ix_entropy_curvature

def run_dx_single(filename, args, N_value, SERSIC_DB):
    # Parse
    ext = os.path.splitext(filename)[1].lower()
    meta = {}
    if ext in ('.dat', '.txt'):
        has_trip, dpath, bpath = find_etg_triplet(filename)
        if has_trip:
            params = None
            stem_key = os.path.basename(filename)[:-11].strip().lower()
            if SERSIC_DB: params = SERSIC_DB.get(stem_key) or SERSIC_DB.get(stem_key.replace('_',''))
            r, Vobs, errV, Vgas, Vdisk, Vbul, SBdisk, SBbul, meta, _ = merge_etg_triplet(filename, dpath, bpath, params, verbose=args.verbose)
        else:
            r, Vobs, errV, Vgas, Vdisk, Vbul, SBdisk, SBbul = parse_rotmod_file(filename)
    elif ext == '.csv':
        # You can add your CSV parser here if needed; for now reuse rotmod-style columns
        r, Vobs, errV, Vgas, Vdisk, Vbul, SBdisk, SBbul = parse_rotmod_file(filename)
    else:
        raise ValueError(f"Unsupported extension: {ext}")

    # Mass proxy (consistent with your script)
    M_all = (Vdisk**2 + Vbul**2 + Vgas**2) * r / G
    D = r

    # I(x) via entropy–curvature using SBdisk or SBbul depending on dx_method metadata.
    SB_source = SBbul if meta.get("dx_method") == "bulge" else SBdisk
    SB_safe = np.where(np.isfinite(SB_source) & (SB_source > 0), SB_source, 0.0)
    I = derive_Ix(r, Vgas, Vdisk, Vbul, SB_safe)
    if I is None or len(I) == 0:
        raise ValueError(f"No valid I(x) for {filename}")

    # Feed ΔX with UNNORMALIZED inputs matching the legacy solver:
    # M_all = (Vdisk^2 + Vbul^2 + Vgas^2) * r / G,  D = r,  I as above.
    dx_pred, damping_arr, feedback_arr = deltaX_with_metrics(M_all, D, I, default_a, default_b, default_c, N_value)
    # Guard against tiny negative noise and NaNs
    dx_pred = np.where(np.isfinite(dx_pred), dx_pred, 0.0)
    dx_pred = np.clip(dx_pred, 0.0, None)

    v_lum_full = luminous_velocity(Vdisk, Vbul, Vgas)
    with np.errstate(divide='ignore', invalid='ignore'):
        dx_obs_full = Vobs / v_lum_full
    valid_mask = valid_mask_from(v_lum_full, Vobs, min_vlum=getattr(args,'min_vlum',0.0)) & (~np.isnan(dx_obs_full))

    r_valid = r[valid_mask]
    Vobs_valid = Vobs[valid_mask]
    errV_valid = errV[valid_mask]
    SBdisk_valid = SBdisk[valid_mask]
    # If the disc SB has no usable signal but bulge SB does, use bulge SB to seed the exponential scale
    try:
        SBbul_valid = SBbul[valid_mask]
    except Exception:
        SBbul_valid = None
    if (not np.any(np.isfinite(SBdisk_valid) & (SBdisk_valid > 0))) and (SBbul_valid is not None) and np.any(np.isfinite(SBbul_valid) & (SBbul_valid > 0)):
        SBdisk_valid = SBbul_valid
    dx_obs = dx_obs_full[valid_mask]
    dx_pred_valid = dx_pred[valid_mask]
    Vdisk_valid = Vdisk[valid_mask]
    Vbul_valid  = Vbul[valid_mask]
    Vgas_valid  = Vgas[valid_mask]
    v_lum_valid = v_lum_full[valid_mask]
    r_s = fit_r_s_exponential(r_valid, SBdisk_valid)

    # Integrated velocity + anchoring
    Vpred_direct_full = dx_pred * v_lum_full
    Vint_raw_full = integrate_velocity_from_deltax(r, Vpred_direct_full, dx_pred, r_s if np.isfinite(r_s) else np.nan)
    # Use scale_only anchoring instead of full anchor_velocity_profile
    Vanch_full = anchor_velocity_scale_only(
        r, Vint_raw_full, Vobs, errV, v_lum_full, r_s,
        sigma_floor=float(getattr(args, 'sigma_floor_kms', 0.0)),
        inner_trim_frac=float(getattr(args, 'vel_inner_trim_frac', 0.0))
    )
    v_pred_int_valid = Vanch_full[valid_mask]
    v_pred_valid = dx_pred_valid * v_lum_full[valid_mask]

    median_damping = float(np.median(damping_arr[valid_mask])) if np.any(valid_mask) else np.nan
    median_feedback = float(np.median(feedback_arr[valid_mask])) if np.any(valid_mask) else np.nan

    df = pd.DataFrame({
        'r': r_valid, 'Vobs': Vobs_valid, 'errV': errV_valid,
        'dx_obs': dx_obs, 'dx_pred_raw': dx_pred_valid,
        'dx_pred_transformed': dx_pred_valid,
        'v_pred': v_pred_valid, 'v_pred_int': v_pred_int_valid,
        'integrated_velocity': True,
        'anchor_mode': str(getattr(args, 'vel_anchor', 'lsq')).lower(),
        'anchor_pivot_k': float(getattr(args, 'anchor_pivot_k', 2.2)),
        'sigma_floor_kms': float(getattr(args, 'sigma_floor_kms', 0.0)),
        'vel_inner_trim_frac': float(getattr(args, 'vel_inner_trim_frac', 0.0)),
        'anchor_pivot_target': 'Vobs',
        'SBdisk': SBdisk_valid,
        'filename': os.path.basename(filename),
        'r_s': r_s,
        'module': 'legacy_inputs',
        'a': default_a, 'b': default_b, 'c': default_c, 'N': N_value,
        'median_damping': median_damping,
        'median_feedback': median_feedback,
        'Vdisk': Vdisk_valid,
        'Vbul': Vbul_valid,
        'Vgas': Vgas_valid,
        'SBbul': SBbul_valid if SBbul_valid is not None else np.full_like(r_valid, np.nan, dtype=float),
        'v_lum': v_lum_valid,
        'dx_method': meta.get("dx_method", "disk"),
        'M': M_all[valid_mask],
        'D': D[valid_mask],
    })
    df["I_predicted"] = np.nan
    return df

def fit_r_s_exponential(r, sb):
    mask = (sb > 0) & (r > 0)
    r_fit = r[mask]; sb_fit = sb[mask]
    if len(r_fit) < 2: return np.nan
    ln_sb = np.log(sb_fit)
    A = np.vstack([r_fit, np.ones_like(r_fit)]).T
    m, c = np.linalg.lstsq(A, ln_sb, rcond=None)[0]
    return (-1/m) if m < 0 else np.nan

def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='ΔX galaxy engine (modular CLI)')
    parser.add_argument('input_dir')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--paper_lite', action='store_true')
    parser.add_argument('--min_vlum', type=float, default=0.0)
    parser.add_argument('--integrated_velocity', action='store_true')
    parser.add_argument('--vel_anchor', type=str, default='scale_only', choices=['inner+pivot','lsq','scale_only'])
    parser.add_argument('--sigma_floor_kms', type=float, default=0.0)
    parser.add_argument('--vel_inner_trim_frac', type=float, default=0.0)
    parser.add_argument('--anchor_pivot_k', type=float, default=2.2)
    parser.add_argument('--anchor_pivot_target', type=str, default='Vobs')
    parser.add_argument('--sersic_table', type=str, default='')
    parser.add_argument('--out', type=str, default='', help='Optional output directory. If not provided, a folder named dx_<input_basename>_outputs is created in the CWD.')
    parser.add_argument('--inverse_entropy', action='store_true', help='Run inverse entropy solver on ETG galaxies')
    parser.add_argument('--use_predicted_I', action='store_true', help='Use predicted I values from I_predicted_output.csv if available')
    args = parser.parse_args(argv)

    SERSIC_DB = {}
    if args.sersic_table.strip():
        try:
            SERSIC_DB = parse_sersic_table(args.sersic_table.strip())
            if args.verbose: print(f"[Sérsic] Loaded {len(SERSIC_DB)} entries")
        except Exception as e:
            print(f"[Sérsic] Failed to load '{args.sersic_table}': {e}")

    input_path = args.input_dir
    input_is_file = os.path.isfile(input_path)

    # Gather files (supports single file or directory)
    all_files = []
    if input_is_file:
        all_files = [input_path]
    else:
        for pat in ('**/*.dat','**/*.DAT','**/*.csv','**/*.CSV'):
            all_files += glob.glob(os.path.join(input_path, pat), recursive=True)
        # filter out our dx_ outputs
        def is_self_output(path):
            base = os.path.basename(path)
            if '/dx_' in path or os.path.basename(os.path.dirname(path)).startswith('dx_'):
                return True
            if base.startswith(('all_data_dx_','beta_per_galaxy_')): return True
            if base.endswith(('_summary.csv','_summary.csv.gz')): return True
            return False
        all_files = [p for p in all_files if not is_self_output(p)]

    if len(all_files) == 0:
        print(f"No input files found: {input_path}")
        return 2

    # --- Predicted I loading if requested ---
    predicted_I_df = None
    if args.use_predicted_I:
        try:
            I_csv_path = os.path.join(os.getcwd(), "dx_Rotmod_ETG_outputs", "I_predicted_output.csv")
            predicted_I_df = pd.read_csv(I_csv_path)
            if args.verbose:
                print(f"[PredictedI] Loaded {len(predicted_I_df)} I predictions from {I_csv_path}")
        except Exception as e:
            print(f"[PredictedI] Failed to load predicted I data: {e}")
            predicted_I_df = None

    N_value = default_N  # can expose N_scale later
    dfs = []
    for f in all_files:
        try:
            if args.verbose: print(f"Processing: {f}")
            df = run_dx_single(f, args, N_value, SERSIC_DB)
            # --- Insert predicted I values from CSV if requested ---
            if args.use_predicted_I and predicted_I_df is not None and df is not None:
                try:
                    merged = df.merge(
                        predicted_I_df[['filename', 'r', 'I_predicted']],
                        on=['filename', 'r'],
                        how='left',
                        suffixes=('', '_from_csv')
                    )
                    df['I_predicted'] = merged['I_predicted_from_csv'].combine_first(merged['I_predicted'])
                except Exception as e:
                    print(f"[PredictedI] Merge failed for {os.path.basename(f)}: {e}")
            # Inverse entropy solver for ETG galaxies, if requested
            if args.inverse_entropy and df is not None and not df.empty:
                try:
                    # Use only valid points for inverse inference
                    df_valid = df[np.isfinite(df['r']) & np.isfinite(df['Vobs']) & np.isfinite(df['v_lum'])]
                    print(f"[InverseEntropy] {os.path.basename(f)} → valid points: {len(df_valid)}")
                    if len(df_valid) >= 2:
                        r = df_valid['r'].to_numpy()
                        Vobs = df_valid['Vobs'].to_numpy()
                        Vlum = df_valid['v_lum'].to_numpy()
                        Ir_fit, dx_obs, dx_fit, result_meta = inverse_entropy_solver(r, Vobs, Vlum)
                        # Directly assign to the main DataFrame
                        df.loc[df_valid.index, 'I_predicted'] = Ir_fit
                    elif len(df_valid) == 1:
                        print(f"[InverseEntropy] {os.path.basename(f)} has only 1 valid point — skipping.")
                    else:
                        print(f"[InverseEntropy] {os.path.basename(f)} has 0 usable data — skipping.")
                except Exception as e:
                    print(f"[InverseEntropy] {os.path.basename(f)}: {e}")
            if df is not None: dfs.append(df)
        except Exception as e:
            print(f"[Error] {os.path.basename(f)}: {e}")

    all_data = pd.concat(dfs, ignore_index=True) if len(dfs)>0 else pd.DataFrame()
    if args.out.strip():
        outdir = os.path.abspath(args.out.strip())
    else:
        base_in = os.path.basename(input_path.rstrip(os.sep))
        if input_is_file:
            base_in = os.path.splitext(base_in)[0]
        outdir = os.path.join(os.getcwd(), f"dx_{base_in}_outputs")
    os.makedirs(outdir, exist_ok=True)

    base_tag = os.path.basename(input_path.rstrip(os.sep))
    if input_is_file:
        base_tag = os.path.splitext(base_tag)[0]

    all_path = os.path.join(outdir, f"all_data_dx_{base_tag}.csv")
    all_data.to_csv(all_path, index=False)

    # Summary rows (add your existing summary_rows assembly here if you keep it)
    summary_rows = [{'filename': f} for f in sorted(set(all_data['filename']))]
    summary_df = pd.DataFrame(summary_rows)

    # --- Compute and merge curvature (Δx) metrics per galaxy ---
    def _per_gal_curv_metrics(grp: pd.DataFrame):
        import numpy as _np
        # Use transformed dx if present; else raw
        if 'dx_obs' not in grp.columns:
            return pd.Series({'RMSE_dx': _np.nan, 'MAE_dx': _np.nan, 'n_dx_pts': 0})
        dxo = grp['dx_obs'].to_numpy(dtype=float)
        if 'dx_pred_transformed' in grp.columns:
            dxp = grp['dx_pred_transformed'].to_numpy(dtype=float)
        elif 'dx_pred_raw' in grp.columns:
            dxp = grp['dx_pred_raw'].to_numpy(dtype=float)
        else:
            return pd.Series({'RMSE_dx': _np.nan, 'MAE_dx': _np.nan, 'n_dx_pts': 0})
        m = _np.isfinite(dxo) & _np.isfinite(dxp)
        n = int(_np.sum(m))
        resid = dxo[m] - dxp[m]
        rmse = float(_np.sqrt(_np.mean(resid**2))) if n > 0 else _np.nan
        mae  = float(_np.mean(_np.abs(resid))) if n > 0 else _np.nan
        return pd.Series({'RMSE_dx': rmse, 'MAE_dx': mae, 'n_dx_pts': n})

    if not all_data.empty:
        curv_summary = all_data.groupby('filename').apply(_per_gal_curv_metrics).reset_index()
        summary_df = summary_df.merge(curv_summary, on='filename', how='left')

    # Merge LSQ velocity metrics
    lsq_summary = build_lsq_summary(all_data, sigma_floor_kms=args.sigma_floor_kms, inner_trim_frac=args.vel_inner_trim_frac)
    if not lsq_summary.empty:
        summary_df = summary_df.merge(lsq_summary, on='filename', how='left')

    # Merge direct velocity metrics (RMSE_v_direct, chi2_v_direct_reduced, etc)
    direct_summary = build_direct_velocity_summary(all_data, sigma_floor_kms=args.sigma_floor_kms, inner_trim_frac=args.vel_inner_trim_frac)
    if not direct_summary.empty:
        summary_df = summary_df.merge(direct_summary, on='filename', how='left')

    # Per-galaxy scalar β on LSQ-anchored integrated velocity (nudges that keep curvature intact)
    beta_sum = build_beta_on_vint_summary(
        all_data,
        sigma_floor_kms=args.sigma_floor_kms,
        inner_trim_frac=args.vel_inner_trim_frac,
        beta_bounds=(0.9, 1.1)  # ±10% as per journal-safe nudges
    )
    if not beta_sum.empty:
        summary_df = summary_df.merge(beta_sum, on='filename', how='left')

    # --- Per-galaxy affine calibration on direct predictor (preserves curvature) ---
    # Model: Vobs ≈ s * v_pred + alpha * r + beta  (weighted LSQ, sigma floor applied)
    def _per_gal_affine(grp: pd.DataFrame):
        import numpy as _np
        r     = grp['r'].to_numpy(float)
        Vobs  = grp['Vobs'].to_numpy(float)
        errV  = grp['errV'].to_numpy(float)
        if 'v_pred' not in grp.columns:
            return pd.Series({
                's_affine': _np.nan,
                'alpha_affine': _np.nan,
                'beta_affine': _np.nan,
                'RMSE_v_affine': _np.nan,
                'chi2_v_affine_reduced': _np.nan,
                'n_affine_pts': int(len(r))
            })
        vpred = grp['v_pred'].to_numpy(float)
        m = _np.isfinite(r) & _np.isfinite(Vobs) & _np.isfinite(errV) & _np.isfinite(vpred)
        n = int(_np.sum(m))
        if n < 3:
            return pd.Series({
                's_affine': _np.nan,
                'alpha_affine': _np.nan,
                'beta_affine': _np.nan,
                'RMSE_v_affine': _np.nan,
                'chi2_v_affine_reduced': _np.nan,
                'n_affine_pts': n
            })
        r = r[m]; Vobs = Vobs[m]; errV = errV[m]; vpred = vpred[m]
        w = 1.0 / (_np.maximum(errV, 1e-6)**2 + float(getattr(args, 'sigma_floor_kms', 0.0))**2)
        sw = _np.sqrt(w)
        X  = _np.vstack([vpred, r, _np.ones_like(r)]).T
        Xw = X * sw[:, None]
        yw = Vobs * sw
        try:
            theta, *_ = _np.linalg.lstsq(Xw, yw, rcond=None)
            s_hat, a_hat, b_hat = [float(t) for t in theta]
        except Exception:
            s_hat, a_hat, b_hat = _np.nan, 0.0, 0.0
        # Guard against unphysical sign flip; keep s >= 0 (can tighten to [0.9,1.1] if desired)
        s = float(s_hat) if _np.isfinite(s_hat) else np.nan
        if _np.isfinite(s):
            s = max(0.0, s)
        alpha = float(a_hat) if _np.isfinite(a_hat) else 0.0
        beta  = float(b_hat) if _np.isfinite(b_hat) else 0.0
        vfit = s * vpred + alpha * r + beta
        resid = Vobs - vfit
        rmse = float(_np.sqrt(_np.nanmean(resid**2)))
        dof = max(n - 3, 1)
        chi2_red = float(_np.sum(w * resid**2) / dof)
        return pd.Series({
            's_affine': s,
            'alpha_affine': alpha,
            'beta_affine': beta,
            'RMSE_v_affine': rmse,
            'chi2_v_affine_reduced': chi2_red,
            'n_affine_pts': n
        })

    if not all_data.empty:
        affine_summary = all_data.groupby('filename').apply(_per_gal_affine).reset_index()
        summary_df = summary_df.merge(affine_summary, on='filename', how='left')

    # --- ETG-lite metrics: for galaxies with <3 points, report outer-radius diagnostics ---
    def _per_gal_etg_lite(grp: pd.DataFrame):
        import numpy as _np
        result = {
            'delta_dx_outer': _np.nan,
            'dx_ratio_outer': _np.nan,
            'vel_resid_outer_affine': _np.nan,
            'v_scale_ratio_outer': _np.nan,
            'n_outer_pts': int(len(grp))
        }
        if 'dx_obs' not in grp.columns or 'dx_pred_transformed' not in grp.columns:
            return pd.Series(result)
        r = grp['r'].to_numpy(float)
        dxo = grp['dx_obs'].to_numpy(float)
        dxp = grp['dx_pred_transformed'].to_numpy(float)
        Vobs = grp['Vobs'].to_numpy(float)
        vpred = grp['v_pred'].to_numpy(float) if 'v_pred' in grp.columns else None
        errV = grp['errV'].to_numpy(float)
        m = _np.isfinite(r) & _np.isfinite(dxo) & _np.isfinite(dxp) & _np.isfinite(Vobs)
        if vpred is not None:
            m &= _np.isfinite(vpred)
        if _np.sum(m) == 0:
            return pd.Series(result)
        # Take outermost valid point
        idx = _np.nanargmax(r[m])
        dxo_out, dxp_out = dxo[m][idx], dxp[m][idx]
        Vobs_out = Vobs[m][idx]
        vpred_out = vpred[m][idx] if vpred is not None else _np.nan
        result['delta_dx_outer'] = float(abs(dxo_out - dxp_out)) if _np.isfinite(dxo_out) and _np.isfinite(dxp_out) else _np.nan
        result['dx_ratio_outer'] = float(dxp_out / dxo_out) if (dxo_out not in (0, _np.nan) and _np.isfinite(dxo_out) and _np.isfinite(dxp_out)) else _np.nan
        result['v_scale_ratio_outer'] = float(vpred_out / Vobs_out) if (Vobs_out not in (0, _np.nan) and _np.isfinite(Vobs_out) and _np.isfinite(vpred_out)) else _np.nan
        # Simple affine fit on <=2 points: just scale and offset (no slope)
        if vpred is not None and _np.sum(m) >= 1:
            X = _np.vstack([vpred[m], _np.ones_like(vpred[m])]).T
            w = 1.0 / (_np.maximum(errV[m], 1e-6)**2 + float(getattr(args,'sigma_floor_kms',0.0))**2)
            sw = _np.sqrt(w)
            Xw = X * sw[:, None]
            yw = Vobs[m] * sw
            try:
                theta, *_ = _np.linalg.lstsq(Xw, yw, rcond=None)
                s_hat, b_hat = [float(t) for t in theta]
            except Exception:
                s_hat, b_hat = _np.nan, 0.0
            if _np.isfinite(s_hat) and _np.isfinite(vpred_out) and _np.isfinite(Vobs_out):
                vfit_out = s_hat * vpred_out + b_hat
                result['vel_resid_outer_affine'] = float(abs(Vobs_out - vfit_out))
        return pd.Series(result)

    if not all_data.empty:
        etg_lite_summary = all_data.groupby('filename').apply(_per_gal_etg_lite).reset_index()
        summary_df = summary_df.merge(etg_lite_summary, on='filename', how='left')

    sum_path = os.path.join(outdir, f"dx_{base_tag}_summary.csv")
    summary_df.to_csv(sum_path, index=False)

    if args.verbose:
        print(f"[Write] all_data -> {all_path}")
        print(f"[Write] summary -> {sum_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())