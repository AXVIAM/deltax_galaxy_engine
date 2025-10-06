import numpy as np

def anchor_velocity_profile(r, Vint_raw, Vobs, errV, Vlum, r_s,
                            mode='inner+pivot', pivot_k=2.2, pivot_target='Vobs',
                            sigma_floor=0.0, inner_trim_frac=0.0):
    r = np.asarray(r, dtype=float)
    Vint_raw = np.asarray(Vint_raw, dtype=float)
    Vobs = np.asarray(Vobs, dtype=float) if Vobs is not None else None
    Vlum = np.asarray(Vlum, dtype=float)
    errV = np.asarray(errV, dtype=float) if errV is not None else None

    if not np.isfinite(r_s) or r_s <= 0:
        rmax = float(np.nanmax(r)) if np.any(np.isfinite(r)) else 1.0
        r_s_eff = max(rmax * 0.5, 1e-6)
    else:
        r_s_eff = float(r_s)

    if str(mode).lower() == 'lsq' and (Vobs is not None) and (errV is not None):
        ro, yo, ei, vin = r, Vobs, errV, Vint_raw
        if np.any(np.isfinite(ei)) and float(sigma_floor) > 0:
            ei = np.sqrt(ei**2 + float(sigma_floor)**2)
        m = np.isfinite(ro) & np.isfinite(yo) & np.isfinite(ei) & (ei > 0) & np.isfinite(vin)
        if float(inner_trim_frac) > 0 and np.isfinite(r_s_eff) and r_s_eff > 0:
            m &= (ro >= float(inner_trim_frac) * r_s_eff)
        if np.sum(m) >= 3:
            r_fit = ro[m]
            y_fit = yo[m] - vin[m]
            w = 1.0 / (ei[m] ** 2)
            X = np.vstack([r_fit, np.ones_like(r_fit)]).T
            sw = np.sqrt(w)
            Xw = X * sw[:, None]
            yw = y_fit * sw
            try:
                theta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
                alpha_ls, beta_ls = float(theta[0]), float(theta[1])
            except Exception:
                alpha_ls, beta_ls = 0.0, 0.0
            Vanch = vin + alpha_ls * ro + beta_ls
            return np.where(np.isfinite(Vanch), Vanch, 0.0)
        # else fall through

    def _interp1(x, y, x0, left=None, right=None):
        xm = np.asarray(x, dtype=float); ym = np.asarray(y, dtype=float)
        m = np.isfinite(xm) & np.isfinite(ym)
        if np.sum(m) < 2: return np.nan
        xs, ys = xm[m], ym[m]; order = np.argsort(xs); xs, ys = xs[order], ys[order]
        if left is None: left = ys[0]
        if right is None: right = ys[-1]
        return float(np.interp(x0, xs, ys, left=left, right=right))

    rmax = float(np.nanmax(r)) if np.any(np.isfinite(r)) else 1.0
    r_inner_lim = 0.3 * r_s_eff if np.isfinite(r_s_eff) and r_s_eff > 0 else 0.2 * rmax
    inner_mask = (r > 0) & np.isfinite(r) & (r <= r_inner_lim) & np.isfinite(Vlum)
    if np.sum(inner_mask) < 3:
        valid_idx = np.where((r > 0) & np.isfinite(r) & np.isfinite(Vlum))[0]
        inner_mask = np.zeros_like(r, dtype=bool)
        inner_mask[valid_idx[:max(3, min(5, valid_idx.size))]] = True
    slope_tgt = np.nanmedian(Vlum[inner_mask] / np.clip(r[inner_mask], 1e-9, None))
    if not np.isfinite(slope_tgt): slope_tgt = 0.0

    dVint = np.gradient(Vint_raw, r)
    slope_int = np.nanmedian(dVint[inner_mask])
    if not np.isfinite(slope_int): slope_int = 0.0
    alpha = slope_tgt - slope_int

    rp = pivot_k * r_s_eff
    if not np.isfinite(rp) or rp <= np.nanmin(r):
        rp = 0.35 * rmax
    Vint_at_rp = _interp1(r, Vint_raw, rp)
    if str(pivot_target).lower() == 'vobs' and Vobs is not None:
        Vtar = _interp1(r, Vobs, rp)
        if not np.isfinite(Vtar):
            Vtar = _interp1(r, Vlum, rp)
    else:
        Vtar = _interp1(r, Vlum, rp)
    if not np.isfinite(Vtar): Vtar = 0.0
    beta = Vtar - (Vint_at_rp + alpha * rp)

    Vanch = Vint_raw + alpha * r + beta
    return np.where(np.isfinite(Vanch), Vanch, 0.0)

def anchor_velocity_scale_only(r, Vint_raw, Vobs, errV, Vlum, r_s,
                               pivot_k=2.2, pivot_target='Vobs',
                               sigma_floor=0.0, inner_trim_frac=0.0, **kwargs):
    r = np.asarray(r, dtype=float)
    Vint_raw = np.asarray(Vint_raw, dtype=float)
    Vobs = np.asarray(Vobs, dtype=float) if Vobs is not None else None
    Vlum = np.asarray(Vlum, dtype=float)

    if not np.isfinite(r_s) or r_s <= 0:
        rmax = float(np.nanmax(r)) if np.any(np.isfinite(r)) else 1.0
        r_s_eff = max(rmax * 0.5, 1e-6)
    else:
        r_s_eff = float(r_s)

    def _interp1(x, y, x0, left=None, right=None):
        xm = np.asarray(x, dtype=float); ym = np.asarray(y, dtype=float)
        m = np.isfinite(xm) & np.isfinite(ym)
        if np.sum(m) < 2: return np.nan
        xs, ys = xm[m], ym[m]; order = np.argsort(xs); xs, ys = xs[order], ys[order]
        if left is None: left = ys[0]
        if right is None: right = ys[-1]
        return float(np.interp(x0, xs, ys, left=left, right=right))

    rmax = float(np.nanmax(r)) if np.any(np.isfinite(r)) else 1.0
    rp = pivot_k * r_s_eff
    if not np.isfinite(rp) or rp <= np.nanmin(r):
        rp = 0.35 * rmax

    Vint_at_rp = _interp1(r, Vint_raw, rp)
    if str(pivot_target).lower() == 'vobs' and Vobs is not None:
        Vtar = _interp1(r, Vobs, rp)
        if not np.isfinite(Vtar):
            Vtar = _interp1(r, Vlum, rp)
    else:
        Vtar = _interp1(r, Vlum, rp)
    if not np.isfinite(Vint_at_rp) or Vint_at_rp == 0:
        s = 1.0
    else:
        s = Vtar / Vint_at_rp

    Vanch = s * Vint_raw
    return np.where(np.isfinite(Vanch), Vanch, 0.0)