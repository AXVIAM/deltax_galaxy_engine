import numpy as np
from .utils import cumtrapz

# Paper constants (kept configurable by caller)
default_a = np.pi/5
default_b = np.pi/3
default_c = np.pi/4
default_N = (7.0/12.0) * np.pi

def deltaX(M, D, I, a=default_a, b=default_b, c=default_c, N=default_N):
    core = np.pi * np.power(M + 1e-30, a) * np.power(D + 1e-30, b) * np.power(I + 1e-30, c)
    damp_arg = M * D * I
    damp_arg = np.where(damp_arg < 0, 0, damp_arg)
    damping = 1.0 / (1.0 + np.log1p(damp_arg + 1e-12))
    dx = core * damping
    for _ in range(5):
        feedback = 1.0 / (1.0 + N * np.maximum(dx, 1e-10))
        dx_new = core * damping * feedback
        if np.allclose(dx_new, dx, rtol=1e-5, atol=1e-8):
            break
        dx = dx_new
    return dx

def deltaX_with_metrics(M, D, I, a=default_a, b=default_b, c=default_c, N=default_N):
    core = np.pi * np.power(M + 1e-30, a) * np.power(D + 1e-30, b) * np.power(I + 1e-30, c)
    damp_arg = M * D * I
    damp_arg = np.where(damp_arg < 0, 0, damp_arg)
    damping = 1.0 / (1.0 + np.log1p(damp_arg + 1e-12))
    dx = core * damping
    feedback = np.ones_like(dx)
    for _ in range(5):
        feedback = 1.0 / (1.0 + N * np.maximum(dx, 1e-10))
        dx_new = core * damping * feedback
        if np.allclose(dx_new, dx, rtol=1e-5, atol=1e-8):
            dx = dx_new
            break
        dx = dx_new
    return dx, damping, feedback

def integrate_velocity_from_deltax(r, v_lum, dx, r_s):
    """
    Double-integrate curvature to velocity with stabilization:
      1) compute kappa = dx * v_lum / r_s_eff^2
      2) integrate twice to get Vint_raw
      3) stabilize with an outer-slope guard (match slope of direct predictor at large r)
      4) normalize at a pivot radius by matching the direct predictor value
    This keeps curvature-driven shape while preventing runaway parabolic growth.
    """
    r = np.asarray(r, dtype=float)
    v_lum = np.asarray(v_lum, dtype=float)
    dx = np.asarray(dx, dtype=float)

    # Effective scale length
    if not np.isfinite(r_s) or r_s <= 0:
        rmax = float(np.nanmax(r)) if np.any(np.isfinite(r)) else 1.0
        r_s_eff = max(rmax * 0.5, 1e-6)
    else:
        r_s_eff = float(r_s)

    # 1) curvature
    kappa = dx * (v_lum / (r_s_eff**2 + 1e-12))

    # 2) raw double integration
    Vprime = cumtrapz(kappa, r)
    Vint_raw = cumtrapz(Vprime, r)

    # Direct predictor for reference (used only for stabilization targets)
    Vpred_direct = dx * v_lum

    def _median_slope(y, x, frac=0.2):
        m = np.isfinite(x) & np.isfinite(y)
        if np.sum(m) < 3:
            return 0.0
        xs = x[m]; ys = y[m]
        order = np.argsort(xs)
        xs = xs[order]; ys = ys[order]
        n = len(xs)
        i0 = max(0, int((1.0 - frac) * n))
        xs_o = xs[i0:]; ys_o = ys[i0:]
        if len(xs_o) < 3:
            xs_o = xs; ys_o = ys
        with np.errstate(invalid='ignore'):
            d = np.gradient(ys_o, xs_o)
        d = d[np.isfinite(d)]
        return float(np.median(d)) if d.size else 0.0

    def _interp1(x, y, x0):
        xm = np.asarray(x, dtype=float); ym = np.asarray(y, dtype=float)
        m = np.isfinite(xm) & np.isfinite(ym)
        if np.sum(m) < 2:
            return np.nan
        xs = xm[m]; ys = ym[m]
        order = np.argsort(xs)
        xs = xs[order]; ys = ys[order]
        return float(np.interp(x0, xs, ys, left=ys[0], right=ys[-1]))

    # 3) Outer-slope guard: align the outer median slope of Vint to that of the direct predictor
    slope_int_outer = _median_slope(Vint_raw, r, frac=0.2)
    slope_dir_outer = _median_slope(Vpred_direct, r, frac=0.2)
    a = slope_dir_outer - slope_int_outer
    Vint_corr = Vint_raw + a * r

    # 4) Pivot normalization: match value at rp to direct predictor
    rmax = float(np.nanmax(r)) if np.any(np.isfinite(r)) else 1.0
    rp = 2.2 * r_s_eff
    if not np.isfinite(rp) or rp <= np.nanmin(r):
        rp = 0.35 * rmax
    Vint_at_rp = _interp1(r, Vint_corr, rp)
    Vdir_at_rp = _interp1(r, Vpred_direct, rp)
    if not np.isfinite(Vint_at_rp) or not np.isfinite(Vdir_at_rp):
        b = 0.0
    else:
        b = Vdir_at_rp - Vint_at_rp

    Vout = Vint_corr + b
    return np.where(np.isfinite(Vout), Vout, 0.0)