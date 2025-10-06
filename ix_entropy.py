import numpy as np
import pandas as pd
from .utils import moving_average

def derive_Ix_entropy_curvature(r, Vgas, Vdisk, Vbul, SB, entropy_as_prob: bool = False):
    """
    Deterministic I(x) from entropy–curvature relation (R3 method).
    SB input may be disk or bulge surface brightness depending on structural dominance.
    """
    r = np.asarray(r, dtype=float)
    Vgas = np.asarray(Vgas, dtype=float)
    Vdisk = np.asarray(Vdisk, dtype=float)
    Vbul = np.asarray(Vbul, dtype=float)
    SB = np.asarray(SB, dtype=float)

    v_lum = np.sqrt(np.maximum(Vdisk,0)**2 + np.maximum(Vbul,0)**2 + np.maximum(Vgas,0)**2)

    # curvature proxy from luminous
    dv   = np.gradient(v_lum, r)
    d2v  = np.gradient(dv, r)
    curvature = np.abs(d2v)

    # entropy memory from SB
    SB_pos = np.where(SB > 0, SB, 1e-12)
    if entropy_as_prob:
        p = SB_pos / (np.nansum(SB_pos) + 1e-12)
    else:
        p = SB_pos / (np.nanmax(SB_pos) + 1e-12)
    entropy_core = -p * np.log(np.clip(p, 1e-12, None))
    dS   = np.gradient(entropy_core, r)
    d2S  = np.gradient(dS, r)
    entropy_memory = np.abs(d2S)

    if not np.any(np.isfinite(entropy_memory)) or np.nanmax(entropy_memory) <= 1e-12:
        I = curvature
    else:
        I = np.sqrt(curvature * entropy_memory)

    if not np.all(np.isfinite(I)):
        I = np.where(np.isfinite(I), I, np.nan)
        I = pd.Series(I).interpolate(limit_direction='both').fillna(method='bfill').fillna(method='ffill').values
    return I