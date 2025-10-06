import os, subprocess, numpy as np, pandas as pd

G = 4.30091e-6  # kpc * (km/s)^2 / Msun

def git_hash():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None

def moving_average(arr, win=5):
    x = np.asarray(arr, dtype=float)
    if win is None or win <= 1 or x.size == 0:
        return np.copy(x)
    w = int(win)
    if w % 2 == 0: w += 1
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode='edge')
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(xp, kernel, mode='valid')

def cumtrapz(y, x):
    y = np.asarray(y, dtype=float); x = np.asarray(x, dtype=float)
    if y.size != x.size or y.size == 0:
        return np.zeros_like(y, dtype=float)
    dx = np.diff(x)
    avg = 0.5 * (y[1:] + y[:-1])
    integ = np.concatenate(([0.0], np.cumsum(dx * avg)))
    return integ

def luminous_velocity(Vdisk, Vbul, Vgas):
    return np.sqrt(np.maximum(Vdisk,0)**2 + np.maximum(Vbul,0)**2 + np.maximum(Vgas,0)**2)

def valid_mask_from(v_lum, Vobs, min_vlum=0.0):
    return (v_lum > float(min_vlum)) & (Vobs > 0) & np.isfinite(v_lum) & np.isfinite(Vobs)