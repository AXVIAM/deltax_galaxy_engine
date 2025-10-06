import os, numpy as np
from .sersic import build_sersic_sb

def parse_rotmod_file(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'): continue
            parts = s.split()
            try:
                vals = list(map(float, parts[:8]))
                data.append(vals)
            except: continue
    arr = np.array(data, dtype=float)
    if arr.shape[1] < 8:
        pad = 8 - arr.shape[1]
        arr = np.pad(arr, ((0,0),(0,pad)), mode='constant')
    elif arr.shape[1] > 8:
        arr = arr[:, :8]
    r, Vobs, errV, Vgas, Vdisk, Vbul, SBdisk, SBbul = [arr[:,i] for i in range(8)]
    return r, Vobs, errV, Vgas, Vdisk, Vbul, SBdisk, SBbul

def parse_component_dat(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#') or s.startswith('!'):
                continue
            parts = s.split()
            nums = []
            for tok in parts:
                try:
                    nums.append(float(tok))
                except ValueError:
                    break
            if len(nums) >= 3:
                data.append((nums[0], nums[-1], nums[1]))  # r, v, density
            elif len(nums) == 2:
                data.append((nums[0], nums[1], np.nan))

    if len(data) == 0:
        raise ValueError(f"No numeric data found in {os.path.basename(filename)}")
    arr = np.array(data, dtype=float)
    r = arr[:, 0]
    V = arr[:, 1]
    SB = arr[:, 2]
    if not np.any(np.isfinite(SB)):
        SB = None
    return r, V, SB

def find_etg_triplet(rotmod_path):
    base = os.path.basename(rotmod_path)
    if not base.lower().endswith('_rotmod.dat'):
        return False, None, None
    stem = base[:-11]
    dpath = os.path.join(os.path.dirname(rotmod_path), f"{stem}_disk.dat")
    bpath = os.path.join(os.path.dirname(rotmod_path), f"{stem}_bulge.dat")
    return os.path.exists(dpath) and os.path.exists(bpath), dpath, bpath

def _interp_safe(x_src, y_src, x_new, fill=0.0):
    mask = np.isfinite(x_src) & np.isfinite(y_src)
    if np.sum(mask) < 2: return np.full_like(x_new, fill, dtype=float)
    order = np.argsort(x_src[mask])
    xs = x_src[mask][order]; ys = y_src[mask][order]
    return np.interp(x_new, xs, ys, left=ys[0], right=ys[-1])

def merge_etg_triplet(rotmod_path, disk_path, bulge_path, sersic_params=None, verbose=False):
    base = os.path.basename(rotmod_path)
    r_rot, Vobs, errV, Vgas_rm, Vdisk_rm, Vbul_rm, SBdisk_rm, SBbul_rm = parse_rotmod_file(rotmod_path)
    r_d, Vdisk_d, SBdisk_d = parse_component_dat(disk_path)
    r_b, Vbul_b,  SBbul_b  = parse_component_dat(bulge_path)

    # heuristic pc->kpc
    def maybe_pc_to_kpc(rarr):
        rmed = float(np.nanmedian(rarr[np.isfinite(rarr)])) if np.any(np.isfinite(rarr)) else np.nan
        if np.isfinite(rmed) and (300.0 <= rmed <= 50000.0):
            return rarr / 1000.0, True
        return rarr, False
    r_rot, did_rot = maybe_pc_to_kpc(r_rot)
    r_d,   did_d   = maybe_pc_to_kpc(r_d)
    r_b,   did_b   = maybe_pc_to_kpc(r_b)
    if verbose and (did_rot or did_d or did_b):
        print(f"[ETG radius guard] pc→kpc in {base} (rot:{did_rot} disk:{did_d} bulge:{did_b})")

    Vdisk = _interp_safe(r_d, Vdisk_d, r_rot, 0.0)
    Vbul  = _interp_safe(r_b, Vbul_b,  r_rot, 0.0)

    # Fallbacks: if component files contribute ~0, try rotmod columns
    def _has_signal(v):
        v = np.asarray(v, dtype=float)
        return np.isfinite(v).any() and np.nanmax(np.abs(v)) > 1e-3

    if not _has_signal(Vdisk) and _has_signal(Vdisk_rm):
        Vdisk = np.array(Vdisk_rm, dtype=float)
    if not _has_signal(Vbul) and _has_signal(Vbul_rm):
        Vbul = np.array(Vbul_rm, dtype=float)

    SBdisk = SBdisk_rm
    SBbul  = SBbul_rm
    sersic_sb = None
    if sersic_params and all(k in sersic_params for k in ('Re_kpc','n')):
        sersic_sb = build_sersic_sb(r_rot, sersic_params['Re_kpc'], sersic_params['n'], sersic_params.get('Ie'))

    def sanitize(sb):
        sb = np.array(sb, dtype=float)
        if not np.any(np.isfinite(sb)): sb = np.ones_like(r_rot)
        mn = np.nanmin(sb); mx = np.nanmax(sb)
        if not np.isfinite(mx) or not np.isfinite(mn) or mx <= mn:
            return np.clip(np.nan_to_num(sb, nan=1.0, posinf=1.0, neginf=1.0), 1e-6, None)
        sb = (sb - mn) / (mx - mn)
        return np.clip(sb, 1e-6, None)

    # prefer explicit component SB; fallback to Sérsic; else neutral
    if SBdisk_d is None or (isinstance(SBdisk_d, np.ndarray) and not np.any(np.isfinite(SBdisk_d))):
        SBdisk = sanitize(sersic_sb if sersic_sb is not None else np.ones_like(r_rot))
    else:
        SBdisk = sanitize(_interp_safe(r_d, SBdisk_d, r_rot, fill=np.nan))

    if SBbul_b is None or (isinstance(SBbul_b, np.ndarray) and not np.any(np.isfinite(SBbul_b))):
        SBbul = sanitize(sersic_sb if sersic_sb is not None else np.zeros_like(r_rot))
    else:
        SBbul = sanitize(_interp_safe(r_b, SBbul_b, r_rot, fill=np.nan))

    # If there is no measurable disk but a clear bulge, let bulge SB seed the scale for Δx
    if (not _has_signal(Vdisk)) and _has_signal(Vbul):
        # Use bulge SB as the primary surface-brightness driver when the disc is absent
        SBdisk = sanitize(SBbul)

    Vgas = Vgas_rm if np.any(np.isfinite(Vgas_rm)) else np.zeros_like(r_rot)

    # Final safety: replace non-finite with zeros (velocities) or small positive (SB)
    def _safe_arr(a, fill=0.0):
        a = np.array(a, dtype=float)
        a[~np.isfinite(a)] = fill
        return a
    Vgas  = _safe_arr(Vgas, 0.0)
    Vdisk = _safe_arr(Vdisk, 0.0)
    Vbul  = _safe_arr(Vbul, 0.0)
    SBdisk = _safe_arr(SBdisk, 1e-6)
    SBbul  = _safe_arr(SBbul, 1e-6)

    result_dict = {"dx_method": "bulge" if (not _has_signal(Vdisk) and _has_signal(Vbul)) else "disk"}
    result_dict["I_predicted"] = np.full_like(r_rot, np.nan)
    return r_rot, Vobs, errV, Vgas, Vdisk, Vbul, SBdisk, SBbul, result_dict, None