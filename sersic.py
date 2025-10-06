import numpy as np, pandas as pd

def sersic_bn(n):
    n = float(n)
    return 2.0*n - 1.0/3.0 + 0.009876/max(n,1e-6)

def build_sersic_sb(r_kpc, Re_kpc, n, Ie=None):
    r = np.asarray(r_kpc, dtype=float)
    Re = float(Re_kpc) if (Re_kpc is not None and np.isfinite(Re_kpc)) else np.nan
    n  = float(n) if (n is not None and np.isfinite(n)) else np.nan
    if not (np.isfinite(Re) and Re > 0 and np.isfinite(n) and n > 0):
        return np.ones_like(r)
    b  = sersic_bn(n)
    Ie_val = 1.0 if (Ie is None or not np.isfinite(Ie)) else float(Ie)
    with np.errstate(over='ignore', invalid='ignore'):
        x = np.power(r/(Re + 1e-12), 1.0/max(n,1e-6))
        I = Ie_val * np.exp(-b*(x - 1.0))
    I = np.where(np.isfinite(I), I, 0.0)
    if np.nanmax(I) > 0:
        I = I / np.nanmax(I)
    return np.clip(I, 1e-6, None)

def parse_sersic_table(table_path):
    df = pd.read_csv(table_path)
    cols = {c.lower(): c for c in df.columns}
    def pick(cands, required=False):
        for c in cands:
            if c in cols: return cols[c]
        if required: raise ValueError(f"Missing column among: {cands}")
        return None
    id_col = pick(['id','name','galaxy','object','stem'], required=True)
    re_col = pick(['re_kpc','re','r_e'])
    n_col  = pick(['n','sersic_n','index'])
    ie_col = pick(['ie','i_e','mu_e'])
    m = {}
    for _, row in df.iterrows():
        gid = str(row[id_col]).strip()
        stem = gid.replace('.dat','').replace('_rotmod','').strip().lower()
        Re = float(row[re_col]) if (re_col and pd.notna(row[re_col])) else np.nan
        n  = float(row[n_col])  if (n_col  and pd.notna(row[n_col]))  else np.nan
        Ie = float(row[ie_col]) if (ie_col and pd.notna(row[ie_col])) else np.nan
        m[stem] = {'Re_kpc': Re, 'n': n, 'Ie': Ie}
    return m