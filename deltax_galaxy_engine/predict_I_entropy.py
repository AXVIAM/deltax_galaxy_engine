import os
import pandas as pd
import numpy as np
from deltax_galaxy_engine.io import merge_etg_triplet
from deltax_galaxy_engine.inverse_entropy import inverse_entropy_solver

OUTPUT_CSV = "dx_Rotmod_ETG_outputs/I_predicted_output.csv"

def extract_I_for_all(folder):
    records = []

    for fname in os.listdir(folder):
        if not fname.endswith("_rotmod.dat"):
            continue

        full_path = os.path.join(folder, fname)

        try:
            disk_path = full_path.replace("_rotmod.dat", "_disk.dat")
            bulge_path = full_path.replace("_rotmod.dat", "_bulge.dat")
            r, Vobs, _, _, Vdisk, Vbul, *_ = merge_etg_triplet(full_path, disk_path, bulge_path)
            Vlum = np.sqrt(Vdisk**2 + Vbul**2)

            # Convert to arrays and filter valid entries
            r = np.array(r)
            Vobs = np.array(Vobs)
            Vlum = np.array(Vlum)
            valid_mask = np.isfinite(r) & np.isfinite(Vobs) & np.isfinite(Vlum)

            r = r[valid_mask]
            Vobs = Vobs[valid_mask]
            Vlum = Vlum[valid_mask]

            if len(r) >= 2:
                I_r, *_ = inverse_entropy_solver(r, Vobs, Vlum)

                for radius, I_val in zip(r, I_r):
                    records.append({"filename": fname, "r": radius, "I_predicted": I_val})
            else:
                print(f"[Skip] {fname} — not enough valid points.")

        except Exception as e:
            print(f"[Error] {fname}: {e}")

    if records:
        df_out = pd.DataFrame(records)
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        df_out.to_csv(OUTPUT_CSV, index=False)
        print(f"[✓] Wrote {len(df_out)} predictions to {OUTPUT_CSV}")
    else:
        print("[!] No valid I predictions generated.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m deltax_galaxy_engine.predict_I_entropy /path/to/Rotmod_ETG/Rotmod_ETG")
        sys.exit(1)
    extract_I_for_all(sys.argv[1])