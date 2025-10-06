


import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d

def dx_model(I_r, beta=np.pi/5):
    """Compute predicted DeltaX from I(r) using known beta scaling."""
    I_r = np.asarray(I_r)
    return beta * I_r

def inverse_entropy_solver(r, v_obs, v_lum, I0_guess=None, beta=np.pi/5):
    """
    Given r, observed and luminous velocities, solve for I(r) such that
    DeltaX_pred = log10(v_obs^2 / v_lum^2) ≈ beta * I(r)
    """
    r = np.asarray(r)
    v_obs = np.asarray(v_obs)
    v_lum = np.asarray(v_lum)
    dx_obs = np.log10(np.clip(v_obs**2 / v_lum**2, 1e-10, 1e10))

    if I0_guess is None:
        I0_guess = np.zeros_like(r)

    def loss(I_r_flat):
        dx_pred = dx_model(I_r_flat, beta=beta)
        return np.mean((dx_pred - dx_obs)**2)

    result = minimize(loss, I0_guess, method="L-BFGS-B")
    I_r_opt = result.x

    # Diagnostic logging block
    print("[EntropySolver] Inputs:")
    print("  r:", r)
    print("  Vobs:", v_obs)
    print("  Vlum:", v_lum)
    print("  dx_obs:", dx_obs)
    print("  I0_guess:", I0_guess)
    print("  Optimized I_r:", I_r_opt)
    print("  Success:", result.success)
    print("  Message:", result.message)

    return I_r_opt, dx_obs, dx_model(I_r_opt, beta=beta), result