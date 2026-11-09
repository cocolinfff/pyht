import math
import numpy as np

def nusselt_number(Re: np.ndarray, Pr: np.ndarray, laminar_nu: float=3.66) -> np.ndarray:
    Re = np.maximum(Re, 1.0)
    Pr = np.maximum(Pr, 0.001)
    Nu = np.zeros_like(Re)
    lam = Re < 2300
    turb = ~lam
    Nu[lam] = laminar_nu
    Nt = turb
    f = (0.79 * np.log(np.maximum(Re[Nt], 2301)) - 1.64) ** (-2)
    Nu[Nt] = f / 8 * (np.maximum(Re[Nt], 2301) - 1000) * Pr[Nt] / (1 + 12.7 * (f / 8) ** 0.5 * (Pr[Nt] ** (2 / 3) - 1))
    return np.maximum(Nu, 0.1)

def _nu_scalar(Re: float, Pr: float, laminar_nu: float=3.66) -> float:
    if Re < 2300:
        return laminar_nu
    Re_s = max(Re, 2301)
    f = (0.79 * math.log(Re_s) - 1.64) ** (-2)
    Nu = f / 8 * (Re_s - 1000) * Pr / (1 + 12.7 * (f / 8) ** 0.5 * (Pr ** (2 / 3) - 1))
    return max(Nu, 0.1)

def chen_two_phase(x, G, Dh, rho_l, rho_g, mu_l, mu_g, k_l, cp_l, sigma, h_fg, dT_sat, dP_sat):
    if x <= 0 or x >= 1:
        return 0.0
    Pr_l = mu_l * cp_l / max(k_l, 1e-10)
    X_tt = ((1 - x) / max(x, 1e-10)) ** 0.9 * (rho_g / rho_l) ** 0.5 * (mu_l / max(mu_g, 1e-10)) ** 0.1
    X_inv = 1.0 / max(X_tt, 1e-10)
    F = 1.0 if X_inv <= 0.1 else 2.35 * (X_inv + 0.213) ** 0.736
    Re_l = G * (1 - x) * Dh / max(mu_l, 1e-10)
    h_macro = 0.023 * Re_l ** 0.8 * Pr_l ** 0.4 * k_l / Dh * F
    Re_tp = Re_l * F ** 1.25
    S = 1.0 / (1.0 + 2.53e-06 * Re_tp ** 1.17)
    h_micro = 0.00122 * dT_sat ** 0.24 * dP_sat ** 0.75 / (sigma ** 0.5 * mu_l ** 0.29 * h_fg ** 0.24 * rho_g ** 0.24) * S
    return h_macro + max(h_micro, 0.0)

def chen_two_phase_c_scalar(x, G, Dh, rho_l, rho_g, mu_l, mu_g, k_l, cp_l, sigma, h_fg, dT_sat, dP_sat):
    if x <= 0.0 or x >= 1.0:
        return 0.0
    Pr_l = mu_l * cp_l / max(k_l, 1e-10)
    x_s = max(x, 1e-10)
    X_tt = ((1.0 - x) / x_s) ** 0.9 * (rho_g / rho_l) ** 0.5 * (mu_l / max(mu_g, 1e-10)) ** 0.1
    X_inv = 1.0 / max(X_tt, 1e-10)
    F = 1.0 if X_inv <= 0.1 else 2.35 * (X_inv + 0.213) ** 0.736
    Re_l = G * (1.0 - x) * Dh / max(mu_l, 1e-10)
    h_macro = 0.023 * Re_l ** 0.8 * Pr_l ** 0.4 * k_l / Dh * F
    Re_tp = Re_l * F ** 1.25
    S = 1.0 / (1.0 + 2.53e-06 * Re_tp ** 1.17)
    h_micro = 0.00122 * dT_sat ** 0.24 * dP_sat ** 0.75 / (sigma ** 0.5 * mu_l ** 0.29 * h_fg ** 0.24 * rho_g ** 0.24) * S
    return h_macro + h_micro if h_micro > 0 else h_macro

def wall_heat_flux(T_fluid: np.ndarray, T_wall: np.ndarray, htc: np.ndarray, P_heated: float) -> np.ndarray:
    return htc * P_heated * (T_wall - T_fluid)
