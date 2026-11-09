"""
Single-phase heat transfer correlations.
"""

import math
import numpy as np


def nusselt_number(Re: np.ndarray, Pr: np.ndarray,
                   laminar_nu: float = 3.66) -> np.ndarray:
    """Gnielinski correlation: covers transition and turbulent (2300 < Re < 5e6).

    Parameters
    ----------
    Re, Pr : ndarray
        Reynolds and Prandtl numbers.
    laminar_nu : float
        Laminar Nusselt number for fully developed flow.
        Common values:
        - 3.66  : circular tube, constant wall temperature (default)
        - 4.36  : circular tube, constant heat flux
        - 2.98  : square duct, constant wall temperature
        - 3.61  : square duct, constant heat flux
        - 7.54  : parallel plates, constant wall temperature
    """
    Re = np.maximum(Re, 1.0)
    Pr = np.maximum(Pr, 0.001)

    Nu = np.zeros_like(Re)

    lam = Re < 2300
    turb = ~lam

    Nu[lam] = laminar_nu

    # turbulent: Gnielinski
    Nt = turb
    f = (0.79 * np.log(np.maximum(Re[Nt], 2301)) - 1.64) ** (-2)
    Nu[Nt] = (f / 8) * (np.maximum(Re[Nt], 2301) - 1000) * Pr[Nt] / (
        1 + 12.7 * (f / 8) ** 0.5 * (Pr[Nt] ** (2 / 3) - 1)
    )

    return np.maximum(Nu, 0.1)


def _nu_scalar(Re: float, Pr: float, laminar_nu: float = 3.66) -> float:
    """Gnielinski — fast scalar path (no numpy overhead)."""
    if Re < 2300:
        return laminar_nu
    Re_s = max(Re, 2301)
    f = (0.79 * math.log(Re_s) - 1.64) ** (-2)
    Nu = (f / 8) * (Re_s - 1000) * Pr / (1 + 12.7 * (f / 8) ** 0.5 * (Pr ** (2 / 3) - 1))
    return max(Nu, 0.1)


def chen_two_phase(x, G, Dh, rho_l, rho_g, mu_l, mu_g, k_l, cp_l,
                   sigma, h_fg, dT_sat, dP_sat):
    """Chen (1966) saturated boiling heat transfer coefficient [W/m2K].

    h_tp = h_macro * F + h_micro * S

    Parameters
    ----------
    x : float        — thermodynamic quality [0..1]
    G : float        — total mass flux [kg/m2s]
    Dh : float       — hydraulic diameter [m]
    rho_l, rho_g     — liquid / vapor density [kg/m3]
    mu_l, mu_g       — liquid / vapor viscosity [Pa*s]
    k_l              — liquid thermal conductivity [W/mK]
    cp_l             — liquid specific heat [J/kgK]
    sigma            — surface tension [N/m]
    h_fg             — latent heat [J/kg]
    dT_sat           — wall superheat T_w - Tsat [K]
    dP_sat           — P_sat(T_w) - P [Pa]

    Returns
    -------
    h_tp : float     — two-phase heat transfer coefficient [W/m2K]
    """
    if x <= 0 or x >= 1:
        return 0.0
    Pr_l = mu_l * cp_l / max(k_l, 1e-10)

    # Martinelli parameter
    X_tt = ((1 - x) / max(x, 1e-10)) ** 0.9 * (rho_g / rho_l) ** 0.5 * (mu_l / max(mu_g, 1e-10)) ** 0.1

    # Convective enhancement factor F
    X_inv = 1.0 / max(X_tt, 1e-10)
    F = 1.0 if X_inv <= 0.1 else 2.35 * (X_inv + 0.213) ** 0.736

    # Macroscopic convective contribution (Dittus-Boelter × F)
    Re_l = G * (1 - x) * Dh / max(mu_l, 1e-10)
    h_macro = 0.023 * Re_l ** 0.8 * Pr_l ** 0.4 * k_l / Dh * F

    # Nucleate boiling suppression factor
    Re_tp = Re_l * F ** 1.25
    S = 1.0 / (1.0 + 2.53e-6 * Re_tp ** 1.17)

    # Microscopic (nucleate boiling) contribution (Forster-Zuber)
    h_micro = (0.00122 * (dT_sat ** 0.24) * (dP_sat ** 0.75) /
               (sigma ** 0.5 * mu_l ** 0.29 * h_fg ** 0.24 * rho_g ** 0.24)) * S

    return h_macro + max(h_micro, 0.0)


def chen_two_phase_c_scalar(x, G, Dh, rho_l, rho_g, mu_l, mu_g, k_l, cp_l,
                             sigma, h_fg, dT_sat, dP_sat):
    """Chen correlation — scalar fast path (no numpy calls)."""
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
    S = 1.0 / (1.0 + 2.53e-6 * Re_tp ** 1.17)
    h_micro = (0.00122 * (dT_sat ** 0.24) * (dP_sat ** 0.75) /
               (sigma ** 0.5 * mu_l ** 0.29 * h_fg ** 0.24 * rho_g ** 0.24)) * S
    return h_macro + h_micro if h_micro > 0 else h_macro


def wall_heat_flux(
    T_fluid: np.ndarray, T_wall: np.ndarray,
    htc: np.ndarray, P_heated: float
) -> np.ndarray:
    """Heat flux per unit axial length [W/m]."""
    return htc * P_heated * (T_wall - T_fluid)
