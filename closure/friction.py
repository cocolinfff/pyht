"""
Single-phase + two-phase friction factor correlations.
All return Darcy friction factor f.
"""

import math
import numpy as np


def friction_factor(Re: np.ndarray, Dh: float, roughness: float = 1.5e-6) -> np.ndarray:
    """Churchill correlation — array version."""
    Re = np.maximum(Re, 1.0)
    f_lam = 64.0 / Re
    A = (2.457 * np.log(1.0 / ((7.0 / Re) ** 0.9 + 0.27 * roughness / Dh))) ** 16
    B = (37530.0 / Re) ** 16
    f_turb = 8.0 * ((8.0 / Re) ** 12 + 1.0 / (A + B) ** 1.5) ** (1.0 / 12.0)
    return np.where(Re <= 2300, f_lam, np.where(Re < 4000, f_lam + (f_turb - f_lam) * (Re - 2300) / 1700, f_turb))


def _ff_scalar(Re: float, Dh: float, roughness: float = 1.5e-6) -> float:
    """Churchill correlation — fast scalar path (no numpy overhead)."""
    Re = Re if Re > 1.0 else 1.0
    f_lam = 64.0 / Re
    A = (2.457 * math.log(1.0 / ((7.0 / Re) ** 0.9 + 0.27 * roughness / Dh))) ** 16
    B = (37530.0 / Re) ** 16
    f_turb = 8.0 * ((8.0 / Re) ** 12 + 1.0 / (A + B) ** 1.5) ** (1.0 / 12.0)
    if Re <= 2300:
        return f_lam
    if Re < 4000:
        return f_lam + (f_turb - f_lam) * (Re - 2300) / 1700
    return f_turb


def Re_from_mass_flux(G: np.ndarray, Dh: float, mu: np.ndarray) -> np.ndarray:
    """Reynolds number from mass flux."""
    return np.abs(G) * Dh / np.maximum(mu, 1e-10)


def friedel_two_phi(x: float, rho_l: float, rho_g: float,
                     mu_l: float, mu_g: float, G: float, Dh: float,
                     sigma: float = 0.05, g: float = 9.81) -> float:
    """Friedel (1979) two-phase friction multiplier — scalar, fast."""
    if x <= 0:
        return 1.0
    if x >= 1:
        return 1.0
    if G <= 1e-10:
        return 1.0

    Re_lo = G * Dh / mu_l
    Fr = max(G * G / (g * Dh * rho_l * rho_l), 1e-10)
    We = max(G * G * Dh / (sigma * rho_l), 1e-10)
    rho_ratio = rho_l / rho_g
    mu_ratio = mu_l / mu_g

    f_mu = _ff_scalar(Re_lo * mu_ratio, Dh)
    f_lo = max(_ff_scalar(Re_lo, Dh), 1e-10)
    E = (1 - x) * (1 - x) + x * x * rho_ratio * f_mu / f_lo
    F = x ** 0.78 * (1 - x) ** 0.224
    base_h = 1.0 - mu_ratio / rho_ratio
    H = (rho_ratio ** 0.91) * (mu_ratio ** 0.19) * (base_h ** 0.7 if base_h > 0 else 0.0)

    Fr_h = max(Fr / (rho_ratio ** 0.5), 1e-10)
    We_h = max(We / (rho_ratio ** 0.5), 1e-10)

    phi2 = E + 3.24 * F * H / (Fr_h ** 0.045 * We_h ** 0.035)
    return max(phi2, 1.0)
