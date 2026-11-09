import numpy as np
RMAX = 1.0 - 1.0 / np.sqrt(2.0)
RE_LAM = 2300.0
RE_TURB = 3000.0
RE_FIT_MAX = 11000.0
_P0 = 0.50876946
_P1 = -0.0756576
_P2 = -0.07179806
_P3 = -0.42066745

def c4_th(Re, Pr):
    X = np.log(np.asarray(Pr, float) / 0.91528)
    A = 0.03846 + 0.0015 * X + 0.0013 * X * X
    return A / np.log(np.asarray(Re, float)) ** 2

def d_power(Re, Pr, r):
    Re = np.asarray(Re, float)
    Pr = np.asarray(Pr, float)
    R = np.clip(np.asarray(r, float), 0.0, RMAX) / RMAX
    Y = np.log(Re / 6000.0)
    X = np.log(Pr / 0.91528)
    return np.exp(_P0 + _P1 * Y + _P2 * X + _P3 * np.log1p(R))

def c4_turb(Re, Pr, r):
    Re = np.asarray(Re, float)
    return c4_th(Re, Pr) * d_power(np.minimum(Re, RE_FIT_MAX), Pr, r)

def c4_lam(r):
    r = np.clip(np.asarray(r, float), 0.0, RMAX)
    return 0.04280504 * r ** 3 - 0.036259 * r ** 2 - 0.00041858 * r + 0.00596613

def _blend(Re, c_lam, c_turb):
    Re = np.asarray(Re, float)
    t = np.clip((np.log(np.maximum(Re, 1e-12)) - np.log(RE_LAM)) / (np.log(RE_TURB) - np.log(RE_LAM)), 0.0, 1.0)
    w = t * t * (3.0 - 2.0 * t)
    return np.where(Re <= RE_LAM, c_lam, np.where(Re >= RE_TURB, c_turb, (1.0 - w) * c_lam + w * c_turb))

def c4_water(Re, Pr, r):
    return _blend(Re, c4_lam(r), c4_turb(Re, Pr, r))

def c4_hem(Re_m, Pr_m, r):
    return c4_water(Re_m, Pr_m, r)
HE_FACTOR = 1.01944

def helium_factor(r):
    r = np.asarray(r, float)
    return np.full_like(r, HE_FACTOR)

def c4_helium(Re, Pr, r):
    return _blend(Re, c4_lam(r), c4_turb(Re, Pr, r) * helium_factor(r))
