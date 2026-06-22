"""
Thermophysical property interface — CoolProp with multi-fluid support.

Fluids: Water (IAPWS-IF97, with bilinear lookup table), Helium (direct).
"""

import os, hashlib, json, numpy as np
import CoolProp.CoolProp as _CP

_KEYS = {"rho": "D", "mu": "V", "k": "L", "cp": "C", "h": "H", "s": "S"}

def _props(key, p, T, fluid="Water"):
    p = np.asarray(p, dtype=float)
    T = np.asarray(T, dtype=float)
    scalar = p.ndim == 0 and T.ndim == 0
    out = np.array([
        _CP.PropsSI(key, "P", float(pi), "T", float(Ti), fluid)
        for pi, Ti in zip(np.atleast_1d(p).flat, np.atleast_1d(T).flat)
    ])
    return out.item() if scalar else out.reshape(p.shape)


def _props_safe(key, p, T, fluid="Water", fallback=0.0):
    try:
        return _CP.PropsSI(key, "P", float(p), "T", float(T), fluid)
    except (ValueError, RuntimeError, TypeError):
        return fallback


_props_cache = {}
def _props_Np1(key, p, T, fluid="Water", fallback=0.0):
    p = np.asarray(p, dtype=float)
    T = np.asarray(T, dtype=float)
    scalar = p.ndim == 0 and T.ndim == 0
    out = []
    for pi, Ti in zip(np.atleast_1d(p).flat, np.atleast_1d(T).flat):
        ck = (fluid, key, round(float(pi), -2), round(float(Ti), 1))
        if ck in _props_cache:
            val = _props_cache[ck]
        else:
            try:
                val = _CP.PropsSI(key, "P", float(pi), "T", float(Ti), fluid)
                _props_cache[ck] = val
            except (ValueError, RuntimeError, TypeError):
                val = fallback
        out.append(val)
    out = np.array(out)
    return out.item() if scalar else out.reshape(p.shape)


# ---- lookup table (Water only — IAPWS-IF97 is expensive) ----

class PropertyTable:
    """Bilinear interpolation table for a fluid's properties."""

    def __init__(self, fluid="Water", p_lim=(1e4, 2e7), T_lim=(273.15, 600.0), n_p=50, n_T=50):
        self.fluid = fluid
        self.p_lim = tuple(p_lim)
        self.T_lim = tuple(T_lim)
        self._built = False
        self._build(n_p, n_T)

    def rebuild(self, p_range=None, T_range=None, n_p=50, n_T=50):
        if p_range:
            self.p_lim = tuple(p_range)
        if T_range:
            self.T_lim = tuple(T_range)
        self._build(n_p, n_T)

    def _cache_path(self, n_p, n_T):
        """Deterministic cache file path; includes params so invalidation is automatic."""
        key = json.dumps([self.fluid, n_p, n_T, self.p_lim, self.T_lim], sort_keys=True)
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".table_cache")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{h}.npz")

    def _build(self, n_p, n_T):
        cpath = self._cache_path(n_p, n_T)
        if os.path.isfile(cpath):
            data = np.load(cpath)
            self.p_grid = data["p_grid"]
            self.T_grid = data["T_grid"]
            self._vals = {}
            for name in _KEYS:
                self._vals[name] = data[name]
            return

        from scipy.interpolate import NearestNDInterpolator
        p_grid = np.logspace(np.log10(self.p_lim[0]), np.log10(self.p_lim[1]), n_p)
        T_grid = np.linspace(self.T_lim[0], self.T_lim[1], n_T)
        pp, TT = np.meshgrid(p_grid, T_grid, indexing="ij")
        self.p_grid = p_grid
        self.T_grid = T_grid
        self._vals = {}
        for name, key in _KEYS.items():
            vals = np.array([
                _props_safe(key, pp[i, j], TT[i, j], self.fluid, 0.0)
                for i in range(n_p) for j in range(n_T)
            ]).reshape(n_p, n_T)
            mask = vals == 0
            if mask.any():
                pts = np.column_stack((pp[~mask], TT[~mask]))
                vals[mask] = NearestNDInterpolator(pts, vals[~mask])(pp[mask], TT[mask])
            self._vals[name] = vals
        # save to cache
        save = {"p_grid": self.p_grid, "T_grid": self.T_grid}
        save.update(self._vals)
        np.savez_compressed(cpath, **save)

    def __call__(self, name, p, T):
        """Fast bilinear interpolation — pure numpy, no scipy."""
        vals = self._vals[name]
        n_p, n_T = vals.shape
        p = np.asarray(p, dtype=float); T = np.asarray(T, dtype=float)
        scalar = p.ndim == 0 and T.ndim == 0
        # log-spaced p-grid → direct index
        log_p0 = np.log(self.p_grid[0]); log_p1 = np.log(self.p_grid[-1])
        ratio = (np.log(np.maximum(p, self.p_grid[0])) - log_p0) / (log_p1 - log_p0)
        i = np.clip(np.floor(ratio * (n_p - 1)).astype(int), 0, n_p - 2)
        # linear T-grid
        ratio_T = (np.maximum(T, self.T_grid[0]) - self.T_grid[0]) / (self.T_grid[-1] - self.T_grid[0])
        j = np.clip(np.floor(ratio_T * (n_T - 1)).astype(int), 0, n_T - 2)
        wp = (p - self.p_grid[i]) / (self.p_grid[i + 1] - self.p_grid[i])
        wT = (T - self.T_grid[j]) / (self.T_grid[j + 1] - self.T_grid[j])
        result = ((1 - wp) * (1 - wT) * vals[i, j] +
                  wp * (1 - wT) * vals[i + 1, j] +
                  (1 - wp) * wT * vals[i, j + 1] +
                  wp * wT * vals[i + 1, j + 1])
        return float(result) if scalar else result


# ---- module-level singletons ----

_tables: dict[str, PropertyTable] = {}

def _get_table(fluid="Water"):
    if fluid not in _tables:
        _tables[fluid] = PropertyTable(fluid=fluid)
    return _tables[fluid]


# ---- public API ----

class CoolPropWrapper:
    """Multi-fluid property wrapper. Use via classmethods::

        CoolPropWrapper.rho(p, T, fluid="Water")
        CoolPropWrapper.saturation_T(p, fluid="Water")
    """
    _use_table = True

    # --- properties (use table for any fluid when _use_table is True) ---
    @classmethod
    def rho(cls, p, T, fluid="Water"):
        return cls._lookup("rho", p, T, fluid) if cls._use_table else _props_Np1("D", p, T, fluid)

    @classmethod
    def cp(cls, p, T, fluid="Water"):
        return cls._lookup("cp", p, T, fluid) if cls._use_table else _props_Np1("C", p, T, fluid)

    @classmethod
    def h(cls, p, T, fluid="Water"):
        return cls._lookup("h", p, T, fluid) if cls._use_table else _props_Np1("H", p, T, fluid)

    @classmethod
    def mu(cls, p, T, fluid="Water"):
        return cls._lookup("mu", p, T, fluid) if cls._use_table else _props_Np1("V", p, T, fluid)

    @classmethod
    def k(cls, p, T, fluid="Water"):
        return cls._lookup("k", p, T, fluid) if cls._use_table else _props_Np1("L", p, T, fluid)

    @classmethod
    def _lookup(cls, name, p, T, fluid):
        val = _get_table(fluid)(name, p, T)
        if isinstance(val, np.ndarray) and val.ndim == 0:
            return val.item()
        if isinstance(val, np.ndarray) and val.size == 1:
            return val.item()
        return val

    # --- helpers ---
    @classmethod
    def set_table_range(cls, p_lim=(1e4, 2e7), T_lim=(273.15, 600.0), n_p=50, n_T=50, fluid="Water"):
        _get_table(fluid).rebuild(p_range=p_lim, T_range=T_lim, n_p=n_p, n_T=n_T)

    @staticmethod
    def saturation_T(p, fluid="Water"):
        if fluid != "Water":
            return None
        p = np.asarray(p, dtype=float)
        scalar = p.ndim == 0
        out = []
        for pi in np.atleast_1d(p).flat:
            try:
                out.append(_CP.PropsSI("T", "P", float(pi), "Q", 0.0, "Water"))
            except ValueError:
                out.append(373.15)
        out = np.array(out)
        return out.item() if scalar else out.reshape(p.shape)

    @staticmethod
    def d_rho_dp(p, T, fluid="Water"):
        return _props("d(D)/d(P)|T", p, T, fluid)

    @staticmethod
    def d_rho_dT(p, T, fluid="Water"):
        return _props("d(D)/d(T)|P", p, T, fluid)


# backward-compatible alias
WaterProperties = CoolPropWrapper
