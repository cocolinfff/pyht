import os, hashlib, json, numpy as np
import CoolProp.CoolProp as _CP
_KEYS = {'rho': 'D', 'mu': 'V', 'k': 'L', 'cp': 'C', 'h': 'H', 's': 'S'}

def _props(key, p, T, fluid='Water'):
    p = np.asarray(p, dtype=float)
    T = np.asarray(T, dtype=float)
    scalar = p.ndim == 0 and T.ndim == 0
    out = np.array([_CP.PropsSI(key, 'P', float(pi), 'T', float(Ti), fluid) for pi, Ti in zip(np.atleast_1d(p).flat, np.atleast_1d(T).flat)])
    return out.item() if scalar else out.reshape(p.shape)

def _props_safe(key, p, T, fluid='Water', fallback=0.0):
    try:
        return _CP.PropsSI(key, 'P', float(p), 'T', float(T), fluid)
    except (ValueError, RuntimeError, TypeError):
        return fallback
_props_cache = {}

def _props_Np1(key, p, T, fluid='Water', fallback=0.0):
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
                val = _CP.PropsSI(key, 'P', float(pi), 'T', float(Ti), fluid)
                _props_cache[ck] = val
            except (ValueError, RuntimeError, TypeError):
                val = fallback
        out.append(val)
    out = np.array(out)
    return out.item() if scalar else out.reshape(p.shape)

class PropertyTable:
    normalised = False

    def __init__(self, fluid='Water', p_lim=(10000.0, 20000000.0), T_lim=(273.15, 600.0), n_p=50, n_T=50):
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
        key = json.dumps(['v3norm', self.fluid, n_p, n_T, self.p_lim, self.T_lim], sort_keys=True)
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.table_cache')
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f'{h}.npz')

    def _build(self, n_p, n_T):
        cpath = self._cache_path(n_p, n_T)
        if os.path.isfile(cpath):
            data = np.load(cpath)
            self.p_grid = data['p_grid']
            self.T_grid = data['T_grid']
            self.u_grid = self.T_grid
            self.Tref_grid = data['Tref_grid']
            self.k_sat = int(data['k_sat'])
            self.normalised = bool(data['normalised'])
            self._vals = {}
            for name in _KEYS:
                self._vals[name] = data[name]
            return
        from scipy.interpolate import NearestNDInterpolator
        p_grid = np.logspace(np.log10(self.p_lim[0]), np.log10(self.p_lim[1]), n_p)
        if self.fluid == 'Water':
            Tc = float(_CP.PropsSI('Tcrit', 'Water'))
            Pc = float(_CP.PropsSI('Pcrit', 'Water'))
            Tref = np.array([Tc if p >= Pc * 0.9999 else float(_CP.PropsSI('T', 'P', float(p), 'Q', 0.0, 'Water')) for p in p_grid], dtype=float)
            self.normalised = True
        else:
            Tref = np.ones(n_p, dtype=float)
            self.normalised = False
        if self.normalised:
            u_lo = float((self.T_lim[0] / Tref).min())
            u_hi = float((self.T_lim[1] / Tref).max())
            k = int(round((1.0 - u_lo) / (u_hi - u_lo) * (n_T - 1)))
            k = max(3, min(n_T - 3, k))
            du = (1.0 - u_lo) / k
            below = u_lo + du * np.arange(k)
            above = np.linspace(1.0, u_hi, n_T - k - 1)[1:]
            u_grid = np.concatenate([below, [1.0, 1.0], above])
            assert len(u_grid) == n_T, (len(u_grid), n_T)
            assert np.sum(u_grid == 1.0) == 2, u_grid
            self.k_sat = k
            self.u_grid = u_grid
            TT = u_grid[None, :] * Tref[:, None]
            TT[:, k] = Tref - 0.001
            TT[:, k + 1] = Tref + 0.001
            self.T_grid = u_grid
        else:
            u_grid = np.linspace(self.T_lim[0], self.T_lim[1], n_T)
            self.k_sat = -1
            self.u_grid = u_grid
            self.T_grid = u_grid
            TT = np.repeat(u_grid[None, :], n_p, axis=0)
        self.Tref_grid = Tref
        pp = np.repeat(p_grid[:, None], n_T, axis=1)
        self.p_grid = p_grid
        self._vals = {}
        for name, key in _KEYS.items():
            vals = np.array([_props_safe(key, pp[i, j], TT[i, j], self.fluid, 0.0) for i in range(n_p) for j in range(n_T)]).reshape(n_p, n_T)
            mask = vals == 0
            if mask.any():
                pts = np.column_stack((pp[~mask], TT[~mask]))
                vals[mask] = NearestNDInterpolator(pts, vals[~mask])(pp[mask], TT[mask])
            self._vals[name] = vals
        save = {'p_grid': self.p_grid, 'T_grid': self.T_grid, 'Tref_grid': self.Tref_grid, 'k_sat': np.array(self.k_sat), 'normalised': np.array(self.normalised)}
        save.update(self._vals)
        np.savez_compressed(cpath, **save)

    def Tsat(self, p):
        if not self.normalised:
            return None
        p = np.asarray(p, dtype=float)
        scalar = p.ndim == 0
        cache = self.__dict__.setdefault('_tsat_cache', {})
        out = np.empty(np.atleast_1d(p).size, dtype=float)
        for n, pv in enumerate(np.atleast_1d(p).flat):
            key = round(float(pv), -1)
            v = cache.get(key)
            if v is None:
                try:
                    v = float(_CP.PropsSI('T', 'P', key, 'Q', 0.0, 'Water'))
                except ValueError:
                    v = float(_CP.PropsSI('Tcrit', 'Water'))
                cache[key] = v
            out[n] = v
        return out.item() if scalar else out.reshape(p.shape)

    def __call__(self, name, p, T):
        vals = self._vals[name]
        n_p, n_T = vals.shape
        p = np.asarray(p, dtype=float)
        T = np.asarray(T, dtype=float)
        scalar = p.ndim == 0 and T.ndim == 0
        log_p0 = np.log(self.p_grid[0])
        log_p1 = np.log(self.p_grid[-1])
        ratio = (np.log(np.maximum(p, self.p_grid[0])) - log_p0) / (log_p1 - log_p0)
        i = np.clip(np.floor(ratio * (n_p - 1)).astype(int), 0, n_p - 2)
        wp = (p - self.p_grid[i]) / (self.p_grid[i + 1] - self.p_grid[i])
        if self.normalised:
            Tref = (1 - wp) * self.Tref_grid[i] + wp * self.Tref_grid[i + 1]
            u = T / Tref
            j = np.searchsorted(self.u_grid, u, side='right') - 1
            j = np.clip(j, 0, n_T - 2)
            j = np.where(j == self.k_sat, self.k_sat - 1, j)
            j = np.clip(j, 0, n_T - 2)
            wT = (u - self.u_grid[j]) / (self.u_grid[j + 1] - self.u_grid[j])
            wT = np.clip(wT, 0.0, 1.0)
        else:
            u = T
            ratio_T = (np.maximum(u, self.u_grid[0]) - self.u_grid[0]) / (self.u_grid[-1] - self.u_grid[0])
            j = np.clip(np.floor(ratio_T * (n_T - 1)).astype(int), 0, n_T - 2)
            wT = (u - self.u_grid[j]) / (self.u_grid[j + 1] - self.u_grid[j])
        result = (1 - wp) * (1 - wT) * vals[i, j] + wp * (1 - wT) * vals[i + 1, j] + (1 - wp) * wT * vals[i, j + 1] + wp * wT * vals[i + 1, j + 1]
        return float(result) if scalar else result
_tables: dict[str, PropertyTable] = {}

def _get_table(fluid='Water'):
    if fluid not in _tables:
        _tables[fluid] = PropertyTable(fluid=fluid)
    return _tables[fluid]

class CoolPropWrapper:
    _use_table = True

    @classmethod
    def rho(cls, p, T, fluid='Water'):
        return cls._lookup('rho', p, T, fluid) if cls._use_table else _props_Np1('D', p, T, fluid)

    @classmethod
    def cp(cls, p, T, fluid='Water'):
        return cls._lookup('cp', p, T, fluid) if cls._use_table else _props_Np1('C', p, T, fluid)

    @classmethod
    def h(cls, p, T, fluid='Water'):
        return cls._lookup('h', p, T, fluid) if cls._use_table else _props_Np1('H', p, T, fluid)

    @classmethod
    def mu(cls, p, T, fluid='Water'):
        return cls._lookup('mu', p, T, fluid) if cls._use_table else _props_Np1('V', p, T, fluid)

    @classmethod
    def k(cls, p, T, fluid='Water'):
        return cls._lookup('k', p, T, fluid) if cls._use_table else _props_Np1('L', p, T, fluid)

    @classmethod
    def _lookup(cls, name, p, T, fluid):
        val = _get_table(fluid)(name, p, T)
        if isinstance(val, np.ndarray) and val.ndim == 0:
            return val.item()
        if isinstance(val, np.ndarray) and val.size == 1:
            return val.item()
        return val

    @classmethod
    def set_table_range(cls, p_lim=(10000.0, 20000000.0), T_lim=(273.15, 600.0), n_p=50, n_T=50, fluid='Water'):
        _get_table(fluid).rebuild(p_range=p_lim, T_range=T_lim, n_p=n_p, n_T=n_T)

    @staticmethod
    def saturation_T(p, fluid='Water'):
        if fluid != 'Water':
            return None
        p = np.asarray(p, dtype=float)
        scalar = p.ndim == 0
        out = []
        for pi in np.atleast_1d(p).flat:
            try:
                out.append(_CP.PropsSI('T', 'P', float(pi), 'Q', 0.0, 'Water'))
            except ValueError:
                out.append(373.15)
        out = np.array(out)
        return out.item() if scalar else out.reshape(p.shape)

    @staticmethod
    def d_rho_dp(p, T, fluid='Water'):
        return _props('d(D)/d(P)|T', p, T, fluid)

    @staticmethod
    def d_rho_dT(p, T, fluid='Water'):
        return _props('d(D)/d(T)|P', p, T, fluid)
WaterProperties = CoolPropWrapper
