import os, time, ctypes
import numpy as np
import CoolProp.CoolProp as _CP
from ..core.topology import SubchannelMesh
from ..core.variables import StateVector, ResidualHistory
from ..fluid import _get_table
_CDIR = os.path.dirname(__file__)
_DLL = os.environ.get('PYHT_SOLVER_DLL') or '_solver_c.dll'
_C_LIB = ctypes.CDLL(os.path.join(_CDIR, _DLL))
print('[pyht] C extension loaded: ' + _DLL)
DIVERGED_T = 100000000.0
_prop_cache = {}

def _cached_CP(key_str, p, T, fluid='Water'):
    ck = (fluid, key_str, round(p, -2), round(T, 1))
    if ck in _prop_cache:
        return _prop_cache[ck]
    try:
        v = float(_CP.PropsSI(key_str, 'P', float(p), 'T', float(T), fluid))
        _prop_cache[ck] = v
        return v
    except Exception:
        return 0.0

def _rho(p, T, fluid='Water'):
    v = _cached_CP('D', p, T, fluid)
    return v if v > 0 else 1000.0

def _h(p, T, fluid='Water'):
    v = _cached_CP('H', p, T, fluid)
    return v if v > 0 else 400000.0

def _h_direct(p, T):
    return _cached_CP('H', p, T, 'Water') or 400000.0
_TCRIT = float(_CP.PropsSI('Tcrit', 'Water'))
_PCRIT = float(_CP.PropsSI('Pcrit', 'Water'))
_Tsat_cache = {}

def _Tsat(p):
    key = float(p)
    v = _Tsat_cache.get(key)
    if v is not None:
        return v
    if key >= _PCRIT * 0.9999:
        v = _TCRIT
    else:
        try:
            v = float(_CP.PropsSI('T', 'P', key, 'Q', 0.0, 'Water'))
        except Exception:
            v = _TCRIT
    _Tsat_cache[key] = v
    return v

def _quality(h, h_l, h_g):
    if h_l >= h_g - 1:
        return 0.0
    return float(np.clip((h - h_l) / (h_g - h_l), 0.0, 1.0))

class _AxisT(ctypes.Structure):
    _fields_ = [('p_grid', ctypes.POINTER(ctypes.c_double)), ('n_p', ctypes.c_int), ('u_grid', ctypes.POINTER(ctypes.c_double)), ('n_T', ctypes.c_int), ('Tref', ctypes.POINTER(ctypes.c_double)), ('is_norm', ctypes.c_int), ('k_sat', ctypes.c_int)]

def _axis_of(tab):
    keep = getattr(tab, '_axis_keep', None)
    if keep is not None:
        return keep
    ax = _AxisT()
    ax.p_grid = tab.p_grid.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    ax.n_p = ctypes.c_int(len(tab.p_grid))
    ax.u_grid = tab.T_grid.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    ax.n_T = ctypes.c_int(len(tab.T_grid))
    ax.Tref = tab.Tref_grid.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    ax.is_norm = ctypes.c_int(1 if tab.normalised else 0)
    ax.k_sat = ctypes.c_int(getattr(tab, 'k_sat', -1))
    tab._axis_keep = (ax, tab.p_grid, tab.T_grid, tab.Tref_grid)
    return tab._axis_keep

def _build_C_argtypes():
    lib = _C_LIB
    lib.momentum_loop_c.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.c_double, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(_AxisT), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(_AxisT), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    lib.momentum_loop_c.restype = None
    lib.energy_loop_c.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(_AxisT), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(_AxisT), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
    lib.energy_loop_c.restype = None
    lib.table_lookup_bulk.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(_AxisT), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    lib.table_lookup_bulk.restype = None
    lib.update_wall_temps_c.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int]
    lib.update_wall_temps_c.restype = None
    lib.htc_bulk_c.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    lib.htc_bulk_c.restype = None

def _as_ptr(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

def _as_iptr(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

def _safe_ratio(cut, a):
    try:
        a = float(a)
        c = float(cut)
    except (TypeError, ValueError):
        return 0.0
    if a <= 0.0 or c <= 0.0:
        return 0.0
    return min(1.0, c / a)

def _table_lookup_bulk(tab, name, p, T):
    vals = np.ascontiguousarray(tab._vals[name].ravel(), dtype=np.float64)
    p = np.ascontiguousarray(np.asarray(p, dtype=np.float64).ravel())
    T = np.ascontiguousarray(np.asarray(T, dtype=np.float64).ravel())
    if p.size != T.size:
        raise ValueError("p and T must have the same size")
    out = np.empty(p.size, dtype=np.float64)
    _C_LIB.table_lookup_bulk(
        _as_ptr(vals), ctypes.byref(_axis_of(tab)[0]),
        _as_ptr(p), _as_ptr(T), _as_ptr(out), ctypes.c_int(p.size),
    )
    return out

_build_C_argtypes()
try:
    _C_LIB.prof_print_c.argtypes = []
    _C_LIB.prof_print_c.restype = None
except AttributeError:
    pass
lib = _C_LIB
lib.evaluate_post_c.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
lib.evaluate_post_c.restype = None

def steady_ice(mesh: SubchannelMesh, T_wall: np.ndarray | None=None, max_outer: int=200, tol: float=1e-06, verbose: bool=True, two_phase: bool=False, profile: bool=False, wall_max: float=0.0, stall_window: int=20, duty_tol: float=0.002, urf_htc: float=0.35, callback=None, dtw_tol: float=0.1, pt_tol: float=0.5, urf_p0: float=0.5, urf_T0: float=0.5, adapt_urf: bool=True, balance_flow: bool=False, balance_relax: float=1.0, balance_start: int=1, warm_start=None, want_f: bool=True) -> tuple[StateVector, ResidualHistory]:
    n_ch, n_ax = (mesh.n_ch(), mesh.n_ax())
    areas = np.array([ch.area for ch in mesh.channels])
    Dh = np.array([ch.Dh for ch in mesh.channels])
    P_h = np.array([ch.P_heated for ch in mesh.channels])
    _ki = np.array([float(getattr(ch, 'K_in', 0.0) or 0.0) for ch in mesh.channels])
    _ko = np.array([float(getattr(ch, 'K_out', 0.0) or 0.0) for ch in mesh.channels])
    k_in_c = np.ascontiguousarray(_ki, dtype=np.float64) if np.any(_ki > 0) else None
    k_out_c = np.ascontiguousarray(_ko, dtype=np.float64) if np.any(_ko > 0) else None
    dz = mesh.channels[0].dz
    n_wall = len(mesh.walls)
    has_ht = T_wall is not None or n_wall > 0
    ch_fluid = [getattr(ch, 'fluid', 'Water') for ch in mesh.channels]
    ch_tp = [two_phase and f == 'Water' for f in ch_fluid]
    wall_adj = [[] for _ in range(n_ch)]
    for wi, w in enumerate(mesh.walls):
        _pi, _pj, _ = w.widths()
        wall_adj[w.ch_i].append((wi, w.ch_j, w.t_w, w.k_w, _pi))
        wall_adj[w.ch_j].append((wi, w.ch_i, w.t_w, w.k_w, _pj))
    wall_offsets = [0]
    wall_other_l, wall_wi_l, wall_tw_l, wall_kw_l, wall_Pw_l = ([], [], [], [], [])
    for c in range(n_ch):
        for wi, other, tw, kw, Pw in wall_adj[c]:
            wall_other_l.append(other)
            wall_wi_l.append(wi)
            wall_tw_l.append(tw)
            wall_kw_l.append(kw)
            wall_Pw_l.append(Pw)
        wall_offsets.append(len(wall_other_l))
    wall_other = np.array(wall_other_l, dtype=np.int32)
    wall_wi_a = np.array(wall_wi_l, dtype=np.int32)
    wall_tw_a = np.array(wall_tw_l, dtype=np.float64)
    wall_kw_a = np.array(wall_kw_l, dtype=np.float64)
    wall_Pw_a = np.array(wall_Pw_l, dtype=np.float64)
    wall_offsets_a = np.array(wall_offsets, dtype=np.int32)
    wall_ch_i_a = np.array([w.ch_i for w in mesh.walls], dtype=np.int32)
    wall_ch_j_a = np.array([w.ch_j for w in mesh.walls], dtype=np.int32)
    _widths = [w.widths() for w in mesh.walls]
    wall_Pi_a = np.array([a for a, _, _ in _widths], dtype=np.float64)
    wall_Pj_a = np.array([b for _, b, _ in _widths], dtype=np.float64)
    wall_Pcond_a = np.array([c for _, _, c in _widths], dtype=np.float64)
    wall_one_a = np.array([1 if getattr(w, 'one_sided', False) else 0 for w in mesh.walls], dtype=np.int32)
    wall_idx_map = {}
    for wi, w in enumerate(mesh.walls):
        wall_idx_map[w.ch_i, w.ch_j] = wi
        wall_idx_map[w.ch_j, w.ch_i] = wi
    sv = StateVector(n_ch, n_ax, n_wall)
    for c in range(n_ch):
        fl = ch_fluid[c]
        ch = mesh.channels[c]
        sv.p[c, :] = ch.p_out
        sv.G[c, :] = ch.m_dot_in / ch.area
        T_in = ch.T_in
        T_out_est = T_in - 200.0 if ch.fluid == 'Helium' else T_in + 30.0
        fs = getattr(ch, 'flow_sign', 1)
        for j in range(n_ax):
            xi = j / (n_ax - 1.0)
            sv.T[c, j] = T_in * (1 - xi) + T_out_est * xi if fs == 1 else T_out_est * (1 - xi) + T_in * xi
        sv.rho[c] = _rho(ch.p_out, sv.T[c, 0], fluid=fl)
        sv.h[c] = _h(ch.p_out, sv.T[c, 0], fluid=fl)
    _ws_ok = False
    if warm_start is not None:
        try:
            if warm_start.T.shape == sv.T.shape and warm_start.rho.shape == sv.rho.shape and (warm_start.h.shape == sv.h.shape):
                sv.T[:] = warm_start.T
                sv.rho[:] = warm_start.rho
                sv.h[:] = warm_start.h
                _ws_ok = True
        except AttributeError:
            pass
    if _ws_ok and n_wall > 0:
        try:
            if warm_start.T_wall.shape == sv.T_wall.shape:
                sv.T_wall[:] = warm_start.T_wall
            else:
                _ws_ok = False
        except AttributeError:
            _ws_ok = False
    if _ws_ok:
        try:
            if warm_start.alpha.shape == sv.alpha.shape:
                sv.alpha[:] = warm_start.alpha
        except AttributeError:
            pass
    if n_wall > 0 and (not _ws_ok):
        for wi, w in enumerate(mesh.walls):
            T0 = 0.5 * (sv.T[w.ch_i, 0] + sv.T[w.ch_j, 0])
            sv.T_wall[wi, 0, :] = T0
            sv.T_wall[wi, 1, :] = T0
    if not _ws_ok:
        sv.alpha = np.zeros((n_ch, n_ax))
    urf_p = float(urf_p0)
    urf_T = float(urf_T0)
    urf_htc = float(urf_htc)
    dtw_tol = float(dtw_tol)
    pt_tol = float(pt_tol)
    stall_best = float('inf')
    stall = 0
    duty_q = 0.0
    duty_hist = []
    n_over = 0
    prev_r_p = 1.0
    prev_r_t = 1.0
    dp_ref = None
    dT_ref = None
    dh_ref = None
    last_change = 0
    hist = ResidualHistory()
    tab_w = _get_table('Water')
    p_grid = tab_w.p_grid
    T_grid_w = tab_w.T_grid
    n_p = len(p_grid)
    n_T_w = len(T_grid_w)
    Tsat_arr = np.array([_Tsat(p) for p in p_grid], dtype=np.float64)
    hl_grid = np.array([_cached_CP('H', float(p), _Tsat(p) - 0.5, 'Water') for p in p_grid], dtype=np.float64)
    hg_grid = np.array([_cached_CP('H', float(p), _Tsat(p) + 0.5, 'Water') for p in p_grid], dtype=np.float64)
    rho_l_grid = np.array([_rho(p, _Tsat(p) - 0.5, fluid='Water') for p in p_grid], dtype=np.float64)
    rho_g_grid = np.array([_rho(p, _Tsat(p) + 0.5, fluid='Water') for p in p_grid], dtype=np.float64)
    mu_l_grid = np.array([_cached_CP('V', p, _Tsat(p) - 0.5, 'Water') for p in p_grid], dtype=np.float64)
    mu_g_grid = np.array([_cached_CP('V', p, _Tsat(p) + 0.5, 'Water') for p in p_grid], dtype=np.float64)
    k_l_grid = np.array([_cached_CP('L', p, _Tsat(p) - 0.5, 'Water') for p in p_grid], dtype=np.float64)
    cp_l_grid = np.array([_cached_CP('C', p, _Tsat(p) - 0.5, 'Water') for p in p_grid], dtype=np.float64)
    sigma_arr = np.array([float(_CP.PropsSI('I', 'P', float(p), 'Q', 0.0, 'Water')) if p < _PCRIT * 0.9999 else 0.001 for p in p_grid], dtype=np.float64)
    sv.p = p_c = np.ascontiguousarray(sv.p, dtype=np.float64)
    sv.T = T_c = np.ascontiguousarray(sv.T, dtype=np.float64)
    sv.rho = rho_c = np.ascontiguousarray(sv.rho, dtype=np.float64)
    sv.h = h_c = np.ascontiguousarray(sv.h, dtype=np.float64)
    sv.alpha = alpha_c = np.ascontiguousarray(sv.alpha, dtype=np.float64)
    sv.T_wall = T_wall_c = np.ascontiguousarray(sv.T_wall, dtype=np.float64)
    sv.G = G_c = np.ascontiguousarray(sv.G, dtype=np.float64)
    assert np.shares_memory(G_c, sv.G), 'sv.G 与 C 缓冲不再共享内存，流量更新会静默失效'
    Dh_c = np.ascontiguousarray(Dh, dtype=np.float64)
    areas_c = np.ascontiguousarray(areas, dtype=np.float64)
    P_h_c = np.ascontiguousarray(P_h, dtype=np.float64)
    ch_tp_arr = np.array(ch_tp, dtype=np.int32)
    fluids_arr = np.array([0 if f == 'Water' else 1 for f in ch_fluid], dtype=np.int32)
    _SHAPE_CODE = {'circle': 0, 'square': 1, 'hexagon': 2, 'octagon': 3}

    def _shape_code(ch):
        cut = float(getattr(ch, 'cut', 0.0) or 0.0)
        a = float(getattr(ch, 'a', 0.0) or 0.0)
        nm = getattr(ch, 'shape', 'circle')
        if cut > 0.0 and a > 0.0:
            return 3
        return _SHAPE_CODE.get(nm, 0)
    ch_shape_arr = np.array([_shape_code(ch) for ch in mesh.channels], dtype=np.int32)
    if (ch_shape_arr == 3).any():
        ch_cut_arr = np.array([_safe_ratio(getattr(ch, 'cut', 0.0), getattr(ch, 'a', 0.0)) for ch in mesh.channels], dtype=np.float64)
        ch_cut_ptr = _as_ptr(ch_cut_arr)
    else:
        ch_cut_arr = None
        ch_cut_ptr = None
    p_out_c = np.array([mesh.channels[c].p_out for c in range(n_ch)], dtype=np.float64)
    flow_sign_arr = np.array([getattr(mesh.channels[c], 'flow_sign', 1) for c in range(n_ch)], dtype=np.int32)
    T_in_c = np.array([mesh.channels[c].T_in for c in range(n_ch)], dtype=np.float64)
    mu_tab_f = np.ascontiguousarray(tab_w._vals['mu'].ravel(), dtype=np.float64)
    k_tab_f = np.ascontiguousarray(tab_w._vals['k'].ravel(), dtype=np.float64)
    cp_tab_f = np.ascontiguousarray(tab_w._vals['cp'].ravel(), dtype=np.float64)
    rho_tab_f = np.ascontiguousarray(tab_w._vals['rho'].ravel(), dtype=np.float64)
    h_tab_f = np.ascontiguousarray(tab_w._vals['h'].ravel(), dtype=np.float64)
    p_grid_c = np.ascontiguousarray(p_grid, dtype=np.float64)
    T_grid_c = np.ascontiguousarray(T_grid_w, dtype=np.float64)
    _ax_w = ctypes.byref(_axis_of(tab_w)[0])
    Tsat_c = np.ascontiguousarray(Tsat_arr, dtype=np.float64)
    hl_c = np.ascontiguousarray(hl_grid, dtype=np.float64)
    hg_c = np.ascontiguousarray(hg_grid, dtype=np.float64)
    sigma_c = np.ascontiguousarray(sigma_arr, dtype=np.float64)
    k_l_c = np.ascontiguousarray(k_l_grid, dtype=np.float64)
    cp_l_c = np.ascontiguousarray(cp_l_grid, dtype=np.float64)
    tab_he = _get_table('Helium')
    mu_he_f = np.ascontiguousarray(tab_he._vals['mu'].ravel(), dtype=np.float64)
    k_he_f = np.ascontiguousarray(tab_he._vals['k'].ravel(), dtype=np.float64)
    cp_he_f = np.ascontiguousarray(tab_he._vals['cp'].ravel(), dtype=np.float64)
    rho_he_f = np.ascontiguousarray(tab_he._vals['rho'].ravel(), dtype=np.float64)
    h_he_f = np.ascontiguousarray(tab_he._vals['h'].ravel(), dtype=np.float64)
    p_grid_he_c = np.ascontiguousarray(tab_he.p_grid, dtype=np.float64)
    T_grid_he_c = np.ascontiguousarray(tab_he.T_grid, dtype=np.float64)
    n_p_he = len(tab_he.p_grid)
    n_T_he = len(tab_he.T_grid)
    _ax_he = ctypes.byref(_axis_of(tab_he)[0])
    T_wall_ext_c = np.ascontiguousarray(T_wall, dtype=np.float64) if T_wall is not None else None
    htc_film_all_c = np.zeros((n_ch, n_ax), dtype=np.float64)
    cp_film_all_c = np.zeros((n_ch, n_ax), dtype=np.float64)
    if _ws_ok:
        try:
            if warm_start.htc_film.shape == htc_film_all_c.shape:
                htc_film_all_c[:] = warm_start.htc_film
        except AttributeError:
            pass
    q_net_c = np.zeros((n_ch, n_ax), dtype=np.float64)
    pf = {'total': 0.0, 'momentum': 0.0, 'energy': 0.0, 'residual': 0.0}
    if verbose:
        print(f"{'Step':>6} {'Res(P)':>12} {'Res(T)':>12}")
        print('-' * 42)
    _p_all = np.concatenate([p_out_c, np.asarray(sv.p)[:, 0], np.asarray(sv.p)[:, -1]])
    p_span = max(float(np.ptp(_p_all)), 1000.0)
    _tin = np.asarray(T_in_c, dtype=np.float64)
    T_span = max(float(np.ptp(_tin)), 50.0)
    h_span = max(float(np.ptp(_tin)) * 5200.0, 100000.0)
    if verbose:
        print(f'  scale: p {p_span:.4g} Pa, T {T_span:.4g} K, h {h_span:.4g} J/kg')
    t_total = time.perf_counter()
    t_start = t_total
    _fs_arr = np.asarray(flow_sign_arr, dtype=int)
    _fl_arr = np.asarray(fluids_arr, dtype=int)
    _po_arr = np.asarray([mesh.channels[c].p_out for c in range(n_ch)], dtype=float)
    _m_tot = {}
    for _f in (0, 1):
        _sel0 = np.where(_fl_arr == _f)[0]
        _m_tot[_f] = float(sum((mesh.channels[c].m_dot_in for c in _sel0)))
    _bal_g = 9.81

    def _flow_balance(sv, relax):
        dmax = 0.0
        for _f, _tot in _m_tot.items():
            sel = np.where(_fl_arr == _f)[0]
            if len(sel) < 2 or _tot <= 0.0:
                continue
            fs = _fs_arr[sel].astype(float)
            pins = np.where(fs > 0, sv.p[sel, 0], sv.p[sel, -1])
            pout = _po_arr[sel]
            rho = sv.rho[sel]
            Gc = sv.G[sel]
            dp_grav = fs * _bal_g * dz * np.sum(rho, axis=1)
            dp_acc = fs * np.sum(Gc[:, 1:n_ax] * Gc[:, 1:n_ax] * (1.0 / rho[:, 1:] - 1.0 / rho[:, :-1]), axis=1)
            m = np.array([mesh.channels[c].m_dot_in for c in sel], dtype=float)
            a = a_lin_c[sel] / areas[sel]
            a = np.maximum(a, 1e-06 * np.median(a[a > 0]))
            w = 1.0 / a
            P = (_tot + np.sum(w * (pout + dp_grav + dp_acc))) / np.sum(w)
            m_new = (P - pout - dp_grav - dp_acc) * w
            m_new = np.maximum(m_new, 0.02 * m.mean())
            m_new *= _tot / np.sum(m_new)
            if relax < 1.0:
                m_new = (1.0 - relax) * m + relax * m_new
                m_new *= _tot / np.sum(m_new)
            dmax = max(dmax, float(np.max(np.abs(np.log(m_new / np.maximum(m, 1e-30))))))
            for c, v in zip(sel, m_new):
                mesh.channels[c].m_dot_in = float(v)
            sv.G[sel, :] = (m_new / areas[sel])[:, None]
        return dmax
    bal_hist = []
    a_lin_c = np.zeros(n_ch, dtype=np.float64) if balance_flow else None
    dp_fric_c = np.zeros(n_ch, dtype=np.float64) if want_f else None
    dt_max = dt_p99 = dTw_l2 = r_h = dh_p99 = float('nan')
    for outer in range(1, max_outer + 1):
        if wall_max > 0 and time.perf_counter() - t_start > wall_max:
            if verbose:
                print(f'\nAborted at iter {outer} (wall clock exceeded {wall_max:.0f}s)')
            break
        p_old = sv.p.copy()
        T_old = sv.T.copy()
        Tw_old = sv.T_wall.copy()
        h_old = sv.h.copy()
        tm = time.perf_counter()
        _C_LIB.momentum_loop_c(_as_ptr(p_c), _as_ptr(T_c), _as_ptr(rho_c), _as_ptr(h_c), _as_ptr(G_c), _as_ptr(Dh_c), ctypes.c_double(dz), ctypes.c_int(n_ch), ctypes.c_int(n_ax), _as_ptr(k_in_c) if k_in_c is not None else None, _as_ptr(k_out_c) if k_out_c is not None else None, _as_ptr(a_lin_c) if balance_flow else None, _as_ptr(dp_fric_c) if want_f else None, _as_iptr(ch_tp_arr), _as_ptr(p_out_c), _as_iptr(flow_sign_arr), ctypes.c_double(urf_p), _as_iptr(fluids_arr), _as_ptr(mu_tab_f), _as_ptr(k_tab_f), _as_ptr(cp_tab_f), _as_ptr(rho_tab_f), _as_ptr(h_tab_f), _as_ptr(p_grid_c), ctypes.c_int(n_p), _ax_w, _as_ptr(mu_he_f), _as_ptr(k_he_f), _as_ptr(cp_he_f), _as_ptr(rho_he_f), _as_ptr(h_he_f), _as_ptr(p_grid_he_c), ctypes.c_int(n_p_he), _ax_he, _as_ptr(Tsat_c), _as_ptr(hl_c), _as_ptr(hg_c), _as_ptr(rho_l_grid), _as_ptr(rho_g_grid), _as_ptr(mu_l_grid), _as_ptr(mu_g_grid), _as_ptr(sigma_c))
        pf['momentum'] += time.perf_counter() - tm
        if balance_flow and outer >= balance_start:
            _db = _flow_balance(sv, balance_relax)
            bal_hist.append(_db)
        te = time.perf_counter()
        htc_bulk = np.zeros((n_ch, n_ax))
        htc_bulk_c = np.ascontiguousarray(htc_bulk, dtype=np.float64)
        if n_wall > 0:
            _acc = np.zeros((n_ch, n_ax))
            _cnt = np.zeros((n_ch, n_ax))
            _ci = np.asarray(wall_ch_i_a)
            _cj = np.asarray(wall_ch_j_a)
            np.add.at(_acc, _ci, sv.T_wall[:, 0, :])
            np.add.at(_cnt, _ci, 1.0)
            _two = ~np.asarray(wall_one_a, dtype=bool)
            if _two.any():
                np.add.at(_acc, _cj[_two], sv.T_wall[_two, 1, :])
                np.add.at(_cnt, _cj[_two], 1.0)
            T_film_c = np.ascontiguousarray(0.5 * (sv.T + np.divide(_acc, _cnt, out=np.array(sv.T), where=_cnt > 0)), dtype=np.float64)
        else:
            T_film_c = np.ascontiguousarray(sv.T, dtype=np.float64)
        _C_LIB.energy_loop_c(_as_ptr(p_c), _as_ptr(T_c), _as_ptr(rho_c), _as_ptr(h_c), _as_ptr(G_c), _as_ptr(alpha_c), _as_ptr(T_wall_c), _as_ptr(Dh_c), _as_ptr(areas_c), _as_ptr(P_h_c), ctypes.c_double(dz), ctypes.c_int(n_ch), ctypes.c_int(n_ax), _as_iptr(ch_tp_arr), _as_ptr(T_in_c), _as_iptr(flow_sign_arr), ctypes.c_double(urf_T), ctypes.c_double(urf_htc), _as_iptr(wall_ch_i_a), _as_iptr(wall_ch_j_a), ctypes.c_int(n_wall), _as_ptr(wall_tw_a), _as_ptr(wall_kw_a), _as_ptr(wall_Pi_a), _as_ptr(wall_Pj_a), _as_ptr(wall_Pcond_a), _as_iptr(wall_one_a), _as_iptr(fluids_arr), _as_iptr(ch_shape_arr), ch_cut_ptr, _as_ptr(mu_tab_f), _as_ptr(k_tab_f), _as_ptr(cp_tab_f), _as_ptr(rho_tab_f), _as_ptr(h_tab_f), _as_ptr(p_grid_c), ctypes.c_int(n_p), _ax_w, _as_ptr(mu_he_f), _as_ptr(k_he_f), _as_ptr(cp_he_f), _as_ptr(rho_he_f), _as_ptr(h_he_f), _as_ptr(p_grid_he_c), ctypes.c_int(n_p_he), _ax_he, _as_ptr(Tsat_c), _as_ptr(hl_c), _as_ptr(hg_c), _as_ptr(rho_l_grid), _as_ptr(rho_g_grid), _as_ptr(mu_l_grid), _as_ptr(mu_g_grid), _as_ptr(k_l_c), _as_ptr(cp_l_c), _as_ptr(sigma_c), _as_ptr(htc_bulk_c), _as_ptr(htc_film_all_c), _as_ptr(cp_film_all_c), _as_ptr(q_net_c), _as_ptr(T_wall_ext_c) if T_wall_ext_c is not None else None, _as_ptr(T_film_c))
        if n_wall > 0:
            _C_LIB.update_wall_temps_c(_as_ptr(T_wall_c), _as_ptr(T_c), _as_ptr(htc_bulk_c), _as_iptr(wall_ch_i_a), _as_iptr(wall_ch_j_a), _as_ptr(wall_tw_a), _as_ptr(wall_kw_a), _as_ptr(wall_Pi_a), _as_ptr(wall_Pj_a), _as_ptr(wall_Pcond_a), ctypes.c_int(n_wall), ctypes.c_int(n_ax))
        pf['energy'] += time.perf_counter() - te
        tr = time.perf_counter()
        _wp = max(urf_p, 0.001)
        _wT = max(urf_T, 0.001)
        dp_l2 = float(np.sqrt(np.mean((sv.p - p_old) ** 2))) / _wp
        dT_l2 = float(np.sqrt(np.mean((sv.T - T_old) ** 2))) / _wT if has_ht else 0.0
        dTw_l2 = float(np.sqrt(np.mean((sv.T_wall - Tw_old) ** 2))) if n_wall > 0 else 0.0
        dh_l2 = float(np.sqrt(np.mean((sv.h - h_old) ** 2))) / _wT if has_ht else 0.0
        dh_p99 = float(np.percentile(np.abs(sv.h - h_old), 99.0)) if has_ht else 0.0
        r_h = dh_l2 / h_span if has_ht else 0.0
        _dT_cell = np.abs(sv.T - T_old) if has_ht else None
        if _dT_cell is not None:
            dt_max = float(_dT_cell.max())
            dt_p99 = float(np.percentile(_dT_cell, 99.0))
        else:
            dt_max = dt_p99 = 0.0
        r_p = dp_l2 / p_span
        r_t = dT_l2 / T_span if has_ht else r_p
        hist.append(r_p, r_t, r_p)
        if outer > 1 and adapt_urf:
            if r_p < prev_r_p:
                urf_p = min(urf_p * 1.08, 0.8)
            else:
                urf_p = max(urf_p * 0.7, 0.15)
            if r_t < prev_r_t:
                urf_T = min(urf_T * 1.08, 0.8)
            else:
                urf_T = max(urf_T * 0.7, 0.15)
        prev_r_p = r_p
        prev_r_t = r_t
        pf['residual'] += time.perf_counter() - tr
        if callback is not None:
            if callback(sv, outer, r_p, r_t, dTw_l2):
                last_change = outer
                dp_ref = None
                dT_ref = None
                dh_ref = None
                prev_r_p = 1.0
                prev_r_t = 1.0
                stall_best = float('inf')
                stall = 0
                duty_hist = []
        if verbose and (outer <= 3 or outer % 20 == 0 or r_p < tol):
            print(f'{outer:>6} {r_p:>12.2e} {r_t:>12.2e}')
        p_ref_abs = float(np.mean(sv.p))
        converged_rel = r_p < tol and r_t < tol and (r_h < tol) and (dTw_l2 < dtw_tol or not has_ht) and (dt_p99 < pt_tol or not has_ht)
        converged_abs = False
        stalled_rel = False
        if has_ht and r_p < tol and (r_t < tol) and (dTw_l2 < 5.0):
            if dTw_l2 < stall_best * 0.98:
                stall_best = dTw_l2
                stall = 0
            else:
                stall += 1
                if stall >= stall_window:
                    stalled_rel = True
        stalled_duty = False
        if outer > 2 and (last_change == 0 or r_p < tol):
            _dh = np.abs(sv.h[:, -1] - sv.h[:, 0])
            duty_q = float(np.sum(_dh))
            duty_hist.append(duty_q)
            if len(duty_hist) > stall_window:
                duty_hist.pop(0)
            if len(duty_hist) == stall_window:
                _lo = min(duty_hist)
                _hi = max(duty_hist)
                if _hi > 0.0 and _hi - _lo <= duty_tol * _hi:
                    stalled_duty = True
        _tmax = float(np.max(sv.T)) if np.all(np.isfinite(sv.T)) else float('inf')
        if _tmax > DIVERGED_T:
            n_over += 1
        else:
            n_over = 0
        if n_over >= 3:
            hist.diverged = True
            hist.exit_reason = 'diverged'
            break
        quiet = outer - last_change >= stall_window
        if (converged_rel or stalled_rel or stalled_duty) and outer > 2 and quiet:
            if verbose:
                how = 'Converged' if converged_rel else 'Stalled (limit cycle)' if stalled_rel else 'Stalled (duty frozen)'
                print(f'\n{how} in {outer} iterations')
            hist.stalled = stalled_rel
            hist.exit_reason = 'residual' if converged_rel else 'stall_rel' if stalled_rel else 'stall_duty'
            break
    hist.dt_max_final = dt_max
    hist.dt_p99_final = dt_p99
    hist.dtw_l2_final = dTw_l2
    hist.rh_final = r_h
    hist.balance_hist = bal_hist
    hist.dh_p99_final = dh_p99
    if not hist.exit_reason:
        hist.exit_reason = 'max_outer'
    pf['total'] = time.perf_counter() - t_total
    if profile:
        print('\n===== PROFILE =====')
        for k in ['momentum', 'energy', 'residual', 'total']:
            if pf[k] > 0:
                print(f"  {k:15s}: {pf[k] * 1000:8.2f} ms ({pf[k] / max(pf['total'], 1e-30) * 100:5.1f}%)")
        print('=' * 24)
        _C_LIB.prof_print_c()
    if verbose:
        for c in range(n_ch):
            fs = flow_sign_arr[c]
            dp_flow = sv.p[c, -1] - sv.p[c, 0] if fs == -1 else sv.p[c, 0] - sv.p[c, -1]
            if dp_flow < 0:
                print(f'  WARNING: channel {c} negative flow-direction pressure drop: {dp_flow:.0f} Pa')
            if ch_tp[c]:
                alpha_max = float(np.max(sv.alpha[c]))
                if alpha_max > 0.95:
                    print(f'  WARNING: channel {c} high void fraction: alpha_max={alpha_max:.3f}')
    sv.htc_film[:] = htc_film_all_c
    if want_f and dp_fric_c is not None:
        sv.dp_fric = np.array(dp_fric_c, dtype=float)
    return (sv, hist)
