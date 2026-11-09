"""
pyHT steady-state ICE solver — C-accelerated hot loops only.

The C extension (_solver_c.dll) is required; there is no Python fallback.
Pre-computed property tables and pre-allocated working arrays in C
give 10-50x speedup over pure Python.
"""

import os, time, ctypes
import numpy as np
import CoolProp.CoolProp as _CP
from ..core.topology import SubchannelMesh
from ..core.variables import StateVector, ResidualHistory
from ..fluid import _get_table

# ---- require C extension ----
_CDIR = os.path.dirname(__file__)
_C_LIB = ctypes.CDLL(os.path.join(_CDIR, '_solver_c.dll'))
print("[pyht] C extension loaded: _solver_c.dll")


# ---- direct CoolProp scalar wrappers with unified cache ----
_prop_cache = {}
def _cached_CP(key_str, p, T, fluid="Water"):
    ck = (fluid, key_str, round(p, -2), round(T, 1))
    if ck in _prop_cache: return _prop_cache[ck]
    try:
        v = float(_CP.PropsSI(key_str, "P", float(p), "T", float(T), fluid))
        _prop_cache[ck] = v; return v
    except Exception:
        return 0.0

def _rho(p, T, fluid="Water"):
    v = _cached_CP("D", p, T, fluid)
    return v if v > 0 else 1000.0

def _h(p, T, fluid="Water"):
    v = _cached_CP("H", p, T, fluid)
    return v if v > 0 else 4e5

def _h_direct(p, T):
    """Direct CoolProp enthalpy for water (used for saturation comparisons)."""
    return _cached_CP("H", p, T, "Water") or 4e5

_Tsat_cache = {}
def _Tsat(p):
    key = round(p, -2)
    if key in _Tsat_cache: return _Tsat_cache[key]
    try:
        v = float(_CP.PropsSI("T", "P", float(p), "Q", 0.0, "Water"))
        _Tsat_cache[key] = v; return v
    except Exception: return 373.15

def _quality(h, h_l, h_g):
    if h_l >= h_g - 1: return 0.0
    return float(np.clip((h - h_l) / (h_g - h_l), 0.0, 1.0))


# ---- C extension wrappers ----

def _build_C_argtypes():
    """Define argument types for C functions."""
    lib = _C_LIB
    # momentum_loop_c
    lib.momentum_loop_c.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # p
        ctypes.POINTER(ctypes.c_double),  # T
        ctypes.POINTER(ctypes.c_double),  # rho
        ctypes.POINTER(ctypes.c_double),  # h
        ctypes.POINTER(ctypes.c_double),  # G
        ctypes.POINTER(ctypes.c_double),  # Dh
        ctypes.c_double,                   # dz
        ctypes.c_int, ctypes.c_int,        # n_ch, n_ax
        ctypes.POINTER(ctypes.c_int),     # ch_tp
        ctypes.POINTER(ctypes.c_double),  # p_out
        ctypes.POINTER(ctypes.c_int),     # flow_sign
        ctypes.c_double,                   # urf_p
        ctypes.POINTER(ctypes.c_int),     # fluids (0=Water, 1=Helium)
        ctypes.POINTER(ctypes.c_double),  # mu_tab_w
        ctypes.POINTER(ctypes.c_double),  # k_tab_w
        ctypes.POINTER(ctypes.c_double),  # cp_tab_w
        ctypes.POINTER(ctypes.c_double),  # rho_tab_w
        ctypes.POINTER(ctypes.c_double),  # h_tab_w
        ctypes.POINTER(ctypes.c_double),  # p_grid_w
        ctypes.c_int,                      # n_p_w
        ctypes.POINTER(ctypes.c_double),  # T_grid_w
        ctypes.c_int,                      # n_T_w
        ctypes.POINTER(ctypes.c_double),  # mu_tab_he
        ctypes.POINTER(ctypes.c_double),  # k_tab_he
        ctypes.POINTER(ctypes.c_double),  # cp_tab_he
        ctypes.POINTER(ctypes.c_double),  # rho_tab_he
        ctypes.POINTER(ctypes.c_double),  # h_tab_he
        ctypes.POINTER(ctypes.c_double),  # p_grid_he
        ctypes.c_int,                      # n_p_he
        ctypes.POINTER(ctypes.c_double),  # T_grid_he
        ctypes.c_int,                      # n_T_he
        ctypes.POINTER(ctypes.c_double),  # Tsat_grid
        ctypes.POINTER(ctypes.c_double),  # hl_sat_grid (1D)
        ctypes.POINTER(ctypes.c_double),  # hg_sat_grid (1D)
        ctypes.POINTER(ctypes.c_double),  # rho_l_sat_grid (1D)
        ctypes.POINTER(ctypes.c_double),  # rho_g_sat_grid
        ctypes.POINTER(ctypes.c_double),  # mu_l_sat_grid
        ctypes.POINTER(ctypes.c_double),  # mu_g_sat_grid
        ctypes.POINTER(ctypes.c_double),  # sigma_tab
    ]
    lib.momentum_loop_c.restype = None

    # energy_loop_c
    lib.energy_loop_c.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # p
        ctypes.POINTER(ctypes.c_double),  # T
        ctypes.POINTER(ctypes.c_double),  # rho
        ctypes.POINTER(ctypes.c_double),  # h
        ctypes.POINTER(ctypes.c_double),  # G
        ctypes.POINTER(ctypes.c_double),  # alpha
        ctypes.POINTER(ctypes.c_double),  # T_wall
        ctypes.POINTER(ctypes.c_double),  # Dh
        ctypes.POINTER(ctypes.c_double),  # areas
        ctypes.POINTER(ctypes.c_double),  # P_h
        ctypes.c_double,                   # dz
        ctypes.c_int, ctypes.c_int,        # n_ch, n_ax
        ctypes.POINTER(ctypes.c_int),     # ch_tp
        ctypes.POINTER(ctypes.c_double),  # T_in
        ctypes.POINTER(ctypes.c_int),     # flow_sign
        ctypes.c_double,                   # urf_T
        ctypes.POINTER(ctypes.c_int),     # wall_ch_i
        ctypes.POINTER(ctypes.c_int),     # wall_ch_j
        ctypes.c_int,                      # n_wall
        ctypes.POINTER(ctypes.c_double),  # wall_tw
        ctypes.POINTER(ctypes.c_double),  # wall_kw
        ctypes.POINTER(ctypes.c_double),  # wall_Pw
        ctypes.POINTER(ctypes.c_int),     # fluids (0=Water, 1=Helium)
        ctypes.POINTER(ctypes.c_double),  # mu_tab_w
        ctypes.POINTER(ctypes.c_double),  # k_tab_w
        ctypes.POINTER(ctypes.c_double),  # cp_tab_w
        ctypes.POINTER(ctypes.c_double),  # rho_tab_w
        ctypes.POINTER(ctypes.c_double),  # h_tab_w
        ctypes.POINTER(ctypes.c_double),  # p_grid_w
        ctypes.c_int,                      # n_p_w
        ctypes.POINTER(ctypes.c_double),  # T_grid_w
        ctypes.c_int,                      # n_T_w
        ctypes.POINTER(ctypes.c_double),  # mu_tab_he
        ctypes.POINTER(ctypes.c_double),  # k_tab_he
        ctypes.POINTER(ctypes.c_double),  # cp_tab_he
        ctypes.POINTER(ctypes.c_double),  # rho_tab_he
        ctypes.POINTER(ctypes.c_double),  # h_tab_he
        ctypes.POINTER(ctypes.c_double),  # p_grid_he
        ctypes.c_int,                      # n_p_he
        ctypes.POINTER(ctypes.c_double),  # T_grid_he
        ctypes.c_int,                      # n_T_he
        ctypes.POINTER(ctypes.c_double),  # Tsat_grid
        ctypes.POINTER(ctypes.c_double),  # hl_grid (sat liquid, 1D p)
        ctypes.POINTER(ctypes.c_double),  # hg_grid (sat vapor, 1D p)
        ctypes.POINTER(ctypes.c_double),  # rho_l_sat_grid (1D, new in v0.2)
        ctypes.POINTER(ctypes.c_double),  # rho_g_sat_grid
        ctypes.POINTER(ctypes.c_double),  # mu_l_sat_grid
        ctypes.POINTER(ctypes.c_double),  # mu_g_sat_grid
        ctypes.POINTER(ctypes.c_double),  # sigma_tab
        ctypes.POINTER(ctypes.c_double),  # htc_bulk
        ctypes.POINTER(ctypes.c_double),  # htc_film_all (pre-allocated)
        ctypes.POINTER(ctypes.c_double),  # cp_film_all (pre-allocated)
        ctypes.POINTER(ctypes.c_double),  # q_net (pre-allocated)
        ctypes.c_void_p,                   # T_wall_ext (NULL = none)
        ctypes.POINTER(ctypes.c_double),  # T_film
    ]
    lib.energy_loop_c.restype = None

    # update_wall_temps_c
    lib.update_wall_temps_c.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # T_wall
        ctypes.POINTER(ctypes.c_double),  # T
        ctypes.POINTER(ctypes.c_double),  # htc_bulk
        ctypes.POINTER(ctypes.c_int),     # wall_ch_i
        ctypes.POINTER(ctypes.c_int),     # wall_ch_j
        ctypes.POINTER(ctypes.c_double),  # wall_tw
        ctypes.POINTER(ctypes.c_double),  # wall_kw
        ctypes.POINTER(ctypes.c_double),  # wall_Pw
        ctypes.c_int,                      # n_wall
        ctypes.c_int,                      # n_ax
    ]
    lib.update_wall_temps_c.restype = None

    # htc_bulk_c
    lib.htc_bulk_c.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # htc_out
        ctypes.POINTER(ctypes.c_double),  # p
        ctypes.POINTER(ctypes.c_double),  # T
        ctypes.POINTER(ctypes.c_double),  # G
        ctypes.POINTER(ctypes.c_double),  # Dh
        ctypes.c_int, ctypes.c_int,        # n_ch, n_ax
        ctypes.POINTER(ctypes.c_int),     # fluids
        ctypes.POINTER(ctypes.c_double),  # mu_w etc
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
    ]
    lib.htc_bulk_c.restype = None

def _as_ptr(arr):
    """Convert numpy array to ctypes pointer."""
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

def _as_iptr(arr):
    """Convert int32 numpy array to ctypes pointer."""
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

_build_C_argtypes()
try:
    _C_LIB.prof_print_c.argtypes = []
    _C_LIB.prof_print_c.restype = None
except AttributeError:
    pass

lib = _C_LIB
lib.evaluate_post_c.argtypes = [
    ctypes.POINTER(ctypes.c_double),   # p
    ctypes.POINTER(ctypes.c_double),   # T
    ctypes.POINTER(ctypes.c_double),   # rho
    ctypes.POINTER(ctypes.c_double),   # h
    ctypes.POINTER(ctypes.c_double),   # T_wall
    ctypes.c_int,                       # n_ch
    ctypes.c_int,                       # n_ax
    ctypes.c_int,                       # n_wall
    ctypes.POINTER(ctypes.c_int),      # flow_sign
    ctypes.POINTER(ctypes.c_int),      # fluids
    ctypes.POINTER(ctypes.c_double),   # m_dot_in
    ctypes.POINTER(ctypes.c_double),   # Qh (out)
    ctypes.POINTER(ctypes.c_double),   # Qw (out)
    ctypes.POINTER(ctypes.c_double),   # dph (out)
    ctypes.POINTER(ctypes.c_double),   # dpw (out)
    ctypes.POINTER(ctypes.c_double),   # Tha (out)
    ctypes.POINTER(ctypes.c_double),   # Twa (out)
    ctypes.POINTER(ctypes.c_double),   # rha (out)
    ctypes.POINTER(ctypes.c_double),   # rwa (out)
    ctypes.POINTER(ctypes.c_double),   # Twm (out)
]
lib.evaluate_post_c.restype = None


# ---- solver ----

def steady_ice(
    mesh: SubchannelMesh,
    T_wall: np.ndarray | None = None,
    max_outer: int = 200,
    tol: float = 1e-6,
    verbose: bool = True,
    two_phase: bool = False,
    profile: bool = False,
    wall_max: float = 0.0,
) -> tuple[StateVector, ResidualHistory]:
    """C-accelerated steady-state ICE solver for subchannel heat exchangers.
    If *wall_max* > 0, abort after that many seconds and return partial state.
    """
    n_ch, n_ax = mesh.n_ch(), mesh.n_ax()
    areas = np.array([ch.area for ch in mesh.channels])
    Dh = np.array([ch.Dh for ch in mesh.channels])
    P_h = np.array([ch.P_heated for ch in mesh.channels])
    dz = mesh.channels[0].dz
    n_wall = len(mesh.walls)
    has_ht = T_wall is not None or n_wall > 0

    ch_fluid = [getattr(ch, 'fluid', 'Water') for ch in mesh.channels]
    ch_tp = [two_phase and f == "Water" for f in ch_fluid]

    wall_adj = [[] for _ in range(n_ch)]
    for wi, w in enumerate(mesh.walls):
        wall_adj[w.ch_i].append((wi, w.ch_j, w.t_w, w.k_w, w.P_w))
        wall_adj[w.ch_j].append((wi, w.ch_i, w.t_w, w.k_w, w.P_w))

    # ---- build C-compatible wall adjacency arrays ----
    wall_offsets = [0]
    wall_other_l, wall_wi_l, wall_tw_l, wall_kw_l, wall_Pw_l = [], [], [], [], []
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
    # wall index lookup: for each (ch_i, ch_j) pair, find the original wall index
    # We rebuild this from mesh data
    wall_idx_map = {}
    for wi, w in enumerate(mesh.walls):
        wall_idx_map[(w.ch_i, w.ch_j)] = wi
        wall_idx_map[(w.ch_j, w.ch_i)] = wi

    sv = StateVector(n_ch, n_ax, n_wall)
    for c in range(n_ch):
        fl = ch_fluid[c]; ch = mesh.channels[c]
        sv.p[c, :] = ch.p_out; sv.G[c, :] = ch.m_dot_in / ch.area
        # linear T gradient from inlet to estimated outlet
        T_in = ch.T_in
        T_out_est = T_in - 200.0 if ch.fluid == "Helium" else T_in + 30.0
        fs = getattr(ch, 'flow_sign', 1)
        for j in range(n_ax):
            xi = j / (n_ax - 1.0)
            sv.T[c, j] = T_in * (1 - xi) + T_out_est * xi if fs == 1 else T_out_est * (1 - xi) + T_in * xi
        sv.rho[c] = _rho(ch.p_out, sv.T[c, 0], fluid=fl)
        sv.h[c] = _h(ch.p_out, sv.T[c, 0], fluid=fl)

    if n_wall > 0:
        for wi, w in enumerate(mesh.walls):
            T0 = 0.5 * (sv.T[w.ch_i, 0] + sv.T[w.ch_j, 0])
            sv.T_wall[wi, 0, :] = T0; sv.T_wall[wi, 1, :] = T0

    sv.alpha = np.zeros((n_ch, n_ax))
    urf_p = 0.5; urf_T = 0.5
    prev_r_p = 1.0; prev_r_t = 1.0
    hist = ResidualHistory()

    # ---- prepare property table data for C ----
    tab_w = _get_table("Water")
    p_grid = tab_w.p_grid
    T_grid_w = tab_w.T_grid
    n_p = len(p_grid)
    n_T_w = len(T_grid_w)
    # saturation property grids (1D, function of pressure only — CoolProp sat lookup)
    Tsat_arr = np.array([_Tsat(p) for p in p_grid], dtype=np.float64)
    hl_grid = np.array([_cached_CP("H", float(p), _Tsat(p) - 0.5, "Water") for p in p_grid], dtype=np.float64)
    hg_grid = np.array([_cached_CP("H", float(p), _Tsat(p) + 0.5, "Water") for p in p_grid], dtype=np.float64)
    rho_l_grid = np.array([_rho(p, _Tsat(p) - 0.5, fluid="Water") for p in p_grid], dtype=np.float64)
    rho_g_grid = np.array([_rho(p, _Tsat(p) + 0.5, fluid="Water") for p in p_grid], dtype=np.float64)
    mu_l_grid = np.array([_cached_CP("V", p, _Tsat(p) - 0.5, "Water") for p in p_grid], dtype=np.float64)
    mu_g_grid = np.array([_cached_CP("V", p, _Tsat(p) + 0.5, "Water") for p in p_grid], dtype=np.float64)
    sigma_arr = np.array([
        float(_CP.PropsSI("I", "P", float(p), "Q", 0.0, "Water"))
        if p < 2.2e7 else 0.001 for p in p_grid], dtype=np.float64)

    # arrays must be C-contiguous float64; share memory with sv to avoid copy-back
    sv.p = p_c = np.ascontiguousarray(sv.p, dtype=np.float64)
    sv.T = T_c = np.ascontiguousarray(sv.T, dtype=np.float64)
    sv.rho = rho_c = np.ascontiguousarray(sv.rho, dtype=np.float64)
    sv.h = h_c = np.ascontiguousarray(sv.h, dtype=np.float64)
    sv.alpha = alpha_c = np.ascontiguousarray(sv.alpha, dtype=np.float64)
    sv.T_wall = T_wall_c = np.ascontiguousarray(sv.T_wall, dtype=np.float64)
    sv.G = G_c = np.ascontiguousarray(sv.G, dtype=np.float64)
    Dh_c = np.ascontiguousarray(Dh, dtype=np.float64)
    areas_c = np.ascontiguousarray(areas, dtype=np.float64)
    P_h_c = np.ascontiguousarray(P_h, dtype=np.float64)
    ch_tp_arr = np.array(ch_tp, dtype=np.int32)
    fluids_arr = np.array([0 if f == "Water" else 1 for f in ch_fluid], dtype=np.int32)
    p_out_c = np.array([mesh.channels[c].p_out for c in range(n_ch)], dtype=np.float64)
    flow_sign_arr = np.array([getattr(mesh.channels[c], 'flow_sign', 1) for c in range(n_ch)], dtype=np.int32)
    T_in_c = np.array([mesh.channels[c].T_in for c in range(n_ch)], dtype=np.float64)
    # property table values (flatten to 1D)
    mu_tab_f = np.ascontiguousarray(tab_w._vals["mu"].ravel(), dtype=np.float64)
    k_tab_f = np.ascontiguousarray(tab_w._vals["k"].ravel(), dtype=np.float64)
    cp_tab_f = np.ascontiguousarray(tab_w._vals["cp"].ravel(), dtype=np.float64)
    rho_tab_f = np.ascontiguousarray(tab_w._vals["rho"].ravel(), dtype=np.float64)
    h_tab_f = np.ascontiguousarray(tab_w._vals["h"].ravel(), dtype=np.float64)
    p_grid_c = np.ascontiguousarray(p_grid, dtype=np.float64)
    T_grid_c = np.ascontiguousarray(T_grid_w, dtype=np.float64)
    Tsat_c = np.ascontiguousarray(Tsat_arr, dtype=np.float64)
    hl_c = np.ascontiguousarray(hl_grid, dtype=np.float64)
    hg_c = np.ascontiguousarray(hg_grid, dtype=np.float64)
    sigma_c = np.ascontiguousarray(sigma_arr, dtype=np.float64)
    # Helium tables
    tab_he = _get_table("Helium")
    mu_he_f = np.ascontiguousarray(tab_he._vals["mu"].ravel(), dtype=np.float64)
    k_he_f  = np.ascontiguousarray(tab_he._vals["k"].ravel(), dtype=np.float64)
    cp_he_f = np.ascontiguousarray(tab_he._vals["cp"].ravel(), dtype=np.float64)
    rho_he_f = np.ascontiguousarray(tab_he._vals["rho"].ravel(), dtype=np.float64)
    h_he_f   = np.ascontiguousarray(tab_he._vals["h"].ravel(), dtype=np.float64)
    p_grid_he_c = np.ascontiguousarray(tab_he.p_grid, dtype=np.float64)
    T_grid_he_c = np.ascontiguousarray(tab_he.T_grid, dtype=np.float64)
    n_p_he = len(tab_he.p_grid)
    n_T_he = len(tab_he.T_grid)
    T_wall_ext_c = np.ascontiguousarray(T_wall, dtype=np.float64) if T_wall is not None else None
    # Pre-allocated work arrays for energy_loop_c (reused per iteration)
    htc_film_all_c = np.zeros((n_ch, n_ax), dtype=np.float64)
    cp_film_all_c = np.zeros((n_ch, n_ax), dtype=np.float64)
    q_net_c = np.zeros((n_ch, n_ax), dtype=np.float64)

    # ---- profile accumulators ----
    pf = {"total": 0.0, "momentum": 0.0, "energy": 0.0, "residual": 0.0}

    if verbose:
        print(f"{'Step':>6} {'Res(P)':>12} {'Res(T)':>12}")
        print("-" * 42)

    t_total = time.perf_counter()
    t_start = t_total

    for outer in range(1, max_outer + 1):
        if wall_max > 0 and (time.perf_counter() - t_start) > wall_max:
            if verbose:
                print(f"\nAborted at iter {outer} (wall clock exceeded {wall_max:.0f}s)")
            break
        p_old = sv.p.copy(); T_old = sv.T.copy(); Tw_old = sv.T_wall.copy()

        # ======== [1] Momentum (C) ========
        tm = time.perf_counter()
        _C_LIB.momentum_loop_c(
            _as_ptr(p_c), _as_ptr(T_c), _as_ptr(rho_c), _as_ptr(h_c),
            _as_ptr(G_c), _as_ptr(Dh_c), ctypes.c_double(dz),
            ctypes.c_int(n_ch), ctypes.c_int(n_ax),
            _as_iptr(ch_tp_arr), _as_ptr(p_out_c),
            _as_iptr(flow_sign_arr),
            ctypes.c_double(urf_p),
            _as_iptr(fluids_arr),
            _as_ptr(mu_tab_f), _as_ptr(k_tab_f),
            _as_ptr(cp_tab_f), _as_ptr(rho_tab_f), _as_ptr(h_tab_f),
            _as_ptr(p_grid_c), ctypes.c_int(n_p),
            _as_ptr(T_grid_c), ctypes.c_int(n_T_w),
            _as_ptr(mu_he_f), _as_ptr(k_he_f),
            _as_ptr(cp_he_f), _as_ptr(rho_he_f), _as_ptr(h_he_f),
            _as_ptr(p_grid_he_c), ctypes.c_int(n_p_he),
            _as_ptr(T_grid_he_c), ctypes.c_int(n_T_he),
            _as_ptr(Tsat_c),
            _as_ptr(hl_c), _as_ptr(hg_c),
            _as_ptr(rho_l_grid), _as_ptr(rho_g_grid),
            _as_ptr(mu_l_grid), _as_ptr(mu_g_grid),
            _as_ptr(sigma_c))
        pf["momentum"] += time.perf_counter() - tm

        # ======== [2b] Energy (C) — also fills htc_bulk via internal Pass 0 ========
        te = time.perf_counter()
        htc_bulk = np.zeros((n_ch, n_ax))
        htc_bulk_c = np.ascontiguousarray(htc_bulk, dtype=np.float64)
        T_film_c = np.ascontiguousarray(sv.T, dtype=np.float64)
        _C_LIB.energy_loop_c(
            _as_ptr(p_c), _as_ptr(T_c), _as_ptr(rho_c), _as_ptr(h_c),
            _as_ptr(G_c), _as_ptr(alpha_c), _as_ptr(T_wall_c),
            _as_ptr(Dh_c), _as_ptr(areas_c), _as_ptr(P_h_c),
            ctypes.c_double(dz),
            ctypes.c_int(n_ch), ctypes.c_int(n_ax),
            _as_iptr(ch_tp_arr), _as_ptr(T_in_c), _as_iptr(flow_sign_arr),
            ctypes.c_double(urf_T),
            _as_iptr(wall_ch_i_a), _as_iptr(wall_ch_j_a), ctypes.c_int(n_wall),
            _as_ptr(wall_tw_a), _as_ptr(wall_kw_a), _as_ptr(wall_Pw_a),
            _as_iptr(fluids_arr),
            _as_ptr(mu_tab_f), _as_ptr(k_tab_f),
            _as_ptr(cp_tab_f), _as_ptr(rho_tab_f), _as_ptr(h_tab_f),
            _as_ptr(p_grid_c), ctypes.c_int(n_p),
            _as_ptr(T_grid_c), ctypes.c_int(n_T_w),
            _as_ptr(mu_he_f), _as_ptr(k_he_f),
            _as_ptr(cp_he_f), _as_ptr(rho_he_f), _as_ptr(h_he_f),
            _as_ptr(p_grid_he_c), ctypes.c_int(n_p_he),
            _as_ptr(T_grid_he_c), ctypes.c_int(n_T_he),
            _as_ptr(Tsat_c), _as_ptr(hl_c), _as_ptr(hg_c),
            _as_ptr(rho_l_grid), _as_ptr(rho_g_grid),
            _as_ptr(mu_l_grid), _as_ptr(mu_g_grid),
            _as_ptr(sigma_c), _as_ptr(htc_bulk_c),
            _as_ptr(htc_film_all_c), _as_ptr(cp_film_all_c), _as_ptr(q_net_c),
            _as_ptr(T_wall_ext_c) if T_wall_ext_c is not None else None,
            _as_ptr(T_film_c))
        if n_wall > 0:
            _C_LIB.update_wall_temps_c(
                _as_ptr(T_wall_c), _as_ptr(T_c), _as_ptr(htc_bulk_c),
                _as_iptr(wall_ch_i_a), _as_iptr(wall_ch_j_a),
                _as_ptr(wall_tw_a), _as_ptr(wall_kw_a), _as_ptr(wall_Pw_a),
                ctypes.c_int(n_wall), ctypes.c_int(n_ax))
        pf["energy"] += time.perf_counter() - te

        # ======== [3] Residuals ========
        tr = time.perf_counter()
        dp_l2 = float(np.sqrt(np.mean((sv.p - p_old) ** 2)))
        dT_l2 = float(np.sqrt(np.mean((sv.T - T_old) ** 2))) if has_ht else 0.0
        dTw_l2 = float(np.sqrt(np.mean((sv.T_wall - Tw_old) ** 2))) if n_wall > 0 else 0.0
        if outer == 1:
            dp_ref = max(dp_l2, 1e-10); dT_ref = max(dT_l2, 1e-10) if has_ht else 1.0
        else:
            dp_ref = max(dp_ref, dp_l2, 1e-10)
            dT_ref = max(dT_ref, dT_l2, 1e-10) if has_ht else 1.0
        r_p = dp_l2 / dp_ref; r_t = dT_l2 / dT_ref if has_ht else r_p
        hist.append(r_p, r_t, r_p)
        if outer > 1:
            if r_p < prev_r_p: urf_p = min(urf_p * 1.08, 0.8)
            else: urf_p = max(urf_p * 0.70, 0.15)
            if r_t < prev_r_t: urf_T = min(urf_T * 1.08, 0.8)
            else: urf_T = max(urf_T * 0.70, 0.15)
        prev_r_p = r_p; prev_r_t = r_t
        pf["residual"] += time.perf_counter() - tr

        if verbose and (outer <= 3 or outer % 20 == 0 or r_p < tol):
            print(f"{outer:>6} {r_p:>12.2e} {r_t:>12.2e}")
        p_ref_abs = float(np.mean(sv.p))
        converged_rel = r_p < tol and r_t < tol and (dTw_l2 < 0.1 or not has_ht)
        # Disable loose absolute criterion; it masks poor energy balance.
        converged_abs = False
        if converged_rel and outer > 2:
            if verbose: print(f"\nConverged in {outer} iterations")
            break

    pf["total"] = time.perf_counter() - t_total

    if profile:
        print("\n===== PROFILE =====")
        for k in ["momentum", "energy", "residual", "total"]:
            if pf[k] > 0:
                print(f"  {k:15s}: {pf[k]*1000:8.2f} ms ({pf[k]/max(pf['total'],1e-30)*100:5.1f}%)")
        print("=" * 24)
        _C_LIB.prof_print_c()

    # ---- outlet boundary condition check ----
    if verbose:
        for c in range(n_ch):
            fs = flow_sign_arr[c]
            dp_flow = sv.p[c, -1] - sv.p[c, 0] if fs == -1 else sv.p[c, 0] - sv.p[c, -1]
            if dp_flow < 0:
                print(f"  WARNING: channel {c} negative flow-direction pressure drop: {dp_flow:.0f} Pa")
            if ch_tp[c]:
                alpha_max = float(np.max(sv.alpha[c]))
                if alpha_max > 0.95:
                    print(f"  WARNING: channel {c} high void fraction: alpha_max={alpha_max:.3f}")

    return sv, hist
