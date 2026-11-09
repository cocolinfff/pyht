import ctypes
import numpy as np

def solve_mesh(mesh, *, max_outer, tol, two_phase=False, wall_max=0.0, T_wall=None, stall_window=50, urf_htc=0.35, callback=None, duty_tol=0.002, profile=False, dtw_tol=0.1, pt_tol=0.5, urf_p0=0.5, urf_T0=0.5, adapt_urf=True, balance_flow=False, balance_relax=1.0, balance_start=1, warm_start=None, want_f=True):
    from pyht.solver.ice import steady_ice
    return steady_ice(mesh, T_wall=T_wall, max_outer=max_outer, tol=tol, verbose=False, two_phase=two_phase, wall_max=wall_max, stall_window=stall_window, urf_htc=urf_htc, callback=callback, duty_tol=duty_tol, profile=profile, dtw_tol=dtw_tol, pt_tol=pt_tol, urf_p0=urf_p0, urf_T0=urf_T0, adapt_urf=adapt_urf, balance_flow=balance_flow, balance_relax=balance_relax, balance_start=balance_start, warm_start=warm_start, want_f=want_f)

def postprocess(mesh, state):
    from pyht.solver.ice import _C_LIB as _cpost
    n_ch = mesh.n_ch()
    n_ax = state.T.shape[1]
    n_wall = state.T_wall.shape[0]
    fs_arr = np.array([getattr(ch, 'flow_sign', 1) for ch in mesh.channels], dtype=np.int32)
    fl_arr = np.array([1 if ch.fluid == 'Helium' else 0 for ch in mesh.channels], dtype=np.int32)
    md_arr = np.array([ch.m_dot_in for ch in mesh.channels], dtype=np.float64)
    _Qh = ctypes.c_double(0)
    _Qw = ctypes.c_double(0)
    _dph = ctypes.c_double(0)
    _dpw = ctypes.c_double(0)
    _Tha = ctypes.c_double(0)
    _Twa = ctypes.c_double(0)
    _rha = ctypes.c_double(0)
    _rwa = ctypes.c_double(0)
    _Twm = ctypes.c_double(0)
    _cpost.evaluate_post_c(state.p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), state.T.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), state.rho.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), state.h.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), state.T_wall.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n_ch, n_ax, n_wall, fs_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), fl_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), md_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.byref(_Qh), ctypes.byref(_Qw), ctypes.byref(_dph), ctypes.byref(_dpw), ctypes.byref(_Tha), ctypes.byref(_Twa), ctypes.byref(_rha), ctypes.byref(_rwa), ctypes.byref(_Twm))
    Qh, Qw = (_Qh.value, _Qw.value)
    Twa = _Twa.value
    Twm = _Twm.value if n_wall > 0 else Twa
    return {'Qh': Qh, 'Qw': Qw, 'Q': 0.5 * (Qh + Qw), 'dph': _dph.value, 'dpw': _dpw.value, 'Tha': _Tha.value, 'Twa': Twa, 'rha': _rha.value, 'rwa': _rwa.value, 'Twm': Twm}

def state_is_valid(state, mesh):
    if state is None:
        return False
    for name, arr in [('p', state.p), ('T', state.T), ('h', state.h), ('rho', state.rho), ('G', state.G)]:
        if not np.all(np.isfinite(arr)):
            return False
    if state.T_wall.size > 0 and (not np.all(np.isfinite(state.T_wall))):
        return False
    if np.any(state.T < 200) or np.any(state.T > 2000):
        return False
    if np.any(state.p < 1000.0) or np.any(state.p > 500000000.0):
        return False
    return True
