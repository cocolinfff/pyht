import numpy as np
from .j4_octagon import RE_TURB, RMAX, c4_helium, c4_hem, c4_water
C4_LAMINAR = 0.005848
C4_RE = np.array([4000.0, 6000.0, 10000.0])
C4_VAL = np.array([0.005624, 0.005836, 0.006088])

def c4_of(Re):
    Re = np.asarray(Re, dtype=float)
    out = np.full(Re.shape, C4_LAMINAR)
    hi = Re >= C4_RE[0]
    out[hi] = np.interp(Re[hi], C4_RE, C4_VAL)
    return out

def _r_of(ch):
    if getattr(ch, 'shape', '') != 'octagon':
        return 0.0
    try:
        a = float(getattr(ch, 'a', 0.0) or 0.0)
        cut = float(getattr(ch, 'cut', 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if a <= 0.0 or cut <= 0.0:
        return 0.0
    return min(cut / a, RMAX)

def _churchill(Re, eps, Dh):
    Re = np.maximum(Re, 1e-12)
    A = (-2.457 * np.log((7.0 / Re) ** 0.9 + 0.27 * eps / Dh)) ** 16
    B = (37530.0 / Re) ** 16
    return 8.0 * ((8.0 / Re) ** 12 + 1.0 / (A + B) ** 1.5) ** (1.0 / 12.0)

def jk_from_state(mesh, state, n_h=4, flu='Water', k=4, fluids=None):
    n_ch, n_ax = state.T.shape
    fluid_of = np.array([getattr(ch, 'fluid', 'Water') for ch in mesh.channels])
    if fluids is None:
        fluids = (flu,)
    fluids = tuple(fluids)
    sel_by_fluid = {f: fluid_of == f for f in fluids}
    if not any((m.any() for m in sel_by_fluid.values())):
        raise ValueError('no channels with fluids %r' % (fluids,))
    sel = sel_by_fluid.get('Water')
    if sel is None or not sel.any():
        sel = sel_by_fluid[fluids[0]]
    r_ch = np.array([_r_of(ch) for ch in mesh.channels], dtype=float)
    P = np.array([ch.P_heated for ch in mesh.channels], dtype=float)
    Dh = np.array([ch.Dh for ch in mesh.channels], dtype=float)
    A = np.array([ch.area for ch in mesh.channels], dtype=float)
    dz = float(mesh.channels[0].dz) if n_ax > 1 else 1.0
    Gc = 0.5 * (state.G[:, :-1] + state.G[:, 1:])
    rho = state.rho
    from pyht.fluid import _get_table
    mu = np.full(state.T.shape, np.nan)
    kk = np.full(state.T.shape, np.nan)
    cp = np.full(state.T.shape, np.nan)
    for fname in np.unique(fluid_of):
        m = fluid_of == fname
        try:
            tf = _get_table(str(fname))
        except Exception:
            tf = _get_table(flu)
        mu[m] = tf('mu', state.p[m], state.T[m])
        kk[m] = tf('k', state.p[m], state.T[m])
        cp[m] = tf('cp', state.p[m], state.T[m])
    sel_water = sel_by_fluid.get('Water', np.zeros(n_ch, dtype=bool))
    al = np.clip(np.asarray(getattr(state, 'alpha', np.zeros_like(state.T)), dtype=float), 0.0, 1.0)
    tp_all = sel_water[:, None] & (al > 1e-09)
    if sel_water.any():
        from pyht.solver.ice import _Tsat, _cached_CP
        tab_w = _get_table('Water')
        pg = tab_w.p_grid
        p_safe = np.clip(state.p, pg[0], pg[-1])

        def _sat(key, dT):
            return np.array([_cached_CP(key, float(p_), _Tsat(p_) + dT, 'Water') for p_ in pg])
        i_p = np.clip(np.searchsorted(pg, p_safe), 0, len(pg) - 1)
        k_l = _sat('L', -0.5)[i_p]
        k_g = _sat('L', +0.5)[i_p]
        cp_l = _sat('C', -0.5)[i_p]
        cp_g = _sat('C', +0.5)[i_p]
        mu_l = _sat('V', -0.5)[i_p]
        mu_g = _sat('V', +0.5)[i_p]
        tp = tp_all
        if tp.any():
            mu = np.where(tp, (1.0 - al) * mu_l + al * mu_g, mu)
            kk = np.where(tp, (1.0 - al) * k_l + al * k_g, kk)
            cp = np.where(tp, (1.0 - al) * cp_l + al * cp_g, cp)
    Re = np.abs(Gc) * Dh[:, None] / np.maximum(mu, 1e-12)
    Pr = mu * cp / np.maximum(kk, 1e-12)
    RePr = Re * Pr
    mdot = np.array([ch.m_dot_in for ch in mesh.channels], dtype=float)
    dhdz = np.gradient(state.h, dz, axis=1)
    sign = np.array([getattr(ch, 'flow_sign', 1) for ch in mesh.channels], dtype=float)
    qcell = mdot[:, None] * dhdz * sign[:, None]
    qflux = np.abs(qcell) / np.maximum(P[:, None] * dz, 1e-30)
    w = np.abs(Gc) / np.maximum(rho, 1e-12) * qflux / np.maximum(kk, 1e-12)
    dV = A[:, None] * dz
    W = w * dV
    r_cell = r_ch[:, None]
    ck_w = np.where(tp_all, c4_hem(Re, Pr, r_cell), c4_water(Re, Pr, r_cell))
    ck_he = c4_helium(Re, Pr, r_cell)
    ck_by_fluid = {'Water': ck_w, 'Helium': ck_he}
    num = 0.0
    den = 0.0
    J_by_fluid = {}
    W_by_fluid = {}
    for fname in fluids:
        m = sel_by_fluid[fname][:, None]
        Wm = np.where(m, W, 0.0)
        ck = ck_by_fluid.get(fname, ck_w)
        ff = ck * RePr ** k
        num_f = float((Wm * ff).sum())
        den_f = float(Wm.sum())
        W_by_fluid[fname] = den_f
        J_by_fluid[fname] = num_f / den_f if den_f > 0 else float('nan')
        num += num_f
        den += den_f
    J = float(num / den) if den > 0 else float('nan')
    Wmw = np.where(sel_water[:, None], W, 0.0)
    den_old = float(Wmw.sum())
    J_legacy = float((Wmw * c4_of(Re) * RePr ** k).sum()) / den_old if den_old > 0 else float('nan')
    ck = ck_w if sel_water.any() else ck_he
    Jch = np.full(n_ch, np.nan)
    for c in range(n_ch):
        if fluid_of[c] in fluids and W[c].sum() > 0:
            ck_c = ck_he[c] if fluid_of[c] == 'Helium' else ck_w[c]
            Jch[c] = float((W[c] * ck_c * RePr[c] ** k).sum() / W[c].sum())
    _dpf = getattr(state, 'dp_fric', None)
    f_src = 'dp_fric'
    if _dpf is None or not np.all(np.isfinite(np.asarray(_dpf))):
        f_src = 'closure'
        Re_c = np.abs(Gc) * Dh[:, None] / np.maximum(mu, 1e-12)
        _eps = 1e-05
        f_cell = np.where(Re_c < 2300.0, 56.91 / np.maximum(Re_c, 1e-12), _churchill(Re_c, _eps, Dh[:, None]))
        f_cell = np.where(np.isfinite(f_cell) & (f_cell > 0), f_cell, np.nan)
        f_ch = np.full(n_ch, np.nan)
        Wc = np.where(sel[:, None], W, 0.0)
        ok = np.isfinite(f_cell)
        num = np.where(ok, Wc * np.nan_to_num(f_cell), 0.0).sum(axis=1)
        den = np.where(ok, Wc, 0.0).sum(axis=1)
        f_ch = np.where(den > 0, num / np.maximum(den, 1e-300), np.nan)
    else:
        _dpf = np.asarray(_dpf, dtype=float)
        rho_bar = np.mean(rho, axis=1)
        Lch = dz * state.p.shape[1]
        G_ch = mdot / np.maximum(A, 1e-30)
        with np.errstate(divide='ignore', invalid='ignore'):
            f_ch = 2.0 * rho_bar * _dpf * Dh / (Lch * np.maximum(G_ch ** 2, 1e-30))
        f_ch = np.where(np.isfinite(f_ch) & (f_ch > 0), f_ch, np.nan)
    Wc = np.where(sel[:, None], W, 0.0)
    Wsum = Wc.sum(axis=1)
    good = sel & np.isfinite(f_ch) & (Wsum > 0)
    f_w = float((Wsum[good] * f_ch[good]).sum() / Wsum[good].sum()) if good.any() else float('nan')
    mw = np.where(sel & np.isfinite(f_ch), mdot, 0.0)
    f_m = float((mw * np.nan_to_num(f_ch)).sum() / mw.sum()) if mw.sum() > 0 else float('nan')
    f_cell = f_ch[:, None] * np.ones((1, state.T.shape[1]))
    sel_he = sel_by_fluid.get('Helium', np.zeros(n_ch, dtype=bool))
    W_w = np.where(sel_water[:, None], W, 0.0)
    W_w_sum = float(W_w.sum())
    tp_frac = float(W_w[tp_all].sum()) / W_w_sum if W_w_sum > 0 else 0.0
    ff_w = ck_w * RePr ** k
    den_wff = float((W_w * ff_w).sum())
    low_mask = sel_water[:, None] & (Re < RE_TURB)
    low_frac = float((W_w * ff_w)[low_mask].sum()) / den_wff if den_wff > 0 else 0.0
    tp_j_frac = float((W_w * ff_w)[tp_all].sum()) / den_wff if den_wff > 0 else 0.0
    return {'J%d' % k: J, 'J%d_water' % k: J_by_fluid.get('Water', float('nan')), 'J%d_He' % k: J_by_fluid.get('Helium', float('nan')), 'J%d_legacy' % k: J_legacy, 'J%d_ch' % k: Jch, 'W_water': W_by_fluid.get('Water', 0.0), 'W_He': W_by_fluid.get('Helium', 0.0), 'Re_med': float(np.median(Re[sel])), 'Re_max': float(Re[sel].max()), 'RePr_med': float(np.median(RePr[sel])), 'Re_med_He': float(np.median(Re[sel_he])) if sel_he.any() else float('nan'), 'c4_med': float(np.median(ck[sel])), 'c4_med_He': float(np.median(ck_he[sel_he])) if sel_he.any() else float('nan'), 'alpha_max': float(al[sel_water].max()) if sel_water.any() else 0.0, 'W_2ph_frac': tp_frac, 'J4_lowRe_frac': low_frac, 'J4_2ph_frac': tp_j_frac, 'w_med': float(np.median(w[sel])), 'f_w': f_w, 'f_m': f_m, 'f_src': f_src, 'f_med': float(np.nanmedian(f_ch[sel])) if np.isfinite(f_ch[sel]).any() else float('nan')}
