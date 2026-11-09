import unittest

import numpy as np

from pyht.core.topology import Channel, SubchannelMesh, Wall
from pyht.solver.ice import _C_LIB, steady_ice
from pyht.solver import get_chen_nu, set_chen_nu
from tests._helpers import configure_tables


class TestChenNuSwitch(unittest.TestCase):
    def setUp(self):
        configure_tables()
        set_chen_nu(1)

    def tearDown(self):
        set_chen_nu(1)
        configure_tables()

    def test_default_is_gnielinski_and_roundtrips(self):
        self.assertEqual(get_chen_nu(), 1)
        set_chen_nu(0)
        self.assertEqual(get_chen_nu(), 0)
        set_chen_nu(1)
        self.assertEqual(get_chen_nu(), 1)
        with self.assertRaises(ValueError):
            set_chen_nu(2)
        with self.assertRaises(ValueError):
            set_chen_nu(1.5)

    def test_compiled_default_is_gnielinski(self):
        import os
        import subprocess
        import sys
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.check_output(
            [sys.executable, "-c",
             "from pyht.solver import get_chen_nu; print(get_chen_nu())"],
            cwd=root, text=True)
        self.assertEqual(out.strip().splitlines()[-1].strip(), "1")

    def test_macro_switch_matches_formula(self):
        x, G, Dh = 0.2, 500.0, 2.0e-3
        rho_l, rho_g = 799.2, 20.05
        mu_l, mu_g = 1.06e-4, 1.75e-5
        k_l, cp_l, htc_sp = 0.6171, 4868.0, 12345.0
        args = (x, G, Dh, rho_l, rho_g, mu_l, mu_g, k_l, cp_l, htc_sp)

        set_chen_nu(0)
        h_db = _C_LIB.pyht_chen_macro_c(*args)
        set_chen_nu(1)
        h_gn = _C_LIB.pyht_chen_macro_c(*args)

        xs = x
        X_tt = ((1.0 - xs) / xs) ** 0.9 * (rho_g / rho_l) ** 0.5 \
            * (mu_l / mu_g) ** 0.1
        F = min(70.0, 2.35 * (1.0 / X_tt + 0.213) ** 0.736)
        Re_l = G * (1.0 - xs) * Dh / mu_l
        Pr_l = mu_l * cp_l / k_l
        db = 0.023 * Re_l ** 0.8 * Pr_l ** 0.4 * k_l / Dh * F

        self.assertAlmostEqual(h_gn, F * htc_sp, places=6)
        self.assertAlmostEqual(h_db, db, places=6)
        self.assertGreater(h_gn, h_db)

    def _mesh(self):
        p_w = 4.0e6
        import CoolProp.CoolProp as CP
        Ts = float(CP.PropsSI("T", "P", p_w, "Q", 0, "Water"))
        n_ax, L = 20, 1.0
        dz = L / n_ax
        z = np.linspace(dz / 2, L - dz / 2, n_ax)
        x = np.linspace(0.0, L, n_ax + 1)
        area, Dh = 1.0e-4, 1.0e-2
        m = SubchannelMesh()
        m.add_channel(Channel(
            id=0, area=area, Dh=Dh, P_heated=0.0, x=x, z=z, dz=dz,
            m_dot_in=1.0e-2, T_in=973.15, p_out=3.0e6,
            fluid="Helium", flow_sign=-1))
        m.add_channel(Channel(
            id=1, area=area, Dh=Dh, P_heated=0.0, x=x, z=z, dz=dz,
            m_dot_in=3.0e-2, T_in=Ts - 5.0, p_out=p_w,
            fluid="Water", flow_sign=1))
        m.add_wall(Wall(0, 0, 1, 1.0e-3, 2.0e-2, 15.0))
        return m

    def _run(self, mode):
        set_chen_nu(mode)
        m = self._mesh()
        state, hist = steady_ice(m, max_outer=200, tol=1e-3, verbose=False,
                                 two_phase=True, urf_htc=0.35)
        self.assertFalse(hist.diverged)
        self.assertTrue(np.all(np.isfinite(state.T)))
        return state, m

    def test_solver_modes_differ_in_boiling_region(self):
        state_db, m_db = self._run(0)
        state_gn, m_gn = self._run(1)
        w_db, w_gn = m_db.ch_idx(1), m_gn.ch_idx(1)
        h_db = np.asarray(state_db.htc_film[w_db])
        h_gn = np.asarray(state_gn.htc_film[w_gn])
        self.assertGreater(float(np.max(h_db)), 2.0e4)
        self.assertGreater(float(np.max(h_gn)), 2.0e4)
        rel = np.abs(h_gn - h_db) / np.maximum(np.abs(h_db), 1.0)
        self.assertGreater(float(np.max(rel)), 1.0e-3,
                           "Gnielinski and Dittus-Boelter modes produced "
                           "identical htc; the switch is not wired")
