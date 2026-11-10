import unittest

import CoolProp.CoolProp as CP
import numpy as np

from pyht.core.topology import Channel, SubchannelMesh, Wall
from pyht.fluid import CoolPropWrapper
from pyht.solver.ice import steady_ice


class TestTwoPhaseHtc(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        CoolPropWrapper.set_table_range((5e4, 3e7), (300.0, 1073.15),
                                        50, 150, "Water")
        CoolPropWrapper.set_table_range((5e5, 1e7), (300.0, 1173.15),
                                        40, 120, "Helium")

    def test_chen_nucleate_term_is_not_blown_up(self):
        p_w = 4.0e6
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

        state, hist = steady_ice(m, max_outer=200, tol=1e-3, verbose=False,
                                 two_phase=True, urf_htc=0.35)
        self.assertFalse(hist.diverged)
        self.assertTrue(np.all(np.isfinite(state.T)))

        w = m.ch_idx(1)
        self.assertGreater(float(np.max(state.alpha[w])), 0.1)
        self.assertTrue(np.isfinite(state.dp_fric[w]))
        self.assertGreater(float(np.max(state.dp_fric)), 0.0)
        self.assertGreater(float(state.dp_fric[w]), 0.0)
        hmax = float(np.max(state.htc_film[w]))
        self.assertGreater(hmax, 2.0e4)
        self.assertLess(
            hmax, 1.0e5,
            "htc too large: check fz_liq exponents and q_flux units")
