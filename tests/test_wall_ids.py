import unittest

import numpy as np

from pyht.core.topology import Channel, SubchannelMesh, Wall
from pyht.solver.ice import steady_ice
from tests._helpers import configure_tables


class TestWallChannelIds(unittest.TestCase):
    def setUp(self):
        configure_tables()

    def _mesh(self):
        n_ax, L = 4, 0.4
        dz = L / n_ax
        z = np.linspace(dz / 2, L - dz / 2, n_ax)
        x = np.linspace(0.0, L, n_ax + 1)
        m = SubchannelMesh()
        for cid, tin in ((2, 360.0), (0, 300.0), (1, 330.0)):
            m.add_channel(Channel(
                id=cid, area=1e-4, Dh=1e-2, P_heated=0.0,
                x=x, z=z, dz=dz, m_dot_in=0.05, T_in=tin,
                p_out=1e5, fluid="Water", flow_sign=1))
        m.add_wall(Wall(0, 2, 1, 1e-3, 1e-2, 15.0))
        return m

    def test_ids_resolve_to_list_positions(self):
        m = self._mesh()
        s, _ = steady_ice(m, max_outer=2, tol=1e-3, verbose=False, urf_T0=1.0)
        pos_id0 = m.ch_idx(0)
        np.testing.assert_allclose(s.T[pos_id0], 300.0, atol=1e-9)
        pos_id2, pos_id1 = m.ch_idx(2), m.ch_idx(1)
        moved = (abs(s.T[pos_id2, -1] - 360.0)
                 + abs(s.T[pos_id1, -1] - 330.0))
        self.assertGreater(moved, 1e-6)
        self.assertLess(float(s.T[pos_id2, -1]), 360.0)
        self.assertGreater(float(s.T[pos_id1, -1]), 330.0)

    def test_unknown_id_raises_valueerror(self):
        m = self._mesh()
        m.walls[0].ch_j = 99
        with self.assertRaises(ValueError):
            steady_ice(m, max_outer=1, tol=1e-3, verbose=False)
