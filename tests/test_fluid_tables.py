import unittest

import CoolProp.CoolProp as CP

from pyht.fluid import PropertyTable
from pyht.solver.ice import _axis_of
from tests._helpers import configure_tables


class TestAxisInvalidation(unittest.TestCase):
    def test_rebuild_replaces_cached_axis(self):
        tab = PropertyTable("Helium", p_lim=(5e5, 1e7),
                            T_lim=(300.0, 700.0), n_p=20, n_T=30)
        ax = _axis_of(tab)[0]
        self.assertEqual((ax.n_p, ax.n_T), (20, 30))

        tab.rebuild(p_range=(5e5, 1e7), T_range=(300.0, 1173.15),
                    n_p=25, n_T=80)
        ax2 = _axis_of(tab)[0]
        self.assertEqual((ax2.n_p, ax2.n_T), (25, 80))


class TestHeliumDefaultRange(unittest.TestCase):
    def test_default_covers_973k(self):
        tab = PropertyTable("Helium", n_p=40, n_T=120)
        p, T = 3.0e6, 973.15
        for name, key, rtol in (("rho", "D", 1e-2), ("h", "H", 1e-2),
                                ("cp", "C", 1e-2), ("mu", "V", 1e-2),
                                ("k", "L", 1e-2)):
            got = float(tab(name, p, T))
            want = float(CP.PropsSI(key, "P", p, "T", T, "Helium"))
            self.assertLess(abs(got / want - 1.0), rtol,
                            "%s got=%g want=%g" % (name, got, want))


class TestSetTableRangeDefault(unittest.TestCase):
    def test_helium_none_t_lim_uses_1500k_default(self):
        from pyht.fluid import CoolPropWrapper, _get_table
        try:
            CoolPropWrapper.set_table_range(p_lim=(5e5, 1e7), T_lim=None,
                                            n_p=20, n_T=40, fluid="Helium")
            tab = _get_table("Helium")
            self.assertAlmostEqual(float(tab.T_grid[-1]), 1500.0, places=9)
            self.assertAlmostEqual(float(tab.u_grid[-1]), 1500.0, places=9)
            got = float(tab("rho", 3.0e6, 973.15))
            want = float(CP.PropsSI("D", "P", 3.0e6, "T", 973.15, "Helium"))
            self.assertLess(abs(got / want - 1.0), 1e-2,
                            "rho got=%g want=%g" % (got, want))
        finally:
            configure_tables()
