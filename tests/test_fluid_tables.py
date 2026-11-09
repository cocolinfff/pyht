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


import numpy as np

from pyht.fluid import _get_table


class TestPythonLookupDelegation(unittest.TestCase):
    def test_out_of_range_clamps_like_c_and_warns_once(self):
        import warnings

        tab = PropertyTable("Helium", p_lim=(5e5, 1e7),
                            T_lim=(300.0, 600.0), n_p=20, n_T=40)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            v900 = float(tab("rho", 3.0e6, 900.0))
            v901 = float(tab("rho", 3.0e6, 901.0))
        self.assertEqual(len(caught), 1,
                         "second out-of-range query must not warn again")
        self.assertTrue(issubclass(caught[0].category, RuntimeWarning))
        v600 = float(tab("rho", 3.0e6, 600.0))
        self.assertAlmostEqual(v900, v600, places=12)
        self.assertAlmostEqual(v901, v600, places=12)

    def test_array_shape_is_preserved(self):
        tab = PropertyTable("Helium", n_p=30, n_T=80)
        p = np.full((2, 3), 3.0e6)
        T = np.full((2, 3), 973.15)
        out = tab("rho", p, T)
        self.assertEqual(out.shape, (2, 3))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_mixed_scalar_and_array_inputs_broadcast(self):
        tab = PropertyTable("Helium", n_p=30, n_T=80)
        p = 3.0e6
        T = np.array([300.0, 500.0, 900.0])
        out = tab("rho", p, T)
        self.assertEqual(out.shape, (3,))
        want = np.array([float(tab("rho", p, float(t))) for t in T])
        np.testing.assert_allclose(out, want, rtol=0.0, atol=1e-9)

        p2 = np.array([[3.0e6], [5.0e6]])
        T2 = np.array([[300.0, 900.0]])
        out2 = tab("rho", p2, T2)
        self.assertEqual(out2.shape, (2, 2))
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(
                    float(out2[i, j]),
                    float(tab("rho", float(p2[i, 0]), float(T2[0, j]))),
                    places=9)

    def test_same_size_different_shape_broadcast_pairing(self):
        tab = PropertyTable("Helium", n_p=30, n_T=80)
        p = np.array([[1.0e6, 2.0e6, 3.0e6, 4.0e6, 5.0e6, 6.0e6]])
        T = np.array([[300.0], [500.0], [700.0], [900.0], [1100.0],
                      [1300.0]])
        out = tab("rho", p, T)
        self.assertEqual(out.shape, (6, 6))
        for i in range(6):
            for j in range(6):
                self.assertAlmostEqual(
                    float(out[i, j]),
                    float(tab("rho", float(p[0, j]), float(T[i, 0]))),
                    places=9)
        with self.assertRaises(ValueError):
            tab("rho", np.zeros((2, 3)) + 3.0e6,
                np.zeros((3, 2)) + 500.0)


class TestSaturationPhase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_tables()
        cls.tab = _get_table("Water")
        cls.p = 4.0e6
        cls.Ts = float(CP.PropsSI("T", "P", cls.p, "Q", 0, "Water"))

    def test_exact_saturation_is_saturated_liquid(self):
        for name, key, rtol in (("mu", "V", 1e-2), ("k", "L", 1e-2),
                                ("cp", "C", 1e-2), ("rho", "D", 1e-2),
                                ("h", "H", 1e-2)):
            got = float(self.tab(name, self.p, self.Ts))
            liq = float(CP.PropsSI(key, "P", self.p, "T", self.Ts - 0.5, "Water"))
            gas = float(CP.PropsSI(key, "P", self.p, "T", self.Ts + 0.5, "Water"))
            self.assertLess(
                abs(got / liq - 1.0), rtol,
                "%s got=%g liquid=%g gas=%g" % (name, got, liq, gas))

    def test_just_below_saturation_stays_liquid(self):
        got = float(self.tab("k", self.p, self.Ts - 0.05))
        liq = float(CP.PropsSI("L", "P", self.p, "T", self.Ts - 0.5, "Water"))
        self.assertLess(abs(got / liq - 1.0), 2e-2)

    def test_clearly_superheated_is_vapour(self):
        got = float(self.tab("k", self.p, self.Ts + 2.0))
        liq = float(CP.PropsSI("L", "P", self.p, "T", self.Ts - 0.5, "Water"))
        gas = float(CP.PropsSI("L", "P", self.p, "T", self.Ts + 2.0, "Water"))
        self.assertLess(abs(got - gas), abs(got - liq))
