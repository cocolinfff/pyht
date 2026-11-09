import unittest


class TestCleanup(unittest.TestCase):
    def test_reference_closures_removed(self):
        import pyht.closure as c
        self.assertFalse(hasattr(c, "friction_factor"))
        self.assertFalse(hasattr(c, "friedel_two_phi"))
        self.assertFalse(hasattr(c, "nusselt_number"))
        self.assertTrue(hasattr(c, "jk_from_state"))

    def test_dead_fluid_helpers_removed(self):
        from pyht import fluid
        self.assertFalse(hasattr(fluid, "_props"))
        self.assertFalse(hasattr(fluid.PropertyTable, "Tsat"))
        self.assertFalse(hasattr(fluid.CoolPropWrapper, "saturation_T"))
        self.assertFalse(hasattr(fluid.CoolPropWrapper, "d_rho_dp"))
        self.assertFalse(hasattr(fluid.CoolPropWrapper, "d_rho_dT"))

    def test_dead_ice_helpers_removed(self):
        from pyht.solver import ice
        self.assertFalse(hasattr(ice, "_h_direct"))
        self.assertFalse(hasattr(ice, "_quality"))
        self.assertFalse(hasattr(ice._C_LIB, "htc_bulk_c"))


class _Stub:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _stub_state_and_mesh(n_ch=2, n_ax=3, dp_fric=1000.0):
    import numpy as np

    dz, Dh, area, P_heated, mdot = 0.1, 0.01, 1.0e-4, 0.04, 0.1
    G_cell = mdot / area
    z = np.arange(n_ax, dtype=float)
    state = _Stub(
        T=np.full((n_ch, n_ax), 300.0),
        G=np.full((n_ch, n_ax + 1), G_cell),
        rho=np.full((n_ch, n_ax), 1000.0),
        p=np.full((n_ch, n_ax), 1.0e5),
        h=1.0e5 + 1000.0 * z[None, :] * np.ones((n_ch, 1)),
        dp_fric=(None if dp_fric is None
                 else np.full(n_ch, dp_fric)),
    )
    channels = [
        _Stub(fluid="Water", P_heated=P_heated, Dh=Dh, area=area, dz=dz,
              m_dot_in=mdot, flow_sign=1)
        for _ in range(n_ch)
    ]
    return _Stub(channels=channels), state


class TestJkFromStateContract(unittest.TestCase):
    def test_finite_dp_fric_uses_dp_fric(self):
        import numpy as np
        from pyht.closure import jk_from_state

        mesh, state = _stub_state_and_mesh(dp_fric=1000.0)
        out = jk_from_state(mesh, state)
        self.assertEqual(out["f_src"], "dp_fric")
        self.assertTrue(np.isfinite(out["f_w"]))
        self.assertTrue(np.isfinite(out["f_m"]))
        self.assertGreater(out["f_w"], 0.0)
        self.assertGreater(out["f_m"], 0.0)

    def test_missing_dp_fric_gives_nan(self):
        import numpy as np
        from pyht.closure import jk_from_state

        mesh, state = _stub_state_and_mesh(dp_fric=None)
        out = jk_from_state(mesh, state)
        self.assertEqual(out["f_src"], "missing")
        self.assertTrue(np.isnan(out["f_w"]))
        self.assertTrue(np.isnan(out["f_m"]))
