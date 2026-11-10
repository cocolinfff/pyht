import dataclasses
import unittest

import numpy as np

from pyht.core.topology import Channel, SubchannelMesh


class TestChannelFields(unittest.TestCase):
    def test_dead_fields_removed(self):
        names = {f.name for f in dataclasses.fields(Channel)}
        self.assertNotIn("q_wall", names)
        self.assertNotIn("heated_sign", names)


class TestChannelIds(unittest.TestCase):
    def test_duplicate_id_rejected(self):
        m = SubchannelMesh()
        x = np.linspace(0.0, 1.0, 3)
        z = np.array([0.25, 0.75])
        m.add_channel(Channel(id=7, area=1e-4, Dh=1e-2, P_heated=0.0,
                              x=x, z=z))
        with self.assertRaises(ValueError):
            m.add_channel(Channel(id=7, area=1e-4, Dh=1e-2, P_heated=0.0,
                                  x=x, z=z))
