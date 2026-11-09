import dataclasses
import unittest

from pyht.core.topology import Channel


class TestChannelFields(unittest.TestCase):
    def test_dead_fields_removed(self):
        names = {f.name for f in dataclasses.fields(Channel)}
        self.assertNotIn("q_wall", names)
        self.assertNotIn("heated_sign", names)
