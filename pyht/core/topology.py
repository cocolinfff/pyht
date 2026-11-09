"""
Subchannel topology: channel geometry and wall adjacency graph.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class Channel:
    id: int
    area: float           # flow area [m²]
    Dh: float             # hydraulic diameter [m]
    P_heated: float       # heated perimeter (external heating) [m]

    # axial mesh
    x: np.ndarray = field(repr=False)        # node positions [m]
    z: np.ndarray = field(repr=False)        # cell centers [m]
    dz: float = field(repr=False, default=0.0)

    # boundary conditions
    m_dot_in: float = 0.0
    T_in: float = 0.0
    p_out: float = 0.0

    q_wall: np.ndarray = field(repr=False, default=None)
    fluid: str = "Water"  # CoolProp fluid (Water, Helium, ...)

    # extended properties for momentum/energy
    flow_sign: int = 1      # flow direction: +1 upward, -1 downward
    K_in: float = 0.5       # inlet form loss coefficient
    K_out: float = 1.0      # outlet form loss coefficient
    heated_sign: int = 1    # heating direction: +1 wall→fluid

    def num_cells(self) -> int:
        return len(self.z)


@dataclass
class Wall:
    """Solid wall connecting two channels (thermal coupling, no mass exchange)."""
    id: int
    ch_i: int             # first channel index
    ch_j: int             # second channel index
    t_w: float            # wall thickness [m]
    P_w: float            # wall heat transfer width [m] (contact width)
    k_w: float = 15.0     # wall thermal conductivity [W/m·K] (SS304 default)


@dataclass
class SubchannelMesh:
    channels: list[Channel] = field(default_factory=list)
    walls: list[Wall] = field(default_factory=list)
    _ch_map: dict = field(default_factory=dict)

    def add_channel(self, ch: Channel):
        self._ch_map[ch.id] = len(self.channels)
        self.channels.append(ch)

    def add_wall(self, w: Wall):
        self.walls.append(w)

    def n_ch(self) -> int:
        return len(self.channels)

    def n_ax(self) -> int:
        return self.channels[0].num_cells() if self.channels else 0

    def ch_idx(self, ch_id: int) -> int:
        return self._ch_map[ch_id]



