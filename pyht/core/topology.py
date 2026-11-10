import numpy as np
from dataclasses import dataclass, field

@dataclass
class Channel:
    id: int
    area: float
    Dh: float
    P_heated: float
    x: np.ndarray = field(repr=False)
    z: np.ndarray = field(repr=False)
    dz: float = field(repr=False, default=0.0)
    m_dot_in: float = 0.0
    T_in: float = 0.0
    p_out: float = 0.0
    fluid: str = 'Water'
    shape: str = 'circle'
    cut: float = 0.0
    a: float = 0.0
    flow_sign: int = 1
    K_in: float = 0.0
    K_out: float = 0.0

    def num_cells(self) -> int:
        return len(self.z)

@dataclass
class Wall:
    id: int
    ch_i: int
    ch_j: int
    t_w: float
    P_w: float
    k_w: float = 15.0
    one_sided: bool = False
    P_i: float | None = None
    P_j: float | None = None
    P_cond: float | None = None

    def widths(self) -> tuple[float, float, float]:
        pi = self.P_w if self.P_i is None else self.P_i
        pj = self.P_w if self.P_j is None else self.P_j
        pc = self.P_w if self.P_cond is None else self.P_cond
        return (float(pi), float(pj), float(pc))

@dataclass
class SubchannelMesh:
    channels: list[Channel] = field(default_factory=list)
    walls: list[Wall] = field(default_factory=list)
    _ch_map: dict = field(default_factory=dict)

    def add_channel(self, ch: Channel):
        if ch.id in self._ch_map:
            raise ValueError(f"duplicate Channel.id {ch.id}")
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
