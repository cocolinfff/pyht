import numpy as np

class StateVector:

    def __init__(self, n_ch: int, n_ax: int, n_walls: int=0):
        self.p = np.zeros((n_ch, n_ax))
        self.T = np.zeros((n_ch, n_ax))
        self.rho = np.zeros((n_ch, n_ax))
        self.h = np.zeros((n_ch, n_ax))
        self.alpha = np.zeros((n_ch, n_ax))
        self.G = np.zeros((n_ch, n_ax + 1))
        self.T_wall = np.zeros((n_walls, 2, n_ax))
        self.dp_fric = np.zeros(n_ch)
        self.htc_film = np.zeros((n_ch, n_ax))

    def u(self, ch: int) -> np.ndarray:
        return self.G[ch] / np.maximum(self._rho_face(ch), 1e-10)

    def _rho_face(self, ch: int) -> np.ndarray:
        rho = self.rho[ch]
        G = self.G[ch]
        rho_f = np.zeros(len(G))
        for j in range(len(G)):
            if j == 0:
                rho_f[j] = rho[0]
            elif j == len(rho):
                rho_f[j] = rho[-1]
            else:
                rho_f[j] = rho[j - 1] if G[j] >= 0 else rho[j]
        return rho_f

class ResidualHistory:

    def __init__(self):
        self.mass = []
        self.moment = []
        self.energy = []
        self.stalled = False
        self.diverged = False
        self.exit_reason = ''
        self.dt_max_final = float('nan')
        self.dt_p99_final = float('nan')
        self.dtw_l2_final = float('nan')
        self.rh_final = float('nan')
        self.dh_p99_final = float('nan')

    def append(self, r_mass: float, r_mom: float, r_ener: float):
        self.mass.append(r_mass)
        self.moment.append(r_mom)
        self.energy.append(r_ener)

    def converged(self, tol: float=1e-06, window: int=5) -> bool:
        if len(self.mass) < window + 1:
            return False
        return all((max(r[-window:]) < tol for r in [self.mass, self.moment, self.energy]))
