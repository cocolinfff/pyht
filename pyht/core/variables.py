"""
State vectors for staggered-grid variables.
"""

import numpy as np


class StateVector:
    """Stores all solution variables for the subchannel domain."""

    def __init__(self, n_ch: int, n_ax: int, n_walls: int = 0):
        # Cell-centered (n_ch x n_ax)
        self.p   = np.zeros((n_ch, n_ax))   # pressure [Pa]
        self.T   = np.zeros((n_ch, n_ax))   # temperature [K]
        self.rho = np.zeros((n_ch, n_ax))   # density [kg/m³]
        self.h   = np.zeros((n_ch, n_ax))   # specific enthalpy [J/kg]
        self.alpha = np.zeros((n_ch, n_ax)) # void fraction (0 for SP)

        # Face-centered (n_ch x n_ax+1)
        self.G = np.zeros((n_ch, n_ax + 1)) # mass flux rho*u [kg/m²s]

        # Wall surface temperatures (n_walls x 2 x n_ax)
        # [wi, 0, j] = T_wall on ch_i side, [wi, 1, j] = T_wall on ch_j side
        self.T_wall = np.zeros((n_walls, 2, n_ax))

    def u(self, ch: int) -> np.ndarray:
        """Velocity [m/s] at cell faces"""
        return self.G[ch] / np.maximum(self._rho_face(ch), 1e-10)

    def _rho_face(self, ch: int) -> np.ndarray:
        """Density at cell faces by donor-cell upwinding"""
        rho = self.rho[ch]
        G   = self.G[ch]
        rho_f = np.zeros(len(G))
        for j in range(len(G)):
            if j == 0:
                rho_f[j] = rho[0]
            elif j == len(rho):
                rho_f[j] = rho[-1]
            else:
                rho_f[j] = rho[j-1] if G[j] >= 0 else rho[j]
        return rho_f


class ResidualHistory:
    """Records convergence history."""

    def __init__(self):
        self.mass   = []
        self.moment = []
        self.energy = []

    def append(self, r_mass: float, r_mom: float, r_ener: float):
        self.mass.append(r_mass)
        self.moment.append(r_mom)
        self.energy.append(r_ener)

    def converged(self, tol: float = 1e-6, window: int = 5) -> bool:
        if len(self.mass) < window + 1:
            return False
        return all(max(r[-window:]) < tol for r in [self.mass, self.moment, self.energy])
