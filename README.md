# pyHT — A Parallel-Channel Thermal-Hydraulic Solver

pyHT is a 1D steady-state CFD solver for parallel flow channels coupled by a thermal resistance network through solid walls. It supports single-phase and two-phase (boiling) flow of water and helium.

The code employs the Implicit Continuous-fluid Eulerian (ICE) scheme, uses CoolProp for fluid properties (IAPWS-IF97 water, Helmholtz EOS helium), and implements a C extension for performance-critical loops.

Each channel is solved independently with a fixed mass flow rate — there is no cross-flow or lateral mass exchange between channels. Channels are thermally coupled through solid walls modeled as a 1D series resistance network (convection on both sides + conduction through the wall, i.e. `1/h_i + t_w/k_w + 1/h_j`).

## Quick Start

```python
import numpy as np
from pyht import core
from pyht.solver import steady_ice

n_ax = 40
L = 1.0                           # heated length
z = np.linspace(0, L, n_ax)
dz = np.full(n_ax, L / n_ax)

channel = core.Channel(
    area=1e-4,                     # flow area [m^2]
    Dh=0.01,                       # hydraulic diameter
    P_heated=0.0314,               # heated perimeter
    x=np.linspace(0, 1, n_ax + 1),
    z=z,
    dz=dz,
    m_dot_in=0.1,                  # inlet mass flow
    T_in=300.0,                    # inlet temperature [K]
    p_out=15.5e6,                  # outlet pressure [Pa]
    fluid="Water",
)

mesh = core.SubchannelMesh(
    channels=[channel],
    x=[np.linspace(0, 1, n_ax + 1)],
    walls=[],
)

T_wall = np.full((0, 2, n_ax), np.nan)   # no wall regions

state, residuals = steady_ice(
    mesh, T_wall, max_outer=500, tol=1e-6, verbose=True
)

from pyht.io import print_state
print_state(mesh, state)
```

## Module Structure

pyht is organized into subpackages: `core` (data structures), `closure` (friction, heat transfer, two-phase models), `solver` (ICE solver), `fluid` (CoolProp wrappers with table caching), `io` (output utilities), and the C extension `_solver_c` for tridiagonal solves and residual assembly.

## Building from Source

- **Windows (MinGW-w64):** `gcc -O3 -march=native -shared -o pyht\_solver_c.dll pyht\_solver_c.c`
- **Linux:** `gcc -O3 -march=native -shared -fPIC -o pyht/_solver_c.so pyht/_solver_c.c`

Place the compiled extension in the `pyht/` package directory.

## Dependencies

- Python >= 3.9
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- CoolProp >= 6.4.1
- A C99 compiler (if building from source)

## License

Licensed under GPL-3.0. See [LICENSE](LICENSE) for details.

---

## References

- Harlow, F. H., & Amsden, A. A. (1971). *A numerical fluid dynamics calculation method for all flow speeds*. Journal of Computational Physics, 8(2), 197–213.
- Chen, J. C. (1966). *Correlation for boiling heat transfer to saturated fluids in convective flow*. Industrial & Engineering Chemistry Process Design and Development, 5(3), 322–329.
- Friedel, L. (1979). *Improved friction pressure drop correlations for horizontal and vertical two-phase pipe flow*. European Two-Phase Flow Group Meeting, Paper E2.
- Churchill, S. W. (1977). *Friction-factor equation spans all fluid-flow regimes*. Chemical Engineering, 84(24), 91–92.
- Gnielinski, V. (1976). *New equations for heat and mass transfer in turbulent pipe and channel flow*. International Chemical Engineering, 16(2), 359–368.
- Zuber, N., & Findlay, J. A. (1965). *Average volumetric concentration in two-phase flow systems*. Journal of Heat Transfer, 87(4), 453–468.
