# pyHT — High-Performance Subchannel Thermal-Hydraulic Code

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()

**pyHT** is a high-performance subchannel thermal-hydraulic analysis code for nuclear reactor core simulations. It models steady-state single-phase and two-phase (boiling) flows in parallel channels with wall heat transfer coupling.

The code implements an **Implicit Continuous-fluid Eulerian (ICE)** solution scheme — a semi-implicit pressure-based method — to solve the conservation equations for mass, momentum, and energy. Fluid properties are evaluated using CoolProp (IAPWS-IF97 for water, Helmholtz EOS for helium), with fast property table caching for performance.

---

## Features

- **Working fluids**: Water (IAPWS-IF97) and Helium via CoolProp
- **Single-phase flow**: Churchill friction factor, Gnielinski Nusselt number
- **Two-phase boiling**: Chen nucleate + forced convective boiling correlation
- **Two-phase pressure drop**: Friedel multiplier
- **Void fraction**: Zuber-Findlay drift-flux model
- **Wall heat transfer**: Conjugate wall-to-fluid coupling
- **Solver**: Implicit Continuous-fluid Eulerian (ICE) with adaptive under-relaxation
- **Performance**: C extension (`_solver_c`) for tight-loop linear algebra and residual assembly
- **Fast property tables**: Pre-computed property grids for minimal CoolProp calls at runtime

---

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/placeholder/pyht.git
```

The C extension (`_solver_c`) must be compiled. See *Building from source* below.

### Quick install with pre-built extension

If a pre-built `_solver_c.dll` (Windows) or `_solver_c.so` (Linux) is present in the package directory, `pip install` will pick it up automatically — no compiler required.

---

## Quick Start

```python
import numpy as np
from pyht import core
from pyht.solver import steady_ice

# Define a simple single-channel problem
n_ax = 50
L = 1.0                          # heated length [m]
z = np.linspace(0, L, n_ax)
dz = np.full(n_ax, L / n_ax)

channel = core.Channel(
    area=1e-4,                   # flow area [m^2]
    Dh=0.01,                     # hydraulic diameter [m]
    P_heated=0.0314,             # heated perimeter [m]
    x=np.linspace(0, 1, n_ax + 1),
    z=z,
    dz=dz,
    m_dot_in=0.1,                # inlet mass flow [kg/s]
    T_in=300.0,                  # inlet temperature [K]
    p_out=15.5e6,                # outlet pressure [Pa]
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

# Print results
from pyht.io import print_state
print_state(mesh, state)
```

---

## Module Structure

| Module | Description |
|---|---|
| `pyht.core` | Data structures: `Channel`, `SubchannelMesh`, axial/node geometry |
| `pyht.closure` | Closure models: friction (Churchill), heat transfer (Gnielinski, Chen), two-phase multiplier (Friedel), void fraction (Zuber-Findlay) |
| `pyht.solver` | ICE solver: `steady_ice()`, under-relaxation, outer iteration control |
| `pyht.fluid` | Fluid property wrappers around CoolProp with fast table caching |
| `pyht.io` | Output utilities: `print_state()`, convergence history formatting |

The C extension lives at `pyht/_solver_c.c` / `_solver_c.dll` and handles the performance-critical loops (tridiagonal solves, residual assembly).

---

## Building from Source

The C extension **must be compiled** for the solver to work. A C99 compiler (gcc / MinGW) is required.

### Windows (MinGW-w64)

```batch
gcc -O3 -march=native -shared -o pyht\_solver_c.dll pyht\_solver_c.c
```

Place the resulting `_solver_c.dll` in the `pyht/` package directory.

### Linux

```bash
gcc -O3 -march=native -shared -fPIC -o pyht/_solver_c.so pyht/_solver_c.c
```

### Via setuptools (editable install)

```bash
pip install -e .
```

This compiles the extension automatically if a `setup.py` or extension configuration is provided.

---

## Dependencies

- Python >= 3.9
- [NumPy](https://numpy.org/) >= 1.21.0
- [SciPy](https://scipy.org/) >= 1.7.0
- [CoolProp](http://www.coolprop.org/) >= 6.4.1
- A C99 compiler (if building from source)

---

## License

**pyHT** is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but **without any warranty**; without even the implied warranty of merchantability or fitness for a particular purpose. See the [GNU General Public License v3](LICENSE) for details.

---

## References

- Harlow, F. H., & Amsden, A. A. (1971). *A numerical fluid dynamics calculation method for all flow speeds*. Journal of Computational Physics, 8(2), 197–213.
- Chen, J. C. (1966). *Correlation for boiling heat transfer to saturated fluids in convective flow*. Industrial & Engineering Chemistry Process Design and Development, 5(3), 322–329.
- Friedel, L. (1979). *Improved friction pressure drop correlations for horizontal and vertical two-phase pipe flow*. European Two-Phase Flow Group Meeting, Paper E2.
- Churchill, S. W. (1977). *Friction-factor equation spans all fluid-flow regimes*. Chemical Engineering, 84(24), 91–92.
- Gnielinski, V. (1976). *New equations for heat and mass transfer in turbulent pipe and channel flow*. International Chemical Engineering, 16(2), 359–368.
- Zuber, N., & Findlay, J. A. (1965). *Average volumetric concentration in two-phase flow systems*. Journal of Heat Transfer, 87(4), 453–468.
