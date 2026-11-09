# pyHT

pyHT is a one-dimensional steady-state subchannel thermal-hydraulic solver for
parallel channels coupled through solid walls. It supports single-phase and
two-phase water/helium flow, uses CoolProp properties, and provides a C
extension for the momentum and energy loops plus a generic NSGA-II/III
optimization interface.

## Layout

| Path | Contents |
| --- | --- |
| `pyht/core/` | channel, wall and subchannel topology |
| `pyht/solver/` | steady-state driver and C kernel |
| `pyht/closure/` | friction, heat-transfer and J4 field-synergy closures |
| `pyht/fluid.py` | CoolProp wrapper and property tables |
| `pyht/optim/` | case-injection interface and NSGA-II/III runner |

## Build

Requirements: Python >= 3.10, numpy, scipy and CoolProp. `pymoo` and
`matplotlib` are optional extras.

```bash
pip install -e ".[opt,plot]"
python build_c_ext.py
```

`build_c_ext.py` compiles `pyht/solver/_solver_c.c` into `_solver_c.dll` with
GCC/MinGW. The C extension is required at runtime.

## Usage

```python
import numpy as np
from pyht.core.topology import Channel, SubchannelMesh
from pyht.solver.ice import steady_ice

L, n_ax = 1.0, 20
mesh = SubchannelMesh()
mesh.add_channel(Channel(
    0, 1e-4, 0.01, 0.04,
    np.linspace(0.0, L, n_ax + 1),
    np.linspace(L / n_ax / 2, L - L / n_ax / 2, n_ax),
    L / n_ax, 0.1, 300.0, 15.5e6, np.zeros(n_ax),
    fluid="Water", flow_sign=1,
))
state, history = steady_ice(mesh, max_outer=200, tol=1e-6)
```

`state` holds the converged temperature, enthalpy, pressure, density and mass
flux arrays; `history` holds the residual and convergence trace.
