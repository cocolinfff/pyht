"""
Console output utilities for solver results and convergence history.
"""


def print_state(mesh, state, label: str = "State"):
    """Print key state variables per channel."""
    n_ch = mesh.n_ch()
    n_ax = mesh.n_ax()
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"{'ch':>3} {'T_in(C)':>10} {'T_out(C)':>10} {'p_out(bar)':>10} {'m_dot(g/s)':>11}")
    print("-" * 60)
    for c in range(n_ch):
        T_out = state.T[c, -1] - 273.15
        T_in = state.T[c, 0] - 273.15
        p_out = state.p[c, -1] / 1e5
        m = state.G[c, 0] * mesh.channels[c].area * 1000
        print(f"{c:>3} {T_in:>10.2f} {T_out:>10.2f} {p_out:>10.5f} {m:>11.3f}")


def print_residuals(history):
    """Print convergence history summary."""
    print(f"\n{'=' * 60}")
    print(f"  Convergence History (last 5 steps)")
    print(f"{'=' * 60}")
    print(f"{'Step':>6} {'Mass Res':>12} {'Mom Res':>12} {'En Res':>12}")
    print("-" * 42)
    n = len(history.mass)
    start = max(0, n - 5)
    for i in range(start, n):
        print(f"{i:>6} {history.mass[i]:>12.2e} {history.moment[i]:>12.2e} {history.energy[i]:>12.2e}")
