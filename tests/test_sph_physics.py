"""Tests for SPH physics operators."""

from __future__ import annotations

import numpy as np

from sph_solver.config import SimConfig
from sph_solver.core import Particles
from sph_solver.physics import compute_pressure


def test_hydrostatic_column_pressure_linear_with_depth() -> None:
    cfg = SimConfig(rho0=1000.0, c_s=25.0, gamma=7.0, gravity=(0.0, -9.81, 0.0))

    n = 32
    y = np.linspace(0.0, 0.62, n)
    depth = y.max() - y

    hydrostatic_p = cfg.rho0 * abs(cfg.gravity[1]) * depth
    rho = cfg.rho0 * (1.0 + (cfg.gamma * hydrostatic_p) / (cfg.rho0 * cfg.c_s**2)) ** (
        1.0 / cfg.gamma
    )

    pos = np.zeros((n, 3), dtype=np.float64)
    pos[:, 1] = y
    p = Particles(
        pos=pos,
        vel=np.zeros((n, 3), dtype=np.float64),
        acc=np.zeros((n, 3), dtype=np.float64),
        rho=rho,
        pressure=np.zeros(n, dtype=np.float64),
        mass=np.full(n, cfg.rho0 * cfg.particle_volume, dtype=np.float64),
    )

    pressure = compute_pressure(p, cfg)

    rel_err = np.max(np.abs(pressure - hydrostatic_p) / np.maximum(hydrostatic_p, 1e-12))
    assert rel_err < 0.05

    slopes = np.diff(pressure) / np.diff(depth)
    expected_slope = cfg.rho0 * abs(cfg.gravity[1])
    assert np.allclose(slopes, expected_slope, rtol=0.05, atol=1e-9)
