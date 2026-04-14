"""Tests for SPH integrator and boundary handling."""

from __future__ import annotations

import numpy as np

from sph_solver.boundary import enforce_box
from sph_solver.config import SimConfig
from sph_solver.core import Particles
from sph_solver.integrator import leapfrog_step


def test_gravity_only_particles_stay_in_box_and_energy_bounded() -> None:
    cfg = SimConfig(
        box_size=(1.0, 1.0, 1.0),
        particle_spacing=0.05,
        h=0.05,
        dt=0.01,
        gravity=(0.0, -9.81, 0.0),
    )

    n = 100
    rng = np.random.default_rng(7)
    pos = rng.uniform(0.1, 0.9, size=(n, 3)).astype(np.float64)
    p = Particles(
        pos=pos,
        vel=np.zeros((n, 3), dtype=np.float64),
        acc=np.zeros((n, 3), dtype=np.float64),
        rho=np.full(n, cfg.rho0, dtype=np.float64),
        pressure=np.zeros(n, dtype=np.float64),
        mass=np.ones(n, dtype=np.float64),
    )

    def neighbors_fn(x: np.ndarray, h: float) -> list[np.ndarray]:
        _ = x, h
        return [np.empty(0, dtype=np.int64) for _ in range(n)]

    kinetic = []
    for _ in range(50):
        leapfrog_step(p, cfg, neighbors_fn)
        enforce_box(p, cfg)
        kinetic.append(float(0.5 * np.sum(p.mass * np.sum(p.vel * p.vel, axis=1))))

    box = np.asarray(cfg.box_size)
    assert np.all(p.pos >= -1e-12)
    assert np.all(p.pos <= box[None, :] + 1e-12)
    assert max(kinetic) < 5_000.0
