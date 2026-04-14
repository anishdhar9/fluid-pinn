"""SPH time integration routines."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from sph_solver.config import SimConfig
from sph_solver.core import Particles, wendland_c2
from sph_solver.physics import compute_density, compute_forces, compute_pressure


def _neighbor_pairs(neighbors: list[np.ndarray], n_particles: int) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.array([ids.size for ids in neighbors], dtype=np.int64)
    if lengths.shape[0] != n_particles:
        raise ValueError("neighbors length must match number of particles")
    if lengths.sum() == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    i_idx = np.repeat(np.arange(n_particles, dtype=np.int64), lengths)
    j_idx = np.concatenate(neighbors).astype(np.int64, copy=False)
    return i_idx, j_idx


def leapfrog_step(
    p: Particles,
    cfg: SimConfig,
    neighbors_fn: Callable[[np.ndarray, float], list[np.ndarray]],
) -> None:
    """Advance one SPH step with kick-drift-kick leapfrog + XSPH correction."""

    dt = cfg.dt
    h = cfg.smoothing_length
    eps = 0.5

    neighbors = neighbors_fn(p.pos, h)
    compute_density(p, neighbors, cfg)
    compute_pressure(p, cfg)
    compute_forces(p, neighbors, cfg)

    p.vel += 0.5 * dt * p.acc
    p.pos += dt * p.vel

    neighbors_new = neighbors_fn(p.pos, h)
    compute_density(p, neighbors_new, cfg)
    compute_pressure(p, cfg)
    compute_forces(p, neighbors_new, cfg)

    p.vel += 0.5 * dt * p.acc

    n = p.pos.shape[0]
    i_idx, j_idx = _neighbor_pairs(neighbors_new, n)
    if i_idx.size == 0:
        return

    r_ij = p.pos[i_idx] - p.pos[j_idx]
    r = np.linalg.norm(r_ij, axis=1)
    w_ij = wendland_c2(r, h)
    rho_bar = 0.5 * (p.rho[i_idx] + p.rho[j_idx])

    v_ij = p.vel[j_idx] - p.vel[i_idx]
    coeff = eps * p.mass[j_idx] * w_ij / np.maximum(rho_bar, 1e-12)

    xsph = np.zeros_like(p.vel)
    np.add.at(xsph, i_idx, coeff[:, None] * v_ij)
    p.vel += xsph
