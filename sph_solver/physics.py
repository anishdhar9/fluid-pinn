"""SPH physics operators (density, EOS pressure, and accelerations)."""

from __future__ import annotations

import numpy as np

from sph_solver.config import SimConfig
from sph_solver.core import Particles, grad_wendland_c2, wendland_c2


def _neighbor_pairs(neighbors: list[np.ndarray], n_particles: int) -> tuple[np.ndarray, np.ndarray]:
    """Flatten neighbor list-of-arrays into vectorized (i, j) interaction pairs."""

    lengths = np.fromiter((len(ids) for ids in neighbors), dtype=np.int64, count=n_particles)
    total = int(lengths.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    i_idx = np.repeat(np.arange(n_particles, dtype=np.int64), lengths)
    j_idx = np.concatenate(neighbors).astype(np.int64, copy=False)
    return i_idx, j_idx


def compute_density(p: Particles, neighbors: list[np.ndarray], cfg: SimConfig) -> np.ndarray:
    """Compute SPH density via summation ρᵢ = Σⱼ mⱼ W(|rᵢ-rⱼ|, h)."""

    h = cfg.smoothing_length
    n = p.pos.shape[0]
    i_idx, j_idx = _neighbor_pairs(neighbors, n)
    if i_idx.size == 0:
        p.rho = np.zeros(n, dtype=np.float64)
        return p.rho

    r_ij = p.pos[i_idx] - p.pos[j_idx]
    r = np.linalg.norm(r_ij, axis=1)
    w = wendland_c2(r, h)
    contrib = p.mass[j_idx] * w
    rho = np.bincount(i_idx, weights=contrib, minlength=n).astype(np.float64)
    p.rho = rho
    return rho


def compute_pressure(p: Particles, cfg: SimConfig) -> np.ndarray:
    """Compute pressure from density using the Tait equation of state."""

    b = cfg.rho0 * cfg.c_s**2 / cfg.gamma
    pressure = b * ((p.rho / cfg.rho0) ** cfg.gamma - 1.0)
    p.pressure = np.maximum(pressure, 0.0)
    return p.pressure


def compute_forces(p: Particles, neighbors: list[np.ndarray], cfg: SimConfig) -> np.ndarray:
    """Compute accelerations from pressure, viscosity, and gravity."""

    h = cfg.smoothing_length
    n = p.pos.shape[0]
    gravity = np.asarray(cfg.gravity, dtype=np.float64)

    i_idx, j_idx = _neighbor_pairs(neighbors, n)
    if i_idx.size == 0:
        p.acc = np.broadcast_to(gravity, (n, 3)).copy()
        return p.acc

    r_ij = p.pos[i_idx] - p.pos[j_idx]
    v_ij = p.vel[i_idx] - p.vel[j_idx]
    grad_w = grad_wendland_c2(r_ij, h)

    rho_i = p.rho[i_idx]
    rho_j = p.rho[j_idx]

    pressure_pair = -p.mass[j_idx] * (
        p.pressure[i_idx] / (rho_i**2) + p.pressure[j_idx] / (rho_j**2)
    )
    pressure_terms = pressure_pair[:, None] * grad_w

    r2 = np.sum(r_ij * r_ij, axis=1)
    vr = np.sum(v_ij * r_ij, axis=1)
    visc_pair = (
        p.mass[j_idx]
        * cfg.mu
        * (4.0 * vr)
        / (rho_i * rho_j * (r2 + 0.01 * h * h))
    )
    visc_terms = visc_pair[:, None] * grad_w

    pressure_acc = np.column_stack(
        [
            np.bincount(i_idx, weights=pressure_terms[:, d], minlength=n)
            for d in range(3)
        ]
    )
    visc_acc = np.column_stack(
        [np.bincount(i_idx, weights=visc_terms[:, d], minlength=n) for d in range(3)]
    )

    p.acc = pressure_acc + visc_acc + gravity
    return p.acc
