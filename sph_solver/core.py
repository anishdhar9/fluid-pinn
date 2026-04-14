"""SPH core data structures and spatial/kernel helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from sph_solver.config import SimConfig

_WENDLAND_C2_ALPHA_3D = 21.0 / (16.0 * np.pi)


@dataclass(slots=True)
class Particles:
    """Particle state arrays for SPH simulation."""

    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray
    rho: np.ndarray
    pressure: np.ndarray
    mass: np.ndarray

    def __post_init__(self) -> None:
        self.pos = np.asarray(self.pos, dtype=np.float64)
        self.vel = np.asarray(self.vel, dtype=np.float64)
        self.acc = np.asarray(self.acc, dtype=np.float64)
        self.rho = np.asarray(self.rho, dtype=np.float64)
        self.pressure = np.asarray(self.pressure, dtype=np.float64)
        self.mass = np.asarray(self.mass, dtype=np.float64)

        n = self.pos.shape[0]
        if self.pos.ndim != 2 or self.pos.shape[1] != 3:
            raise ValueError("pos must have shape (N, 3)")
        if self.vel.shape != (n, 3):
            raise ValueError("vel must have shape (N, 3)")
        if self.acc.shape != (n, 3):
            raise ValueError("acc must have shape (N, 3)")
        if self.rho.shape != (n,):
            raise ValueError("rho must have shape (N,)")
        if self.pressure.shape != (n,):
            raise ValueError("pressure must have shape (N,)")
        if self.mass.shape != (n,):
            raise ValueError("mass must have shape (N,)")


def init_dam_break(cfg: SimConfig) -> Particles:
    """Initialize a classic dam-break column in the left third of the domain.

    The water block dimensions are fixed to 0.4 × 0.8 × 0.4 meters and seeded on a
    lattice perturbed by small uniform jitter.
    """

    dx = cfg.particle_spacing
    half = 0.5 * dx

    x_max = min(0.4, cfg.box_size[0] / 3.0)
    y_max = min(0.8, cfg.box_size[1])
    z_max = min(0.4, cfg.box_size[2])

    xs = np.arange(half, x_max, dx)
    ys = np.arange(half, y_max, dx)
    zs = np.arange(half, z_max, dx)

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    pos = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)

    rng = np.random.default_rng(cfg.seed)
    jitter_mag = cfg.jitter * dx
    pos += rng.uniform(-jitter_mag, jitter_mag, size=pos.shape)

    pos[:, 0] = np.clip(pos[:, 0], 0.0, x_max)
    pos[:, 1] = np.clip(pos[:, 1], 0.0, y_max)
    pos[:, 2] = np.clip(pos[:, 2], 0.0, z_max)

    n = pos.shape[0]
    vel = np.zeros((n, 3), dtype=np.float64)
    acc = np.zeros((n, 3), dtype=np.float64)
    rho = np.full(n, cfg.rho0, dtype=np.float64)
    pressure = np.zeros(n, dtype=np.float64)
    mass = np.full(n, cfg.rho0 * cfg.particle_volume, dtype=np.float64)
    return Particles(pos=pos, vel=vel, acc=acc, rho=rho, pressure=pressure, mass=mass)


def wendland_c2(r: np.ndarray, h: float) -> np.ndarray:
    """Evaluate the 3D Wendland C2 kernel W(r, h) vectorized over ``r``."""

    r_arr = np.asarray(r, dtype=np.float64)
    q = r_arr / h
    val = np.zeros_like(q)
    mask = q < 2.0
    qm = q[mask]
    term = 1.0 - 0.5 * qm
    val[mask] = (_WENDLAND_C2_ALPHA_3D / h**3) * (term**4) * (2.0 * qm + 1.0)
    return val


def grad_wendland_c2(r_vec: np.ndarray, h: float) -> np.ndarray:
    """Evaluate ∇W(r_vec, h) for the 3D Wendland C2 kernel."""

    rv = np.asarray(r_vec, dtype=np.float64)
    if rv.ndim == 1:
        rv = rv[None, :]

    r = np.linalg.norm(rv, axis=1)
    q = r / h
    grad = np.zeros_like(rv)

    mask = q < 2.0
    coeff = np.zeros_like(r)
    coeff[mask] = -5.0 * (_WENDLAND_C2_ALPHA_3D / h**5) * (1.0 - 0.5 * q[mask]) ** 3
    grad[mask] = coeff[mask, None] * rv[mask]
    return grad


def integrate_wendland_c2_over_sphere(h: float, n_samples: int = 4096) -> float:
    """Numerically approximate ∫ W dV over the support sphere (r ∈ [0, 2h])."""

    r = np.linspace(0.0, 2.0 * h, n_samples)
    integrand = 4.0 * np.pi * (r**2) * wendland_c2(r, h)
    return float(np.trapz(integrand, r))


def kernel_quick_plot_test(h: float = 0.1, n_samples: int = 512) -> float:
    """Quick visual + numeric sanity check for the Wendland C2 kernel.

    Returns the numeric normalization integral, expected to be approximately 1.0.
    """

    import matplotlib.pyplot as plt

    r = np.linspace(0.0, 2.0 * h, n_samples)
    w = wendland_c2(r, h)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r, w)
    ax.set_title("Wendland C2 kernel")
    ax.set_xlabel("r")
    ax.set_ylabel("W(r, h)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return integrate_wendland_c2_over_sphere(h=h, n_samples=max(2048, n_samples))


_TREE_CACHE: dict[tuple[int, tuple[int, ...], tuple[int, ...]], cKDTree] = {}


def _tree_cache_key(pos: np.ndarray) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    return (pos.__array_interface__["data"][0], pos.shape, pos.strides)


def find_neighbors(pos: np.ndarray, h: float) -> list[np.ndarray]:
    """Return neighbor indices within ``2h`` for each particle using ``cKDTree``."""

    pos_arr = np.asarray(pos, dtype=np.float64)
    key = _tree_cache_key(pos_arr)
    tree = _TREE_CACHE.get(key)
    if tree is None:
        tree = cKDTree(pos_arr)
        _TREE_CACHE.clear()
        _TREE_CACHE[key] = tree

    neighbor_ids = tree.query_ball_point(pos_arr, r=2.0 * h)
    return [np.asarray(ids, dtype=np.int64) for ids in neighbor_ids]


if __name__ == "__main__":
    value = kernel_quick_plot_test()
    print(f"Kernel normalization integral ≈ {value:.6f}")
