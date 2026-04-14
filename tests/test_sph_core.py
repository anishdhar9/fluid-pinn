"""Tests for SPH core kernel and neighbor search utilities."""

from __future__ import annotations

import numpy as np

from sph_solver.core import find_neighbors, integrate_wendland_c2_over_sphere


def test_wendland_c2_kernel_normalization_within_two_percent() -> None:
    integral = integrate_wendland_c2_over_sphere(h=0.1, n_samples=8192)
    assert abs(integral - 1.0) < 0.02


def test_find_neighbors_lattice_counts_reasonable() -> None:
    # 5x5x5 regular lattice with spacing dx.
    dx = 0.05
    n = 5
    grid = np.stack(
        np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij"), axis=-1
    ).reshape(-1, 3)
    pos = grid.astype(np.float64) * dx

    neighbors = find_neighbors(pos, h=dx)
    counts = np.array([len(ids) - 1 for ids in neighbors])

    center_index = np.where(np.all(grid == np.array([2, 2, 2]), axis=1))[0][0]

    assert counts[center_index] == 32
    assert counts.min() >= 10
    assert counts.max() <= 32
