"""Boundary handling utilities for boxed SPH domains."""

from __future__ import annotations

import numpy as np

from sph_solver.config import SimConfig
from sph_solver.core import Particles


def enforce_box(p: Particles, cfg: SimConfig) -> None:
    """Reflect particles off axis-aligned box walls with restitution 0.5."""

    box = np.asarray(cfg.box_size, dtype=np.float64)

    lower = p.pos < 0.0
    if np.any(lower):
        p.pos[lower] = -p.pos[lower]
        p.vel[lower] = -0.5 * p.vel[lower]

    upper = p.pos > box[None, :]
    if np.any(upper):
        box_full = np.broadcast_to(box, p.pos.shape)
        p.pos[upper] = 2.0 * box_full[upper] - p.pos[upper]
        p.vel[upper] = -0.5 * p.vel[upper]
