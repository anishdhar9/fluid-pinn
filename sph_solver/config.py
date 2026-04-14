"""Simulation configuration for the SPH scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(slots=True)
class SimConfig:
    """Basic simulation configuration used by the SPH core utilities."""

    box_size: Tuple[float, float, float] = (1.2, 0.8, 0.4)
    particle_spacing: float = 0.02
    dt: float = 1.0e-3
    h: float | None = None
    rho0: float = 1000.0
    gamma: float = 7.0
    c_s: float = 20.0
    mu: float = 1.0e-3
    gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0)
    jitter: float = 0.1
    seed: int = 42

    @property
    def particle_volume(self) -> float:
        """Reference volume represented by one particle."""
        return self.particle_spacing**3

    @property
    def smoothing_length(self) -> float:
        """SPH smoothing length used by kernels/operators."""
        return self.particle_spacing if self.h is None else self.h
