"""Simulation configuration for the SPH scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(slots=True)
class SimConfig:
    """Basic simulation configuration used by the SPH core utilities."""

    box_size: Tuple[float, float, float] = (1.2, 0.8, 0.4)
    particle_spacing: float = 0.02
    rho0: float = 1000.0
    jitter: float = 0.1
    seed: int = 42

    @property
    def particle_volume(self) -> float:
        """Reference volume represented by one particle."""
        return self.particle_spacing**3
