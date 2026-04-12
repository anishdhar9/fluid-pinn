"""Simulation configuration for SPH scaffolding."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SimConfig:
    """Immutable simulation configuration for the SPH solver."""

    n_particles: int = 2000
    box: tuple[float, float, float] = (1.0, 1.0, 1.0)
    rho0: float = 1000.0
    viscosity: float = 0.001
    surface_tension: float = 0.0
    gravity: tuple[float, float, float] = (0.0, -9.81, 0.0)
    dt: float = 1e-4
    t_end: float = 2.0
    kernel_h: float = 0.05
    c_s: float = 10.0
    gamma: float = 7.0
    output_every: int = 10

    def __post_init__(self) -> None:
        if self.dt >= self.kernel_h / self.c_s:
            raise ValueError("CFL condition failed: require dt < kernel_h / c_s.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimConfig":
        """Build a configuration from a plain dictionary."""

        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a plain dictionary."""

        return asdict(self)
