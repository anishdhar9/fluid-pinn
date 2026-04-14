"""Snapshot/export helpers."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def write_hdf5_trajectory(
    path: str | Path,
    positions: np.ndarray,
    velocities: np.ndarray,
    dt: float,
) -> str:
    """Write sampled trajectory arrays to an HDF5 file and return the file path."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out, "w") as f:
        f.create_dataset("positions", data=positions)
        f.create_dataset("velocities", data=velocities)
        f.attrs["dt"] = float(dt)

    return str(out)
