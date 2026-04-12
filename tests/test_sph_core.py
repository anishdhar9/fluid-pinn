"""SPH core tests."""

from sph_solver.config import SimConfig


def test_sim_config_defaults_satisfy_cfl_condition() -> None:
    cfg = SimConfig()
    assert cfg.dt < cfg.kernel_h / cfg.c_s
