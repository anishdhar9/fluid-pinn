"""Simulation tab UI and callbacks."""

from __future__ import annotations

import queue
import tempfile
import threading
import time
from pathlib import Path

import gradio as gr
import numpy as np
import plotly.graph_objects as go

from sph_solver.boundary import enforce_box
from sph_solver.config import SimConfig
from sph_solver.core import Particles, find_neighbors
from sph_solver.export import write_hdf5_trajectory
from sph_solver.integrator import leapfrog_step


def _make_frame_figure(pos: np.ndarray, vel: np.ndarray, box_size: tuple[float, float, float]) -> go.Figure:
    speed = np.linalg.norm(vel, axis=1)
    max_speed = max(float(np.max(speed)), 1e-9)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=pos[:, 0],
                y=pos[:, 1],
                z=pos[:, 2],
                mode="markers",
                marker={
                    "size": 3,
                    "color": speed,
                    "colorscale": "Viridis",
                    "cmin": 0.0,
                    "cmax": max_speed,
                    "opacity": 0.9,
                },
            )
        ]
    )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        title="SPH Particles (color = |v|)",
        scene={
            "xaxis": {"range": [0, box_size[0]]},
            "yaxis": {"range": [0, box_size[1]]},
            "zaxis": {"range": [0, box_size[2]]},
        },
    )
    return fig


def _simulate_worker(
    out_q: queue.Queue,
    n_particles: int,
    viscosity_log10: float,
    box_x: float,
    box_y: float,
    box_z: float,
    dt: float,
    t_end: float,
) -> None:
    try:
        viscosity = 10.0 ** viscosity_log10
        spacing = (box_x * box_y * box_z / max(n_particles, 1)) ** (1.0 / 3.0)
        cfg = SimConfig(
            box_size=(box_x, box_y, box_z),
            particle_spacing=spacing,
            h=1.5 * spacing,
            dt=dt,
            mu=viscosity,
        )

        rng = np.random.default_rng(123)
        pos = rng.uniform([0.0, 0.0, 0.0], [box_x, box_y, box_z], size=(n_particles, 3)).astype(np.float64)
        vel = np.zeros((n_particles, 3), dtype=np.float64)

        p = Particles(
            pos=pos,
            vel=vel,
            acc=np.zeros((n_particles, 3), dtype=np.float64),
            rho=np.full(n_particles, cfg.rho0, dtype=np.float64),
            pressure=np.zeros(n_particles, dtype=np.float64),
            mass=np.full(n_particles, cfg.rho0 * cfg.particle_volume, dtype=np.float64),
        )

        n_steps = max(1, int(np.ceil(t_end / dt)))
        sample_every = 10
        sampled_pos = []
        sampled_vel = []

        def neighbors_fn(curr_pos: np.ndarray, h: float) -> list[np.ndarray]:
            return find_neighbors(curr_pos, h)

        for step in range(n_steps):
            leapfrog_step(p, cfg, neighbors_fn)
            enforce_box(p, cfg)

            if step % sample_every == 0 or step == n_steps - 1:
                sampled_pos.append(p.pos.copy())
                sampled_vel.append(p.vel.copy())
                out_q.put(("frame", _make_frame_figure(p.pos, p.vel, cfg.box_size), (step + 1) / n_steps))

        sampled_pos_arr = np.stack(sampled_pos, axis=0)
        sampled_vel_arr = np.stack(sampled_vel, axis=0)

        out_dir = Path(tempfile.mkdtemp(prefix="sph_run_"))
        out_path = out_dir / "sph_trajectory.h5"
        file_path = write_hdf5_trajectory(out_path, sampled_pos_arr, sampled_vel_arr, dt=dt * sample_every)
        out_q.put(("done", file_path, 1.0))
    except Exception as exc:  # runtime callback safety
        out_q.put(("error", str(exc), 1.0))


def run_simulation_stream(
    n_particles: int,
    viscosity_log10: float,
    box_x: float,
    box_y: float,
    box_z: float,
    dt: float,
    t_end: float,
    progress: gr.Progress = gr.Progress(),
):
    """Run simulation in a worker thread and stream plot updates to Gradio."""

    out_q: queue.Queue = queue.Queue()
    worker = threading.Thread(
        target=_simulate_worker,
        args=(out_q, n_particles, viscosity_log10, box_x, box_y, box_z, dt, t_end),
        daemon=True,
    )
    worker.start()

    latest_fig = go.Figure()
    yielded_final = False

    while worker.is_alive() or not out_q.empty():
        try:
            kind, payload, ratio = out_q.get(timeout=0.1)
        except queue.Empty:
            time.sleep(0.02)
            continue

        progress(ratio)
        if kind == "frame":
            latest_fig = payload
            yield gr.update(value=latest_fig), gr.update(value=None)
        elif kind == "done":
            yielded_final = True
            yield gr.update(value=latest_fig), gr.update(value=payload)
        elif kind == "error":
            yielded_final = True
            raise gr.Error(f"Simulation failed: {payload}")

    if not yielded_final:
        yield gr.update(value=latest_fig), gr.update(value=None)


def build_sim_tab() -> None:
    """Render the 'Run SPH Simulation' tab."""

    with gr.Row():
        n_particles = gr.Slider(100, 5000, value=1000, step=50, label="n_particles")
        viscosity = gr.Slider(-4.0, -1.0, value=-3.0, step=0.1, label="log10(viscosity)")

    with gr.Row():
        box_x = gr.Slider(0.5, 3.0, value=1.2, step=0.1, label="box_x")
        box_y = gr.Slider(0.5, 3.0, value=0.8, step=0.1, label="box_y")
        box_z = gr.Slider(0.5, 3.0, value=0.8, step=0.1, label="box_z")

    with gr.Row():
        dt = gr.Slider(1e-4, 5e-2, value=5e-3, step=1e-4, label="dt")
        t_end = gr.Slider(0.1, 10.0, value=2.0, step=0.1, label="t_end")

    run_btn = gr.Button("Run simulation", variant="primary")
    plot_out = gr.Plot(label="Particle animation")
    file_out = gr.File(label="Download HDF5 output")

    run_btn.click(
        fn=run_simulation_stream,
        inputs=[n_particles, viscosity, box_x, box_y, box_z, dt, t_end],
        outputs=[plot_out, file_out],
    )
