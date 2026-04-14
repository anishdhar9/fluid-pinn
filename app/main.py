"""Gradio entrypoint for the fluid-pinn app."""

from __future__ import annotations

import gradio as gr

from app.compare_tab import build_compare_tab
from app.pinn_tab import build_pinn_tab
from app.sim_tab import build_sim_tab


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Fluid-PINN") as demo:
        gr.Markdown("# Fluid-PINN")
        with gr.Tabs():
            with gr.Tab("Run SPH Simulation"):
                build_sim_tab()
            with gr.Tab("Compare"):
                build_compare_tab()
            with gr.Tab("PINN"):
                build_pinn_tab()
    return demo


demo = build_app()

if __name__ == "__main__":
    demo.queue().launch()
