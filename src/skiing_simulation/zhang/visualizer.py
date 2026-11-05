"""Wizualizacje trajektorii dla automatu komórkowego Zhang i in. (2024)."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from skiing_simulation.zhang.model import ZhangModel


SKIER_COLORS = ["#1f77b4", "#e41a1c", "#2ca02c", "#ff7f0e", "#984ea3", "#00a6a6"]


class ZhangVisualizer:
    """Wykres statyczny i animacja o układzie zgodnym z obecnymi wizualizacjami."""

    def __init__(self, model: ZhangModel) -> None:
        self.model = model

    def _setup_figure(self):
        fig = plt.figure(figsize=(13, 8), dpi=100)
        grid = fig.add_gridspec(
            4, 3, width_ratios=[1.35, 1.0, 1.0], height_ratios=[1, 1, 1, 0.30],
            hspace=0.46, wspace=0.38,
        )
        ax_slope = fig.add_subplot(grid[0:3, 0])
        ax_status = fig.add_subplot(grid[3, 0])
        ax_probability = fig.add_subplot(grid[0, 1:])
        ax_position = fig.add_subplot(grid[1, 1:])
        ax_state = fig.add_subplot(grid[2, 1:])
        ax_status.axis("off")

        ax_slope.set_xlim(-1, self.model.width)
        ax_slope.set_ylim(self.model.length + 2, -2)
        ax_slope.set_xlabel("x [komórka]")
        ax_slope.set_ylabel("y [komórka; 0 = szczyt]")
        ax_slope.set_title("Stok — automat komórkowy Zhang i in. (2024)", fontweight="bold", fontsize=10)
        ax_slope.set_facecolor("aliceblue")
        ax_slope.axvline(0, color="firebrick", linestyle="--", linewidth=1.4, label="granica wizualizacji")
        ax_slope.axvline(self.model.width - 1, color="firebrick", linestyle="--", linewidth=1.4)
        ax_slope.grid(True, linestyle=":", alpha=0.35)

        ax_probability.set_title("Aktywne prawdopodobieństwo kroku poprzecznego", fontweight="bold", fontsize=10)
        ax_probability.set_ylabel("p")
        ax_probability.set_ylim(-0.03, 1.03)
        ax_probability.axhline(self.model.epsilon, color="black", linestyle="--", linewidth=1, label="próg ε")
        ax_probability.grid(True, linestyle=":", alpha=0.45)

        ax_position.set_title("Pozycja poprzeczna x w funkcji postępu w dół", fontweight="bold", fontsize=10)
        ax_position.set_ylabel("x [komórka]")
        ax_position.set_ylim(-1, self.model.width)
        ax_position.grid(True, linestyle=":", alpha=0.45)

        ax_state.set_title("Stan pamięci skrętu", fontweight="bold", fontsize=10)
        ax_state.set_xlabel("krok automatu")
        ax_state.set_ylabel("strona")
        ax_state.set_yticks([0, 1], labels=["lewo", "prawo"])
        ax_state.set_ylim(-0.2, 1.2)
        ax_state.grid(True, linestyle=":", alpha=0.45)
        return fig, (ax_slope, ax_probability, ax_position, ax_state, ax_status)

    @staticmethod
    def _data(skier, end: int | None = None):
        positions = np.asarray(skier.trajectory[:end], dtype=float)
        probabilities = np.asarray(skier.probability_history[:end], dtype=float)
        states = np.asarray(skier.direction_history[:end], dtype=float)
        steps = np.arange(len(positions))
        return positions, probabilities, states, steps

    def render_static_result(self, output_path: str | Path) -> None:
        """Zapisuje kompletne podsumowanie już wykonanego przebiegu."""
        fig, axes = self._setup_figure()
        ax_slope, ax_probability, ax_position, ax_state, ax_status = axes
        for idx, skier in enumerate(self.model.skiers):
            color = SKIER_COLORS[idx % len(SKIER_COLORS)]
            positions, probabilities, states, steps = self._data(skier)
            if not len(positions):
                continue
            label = f"Narciarz #{skier.skier_id}"
            ax_slope.plot(positions[:, 0], positions[:, 1], color=color, linewidth=1.8, label=label)
            ax_slope.scatter(*positions[0], color=color, edgecolor="white", s=42, zorder=3)
            ax_slope.scatter(*positions[-1], color=color, marker="*", edgecolor="black", s=82, zorder=3)
            ax_probability.plot(steps, probabilities, color=color, linewidth=1.2, label=label)
            ax_position.plot(positions[:, 1], positions[:, 0], color=color, linewidth=1.2, label=label)
            ax_state.step(steps, states, where="post", color=color, linewidth=1.2, label=label)

        summary = self.model.summary()
        ax_status.text(
            0.5, 0.5,
            f"δ = {self.model.delta:.3f} | ε = {self.model.epsilon:.3f} | kroki: {summary['steps']} | "
            f"ukończyli: {summary['finished']}/{summary['skiers']} | kroki boczne: {summary['lateral_share']:.1%}",
            transform=ax_status.transAxes, ha="center", va="center", fontsize=8.8,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#f8f9fa", edgecolor="#ced4da"),
        )
        for axis in (ax_slope, ax_probability, ax_position, ax_state):
            axis.legend(loc="best", fontsize=7.3, framealpha=0.88)
        fig.subplots_adjust(left=0.07, right=0.97, top=0.94, bottom=0.06)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
        plt.close(fig)

    def animate(
        self,
        output_path: str | Path | None = None,
        steps_per_frame: int = 3,
        max_frames: int = 1_000,
        interval_ms: int = 50,
    ) -> None:
        """Uruchamia lub zapisuje animację; zapis GIF używa Pillow."""
        if steps_per_frame < 1:
            raise ValueError("steps_per_frame musi być dodatnie")
        fig, axes = self._setup_figure()
        ax_slope, ax_probability, ax_position, ax_state, ax_status = axes
        trails, dots, probability_lines, position_lines, state_lines = [], [], [], [], []
        for idx, skier in enumerate(self.model.skiers):
            color = SKIER_COLORS[idx % len(SKIER_COLORS)]
            label = f"Narciarz #{skier.skier_id}"
            (trail,) = ax_slope.plot([], [], color=color, linewidth=1.8, label=label)
            (dot,) = ax_slope.plot([], [], "o", color=color, markeredgecolor="black", markersize=6)
            (probability_line,) = ax_probability.plot([], [], color=color, linewidth=1.2, label=label)
            (position_line,) = ax_position.plot([], [], color=color, linewidth=1.2, label=label)
            (state_line,) = ax_state.step([], [], where="post", color=color, linewidth=1.2, label=label)
            trails.append(trail)
            dots.append(dot)
            probability_lines.append(probability_line)
            position_lines.append(position_line)
            state_lines.append(state_line)
        for axis in (ax_slope, ax_probability, ax_position, ax_state):
            axis.legend(loc="best", fontsize=7.3, framealpha=0.88)
        status = ax_status.text(0.5, 0.5, "", transform=ax_status.transAxes, ha="center", va="center", fontsize=8.8)

        def update(_frame: int):
            for _ in range(steps_per_frame):
                if self.model.active_skiers:
                    self.model.step()
            for idx, skier in enumerate(self.model.skiers):
                positions, probabilities, states, steps = self._data(skier)
                trails[idx].set_data(positions[:, 0], positions[:, 1])
                dots[idx].set_data([positions[-1, 0]], [positions[-1, 1]])
                probability_lines[idx].set_data(steps, probabilities)
                position_lines[idx].set_data(positions[:, 1], positions[:, 0])
                state_lines[idx].set_data(steps, states)
            xmax = max(20, self.model.step_count + 5)
            ax_probability.set_xlim(0, xmax)
            ax_state.set_xlim(0, xmax)
            summary = self.model.summary()
            status.set_text(
                f"krok: {summary['steps']} | aktywni: {summary['active']} | "
                f"δ = {self.model.delta:.3f}, ε = {self.model.epsilon:.3f}"
            )
            return trails + dots + probability_lines + position_lines + state_lines + [status]

        fig.subplots_adjust(left=0.07, right=0.97, top=0.94, bottom=0.06)
        animation = FuncAnimation(fig, update, frames=max_frames, interval=interval_ms, blit=False, repeat=False)
        if output_path is None:
            plt.show()
        else:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            animation.save(output, writer="pillow", fps=max(1, round(1000 / interval_ms)))
            plt.close(fig)
