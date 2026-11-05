import os
from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

from skiing_simulation.slope.SlopeModel import SkiModel
from skiing_simulation.agents.base_skier import BaseSkier

# Paleta unikalnych, wyraźnych kolorów dla narciarzy
SKIER_COLORS = [
    "#1f77b4",  # niebieski
    "#e41a1c",  # czerwony
    "#2ca02c",  # zielony
    "#ff7f0e",  # pomarańczowy
    "#984ea3",  # fioletowy
    "#00ced1",  # morski / turkusowy
    "#a65628",  # brązowy
    "#f781bf",  # różowy
]


class SlopeVisualizer:
    """
    Moduł wizualizacji symulacji stoku narciarskiego (Top-Down)
    zgodny z artykułem Holleczek & Tröster (2012).
    """

    def __init__(self, model: SkiModel):
        self.model = model
        self.terrain = model.terrain

    def setup_figure(self) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes, plt.Axes, plt.Axes, plt.Axes]]:
        """Przygotowuje układ wykresów: rzut stoku z góry + telemetria (prędkość, tarcie, wysokość)."""
        fig = plt.figure(figsize=(14, 8.8), dpi=100)
        gs = fig.add_gridspec(
            4, 3,
            width_ratios=[1.7, 1.0, 1.0],
            height_ratios=[1.0, 1.0, 1.0, 0.32],
            hspace=0.38, wspace=0.35,
        )

        ax_slope = fig.add_subplot(gs[0:3, 0])
        ax_status = fig.add_subplot(gs[3, 0])
        ax_status.axis("off")  # Czyste pole pod stokiem na panel telemetryczny

        ax_speed = fig.add_subplot(gs[0, 1:])
        ax_friction = fig.add_subplot(gs[1, 1:])
        ax_elev = fig.add_subplot(gs[2:, 1:])

        self._draw_static_terrain(ax_slope)

        ax_slope.set_xlim(-2, self.terrain.width + 2)
        ax_slope.set_ylim(self.terrain.length + 5, -5)  # Odwrócona oś Y (0 = szczyt)
        ax_slope.set_aspect("auto")
        ax_slope.set_title("Stok narciarski (Model Holleczek & Tröster 2012)", fontsize=11, fontweight="bold")
        ax_slope.set_xlabel("Szerokość trasy [m]")
        ax_slope.set_ylabel("Długość stoku [m] (0 = Szczyt)")
        ax_slope.grid(True, linestyle=":", alpha=0.4)

        ax_speed.set_title("Prędkość narciarzy [km/h]", fontsize=10, fontweight="bold")
        ax_speed.set_ylabel("km/h")
        ax_speed.set_ylim(0.0, 90.0)
        ax_speed.grid(True, linestyle=":", alpha=0.5)

        ax_friction.set_title("Współczynnik tarcia podłoża (μ)", fontsize=10, fontweight="bold")
        ax_friction.set_ylabel("μ")
        ax_friction.set_ylim(0.0, 0.20)
        ax_friction.grid(True, linestyle=":", alpha=0.5)

        slope_label = f"Stok {self.terrain.slope_name.upper()} (Engelberg, Szwajcaria)"
        ax_elev.set_title(f"Profil wysokości Z(y) [m n.p.m.] - {slope_label}", fontsize=10, fontweight="bold")
        ax_elev.set_xlabel("Długość zjazdu [m]")
        ax_elev.set_ylabel("Wysokość [m]")
        ax_elev.set_ylim(self.terrain.finish_altitude - 20, self.terrain.start_altitude + 20)
        ax_elev.grid(True, linestyle=":", alpha=0.5)

        return fig, (ax_slope, ax_speed, ax_friction, ax_elev, ax_status)

    def _draw_static_terrain(self, ax: plt.Axes) -> None:
        """Rysuje statyczne elementy stoku: poziomice, granice, przeszkody."""
        # 1. Poziomice wysokości terenu
        ax.contourf(
            self.terrain.X,
            self.terrain.Y,
            self.terrain.elevation,
            levels=18,
            cmap="Blues_r",
            alpha=0.25,
        )

        # 2. Granice trasy (linie brzegowe)
        ax.plot(self.terrain.left_border, self.terrain.y_coords, "r--", linewidth=2.0, label="Krawędź stoku")
        ax.plot(self.terrain.right_border, self.terrain.y_coords, "r--", linewidth=2.0)

        # 3. Płaty lodu (o ile występują)
        ice_mask = self.terrain.friction_map < 0.08
        if np.any(ice_mask):
            ax.contourf(
                self.terrain.X,
                self.terrain.Y,
                self.terrain.friction_map,
                levels=[0.0, 0.06, 0.09],
                colors=["#00ffff", "#80e5ff"],
                alpha=0.45,
            )
            ax.plot([], [], color="#00cccc", linewidth=5, alpha=0.6, label="Strefa lodu")

        # 4. Przeszkody
        for obs in self.terrain.obstacles:
            color = "forestgreen" if obs.obstacle_type == "tree" else "dimgray"
            circle = patches.Circle(
                (obs.x, obs.y),
                obs.radius,
                facecolor=color,
                edgecolor="black",
                linewidth=1.2,
                alpha=0.85,
            )
            ax.add_patch(circle)

    def render_static_result(self, output_path: str = "simulation_result.png") -> None:
        """Zapisuje podsumowanie ukończonej symulacji z trajektorią narciarza i telemetrią."""
        fig, (ax_slope, ax_speed, ax_friction, ax_elev, ax_status) = self.setup_figure()

        skiers = [a for a in self.model.agents if isinstance(a, BaseSkier)]

        for i, skier in enumerate(skiers):
            if not skier.trajectory:
                continue

            traj = np.array([(p[0], p[1]) for p in skier.trajectory])
            speeds = np.array([p[2] * 3.6 for p in skier.trajectory])
            frictions = np.array([p[3] for p in skier.trajectory])
            ys = traj[:, 1]
            elevations = [self.terrain.get_elevation(p[0], p[1]) for p in skier.trajectory]

            c = SKIER_COLORS[i % len(SKIER_COLORS)]

            # Rysowanie punktów nawigacyjnych (waypoints) w kolorze danego narciarza
            if hasattr(skier, "waypoints") and skier.waypoints:
                wps = np.array(skier.waypoints)
                ax_slope.scatter(
                    wps[:, 0], wps[:, 1],
                    color=c, s=50, marker="D",
                    edgecolors="black", linewidths=1.2,
                    zorder=5, label=f"Cele #{skier.skier_id}" if len(skiers) <= 4 else None,
                )

            # Rysowanie trajektorii zjazdu
            ax_slope.plot(traj[:, 0], traj[:, 1], color=c, linewidth=2.2, label=f"Narciarz #{skier.skier_id}", zorder=4)

            # Wektory orientacji nart co kilkadziesiąt kroków
            step_skip = max(len(traj) // 25, 1)
            for idx in range(0, len(traj) - 1, step_skip):
                p_curr = traj[idx]
                p_next = traj[min(idx + 5, len(traj) - 1)]
                dir_vec = p_next - p_curr
                dir_norm = np.hypot(dir_vec[0], dir_vec[1])
                if dir_norm > 1e-4:
                    dir_u = (dir_vec / dir_norm) * 5.0
                    ax_slope.annotate(
                        "",
                        xy=(p_curr[0] + dir_u[0], p_curr[1] + dir_u[1]),
                        xytext=(p_curr[0], p_curr[1]),
                        arrowprops=dict(arrowstyle="->", color=c, lw=1.2, mutation_scale=9),
                        zorder=6,
                    )

            # Start i meta dla danego narciarza
            ax_slope.scatter(traj[0, 0], traj[0, 1], color=c, s=90, zorder=7, marker="o", edgecolors="white", linewidths=1.5)
            end_marker = "*" if skier.status == "finished" else "X"
            ax_slope.scatter(traj[-1, 0], traj[-1, 1], color=c, s=130, zorder=7, marker=end_marker, edgecolors="black", linewidths=1.5)

            # Wykres prędkości każdego narciarza z osobna
            ax_speed.plot(ys, speeds, color=c, linewidth=1.8, label=f"Narciarz #{skier.skier_id}")

            # Wykres tarcia podłoża każdego narciarza z osobna
            ax_friction.plot(ys, frictions, color=c, linewidth=1.8, label=f"μ #{skier.skier_id}")

            # Profil wysokości (wystarczy dla pierwszego narciarza)
            if i == 0:
                ax_elev.plot(ys, elevations, color="navy", linewidth=1.8)

        # Linie referencyjne
        ax_speed.axhline(54.0, color="green", linestyle=":", alpha=0.7, label="Średnia w artykule (~54 km/h)")
        ax_friction.axhline(0.10, color="black", linestyle="--", alpha=0.5, label="Śnieg bazowy (μ=0.10)")

        # Statystyki zjazdu - umieszczone pod stokiem w ax_status (nie zachodzą na stok!)
        stats = self.model.get_summary_stats()
        status_text = (
            f"Czas: {stats['time_s']:.1f} s  |  Narciarze: {stats['total_skiers']} (ukończyli: {stats['finished']})  |  "
            f"V_avg: {stats['avg_speed_kmh']:.1f} km/h\n"
            f"Model: Holleczek & Tröster (2012) Phys. Rev. E 85, 056101"
        )
        ax_status.text(
            0.5, 0.5,
            status_text,
            transform=ax_status.transAxes,
            ha="center", va="center",
            fontsize=9.0,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", alpha=0.95, edgecolor="#ced4da"),
        )

        ax_slope.legend(loc="upper right", fontsize=7.5, framealpha=0.85)
        ax_speed.legend(loc="upper right", fontsize=7.5, framealpha=0.85)
        ax_friction.legend(loc="upper right", fontsize=7.5, framealpha=0.85)
        fig.subplots_adjust(left=0.07, right=0.96, top=0.94, bottom=0.05, wspace=0.35, hspace=0.38)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"[SlopeVisualizer] Wykres zapisany pomyślnie w: {output_path}")

    def animate(
        self,
        max_steps: int = 15000,
        interval: int = 20,
        steps_per_frame: int = 6,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Interaktywna animacja zjazdu ze zoptymalizowanym renderowaniem
        oraz możliwością dynamicznej zmiany tempa klawiszami (+ / - / Spacja).
        """
        fig, (ax_slope, ax_speed, ax_friction, ax_elev, ax_status) = self.setup_figure()

        # Ustawienie stałych limitów osi (eliminuje kosztowne przeliczanie widoku co klatkę)
        ax_speed.set_xlim(0, self.terrain.length)
        ax_speed.set_ylim(0, 95)
        ax_friction.set_xlim(0, self.terrain.length)
        ax_friction.set_ylim(0, 0.20)
        ax_elev.set_xlim(0, self.terrain.length)
        ax_elev.set_ylim(self.terrain.finish_altitude - 20, self.terrain.start_altitude + 20)
        
        y_pts = self.terrain.y_coords
        elev_pts = [self.terrain.get_elevation(self.terrain.width / 2.0, y) for y in y_pts]
        ax_elev.plot(y_pts, elev_pts, color="navy", linewidth=1.8)

        skiers = [a for a in self.model.agents if isinstance(a, BaseSkier)]
        skier_dots = []
        skier_trails = []
        speed_lines = []
        friction_lines = []

        x_trails = [[] for _ in skiers]
        y_trails = [[] for _ in skiers]
        speed_trails = [[] for _ in skiers]
        friction_trails = [[] for _ in skiers]

        for i, skier in enumerate(skiers):
            c = SKIER_COLORS[i % len(SKIER_COLORS)]
            if hasattr(skier, "waypoints") and skier.waypoints:
                wps = np.array(skier.waypoints)
                ax_slope.scatter(
                    wps[:, 0], wps[:, 1],
                    color=c, s=45, marker="D",
                    edgecolors="black", linewidths=1.2, zorder=5,
                    label=f"Cele #{skier.skier_id}" if len(skiers) <= 4 else None,
                )

            (trail,) = ax_slope.plot([], [], "-", color=c, linewidth=2.0, alpha=0.85, zorder=4, label=f"Narciarz #{skier.skier_id}")
            (dot,) = ax_slope.plot([], [], "o", color=c, markersize=8, markeredgecolor="black", markeredgewidth=1.0, zorder=6)
            skier_trails.append(trail)
            skier_dots.append(dot)

            (s_line,) = ax_speed.plot([], [], "-", color=c, linewidth=1.8, label=f"Narciarz #{skier.skier_id}")
            speed_lines.append(s_line)

            (f_line,) = ax_friction.plot([], [], "-", color=c, linewidth=1.8, label=f"μ #{skier.skier_id}")
            friction_lines.append(f_line)

            # Punkt początkowy
            x_trails[i].append(skier.pos[0])
            y_trails[i].append(skier.pos[1])
            speed_trails[i].append(skier.speed_kmh)
            friction_trails[i].append(self.terrain.get_friction(skier.pos[0], skier.pos[1]))

        # Linie referencyjne
        ax_speed.axhline(54.0, color="green", linestyle=":", alpha=0.7, label="Średnia z artykułu (54 km/h)")
        ax_speed.legend(loc="upper right", fontsize=7.5, framealpha=0.85)

        ax_friction.axhline(0.10, color="black", linestyle="--", alpha=0.6, label="Śnieg bazowy (μ=0.10)")
        ax_friction.legend(loc="upper right", fontsize=7.5, framealpha=0.85)

        ax_slope.legend(loc="upper right", fontsize=7.5, framealpha=0.85)

        # Panel telemetryczny umieszczony pod stokiem w ax_status (nie zasłania stoku!)
        title_text = ax_status.text(
            0.5, 0.5, "",
            transform=ax_status.transAxes,
            ha="center", va="center",
            fontsize=9.0,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", alpha=0.95, edgecolor="#ced4da")
        )

        # Stan interaktywności (sterowanie klawiaturą)
        sim_state = {
            "paused": False,
            "speedup": steps_per_frame,
        }

        def on_key(event):
            if event.key == " ":
                sim_state["paused"] = not sim_state["paused"]
            elif event.key in ["+", "=", "]", "up"]:
                sim_state["speedup"] = min(sim_state["speedup"] + 2, 40)
            elif event.key in ["-", "_", "[", "down"]:
                sim_state["speedup"] = max(sim_state["speedup"] - 2, 1)

        fig.canvas.mpl_connect("key_press_event", on_key)

        def update(frame: int):
            if not sim_state["paused"]:
                # Wykonujemy N kroków fizyki na jedną klatkę animacji
                for _ in range(sim_state["speedup"]):
                    if any(s.status == "active" for s in skiers):
                        self.model.step()

            # Zoptymalizowana aktualizacja współrzędnych bez alokowania całych macierzy
            for i, skier in enumerate(skiers):
                x_trails[i].append(skier.pos[0])
                y_trails[i].append(skier.pos[1])
                speed_trails[i].append(skier.speed_kmh)
                friction_trails[i].append(self.terrain.get_friction(skier.pos[0], skier.pos[1]))

                skier_trails[i].set_data(x_trails[i], y_trails[i])
                skier_dots[i].set_data([skier.pos[0]], [skier.pos[1]])
                speed_lines[i].set_data(y_trails[i], speed_trails[i])
                friction_lines[i].set_data(y_trails[i], friction_trails[i])

                if skier.status == "finished":
                    skier_dots[i].set_marker("*")
                    skier_dots[i].set_color("gold")
                    skier_dots[i].set_markersize(12)

            stats = self.model.get_summary_stats()
            pause_str = " [PAUZA]" if sim_state["paused"] else ""
            title_text.set_text(
                f"Czas: {stats['time_s']:.1f}s | "
                f"V_avg: {stats['avg_speed_kmh']:.1f} km/h | "
                f"Tempo: {sim_state['speedup']}x{pause_str}\n"
                f"[Spacja: Pauza | +/-: Zmiana tempa]"
            )
            return skier_trails + skier_dots + speed_lines + friction_lines + [title_text]

        fig.subplots_adjust(left=0.07, right=0.96, top=0.94, bottom=0.05, wspace=0.35, hspace=0.38)
        total_frames = max_steps // steps_per_frame
        anim = FuncAnimation(fig, update, frames=total_frames, interval=interval, blit=False, repeat=False)

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            print(f"[SlopeVisualizer] Zapisywanie animacji do {save_path}...")
            anim.save(save_path, writer="pillow", fps=25)
            print(f"[SlopeVisualizer] Animacja zapisana pomyślnie.")
            plt.close(fig)
        else:
            plt.show()
