"""Narzędzia wizualizacji i analityki dla modelu Kamińskiego (2026 - wersja rozszerzona).

Generuje zestaw wykresów walidacyjnych i diagnostycznych:
- Trajektorie narciarzy na tle granic, przeszkód (słupów wyciągu),
- Dynamiczną stygmergię stoku: mapę wyślizganego lodu i narastających muld,
- Kolizje, zderzenia i strefy wypadków,
- Zależność prędkości od nachylenia stoku (Rys. 6.13),
- Wskaźnik proporcji zakrętu dx/dy w funkcji stromości (Rys. 6.14),
- Animacje GIF z dynamiczną ewolucją stoku i ruchu narciarzy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Ellipse

from skiing_simulation.kaminski.model import SkiModel


class KaminskiVisualizer:
    """Wizualizator wyników symulacji stochastycznej F. Kamińskiego."""

    def __init__(self, model: SkiModel) -> None:
        self.model = model

    def render_dashboard(
        self,
        output_path: str | Path = "data/kaminski_simulation_result.png",
        dpi: int = 150,
    ) -> Path:
        """Tworzy 4-panelowy panel diagnostyczny weryfikujący trajektorie, stygmergię i dynamikę."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axs = plt.subplots(1, 4, figsize=(22, 8), gridspec_kw={"width_ratios": [1.3, 1.3, 1.4, 1.4]})
        plt.subplots_adjust(wspace=0.35)

        terrain = self.model.terrain
        y_vals = np.arange(terrain.length)
        left_borders = [terrain.get_borders_at(y)[0] for y in y_vals]
        right_borders = [terrain.get_borders_at(y)[1] for y in y_vals]

        # ----------------------------------------------------
        # Panel 1: Trajektorie na stoku z przeszkodami i stylami
        # ----------------------------------------------------
        ax1 = axs[0]
        ax1.plot(left_borders, y_vals, color="black", linewidth=2.0, label="Granica trasy")
        ax1.plot(right_borders, y_vals, color="black", linewidth=2.0)
        ax1.fill_betweenx(y_vals, 0, left_borders, color="gainsboro", alpha=0.8)
        ax1.fill_betweenx(y_vals, right_borders, terrain.width, color="gainsboro", alpha=0.8)

        # Rysowanie przeszkód stałych (słupy wyciągu)
        for obs in terrain.obstacles:
            c = Circle((obs["cx"], obs["cy"]), obs["radius"], color="black", zorder=8)
            ax1.add_patch(c)
            ax1.text(
                obs["cx"] + 2.5,
                obs["cy"],
                obs["label"],
                color="black",
                fontsize=7.5,
                fontweight="bold",
                va="center",
                zorder=9,
            )

        # Meta
        ax1.axhline(terrain.length - 1, color="crimson", linestyle="-", linewidth=2.5, label="Linia mety (350m)")
        ax1.fill_between([0, terrain.width], terrain.length - 1, terrain.length + 5, color="lightpink", alpha=0.3)

        # Trajektorie
        colors = plt.cm.tab10.colors
        for i, skier in enumerate(self.model.agents):
            if len(skier.trajectory) > 1:
                traj = np.array(skier.trajectory)
                c = colors[i % len(colors)]
                t_finish = len(skier.trajectory)
                lbl = f"#{i} {skier.style} ({skier.skill_level})" if len(self.model.agents) <= 8 else None
                ax1.plot(traj[:, 0], traj[:, 1], color=c, alpha=0.85, linewidth=1.8, label=lbl)
                ax1.scatter(traj[0, 0], traj[0, 1], color=c, s=25, zorder=5)
                end_x, end_y = traj[-1, 0], traj[-1, 1]
                ax1.scatter(end_x, end_y, marker="X", color=c, edgecolor="black", s=50, zorder=6)

        # Miejsca wypadków / zderzeń
        if self.model.crash_locations:
            crash_x = [p[0] for p in self.model.crash_locations]
            crash_y = [p[1] for p in self.model.crash_locations]
            ax1.scatter(crash_x, crash_y, marker="*", color="red", s=120, edgecolor="yellow", zorder=10, label=f"Wypadki ({len(crash_x)})")

        ax1.set_xlim(0, terrain.width)
        ax1.set_ylim(terrain.length + 6, 0)
        ax1.set_xlabel("Szerokość x [m]")
        ax1.set_ylabel("Długość stoku y [m]")
        ax1.set_title("Trajektorie i przeszkody\n(zróżnicowane style i wypadki)")
        if len(self.model.agents) <= 8:
            ax1.legend(loc="upper right", fontsize=7.5, framealpha=0.85)
        ax1.grid(True, linestyle=":", alpha=0.5)

        # ----------------------------------------------------
        # Panel 2: Dynamiczna stygmergia stoku (Zużycie, Lód i Muldy)
        # ----------------------------------------------------
        ax2 = axs[1]
        # Tworzymy barwną kompozycję nawierzchni:
        # Błękitny = lód (I), Brązowo-złoty = muldy (B)
        surface_rgb = np.ones((terrain.length, terrain.width, 3), dtype=float)
        # Dodajemy lód: zmniejsza czerwony i zielony -> niebieski [0.0, 0.7, 1.0]
        ice = terrain.ice_map[:, :, np.newaxis]
        moguls = terrain.moguls_map[:, :, np.newaxis]

        surface_rgb = surface_rgb * (1.0 - 0.7 * ice) + np.array([0.0, 0.6, 0.95]) * (0.7 * ice)
        surface_rgb = surface_rgb * (1.0 - 0.6 * moguls) + np.array([0.85, 0.45, 0.05]) * (0.6 * moguls)

        ax2.imshow(surface_rgb, aspect="auto", extent=[0, terrain.width, terrain.length, 0])
        ax2.plot(left_borders, y_vals, color="black", linewidth=1.5, linestyle="--")
        ax2.plot(right_borders, y_vals, color="black", linewidth=1.5, linestyle="--")

        # Rysowanie przeszkód stałych na mapie stygmergii
        for obs in terrain.obstacles:
            c = Circle((obs["cx"], obs["cy"]), obs["radius"], color="black", zorder=8)
            ax2.add_patch(c)

        ax2.set_xlabel("Szerokość x [m]")
        ax2.set_ylabel("Długość stoku y [m]")
        ax2.set_title(f"Stygmergia stoku (Zużycie)\nNiebieski=Lód, Pomarańczowy=Muldy")

        # ----------------------------------------------------
        # Panel 3: Prędkość w zakręcie a nachylenie (Rys. 6.13)
        # ----------------------------------------------------
        ax3 = axs[2]
        all_metrics = []
        for skier in self.model.agents:
            all_metrics.extend(skier.turns_metrics)

        if all_metrics:
            slopes = [m["slope_inclination"] for m in all_metrics]
            speeds = [m["avg_speed"] for m in all_metrics]
            ax3.scatter(slopes, speeds, alpha=0.55, edgecolors="none", s=25, c="dodgerblue")

            if len(slopes) > 5:
                z = np.polyfit(slopes, speeds, 1)
                p = np.poly1d(z)
                x_line = np.linspace(min(slopes), max(slopes), 50)
                ax3.plot(x_line, p(x_line), "r--", linewidth=2.0, label="Trend liniowy")
                ax3.legend()

        ax3.set_xlabel("Lokalne nachylenie stoku Δh/Δy")
        ax3.set_ylabel("Średnia prędkość w zakręcie [m/s]")
        ax3.set_title("Prędkość w zakręcie a nachylenie stoku\n(porównanie z Rys. 6.13)")
        ax3.grid(True, linestyle=":", alpha=0.5)

        # ----------------------------------------------------
        # Panel 4: Stosunek dx/dy a nachylenie stoku (Rys. 6.14)
        # ----------------------------------------------------
        ax4 = axs[3]
        if all_metrics:
            dx_dy = [m["dx_dy_ratio"] for m in all_metrics]
            ax4.scatter(slopes, dx_dy, alpha=0.55, edgecolors="none", s=25, c="darkorange")
            ax4.axhline(1.0, color="grey", linestyle="--", label="dx/dy = 1 (granica)")
            ax4.legend()

        ax4.set_xlabel("Lokalne nachylenie stoku Δh/Δy")
        ax4.set_ylabel("Stosunek dx / dy (szerokość / długość)")
        ax4.set_title("Stosunek dx/dy a nachylenie stoku\n(porównanie z Rys. 6.14)")
        ax4.set_ylim(0, 1.6)
        ax4.grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi)
        plt.close(fig)
        return output_path

    def render_animation(
        self,
        output_path: str | Path = "data/kaminski_simulation_demo.gif",
        fps: int = 8,
        dpi: int = 100,
    ) -> Path:
        """Tworzy animację GIF prezentującą zjazd narciarzy, anizotropowe uniki i degradację stoku."""
        from matplotlib.animation import FuncAnimation

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        terrain = self.model.terrain
        total_skiers = len(self.model.agents)
        max_frames = max(len(s.trajectory) for s in self.model.agents) if self.model.agents else 1

        fig = plt.figure(figsize=(15, 8.5), dpi=dpi)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0], height_ratios=[1.0, 1.0], hspace=0.35, wspace=0.25)

        ax_slope = fig.add_subplot(gs[:, 0])
        ax_speed = fig.add_subplot(gs[0, 1])
        ax_turn = fig.add_subplot(gs[1, 1])

        y_vals = np.arange(terrain.length)
        left_borders = [terrain.get_borders_at(y)[0] for y in y_vals]
        right_borders = [terrain.get_borders_at(y)[1] for y in y_vals]

        # Podkład stoku z dynamicznym podglądem powierzchni
        surface_img = ax_slope.imshow(
            np.ones((terrain.length, terrain.width, 3)),
            aspect="auto",
            extent=[0, terrain.width, terrain.length, 0],
            zorder=1,
        )

        ax_slope.plot(left_borders, y_vals, color="black", linewidth=2.0, zorder=3)
        ax_slope.plot(right_borders, y_vals, color="black", linewidth=2.0, zorder=3)
        ax_slope.fill_betweenx(y_vals, 0, left_borders, color="#d0d0d0", alpha=0.9, zorder=2)
        ax_slope.fill_betweenx(y_vals, right_borders, terrain.width, color="#d0d0d0", alpha=0.9, zorder=2)

        # Przeszkody stałe
        for obs in terrain.obstacles:
            c = Circle((obs["cx"], obs["cy"]), obs["radius"], color="black", zorder=6)
            ax_slope.add_patch(c)
            ax_slope.text(
                obs["cx"],
                obs["cy"] - 3.0,
                obs["label"],
                color="black",
                fontsize=7.5,
                fontweight="bold",
                ha="center",
                zorder=7,
            )

        finish_y = terrain.length - 1
        ax_slope.axhline(finish_y, color="crimson", linestyle="-", linewidth=2.5, zorder=4)
        ax_slope.fill_between([0, terrain.width], finish_y, terrain.length + 15, color="#d8f3dc", alpha=0.55, zorder=2)
        ax_slope.text(
            terrain.width / 2.0,
            finish_y - 3.5,
            "=== LINIA METY (350m) ===",
            color="crimson",
            fontweight="bold",
            ha="center",
            fontsize=9,
            zorder=5,
        )

        ax_slope.set_xlim(-1, terrain.width + 1)
        ax_slope.set_ylim(terrain.length + 14, -5)
        ax_slope.set_xlabel("Szerokość x [m]")
        ax_slope.set_ylabel("Długość stoku y [m] (0 = Start)")
        ax_slope.set_title("Stok Kasprowy Wierch (Model Rozszerzony)", fontweight="bold")
        ax_slope.grid(True, linestyle=":", alpha=0.4)

        ax_speed.set_title("Prędkość chwilowa narciarzy [km/h]", fontweight="bold")
        ax_speed.set_xlabel("Czas [s]")
        ax_speed.set_ylabel("Prędkość [km/h]")
        ax_speed.set_xlim(0, max_frames + 1)
        ax_speed.set_ylim(-2, 70)
        ax_speed.grid(True, linestyle=":", alpha=0.4)

        ax_turn.set_title("Wskaźnik szerokości skrętu (dx / dy)", fontweight="bold")
        ax_turn.set_xlabel("Czas [s]")
        ax_turn.set_ylabel("dx / dy")
        ax_turn.set_xlim(0, max_frames + 1)
        ax_turn.set_ylim(-0.05, 1.5)
        ax_turn.grid(True, linestyle=":", alpha=0.4)

        colors = plt.cm.tab10.colors

        trail_lines = []
        head_dots = []
        finish_badges = []
        speed_lines = []
        ratio_lines = []

        finish_offsets_x = np.linspace(8, terrain.width - 8, total_skiers)

        for i, skier in enumerate(self.model.agents):
            c = colors[i % len(colors)]
            lbl = f"#{i} {skier.style[:4]} ({skier.skill_level[:3]})"
            (line,) = ax_slope.plot([], [], color=c, linewidth=1.8, alpha=0.85, zorder=5)
            (dot,) = ax_slope.plot([], [], marker="o", markersize=7, color=c, markeredgecolor="black", markeredgewidth=0.5, zorder=7)
            (badge,) = ax_slope.plot([], [], marker="X", markersize=8, color=c, markeredgecolor="black", markeredgewidth=1.0, zorder=8)
            (s_line,) = ax_speed.plot([], [], color=c, linewidth=1.6, alpha=0.9, label=lbl)
            (r_line,) = ax_turn.plot([], [], color=c, linewidth=1.4, alpha=0.85, label=lbl)

            trail_lines.append(line)
            head_dots.append(dot)
            finish_badges.append(badge)
            speed_lines.append(s_line)
            ratio_lines.append(r_line)

        if total_skiers <= 8:
            ax_speed.legend(loc="upper right", fontsize=7.0, framealpha=0.85)
            ax_turn.legend(loc="upper right", fontsize=7.0, framealpha=0.85)

        status_box = ax_slope.text(
            0.03,
            0.97,
            "",
            transform=ax_slope.transAxes,
            fontsize=8.5,
            fontfamily="monospace",
            va="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="dimgray", alpha=0.92),
            zorder=10,
        )

        celebration_banner = ax_slope.text(
            0.50,
            0.50,
            "",
            transform=ax_slope.transAxes,
            fontsize=13,
            fontweight="bold",
            ha="center",
            va="center",
            visible=False,
            zorder=20,
        )

        pause_frames = 20
        total_anim_frames = max_frames + pause_frames

        # Prekalkulacja ewolucji stygmergii stoku klatka po klatce
        stygmergy_frames: list[np.ndarray] = []
        temp_snow = np.ones((terrain.length, terrain.width), dtype=float)
        temp_ice = np.zeros((terrain.length, terrain.width), dtype=float)
        temp_moguls = np.zeros((terrain.length, terrain.width), dtype=float)

        # Inicjalizacja z przeszkodami/cechami
        for ip in terrain.ice_patches:
            cx, cy, rx, ry = ip["cx"], ip["cy"], ip["rx"], ip["ry"]
            for y in range(max(0, int(cy - ry)), min(terrain.length, int(cy + ry + 1))):
                for x in range(max(0, int(cx - rx)), min(terrain.width, int(cx + rx + 1))):
                    if ((x - cx) / rx)**2 + ((y - cy) / ry)**2 <= 1.0:
                        temp_ice[y, x] = max(temp_ice[y, x], ip["intensity"])

        for f in range(max_frames):
            for skier in self.model.agents:
                if f < len(skier.trajectory) - 1:
                    ox, oy = skier.trajectory[f]
                    nx, ny = skier.trajectory[f + 1]
                    v_val = skier.speed_history[f]
                    dx = abs(nx - ox)
                    dy = max(1, abs(ny - oy))
                    ratio = float(dx) / float(dy)
                    scrape = (v_val / 15.0) * (0.05 + 0.15 * ratio)

                    for cy in range(min(oy, ny), max(oy, ny) + 1):
                        for cx in range(min(ox, nx), max(ox, nx) + 1):
                            if 0 <= cx < terrain.width and 0 <= cy < terrain.length:
                                temp_snow[cy, cx] = max(0.0, temp_snow[cy, cx] - 0.05 * scrape)
                                if temp_snow[cy, cx] < 0.65:
                                    temp_ice[cy, cx] = min(1.0, temp_ice[cy, cx] + 0.15 * ((0.65 - temp_snow[cy, cx]) / 0.65))

                    if dx > 0:
                        dep_off = 2 if nx < ox else -2
                        dep_x = int(np.clip(nx + dep_off, 0, terrain.width - 1))
                        dep_y = int(np.clip(ny + 1, 0, terrain.length - 1))
                        temp_moguls[dep_y, dep_x] = min(1.0, temp_moguls[dep_y, dep_x] + 0.10 * scrape)

            # Tworzymy mapę RGB
            rgb = np.ones((terrain.length, terrain.width, 3), dtype=float)
            ice_3d = temp_ice[:, :, np.newaxis]
            mog_3d = temp_moguls[:, :, np.newaxis]
            rgb = rgb * (1.0 - 0.7 * ice_3d) + np.array([0.0, 0.6, 0.95]) * (0.7 * ice_3d)
            rgb = rgb * (1.0 - 0.6 * mog_3d) + np.array([0.85, 0.45, 0.05]) * (0.6 * mog_3d)
            stygmergy_frames.append(rgb.copy())

        def update(frame_idx: int):
            is_celebration = frame_idx >= max_frames
            frame = min(frame_idx, max_frames - 1)

            # Aktualizacja podkładu stygmergii stoku
            surface_img.set_data(stygmergy_frames[frame])

            finished_count = 0
            for i, skier in enumerate(self.model.agents):
                traj = skier.trajectory
                t_fin = len(traj) - 1
                if frame >= t_fin:
                    finished_count += 1

            status_lines = [
                f"CZAS: {frame:2d}s | META: {finished_count}/{total_skiers}",
                f"KOLIZJE: {self.model.total_collisions} | WYPADKI: {self.model.total_crashes}",
                "-" * 24,
            ]
            if total_skiers <= 6:
                for i, skier in enumerate(self.model.agents):
                    traj = skier.trajectory
                    t_fin = len(traj) - 1
                    if frame >= t_fin:
                        status_lines.append(f"#{i} {skier.style[:4]}: META ({t_fin}s)")
                    else:
                        spd_ms = skier.speed_history[frame]
                        status_lines.append(f"#{i} {skier.style[:4]}: {spd_ms*3.6:4.1f}km/h")
            status_box.set_text("\n".join(status_lines))

            for i, skier in enumerate(self.model.agents):
                traj = skier.trajectory
                speeds = skier.speed_history
                t_fin = len(traj) - 1

                cur_len = min(frame + 1, len(traj))
                pts = np.array(traj[:cur_len])

                trail_lines[i].set_data(pts[:, 0], pts[:, 1])

                if frame >= t_fin:
                    fin_x = finish_offsets_x[i]
                    fin_y = finish_y + 4.5
                    head_dots[i].set_data([fin_x], [fin_y])
                    finish_badges[i].set_data([pts[-1, 0]], [pts[-1, 1]])

                    active_t = np.arange(len(speeds))
                    active_spd = np.array(speeds) * 3.6
                    extra_len = frame - t_fin
                    if extra_len > 0:
                        post_t = np.arange(t_fin + 1, frame + 1)
                        post_spd = np.zeros(extra_len)
                        full_t = np.concatenate([active_t, post_t])
                        full_spd = np.concatenate([active_spd, post_spd])
                    else:
                        full_t = active_t
                        full_spd = active_spd
                    speed_lines[i].set_data(full_t, full_spd)
                else:
                    head_dots[i].set_data([pts[-1, 0]], [pts[-1, 1]])
                    finish_badges[i].set_data([], [])

                    t_axis = np.arange(cur_len)
                    spd_kmh = np.array(speeds[:cur_len]) * 3.6
                    speed_lines[i].set_data(t_axis, spd_kmh)

                if cur_len > 1:
                    dxs = np.abs(np.diff(pts[:, 0]))
                    dys = np.maximum(1, np.diff(pts[:, 1]))
                    ratios = dxs / dys
                    ratio_lines[i].set_data(np.arange(1, cur_len), ratios)

            if is_celebration:
                celeb_step = frame_idx - max_frames + 1
                blink_color = "#d90429" if (celeb_step // 2) % 2 == 0 else "#2b9348"
                celebration_banner.set_visible(True)
                celebration_banner.set_text(
                    f"=== SYMULACJA ZAKOŃCZONA ===\n"
                    f"Zjazdy ukończone, wygenerowano muldy i lód\n"
                    f"Powtórka za: {((pause_frames - celeb_step) / fps):.1f}s"
                )
                celebration_banner.set_bbox(
                    dict(boxstyle="round,pad=0.6", fc="white", ec=blink_color, linewidth=2.5, alpha=0.96)
                )
            else:
                celebration_banner.set_visible(False)

            return [surface_img] + trail_lines + head_dots + finish_badges + speed_lines + ratio_lines + [status_box, celebration_banner]

        anim = FuncAnimation(fig, update, frames=total_anim_frames, blit=True, interval=1000 // fps)
        anim.save(output_path, writer="pillow", fps=fps)
        plt.close(fig)
        return output_path
