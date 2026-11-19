"""Narzędzia wizualizacji i analityki dla modelu Kamińskiego (2026).

Generuje zestaw wykresów walidacyjnych i diagnostycznych:
- Trajektorie narciarzy na tle granic i stref buforowych stoku (Rys. 6.2, 6.12),
- Mapę kolizji i stref podwyższonego ryzyka (Heatmapa),
- Zależność prędkości od nachylenia stoku (Rys. 6.13),
- Wskaźnik proporcji zakrętu dx/dy w funkcji stromości (Rys. 6.14).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

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
        """Tworzy 4-panelowy panel diagnostyczny weryfikujący trajektorie i statystyki ruchu."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axs = plt.subplots(1, 4, figsize=(20, 8), gridspec_kw={"width_ratios": [1.2, 1.2, 1.5, 1.5]})
        plt.subplots_adjust(wspace=0.35)

        # ----------------------------------------------------
        # Panel 1: Trajektorie na stoku (Rysunek 6.12 z doktoratu)
        # ----------------------------------------------------
        ax1 = axs[0]
        terrain = self.model.terrain
        y_vals = np.arange(terrain.length)
        left_borders = [terrain.get_borders_at(y)[0] for y in y_vals]
        right_borders = [terrain.get_borders_at(y)[1] for y in y_vals]

        # Rysowanie granic i stref wyłączonych
        ax1.plot(left_borders, y_vals, color="black", linewidth=2.0, label="Granica trasy")
        ax1.plot(right_borders, y_vals, color="black", linewidth=2.0)
        ax1.fill_betweenx(y_vals, 0, left_borders, color="gainsboro", alpha=0.8)
        ax1.fill_betweenx(y_vals, right_borders, terrain.width, color="gainsboro", alpha=0.8)

        # Rysowanie pól muld i płatów lodu (Opcja A)
        from matplotlib.patches import Ellipse, Rectangle
        for mf in terrain.mogul_fields:
            rect = Rectangle(
                (mf["x_min"], mf["y_min"]),
                mf["x_max"] - mf["x_min"],
                mf["y_max"] - mf["y_min"],
                color="#ffeaa7",
                alpha=0.65,
                zorder=2,
            )
            ax1.add_patch(rect)
            ax1.text(
                (mf["x_min"] + mf["x_max"]) / 2.0,
                (mf["y_min"] + mf["y_max"]) / 2.0,
                "POLE MULD",
                color="#d35400",
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=3,
            )

        for ip in terrain.ice_patches:
            ellipse = Ellipse(
                (ip["cx"], ip["cy"]),
                width=2 * ip["rx"],
                height=2 * ip["ry"],
                color="#00b4d8",
                alpha=0.45,
                zorder=3,
            )
            ax1.add_patch(ellipse)
            ax1.text(
                ip["cx"],
                ip["cy"],
                "LÓD",
                color="#03045e",
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=4,
            )

        # Wyraźna linia mety i strefa wybiegu
        ax1.axhline(terrain.length - 1, color="crimson", linestyle="-", linewidth=2.5, label="Linia mety (350m)")
        ax1.fill_between([0, terrain.width], terrain.length - 1, terrain.length + 5, color="lightpink", alpha=0.3)
        ax1.text(terrain.width / 2.0, terrain.length - 2, "=== LINIA METY ===", color="crimson", fontweight="bold", ha="center", fontsize=9)

        # Trajektorie
        colors = plt.cm.tab10.colors
        for i, skier in enumerate(self.model.agents):
            if len(skier.trajectory) > 1:
                traj = np.array(skier.trajectory)
                c = colors[i % len(colors)]
                t_finish = len(skier.trajectory)
                lbl = f"#{i} {skier.skill_level} ({t_finish}s)" if len(self.model.agents) <= 8 else None
                ax1.plot(traj[:, 0], traj[:, 1], color=c, alpha=0.85, linewidth=1.8, label=lbl)
                # punkt startu i końca
                ax1.scatter(traj[0, 0], traj[0, 1], color=c, s=30, zorder=5)
                # Rozróżnienie punktów mety - mały pionowy/poziomy rozrzut etykiet
                end_x, end_y = traj[-1, 0], traj[-1, 1]
                ax1.scatter(end_x, end_y, marker="X", color=c, edgecolor="black", s=65, zorder=6)
                if len(self.model.agents) <= 10:
                    offset_x = 1.2 if i % 2 == 0 else -1.2
                    ax1.text(end_x + offset_x, end_y + 3, f"#{i}", color=c, fontweight="bold", fontsize=8, ha="center")

        ax1.set_xlim(0, terrain.width)
        ax1.set_ylim(terrain.length + 6, 0)  # y rośnie w dół
        ax1.set_xlabel("Szerokość x [m]")
        ax1.set_ylabel("Długość stoku y [m]")
        ax1.set_title("Trajektorie narciarzy\n(Wszyscy ukończyli zjazd na dole)")
        if len(self.model.agents) <= 8:
            ax1.legend(loc="upper right", fontsize=8, framealpha=0.85)
        ax1.grid(True, linestyle=":", alpha=0.5)

        # ----------------------------------------------------
        # Panel 2: Mapa kolizji i stref ryzyka (Heatmapa)
        # ----------------------------------------------------
        ax2 = axs[1]
        im = ax2.imshow(
            self.model.collision_heatmap,
            cmap="YlOrRd",
            aspect="auto",
            extent=[0, terrain.width, terrain.length, 0],
        )
        ax2.plot(left_borders, y_vals, color="black", linewidth=1.5, linestyle="--")
        ax2.plot(right_borders, y_vals, color="black", linewidth=1.5, linestyle="--")
        plt.colorbar(im, ax=ax2, label="Liczba kolizji")
        ax2.set_xlabel("Szerokość x [m]")
        ax2.set_ylabel("Długość stoku y [m]")
        ax2.set_title(f"Mapa kolizji na stoku\n(Łącznie: {self.model.total_collisions})")

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
            ax3.scatter(slopes, speeds, alpha=0.6, edgecolors="none", s=25, c="dodgerblue")

            # Średnia linia trendu
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
            ax4.scatter(slopes, dx_dy, alpha=0.6, edgecolors="none", s=25, c="darkorange")
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
        """Tworzy animowany plik GIF prezentujący zjazd narciarzy krok po kroku."""
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

        # Rysowanie tła stoku
        y_vals = np.arange(terrain.length)
        left_borders = [terrain.get_borders_at(y)[0] for y in y_vals]
        right_borders = [terrain.get_borders_at(y)[1] for y in y_vals]

        ax_slope.plot(left_borders, y_vals, color="black", linewidth=2.0)
        ax_slope.plot(right_borders, y_vals, color="black", linewidth=2.0)
        ax_slope.fill_betweenx(y_vals, 0, left_borders, color="#e0e0e0", alpha=0.9)
        ax_slope.fill_betweenx(y_vals, right_borders, terrain.width, color="#e0e0e0", alpha=0.9)

        # Rysowanie pól muld i płatów lodu na animacji
        from matplotlib.patches import Ellipse, Rectangle
        for mf in terrain.mogul_fields:
            rect = Rectangle(
                (mf["x_min"], mf["y_min"]),
                mf["x_max"] - mf["x_min"],
                mf["y_max"] - mf["y_min"],
                color="#ffeaa7",
                alpha=0.65,
                zorder=2,
            )
            ax_slope.add_patch(rect)
            ax_slope.text(
                (mf["x_min"] + mf["x_max"]) / 2.0,
                (mf["y_min"] + mf["y_max"]) / 2.0,
                "POLE MULD",
                color="#d35400",
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=3,
            )

        for ip in terrain.ice_patches:
            ellipse = Ellipse(
                (ip["cx"], ip["cy"]),
                width=2 * ip["rx"],
                height=2 * ip["ry"],
                color="#00b4d8",
                alpha=0.45,
                zorder=3,
            )
            ax_slope.add_patch(ellipse)
            ax_slope.text(
                ip["cx"],
                ip["cy"],
                "LÓD",
                color="#03045e",
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=4,
            )

        # Wyraźna linia mety i strefa wybiegu za metą (350m - 364m)
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
        ax_slope.text(
            terrain.width / 2.0,
            finish_y + 10.5,
            "STREFA WYBIEGU / META",
            color="#2d6a4f",
            fontweight="bold",
            ha="center",
            fontsize=8,
            zorder=3,
        )

        ax_slope.set_xlim(-1, terrain.width + 1)
        ax_slope.set_ylim(terrain.length + 14, -5)
        ax_slope.set_xlabel("Szerokość x [m]")
        ax_slope.set_ylabel("Długość stoku y [m] (0 = Start)")
        ax_slope.set_title("Stok Kasprowy Wierch (Model Kamiński 2026)", fontweight="bold")
        ax_slope.grid(True, linestyle=":", alpha=0.4)

        ax_speed.set_title("Prędkość chwilowa narciarzy [km/h]", fontweight="bold")
        ax_speed.set_xlabel("Czas [s]")
        ax_speed.set_ylabel("Prędkość [km/h]")
        ax_speed.set_xlim(0, max_frames + 1)
        ax_speed.set_ylim(-2, 70)
        ax_speed.axhline(0, color="gray", linestyle=":", linewidth=1)
        ax_speed.grid(True, linestyle=":", alpha=0.4)

        ax_turn.set_title("Wskaźnik szerokości skrętu (dx / dy)", fontweight="bold")
        ax_turn.set_xlabel("Czas [s]")
        ax_turn.set_ylabel("dx / dy")
        ax_turn.set_xlim(0, max_frames + 1)
        ax_turn.set_ylim(-0.05, 1.5)
        ax_turn.grid(True, linestyle=":", alpha=0.4)

        colors = plt.cm.tab10.colors

        # Dynamiczne linie i punkty
        trail_lines = []
        head_dots = []
        speed_lines = []
        ratio_lines = []
        finish_badges = []

        # Obliczenie rozłożenia narciarzy w strefie mety, aby nie nachodzili na siebie
        finish_offsets_x = np.linspace(8, terrain.width - 8, total_skiers)

        for i, skier in enumerate(self.model.agents):
            c = colors[i % len(colors)]
            lbl = f"#{i} ({skier.skill_level})"
            (line,) = ax_slope.plot([], [], color=c, linewidth=1.8, alpha=0.85)
            (dot,) = ax_slope.plot([], [], marker="o", markersize=7, color=c, markeredgecolor="black", markeredgewidth=0.5, zorder=6)
            (badge,) = ax_slope.plot([], [], marker="X", markersize=9, color=c, markeredgecolor="black", markeredgewidth=1.0, zorder=7)
            (s_line,) = ax_speed.plot([], [], color=c, linewidth=1.6, alpha=0.9, label=lbl)
            (r_line,) = ax_turn.plot([], [], color=c, linewidth=1.4, alpha=0.85, label=lbl)

            trail_lines.append(line)
            head_dots.append(dot)
            badge.set_data([], [])
            finish_badges.append(badge)
            speed_lines.append(s_line)
            ratio_lines.append(r_line)

        if total_skiers <= 8:
            ax_speed.legend(loc="upper right", fontsize=7.5, framealpha=0.85)
            ax_turn.legend(loc="upper right", fontsize=7.5, framealpha=0.85)

        # Pływający baner ze stanem wyścigu
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

        # Baner celebracji na mecie (migający na końcu)
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

        # Liczba klatek pauzy na mecie (24 klatki przy 8 fps = 3.0 sekundy)
        pause_frames = 24
        total_anim_frames = max_frames + pause_frames

        def update(frame_idx: int):
            is_celebration = frame_idx >= max_frames
            frame = min(frame_idx, max_frames - 1)

            # Zliczanie narciarzy na mecie w danej chwili
            finished_skiers_info = []
            finished_count = 0

            for i, skier in enumerate(self.model.agents):
                traj = skier.trajectory
                t_fin = len(traj) - 1
                if frame >= t_fin:
                    finished_count += 1
                    finished_skiers_info.append((i, skier.skill_level, t_fin, True))
                else:
                    spd_ms = skier.speed_history[frame]
                    cur_pt = traj[frame]
                    is_ice = terrain.get_ice_intensity(cur_pt[0], cur_pt[1]) > 0.25
                    is_mog = terrain.get_mogul_roughness(cur_pt[0], cur_pt[1]) > 0.25
                    tag = " [LOD]" if is_ice else (" [MULD]" if is_mog else "")
                    finished_skiers_info.append((i, skier.skill_level, spd_ms * 3.6, False, tag))

            # Aktualizacja tekstu statusu
            status_lines = [
                f"CZAS: {frame:2d}s | META: {finished_count}/{total_skiers}",
                "-" * 24,
            ]
            if total_skiers <= 6:
                for item in finished_skiers_info:
                    if item[3]:  # is_fin
                        i, skill, val, _ = item
                        status_lines.append(f"#{i} {skill[:5]:<5}: META ({val}s)")
                    else:
                        i, skill, val, _, tag = item
                        status_lines.append(f"#{i} {skill[:5]:<5}: {val:4.1f}km/h{tag}")
            status_box.set_text("\n".join(status_lines))

            # Aktualizacja trajektorii i telemetrii dla każdego narciarza
            for i, skier in enumerate(self.model.agents):
                traj = skier.trajectory
                speeds = skier.speed_history
                t_fin = len(traj) - 1

                cur_len = min(frame + 1, len(traj))
                pts = np.array(traj[:cur_len])

                trail_lines[i].set_data(pts[:, 0], pts[:, 1])

                if frame >= t_fin:
                    # Narciarz jest na mecie: rozkładamy punkty w strefie wybiegu
                    fin_x = finish_offsets_x[i]
                    fin_y = finish_y + 4.5
                    head_dots[i].set_data([fin_x], [fin_y])
                    finish_badges[i].set_data([pts[-1, 0]], [pts[-1, 1]])

                    # Telemetria prędkości: ciągła linia, po mecie opada do 0 km/h
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

            # Obsługa unikalnych klatek świętowania (zapobiega deduplikacji w Pillow)
            if is_celebration:
                celeb_step = frame_idx - max_frames + 1
                blink_color = "#d90429" if (celeb_step // 2) % 2 == 0 else "#2b9348"
                celebration_banner.set_visible(True)
                celebration_banner.set_text(
                    f"=== WSZYSCY NA MECIE ({total_skiers}/{total_skiers}) ===\n"
                    f"Koniec zjazdu po {max_frames-1}s\n"
                    f"Powtórka za: {((pause_frames - celeb_step) / fps):.1f}s"
                )
                celebration_banner.set_bbox(
                    dict(boxstyle="round,pad=0.6", fc="white", ec=blink_color, linewidth=2.5, alpha=0.96)
                )
            else:
                celebration_banner.set_visible(False)

            return trail_lines + head_dots + finish_badges + speed_lines + ratio_lines + [status_box, celebration_banner]

        anim = FuncAnimation(fig, update, frames=total_anim_frames, blit=True, interval=1000 // fps)
        anim.save(output_path, writer="pillow", fps=fps)
        plt.close(fig)
        return output_path
