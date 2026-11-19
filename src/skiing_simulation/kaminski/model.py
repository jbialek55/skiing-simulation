"""Klasa SkiModel dla modelu stochastycznego narciarza (Kamiński 2026).

Obsługuje:
- Środowisko symulacji oparte na MultiGrid,
- Wypuszczanie narciarzy w partiach (np. batch_size co batch_interval kroków),
- Anizotropowe stożkowe strefy unikania kolizji,
- Detekcję kolizji: bezpośrednich (współdzielenie komórki) oraz geometrycznych (przecięcie trajektorii w danym kroku),
- Detekcję zderzeń ze słupami wyciągu / przeszkodami stałymi (natychmiastowy wypadek),
- Gromadzenie danych telemetrycznych.
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from mesa import Model
from mesa.space import MultiGrid

from skiing_simulation.kaminski.terrain import SlopeGrid25D
from skiing_simulation.kaminski.agent import Skier, SkillLevel, SkiStyle


class SkiModel(Model):
    """Główny model symulacji wieloagentowej stoku narciarskiego."""

    def __init__(
        self,
        width: int = 40,
        length: int = 350,
        buffer_zone_r: int = 3,
        use_kasprowy_profile: bool = True,
        enable_terrain_features: bool = True,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__(rng=seed)
        self.rng = np.random.default_rng(seed)
        self.terrain = SlopeGrid25D(
            width=width,
            length=length,
            buffer_zone_r=buffer_zone_r,
            use_kasprowy_profile=use_kasprowy_profile,
        )
        if enable_terrain_features:
            self.terrain.setup_default_features()

        self.grid = MultiGrid(width=width, height=length, torus=False)
        self.simulation_step = 0
        self.total_collisions = 0
        self.total_crashes = 0
        self.total_obstacle_hits = 0

        # Harmonogram wypuszczania partii (batches)
        self.pending_skiers_queue: List[Dict[str, Any]] = []
        self.batch_interval: int = 15

        # Mapy rejestracji zdarzeń
        self.collision_heatmap = np.zeros((length, width), dtype=int)
        self.crash_locations: List[Tuple[int, int]] = []

    def add_skier(
        self,
        x: Optional[int] = None,
        y: int = 2,
        skill_level: Optional[SkillLevel] = None,
        style: Optional[SkiStyle] = None,
        p_zw: float = 0.20,
    ) -> Skier:
        """Dodaje nowego narciarza o zadanym stylu i poziomie zaawansowania."""
        left, right = self.terrain.get_borders_at(y)
        if x is None:
            margin = self.terrain.r + 1
            start_left = min(left + margin, right - 1)
            start_right = max(right - margin, start_left + 1)
            x = int(self.rng.integers(start_left, start_right))

        if skill_level is None:
            choices: List[SkillLevel] = ["beginner", "intermediate", "expert"]
            skill_level = self.rng.choice(choices, p=[0.25, 0.50, 0.25])

        if style is None:
            if skill_level == "beginner":
                style = self.rng.choice(["plow", "snowboard"], p=[0.70, 0.30])
            elif skill_level == "expert":
                style = self.rng.choice(["carving", "short_turns", "snowboard"], p=[0.45, 0.40, 0.15])
            else:
                style = self.rng.choice(["standard", "short_turns", "carving", "snowboard"], p=[0.40, 0.25, 0.20, 0.15])

        skier = Skier(
            model=self,
            pos=(x, y),
            skill_level=skill_level,
            style=style,
            p_zw=p_zw,
        )
        self.grid.place_agent(skier, (x, y))
        return skier

    def populate_batches(
        self,
        total_skiers: int = 24,
        batch_size: int = 6,
        batch_interval: int = 15,
    ) -> None:
        """Kolejkuje wypuszczanie narciarzy w seriach/partiach."""
        self.batch_interval = batch_interval
        self.pending_skiers_queue = []

        # Wypuść pierwszą partię natychmiast na starcie
        first_batch = min(total_skiers, batch_size)
        for i in range(first_batch):
            start_y = (i * 3) % 10 + 2
            self.add_skier(y=start_y)

        # Resztę zakolejkuj
        remaining = total_skiers - first_batch
        batch_idx = 1
        while remaining > 0:
            cur_b = min(remaining, batch_size)
            release_step = batch_idx * batch_interval
            for i in range(cur_b):
                self.pending_skiers_queue.append({
                    "release_step": release_step,
                    "y": (i * 3) % 10 + 2,
                })
            remaining -= cur_b
            batch_idx += 1

    def populate_skiers(
        self,
        count: int = 20,
        stagger_y: bool = True,
    ) -> None:
        """Inicjalizuje narciarzy w partiach (domyślnie po 6 co 15 kroków)."""
        if count <= 8:
            for i in range(count):
                start_y = (i * 5) % 25 + 2 if stagger_y else 2
                self.add_skier(y=start_y)
        else:
            self.populate_batches(total_skiers=count, batch_size=6, batch_interval=15)

    def extract_anisotropic_collision_mask(
        self,
        agent_x: int,
        agent_y: int,
        v: int,
        calling_agent: Skier,
    ) -> np.ndarray:
        """Anizotropowa (eliptyczna/stożkowa) maska unikania kolizji."""
        w = 2 * v + 1
        h = v + 1
        mask = np.ones((h, w), dtype=float)

        x_start = agent_x - v
        y_start = agent_y

        for other in self.agents:
            if other is calling_agent or other.is_finished:
                continue

            ox, oy = other.pos
            ov = max(1.0, other.v)

            if other.crash_recovery_timer > 0:
                for r in range(h):
                    gy = y_start + r
                    for c in range(w):
                        gx = x_start + c
                        dist = np.hypot(gx - ox, gy - oy)
                        if dist < 3.0:
                            mask[r, c] *= (dist / 3.0)
                continue

            if (agent_x - v - ov * 1.5 <= ox <= agent_x + v + ov * 1.5) and (
                agent_y - ov <= oy <= agent_y + v + ov * 2.0
            ):
                turn_dir = other.current_direction
                # Wektor kierunkowy: lewo to -x, prawo to +x
                vx_other = -0.5 * ov if turn_dir == "left" else (0.5 * ov if turn_dir == "right" else 0.0)
                vy_other = ov

                v_len = np.hypot(vx_other, vy_other)
                dir_x = vx_other / v_len
                dir_y = vy_other / v_len

                perp_x = -dir_y
                perp_y = dir_x

                a_long = 1.3 * ov
                b_lat = 0.5 * ov + 1.0

                for r in range(h):
                    gy = y_start + r
                    for c in range(w):
                        gx = x_start + c
                        dx_rel = gx - ox
                        dy_rel = gy - oy

                        proj_long = dx_rel * dir_x + dy_rel * dir_y
                        proj_lat = dx_rel * perp_x + dy_rel * perp_y

                        if proj_long >= -1.0:
                            norm_dist_sq = (proj_long / a_long) ** 2 + (proj_lat / b_lat) ** 2
                            if norm_dist_sq < 1.0:
                                factor = np.sqrt(norm_dist_sq)
                                mask[r, c] *= np.clip(factor, 0.0, 1.0)

        return mask

    def _check_trajectory_intersection(
        self, p1: Tuple[int, int], p2: Tuple[int, int], q1: Tuple[int, int], q2: Tuple[int, int]
    ) -> bool:
        """Sprawdza, czy odcinki przemieszczenia dwóch narciarzy p1->p2 i q1->q2 przecinają się."""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))

    def step(self) -> None:
        """Krok symulacji dla wszystkich narciarzy (staged activation)."""
        self.simulation_step += 1

        # 1. Wypuszczanie zakolejkowanych narciarzy z nowej partii
        if self.pending_skiers_queue:
            to_release = [s for s in self.pending_skiers_queue if s["release_step"] <= self.simulation_step]
            for s_info in to_release:
                self.add_skier(y=s_info["y"])
                self.pending_skiers_queue.remove(s_info)

        active_skiers = [a for a in self.agents if not a.is_finished]
        if not active_skiers:
            return

        # Faza 1: Obliczenie intencji i wylosowanie pozycji
        for skier in active_skiers:
            skier.step()

        # Zapamiętanie starych pozycji do testu przecięć
        old_positions = {skier: skier.pos for skier in active_skiers}

        # Faza 2: Przesunięcie, stygmergia stoku i aktualizacja prędkości
        for skier in active_skiers:
            skier.advance()

        # Faza 3: Detekcja kolizji oraz kolizji z przeszkodami stałymi (słupami)
        # A. Kolizje z przeszkodami stałymi (słupy wyciągu)
        for skier in active_skiers:
            if skier.is_finished or skier.crash_recovery_timer > 0 or skier.v < 1.0:
                continue
            sx, sy = skier.pos
            for obs in self.terrain.obstacles:
                dist = np.hypot(sx - obs["cx"], sy - obs["cy"])
                if dist <= obs["radius"] + 0.3:
                    # Uderzenie w słup wyciągu!
                    self.total_obstacle_hits += 1
                    self.total_crashes += 1
                    self.crash_locations.append((sx, sy))
                    skier.trigger_crash(recovery_time=int(self.rng.integers(8, 14)))
                    # Bezpieczne odsunięcie leżącego narciarza na skraj słupa
                    angle = np.arctan2(sy - obs["cy"], sx - obs["cx"])
                    if dist < 0.1:
                        angle = 0.0 if sx >= obs["cx"] else np.pi
                    safe_dist = obs["radius"] + 1.2
                    nx = int(np.clip(round(obs["cx"] + safe_dist * np.cos(angle)), 1, self.terrain.width - 2))
                    ny = int(np.clip(round(obs["cy"] + safe_dist * np.sin(angle)), 1, self.terrain.length - 2))
                    self.grid.move_agent(skier, (nx, ny))
                    skier.pos = (nx, ny)
                    break

        # B. Kolizje między narciarzami: współdzielenie komórki LUB bliski dystans (<1.4m)
        n_active = len(active_skiers)
        for i in range(n_active):
            s1 = active_skiers[i]
            if s1.is_finished:
                continue
            for j in range(i + 1, n_active):
                s2 = active_skiers[j]
                if s2.is_finished:
                    continue

                # Jeśli obaj leżą po wypadku lub obaj stoją w miejscu, nie dochodzi do nowej kolizji
                if s1.crash_recovery_timer > 0 and s2.crash_recovery_timer > 0:
                    continue
                if s1.v < 1.2 and s2.v < 1.2:
                    continue

                p1_old = old_positions[s1]
                p1_new = s1.pos
                p2_old = old_positions[s2]
                p2_new = s2.pos

                dist_now = np.hypot(p1_new[0] - p2_new[0], p1_new[1] - p2_new[1])
                is_intersecting = self._check_trajectory_intersection(p1_old, p1_new, p2_old, p2_new)

                if dist_now <= 1.4 or is_intersecting:
                    self.total_collisions += 1
                    cx = (p1_new[0] + p2_new[0]) // 2
                    cy = (p1_new[1] + p2_new[1]) // 2
                    self.collision_heatmap[cy, cx] += 1

                    # Jeśli prędkość była znaczna, dochodzi do wypadku
                    max_v = max(s1.v, s2.v)
                    if max_v > 5.5:
                        if s1.crash_recovery_timer == 0 and s1.v > 1.2:
                            s1.trigger_crash(recovery_time=int(self.rng.integers(6, 11)))
                        if s2.crash_recovery_timer == 0 and s2.v > 1.2:
                            s2.trigger_crash(recovery_time=int(self.rng.integers(6, 11)))
                        self.total_crashes += 1
                        self.crash_locations.append((cx, cy))

    @property
    def all_finished(self) -> bool:
        """Zwraca True, jeśli wszyscy narciarze ukończyli zjazd i kolejka partii jest pusta."""
        return (
            len(self.pending_skiers_queue) == 0
            and len(self.agents) > 0
            and all(a.is_finished for a in self.agents)
        )

    def run_simulation(self, max_steps: Optional[int] = None) -> None:
        """Uruchamia symulację aż wszyscy narciarze dojadą do końca stoku."""
        while not self.all_finished:
            if max_steps is not None and self.simulation_step >= max_steps:
                break
            self.step()

    def get_summary_stats(self) -> Dict[str, Any]:
        """Zwraca szczegółowe statystyki symulacji."""
        all_speeds = []
        all_turn_metrics = []

        for a in self.agents:
            all_speeds.extend(a.speed_history)
            all_turn_metrics.extend(a.turns_metrics)

        avg_speed = float(np.mean(all_speeds)) if all_speeds else 0.0
        max_speed = float(np.max(all_speeds)) if all_speeds else 0.0

        total_time_ice = sum(a.time_on_ice for a in self.agents)
        total_time_moguls = sum(a.time_on_moguls for a in self.agents)
        total_slips = sum(a.ice_slips_count for a in self.agents)
        total_agent_crashes = sum(a.crashes_count for a in self.agents)

        return {
            "total_skiers": len(self.agents),
            "finished_skiers": sum(1 for a in self.agents if a.is_finished),
            "total_steps": self.simulation_step,
            "total_collisions": self.total_collisions,
            "total_crashes": self.total_crashes,
            "total_obstacle_hits": self.total_obstacle_hits,
            "total_agent_crashes": total_agent_crashes,
            "average_speed_ms": avg_speed,
            "average_speed_kmh": avg_speed * 3.6,
            "max_speed_kmh": max_speed * 3.6,
            "total_turns_recorded": len(all_turn_metrics),
            "total_time_ice": total_time_ice,
            "total_time_moguls": total_time_moguls,
            "total_ice_slips": total_slips,
        }
