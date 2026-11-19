"""Klasa SkiModel dla modelu stochastycznego narciarza (Kamiński 2026).

Obsługuje:
- Środowisko symulacji oparte na MultiGrid,
- Anizotropowe (eliptyczne/stożkowe) rzutowanie strefy ryzyka kolizji wzdłuż wektora prędkości narciarza,
- Detekcję kolizji, zderzeń i przecięć trajektorii z wprowadzeniem stanu awaryjnego (upadek/zator),
- Dynamiczną stygmergię stoku (zużycie podłoża),
- Gromadzenie zaawansowanej telemetrii.
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

        # Mapa cieplna kolizji
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

    def populate_skiers(
        self,
        count: int = 20,
        stagger_y: bool = True,
    ) -> None:
        """Inicjalizuje grupę narciarzy na stoku."""
        for i in range(count):
            start_y = (i * 7) % 35 + 2 if stagger_y else 2
            self.add_skier(y=start_y)

    def extract_anisotropic_collision_mask(
        self,
        agent_x: int,
        agent_y: int,
        v: int,
        calling_agent: Skier,
    ) -> np.ndarray:
        """Anizotropowa (eliptyczna/stożkowa) maska kolizji.

        Zamiast symetrycznego koła, każdy inny narciarz rzutuje strefę ryzyka w kierunku
        swojego aktualnego wektora ruchu. Obszar ZA nim nie blokuje innych.
        """
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

            # Jeśli narciarz ma upadek/wypadek, tworzy nieruchomą strefę kolizyjną
            if other.crash_recovery_timer > 0:
                for r in range(h):
                    gy = y_start + r
                    for c in range(w):
                        gx = x_start + c
                        dist = np.hypot(gx - ox, gy - oy)
                        if dist < 3.0:
                            mask[r, c] *= (dist / 3.0)
                continue

            # Sprawdzenie bounding box
            if (agent_x - v - ov * 1.5 <= ox <= agent_x + v + ov * 1.5) and (
                agent_y - ov <= oy <= agent_y + v + ov * 2.0
            ):
                # Kierunkowy wektor innego narciarza
                turn_dir = other.current_direction
                vx_other = -0.5 * ov if turn_dir == "left" else (0.5 * ov if turn_dir == "right" else 0.0)
                vy_other = ov

                v_len = np.hypot(vx_other, vy_other)
                dir_x = vx_other / v_len
                dir_y = vy_other / v_len

                # Oś prostopadła
                perp_x = -dir_y
                perp_y = dir_x

                a_long = 1.3 * ov      # zasięg wzdłużny przed narciarzem
                b_lat = 0.5 * ov + 1.0  # zasięg poprzeczny

                for r in range(h):
                    gy = y_start + r
                    for c in range(w):
                        gx = x_start + c

                        # Wektor od innego narciarza do rozpatrywanego punktu
                        dx_rel = gx - ox
                        dy_rel = gy - oy

                        proj_long = dx_rel * dir_x + dy_rel * dir_y
                        proj_lat = dx_rel * perp_x + dy_rel * perp_y

                        # Strefa ryzyka tylko z przodu i z boków (za narciarzem proj_long < -1.0 nie grozi)
                        if proj_long >= -1.0:
                            # Eliptyczna odległość znormalizowana
                            norm_dist_sq = (proj_long / a_long) ** 2 + (proj_lat / b_lat) ** 2
                            if norm_dist_sq < 1.0:
                                factor = np.sqrt(norm_dist_sq)
                                mask[r, c] *= np.clip(factor, 0.0, 1.0)

        return mask

    def step(self) -> None:
        """Krok symulacji dla wszystkich narciarzy (staged activation: step -> advance)."""
        self.simulation_step += 1

        active_skiers = [a for a in self.agents if not a.is_finished]
        if not active_skiers:
            return

        # Faza 1: Obliczenie intencji i wylosowanie pozycji (z przeciążaniem pędem)
        for skier in active_skiers:
            skier.step()

        # Faza 2: Przesunięcie, stygmergia stoku i aktualizacja prędkości
        for skier in active_skiers:
            skier.advance()

        # Faza 3: Detekcja kolizji oraz konsekwencji wypadków
        # 1. Sprawdzenie współdzielenia komórek
        occupied_cells: Dict[Tuple[int, int], List[Skier]] = {}
        for skier in active_skiers:
            if not skier.is_finished:
                pos = skier.pos
                occupied_cells.setdefault(pos, []).append(skier)

        for pos, skiers_in_cell in occupied_cells.items():
            if len(skiers_in_cell) > 1:
                self.total_collisions += 1
                cx, cy = pos
                self.collision_heatmap[cy, cx] += 1

                # Jeśli prędkość względna była znaczna, dochodzi do wypadku/upadku
                speeds = [s.v for s in skiers_in_cell]
                if max(speeds) > 6.0:
                    for s in skiers_in_cell:
                        s.trigger_crash(recovery_time=self.rng.integers(6, 12))
                    self.total_crashes += 1
                    self.crash_locations.append(pos)

    @property
    def all_finished(self) -> bool:
        """Zwraca True, jeśli wszyscy narciarze ukończyli zjazd."""
        return len(self.agents) > 0 and all(a.is_finished for a in self.agents)

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
            "total_agent_crashes": total_agent_crashes,
            "average_speed_ms": avg_speed,
            "average_speed_kmh": avg_speed * 3.6,
            "max_speed_kmh": max_speed * 3.6,
            "total_turns_recorded": len(all_turn_metrics),
            "total_time_ice": total_time_ice,
            "total_time_moguls": total_time_moguls,
            "total_ice_slips": total_slips,
        }
