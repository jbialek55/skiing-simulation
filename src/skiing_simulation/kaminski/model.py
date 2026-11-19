"""Klasa SkiModel dla modelu stochastycznego narciarza (Kamiński 2026).

Obsługuje środowisko symulacji oparte na MultiGrid, cykl życia agentów,
aktualizację masek kolizyjnych oraz gromadzenie telemetrii i statystyk zderzeń.
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from mesa import Model
from mesa.space import MultiGrid

from skiing_simulation.kaminski.terrain import SlopeGrid25D
from skiing_simulation.kaminski.agent import Skier, SkillLevel


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

        # Mapa cieplna kolizji (zliczanie zderzeń w każdej komórce stoku)
        self.collision_heatmap = np.zeros((length, width), dtype=int)
        self.finished_skiers: List[Skier] = []

    def add_skier(
        self,
        x: Optional[int] = None,
        y: int = 2,
        skill_level: Optional[SkillLevel] = None,
        p_zw: float = 0.20,
        k_d: float = 0.10,
        k_h: float = 0.80,
    ) -> Skier:
        """Dodaje nowego narciarza na stoku."""
        left, right = self.terrain.get_borders_at(y)
        if x is None:
            # Losowa pozycja startowa na szerokości trasy z buforem
            margin = self.terrain.r + 1
            start_left = min(left + margin, right - 1)
            start_right = max(right - margin, start_left + 1)
            x = int(self.rng.integers(start_left, start_right))

        if skill_level is None:
            # Domyślny rozkład poziomów: 20% początkujący, 60% średni, 20% eksperci
            choices: List[SkillLevel] = ["beginner", "intermediate", "expert"]
            skill_level = self.rng.choice(choices, p=[0.20, 0.60, 0.20])

        skier = Skier(
            model=self,
            pos=(x, y),
            skill_level=skill_level,
            p_zw=p_zw,
            k_d=k_d,
            k_h=k_h,
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
            start_y = (i * 8) % 40 + 2 if stagger_y else 2
            self.add_skier(y=start_y)

    def extract_collision_mask(
        self,
        agent_x: int,
        agent_y: int,
        v: int,
        calling_agent: Skier,
    ) -> np.ndarray:
        """Tworzy maskę unikania kolizji dla sąsiedztwa (2v+1) x (v+1) - Równanie 4.0.

        Wokół każdego innego aktywnego narciarza k o prędkości v_k,
        prawdopodobieństwo wjazdu spada proporcjonalnie do d / v_k dla d < v_k.
        """
        w = 2 * v + 1
        h = v + 1
        mask = np.ones((h, w), dtype=float)

        x_start = agent_x - v
        y_start = agent_y

        # Szukamy innych aktywnych narciarzy w strefie wzroku
        for other in self.agents:
            if other is calling_agent or other.is_finished:
                continue

            ox, oy = other.pos
            ov = max(1.0, other.v)

            # Szybki test odległości prostokątnej (bounding box)
            if (agent_x - v - ov <= ox <= agent_x + v + ov) and (
                agent_y - ov <= oy <= agent_y + v + ov
            ):
                # Obliczenie odległości na siatce sąsiedztwa
                for r in range(h):
                    gy = y_start + r
                    for c in range(w):
                        gx = x_start + c
                        dist = np.hypot(gx - ox, gy - oy)
                        if dist < ov:
                            factor = dist / ov
                            mask[r, c] *= factor

        return mask

    def step(self) -> None:
        """Krok symulacji dla wszystkich narciarzy (staged activation: step -> advance)."""
        self.simulation_step += 1

        active_skiers = [a for a in self.agents if not a.is_finished]
        if not active_skiers:
            return

        # Faza 1: Obliczenie intencji i wylosowanie pozycji
        for skier in active_skiers:
            skier.step()

        # Faza 2: Przesunięcie i aktualizacja prędkości
        for skier in active_skiers:
            skier.advance()

        # Detekcja i rejestracja kolizji (komórki z >1 narciarzem)
        for skier in active_skiers:
            if not skier.is_finished:
                cell_contents = self.grid.get_cell_list_contents([skier.pos])
                if len(cell_contents) > 1:
                    self.total_collisions += 1
                    cx, cy = skier.pos
                    self.collision_heatmap[cy, cx] += 1

    @property
    def all_finished(self) -> bool:
        """Zwraca True, jeśli wszyscy narciarze ukończyli zjazd (dotarli do końca stoku)."""
        return len(self.agents) > 0 and all(a.is_finished for a in self.agents)

    def run_simulation(self, max_steps: Optional[int] = None) -> None:
        """Uruchamia symulację aż wszyscy narciarze dojadą do końca stoku.

        Opcjonalny parametr max_steps stanowi jedynie górny limit bezpieczeństwa.
        """
        while not self.all_finished:
            if max_steps is not None and self.simulation_step >= max_steps:
                break
            self.step()

    def get_summary_stats(self) -> Dict[str, Any]:
        """Zwraca podsumowanie statystyczne przejazdów."""
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

        return {
            "total_skiers": len(self.agents),
            "finished_skiers": sum(1 for a in self.agents if a.is_finished),
            "total_steps": self.simulation_step,
            "total_collisions": self.total_collisions,
            "average_speed_ms": avg_speed,
            "average_speed_kmh": avg_speed * 3.6,
            "max_speed_kmh": max_speed * 3.6,
            "total_turns_recorded": len(all_turn_metrics),
            "total_time_ice": total_time_ice,
            "total_time_moguls": total_time_moguls,
            "total_ice_slips": total_slips,
        }
