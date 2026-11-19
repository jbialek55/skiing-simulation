"""Implementacja agenta narciarza Skier (Kamiński 2026).

Klasa Skier dziedziczy po mesa.Agent i realizuje dwufazowy krok symulacji (step + advance):
- Wybór zakrętu, długości i parametrów rozkładu (step),
- Przemnożenie rozkładu przez maski terenu i kolizji (Iloczyn Hadamarda),
- Przemieszczenie na nową pozycję (advance) oraz aktualizacja prędkości
  zgodnie ze wzorem fizycznym v = v + k_d*(dh/dy) - k_h*(dx/dy) i modelem Nagela-Schreckenberga.
"""

from __future__ import annotations

from typing import Tuple, Optional, Literal, List, Dict, Any, TYPE_CHECKING
import numpy as np
from mesa import Agent

from skiing_simulation.kaminski.distributions import (
    TurnDirection,
    get_turn_mus,
    compute_gamma_x,
    sample_turn_direction,
    sample_turn_length,
    sample_scale_sy,
    compute_base_distribution,
    sample_next_position,
)

if TYPE_CHECKING:
    from skiing_simulation.kaminski.model import SkiModel

SkillLevel = Literal["beginner", "intermediate", "expert"]


class Skier(Agent):
    """Agent reprezentujący jednego narciarza na stoku."""

    def __init__(
        self,
        model: "SkiModel",
        pos: Tuple[int, int],
        skill_level: SkillLevel = "intermediate",
        p_zw: float = 0.20,
        k_d: float = 0.10,
        k_h: float = 0.80,
    ) -> None:
        super().__init__(model=model)
        self.skill_level = skill_level
        self.p_zw = p_zw
        self.k_d = k_d
        self.k_h = k_h

        # Początkowa prędkość maksymalna v_max_p wg poziomu umiejętności (Równanie 4.3)
        if skill_level == "beginner":
            self.v_max_p = 10.0
            self.v = 5.0
        elif skill_level == "expert":
            self.v_max_p = 30.0
            self.v = 15.0
        else:  # intermediate
            self.v_max_p = 20.0
            self.v = 10.0

        self.v_max = self.v_max_p
        self.is_finished = False

        # Świadomość i omijanie lodu zależne od poziomu umiejętności
        if skill_level == "expert":
            self.ice_avoidance_weight = 0.85
        elif skill_level == "intermediate":
            self.ice_avoidance_weight = 0.45
        else:  # beginner
            self.ice_avoidance_weight = 0.0

        # Statystyki interakcji z terenem (lód i muldy)
        self.time_on_ice: int = 0
        self.time_on_moguls: int = 0
        self.ice_slips_count: int = 0

        # Stan manewru
        self.current_direction: TurnDirection = "down"
        self.turn_length: int = 1
        self.turn_stage: int = 1
        self.s_y: float = 1.0

        # Zmienna pomocnicza do fazy advance
        self._next_pos: Optional[Tuple[int, int]] = None
        self._prev_turn: Optional[TurnDirection] = None

        # Historia telemetryczna
        self.trajectory: List[Tuple[int, int]] = [pos]
        self.speed_history: List[float] = [self.v]
        self.turn_stages_history: List[Tuple[str, int, int]] = []
        self.turns_metrics: List[Dict[str, Any]] = []

        # Akumulatory dla bieżącego zakrętu
        self._current_turn_speeds: List[float] = [self.v]
        self._current_turn_dx: int = 0
        self._current_turn_dy: int = 0

    def step(self) -> None:
        """Faza 1 kroku: podejmowanie decyzji i wylosowanie nowej pozycji."""
        if self.is_finished:
            return

        x, y = self.pos
        if y >= self.model.terrain.length - 1:
            self.is_finished = True
            return

        rng = self.model.rng

        # 1. Sprawdzenie, czy rozpoczynamy nowy manewr
        if self.turn_stage > self.turn_length:
            self._finalize_turn_metric()
            left_b, right_b = self.model.terrain.get_borders_at(y)
            free_left = x - left_b
            free_right = right_b - x

            self.current_direction = sample_turn_direction(
                prev_turn=self._prev_turn,
                free_left=free_left,
                free_right=free_right,
                rng=rng,
            )

            if self.current_direction == "down":
                self.turn_length = 1
            else:
                self.turn_length = sample_turn_length(self.v, self.v_max, rng=rng)
                if self.turn_length == 1:
                    self.current_direction = "down"

            self.s_y = sample_scale_sy(self.v, self.v_max, rng=rng)

            # W strefie muld wymuszamy krótki skręt (t_l <= 3) i mniejsze tempo wzdłużne
            current_moguls = self.model.terrain.get_mogul_roughness(x, y)
            if current_moguls > 0.35 and self.turn_length > 3:
                self.turn_length = 3
                self.s_y = min(self.s_y, 0.75)

            self.turn_stage = 1
            if self.current_direction in ("left", "right"):
                self._prev_turn = self.current_direction

        # 2. Wyznaczenie sąsiedztwa (2v+1) x (v+1)
        v_int = max(1, int(round(self.v)))
        w = 2 * v_int + 1
        h = v_int + 1

        # 3. Parametry rozkładu bazowego
        mu_x, mu_y = get_turn_mus(
            turn_type=self.current_direction,
            length=self.turn_length,
            step_idx=self.turn_stage,
            w=w,
            h=h,
            s_y=self.s_y,
        )

        left_b, right_b = self.model.terrain.get_borders_at(y)
        slope_w = max(1, right_b - left_b)
        rel_pos = float(x - left_b) / float(slope_w)
        gamma_x = compute_gamma_x(rel_pos=rel_pos, gamma_max=5.0)

        # Rozkład bazowy B
        B = compute_base_distribution(
            w=w,
            h=h,
            mu_x=mu_x,
            mu_y=mu_y,
            gamma_x=gamma_x,
            gamma_y=5.0,
            sigma_x=1.5,
            sigma_y=1.5,
            s_y=self.s_y,
        )

        # 4. Maski środowiska (z uwzględnieniem lodu) i kolizji
        M_env = self.model.terrain.extract_env_mask(
            x, y, v_int, ice_avoidance_weight=self.ice_avoidance_weight
        )
        M_coll = self.model.extract_collision_mask(x, y, v_int, self)

        # 5. Iloczyn Hadamarda i losowanie nowego przemieszczenia (dy, dx)
        dy, dx = sample_next_position(B, M_env, M_coll, rng=rng)

        # Jeśli dy == 0, wymuszamy minimalny ruch w dół o 1 komórkę
        if dy == 0:
            dy = 1

        new_x = int(np.clip(x - v_int + dx, 0, self.model.terrain.width - 1))
        new_y = int(min(self.model.terrain.length - 1, y + dy))

        self._next_pos = (new_x, new_y)

    def advance(self) -> None:
        """Faza 2 kroku: faktyczne wykonanie przemieszczenia i aktualizacja prędkości."""
        if self.is_finished or self._next_pos is None:
            return

        old_x, old_y = self.pos
        new_x, new_y = self._next_pos

        # Przemieszczenie na siatce
        self.model.grid.move_agent(self, (new_x, new_y))
        self.pos = (new_x, new_y)
        self.trajectory.append(self.pos)

        dx = abs(new_x - old_x)
        dy = max(1, new_y - old_y)
        old_h = self.model.terrain.get_altitude(old_x, old_y)
        new_h = self.model.terrain.get_altitude(new_x, new_y)
        dh = max(0.0, old_h - new_h)

        # 1. Odczyt właściwości terenu w nowej pozycji (lód i muldy)
        ice_intensity = self.model.terrain.get_ice_intensity(new_x, new_y)
        mogul_roughness = self.model.terrain.get_mogul_roughness(new_x, new_y)

        if ice_intensity > 0.20:
            self.time_on_ice += 1
        if mogul_roughness > 0.20:
            self.time_on_moguls += 1

        # 2. Efektywne hamowanie krawędzią (na lodzie drastyczny spadek przyczepności!)
        k_h_eff = self.k_h * (1.0 - 0.75 * ice_intensity)

        # 3. Dodatkowy opór głębokiego śniegu w polu muld
        snow_drag = 1.6 * mogul_roughness

        # Aktualizacja prędkości - Równanie 4.2 zmodyfikowane o właściwości podłoża
        # v_new = v_{t-1} + k_d*(dh/dy) - k_h_eff*(dx/dy) - snow_drag
        v_new = self.v + self.k_d * (dh / float(dy)) - k_h_eff * (float(dx) / float(dy)) - snow_drag

        # 4. Ryzyko poślizgu na lodzie (utrata krawędzi)
        if ice_intensity > 0.35 and self.model.rng.random() < (0.28 * ice_intensity):
            self.ice_slips_count += 1
            v_new += 1.5  # przyspieszenie poślizgowe z powodu niemożności wyhamowania

        # 5. Losowe zwolnienie p_zw (Nagel-Schreckenberg)
        if self.model.rng.random() < self.p_zw:
            v_new = max(1.0, v_new - 1.0)

        # 6. Przyspieszanie przy małych prędkościach (rozpędzanie ze startu na gładkim stoku)
        if v_new < 0.5 * self.v_max and mogul_roughness < 0.30:
            v_new += 1.0

        # 7. Aktualizacja v_max na podstawie nachylenia stoku i muld
        local_inclination = self.model.terrain.get_slope_inclination(new_x, new_y)
        self._update_v_max(local_inclination, mogul_roughness)

        # Ograniczenie prędkości
        self.v = float(np.clip(v_new, 1.0, self.v_max))
        self.speed_history.append(self.v)

        # Akumulacja danych zakrętu
        self._current_turn_speeds.append(self.v)
        self._current_turn_dx += dx
        self._current_turn_dy += dy
        self.turn_stages_history.append((self.current_direction, self.turn_length, self.turn_stage))

        self.turn_stage += 1
        if new_y >= self.model.terrain.length - 1:
            self.is_finished = True
            self._finalize_turn_metric()

    def _update_v_max(self, inclination: float, mogul_roughness: float = 0.0) -> None:
        """Wyznacza dopuszczalną prędkość maksymalną w zależności od nachylenia i muld."""
        if inclination < 0.35:
            factor = 1.0
        elif inclination < 0.45:
            factor = 0.90
        elif inclination < 0.50:
            factor = 0.80
        else:
            factor = 0.70
        base_vmax = factor * self.v_max_p

        # W strefie muld prędkość maksymalna jest drastycznie zdławiona amortyzacją nóg
        if mogul_roughness > 0.20:
            mogul_cap = 3.5 + 3.0 * (1.0 - mogul_roughness)
            self.v_max = min(base_vmax, mogul_cap)
        else:
            self.v_max = base_vmax

    def _finalize_turn_metric(self) -> None:
        """Zapisuje statystyki zakończonego zakrętu."""
        if not self._current_turn_speeds:
            return

        avg_speed = float(np.mean(self._current_turn_speeds))
        dy = max(1, self._current_turn_dy)
        dx_dy_ratio = float(self._current_turn_dx) / float(dy)
        length_m = np.sqrt(self._current_turn_dx**2 + self._current_turn_dy**2)
        x, y = self.pos
        inclination = self.model.terrain.get_slope_inclination(x, y)

        self.turns_metrics.append({
            "direction": self.current_direction,
            "length_steps": self.turn_length,
            "length_meters": length_m,
            "avg_speed": avg_speed,
            "dx_dy_ratio": dx_dy_ratio,
            "slope_inclination": inclination,
            "end_y": y,
        })

        self._current_turn_speeds = []
        self._current_turn_dx = 0
        self._current_turn_dy = 0
