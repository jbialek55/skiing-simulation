"""Implementacja agenta narciarza Skier (Kamiński 2026 - z rozszerzeniami).

Zawiera:
- Wybór stylu jazdy (pług, carving, śmig, standard, snowboard),
- Różnicowanie poziomów doświadczenia (początkujący, średniozaawansowany, ekspert),
- Dynamiczny wykładnik sterowalności beta(v_self, skill) - 'przeciążanie' maski pędem,
- Konsekwencje kolizji i wypadków (stan awaryjny: upadek / postój na stoku i wtórny zator),
- Interakcję z nawierzchnią (lód: utrata krawędzi k_h i wektor uślizgu; muldy: dodatkowy opór śniegu),
- Dynamiczną stygmergię stoku: zeskrobywanie podłoża i formowanie muld po łuku.
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
SkiStyle = Literal["plow", "standard", "short_turns", "carving", "snowboard"]


class Skier(Agent):
    """Agent reprezentujący narciarza lub snowboardzistę na stoku."""

    def __init__(
        self,
        model: "SkiModel",
        pos: Tuple[int, int],
        skill_level: SkillLevel = "intermediate",
        style: SkiStyle = "standard",
        p_zw: float = 0.20,
    ) -> None:
        super().__init__(model=model)
        self.skill_level = skill_level
        self.style = style
        self.p_zw = p_zw

        # Konfiguracja parametrów w zależności od poziomu i stylu jazdy
        self._configure_profile()

        self.v_max = self.v_max_p
        self.is_finished = False

        # Stan awaryjny (wypadek / upadek)
        self.crash_recovery_timer: int = 0
        self.crashes_count: int = 0

        # Statystyki interakcji z terenem
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

    def _configure_profile(self) -> None:
        """Ustala parametry kinematyki i sterowalności na podstawie poziomu i stylu."""
        # 1. Bazowa prędkość maksymalna wg poziomu
        if self.skill_level == "beginner":
            self.v_max_p = 10.0
            self.v = 5.0
            self.sigma_base = 2.0         # mniejsza precyzja, większy rozrzut
            self.ice_aversion = 0.90      # wysoka awersja do lodu
            self.mogul_aversion = 0.85
        elif self.skill_level == "expert":
            self.v_max_p = 30.0
            self.v = 15.0
            self.sigma_base = 1.2         # wysoka precyzja prowadzenia
            self.ice_aversion = 0.35      # mniejsza panika na lodzie
            self.mogul_aversion = 0.25
        else:  # intermediate
            self.v_max_p = 20.0
            self.v = 10.0
            self.sigma_base = 1.5
            self.ice_aversion = 0.60
            self.mogul_aversion = 0.50

        # 2. Modyfikatory stylu jazdy
        if self.style == "plow":
            self.k_d = 0.08
            self.k_h = 1.25               # bardzo silne hamowanie pługiem
            self.v_max_p = min(self.v_max_p, 9.0)
            self.v = min(self.v, 5.0)
        elif self.style == "carving":
            self.k_d = 0.12
            self.k_h = 0.45               # cięty łuk carvingowy mniej wytraca prędkość
            self.v_max_p = max(self.v_max_p, 24.0)
        elif self.style == "short_turns":  # śmig
            self.k_d = 0.10
            self.k_h = 0.95               # dynamiczne wytracanie prędkości śmigiem
        elif self.style == "snowboard":
            self.k_d = 0.11
            self.k_h = 0.75
            self.sigma_base += 0.3
        else:  # standard
            self.k_d = 0.10
            self.k_h = 0.80

    def trigger_crash(self, recovery_time: int = 8) -> None:
        """Wprowadza agenta w stan awaryjny po kolizji (upadek na stoku)."""
        self.crash_recovery_timer = recovery_time
        self.crashes_count += 1
        self.v = 0.0                      # natychmiastowe zatrzymanie

    def step(self) -> None:
        """Faza 1 kroku: podejmowanie decyzji i losowanie nowej pozycji z uwzględnieniem pędu."""
        if self.is_finished:
            return

        x, y = self.pos
        if y >= self.model.terrain.length - 1:
            self.is_finished = True
            return

        # Jeśli agent leży po upadku, dochodzi do siebie i nie wykonuje ruchu
        if self.crash_recovery_timer > 0:
            self._next_pos = (x, y)
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
                self.turn_length = sample_turn_length(
                    self.v, self.v_max, rng=rng, style_preference=self.style
                )
                if self.turn_length == 1:
                    self.current_direction = "down"

            self.s_y = sample_scale_sy(self.v, self.v_max, rng=rng)

            # W strefie muld narciarze preferują krótsze skręty
            current_moguls = self.model.terrain.get_mogul_roughness(x, y)
            if current_moguls > 0.40 and self.turn_length > 3 and self.style != "plow":
                self.turn_length = 3
                self.s_y = min(self.s_y, 0.75)

            self.turn_stage = 1
            if self.current_direction in ("left", "right"):
                self._prev_turn = self.current_direction

        # 2. Wyznaczenie sąsiedztwa (2v+1) x (v+1)
        v_int = max(1, int(round(self.v)))
        w = 2 * v_int + 1
        h = v_int + 1

        # 3. Asymetria snowboardu (frontside vs backside)
        asym = 0.0
        if self.style == "snowboard":
            asym = 0.08 if self.current_direction == "right" else -0.08

        # 4. Parametry rozkładu bazowego
        mu_x, mu_y = get_turn_mus(
            turn_type=self.current_direction,
            length=self.turn_length,
            step_idx=self.turn_stage,
            w=w,
            h=h,
            s_y=self.s_y,
            asymmetry=asym,
        )

        # Odczyt warunków w obecnym miejscu: uślizg na lodzie
        current_ice = self.model.terrain.get_ice_intensity(x, y)
        if current_ice > 0.30:
            # Uślizg: rozkład jest 'ściągany' w dół wzdłuż spadku, utrudniając skręt
            mu_x = (1.0 - 0.4 * current_ice) * mu_x + (0.4 * current_ice) * ((w - 1) / 2.0)
            mu_y = min(float(h - 1), mu_y + 1.2 * current_ice * h)

        left_b, right_b = self.model.terrain.get_borders_at(y)
        slope_w = max(1, right_b - left_b)
        rel_pos = float(x - left_b) / float(slope_w)
        gamma_x = compute_gamma_x(rel_pos=rel_pos, gamma_max=5.0)

        # Rozkład bazowy B
        sigma_cur = self.sigma_base
        if current_ice > 0.3:
            sigma_cur += 0.5  # na lodzie tracimy kontrolę, rośnie wariancja

        B = compute_base_distribution(
            w=w,
            h=h,
            mu_x=mu_x,
            mu_y=mu_y,
            gamma_x=gamma_x,
            gamma_y=5.0,
            sigma_x=sigma_cur,
            sigma_y=sigma_cur,
            s_y=self.s_y,
        )

        # 5. Maski środowiska i anizotropowej kolizji
        M_env = self.model.terrain.extract_env_mask(
            x, y, v_int, ice_aversion=self.ice_aversion, mogul_aversion=self.mogul_aversion
        )
        M_coll = self.model.extract_anisotropic_collision_mask(x, y, v_int, self)

        # 6. Wyznaczenie wykładników sterowalności beta(v_self, skill)
        # Przy dużej prędkości beta -> 0, pęd narciarza przeciąża chęć ominięcia!
        ratio = self.v / max(1.0, self.v_max_p)
        skill_factor = 1.4 if self.skill_level == "expert" else (1.0 if self.skill_level == "intermediate" else 0.7)

        # beta_coll: wykładnik dla kolizji
        beta_coll = float(np.clip(skill_factor * (1.2 - 0.9 * ratio), 0.15, 1.5))
        # beta_env: wykładnik dla terenu (nawierzchni)
        beta_env = float(np.clip(skill_factor * (1.1 - 0.8 * ratio), 0.20, 1.3))

        # 7. Iloczyn Hadamarda z 'przeciążaniem' i losowanie
        dy, dx = sample_next_position(
            base_dist=B,
            env_mask=M_env,
            collision_mask=M_coll,
            beta_coll=beta_coll,
            beta_env=beta_env,
            rng=rng,
        )

        if dy == 0:
            dy = 1

        new_x = int(np.clip(x - v_int + dx, 0, self.model.terrain.width - 1))
        new_y = int(min(self.model.terrain.length - 1, y + dy))

        self._next_pos = (new_x, new_y)

    def advance(self) -> None:
        """Faza 2 kroku: faktyczne wykonanie przemieszczenia, stygmergia i aktualizacja prędkości."""
        if self.is_finished or self._next_pos is None:
            return

        old_x, old_y = self.pos
        new_x, new_y = self._next_pos

        # Obsługa powrotu do sprawności po upadku
        if self.crash_recovery_timer > 0:
            self.crash_recovery_timer -= 1
            self.speed_history.append(0.0)
            if self.crash_recovery_timer == 0:
                self.v = 2.0  # powolne ruszanie po wypadku
            return

        # Przemieszczenie na siatce
        self.model.grid.move_agent(self, (new_x, new_y))
        self.pos = (new_x, new_y)
        self.trajectory.append(self.pos)

        dx = abs(new_x - old_x)
        dy = max(1, new_y - old_y)
        old_h = self.model.terrain.get_altitude(old_x, old_y)
        new_h = self.model.terrain.get_altitude(new_x, new_y)
        dh = max(0.0, old_h - new_h)

        # Dynamiczna stygmergia: narciarz ściera stok i odkłada muldy
        self.model.terrain.record_skier_action(
            old_x=old_x,
            old_y=old_y,
            new_x=new_x,
            new_y=new_y,
            v=self.v,
            turn_type=self.current_direction,
        )

        # 1. Odczyt właściwości terenu w nowej pozycji
        ice_intensity = self.model.terrain.get_ice_intensity(new_x, new_y)
        mogul_roughness = self.model.terrain.get_mogul_roughness(new_x, new_y)

        if ice_intensity > 0.20:
            self.time_on_ice += 1
        if mogul_roughness > 0.20:
            self.time_on_moguls += 1

        # 2. Efektywne hamowanie krawędzią (na lodzie drastyczny spadek przyczepności!)
        k_h_eff = self.k_h * (1.0 - 0.80 * ice_intensity)

        # 3. Dodatkowy opór głębokiego śniegu w muldach
        snow_drag = 1.8 * mogul_roughness

        # Aktualizacja prędkości - Równanie 4.2
        v_new = self.v + self.k_d * (dh / float(dy)) - k_h_eff * (float(dx) / float(dy)) - snow_drag

        # 4. Ryzyko uślizgu krawędzi na lodzie
        if ice_intensity > 0.35 and self.model.rng.random() < (0.35 * ice_intensity):
            self.ice_slips_count += 1
            v_new += 1.8  # przyspieszenie ześlizgowe (brak możliwości zahamowania)

        # 5. Losowe zwolnienie p_zw (Nagel-Schreckenberg)
        if self.model.rng.random() < self.p_zw:
            v_new = max(1.0, v_new - 1.0)

        # 6. Rozpędzanie przy małych prędkościach
        if v_new < 0.5 * self.v_max and mogul_roughness < 0.30:
            v_new += 1.0

        # 7. Aktualizacja v_max
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
        length_m = float(np.sqrt(self._current_turn_dx**2 + self._current_turn_dy**2))
        x, y = self.pos
        inclination = self.model.terrain.get_slope_inclination(x, y)

        self.turns_metrics.append({
            "direction": self.current_direction,
            "style": self.style,
            "skill_level": self.skill_level,
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
