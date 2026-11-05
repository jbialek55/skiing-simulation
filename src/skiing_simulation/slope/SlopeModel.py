from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from mesa import Model
from mesa.space import ContinuousSpace

from skiing_simulation.slope.terrain import SlopeTerrain
from skiing_simulation.agents.base_skier import BaseSkier


class SkiModel(Model):
    """
    Model symulacji stoku narciarskiego w bibliotece Mesa, w 100% zgodny z:
    Thomas Holleczek & Gerhard Tröster,
    "Particle-based model for skiing traffic", Phys. Rev. E 85, 056101 (2012).
    """

    def __init__(
        self,
        slope_name: str = "jochstock",  # 'jochstock' (Table I: 889m) lub 'graustock' (1250m)
        dt: float = 0.01,               # krok integracji 10 ms (Sec. IV.A)
        seed: Optional[int] = 42,
    ):
        super().__init__(rng=seed)
        self.dt = dt
        self.simulation_time = 0.0
        self.step_count = 0

        # Środowisko terenu (DEM wysokości, nachylenie, krawędzie, tarcie)
        self.terrain = SlopeTerrain(slope_name=slope_name, resolution=1.0)
        self.width = self.terrain.width
        self.height = self.terrain.length

        # Przestrzeń ciągła dla narciarzy
        self.space = ContinuousSpace(x_max=self.width, y_max=self.height, torus=False)

    def add_skier(
        self,
        x: Optional[float] = None,
        y: float = 5.0,
        mass: float = 85.0,
        sidecut_radius: float = 10.0,
        friction_coeff: float = 0.10,
        threshold_delta_deg: float = 10.0,
        custom_waypoints: Optional[List[Tuple[float, float]]] = None,
    ) -> BaseSkier:
        """Dodaje narciarza zgodnego z modelem cząsteczkowym Holleczek & Tröster (2012)."""
        if x is None:
            left, right = self.terrain.get_borders_at(y)
            x = (left + right) / 2.0

        skier = BaseSkier(
            model=self,
            pos=(x, y),
            mass=mass,
            sidecut_radius=sidecut_radius,
            friction_coeff=friction_coeff,
            threshold_delta_deg=threshold_delta_deg,
            custom_waypoints=custom_waypoints,
        )
        self.space.place_agent(skier, (x, y))
        skier._record_telemetry(self.terrain.get_friction(x, y))
        return skier

    def step(self) -> None:
        """Krok symulacji dla wszystkich agentów."""
        self.simulation_time += self.dt
        self.step_count += 1
        self.agents.shuffle_do("step")

    def get_summary_stats(self) -> Dict[str, Any]:
        """Zwraca statystyki aktualnego stanu symulacji."""
        skiers = [a for a in self.agents if isinstance(a, BaseSkier)]
        active = sum(1 for s in skiers if s.status == "active")
        finished = sum(1 for s in skiers if s.status == "finished")
        avg_speed = np.mean([s.speed_kmh for s in skiers]) if skiers else 0.0

        return {
            "time_s": self.simulation_time,
            "step": self.step_count,
            "total_skiers": len(skiers),
            "active": active,
            "finished": finished,
            "avg_speed_kmh": float(avg_speed),
        }
