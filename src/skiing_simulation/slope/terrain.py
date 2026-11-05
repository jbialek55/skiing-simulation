import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy.interpolate import PchipInterpolator


@dataclass
class Obstacle:
    """Reprezentacja przeszkody na stoku wg Holleczek & Tröster (2012)."""
    x: float
    y: float
    radius: float
    obstacle_type: str = "tree"  # 'tree', 'pylon', 'snow_cannon'


class SlopeTerrain:
    """
    Model terenu stoku narciarskiego 2.5D oparty na siatce wysokości Z(x, y).
    Odwzorowuje stoki ze szwajcarskiego ośrodka Engelberg (Jochstock oraz Graustock)
    opisane w artykule:
    Thomas Holleczek & Gerhard Tröster,
    "Particle-based model for skiing traffic", Phys. Rev. E 85, 056101 (2012).
    """

    def __init__(
        self,
        slope_name: str = "jochstock",  # 'jochstock' (Table I: 889m, intermediate) lub 'graustock' (1250m, expert)
        resolution: float = 1.0,         # rozdzielczość siatki DEM w metrach (Table I / Sec. IV.B)
    ):
        self.slope_name = slope_name.lower()
        self.resolution = resolution

        if self.slope_name == "graustock":
            # Parametry stoku Graustock z Table I (Holleczek & Tröster 2012)
            self.length = 1250.0
            self.width = 81.0
            self.start_altitude = 2178.0
            self.drop = 413.0
            self.finish_altitude = self.start_altitude - self.drop  # 1765.0 m
            self.avg_inclination = 0.33
        else:
            # Parametry stoku Jochstock z Table I (Holleczek & Tröster 2012)
            self.slope_name = "jochstock"
            self.length = 889.0
            self.width = 62.0
            self.start_altitude = 2504.0
            self.drop = 288.0
            self.finish_altitude = self.start_altitude - self.drop  # 2216.0 m
            self.avg_inclination = 0.32

        self.nx = int(np.ceil(self.width / resolution))
        self.ny = int(np.ceil(self.length / resolution))

        self.x_coords = np.linspace(0, self.width, self.nx)
        self.y_coords = np.linspace(0, self.length, self.ny)
        self.X, self.Y = np.meshgrid(self.x_coords, self.y_coords, indexing="ij")

        # Inicjalizacja macierzy
        self.elevation = np.zeros((self.nx, self.ny), dtype=float)
        self.grad_x = np.zeros((self.nx, self.ny), dtype=float)
        self.grad_y = np.zeros((self.nx, self.ny), dtype=float)
        self.slope_angle = np.zeros((self.nx, self.ny), dtype=float)

        # Współczynnik tarcia kinetycznego nart o śnieg (Table IV: mu = 0.10)
        self.default_snow_friction = 0.10
        self.friction_map = np.full((self.nx, self.ny), self.default_snow_friction, dtype=float)

        # Krawędzie stoku (Table I: zmienna szerokość)
        self.left_border = np.zeros(self.ny, dtype=float)
        self.right_border = np.zeros(self.ny, dtype=float)

        # Przeszkody
        self.obstacles: List[Obstacle] = []

        self._generate_borders()
        self._generate_elevation_profile()
        self._calculate_gradients()

    def _generate_borders(self) -> None:
        """
        Krawędzie trasy:
        Dla Jochstock: wąski początek (ok. 38m), poszerzenie w części środkowej (do ok. 74m),
        następnie średnio ok. 60m przy mecie (średnia szerokość 62m z Table I).
        Dla Graustock: początek ok. 50m, poszerzenie do ok. 95m, średnia 81m z Table I.
        """
        norm_y = self.y_coords / self.length
        x_center = self.width / 2.0

        if self.slope_name == "graustock":
            # Start wąski (~50m), rozszerza się do ~95m
            w = 50.0 + (95.0 - 50.0) * np.clip(norm_y / 0.35, 0.0, 1.0)
        else:
            # Jochstock: wąski start (~38m), rozszerzenie w rejonie 0.2 - 0.7 do ~74m, meta ~60m
            w = np.zeros_like(norm_y)
            for i, ny in enumerate(norm_y):
                if ny < 0.20:
                    w[i] = 38.0 + (68.0 - 38.0) * (ny / 0.20)
                elif ny < 0.70:
                    w[i] = 68.0 + 6.0 * np.sin(np.pi * (ny - 0.20) / 0.50)
                else:
                    w[i] = 60.0

        half_w = w / 2.0
        self.left_border = np.maximum(x_center - half_w, 1.0)
        self.right_border = np.minimum(x_center + half_w, self.width - 1.0)

    def _generate_elevation_profile(self) -> None:
        """
        Generuje cyfrowy model wysokości (DEM) odpowiadający stokom Jochstock lub Graustock
        ze szwajcarskiego rejestru swisstopo (Sec. III.A, IV.B).
        """
        if self.slope_name == "graustock":
            # Graustock: start 2178m, meta 1765m (spadek 413m, średnio 33%).
            # Bardziej stromy w górnej części (ścianka ekspert), wypłaszczenie pod koniec (Sec. IV.B, Fig. 7a)
            anchors_y = [0.0, 150.0, 400.0, 700.0, 950.0, 1100.0, 1250.0]
            anchors_Z = [2178.0, 2125.0, 2030.0, 1920.0, 1820.0, 1785.0, 1765.0]
        else:
            # Jochstock: start 2504m, meta 2216m (spadek 288m, średnio 32%).
            # Start łagodniejszy (~20-25%), ścianka środkowa (~33%),
            # wypłaszczenie i przeciwstok (counter-slope) w rejonie y ~ 510-550m,
            # oraz odcinek zjazdowy do mety (Sec. II.B, Fig. 7b, 8b, 10b).
            anchors_y = [0.0, 100.0, 250.0, 420.0, 510.0, 550.0, 600.0, 720.0, 889.0]
            anchors_Z = [2504.0, 2475.0, 2420.0, 2360.0, 2332.0, 2333.5, 2320.0, 2275.0, 2216.0]

        # Monotoniczny spline hermitowski (PCHIP) zapobiega niefizycznym oscylacjom Rungego
        pchip = PchipInterpolator(anchors_y, anchors_Z)
        z_profile_y = pchip(self.y_coords)

        # Ukształtowanie poprzeczne (delikatna rynna / naturalna wklęsłość stoku o głębokości ok. 1m)
        lateral_arch = 1.0 * np.sin(np.pi * (self.X / self.width))
        self.elevation = z_profile_y[np.newaxis, :] + lateral_arch

    def _calculate_gradients(self) -> None:
        """Wyznacza gradient dZ/dx, dZ/dy oraz lokalny kąt nachylenia stoku alpha (Eq. 14)."""
        dZ_dx, dZ_dy = np.gradient(self.elevation, self.resolution, self.resolution)
        self.grad_x = dZ_dx
        self.grad_y = dZ_dy

        # tan(alpha) = sqrt((dZ/dx)^2 + (dZ/dy)^2)
        grad_mag = np.hypot(dZ_dx, dZ_dy)
        self.slope_angle = np.arctan(grad_mag)

    def _get_indices(self, x: float, y: float) -> Tuple[int, int]:
        ix = int(np.clip(round(x / self.resolution), 0, self.nx - 1))
        iy = int(np.clip(round(y / self.resolution), 0, self.ny - 1))
        return ix, iy

    def get_elevation(self, x: float, y: float) -> float:
        ix, iy = self._get_indices(x, y)
        return float(self.elevation[ix, iy])

    def get_slope_angle(self, x: float, y: float) -> float:
        """Lokalny kąt nachylenia stoku alpha w radianach (Eq. 14)."""
        ix, iy = self._get_indices(x, y)
        return float(self.slope_angle[ix, iy])

    def get_fall_line_vector(self, x: float, y: float) -> np.ndarray:
        """
        Zwraca znormalizowany wektor 2D linii spadku stoku (fall line) w punkcie (x, y).
        Kierunek najszybszego spadku to -grad(Z).
        """
        ix, iy = self._get_indices(x, y)
        gx = self.grad_x[ix, iy]
        gy = self.grad_y[ix, iy]

        fall_x = -gx
        fall_y = -gy
        norm = np.hypot(fall_x, fall_y)
        if norm < 1e-6:
            return np.array([0.0, 1.0])
        return np.array([fall_x / norm, fall_y / norm])

    def get_borders_at(self, y: float) -> Tuple[float, float]:
        """Zwraca (x_left, x_right) dla podanej współrzędnej y."""
        iy = int(np.clip(round(y / self.resolution), 0, self.ny - 1))
        return float(self.left_border[iy]), float(self.right_border[iy])

    def get_friction(self, x: float, y: float) -> float:
        ix, iy = self._get_indices(x, y)
        return float(self.friction_map[ix, iy])

    def add_ice_patch(
        self,
        center_x: float,
        center_y: float,
        radius_x: float = 8.0,
        radius_y: float = 20.0,
        ice_friction: float = 0.04,
    ) -> None:
        """Dodanie strefy oblodzenia o obniżonym tarciu mu."""
        dist_sq = ((self.X - center_x) / radius_x) ** 2 + ((self.Y - center_y) / radius_y) ** 2
        mask = dist_sq <= 1.0
        intensity = np.clip(1.0 - dist_sq, 0.0, 1.0)
        self.friction_map = np.where(
            mask,
            self.friction_map * (1.0 - intensity) + ice_friction * intensity,
            self.friction_map,
        )

    def add_obstacle(self, x: float, y: float, radius: float = 1.5, obstacle_type: str = "tree") -> Obstacle:
        obs = Obstacle(x=x, y=y, radius=radius, obstacle_type=obstacle_type)
        self.obstacles.append(obs)
        return obs
