"""Model stoku narciarskiego 2.5D (Kamiński 2026).

Reprezentuje regularną siatkę 1m x 1m z wysokościami h(x, y),
granicami trasy, strefami buforowymi wokół przeszkód i krawędzi (p_c = d/r)
oraz profilem nachylenia opartym na trasie Gąsienicowej na Kasprowym Wierchu.
"""

from __future__ import annotations

from typing import Tuple, Dict, Optional
import numpy as np


# Współczynniki wielomianu 5. stopnia z Tabeli 6.1 z rozprawy doktorskiej
# f_h(x, y) aproksymujące stok Kasprowego Wierchu
KASPROWY_COEFFS: Dict[str, float] = {
    "00": -0.021, "10": -0.232, "01": 0.161, "20": 0.412, "11": -0.136, "02": 0.108,
    "30": -0.227, "21": 0.167, "12": 0.438, "03": -0.083, "40": -0.691, "31": 0.076,
    "22": -0.073, "13": -0.213, "04": 0.037, "50": 1.273, "41": -0.161, "32": -0.008,
    "23": 0.011, "14": 0.029, "05": -0.005
}


class SlopeGrid25D:
    """Dwuiipółwymiarowa regularna siatka reprezentująca stok narciarski.

    Komórki o wymiarach 1m x 1m.
    y = 0: początek (góra stoku)
    y = length - 1: koniec (dół stoku)
    x = 0 .. width - 1: oś poprzeczna
    """

    def __init__(
        self,
        width: int = 40,
        length: int = 350,
        buffer_zone_r: int = 3,
        use_kasprowy_profile: bool = True,
        base_altitude: float = 1987.0,
    ) -> None:
        self.width = width
        self.length = length
        self.r = buffer_zone_r
        self.base_altitude = base_altitude

        # Siatka wysokości h(x, y)
        self.heights = np.zeros((self.length, self.width), dtype=float)
        # Siatka prawdopodobieństw wjazdu p_c(x, y)
        self.p_c = np.zeros((self.length, self.width), dtype=float)
        # Siatka oblodzenia I(x, y) w zakresie [0.0, 1.0]
        self.ice_map = np.zeros((self.length, self.width), dtype=float)
        # Siatka chropowatości/muld R(x, y) w zakresie [0.0, 1.0]
        self.moguls_map = np.zeros((self.length, self.width), dtype=float)

        self.ice_patches: list[dict] = []
        self.mogul_fields: list[dict] = []

        self._init_geometry(use_kasprowy_profile)

    def _init_geometry(self, use_kasprowy_profile: bool) -> None:
        """Inicjalizuje profil wysokościowy oraz maskę granic z buforami."""
        for y in range(self.length):
            left, right = self.get_borders_at(y)
            for x in range(self.width):
                # 1. Wysokość
                if use_kasprowy_profile:
                    # Profil wzorowany na Kasprowym Wierchu (czarna ściana -> czerwony dół):
                    # Górne 180 m: nachylenie ~0.50 - 0.45 (stromy odcinek czarny)
                    # Dolne 170 m: nachylenie ~0.35 - 0.25 (odcinek czerwony, wypłaszczenie)
                    # Dodatkowo lekki profil rynny poprzecznej
                    prog = y / float(self.length)
                    if prog < 0.5:
                        slope_grad = 0.48 - 0.06 * (prog / 0.5)
                    else:
                        slope_grad = 0.42 - 0.15 * ((prog - 0.5) / 0.5)
                    
                    vertical_drop = y * slope_grad
                    cross_profile = 0.015 * ((x - (left + right) / 2.0) ** 2)
                    self.heights[y, x] = self.base_altitude - vertical_drop + cross_profile
                else:
                    # Liniowy spadek 40%
                    self.heights[y, x] = self.base_altitude - 0.40 * y

                # 2. Maska dostępności i strefa buforowa p_c = d / r (Równanie 4.1)
                if x < left or x > right:
                    self.p_c[y, x] = 0.0
                else:
                    dist_to_border = min(x - left, right - x)
                    if dist_to_border < self.r:
                        self.p_c[y, x] = float(dist_to_border) / float(self.r)
                    else:
                        self.p_c[y, x] = 1.0

    def get_borders_at(self, y: int) -> Tuple[int, int]:
        """Zwraca (x_left, x_right) dopuszczalnego obszaru zjazdu na wysokości y.

        Kształt odwzorowuje rynnę Kasprowego Wierchu (Rysunek 6.1 i 6.2 w doktoracie):
        szeroki start, lekkie przewężenie w połowie, rozszerzenie na dole.
        """
        center = self.width / 2.0
        # Zmienna szerokość korytarza: od 18 do 28 metrów
        prog = y / max(1, self.length)
        half_width = 11.0 + 3.0 * np.cos(2.0 * np.pi * prog) + 1.5 * np.sin(np.pi * prog)
        
        # Lekkie wygięcie osi stoku (asymetria opisana w rozdziale 6)
        curvature = 2.5 * np.sin(3.0 * np.pi * prog)
        
        left = int(np.clip(round(center - half_width + curvature), 0, self.width - 2))
        right = int(np.clip(round(center + half_width + curvature), left + 1, self.width - 1))
        return left, right

    def get_altitude(self, x: int, y: int) -> float:
        """Zwraca wysokość n.p.m. w punkcie (x, y)."""
        clamped_x = int(np.clip(x, 0, self.width - 1))
        clamped_y = int(np.clip(y, 0, self.length - 1))
        return float(self.heights[clamped_y, clamped_x])

    def get_slope_inclination(self, x: int, y: int) -> float:
        """Oblicza lokalne nachylenie terenu (h[y] - h[y+1]) wzdłuż stoku (Sec. 4.3.10)."""
        if y >= self.length - 1:
            return 0.30
        h_curr = self.get_altitude(x, y)
        h_next = self.get_altitude(x, y + 1)
        return max(0.0, h_curr - h_next)

    def add_ice_patch(
        self,
        center_x: int,
        center_y: int,
        radius_x: float,
        radius_y: float,
        max_intensity: float = 0.90,
    ) -> None:
        """Dodaje eliptyczny płat lodu o gładko opadającej intensywności."""
        self.ice_patches.append({
            "cx": center_x, "cy": center_y, "rx": radius_x, "ry": radius_y, "intensity": max_intensity
        })
        y_min = max(0, int(center_y - radius_y - 2))
        y_max = min(self.length, int(center_y + radius_y + 3))
        x_min = max(0, int(center_x - radius_x - 2))
        x_max = min(self.width, int(center_x + radius_x + 3))

        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                dist_sq = ((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2
                if dist_sq <= 1.0:
                    # Rozkład Gaussa / gładki spadek do krawędzi
                    val = max_intensity * (1.0 - dist_sq)
                    self.ice_map[y, x] = max(self.ice_map[y, x], float(val))

    def add_mogul_field(
        self,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        max_roughness: float = 0.85,
        bump_amplitude: float = 0.35,
        bump_wavelength: float = 3.0,
    ) -> None:
        """Dodaje pole muld: modyfikuje profil DEM mikronierównościami oraz ustawia siatkę chropowatości."""
        self.mogul_fields.append({
            "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
            "roughness": max_roughness, "amplitude": bump_amplitude
        })
        clamped_y_min = max(0, y_min)
        clamped_y_max = min(self.length, y_max)
        clamped_x_min = max(0, x_min)
        clamped_x_max = min(self.width, x_max)

        omega = 2.0 * np.pi / bump_wavelength
        for y in range(clamped_y_min, clamped_y_max):
            for x in range(clamped_x_min, clamped_x_max):
                self.moguls_map[y, x] = max(self.moguls_map[y, x], max_roughness)
                # Modyfikacja mikro-wysokości DEM (sinusoidalne garby i zagłębienia)
                bump = bump_amplitude * np.sin(omega * x) * np.cos(omega * y)
                self.heights[y, x] += bump

    def setup_default_features(self) -> None:
        """Konfiguruje domyślne strefy trudności na stoku Kasprowego Wierchu (Opcja A).

        1. Płat lodu w naturalnym przewężeniu stoku (y ~ 165m)
        2. Drugi płat lodu na stromym łuku (y ~ 255m)
        3. Pole muld na prawej flance stoku (y ~ 95m - 155m)
        """
        # Płat lodu 1: przewężenie
        self.add_ice_patch(center_x=20, center_y=165, radius_x=5.5, radius_y=14.0, max_intensity=0.92)
        # Płat lodu 2: dolny odcinek
        self.add_ice_patch(center_x=22, center_y=255, radius_x=6.0, radius_y=12.0, max_intensity=0.85)
        # Pole muld: prawa flanka na środkowym odcinku
        self.add_mogul_field(x_min=24, x_max=34, y_min=95, y_max=155, max_roughness=0.85, bump_amplitude=0.35)

    def get_ice_intensity(self, x: int, y: int) -> float:
        """Zwraca stopień oblodzenia w komórce (x, y) w zakresie [0.0, 1.0]."""
        clamped_x = int(np.clip(x, 0, self.width - 1))
        clamped_y = int(np.clip(y, 0, self.length - 1))
        return float(self.ice_map[clamped_y, clamped_x])

    def get_mogul_roughness(self, x: int, y: int) -> float:
        """Zwraca chropowatość/zmuldzenie w komórce (x, y) w zakresie [0.0, 1.0]."""
        clamped_x = int(np.clip(x, 0, self.width - 1))
        clamped_y = int(np.clip(y, 0, self.length - 1))
        return float(self.moguls_map[clamped_y, clamped_x])

    def extract_env_mask(
        self,
        agent_x: int,
        agent_y: int,
        v: int,
        ice_avoidance_weight: float = 0.0,
    ) -> np.ndarray:
        """Wycina fragment maski terenu p_c dla sąsiedztwa (2v+1) x (v+1) wokół narciarza.

        Punkty poza siatką stoku otrzymują prawdopodobieństwo 0.0.
        Uwzględnia również świadome omijanie lodu przez bardziej doświadczonych narciarzy.
        """
        w = 2 * v + 1
        h = v + 1
        mask = np.zeros((h, w), dtype=float)

        x_start = agent_x - v
        x_end = agent_x + v + 1
        y_start = agent_y
        y_end = agent_y + v + 1

        # Wycinanie z obsługą przekroczenia granic siatki
        grid_x_min = max(0, x_start)
        grid_x_max = min(self.width, x_end)
        grid_y_min = max(0, y_start)
        grid_y_max = min(self.length, y_end)

        if grid_x_max > grid_x_min and grid_y_max > grid_y_min:
            mask_x_min = grid_x_min - x_start
            mask_x_max = mask_x_min + (grid_x_max - grid_x_min)
            mask_y_min = grid_y_min - y_start
            mask_y_max = mask_y_min + (grid_y_max - grid_y_min)

            base_pc = self.p_c[grid_y_min:grid_y_max, grid_x_min:grid_x_max]

            if ice_avoidance_weight > 0.0:
                ice_sub = self.ice_map[grid_y_min:grid_y_max, grid_x_min:grid_x_max]
                # Prawdopodobieństwo wyboru spada w strefie lodu proporcjonalnie do wagi omijania
                factor = 1.0 - ice_avoidance_weight * ice_sub
                mask[mask_y_min:mask_y_max, mask_x_min:mask_x_max] = base_pc * np.clip(factor, 0.05, 1.0)
            else:
                mask[mask_y_min:mask_y_max, mask_x_min:mask_x_max] = base_pc

        return mask
