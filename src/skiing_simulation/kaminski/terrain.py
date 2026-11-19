"""Model stoku narciarskiego 2.5D (Kamiński 2026).

Zawiera:
- Reprezentację regularnej siatki 1m x 1m z profilami wysokościowymi h(x, y),
- Dynamiczną stygmergię stoku: degradację pokrywy śnieżnej S(x, y),
  samoistne powstawanie oblodzeń w miejscach ześlizgu i odkładanie muld po bokach,
- Strefy przeszkód stałych (np. słupy podpory wyciągu z twardym zakazem p_c = 0),
- Maski buforowe wokół krawędzi i przeszkód (p_c = d / r).
"""

from __future__ import annotations

from typing import Tuple, Dict, List, Optional
import numpy as np


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

        # Siatka bazowej wysokości terenu h(x, y)
        self.heights = np.zeros((self.length, self.width), dtype=float)
        # Siatka dostępności krawędzi i przeszkód p_c(x, y) [0.0 .. 1.0]
        self.p_c = np.zeros((self.length, self.width), dtype=float)

        # Dynamiczne warstwy nawierzchni (stygmergia)
        # Pokrywa śnieżna S(x, y): startowo 1.0 (idealny ubity sztruks)
        self.snow_cover = np.ones((self.length, self.width), dtype=float)
        # Siatka oblodzenia I(x, y) w zakresie [0.0, 1.0]
        self.ice_map = np.zeros((self.length, self.width), dtype=float)
        # Siatka chropowatości/muld B(x, y) w zakresie [0.0, 1.0]
        self.moguls_map = np.zeros((self.length, self.width), dtype=float)

        # Lista przeszkód stałych (np. podpory wyciągu krzesełkowego)
        self.obstacles: List[Dict[str, float]] = []

        # Ręcznie zdefiniowane obszary (opcjonalne cechy statyczne)
        self.ice_patches: List[Dict[str, float]] = []
        self.mogul_fields: List[Dict[str, float]] = []

        self._init_geometry(use_kasprowy_profile)

    def _init_geometry(self, use_kasprowy_profile: bool) -> None:
        """Inicjalizuje profil wysokościowy oraz maskę granic z buforami."""
        for y in range(self.length):
            left, right = self.get_borders_at(y)
            for x in range(self.width):
                # 1. Profil wysokościowy (Kasprowy Wierch)
                if use_kasprowy_profile:
                    prog = y / float(self.length)
                    if prog < 0.5:
                        slope_grad = 0.48 - 0.06 * (prog / 0.5)
                    else:
                        slope_grad = 0.42 - 0.15 * ((prog - 0.5) / 0.5)

                    vertical_drop = y * slope_grad
                    cross_profile = 0.015 * ((x - (left + right) / 2.0) ** 2)
                    self.heights[y, x] = self.base_altitude - vertical_drop + cross_profile
                else:
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
        """Zwraca (x_left, x_right) dopuszczalnego korytarza zjazdu na wysokości y."""
        center = self.width / 2.0
        prog = y / max(1, self.length)
        half_width = 11.0 + 3.0 * np.cos(2.0 * np.pi * prog) + 1.5 * np.sin(np.pi * prog)
        curvature = 2.5 * np.sin(3.0 * np.pi * prog)

        left = int(np.clip(round(center - half_width + curvature), 0, self.width - 2))
        right = int(np.clip(round(center + half_width + curvature), left + 1, self.width - 1))
        return left, right

    def add_obstacle(self, center_x: int, center_y: int, radius: float = 1.8, label: str = "Wyciąg") -> None:
        """Dodaje stałą przeszkodę (np. słup kolejki linowej/wyciągu) ze strefą zakazu p_c = 0."""
        self.obstacles.append({
            "cx": float(center_x),
            "cy": float(center_y),
            "radius": float(radius),
            "label": label,
        })
        # Wprowadzamy strefę wyłączenia i strefę buforową wokół przeszkody
        total_r = radius + self.r
        x_min = max(0, int(center_x - total_r - 1))
        x_max = min(self.width, int(center_x + total_r + 2))
        y_min = max(0, int(center_y - total_r - 1))
        y_max = min(self.length, int(center_y + total_r + 2))

        for gy in range(y_min, y_max):
            for gx in range(x_min, x_max):
                dist = np.hypot(gx - center_x, gy - center_y)
                if dist <= radius:
                    self.p_c[gy, gx] = 0.0
                elif dist < total_r:
                    buf_factor = (dist - radius) / float(self.r)
                    self.p_c[gy, gx] = min(self.p_c[gy, gx], float(buf_factor))

    def setup_default_features(self) -> None:
        """Konfiguruje przeszkody stałe oraz opcjonalne początkowe cechy trudności stoku."""
        # Dodanie 2 słupów podpory wyciągu krzesełkowego w osi stoku
        self.add_obstacle(center_x=16, center_y=110, radius=1.8, label="Słup wyciągu 1")
        self.add_obstacle(center_x=24, center_y=220, radius=1.8, label="Słup wyciągu 2")

        # Początkowy mały płat lodu w naturalnym przewężeniu
        self.add_ice_patch(center_x=20, center_y=165, radius_x=4.0, radius_y=10.0, max_intensity=0.85)

    def add_ice_patch(
        self,
        center_x: int,
        center_y: int,
        radius_x: float,
        radius_y: float,
        max_intensity: float = 0.90,
    ) -> None:
        """Dodaje statyczny eliptyczny płat lodu."""
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
                    val = max_intensity * (1.0 - dist_sq)
                    self.ice_map[y, x] = max(self.ice_map[y, x], float(val))

    def record_skier_action(
        self,
        old_x: int,
        old_y: int,
        new_x: int,
        new_y: int,
        v: float,
        turn_type: str,
    ) -> None:
        """Dynamiczna stygmergia stoku: degradacja pokrywy śniegu i budowa muld.

        Przejazd narciarza zeskrobuje śnieg pod krawędzią (rośnie oblodzenie I)
        i odrzuca go na zewnątrz łuku (buduje chropowatość/muldę B).
        """
        dx = abs(new_x - old_x)
        dy = max(1, abs(new_y - old_y))
        transverse_ratio = float(dx) / float(dy)

        # Intensywność krawędziowania: rośnie z prędkością i ostrością skrętu
        scrape_intensity = (v / 15.0) * (0.05 + 0.15 * transverse_ratio)

        # Zeskrobywanie w komórkach trajektorii
        min_x, max_x = min(old_x, new_x), max(old_x, new_x)
        min_y, max_y = min(old_y, new_y), max(old_y, new_y)

        for cy in range(min_y, max_y + 1):
            for cx in range(min_x, max_x + 1):
                if 0 <= cx < self.width and 0 <= cy < self.length:
                    # Obniżenie pokrywy śniegu
                    self.snow_cover[cy, cx] = max(0.0, self.snow_cover[cy, cx] - 0.04 * scrape_intensity)
                    # Gdy śnieg spada poniżej 0.65, wyłania się lód
                    if self.snow_cover[cy, cx] < 0.65:
                        ice_val = (0.65 - self.snow_cover[cy, cx]) / 0.65
                        self.ice_map[cy, cx] = min(1.0, self.ice_map[cy, cx] + 0.12 * ice_val)

        # Odkładanie śniegu na muldę na zewnątrz zakrętu
        if turn_type in ("left", "right") and dx > 0:
            # Dla zakrętu w lewo, śnieg odrzucany jest w prawo (i odwrotnie)
            dep_offset_x = 2 if turn_type == "left" else -2
            dep_x = int(np.clip(new_x + dep_offset_x, 0, self.width - 1))
            dep_y = int(np.clip(new_y + 1, 0, self.length - 1))

            left_b, right_b = self.get_borders_at(dep_y)
            if left_b <= dep_x <= right_b:
                self.moguls_map[dep_y, dep_x] = min(1.0, self.moguls_map[dep_y, dep_x] + 0.08 * scrape_intensity)
                # Lekkie podbicie wysokości mikroterenu
                self.heights[dep_y, dep_x] += 0.02 * scrape_intensity

    def get_altitude(self, x: int, y: int) -> float:
        """Zwraca wysokość n.p.m. w punkcie (x, y)."""
        clamped_x = int(np.clip(x, 0, self.width - 1))
        clamped_y = int(np.clip(y, 0, self.length - 1))
        return float(self.heights[clamped_y, clamped_x])

    def get_slope_inclination(self, x: int, y: int) -> float:
        """Oblicza lokalne nachylenie terenu (h[y] - h[y+1]) wzdłuż stoku."""
        if y >= self.length - 1:
            return 0.30
        h_curr = self.get_altitude(x, y)
        h_next = self.get_altitude(x, y + 1)
        return max(0.0, h_curr - h_next)

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
        ice_aversion: float = 0.5,
        mogul_aversion: float = 0.4,
    ) -> np.ndarray:
        """Wycina fragment maski terenu p_c dla sąsiedztwa (2v+1) x (v+1) wokół narciarza.

        Łączy twarde ograniczenia stoku/przeszkód z behawioralną awersją do lodu i muld.
        """
        w = 2 * v + 1
        h = v + 1
        mask = np.zeros((h, w), dtype=float)

        x_start = agent_x - v
        x_end = agent_x + v + 1
        y_start = agent_y
        y_end = agent_y + v + 1

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
            ice_sub = self.ice_map[grid_y_min:grid_y_max, grid_x_min:grid_x_max]
            mogul_sub = self.moguls_map[grid_y_min:grid_y_max, grid_x_min:grid_x_max]

            # Czynnik zniechęcający do gorszej nawierzchni
            surface_penalty = (1.0 - ice_aversion * ice_sub) * (1.0 - mogul_aversion * mogul_sub)
            surface_penalty = np.clip(surface_penalty, 0.05, 1.0)

            mask[mask_y_min:mask_y_max, mask_x_min:mask_x_max] = base_pc * surface_penalty

        return mask
