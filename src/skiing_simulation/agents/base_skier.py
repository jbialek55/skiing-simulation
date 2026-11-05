import numpy as np
from mesa import Agent
from typing import Tuple, List, Dict, Any, Optional


class BaseSkier(Agent):
    """
    Model narciarza w 100% zgodny z artykułem:
    Thomas Holleczek & Gerhard Tröster,
    "Particle-based model for skiing traffic", Physical Review E 85, 056101 (2012).
    
    Łączy:
    1. Siły społeczne (Social Forces - Sec. II.C, Eq. 1-13):
       - Destination force F_D (przyciąganie do kolejnych punktów nawigacyjnych co 50m)
       - Edge repulsion F_L, F_R (odpychanie od lewej i prawej krawędzi stoku)
       - Athlete repulsion F_A (odpychanie od innych narciarzy)
       - Obstacle repulsion F_O (odpychanie od przeszkód)
       - Kąt widzenia 2*phi = 180 deg i próg skrętu delta = 10 deg (Fig. 3)
    2. Siły fizyczne (Physical Forces - Sec. II.D, Eq. 14-29):
       - Siła zsuwająca F_P i poprzeczna F_lat z grawitacji i nachylenia terenu
       - Siła dośrodkowa F_C w skręcie ciętym z promieniem taliowania R_sc = 10m
       - Siła efektywna docisku nart do śniegu F_eff / F_EFF
       - Tarcie kinetyczne nart o śnieg F_ground / F_GROUND (mu = 0.10)
       - Opór powietrza F_air (Cd = 1.0, A = 0.6 m^2, rho = 1.3163 kg/m^3)
    """

    def __init__(
        self,
        model,
        pos: Tuple[float, float],
        skier_id: Optional[int] = None,
        # Parametry fizyczne z Table III & IV (Holleczek & Tröster 2012)
        mass: float = 85.0,              # masa narciarza m = 85 kg (Table III)
        sidecut_radius: float = 10.0,    # promień taliowania nart R_sc = 10 m (Table IV)
        friction_coeff: float = 0.10,    # współczynnik tarcia kinetycznego mu = 0.10 (Table IV)
        drag_coeff: float = 1.0,         # współczynnik oporu powietrza Cd = 1.0 (Table IV)
        frontal_area: float = 0.6,       # pole powierzchni czołowej A = 0.6 m^2 (Table IV)
        air_density: float = 1.3163,     # gęstość powietrza przy -5 deg C rho = 1.3163 kg/m^3 (Table III)
        v_min_kmh: float = 5.0,          # minimalna prędkość startowa / marszu v_min = 5 km/h (Sec. II.B, IV.B)
        # Parametry sił społecznych z Table II (Holleczek & Tröster 2012)
        strength_destination: float = 1.0,     # A0 = 1 (Table II)
        strength_edge_repulsion: float = 1.0,  # U0 = 1 (Table II)
        range_edge_repulsion: float = 50.0,    # RL = RR = 50 m (Table II)
        strength_athlete_repulsion: float = 100.0, # V0 = 100 (Table II)
        range_athlete_repulsion: float = 2.0,      # RA = 2 m (Table II)
        threshold_delta_deg: float = 10.0,     # próg odchylenia kierunku delta = 10 deg (Table II)
        waypoint_spacing: float = 50.0,        # odstęp między punktami nawigacyjnymi = 50 m (Sec. IV.B)
        custom_waypoints: Optional[List[Tuple[float, float]]] = None,
    ):
        super().__init__(model)
        self.pos = None
        self.initial_pos = np.array(pos, dtype=float)
        self.skier_id = skier_id if skier_id is not None else self.unique_id

        # Parametry fizyczne (Table III, IV)
        self.mass = mass
        self.R_sc = sidecut_radius
        self.mu = friction_coeff
        self.Cd = drag_coeff
        self.A = frontal_area
        self.rho = air_density
        self.v_min = v_min_kmh / 3.6  # m/s (~1.389 m/s)
        self.gravity = 9.81

        # Parametry sił społecznych (Table II)
        self.A0 = strength_destination
        self.U0 = strength_edge_repulsion
        self.RL = range_edge_repulsion
        self.RR = range_edge_repulsion
        self.V0 = strength_athlete_repulsion
        self.RA = range_athlete_repulsion
        self.delta = np.radians(threshold_delta_deg)
        self.cos_delta = np.cos(self.delta)
        self.waypoint_spacing = waypoint_spacing

        # Prędkość początkowa (v_min w dół stoku)
        self.velocity = np.array([0.0, self.v_min], dtype=float)

        # Generowanie punktów nawigacyjnych (Sec. IV.B)
        self.waypoints: List[np.ndarray] = []
        if custom_waypoints is not None:
            self.waypoints = [np.array(wp, dtype=float) for wp in custom_waypoints]
        else:
            self._generate_waypoints()

        self.current_wp_idx = 0

        # Stan i telemetria
        self.status = "active"
        self.is_turning = False
        self.turn_direction = 0  # -1 lewo, +1 prawo, 0 prosto
        self.crash_reason: Optional[str] = None
        self.trajectory: List[Tuple[float, float, float, float, bool]] = []  # x, y, speed, mu, turning

    def _generate_waypoints(self) -> None:
        """
        Generuje kolejne punkty nawigacyjne co 50m zgodnie z Sec. IV.B:
        'skiers choose these waypoints randomly every 50 m, using a uniform
        distribution on the corresponding line from the left to the right edge of the slope.'
        """
        terrain = self.model.terrain
        length = terrain.length
        
        y_targets = np.arange(self.waypoint_spacing, length + self.waypoint_spacing, self.waypoint_spacing)
        for y_wp in y_targets:
            y_clamped = min(y_wp, length - 2.0)
            x_left, x_right = terrain.get_borders_at(y_clamped)
            # Losowy punkt z rozkładu jednostajnego między krawędziami (z buforem 5m od brzegu)
            margin = 6.0
            x_wp = np.random.uniform(x_left + margin, x_right - margin)
            self.waypoints.append(np.array([x_wp, y_clamped], dtype=float))

        # Punkt celu poza linią mety
        x_left_end, x_right_end = terrain.get_borders_at(length - 1.0)
        self.waypoints.append(np.array([(x_left_end + x_right_end) / 2.0, length + 25.0], dtype=float))

    @property
    def speed(self) -> float:
        """Prędkość skalarna w m/s."""
        return float(np.linalg.norm(self.velocity))

    @property
    def speed_kmh(self) -> float:
        """Prędkość w km/h."""
        return self.speed * 3.6

    @property
    def direction_of_motion(self) -> np.ndarray:
        """Znormalizowany wektor aktualnego kierunku ruchu e_r_dot."""
        s = self.speed
        if s < 1e-6:
            return np.array([0.0, 1.0])
        return self.velocity / s

    def _record_telemetry(self, current_friction: float) -> None:
        if self.pos is not None:
            self.trajectory.append((
                float(self.pos[0]),
                float(self.pos[1]),
                self.speed,
                current_friction,
                self.is_turning,
            ))

    def _update_waypoint(self, current_pos: np.ndarray) -> np.ndarray:
        """Aktualizacja celu nawigacyjnego po dotarciu w pobliże punktu."""
        if not self.waypoints:
            return np.array([self.model.terrain.width / 2.0, self.model.terrain.length + 20.0])

        current_target = self.waypoints[self.current_wp_idx]
        dist_to_wp = np.linalg.norm(current_target - current_pos)
        
        # Warunek przejścia do kolejnego punktu: dojechanie na odległość < 5m lub przekroczenie linii Y
        if dist_to_wp < 6.0 or current_pos[1] >= current_target[1] - 4.0:
            if self.current_wp_idx < len(self.waypoints) - 1:
                self.current_wp_idx += 1
                current_target = self.waypoints[self.current_wp_idx]
            else:
                return np.array([self.model.terrain.width / 2.0, self.model.terrain.length + 25.0])

        return current_target

    def compute_social_forces(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Oblicza bezwymiarowe siły społeczne F_social oraz pożądany kierunek e_social
        zgodnie z Sec. II.C i Eq. 1 - 13.
        """
        terrain = self.model.terrain
        r_a = np.array(self.pos, dtype=float)
        e_v = self.direction_of_motion

        # 1. Destination force F_D (Eq. 1, 2)
        target_wp = self._update_waypoint(r_a)
        diff_dest = target_wp - r_a
        norm_dest = np.linalg.norm(diff_dest)
        if norm_dest > 1e-6:
            e_dest = diff_dest / norm_dest
        else:
            e_dest = np.array([0.0, 1.0])
        F_D = self.A0 * e_dest

        # 2. Edge repulsion F_L, F_R (Eq. 3, 9, 10, 30, 31)
        x_left, x_right = terrain.get_borders_at(r_a[1])

        # Lewa krawędź: r_aL = r_a - r_L^a = (x - x_left, 0)
        d_L = max(r_a[0] - x_left, 0.0)
        f_L = (self.U0 / self.RL) * np.exp(-d_L / self.RL) * np.array([1.0, 0.0])
        # Waga kąta widzenia w(r_dot_a, -r_aL): wektor do lewej krawędzi to [-1, 0]
        # w = 1 jeśli (e_v . [-1, 0]) >= cos(90 deg) = 0 => e_v[0] <= 0 (narciarz zbliża się do lewej krawędzi)
        w_L = 1.0 if np.dot(e_v, np.array([-1.0, 0.0])) >= 0.0 else 0.0
        F_L = w_L * f_L

        # Prawa krawędź: r_aR = r_a - r_R^a = (x - x_right, 0)
        d_R = max(x_right - r_a[0], 0.0)
        f_R = (self.U0 / self.RR) * np.exp(-d_R / self.RR) * np.array([-1.0, 0.0])
        w_R = 1.0 if np.dot(e_v, np.array([1.0, 0.0])) >= 0.0 else 0.0
        F_R = w_R * f_R

        # 3. Obstacle repulsion F_O (Eq. 6, 12)
        F_O = np.array([0.0, 0.0], dtype=float)
        for obs in terrain.obstacles:
            r_ao = r_a - np.array([obs.x, obs.y])
            d_o = np.linalg.norm(r_ao)
            if d_o > 1e-6 and d_o < 25.0:
                # Wektor od narciarza do przeszkody to -r_ao
                w_O = 1.0 if np.dot(e_v, -r_ao / d_o) >= 0.0 else 0.0
                f_O = (10.0 / 5.0) * np.exp(-d_o / 5.0) * (r_ao / d_o)
                F_O += w_O * f_O

        # 4. Athlete repulsion F_A (Eq. 4, 5, 11, 32)
        F_A = np.array([0.0, 0.0], dtype=float)
        for other in self.model.agents:
            if other is self or not isinstance(other, BaseSkier) or other.status != "active":
                continue
            r_ab = r_a - np.array(other.pos)
            d_ab = np.linalg.norm(r_ab)
            if 1e-6 < d_ab < 30.0:
                v_b = other.speed
                e_b = other.direction_of_motion
                tau = 0.5
                v_tau_eb = v_b * tau * e_b
                term1 = (d_ab + np.linalg.norm(r_ab - v_tau_eb)) ** 2
                term2 = (v_b * tau) ** 2
                s = 0.5 * np.sqrt(max(term1 - term2, 1e-6))
                
                w_A = 1.0 if np.dot(e_v, -r_ab / d_ab) >= 0.0 else 0.0
                f_A = (self.V0 / self.RA) * np.exp(-s / self.RA) * (r_ab / d_ab)
                F_A += w_A * f_A

        # Net social force (Eq. 13)
        F_social = F_D + F_L + F_R + F_O + F_A
        norm_social = np.linalg.norm(F_social)
        if norm_social > 1e-6:
            e_social = F_social / norm_social
        else:
            e_social = e_dest

        return F_social, e_social

    def step(self) -> None:
        """Krok symulacji narciarza (dt = 0.01 s)."""
        if self.status != "active":
            return

        dt = self.model.dt
        terrain = self.model.terrain
        x, y = self.pos

        # 1. Sprawdzenie osiągnięcia końca stoku
        if y >= terrain.length - 2.0:
            self.status = "finished"
            return

        # 2. Odczyt geometrii terenu pod nartami
        alpha = terrain.get_slope_angle(x, y)          # kąt nachylenia stoku (Eq. 14)
        fall_x, fall_y = terrain.get_fall_line_vector(x, y) # kierunek linii spadku
        e_fall = np.array([fall_x, fall_y], dtype=float)
        current_friction = terrain.get_friction(x, y)

        speed = self.speed
        # Reguła przeciwstoku (Sec. II.B): jeśli narciarz zatrzyma się, idzie z prędkością marszu v_min
        if speed < self.v_min:
            speed = self.v_min
            if np.linalg.norm(self.velocity) < 1e-6:
                self.velocity = self.v_min * e_fall
            else:
                self.velocity = self.v_min * (self.velocity / np.linalg.norm(self.velocity))

        e_v = self.direction_of_motion

        # 3. Wyznaczenie sił społecznych i pożądanego kierunku e_social (Sec. II.C)
        _, e_social = self.compute_social_forces()

        # 4. Decyzja o wykonaniu skrętu (Sec. II.B, Fig. 3):
        # Czy kąt między e_v a e_social przekracza próg delta = 10 deg?
        cos_deviation = float(np.clip(np.dot(e_v, e_social), -1.0, 1.0))
        is_turning = cos_deviation < self.cos_delta
        self.is_turning = is_turning

        # ------------------ SIŁY FIZYCZNE (Sec. II.D, Eq. 14 - 29) ------------------
        # Siła grawitacji wzdłuż linii spadku stoku F_S (Eq. 18): ||F_S|| = m * g * sin(alpha)
        F_S = self.mass * self.gravity * np.sin(alpha) * e_fall

        # Składowa przyspieszająca wzdłuż aktualnego toru jazdy F_P (Eq. 19): F_P = (F_S . e_v) * e_v
        F_P_mag = float(np.dot(F_S, e_v))
        F_P = F_P_mag * e_v

        # Siła poprzeczna F_lat prostopadła do toru jazdy (Eq. 20): F_lat = F_S - F_P
        F_lat = F_S - F_P

        # Opór powietrza F_air (Eq. 25): F_air = -0.5 * Cd * A * rho * v^2 * e_v
        F_air = -0.5 * self.Cd * self.A * self.rho * (speed ** 2) * e_v

        # Siła nacisku normalnego F_N = m * g * cos(alpha) (Eq. 17)
        F_N_mag = self.mass * self.gravity * np.cos(alpha)

        if is_turning:
            # Określenie kierunku skrętu w lewo lub w prawo (iloczyn wektorowy w 2D)
            cross_product = e_v[0] * e_social[1] - e_v[1] * e_social[0]
            if cross_product > 0:  # e_social jest po lewej stronie e_v -> skręt w lewo
                u_turn = np.array([-e_v[1], e_v[0]], dtype=float)
                self.turn_direction = -1
            else:                  # e_social jest po prawej stronie e_v -> skręt w prawo
                u_turn = np.array([e_v[1], -e_v[0]], dtype=float)
                self.turn_direction = 1

            # Siła dośrodkowa F_C w skręcie ciętym o promieniu R_sc = 10m (Eq. 22)
            # F_C skierowana prostopadle do e_v w kierunku środka skrętu
            F_C_mag = (self.mass / self.R_sc) * (speed ** 2)
            F_C = F_C_mag * u_turn

            # Siła boczna podczas skrętu F_LAT = F_lat - F_C (Eq. 21)
            F_LAT = F_lat - F_C

            # Siła efektywna docisku nart F_EFF (Eq. 24):
            # ||F_EFF|| = sqrt(||F_LAT||^2 + ||F_N||^2)
            F_EFF_mag = np.sqrt(np.dot(F_LAT, F_LAT) + F_N_mag ** 2)

            # Tarcie kinetyczne w skręcie F_GROUND (Eq. 27): F_GROUND = -mu * ||F_EFF|| * e_v
            F_GROUND = -self.mu * F_EFF_mag * e_v

            # Siła wypadkowa F_NET (Eq. 29): F_NET = F_P + F_air + F_GROUND + F_C
            F_NET = F_P + F_air + F_GROUND + F_C
        else:
            self.turn_direction = 0
            # Zjazd na wprost (Straight line descent)
            # ||F_eff|| = sqrt(||F_lat||^2 + ||F_N||^2) (Eq. 23)
            F_eff_mag = np.sqrt(np.dot(F_lat, F_lat) + F_N_mag ** 2)

            # Tarcie kinetyczne na wprost F_ground (Eq. 26)
            F_ground = -self.mu * F_eff_mag * e_v

            # Siła wypadkowa F_net (Eq. 28): F_net = F_P + F_air + F_ground
            F_NET = F_P + F_air + F_ground

        # ------------------ CAŁKOWANIE RUCHU (Sec. IV.A) ------------------
        acc = F_NET / self.mass
        self.velocity += acc * dt

        # Jeśli prędkość spadnie do zera (np. na przeciwstoku), przejście w marsz (Sec. II.B)
        if self.speed < self.v_min:
            self.velocity = self.v_min * e_social

        new_x = x + self.velocity[0] * dt
        new_y = y + self.velocity[1] * dt

        # ------------------ ODBICIE OD KRAWĘDZI (Sec. II.B) ------------------
        # 'Should skiers collide with either slope edge during their descent,
        # they will be reflected back to the corresponding center point of the slope.'
        x_left, x_right = terrain.get_borders_at(new_y)
        x_center = (x_left + x_right) / 2.0
        if new_x <= x_left:
            new_x = x_left + 1.0
            # Odbicie w stronę środka stoku
            self.velocity[0] = abs(self.velocity[0])
        elif new_x >= x_right:
            new_x = x_right - 1.0
            self.velocity[0] = -abs(self.velocity[0])

        # Aktualizacja pozycji w przestrzeni ciągłej
        self.model.space.move_agent(self, (new_x, new_y))
        self.pos = (new_x, new_y)
        self._record_telemetry(current_friction)
