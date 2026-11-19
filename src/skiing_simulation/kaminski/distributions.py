"""Probabilistyczne rozkłady prawdopodobieństwa i funkcja przejścia (Kamiński 2026).

Zawiera:
- Parametry rozkładów bazowych dla poszczególnych etapów zakrętów (Tabele 4.1 i 4.2),
- Generowanie dwuwymiarowego skośnego rozkładu normalnego (Skew-Normal),
- Iloczyn Hadamarda rozkładu bazowego z maskami środowiskowymi i kolizyjnymi
  oraz dynamicznym wykładnikiem sterowalności beta(v_self, skill),
- Stochastyczny wybór nowej pozycji narciarza.
"""

from __future__ import annotations

from typing import Literal, Tuple, Dict, List
import numpy as np
from scipy.stats import skewnorm

TurnDirection = Literal["left", "right", "down"]

# Tabele 4.1 i 4.2 z rozprawy doktorskiej:
# Wartości względne (jako ułamek w = szerokości sąsiedztwa i h = wysokości sąsiedztwa)
# Dla zakrętów w lewo (Tabela 4.1). Zakręty w prawo to odbicie lustrzane mu_x = 1 - mu_x.
TURN_TABLE_LEFT: Dict[int, List[Tuple[float, float]]] = {
    3: [
        (0.55, 0.70),
        (0.60, 0.65),
        (0.55, 0.70),
    ],
    5: [
        (0.55, 0.60),
        (0.60, 0.55),
        (0.65, 0.50),
        (0.60, 0.55),
        (0.55, 0.60),
    ],
    7: [
        (0.55, 0.70),
        (0.65, 0.65),
        (0.75, 0.60),
        (0.80, 0.55),
        (0.75, 0.60),
        (0.65, 0.65),
        (0.55, 0.70),
    ],
    9: [
        (0.55, 0.70),
        (0.65, 0.60),
        (0.75, 0.50),
        (0.85, 0.40),
        (0.90, 0.30),
        (0.85, 0.40),
        (0.75, 0.50),
        (0.65, 0.60),
        (0.55, 0.70),
    ],
}


def get_turn_mus(
    turn_type: TurnDirection,
    length: int,
    step_idx: int,
    w: int,
    h: int,
    s_y: float,
    asymmetry: float = 0.0,
) -> Tuple[float, float]:
    """Zwraca środki (mu_x, mu_y) rozkładu bazowego w układzie współrzędnych sąsiedztwa.

    step_idx: 1-indexed (1 .. length).
    w: szerokość sąsiedztwa (2v + 1)
    h: wysokość sąsiedztwa (v + 1)
    asymmetry: opcjonalna asymetria skrętu (np. snowboard frontside/backside).
    """
    if turn_type == "down" or length == 1:
        # Jazda prosto w dół (Sec. 4.3.4):
        mu_x = (w - 1) / 2.0
        mu_y = 0.9 * s_y * (h - 1)
        return mu_x, mu_y

    stages = TURN_TABLE_LEFT.get(length)
    if stages is None or step_idx < 1 or step_idx > len(stages):
        raise ValueError(f"Nieprawidłowa długość zakrętu ({length}) lub etap ({step_idx})")

    rel_x, rel_y = stages[step_idx - 1]

    if turn_type == "right":
        rel_x = 1.0 - rel_x

    # Korekta asymetryczna (dla snowboardu)
    if asymmetry != 0.0:
        rel_x = np.clip(rel_x + asymmetry, 0.05, 0.95)

    # Przeliczenie na współrzędne pikselowe sąsiedztwa [0 .. w-1] i [0 .. h-1]
    mu_x = rel_x * (w - 1)
    mu_y = rel_y * (h - 1)
    return mu_x, mu_y


def compute_gamma_x(rel_pos: float, gamma_max: float = 5.0) -> float:
    """Wyznacza współczynnik skośności gamma_x w zależności od pozycji poprzecznej na stoku.

    rel_pos: pozycja na stoku jako ułamek szerokości [0.0 .. 1.0] (od lewej do prawej krawędzi).
    Zgodnie z podrozdziałem 4.3.8 i Rysunkiem 4.10:
    - rel_pos < 0.15 => +1.0 * gamma_max (blisko lewej krawędzi, odpychaj w prawo)
    - 0.15 <= rel_pos < 0.30 => +0.5 * gamma_max
    - 0.30 <= rel_pos < 0.40 => +0.25 * gamma_max
    - 0.40 <= rel_pos <= 0.60 => 0.0 (środek stoku)
    - 0.60 < rel_pos <= 0.70 => -0.25 * gamma_max
    - 0.70 < rel_pos <= 0.85 => -0.5 * gamma_max
    - rel_pos > 0.85 => -1.0 * gamma_max (blisko prawej krawędzi, odpychaj w lewo)
    """
    if rel_pos < 0.15:
        return 1.0 * gamma_max
    elif rel_pos < 0.30:
        return 0.5 * gamma_max
    elif rel_pos < 0.40:
        return 0.25 * gamma_max
    elif rel_pos <= 0.60:
        return 0.0
    elif rel_pos <= 0.70:
        return -0.25 * gamma_max
    elif rel_pos <= 0.85:
        return -0.5 * gamma_max
    else:
        return -1.0 * gamma_max


def sample_turn_direction(
    prev_turn: TurnDirection | None,
    free_left: int,
    free_right: int,
    rng: np.random.Generator,
) -> TurnDirection:
    """Losuje kierunek kolejnego manewru zgodnie z intencją z podrozdziału 4.3.6.

    Gdy poprzednio jazda na wprost lub start, preferowany jest kierunek z WIĘKSZĄ dostępną przestrzenią.
    Poprawiono błąd z doktoratu zamieniający indeksy p_l i p_r.
    """
    if prev_turn == "right":
        # p_r = 0.1, p_l = 0.6, p_d = 0.3
        probs = [0.6, 0.1, 0.3]
        choices: List[TurnDirection] = ["left", "right", "down"]
        return rng.choice(choices, p=probs)
    elif prev_turn == "left":
        # p_r = 0.6, p_l = 0.1, p_d = 0.3
        probs = [0.1, 0.6, 0.3]
        choices = ["left", "right", "down"]
        return rng.choice(choices, p=probs)
    else:
        # Poprzednio jazda prosto lub start:
        l = max(1, free_left)
        r = max(1, free_right)
        if l < r:
            # Więcej miejsca z prawej strony (r > l), więc skręcamy w PRAWO (p_r wyższe)!
            p_r = 0.9 * (1.0 - (l / r))
            p_r = max(0.0, min(0.9, p_r))
            p_l = 0.9 - p_r
            p_d = 0.1
        else:
            # Więcej miejsca z lewej strony (l >= r), więc skręcamy w LEWO (p_l wyższe)!
            p_l = 0.9 * (1.0 - (r / l))
            p_l = max(0.0, min(0.9, p_l))
            p_r = 0.9 - p_l
            p_d = 0.1
        total = p_l + p_r + p_d
        probs = [p_l / total, p_r / total, p_d / total]
        choices = ["left", "right", "down"]
        return rng.choice(choices, p=probs)


def sample_turn_length(
    v: float,
    v_max: float,
    rng: np.random.Generator,
    style_preference: str = "standard",
) -> int:
    """Losuje długość zakrętu t_l w oparciu o aktualną prędkość v, v_max i styl jazdy."""
    ratio = v / max(1.0, v_max)
    lengths = [1, 3, 5, 7, 9]

    if style_preference == "plow":
        # Pług: głównie jazda prosto lub krótkie asekuracyjne skręty
        probs = [0.35, 0.40, 0.20, 0.05, 0.00]
    elif style_preference == "short_turns":
        # Śmig: silna preferencja krótkich, dynamicznych zakrętów
        probs = [0.05, 0.65, 0.25, 0.05, 0.00]
    elif style_preference == "carving":
        # Carving: długie, szerokie łuki carvingowe
        probs = [0.05, 0.10, 0.25, 0.35, 0.25]
    elif style_preference == "snowboard":
        # Snowboard: szerokie łuki z tendencją do trawersowania
        probs = [0.10, 0.15, 0.30, 0.25, 0.20]
    else:
        # Standardowy rozkład z doktoratu Kamińskiego (Sec. 4.3.6)
        if ratio < 0.5:
            probs = [0.30, 0.30, 0.20, 0.10, 0.10]
        elif ratio <= 0.75:
            probs = [0.10, 0.20, 0.40, 0.20, 0.10]
        else:
            probs = [0.05, 0.15, 0.30, 0.25, 0.25]

    return int(rng.choice(lengths, p=probs))


def sample_scale_sy(
    v: float,
    v_max: float,
    rng: np.random.Generator,
) -> float:
    """Losuje współczynnik skalujący s_y w zależności od prędkości (Sec. 4.3.7)."""
    ratio = v / max(1.0, v_max)
    values = [0.7, 0.8, 0.9, 1.0]

    if ratio < 0.25:
        probs = [0.05, 0.15, 0.30, 0.50]
    elif ratio <= 0.50:
        probs = [0.05, 0.30, 0.50, 0.15]
    elif ratio <= 0.75:
        probs = [0.15, 0.50, 0.30, 0.05]
    else:
        probs = [0.50, 0.30, 0.15, 0.05]

    return float(rng.choice(values, p=probs))


def compute_base_distribution(
    w: int,
    h: int,
    mu_x: float,
    mu_y: float,
    gamma_x: float,
    gamma_y: float = 5.0,
    sigma_x: float = 1.5,
    sigma_y: float = 1.5,
    s_y: float = 1.0,
) -> np.ndarray:
    """Oblicza bazowy rozkład prawdopodobieństwa B na siatce sąsiedztwa o wymiarach (h, w).

    w = 2v + 1 (kolumny x od 0 do w-1)
    h = v + 1  (wiersze y od 0 do h-1)
    """
    x_coords = np.arange(w, dtype=float)
    y_coords = np.arange(h, dtype=float)

    # Skośny rozkład normalny dla osi x oraz y
    pdf_x = skewnorm.pdf(x_coords, a=gamma_x, loc=mu_x, scale=max(0.2, sigma_x))
    pdf_y = skewnorm.pdf(y_coords, a=gamma_y, loc=mu_y, scale=max(0.2, sigma_y))

    # Iloczyn zewnętrzny: B(y, x) = s_y * pdf_y(y) * pdf_x(x)
    B = s_y * np.outer(pdf_y, pdf_x)
    return B


def sample_next_position(
    base_dist: np.ndarray,
    env_mask: np.ndarray,
    collision_mask: np.ndarray,
    beta_coll: float,
    beta_env: float,
    rng: np.random.Generator,
) -> Tuple[int, int]:
    """Wyznacza wypadkowy rozkład i losuje nowe przemieszczenie (dy, dx).

    Wykorzystuje mechanizm 'przeciążania' maski pędem:
    P = B * (M_env)^beta_env * (M_coll)^beta_coll
    Gdy beta -> 0 (duży pęd, brak czasu reakcji), pęd narciarza przeciąża maski i wjeżdża w lód/innego narciarza.
    """
    # Zastosowanie wykładników sterowalności
    # Bezpieczne potęgowanie: wartości masek są w [0, 1]
    effective_env = np.power(np.clip(env_mask, 0.0, 1.0), beta_env)
    effective_coll = np.power(np.clip(collision_mask, 0.0, 1.0), beta_coll)

    # Wypadkowy rozkład P (Iloczyn Hadamarda z przeciążeniem)
    P = base_dist * effective_env * effective_coll

    total_prob = np.sum(P)
    if total_prob <= 1e-12:
        # Fallback awaryjny:
        fallback = env_mask * base_dist
        if np.sum(fallback) > 0:
            P = fallback / np.sum(fallback)
        else:
            h, w = base_dist.shape
            return min(1, h - 1), (w - 1) // 2

    P = P / np.sum(P)

    # Losowanie ruletkowe
    flat_idx = rng.choice(P.size, p=P.ravel())
    dy, dx = np.unravel_index(flat_idx, P.shape)
    return int(dy), int(dx)
