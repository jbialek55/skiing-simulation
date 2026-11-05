"""Automat komórkowy do symulacji zjazdu opisany przez Zhang i in. (2024).

Model celowo nie korzysta z fizyki ani agentów modelu Holleczka.  Artykuł
Zhang i in. opisuje niezależny, stochastyczny automat komórkowy dla małych
gęstości, w którym interakcje między narciarzami nie są jeszcze modelowane.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


TurnDirection = Literal["left", "right"]
MoveKind = Literal["left", "right", "down"]


@dataclass
class ZhangSkier:
    """Stan jednego narciarza w automacie Zhang i in.

    ``p_left`` i ``p_right`` są pamiętanymi prawdopodobieństwami kontynuacji
    odpowiednio lewego i prawego skrętu.  Nie są resetowane przy zmianie
    strony: taką regułę zawiera pseudokod z sekcji 4 artykułu.
    """

    skier_id: int
    x: int
    y: int
    direction: TurnDirection
    p_left: float
    p_right: float
    status: str = "active"
    trajectory: list[tuple[int, int]] = field(default_factory=list)
    probability_history: list[float] = field(default_factory=list)
    direction_history: list[bool] = field(default_factory=list)
    move_history: list[MoveKind] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.trajectory.append((self.x, self.y))
        self.probability_history.append(self.active_probability)
        self.direction_history.append(self.direction == "right")

    @property
    def active_probability(self) -> float:
        """Prawdopodobieństwo ruchu poprzecznego w bieżącym stanie pamięci."""
        return self.p_right if self.direction == "right" else self.p_left


class ZhangModel:
    """Stochastyczny model CA z *Simulation of Downhill Skiing Areas*.

    Współrzędna ``y`` rośnie w dół stoku. W każdym kroku aktywny narciarz
    przechodzi o jedną komórkę w lewo/prawo z prawdopodobieństwem aktywnego
    skrętu albo o jedną komórkę w dół z prawdopodobieństwem dopełniającym.
    """

    def __init__(
        self,
        width: int = 450,
        length: int = 750,
        delta: float = 0.012,
        epsilon: float = 0.87,
        initial_probability: float = 0.50,
        seed: int | None = 42,
    ) -> None:
        if width < 3:
            raise ValueError("width musi wynosić co najmniej 3 komórki")
        if length < 1:
            raise ValueError("length musi być dodatnie")
        if not 0 < delta <= 1:
            raise ValueError("delta musi należeć do przedziału (0, 1]")
        if not 0 < epsilon <= 1:
            raise ValueError("epsilon musi należeć do przedziału (0, 1]")
        if not 0 <= initial_probability <= 1:
            raise ValueError("initial_probability musi należeć do przedziału [0, 1]")

        self.width = width
        self.length = length
        self.delta = delta
        self.epsilon = epsilon
        self.initial_probability = initial_probability
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.skiers: list[ZhangSkier] = []

    def add_skier(
        self,
        x: int | None = None,
        direction: TurnDirection | None = None,
    ) -> ZhangSkier:
        """Dodaje narciarza na górnej krawędzi, jak w opisie artykułu.

        Gdy ``x`` nie podano, pozycja startowa jest losowana jednostajnie.
        Początkowa strona skrętu jest także losowa; można ją jawnie ustawić,
        aby odtworzyć eksperymenty inicjowane stanem ``left`` z artykułu.
        """
        if x is None:
            x = int(self.rng.integers(0, self.width))
        if not 0 <= x < self.width:
            raise ValueError(f"x musi należeć do [0, {self.width - 1}]")
        if direction is None:
            direction = "right" if self.rng.random() < 0.5 else "left"
        if direction not in ("left", "right"):
            raise ValueError("direction musi mieć wartość 'left' albo 'right'")

        skier = ZhangSkier(
            skier_id=len(self.skiers) + 1,
            x=x,
            y=0,
            direction=direction,
            p_left=self.initial_probability,
            p_right=self.initial_probability,
        )
        self.skiers.append(skier)
        return skier

    def _move_down(self, skier: ZhangSkier) -> None:
        """Wykonuje ruch w dół i przygotowuje *przeciwny* skręt.

        Jest to kluczowy szczegół pseudokodu z artykułu: po zjeździe w dół
        w stanie ``right`` zwiększa się ``p_left`` (nie ``p_right``), a po
        osiągnięciu ``epsilon`` zmienia się pamięć na ``left``.  Symetrycznie
        zachowuje się stan ``left``.
        """
        skier.y += 1
        if skier.direction == "right":
            skier.p_left = min(skier.p_left * (1.0 + self.delta), self.epsilon)
            if skier.p_left >= self.epsilon:
                skier.direction = "left"
        else:
            skier.p_right = min(skier.p_right * (1.0 + self.delta), self.epsilon)
            if skier.p_right >= self.epsilon:
                skier.direction = "right"

    def _step_skier(self, skier: ZhangSkier) -> None:
        if skier.status != "active":
            return
        if skier.y >= self.length:
            skier.status = "finished"
            return

        probability = skier.active_probability
        lateral = self.rng.random() < probability
        step_x = 1 if skier.direction == "right" else -1

        # Granice są wyłącznie warunkiem brzegowym skończonego obszaru
        # wizualizacji. Nie występują w pseudokodzie dla swobodnej trajektorii:
        # niedozwolony krok boczny zastępujemy ruchem w dół.
        if lateral and 0 <= skier.x + step_x < self.width:
            skier.x += step_x
            if skier.direction == "right":
                skier.p_right *= 1.0 - self.delta
                move: MoveKind = "right"
            else:
                skier.p_left *= 1.0 - self.delta
                move = "left"
        else:
            self._move_down(skier)
            move = "down"

        skier.move_history.append(move)
        skier.trajectory.append((skier.x, skier.y))
        skier.probability_history.append(skier.active_probability)
        skier.direction_history.append(skier.direction == "right")
        if skier.y >= self.length:
            skier.status = "finished"

    def step(self) -> None:
        """Wykonuje jeden dyskretny krok czasu dla wszystkich narciarzy."""
        for skier in self.skiers:
            self._step_skier(skier)
        self.step_count += 1

    @property
    def active_skiers(self) -> int:
        return sum(skier.status == "active" for skier in self.skiers)

    def run(self, max_steps: int = 10_000) -> int:
        """Uruchamia model aż wszyscy dojadą do końca lub limitu kroków."""
        for _ in range(max_steps):
            if self.active_skiers == 0:
                break
            self.step()
        return self.step_count

    def summary(self) -> dict[str, int | float]:
        """Zwraca małe, łatwe do serializacji podsumowanie przebiegu."""
        finished = sum(skier.status == "finished" for skier in self.skiers)
        lateral_moves = sum(
            move != "down" for skier in self.skiers for move in skier.move_history
        )
        total_moves = sum(len(skier.move_history) for skier in self.skiers)
        return {
            "steps": self.step_count,
            "skiers": len(self.skiers),
            "finished": finished,
            "active": self.active_skiers,
            "lateral_share": lateral_moves / total_moves if total_moves else 0.0,
        }
