"""Główny pakiet symulacji narciarskich (Skiing Simulation).

Zawiera implementacje trzech kluczowych modeli ruchu narciarzy:
1. holleczek - model dynamiki cząsteczkowej (Holleczek & Tröster 2012, ETH Zurich)
2. zhang     - model automatu komórkowego z rozkładami prawdopodobieństwa (Zhang i in. 2024)
3. kaminski  - stochastyczny model z adaptacyjnym sąsiedztwem i stygmergią (Kamiński 2026, AGH)
"""

from skiing_simulation.holleczek.cli import main

__all__ = ["main", "holleczek", "zhang", "kaminski"]
