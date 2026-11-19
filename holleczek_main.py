"""Główny punkt wejściowy dla modelu fizycznego Holleczek & Tröster (ETH Zurich 2012)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from skiing_simulation.holleczek.cli import main

if __name__ == "__main__":
    main()
