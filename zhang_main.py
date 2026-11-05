"""Osobny punkt wejściowy dla modelu probabilistycznego Zhang i in. (2024)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from skiing_simulation.zhang.cli import main


if __name__ == "__main__":
    main()
