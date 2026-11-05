"""
Punkt wejścia do symulacji ruchu narciarzy zgodny z:
Thomas Holleczek & Gerhard Tröster (ETH Zurich),
"Particle-based model for skiing traffic", Phys. Rev. E 85, 056101 (2012).
"""
import sys
import os

# Upewnij się, że katalog 'src' jest w ścieżce wyszukiwania modułów
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from skiing_simulation import main

if __name__ == "__main__":
    main()

