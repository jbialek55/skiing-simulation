# Skiing Simulation Framework

Środowisko wieloagentowych i cząsteczkowych symulacji dynamiki ruchu narciarzy na stoku.
Repozytorium zawiera spójną, modułową implementację trzech modeli:

---

## 📁 Architektura pakietów

Cały kod źródłowy znajduje się w katalogu `src/skiing_simulation/`, podzielonym na trzy dedykowane pakiety:

```text
src/skiing_simulation/
├── holleczek/             # Model dynamiki cząsteczkowej (Holleczek & Tröster 2012)
│   ├── agent.py           # Klasa BaseSkier – siły społeczne Helbinga, carving i pęd
│   ├── model.py           # Klasa SkiModel – przestrzeń ciągła ContinuousSpace
│   ├── terrain.py         # SlopeTerrain – profil 2.5D DEM, granice stoku, przeszkody
│   ├── visualizer.py      # SlopeVisualizer – trajektorie, widok z góry, animacje
│   └── cli.py             # Interfejs konsolowy CLI
│
├── zhang/                 # Model automatu komórkowego (Zhang i in. 2024)
│   ├── model.py           # ZhangModel, ZhangSkier – macierze prawdopodobieństwa
│   ├── visualizer.py      # ZhangVisualizer – widok siatki, ruchy boczne
│   └── cli.py             # Interfejs konsolowy CLI
│
└── kaminski/              # Model stochastyczny z adaptacyjnym otoczeniem (Kamiński 2026)
    ├── agent.py           # Skier – style jazdy, stan awaryjny, pęd beta
    ├── model.py           # SkiModel – anizotropowa kolizja, detekcja zderzeń
    ├── distributions.py   # Rozkłady Skew-Normal, tabele faz zakrętów
    ├── terrain.py         # SlopeGrid25D – stygmergia stoku (lód, muldy, wyciąg)
    ├── visualizer.py      # KaminskiVisualizer – pulpit diagnostyczny i animacje GIF
    └── cli.py             # Interfejs konsolowy CLI
```

---

## 🚀 Uruchamianie modeli

Dla każdego modelu przygotowano wygodny skrypt w katalogu głównym oraz komendę CLI w `pyproject.toml`:

### 1. Model Holleczka (ETH Zurich 2012)
* Szczegółowy opis: [`HOLLECZEK.md`](HOLLECZEK.md)
* Uruchomienie:
  ```bash
  uv run python holleczek_main.py --skiers 4 --slope jochstock
  # lub
  uv run skiing-holleczek-simulation --skiers 4
  ```

### 2. Model Zhanga (2024)
* Szczegółowy opis: [`Zhang.md`](Zhang.md)
* Uruchomienie:
  ```bash
  uv run python zhang_main.py --skiers 4 --steps 60
  # lub
  uv run skiing-zhang-simulation --skiers 4
  ```

### 3. Model Kamińskiego (AGH 2026)
* Szczegółowy opis: [`KAMINSKI.md`](KAMINSKI.md)
* Uruchomienie:
  ```bash
  uv run python kaminski_main.py --skiers 6 --gif data/kaminski_simulation_demo.gif
  # lub
  uv run skiing-kaminski-simulation --skiers 16
  ```