# Skiing Simulation Framework

Środowisko symulacji dynamiki ruchu narciarzy na stoku.
Repozytorium zawiera spójną, modułową implementację trzech kluczowych modeli naukowych:

1. **Holleczek & Tröster (ETH Zurich, 2012)** – Model cząsteczkowy oparty na klasycznej mechanice Newtona, siłach społecznych Helbinga i geometrii carvingu (*Phys. Rev. E 85, 056101*).
2. **Zhang i in. (2024)** – Model stochastycznego automatu komórkowego z rozkładami prawdopodobieństwa w sąsiedztwie komórkowym.
3. **Kamiński (AGH, 2026)** – Stochastyczny model wieloagentowy z prędkościowym sąsiedztwem $(2v+1) \times (v+1)$, skośnym rozkładem Gaussa, anizotropowymi stożkami kolizji, sterowalnością pędem ($\beta$) oraz dynamiczną stygmergią stoku (powstawanie lodu i muld).

---

## 📁 Czysta i modułowa struktura projektu

Kod, dokumentacja oraz wyniki (wykresy i animacje) są **ściśle posegregowane według modeli**:

```text
skiing-simulation/
├── docs/                             # Dokumentacja naukowa każdego modelu
│   ├── HOLLECZEK.md                  # Szczegóły matematyczne modelu Holleczka
│   ├── Zhang.md                      # Szczegóły modelu Zhanga
│   └── KAMINSKI.md                   # Szczegóły modelu Kamińskiego
│
├── data/                             # Dane wyjściowe (wykresy PNG i animacje GIF)
│   ├── holleczek/                    # Wykresy i GIF-y modelu Holleczka
│   ├── zhang/                        # Wykresy i GIF-y modelu Zhanga
│   └── kaminski/                     # Wykresy i GIF-y modelu Kamińskiego
│
├── src/skiing_simulation/            # Pakiety źródłowe modeli
│   ├── holleczek/                    # [Pakiet 1] agent.py, model.py, terrain.py, visualizer.py, cli.py
│   ├── zhang/                        # [Pakiet 2] model.py, visualizer.py, cli.py
│   └── kaminski/                     # [Pakiet 3] agent.py, model.py, terrain.py, visualizer.py, distributions.py, cli.py
│
├── holleczek_main.py                 # Skrót uruchomieniowy dla modelu Holleczka
├── zhang_main.py                     # Skrót uruchomieniowy dla modelu Zhanga
├── kaminski_main.py                  # Skrót uruchomieniowy dla modelu Kamińskiego
├── main.py                           # Domyślny punkt startowy
└── pyproject.toml                    # Konfiguracja środowiska i skrótów CLI
```

---

## 🚀 Uruchamianie modeli

Dla każdego modelu przygotowano dedykowany skrypt oraz komendę CLI w `pyproject.toml`:

### 1. Model Holleczka (ETH Zurich 2012)
* Dokumentacja: [`docs/HOLLECZEK.md`](docs/HOLLECZEK.md)
* Uruchomienie:
  ```bash
  uv run python holleczek_main.py --skiers 4 --slope jochstock
  # lub
  uv run skiing-holleczek-simulation --skiers 4
  ```
* Wyniki trafiają do: `data/holleczek/`

### 2. Model Zhanga (2024)
* Dokumentacja: [`docs/Zhang.md`](docs/Zhang.md)
* Uruchomienie:
  ```bash
  uv run python zhang_main.py --skiers 4 --steps 60
  # lub
  uv run skiing-zhang-simulation --skiers 4
  ```
* Wyniki trafiają do: `data/zhang/`

### 3. Model Kamińskiego (AGH 2026)
* Dokumentacja: [`docs/KAMINSKI.md`](docs/KAMINSKI.md)
* Uruchomienie:
  ```bash
  uv run python kaminski_main.py --skiers 6 --gif data/kaminski/kaminski_simulation_demo.gif
  # lub
  uv run skiing-kaminski-simulation --skiers 16
  ```
* Wyniki trafiają do: `data/kaminski/`