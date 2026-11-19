# Model probabilistyczny Zhang i in. (2024)

Ten katalog zawiera **osobną** implementację automatu komórkowego z artykułu:

> B. Zhang, T. Dressler, A. Maurer, M. Nader, A. Schadschneider,
> *Simulation of Downhill Skiing Areas*, Collective Dynamics 9, A166:1–7 (2024),
> DOI: 10.17815/CD.2024.166.

Nie modyfikuje ona modelu siłowego Holleczka i Tröstera. Oba modele mogą być uruchamiane niezależnie: dotychczasowy `main.py` pozostaje punktem wejścia modelu Holleczka, a `zhang_main.py` uruchamia automat Zhang i in.

## Na czym polega model

Stok jest prostokątną siatką komórek, a współrzędna `y` rośnie w kierunku zjazdu. Każdy narciarz ma:

- pozycję `(x, y)`;
- pamięć aktualnego skrętu: `left` albo `right`;
- dwa pamiętane prawdopodobieństwa `p_left` i `p_right`.

W pojedynczym kroku automat wybiera dokładnie jeden ruch. Dla stanu `right` wykonuje ruch o jedną komórkę w prawo z prawdopodobieństwem `p_right`; w przeciwnym przypadku przesuwa się o jedną komórkę w dół. Stan `left` działa symetrycznie. Ruch pod górę nie występuje.

Po ruchu bocznym prawdopodobieństwo kontynuacji aktualnego skrętu maleje:

```text
p <- p * (1 - δ)
```

Po ruchu w dół rośnie prawdopodobieństwo **przeciwnego** skrętu aż do progu `ε`:

```text
p_opposite <- min(p_opposite * (1 + δ), ε)
```

Po osiągnięciu `ε` pamięć skrętu zmienia stronę. Implementacja zachowuje oba `p_left` i `p_right` między zmianami strony — zgodnie z pseudokodem w sekcji 4 publikacji, który nie przewiduje ich resetowania. Ta pamięć jest źródłem bezwładności oraz nieregularnych łuków trajektorii. W obecnej postaci artykuł analizuje małą gęstość, więc narciarze są niezależni: model nie zawiera unikania kolizji, sił ani fizycznej prędkości.

`δ` i `ε` muszą należeć do `(0, 1]`. Większe `δ` szybciej zmienia prawdopodobieństwo; `ε` wyznacza poziom, przy którym następuje zmiana strony skrętu. Domyślny profil demonstracyjny **`δ=0.012`, `ε=0.87`** został dobrany pomiędzy wariantami z rys. 2, aby łuki pozostały szerokie, ale nieregularne. Najbliższa kalibracja pokazana bezpośrednio w artykule to `δ=0.01`, `ε=0.86`; autorzy porównują też `ε=0.84` i `0.88` (przy `δ=0.01`) oraz `δ=0.006` i `0.014` (przy `ε=0.86`).

## Uruchamianie

Z katalogu projektu:

```bash
# Pełny przebieg i PNG z trajektorią oraz stanem automatu
.venv/bin/python zhang_main.py

# Wiele niezależnych narciarzy z powtarzalnym losowaniem
.venv/bin/python zhang_main.py --skiers 4 --seed 2024

# GIF; ustalony początkowy skręt ułatwia porównywanie przebiegów
.venv/bin/python zhang_main.py --initial-direction left \
  --gif data/moj_zjazd_zhang.gif --steps-per-frame 3

# Zainstalowany skrypt pakietu (po `uv sync`)
uv run skiing-zhang-simulation --delta 0.10 --epsilon 0.75
```

Najważniejsze opcje:

| Opcja | Domyślnie | Znaczenie |
| --- | ---: | --- |
| `--width` | `450` | szerokość symulowanego stoku w komórkach |
| `--length` | `750` | długość stoku w komórkach |
| `--skiers` | `1` | liczba niezależnych narciarzy |
| `--delta` | `0.012` | względna zmiana prawdopodobieństwa `δ` — profil demonstracyjny |
| `--epsilon` | `0.87` | próg zmiany strony skrętu `ε` — profil demonstracyjny |
| `--seed` | `42` | ziarno RNG — identyczne dane dają identyczny przebieg |
| `--initial-direction` | `random` | `left`, `right` albo losowo |
| `--start-margin` | `125` | margines od boków przy losowaniu pozycji startowej |
| `--gif` | — | zapis animacji GIF |
| `--output` | `data/zhang_simulation_result.png` | zapis statycznego podsumowania |

## Struktura implementacji

```text
src/skiing_simulation/zhang/
├── model.py       # ZhangModel i ZhangSkier — reguła CA oraz RNG
├── visualizer.py  # PNG i animacja GIF
└── cli.py         # opcje konsolowe
zhang_main.py      # wygodny, niezależny punkt startowy
```

`ZhangModel` można również wykorzystać bez CLI:

```python
from skiing_simulation.zhang import ZhangModel, ZhangVisualizer

model = ZhangModel(width=450, length=750, delta=0.012, epsilon=0.87, seed=42)
model.add_skier(x=50, direction="left")
model.run()
print(model.summary())
ZhangVisualizer(model).render_static_result("data/wynik.png")
```

## Granice i artefakty w `data/`

Pseudokod z artykułu opisuje pojedynczą trajektorię i nie definiuje bocznych granic skończonego kadru. Dla wizualizacji zastosowano prosty warunek brzegowy: niedozwolony krok poza `x=0` lub `x=width-1` staje się krokiem w dół i aktualizuje pamięć dokładnie jak normalny ruch w dół. Jest to tylko zabezpieczenie skończonego obszaru renderowania, nie dodatkowy mechanizm zachowania narciarza.

W `data/` znajdują się wygenerowane artefakty modelu Zhang odpowiadające istniejącym przykładom: `zhang_single_skier.gif`, `zhang_simulation_demo.gif` oraz `zhang_simulation_result.png`. Panele animacji przedstawiają trajektorię, aktywne prawdopodobieństwo kroku bocznego, `x(y)` i stan pamięci skrętu.
