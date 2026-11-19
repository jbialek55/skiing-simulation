# Stochastyczny Model Narciarski (Kamiński AGH 2026)

Dokumentacja implementacji modelu ruchu narciarzy na stoku w oparciu o rozprawę doktorską:

> **Filip Bernard Kamiński**  
> *"Skalowalny stochastyczny wieloagentowy model do symulacji ruchu narciarzy na stoku"*  
> **Promotor: prof. dr hab. inż. Jarosław Wąs**  
> **Wydział Elektrotechniki, Automatyki, Informatyki i Inżynierii Biomedycznej, AGH Kraków (2026)**  
> *Prezentowane również na: AAMAS 2026 (14th ATT Workshop)*

---

## 1. Koncepcja i Architektura Modelu

W odróżnieniu od modeli opartych na klasycznej mechanice Newtona (jak model Holleczka i Tröstera) oraz prostych automatów komórkowych operujących pojedynczą komórką (jak model Zhanga), model Kamińskiego reprezentuje **podejście quasi-minimalne z probabilistyczną funkcją przejścia**:

1. **Przestrzeń dyskretna 2.5D**:
   * Siatka regularna o komórkach $1\text{ m} \times 1\text{ m}$.
   * Każda komórka przechowuje wysokość $h$ [m n.p.m.] (wielomianowy profil Kasprowego Wierchu 5. stopnia).
   * Krok dyskretny czasu wynosi $\Delta t = 1.0\text{ s}$.
2. **Adaptacyjne sąsiedztwo zależne od prędkości**:
   * Agent o prędkości $v$ rozpatruje prostokątne otoczenie o wymiarach $(2v + 1) \times (v + 1)$ komórek.
   * Wybór komórki w sąsiedztwie jest równoznaczny z wektorem przemieszczenia $(\Delta x, \Delta y)$ w czasie $1\text{ s}$.
3. **Iloczyn Hadamarda (Mnożenie macierzowe element po elemencie)**:
   * Wypadkowy rozkład prawdopodobieństwa powstaje z połączenia intencji zakrętu oraz masek środowiskowych i społecznych:
     $$P = B \odot M_{\text{teren}} \odot M_{\text{kolizje}}$$
   * **Rozkład bazowy $B$**: dwuwymiarowy skośny rozkład Gaussa (Skew-Normal) oparty na empirycznych tabelach faz zakrętu $(\mu_x, \mu_y)$ oraz odpychaniu od brzegów $\gamma_x$.
   * **Maska terenu $M_{\text{teren}}$**: $0.0$ poza trasą (brak możliwości wypadnięcia), strefa buforowa $p_c = d/r$ zniechęcająca do jazdy przy bandzie.
   * **Maska kolizji $M_{\text{kolizje}}$**: dynamiczne koła bezpieczeństwa wokół innych narciarzy skalowane ich prędkościami ($d / v_k$).
4. **Quasi-fizyczna aktualizacja prędkości**:
   $$v_{t_n} = v_{t_{n-1}} + k_d \frac{\Delta h}{\Delta y} - k_h \frac{\Delta x}{\Delta y}$$
   gdzie $k_d = 0.10$ (przyspieszenie grawitacyjne ze spadku terenu), a $k_h = 0.80$ (wytracanie prędkości przy jeździe w poprzek stoku).
   Dodatkowo z prawdopodobieństwem $p_{zw} = 0.20$ następuje losowe zwolnienie o $1\text{ m/s}$ (analogia do modelu Nagela-Schreckenberga).

---

## 2. Struktura Kodu

```text
src/skiing_simulation/kaminski/
├── __init__.py          # Eksport klas i symboli
├── agent.py             # Klasa Skier (Mesa Agent) – stany zakrętów, cykl życia, telemetria
├── model.py             # Klasa SkiModel (Mesa Model, MultiGrid) – unikanie kolizji, zliczanie zderzeń
├── distributions.py     # Tabele zakrętów 4.1 & 4.2, skośny rozkład Gaussa, iloczyn Hadamarda
├── terrain.py           # Siatka 2.5D, wielomian 5. stopnia, granice stoku i bufory
├── visualizer.py        # 4-panelowy pulpit diagnostyczny (trajektorie, heatmapa kolizji, dx/dy)
└── cli.py               # Konsolowy interfejs CLI
kaminski_main.py         # Skrót uruchomieniowy w katalogu głównym
```

---

## 3. Uruchamianie Symulacji

Z katalogu głównego projektu:

```bash
# Podstawowy przebieg (16 narciarzy, 120 kroków):
.venv/bin/python kaminski_main.py

# Wygenerowanie animacji GIF z przejazdu:
.venv/bin/python kaminski_main.py --skiers 6 --steps 65 --gif data/kaminski_simulation_demo.gif

# Symulacja dużej grupy z analizą kolizji (50 narciarzy):
.venv/bin/python kaminski_main.py --skiers 50 --steps 120 --output data/kaminski_50_skiers.png

# Z poziomu zainstalowanego pakietu CLI:
uv run skiing-kaminski-simulation --skiers 25 --width 40 --length 350
```

### Opcje CLI:
| Opcja | Domyślnie | Opis |
| --- | ---: | --- |
| `--skiers` | `16` | Liczba narciarzy na stoku |
| `--width` | `40` | Szerokość stoku w metrach |
| `--length` | `350` | Długość stoku w metrach (odcinek Kasprowego Wierchu) |
| `--steps`, `--max-steps` | `None` | Opcjonalny limit kroków (domyślnie: działa aż wszyscy ukończą zjazd) |
| `--buffer-r` | `3` | Szerokość strefy buforowej przy brzegach trasy w metrach |
| `--seed` | `42` | Ziarno losowe determinizmu eksperymentu |
| `--output` | `data/kaminski_simulation_result.png` | Ścieżka wyjściowa dla wykresów |
| `--gif` | `None` | Opcjonalna ścieżka do zapisu animacji GIF |

---

## 4. Wykresy Diagnostyczne i Walidacja

Moduł `KaminskiVisualizer` generuje 4 panele graficzne:
1. **Trajektorie narciarzy**: Widok stoku 2.5D z lotu ptaka z korytarzem trasy i strefami wyłączonymi.
2. **Mapa kolizji**: Heatmapa rejestrująca punkty o podwyższonym ryzyku zderzeń (przypadki obecności >1 narciarza w komórce $1\text{ m} \times 1\text{ m}$).
3. **Prędkość w zakręcie a nachylenie**: Walidacja zależności prędkości chwilowej od stromości stoku (odpowiadająca Rysunkowi 6.13 z pracy doktorskiej).
4. **Stosunek $dx/dy$ a nachylenie**: Wskaźnik szerokości skrętów w funkcji nachylenia stoku (odpowiadający Rysunkowi 6.14 z pracy).
