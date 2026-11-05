# Model Trafiku Narciarskiego – Dokumentacja Implementacji

Niniejszy projekt stanowi kompletną, zweryfikowaną implementację modelu ruchu narciarzy na stoku w oparciu o artykuł naukowy:

> **Thomas Holleczek & Gerhard Tröster (ETH Zurich)**  
> *"Particle-based model for skiing traffic"*  
> **Physical Review E 85, 056101 (2012)**  
> DOI: [10.1103/PhysRevE.85.056101](https://doi.org/10.1103/PhysRevE.85.056101)

---

## 1. Wprowadzenie i Koncepcja Modelu

Model opisuje zjazd narciarzy jako **dwuwymiarowy, mikroskopowy układ cząstek z masą** (*microscopic-driven many-particle system*). Kluczową cechą pracy Holleczka i Tröstera jest **ścisły podział na dwie fazy**:

1. **Sfera społeczna / decyzyjna (Sec. II.C)**:  
   Narciarz podlega bezwymiarowym siłom społecznym $\vec{F}_{social}$ (przyciąganie do celu, unikanie brzegów stoku, unikanie przeszkód i innych narciarzy). Siły te **nie przyspieszają fizycznie narciarza**, lecz wyznaczają jego **pożądany kierunek jazdy** $\vec{e}_{social}$.
2. **Kryterium skrętu (Sec. II.B, Fig. 3)**:  
   Jeśli kąt między aktualnym wektorem prędkości $\vec{e}_{\dot{r}}$ a kierunkiem pożądanym $\vec{e}_{social}$ przekracza próg $\delta = 10^\circ$, narciarz wchodzi w **skręt cięty (carving)** o stałym promieniu taliowania nart $R_{sc} = 10\text{ m}$. W przeciwnym razie kontynuuje jazdę na wprost.
3. **Sfera fizyczna (Sec. II.D)**:  
   Ruch w każdym kroku czasowym $\Delta t = 10\text{ ms}$ wynika z działania rzeczywistych sił mechanicznych w niutonach $[N]$: grawitacji wzdłuż stoku $\vec{F}_S$, oporu powietrza $\vec{F}_{air}$, siły dośrodkowej $\vec{F}_C$ oraz tarcia kinetycznego nart o śnieg $\vec{F}_{GROUND}$.

---

## 2. Architektura Kodu i Struktura Projektu

Projekt został zaimplementowany w Pythonie 3 przy użyciu biblioteki agentowej **Mesa 3.0**, **NumPy**, **SciPy** oraz **Matplotlib**:

```text
skiing-simulation/
├── HOLLECZEK.md                     # Niniejsza dokumentacja
├── main.py                          # Główny skrót startowy CLI
├── pyproject.toml                   # Konfiguracja środowiska (uv / pip)
├── data/
│   └── simulation_result.png        # Generowane wykresy trajektorii i telemetrii
└── src/
    └── skiing_simulation/
        ├── __init__.py              # Moduł główny CLI i analiza statystyczna
        ├── agents/
        │   └── base_skier.py        # Klasa BaseSkier (Mesa Agent) – siły społeczne i fizyka
        ├── slope/
        │   ├── terrain.py           # SlopeTerrain – model terenu 2.5D DEM, gradienty, krawędzie
        │   └── SlopeModel.py        # SkiModel (Mesa Model) – przestrzeń ciągła ContinuousSpace
        └── visualize/
            └── visualizer.py        # SlopeVisualizer – widok z góry + telemetria wieloagentowa
```

---

## 3. Szczegóły Matematyczne i Fizyczne

### 3.1. Faza 1: Siły Społeczne (Sec. II.C, Eq. 1–13)
Wszystkie siły społeczne są bezwymiarowe:

* **Siła celu $\vec{F}_D$ (Eq. 1, 2)**:
  $$\vec{F}_D(\vec{r}_a, t) = A_0 \frac{\vec{x}_a^k - \vec{r}_a}{\|\vec{x}_a^k - \vec{r}_a\|}$$
  Gdzie $A_0 = 1$ (Table II), a punkty docelowe $\vec{x}_a^k$ są generowane losowo co $50\text{ m}$ wzdłuż stoku z rozkładu jednostajnego na szerokości trasy (Sec. IV.B).
* **Odpychanie od krawędzi stoku $\vec{F}_L, \vec{F}_R$ (Eq. 3, 9, 10, 30, 31)**:
  $$\vec{f}_L = \frac{U_0}{R_L} \exp\left(-\frac{d_L}{R_L}\right) \hat{n}_L, \quad \vec{F}_L = w(\vec{e}_{\dot{r}}, \vec{e}_{brzeg}) \vec{f}_L$$
  Z parametrami $U_0 = 1$, $R_L = R_R = 50\text{ m}$ (Table II). Funkcja $w$ uwzględnia kąt widzenia narciarza $2\varphi = 180^\circ$ (Eq. 8).
* **Odpychanie od przeszkód $\vec{F}_O$ (Eq. 6, 12)**:
  Wykładniczy spadek siły w pobliżu armatek, słupów wyciągu i drzew.
* **Odpychanie od innych narciarzy $\vec{F}_A$ (Eq. 4, 5, 11, 32)**:
  Oparte na eliptycznym potencjale Helbinga z czasem relaksacji $\tau = 0.5\text{ s}$, $V_0 = 100$, $R_A = 2\text{ m}$:
  $$2s = \sqrt{(\|\vec{r}_{ab}\| + \|\vec{r}_{ab} - v_b \tau \vec{e}_b\|)^2 - (v_b \tau)^2}$$
* **Wersor pożądanego kierunku $\vec{e}_{social}$ (Eq. 13)**:
  $$\vec{F}_{social} = \vec{F}_D + \vec{F}_L + \vec{F}_R + \vec{F}_O + \sum_b \vec{F}_{A,b}, \quad \vec{e}_{social} = \frac{\vec{F}_{social}}{\|\vec{F}_{social}\|}$$

### 3.2. Kryterium Skrętu (Sec. II.B, Fig. 3)
Narciarz porównuje wektor bieżącego ruchu $\vec{e}_{\dot{r}}$ z wersorem pożądanym $\vec{e}_{social}$:
$$\cos\theta = \vec{e}_{\dot{r}} \cdot \vec{e}_{social}$$
* Jeśli $\cos\theta < \cos(10^\circ)$ ($\theta > 10^\circ$) $\implies$ **skręt cięty** (`is_turning = True`).
* W przeciwnym razie $\implies$ **jazda na wprost** (`is_turning = False`).

### 3.3. Faza 2: Siły Fizyczne (Sec. II.D, Eq. 14–29)
* **Kąt nachylenia stoku $\alpha$ (Eq. 14)** oraz linia spadku $\hat{u}_{fall}$:
  Wyznaczane z lokalnego gradientu wysokości DEM $\nabla Z(x, y)$.
* **Siła grawitacji wzdłuż stoku $\vec{F}_S$ (Eq. 18)**:
  $$\vec{F}_S = m g \sin\alpha \cdot \hat{u}_{fall}$$
* **Rozkład na składową wzdłuż toru $\vec{F}_P$ i poprzeczną $\vec{F}_{lat}$ (Eq. 19, 20)**:
  $$\vec{F}_P = (\vec{F}_S \cdot \vec{e}_{\dot{r}}) \vec{e}_{\dot{r}}, \quad \vec{F}_{lat} = \vec{F}_S - \vec{F}_P$$
* **Opór powietrza $\vec{F}_{air}$ (Eq. 25)**:
  $$\vec{F}_{air} = -\frac{1}{2} C_d A \rho v^2 \vec{e}_{\dot{r}}$$
  Z wartościami z Table III & IV: $C_d = 1.0$, $A = 0.6\text{ m}^2$, $\rho = 1.3163\text{ kg/m}^3$ (powietrze przy $-5^\circ\text{C}$).
* **Siła nacisku normalnego $F_N$ (Eq. 17)**:
  $$F_N = m g \cos\alpha$$

#### Przypadek A: Skręt cięty (`is_turning = True`)
1. **Siła dośrodkowa $\vec{F}_C$ (Eq. 22)**:
   $$\vec{F}_C = \frac{m v^2}{R_{sc}} \hat{u}_{turn}$$
   gdzie promień taliowania nart wynosi $R_{sc} = 10\text{ m}$ (Table IV).
2. **Siła boczna $\vec{F}_{LAT}$ (Eq. 21)**:
   $$\vec{F}_{LAT} = \vec{F}_{lat} - \vec{F}_C$$
3. **Efektywna siła docisku krawędzi $\|\vec{F}_{EFF}\|$ (Eq. 24)**:
   $$\|\vec{F}_{EFF}\| = \sqrt{\|\vec{F}_{LAT}\|^2 + F_N^2}$$
4. **Tarcie kinetyczne nart o śnieg w skręcie $\vec{F}_{GROUND}$ (Eq. 27)**:
   $$\vec{F}_{GROUND} = -\mu \|\vec{F}_{EFF}\| \vec{e}_{\dot{r}}$$
   > **Kluczowy mechanizm stabilizacji prędkości:**  
   > W trakcie skrętu $\vec{F}_C \propto v^2$ dociska krawędzie do śniegu, gwałtownie podnosząc $\|\vec{F}_{EFF}\|$. W efekcie siła tarcia $\vec{F}_{GROUND}$ rośnie proporcjonalnie do $v^2$, naturalnie ograniczając prędkość narciarza do fizjologicznego poziomu ~50–55 km/h.
5. **Wypadkowa siła $\vec{F}_{NET}$ (Eq. 29)**:
   $$\vec{F}_{NET} = \vec{F}_P + \vec{F}_{air} + \vec{F}_{GROUND} + \vec{F}_C$$

#### Przypadek B: Jazda na wprost (`is_turning = False`)
* $\|\vec{F}_{eff}\| = \sqrt{\|\vec{F}_{lat}\|^2 + F_N^2}$ (Eq. 23)
* $\vec{F}_{ground} = -\mu \|\vec{F}_{eff}\| \vec{e}_{\dot{r}}$ (Eq. 26)
* $\vec{F}_{net} = \vec{F}_P + \vec{F}_{air} + \vec{F}_{ground}$ (Eq. 28)

### 3.4. Reguły Dodatkowe i Całkowanie
* **Krok czasowy (Sec. IV.A)**: $\Delta t = 0.01\text{ s}$ ($10\text{ ms}$).
* **Reguła przeciwstoku (Sec. II.B)**: Jeśli prędkość narciarza spadnie poniżej $v_{min} = 5\text{ km/h}$ (np. na wzniesieniu Jochstock), narciarz przechodzi w marsz pod górę ze stałą prędkością $v_{min}$ w pożądanym kierunku $\vec{e}_{social}$.
* **Odbicie od krawędzi (Sec. II.B)**: Przy uderzeniu w brzeg stoku narciarz odbija się sprężyście w stronę środka trasy ($v_x = -|v_x|$ lub $+|v_x|$).

---

## 4. Topografia Stoków (Engelberg, Szwajcaria – Table I)

Model zawiera parametry dwóch rzeczywistych tras ze szwajcarskiego ośrodka Engelberg:

| Parametr | Jochstock (pośredni) | Graustock (ekspert) |
| :--- | :---: | :---: |
| Długość trasy | $889\text{ m}$ | $1250\text{ m}$ |
| Średnia szerokość | $62\text{ m}$ (start 38m, środek 74m) | $81\text{ m}$ (start 50m, meta 95m) |
| Wysokość startu / mety | $2504\text{ m}$ / $2216\text{ m}$ | $2178\text{ m}$ / $1765\text{ m}$ |
| Różnica wysokości (spadek) | $288\text{ m}$ | $413\text{ m}$ |
| Średnie nachylenie | $32\%$ ($17.7^\circ$) | $33\%$ ($18.3^\circ$) |
| Specyfika profilu | **Przeciwstok na $y \in [510, 550]\text{ m}$** | Stroma ścianka górna, płaski dół |
| Kalibracja tarcia $\mu$ (Sec. V) | **$\mu = 0.10$** | **$\mu = 0.125$** |

### 4.1. Interpolacja Profilu Terenu: PCHIP a Efekt Rungego
Profil wysokościowy $Z(y)$ stoku jest definiowany przez zbiór punktów węzłowych ze szwajcarskich danych topograficznych (np. szczyt, stroma ścianka, punkt wejścia w przeciwstok, szczyt przeciwstoku, meta). Aby wyznaczyć wysokość $Z$ oraz kąt nachylenia $\alpha$ dla każdego metra stoku, konieczna jest interpolacja:

* **Problem zwykłych wielomianów i efekt Rungego**:
  Gdybyśmy użyli klasycznego wielomianu wysokiego stopnia (np. Lagrange'a), w pobliżu krańców przedziału oraz w miejscach gwałtownych zmian spadku pojawiłyby się potężne, dzikie oscylacje (**efekt Rungego**). Zamiast płynnego zjazdu powstałyby niefizyczne fale i sztuczne „skocznie”.
* **Problem tradycyjnego splajnu sześciennego (`CubicSpline`)**:
  Standardowy naturalny splajn sześcienny wymusza ciągłość drugiej pochodnej ($C^2$). Niestety cierpi na tzw. **przeregulowanie (overshoot)** – przy gwałtownym wypłaszczeniu stoku lub wzniesieniu przeciwstoku ($+1.5\text{ m}$) potrafi „wybić” w górę lub „zapaść się” w dół, tworząc niefizyczne doły o nachyleniu 40–50%.
* **Rozwiązanie: Splajn hermitowski PCHIP (`scipy.interpolate.PchipInterpolator`)**:
  **PCHIP** (*Piecewise Cubic Hermite Interpolating Polynomial*) to interpolacja kawałkami sześcienna **ściśle zachowująca monotoniczność (shape-preserving)**:
  1. Jeśli między dwoma punktami węzłowymi wysokość maleje, interpolowana krzywa **gwarantuje brak jakichkolwiek lokalnych górek czy dołków**.
  2. Na szczycie przeciwstoku pochodna jest gładko i dokładnie zerowana bez oscylacji bocznych.
  3. Krzywa jest klasy $C^1$ (ma ciągłą pierwszą pochodną), dzięki czemu kąt nachylenia stoku $\alpha(y)$ oraz siła grawitacji $F_S = mg\sin\alpha$ zmieniają się pod nartami gładko i bez szarpnięć numerycznych.

---

## 5. Jak Korzystać z Projektu?

### 5.1. Wymagania i Środowisko
Projekt korzysta ze standardowego środowiska Python (>=3.10) zarządzanego przez `uv` lub tradycyjny `venv`:
```powershell
# Instalacja zależności przez uv:
uv sync

# LUB tradycyjny pip:
pip install -r pyproject.toml
```

### 5.2. Uruchamianie z Poziomu Konsoli (CLI)

Głównym punktem startowym jest skrypt **`main.py`** w katalogu głównym projektu:

```powershell
# 1. Standardowa szybka symulacja liczbowa z podsumowaniem statystycznym i wykresem PNG:
python main.py

# 2. Uruchomienie interaktywnej animacji zjazdu na żywo:
python main.py --animate

# 3. Zjazd wielu narciarzy (np. 3) z przyspieszeniem animacji 8x:
python main.py --animate --skiers 3 --speedup 8

# 4. Symulacja na stoku Graustock ze skalibrowanym tarciem mu = 0.125:
python main.py --animate --slope graustock --mu 0.125

# 5. Zapis animacji do pliku GIF:
python main.py --gif data/zjazd.gif --speedup 10
```

### 5.3. Dostępne Argumenty Linii Poleceń

| Argument | Typ | Domyślnie | Opis |
| :--- | :---: | :---: | :--- |
| `--slope` | `str` | `"jochstock"` | Wybór stoku: `"jochstock"` lub `"graustock"` (Table I) |
| `--skiers` | `int` | `1` | Liczba narciarzy startujących na stoku |
| `--animate` | flaga | `False` | Otwiera okno interaktywnej animacji Matplotlib |
| `--speedup` | `int` | `6` | Przyspieszenie animacji (liczba kroków fizyki $\Delta t$ na klatkę) |
| `--mu` | `float` | `0.10` | Współczynnik tarcia kinetycznego nart o śnieg (Table IV) |
| `--rsc` | `float` | `10.0` | Promień taliowania nart $R_{sc}$ w metrach (Table IV) |
| `--delta` | `float` | `10.0` | Kąt progowy inicjacji skrętu $\delta$ w stopniach (Table II) |
| `--dt` | `float` | `0.01` | Krok całkowania w sekundach ($10\text{ ms}$, Sec. IV.A) |
| `--gif` | `str` | `None` | Ścieżka do zapisu animacji GIF |
| `--output` | `str` | `"data/simulation_result.png"` | Ścieżka zapisu podsumowującego wykresu PNG |

### 5.4. Sterowanie Animacją (Skróty Klawiszowe)
Podczas wyświetlania okna animacji dostępne są interaktywne klawisze:
* **Spacja (`Space`)**: Pauza / Wznowienie symulacji.
* **Plus (`+` / `=`)**: Przyspieszenie tempa animacji (zwiększenie kroków na klatkę, do 40x).
* **Minus (`-`)**: Spowolnienie tempa animacji (do 1x).

---

## 6. Wizualizacja i Telemetria

W oknie animacji oraz na wygenerowanym wykresie `data/simulation_result.png` wyświetlane są 4 zsynchronizowane panele:

1. **Rzut stoku z góry (Top-Down 2D)**:
   * Kolorowe poziomice wysokości,
   * Czerwone przerywane linie brzegowe stoku (o zmiennej szerokości wg Table I),
   * **Każdy narciarz posiada własny, unikalny kolor** z palety `SKIER_COLORS`,
   * Punkty nawigacyjne (diamenty) w kolorze danego narciarza,
   * Ciągły ślad trajektorii i dynamiczne markery pozycji.
2. **Wykres prędkości $v(y)$ [km/h]**:
   * Osobne krzywe prędkości w czasie rzeczywistym dla każdego narciarza,
   * Pozioma linia odniesienia $\approx 54\text{ km/h}$ z artykułu.
3. **Wykres współczynnika tarcia $\mu(y)$**:
   * Bieżący odczyt tarcia pod nartami każdego zawodnika (bazowe $\mu = 0.10$, spadek do $\mu = 0.04$ w strefach oblodzenia).
4. **Profil wysokościowy $Z(y)$ [m n.p.m.]**:
   * Ukształtowanie pionowe stoku z wyraźnie widocznym wypłaszczeniem i przeciwstokiem.

---

## 7. Weryfikacja Wyników z Artykułem

Wyniki wygenerowane przez naszą symulację wykazują niemal idealną zgodność z wynikami podanymi w publikacji Holleczek & Tröster (2012):

| Metryka | Nasza Symulacja (Jochstock) | Artykuł – Symulacja | Artykuł – Eksperyment GPS |
| :--- | :---: | :---: | :---: |
| **Średnia prędkość** | **$55.1\text{ – }57.0\text{ km/h}$** | **$55.2\text{ km/h}$** (Sec. IV.C) | **$51.8\text{ km/h}$** (Sec. III.B) |
| **Mediana $p_{50}$** | **$61.7\text{ – }63.2\text{ km/h}$** | **$\approx 54\text{ – }60\text{ km/h}$** (Fig. 12b) | **$\approx 54\text{ km/h}$** (Fig. 12b) |
| **Kwartyle $p_{25} - p_{75}$** | **$39.1\text{ – }68.3\text{ km/h}$** | **$45.9\text{ – }67.8\text{ km/h}$** (Fig. 12b) | **$42.0\text{ – }65.0\text{ km/h}$** |
| **Spadek na przeciwstoku** | **do $25\text{ – }35\text{ km/h}$** | **do $\approx 25\text{ km/h}$** (Fig. 8b, 10b) | **do $\approx 20\text{ km/h}$** (Fig. 8b) |
| **Częstość skrętów** | **$1.5\text{ – }3.2\text{ skrętów/100m}$** | **$3.6\text{ skrętów/100m}$** (Sec. IV.E) | **$2.0\text{ skrętów/100m}$** (Sec. IV.E) |

---

## 8. Użycie jako Biblioteka w Pythonie

Model można zaimportować i wykorzystać we własnych skryptach badawczych:

```python
from skiing_simulation.slope.SlopeModel import SkiModel
from skiing_simulation.visualize.visualizer import SlopeVisualizer

# 1. Utworzenie stoku (np. Jochstock)
model = SkiModel(slope_name="jochstock", dt=0.01)

# 2. Dodanie narciarzy o różnych parametrach
skier1 = model.add_skier(x=25.0, y=5.0, mass=85.0, sidecut_radius=10.0)
skier2 = model.add_skier(x=38.0, y=5.0, mass=72.0, sidecut_radius=12.0)

# 3. Krok symulacji lub pełny przebieg
for step in range(10000):
    model.step()
    stats = model.get_summary_stats()
    if stats["active"] == 0:
        break

# 4. Zapis wykresu
viz = SlopeVisualizer(model)
viz.render_static_result("moj_wynik.png")
```

