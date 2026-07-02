# PyMinecraft

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![OpenGL](https://img.shields.io/badge/OpenGL-3.3-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

Voxelowy sandbox w Pythonie — własny silnik na **ModernGL + GLFW + NumPy**.
Nieskończony, proceduralny świat z biomami, jaskiniami, rudami, drzewami,
cyklem dnia i nocy, wodą, kopaniem i budowaniem. Zero zewnętrznych assetów —
wszystkie tekstury generowane proceduralnie przy starcie.

*A voxel sandbox game with a custom Python engine (ModernGL + GLFW + NumPy):
infinite procedural worlds, biomes, caves, day/night cycle, building and
mining. All textures are generated procedurally — no asset files.*

## Szybki start

```
git clone https://github.com/Marak123/PyMinecraft.git
cd PyMinecraft
py -m pip install -r requirements.txt
py launcher.py
```

Wymagania: Python 3.12+, karta z OpenGL 3.3 (czyli praktycznie każda).

## Sterowanie

| Klawisz | Akcja |
|---|---|
| `W A S D` | ruch |
| `Spacja` | skok / pływanie w górę / (w locie) w górę |
| `Lewy Ctrl` | sprint |
| `Lewy Shift` | (w locie) w dół |
| `F` | włącz/wyłącz latanie |
| `LPM` | zniszcz blok |
| `PPM` | postaw blok |
| `ŚPM` | pobierz wskazany blok do hotbara |
| `1–9` / kółko | wybór slotu hotbara |
| `F3` | statystyki (FPS, pozycja, chunki) |
| `F2` | screenshot do `screenshots/` |
| `ESC` | pauza / uwolnij mysz |

Świat zapisuje się automatycznie przy wyjściu (tylko zmodyfikowane chunki)
do `saves/world/`. Ustawienia (rozdzielczość, render distance, FOV, czułość
myszy, seed) w `configs/settings.json` — plik powstaje przy pierwszym starcie.

## Architektura

```
launcher.py          wejście
game/                warstwa gry: pętla, gracz, hotbar, HUD
engine/
  core/              config, logi, zegar, matematyka 3D
  window/  input/    GLFW, snapshot wejścia
  camera/            kamera FPS + frustum
  world/             bloki (data-driven), noise, generator, chunki,
                     streaming async, cykl dnia, zapis świata
  graphics/          mesher (NumPy, AO, kompresja wierzchołków do 8 B),
                     proceduralny atlas (texture array), shadery, renderer
  physics/           kolizje AABB, raycast DDA
configs/             blocks.json (definicje bloków), settings.json
tools/               smoke_test.py (render offscreen), logic_test.py
```

Zasady: silnik nie wie nic o gameplayu; każdy podsystem jest wymienialny;
bloki i przedmioty to **dane** (`configs/blocks.json`), nie klasy — nowy blok
dodajesz wpisem w JSON + 16×16 kafelkiem w atlasie.

### Jak to działa (skrót techniczny)

- **Chunki 16×16×128** (uint8 NumPy). Generacja i meshing na wątkach roboczych
  (NumPy zwalnia GIL), upload na GPU tylko z głównego wątku, z budżetem na klatkę.
- **Mesher zwektoryzowany**: face culling + per-vertex ambient occlusion +
  wybór przekątnej quada eliminujący artefakt AO. Wierzchołek = 2× uint32
  (pozycja, róg, AO, ściana, tekstura, emisja) — dekodowany w vertex shaderze,
  którego tablice stałych są *generowane* z tych samych danych co mesher.
- **Generator wieloprzebiegowy**: kontynenty → góry (ridged noise) → klimat
  (temperatura/wilgotność) → biomy → teren → jaskinie (spaghetti + caverny)
  → rudy → woda → drzewa → rośliny. Wszystko jest czystą funkcją
  `(seed, chunk)`, więc drzewa rosną bezszwowo przez granice chunków.
- **Renderer**: texture array (bez krwawienia UV), frustum culling
  zwektoryzowany, pass nieprzezroczysty front-to-back, cutout (liście/szkło/
  rośliny) bez cullingu, woda blendowana back-to-front z obniżoną, falującą
  taflą; niebo proceduralne z tarczą słońca i gwiazdami w nocy.

## Testy

```
py tools/logic_test.py    # edycje, zapis, fizyka, raycast (bez okna)
py tools/smoke_test.py    # render offscreen do PNG + mikro-benchmark
py launcher.py --frames 300 --screenshot test.png   # pełna gra, auto-zamknięcie
```

Dalszy plan rozwoju: `docs/ROADMAP.md`.

## Autor i licencja

**Autor:** [Marak123](https://github.com/Marak123)

Kod silnika i gry został wygenerowany przy użyciu **Claude Fable 5**
(Claude Code, Anthropic) na podstawie autorskiej specyfikacji projektu.

Licencja: [MIT](LICENSE) — rób z tym, co chcesz, zachowując notkę o prawach
autorskich.
