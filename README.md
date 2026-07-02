# PyMinecraft

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![OpenGL](https://img.shields.io/badge/OpenGL-3.3-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A voxel sandbox game with a custom Python engine built on
**ModernGL + GLFW + NumPy**. Infinite procedural worlds with biomes, caves,
ores and trees, a day/night cycle, water, mining and building. Zero external
assets — every texture is generated procedurally at startup.

## Quick start

```
git clone https://github.com/Marak123/PyMinecraft.git
cd PyMinecraft
py -m pip install -r requirements.txt
py launcher.py
```

Requirements: Python 3.12+ and a GPU with OpenGL 3.3 (practically any).

## Controls

| Key | Action |
|---|---|
| `W A S D` | move |
| `Space` | jump / swim up / (while flying) ascend |
| `Left Ctrl` | sprint |
| `Left Shift` | (while flying) descend |
| `F` | toggle flying |
| `LMB` | break block |
| `RMB` | place block |
| `MMB` | pick targeted block into the hotbar |
| `1–9` / mouse wheel | select hotbar slot |
| `F3` | debug overlay (FPS, position, chunk stats) |
| `F2` | screenshot to `screenshots/` |
| `ESC` | pause / release mouse |

The world saves automatically on exit (modified chunks only) to
`saves/world/`. Settings (resolution, render distance, FOV, mouse
sensitivity, seed) live in `configs/settings.json` — the file is created
with defaults on first run.

## Architecture

```
launcher.py          entry point
game/                game layer: main loop, player, hotbar, HUD
engine/
  core/              config, logging, frame clock, 3D math
  window/  input/    GLFW window, input snapshot
  camera/            FPS camera + frustum
  world/             blocks (data-driven), noise, terrain generator, chunks,
                     async streaming, day cycle, world persistence
  graphics/          mesher (NumPy, AO, 8-byte packed vertices),
                     procedural texture array, shaders, renderer
  physics/           AABB collisions, DDA raycast
configs/             blocks.json (block definitions), settings.json
tools/               smoke_test.py (offscreen render), logic_test.py
```

Principles: the engine knows nothing about gameplay; every subsystem is
replaceable; blocks and items are **data** (`configs/blocks.json`), not
classes — adding a block takes a JSON entry plus a 16×16 tile painter.

### How it works (technical digest)

- **16×16×128 chunks** (NumPy uint8). Generation and meshing run on worker
  threads (NumPy releases the GIL); GPU uploads happen on the main thread
  only, under a per-frame budget.
- **Vectorised mesher**: face culling + per-vertex ambient occlusion +
  quad-diagonal flipping that removes the classic AO anisotropy artifact.
  A vertex is 2× uint32 (position, corner, AO, face, texture, emission),
  decoded in the vertex shader whose constant tables are *generated* from
  the same Python data the mesher uses — a single source of truth.
- **Multi-pass generator**: continents → mountains (ridged noise) → climate
  (temperature × humidity) → biomes → terrain → caves (spaghetti + caverns)
  → ores → water → trees → plants. Everything is a pure function of
  `(seed, chunk)`, so trees grow seamlessly across chunk borders.
- **Renderer**: texture array (no UV bleeding), vectorised frustum culling,
  opaque pass front-to-back, cutout pass (leaves/glass/plants) without
  culling, water blended back-to-front with a lowered, waving surface;
  procedural sky with a sun disc and night-time stars.

## Tests

```
py tools/logic_test.py    # edits, persistence, physics, raycast (headless)
py tools/smoke_test.py    # offscreen render to PNG + micro-benchmark
py launcher.py --frames 300 --screenshot test.png   # full game, auto-close
```

Development plan: see `docs/ROADMAP.md`.

## Author and license

**Author:** [Marak123](https://github.com/Marak123)

The engine and game code were generated with **Claude Fable 5**
(Claude Code, Anthropic) from the author's project specification.

License: [MIT](LICENSE).
