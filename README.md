# PyMinecraft

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![OpenGL](https://img.shields.io/badge/OpenGL-3.3-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A voxel sandbox game with a custom Python engine built on
**ModernGL + GLFW + NumPy**. Infinite procedural worlds with biomes, caves,
ores, trees and scattered ruins, flood-fill lighting with real voxel shadows
and warm torch light, survival mode with health, inventory and crafting,
wandering mobs, rain spells, a third-person view, a day/night cycle with
clouds and stars. Zero external assets — every texture is generated
procedurally at startup (32x32).

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
| `Left Ctrl` | sprint (with an FOV kick) |
| `Left Shift` | sneak — you cannot fall off edges / (while flying) descend |
| `F` | toggle flying (creative mode only) |
| `F4` | switch survival / creative mode |
| `LMB` | break block (hold to dig in survival) / attack mobs |
| `RMB` | place block (consumes from inventory in survival) |
| `MMB` | pick targeted block |
| `1–9` / mouse wheel | select hotbar slot |
| `E` | inventory & crafting (survival) / block picker (creative) |
| `F5` | first / third person view |
| `F11` | fullscreen toggle |
| `F3` | debug overlay (FPS, chunk stats, stage timings) |
| `F2` | screenshot to `screenshots/` |
| `ESC` | pause + settings menu (render distance, FOV, vsync...) |

Survival mode: 10 hearts, fall damage, drowning, lava damage, regeneration,
death & respawn; mined blocks drop into a 36-slot inventory and placing
consumes them; shapeless recipes (`configs/recipes.json`) craft planks,
torches, glass, bricks and more. Creative mode: flying, instant breaking,
no damage, full block picker. Pigs wander the plains; ruined towers with
torches dot the world; rain rolls through now and then.

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

- **16×16×128 chunks** (NumPy uint8). The streaming pipeline has three
  worker-thread stages — generate → light → mesh — with nested radii
  (each stage needs its neighbours from the previous one); GPU uploads
  happen on the main thread only, under a per-frame budget.
- **Flood-fill lighting**: two 0–15 fields per chunk. Sky light beams down
  and floods sideways (real voxel shadows: dark caves, shade under trees);
  block light floods out of torches, glowstone and lava. Fields are
  computed with iterative vectorised dilation on a 3×3-chunk window; edits
  relight a small box seeded with stored boundary light, which handles
  light removal without the classic two-phase BFS.
- **Vectorised mesher**: face culling + per-vertex ambient occlusion +
  smooth light (average of the four corner cells) + quad-diagonal flipping.
  A vertex is 2× uint32 (position, corner, AO, face, texture, emission,
  sky & block light), decoded in the vertex shader whose constant tables
  are *generated* from the same Python data the mesher uses.
- **Multi-pass generator**: continents → mountains (ridged noise) → climate
  (temperature × humidity) → biomes → terrain → caves (spaghetti + caverns,
  3D noise at half resolution, ~3× faster) → ores → glowstone pockets →
  water → trees (oak + birch) → plants. Everything is a pure function of
  `(seed, chunk)`, so trees grow seamlessly across chunk borders.
- **Renderer**: texture array (no UV bleeding), vectorised frustum culling
  with mesh-tight bounds, pooled chunk buffers (remeshes reuse GPU memory),
  opaque front-to-back, cutout without culling, water blended back-to-front
  with a waving surface, drifting procedural clouds; sky with a sun disc,
  a moon and night-time stars.

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
