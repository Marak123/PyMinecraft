# Roadmap — status vs. the design specification

Legend: ✅ done | 🟡 partial | ⬜ planned

## Engine

| Area | Status | Notes |
|---|---|---|
| Window + GL 3.3 context (GLFW/ModernGL) | ✅ | raw mouse motion, vsync, fullscreen |
| Layered engine/game architecture | ✅ | the engine knows nothing about gameplay |
| Data-driven blocks (JSON) | ✅ | hardness, emission, light attenuation, support flags |
| Chunks + async streaming | ✅ | three worker stages: generate → light → mesh, nested radii, per-frame budgets |
| Mesher (culling + AO + smooth light + vertex compression) | ✅ | vectorised NumPy, 8 B/vertex |
| Greedy meshing | ⬜ | conflicts with per-vertex AO/light — needs benchmarking first |
| Frustum culling | ✅ | vectorised, mesh-tight vertical bounds |
| Occlusion culling / LOD | ⬜ | |
| Procedural textures (texture array + mipmaps) | ✅ | 40+ tiles incl. UI icons, zero asset files |
| **Sky light flood fill (voxel shadows)** | ✅ | dark caves, shade under trees/overhangs |
| **Block light (torches, glowstone, lava)** | ✅ | warm tint, incremental relight on edits |
| Coloured light | ⬜ | per-channel flood fill — 3× memory/compute |
| Shadow maps / SSAO / bloom | ⬜ | voxel lighting covers the look until then |
| Day/night cycle, sky, moon, stars, clouds, fog | ✅ | drifting cloud layer, underwater/lava fog |
| World persistence (modified chunks only, npz + meta) | ✅ | async chunk saves on the worker pool |
| Profiler | 🟡 | per-stage frame timings in F3; no flame-graph style tooling |
| Audio (OpenAL) | ⬜ | |
| ECS | ⬜ | the player is the only entity so far; ECS arrives with mobs |
| Multiplayer (authoritative server) | ⬜ | deliberately postponed |
| Modding / scripting | 🟡 | blocks/tiles are data-driven; no mod-pack loading yet |

## World

| Area | Status | Notes |
|---|---|---|
| Heightmap: continents + hills + ridged mountains + domain warp | ✅ | |
| Climate: temperature/humidity → emergent biomes | ✅ | ocean, beach, desert, snowy, mountains, forest, plains |
| Caves (spaghetti + caverns) + lava | ✅ | 3D noise at half resolution (~3× faster generation) |
| Ores (coal/iron/gold/diamond by depth) | ✅ | |
| Glowstone pockets on cavern ceilings | ✅ | natural light landmarks underground |
| Trees: oak + birch, seamless across chunk borders | ✅ | stateless world-coordinate hash |
| Plants: tall grass, flowers, mushrooms, dead bushes | ✅ | biome-dependent |
| Rivers, lakes, villages, structures | ⬜ | the multi-pass pipeline is ready for them |
| Fluid simulation (flowing water) | ⬜ | water is static at sea level |
| Weather (rain/snow/storms) | ⬜ | |

## Gameplay

| Area | Status | Notes |
|---|---|---|
| Movement: walk, sprint (+FOV kick), jump, swim, fly | ✅ | per-axis AABB collisions |
| Sneaking (edge-safe) | ✅ | cannot walk off ledges while sneaking |
| **Survival: health, fall/lava/drowning damage, regen, death/respawn** | ✅ | hearts + air bubbles HUD, damage flash |
| Survival digging (per-block hardness, progress bar) | ✅ | instant in creative |
| Game modes: survival / creative (F4) | ✅ | flying is creative-only |
| Break/place/pick block, target highlight, support rules | ✅ | torches/plants need solid ground, pop when it breaks |
| Hotbar + HUD + F3 + pause | ✅ | |
| Inventory, crafting, hunger, mobs | ⬜ | next milestones (mobs bring ECS) |

## Performance notes (measured on RTX 3050 laptop, Python 3.12)

- Generation: ~16 ms/chunk (was ~60 ms before half-res 3D noise).
- Lighting: ~40 ms/chunk on a 3×3 window (worker threads hide it).
- Meshing: ~6 ms/chunk incl. smooth-light gathering.
- Frame: ~9 ms render at render distance 10 → steady 60 FPS (vsync).
- Next candidates: Numba for the lighting flood, greedy meshing benchmark,
  merged/indirect draws beyond render distance 16.
