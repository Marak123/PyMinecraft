# Roadmap — status vs. the design specification

Legend: ✅ done | 🟡 partial | ⬜ planned

## Engine

| Area | Status | Notes |
|---|---|---|
| Window + GL 3.3 context (GLFW/ModernGL) | ✅ | raw mouse motion, vsync, fullscreen |
| Layered engine/game architecture | ✅ | the engine knows nothing about gameplay |
| Data-driven blocks (JSON) | ✅ | `configs/blocks.json` → NumPy lookup tables |
| Chunks + async streaming | ✅ | thread pool, priority by distance and camera direction, per-frame budgets |
| Mesher (culling + AO + vertex compression) | ✅ | vectorised NumPy, 8 B/vertex |
| Greedy meshing | ⬜ | conflicts with per-vertex AO — needs benchmarking first (spec: measure before optimising) |
| Frustum culling | ✅ | vectorised AABB vs 6 planes |
| Occlusion culling / LOD | ⬜ | |
| Procedural textures (texture array + mipmaps) | ✅ | 25 tiles, zero asset files |
| Directional lighting + AO + emission (lava) | ✅ | per-vertex; no block-light propagation yet |
| Block light (flood fill), coloured light | ⬜ | next big step |
| Day/night cycle, procedural sky, stars, fog | ✅ | |
| World persistence (modified chunks only, npz + meta) | ✅ | corrupt files fall back to regeneration |
| Built-in profiler | 🟡 | timings in smoke_test + F3; no per-stage profiler |
| Audio (OpenAL) | ⬜ | |
| ECS | ⬜ | the player is the only entity so far; ECS arrives with mobs |
| Multiplayer (authoritative server) | ⬜ | world/streaming already separates data from rendering |
| Modding / scripting | 🟡 | blocks/tiles are data-driven; no mod-pack loading yet |

## World

| Area | Status | Notes |
|---|---|---|
| Heightmap: continents + hills + ridged mountains + domain warp | ✅ | |
| Climate: temperature/humidity → emergent biomes | ✅ | ocean, beach, desert, snowy, mountains, forest, plains |
| Spaghetti caves + caverns + lava | ✅ | never breach the surface (no fluid simulation yet) |
| Ores (coal/iron/gold/diamond by depth) | ✅ | |
| Trees seamless across chunk borders | ✅ | stateless world-coordinate hash |
| Plants (tall grass, flowers) | ✅ | |
| Rivers, lakes, villages, structures | ⬜ | the multi-pass pipeline is ready for them |
| Fluid simulation (flowing water) | ⬜ | water is static at sea level |
| Weather (rain/snow/storms) | ⬜ | |

## Gameplay

| Area | Status | Notes |
|---|---|---|
| Movement: walking, sprint, jump, swimming, flying | ✅ | per-axis AABB collisions |
| Break/place/pick block, target highlight | ✅ | DDA raycast, no placing inside the player |
| Hotbar + HUD + F3 + pause | ✅ | |
| Inventory, crafting, survival (HP/hunger), mobs | ⬜ | next milestones |

## Known performance trade-offs (measure before optimising)

- Generation ~60 ms/chunk — dominated by 3D cave/ore noise (8× perlin3);
  candidates: half-resolution + interpolation, Numba, fewer noise fields.
- Saving a modified chunk on unload happens on the main thread (a few ms) —
  move it to the worker pool.
- ~450 draw calls at render distance 10 — fine (60 FPS on an RTX 3050);
  for larger distances consider merged/indirect drawing.
