# Roadmap — status vs. the design specification

> **Development follows the 10-phase master plan in [`docs/plan/`](plan/00_overview_and_priorities.md).**
> **All 10 phases are implemented.** Each was adopted pragmatically — the
> highest-value parts in full, with heavier extras deferred and listed below
> so the plan stays an honest map of what exists.

Legend: ✅ done | 🟡 partial | ⬜ deferred

## Phase-by-phase

| Phase | Title | Status | Deferred to future work |
|---|---|---|---|
| 1 | Rendering pipeline (FBOs, HDR, CSM shadows) | ✅ | — |
| 2 | Shaders & effects (bloom, water, PBR, grading) | 🟡 | SSAO, screen-space reflections/refraction, auto-exposure |
| 3 | Texture system (64px, normal + MRAO maps, packs) | 🟡 | 128px, connected textures, multi-frame animation (UV-scroll used) |
| 4 | World height 256, biomes, rivers, caves, veins | ✅ | dripstone, lush caves, full 15-biome set (13 done) |
| 5 | Fluid simulation (flowing water/lava, falling sand) | 🟡 | per-level water render height, infinite sources |
| 6 | Structures & dimensions (villages, Nether, portals) | 🟡 | End dimension, jigsaw villages, dedicated portal shader |
| 7 | ECS, mobs & AI (behaviour trees, A*, 8 mobs) | ✅ | skeleton arrow projectiles (hitscan now), mob save persistence |
| 8 | Survival depth (hunger, tools, shaped crafting) | 🟡 | furnace/smelting, armor, tool durability, XP, combat cooldown/shield |
| 9 | Audio & particles | 🟡 | 3D positional audio & music (single-shot winsound now) |
| 10 | Performance & polish | 🟡 | greedy meshing, Numba JIT, full clickable main menu |

## Engine

| Area | Status | Notes |
|---|---|---|
| Multi-pass pipeline: HDR target + ACES tonemap + grading | ✅ | scene → HDR → bloom → tonemap → screen |
| Cascaded shadow maps (3 cascades, PCF, texel-snapped) | ✅ | toggle in settings; ~1–2 ms |
| Bloom (threshold pyramid) + god rays + vignette | ✅ | toggle in settings |
| Simplified voxel PBR (procedural normal + MRAO maps) | ✅ | per-block metallic/roughness, sun specular |
| Flood-fill lighting (sky + block) | ✅ | real voxel shadows, warm torch light |
| 16×16×256 chunks, sub-chunk Y-clipping | ✅ | mesher clips to occupied band |
| Async streaming (generate → light → mesh workers) | ✅ | per-frame submit + integration budgets |
| Fluid simulation + block-tick scheduler | ✅ | water/lava flow, water+lava→obsidian/stone, falling sand |
| Structures (villages, dungeons, mineshafts) | ✅ | stateless, cross-chunk |
| Nether dimension + portals | ✅ | obsidian frame + G to ignite, 8:1 coord scale |
| ECS + config-driven mobs + behaviour trees + A* | ✅ | 8 mob types, spawn by light, drops |
| Particle system (instanced billboards) | ✅ | block breaks, explosions, splashes |
| Procedural audio | 🟡 | winsound single-shot cues; no 3D mix |
| Multiplayer | ⬜ | deliberately postponed |

## Gameplay

| Area | Status | Notes |
|---|---|---|
| Movement: walk/sprint/jump/swim/fly/sneak | ✅ | per-axis AABB |
| Survival: health, hunger, fall/lava/drown damage, regen | ✅ | hearts + drumsticks + air HUD |
| Inventory (36 slots) + shaped crafting + creative picker | ✅ | 29 recipes incl. 16 tools |
| Tools: 4 tiers × pickaxe/axe/shovel/sword | ✅ | mining speed + tier-gated drops (no durability yet) |
| Food & eating | ✅ | apples, meat; cooked = more hunger |
| Mobs: pig/cow/sheep/chicken + zombie/skeleton/creeper/spider | ✅ | passive wander, hostile chase/shoot/explode |
| Block break/place/pick, portals, third-person (F5) | ✅ | |

## Performance notes (RTX 3050 laptop, Python 3.12)

- ~45–55 FPS at render distance 8 with **all** effects on; 60 FPS with
  shadows or bloom toggled off in the pause menu.
- Frame cost is dominated by the fullscreen HDR passes (bloom + tonemap),
  not geometry (~0.06–0.1 M verts visible). The two biggest future wins:
  **greedy meshing** (fewer faces; deferred because it must preserve the
  per-vertex AO/smooth-light keys — needs a careful merge + benchmark) and
  collapsing the bloom chain / optional half-res compositing.
- Generation ~18–28 ms/chunk, lighting ~50–90 ms/chunk, meshing ~5–10 ms —
  all on worker threads, hidden from the frame.
