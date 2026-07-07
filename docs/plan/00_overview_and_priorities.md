# PyMinecraft — Full Development Master Plan

**Target Agent:** Claude Code Fable 5 (or equivalent advanced AI coding agent)
**Project Root:** `p:\Programowanie\Github-Projects\PyMinecraft`
**Engine:** Python 3.12+ / ModernGL 5.10+ / GLFW / NumPy
**Current State:** Functional alpha — basic survival, 7 biomes, 25 block types, 1 mob, no audio, no multiplayer, basic shaders.

---

## Current Codebase Summary

| Module | Key Files | Status |
|---|---|---|
| Engine Core | `engine/core/config.py`, `clock.py`, `math.py` | ✅ Working, minimal |
| Window/Input | `engine/window/window.py`, `engine/input/input.py` | ✅ No gamepad, no rebinding |
| Camera | `engine/camera/camera.py` | ✅ Euler angles only |
| Physics | `engine/physics/collision.py`, `raycast.py` | ✅ Player-only AABB, DDA raycast |
| World | `engine/world/*.py` (11 files) | ✅ 16×16×128 chunks, flood-fill light, 7 biomes, basic caves |
| Graphics | `engine/graphics/*.py` (7 files) | ✅ Forward single-pass, no shadows/PBR/post-fx, 16×16 procedural textures |
| Game Layer | `game/*.py` (6 files) | ✅ Survival basics, 1 mob type (pig), shapeless crafting, no hunger/tools |

## Implementation Priority Order

The plan is split into **10 phases**, ordered by impact and dependency:

| Phase | File | What It Adds |
|---|---|---|
| **Phase 1** | `01_rendering_pipeline_overhaul.md` | Multi-pass renderer, FBOs, deferred lighting setup, shadow maps |
| **Phase 2** | `02_shaders_and_visual_effects.md` | PBR materials, normal maps, SSAO, bloom, volumetric light, water shaders |
| **Phase 3** | `03_texture_system_upgrade.md` | Hi-res textures (128×128), texture pack loading, PBR maps, connected textures |
| **Phase 4** | `04_world_height_and_generation.md` | 256/384 world height, new biomes, rivers, better caves, ore veins |
| **Phase 5** | `05_fluid_simulation.md` | Cellular automata water/lava, flowing mechanics, water-lava interactions |
| **Phase 6** | `06_structures_and_dimensions.md` | Villages, dungeons, mine shafts, Nether dimension, End dimension, portals |
| **Phase 7** | `07_entities_ecs_and_mobs.md` | Full ECS, hostile mobs, mob AI behavior trees, A* pathfinding, drops, spawning |
| **Phase 8** | `08_survival_depth.md` | Hunger, tools with tiers/durability, shaped crafting, furnace, enchanting, combat |
| **Phase 9** | `09_audio_and_particles.md` | 3D positional audio, ambient sounds, music, particle system, block break effects |
| **Phase 10** | `10_performance_and_polish.md` | Greedy meshing, Numba JIT, occlusion culling, LOD, main menu, loading screen, multiplayer prep |

## Key Design Principles

1. **Engine knows nothing about gameplay** — preserve this existing boundary.
2. **Data-driven everything** — blocks, items, recipes, mobs, structures defined in JSON/configs.
3. **No external asset files** — keep procedural texture generation but upgrade resolution and quality. Optionally support loading external packs.
4. **Test every system** — extend `tools/logic_test.py` and `tools/smoke_test.py` after each phase.
5. **Backward compatible saves** — version the save format, migrate old worlds gracefully.

## File Naming Convention for New Code

```
engine/
  ecs/                          # Phase 7
    __init__.py
    world.py                    # ECS world container
    components.py               # All component dataclasses
    systems/
      physics_system.py
      ai_system.py
      render_system.py
      spawn_system.py
  graphics/
    pipeline.py                 # Phase 1: Multi-pass render pipeline manager
    framebuffer.py              # Phase 1: FBO management
    postprocess.py              # Phase 2: Post-processing chain
    shadow.py                   # Phase 1: Shadow map rendering
    water_shader.py             # Phase 2: Advanced water rendering
    particle.py                 # Phase 9: GPU particle system
  world/
    fluids.py                   # Phase 5: Fluid simulation
    structures.py               # Phase 6: Structure generation
    dimensions.py               # Phase 6: Dimension management
    block_ticks.py              # Phase 5: Scheduled block update system
  audio/                        # Phase 9
    __init__.py
    audio_engine.py
    sound_manager.py
  network/                      # Phase 10 (prep)
    __init__.py
    protocol.py
    server.py
    client.py
game/
  crafting.py                   # Phase 8: Advanced crafting system
  combat.py                     # Phase 8: Combat mechanics
  items.py                      # Phase 8: Item system with metadata
  mobs/                         # Phase 7
    __init__.py
    pig.py
    zombie.py
    skeleton.py
    creeper.py
    spider.py
  ai/                           # Phase 7
    behavior_tree.py
    pathfinding.py
    goals.py
  screens/                      # Phase 10
    main_menu.py
    world_select.py
    loading_screen.py
configs/
  items.json                    # Phase 8
  mobs.json                     # Phase 7
  structures/                   # Phase 6
    village_house.json
    dungeon.json
    mineshaft.json
  loot_tables.json              # Phase 7
```
