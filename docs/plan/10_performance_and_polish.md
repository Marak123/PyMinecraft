# Phase 10: Performance, Polish & Multiplayer Preparation

**Goal:** Optimize the engine for smooth 60+ FPS at high render distances, add a polished main menu and loading screen, improve the test suite, and lay the networking foundation for future multiplayer.

**Depends on:** All previous phases should be functional before optimization and polish.

---

## Current State Analysis

**Performance (from ROADMAP.md, measured on RTX 3050):**
- Generation: ~16 ms/chunk
- Lighting: ~40 ms/chunk on 3×3 window (worker threads hide it)
- Meshing: ~6 ms/chunk
- Frame: ~9 ms render at RD 10 → 60 FPS (vsync)

**Bottlenecks identified:**
- `engine/world/noise.py`: Pure NumPy noise — no JIT compilation.
- `engine/graphics/mesher.py`: No greedy meshing — every block face emits a separate quad.
- `engine/world/lighting.py`: Iterative vectorized dilation — ~40ms/chunk.
- `engine/graphics/renderer.py`: No occlusion culling, no LOD, single-detail rendering.
- Python GIL: `threading` module doesn't provide true parallelism for CPU-bound work.
- `engine/core/timing.py`: No fixed timestep — physics tied to frame rate.

---

## Implementation Steps

### 10.1 Greedy Meshing [MODIFY engine/graphics/mesher.py]

**Problem:** Each visible block face emits 4 vertices (1 quad). A flat 16×16 grass plain = 256 quads. With greedy meshing = 1 quad.

**Algorithm:**
```
For each face direction (6 faces):
  For each layer perpendicular to the face:
    Create a 2D "face mask" of which blocks have visible faces
    While mask has True cells:
      Find first True cell
      Expand right while same texture + same AO + same light
      Expand down while entire row matches
      Emit one merged quad for the rectangle
      Clear the rectangle from the mask
```

**Challenge:** Per-vertex AO and smooth lighting. Solutions:
- **Option A (recommended):** Only merge faces where all 4 corners have identical AO and light values. This limits merging but preserves visual quality. Typical reduction: 60–70% fewer quads.
- **Option B:** Bake AO/light into a per-face lightmap texture. Maximum merging but requires additional texture storage and UV management. More complex.

**Implementation with Numba:**
```python
from numba import njit

@njit
def _greedy_merge(face_mask, ao_data, light_data, tex_data):
    """Greedy mesh a 16×128 (or 16×256) face layer.
    Returns list of (x, y, w, h, texture, ao_corners, light_corners)."""
```

Use `@njit` for the inner loop — it's pure integer/boolean logic, perfect for Numba.

**Expected performance improvement:**
- Vertex count reduction: 50–70%.
- GPU draw time reduction: ~40% (fewer vertices to shade).
- Meshing time: may increase slightly due to algorithm complexity, but Numba compensates.

### 10.2 Numba JIT for Hot Paths [MODIFY noise.py, lighting.py, mesher.py]

**Noise generation (`noise.py`):**
- Decorate `perlin2`, `perlin3`, `fbm2`, `fbm3`, `ridged2` with `@njit(parallel=True, fastmath=True)`.
- Expected speedup: 10–50× over pure NumPy.
- Chunk generation time: 16ms → ~3ms.

**Lighting (`lighting.py`):**
- The `_spread_once` function does 6 directional shifts and max operations — perfect for `@njit(parallel=True)`.
- Expected speedup: 5–10×.
- Lighting time: 40ms → ~5ms.

**Mesher (`mesher.py`):**
- The face culling and AO calculation loops are already vectorized with NumPy, but the greedy merging inner loop benefits from Numba.

**Fallback:** If Numba is not available (installation issues), fall back to pure NumPy. Use a try/except import pattern:
```python
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(fn): return fn
        return decorator
```

### 10.3 Occlusion Culling [MODIFY engine/graphics/renderer.py]

**Current:** Frustum culling only — chunks behind mountains are still rendered.

**Approach: Hardware Occlusion Queries (ModernGL):**
1. Render chunk bounding boxes (invisible, depth-only, no color write) as occluders.
2. Issue `gl.begin_conditional_render()` for each chunk before drawing its actual geometry.
3. If the bounding box was occluded, skip the full chunk draw.

**Simpler approach (recommended first): Hybrid occlusion with heightmap:**
- Maintain a 2D heightmap of the maximum solid block height per chunk column.
- For each chunk, check if the camera's line of sight passes below the heightmap of any intervening chunk.
- Cull chunks that are fully behind terrain ridges.

### 10.4 LOD (Level of Detail) [MODIFY mesher.py, renderer.py]

For distant chunks (beyond 60% of render distance):
- Use a simplified mesh: skip AO calculation, skip smooth lighting, use flat lighting.
- Optionally: render distant chunks as 2×2×2 block clusters (each 8 blocks merged into 1) — 8× fewer vertices.
- Transition: fade between LOD levels using vertex alpha to avoid popping.

### 10.5 Fixed Timestep Physics [MODIFY engine/core/timing.py, game/game.py]

**Current:** Physics runs at frame rate — faster FPS = different physics behavior.

**Fix:** Implement a fixed timestep accumulator:
```python
PHYSICS_DT = 1.0 / 60.0  # 60 Hz fixed physics
accumulator = 0.0

def update(frame_dt):
    accumulator += frame_dt
    while accumulator >= PHYSICS_DT:
        update_physics(PHYSICS_DT)
        accumulator -= PHYSICS_DT
    alpha = accumulator / PHYSICS_DT  # For render interpolation
```

Render interpolation: store previous and current positions. Render at `lerp(prev, curr, alpha)` for buttery smooth visuals even at lower physics rates.

### 10.6 Main Menu & Loading Screen [NEW: `game/screens/`]

**Main Menu (`game/screens/main_menu.py`):**
- Title: "PyMinecraft" with animated logo (letters bob up and down).
- Buttons: "Singleplayer", "Settings", "Quit".
- Background: slowly rotating panoramic view of a pre-generated world (or a static procedural gradient for simplicity).
- Version number in bottom-left corner.

**World Selection (`game/screens/world_select.py`):**
- List of saved worlds (scan `saves/` directory).
- Each entry shows: world name, seed, last played date, file size.
- Buttons: "Create New World", "Delete", "Play".
- "Create New World": text input for name + seed (or random).

**Loading Screen (`game/screens/loading_screen.py`):**
- Shown during initial chunk generation.
- Progress bar: "Generating terrain... (42/100 chunks)"
- Animated spinner or block icon.
- Tips text rotating at the bottom (random gameplay tips).

### 10.7 Comprehensive Test Suite [MODIFY tools/]

Extend tests to cover all new systems:

```
tools/
  tests/
    test_world.py           # Block edits, persistence, height expansion
    test_physics.py          # Collision, fixed timestep, edge cases
    test_lighting.py         # Sky/block light, relight, colored light
    test_fluids.py           # Water flow, lava interaction, source detection
    test_mesher.py           # Greedy meshing output, vertex count reduction
    test_inventory.py        # Stacking, crafting, durability, item metadata
    test_crafting.py         # Shaped + shapeless recipes, furnace smelting
    test_ecs.py              # Entity creation, component queries, system execution
    test_ai.py               # Behavior tree ticking, pathfinding A*
    test_generation.py       # Biome distribution, structure placement, ore veins
    test_audio.py            # Sound generation, 3D attenuation math
    test_particles.py        # Emission, update, compaction
    test_networking.py       # Packet serialization, protocol
  smoke_test.py              # Existing — extend with new rendering features
  benchmark.py               # Performance regression tracking
```

**Benchmark script (`tools/benchmark.py`):**
- Generate 100 chunks, measure mean/p95 generation time.
- Light 100 chunks, measure mean/p95 lighting time.
- Mesh 100 chunks, measure vertex count and meshing time.
- Render 1000 frames at RD 16, report mean frame time and GPU utilization.
- Compare against baseline numbers from ROADMAP.md.

### 10.8 Multiplayer Preparation [NEW: `engine/network/`]

**This phase only prepares the architecture — full multiplayer is a separate future milestone.**

**Step 1: Headless server mode (`server.py`)**
- Create a new entry point that runs the game loop without any rendering.
- World, physics, ECS, fluids all tick at 20 TPS.
- No ModernGL, no GLFW, no window.
- Accepts commands via stdin or a simple TCP socket.

**Step 2: Input/State separation**
- Refactor `Player` to accept an `InputState`-like interface (could come from keyboard OR from a network packet).
- Refactor `World` state mutations to be serializable as "block change" events.
- Refactor entity updates to produce "entity state" snapshots (position, velocity, animation state).

**Step 3: Protocol skeleton (`engine/network/protocol.py`)**
```python
class PacketType(Enum):
    LOGIN = 0x00
    CHUNK_DATA = 0x01
    PLAYER_POSITION = 0x02
    BLOCK_UPDATE = 0x03
    ENTITY_UPDATE = 0x04
    CHAT_MESSAGE = 0x05

def serialize_packet(packet_type: PacketType, data: dict) -> bytes:
    """Serialize a packet to binary using struct."""

def deserialize_packet(raw: bytes) -> tuple[PacketType, dict]:
    """Deserialize a binary packet."""
```

This is **preparation only** — the actual client-server networking, interpolation, and prediction are a separate project phase beyond this plan.

### 10.9 Bug Fixing & Edge Cases

Address known edge cases discovered during codebase analysis:

1. **Chunk border lighting**: Verify that light propagation across chunk borders works correctly after all phases. The current `relight_box` uses a 16-block radius — ensure this handles edge cases at world height boundaries.
2. **Floating point precision**: `_EPS = 1e-4` in physics — test edge cases at very large coordinates (x > 10000). Consider using chunk-local coordinates for physics to avoid precision loss.
3. **Save format versioning**: After expanding world height to 256, old saves with height 128 must be migrated. Add version checking to `save.py`.
4. **Memory leaks**: Ensure all GPU resources (VAOs, VBOs, textures) are properly released when chunks are unloaded. Add reference counting or weak references.
5. **Window resize**: After Phase 1 (FBOs), ensure all framebuffers are recreated on window resize without crashing.
6. **Thread safety**: `World.set_block()` is called from the main thread. Worker threads read chunk data. Ensure no race conditions exist (currently the code uses chunk state guards — verify they're sufficient).

### 10.10 Configuration & Settings Polish [MODIFY game/ui.py, engine/core/config.py]

Expand the settings menu:

```
Graphics Settings:
  - Render Distance: 4–24 (slider)
  - FOV: 60–110 (slider)
  - VSync: On/Off
  - Shadow Quality: Off / Low / Medium / High
  - SSAO: Off / On
  - Bloom: Off / On
  - Volumetric Light: Off / On
  - Water Quality: Simple / Realistic
  - Particles: Off / Low / High
  - Max FPS: 30 / 60 / 120 / Unlimited

Audio Settings:
  - Master Volume: 0–100 (slider)
  - Music Volume: 0–100 (slider)
  - SFX Volume: 0–100 (slider)
  - Ambient Volume: 0–100 (slider)

Controls:
  - Mouse Sensitivity: 0.02–0.30 (slider)
  - Invert Y: On/Off
  - Key Bindings: (future — display current keys, click to rebind)

Game:
  - Difficulty: Peaceful / Easy / Normal / Hard
  - Auto-save Interval: 1 / 5 / 10 minutes
  - GUI Scale: 1x / 2x / 3x / Auto
```

---

## Verification

1. **Greedy meshing**: Generate flat terrain, count vertices. Should be ~70% fewer than before.
2. **Numba**: Run `tools/benchmark.py`. Generation should be < 5ms/chunk. Lighting < 10ms/chunk.
3. **Fixed timestep**: Run at 30 FPS and 144 FPS — physics should behave identically (same jump height, same fall speed).
4. **Main menu**: Launch game → see title screen → click Singleplayer → see world list → create new world → loading screen → gameplay.
5. **Full integration test**: Start new world → mine wood → craft tools → find cave → mine ores → smelt iron → build house → survive night → enter nether → return. All systems must work together without crashes.
6. **Memory**: Play for 10 minutes while moving — check that memory usage stabilizes (no unbounded growth from GPU buffer leaks).
7. **Save/load**: Save game → restart → load → everything restored (position, inventory, block edits, dimension, time of day).

---

## Final Performance Targets

| Metric | Target (RTX 3050, 1080p) |
|---|---|
| Chunk generation | < 5 ms |
| Chunk lighting | < 8 ms |
| Chunk meshing | < 4 ms |
| Frame time (RD 10) | < 10 ms (100+ FPS) |
| Frame time (RD 16) | < 16 ms (60+ FPS) |
| Memory (RD 16) | < 1.5 GB |
| Startup time | < 8 seconds |
| World load time | < 3 seconds |
