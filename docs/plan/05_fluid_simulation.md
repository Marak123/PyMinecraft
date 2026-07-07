# Phase 5: Fluid Simulation

**Goal:** Replace the static water/lava at sea level with a full cellular automata fluid simulation — flowing water, cascading waterfalls, lava rivers, and water-lava interactions.

**Depends on:** Phase 4 (new world height, block tick system concept).

---

## Current State Analysis

**File:** `engine/world/generation.py` (line ~278)
- Water is placed statically during generation: any air below `SEA_LEVEL` (y=48) is filled with water blocks.
- Caves have a hardcoded rule: "keep ≥5 block roof" specifically because there's no fluid simulation to handle floods.
- Lava fills cave floors at y ≤ 10, also static.

**File:** `engine/world/world.py`
- `set_block()` modifies a block and marks the chunk dirty for remesh. No scheduled updates, no neighbor notification, no tick system.

**File:** `configs/blocks.json`
- Water and Lava are defined with `"render": "liquid"` but have no flow-level metadata.

---

## Implementation Steps

### 5.1 Block Tick System [NEW: `engine/world/block_ticks.py`]

Before implementing fluids, the engine needs a generic scheduled block update system:

```python
class BlockTickScheduler:
    """Schedules and processes block updates at a fixed rate."""
    
    def __init__(self, world):
        self.world = world
        self._queue: list[tuple[float, int, int, int]] = []  # (trigger_time, x, y, z)
        self._queued_set: set[tuple[int, int, int]] = set()   # Dedup
    
    def schedule(self, x, y, z, delay_ticks: int):
        """Schedule a block update. Deduplicates by position."""
        
    def tick(self, current_tick: int, max_updates: int = 512):
        """Process up to max_updates scheduled ticks.
        Returns list of (x, y, z) positions that were updated."""
```

**Rules:**
- When a block is placed/broken adjacent to a fluid, schedule a tick for that fluid (delay: 5 ticks for water, 30 ticks for lava).
- When a fluid block updates, it may schedule ticks for its neighbors.
- Cap at 512 fluid updates per game tick (20 TPS = 10,240 fluid updates/second max) to prevent lag from ocean breaks.

### 5.2 Fluid Block Data [MODIFY configs/blocks.json, engine/world/blocks.py]

Add flow levels to the block registry:

```json
{
  "name": "water",
  "id": 7,
  "render": "liquid",
  "solid": false,
  "opaque": false,
  "flow_levels": 8,
  "flow_speed": 5,
  "textures": {"all": "water"}
}
```

**Block ID scheme for flow levels:**
- Option A: Use block metadata (requires adding a metadata array to chunks — 1 byte per block extra).
- Option B (simpler, recommended for now): Reserve block IDs for each flow level. Water source = ID 7. Water flow levels 1–7 = IDs 64–70. Lava source = ID 8. Lava flow levels 1–3 = IDs 71–73.
- Option B wastes IDs but avoids a metadata refactor. With `uint8` blocks, there are 255 usable IDs — plenty for now.

### 5.3 Fluid Simulation Logic [NEW: `engine/world/fluids.py`]

```python
class FluidSimulator:
    """Cellular automata fluid simulation for water and lava."""
    
    def __init__(self, world, tick_scheduler, registry):
        self.world = world
        self.scheduler = tick_scheduler
```

**Water rules (per-tick update for a single water block at (x, y, z)):**

```
1. If block below is AIR:
   → Set below to WATER_SOURCE (full)
   → Schedule below for next tick
   → If current is a flow block (not source), decrease level or remove
   
2. If block below is SOLID or WATER:
   → Try to spread horizontally (±X, ±Z)
   → For each horizontal neighbor:
     - If AIR: place WATER with level = current_level - 1
     - If WATER with lower level: increase to current_level - 1
   → Schedule all modified neighbors
   
3. If current block has no source feeding it:
   → Decrease level by 1 per tick
   → If level reaches 0, remove block (evaporate)
```

**Source detection:**
- A water source block is permanent (placed by player or generated at sea level).
- A flow block is temporary — it persists only if fed by an adjacent source or higher-level flow.
- Two flow blocks meeting at the same level can create a new source (infinite water trick).

**Lava differences:**
- Flows slower: 30-tick delay vs water's 5-tick.
- Spreads only 3 blocks horizontally (water spreads 7).
- Does not create infinite sources.

### 5.4 Water-Lava Interactions [ADD to fluids.py]

When water and lava meet:
- **Water flows into lava source** → Lava becomes **Obsidian** (new block).
- **Water flows into lava flow** → Lava becomes **Cobblestone**.
- **Lava flows into water** → Water becomes **Stone**.
- Play a hissing sound (Phase 9 audio) and emit steam particles (Phase 9 particles).
- Schedule remesh for affected area.

### 5.5 Mesher Changes for Flow Levels [MODIFY engine/graphics/mesher.py]

Currently water is rendered as a flat surface lowered by 0.115 blocks. With flow levels:

- **Full water (source):** Surface at y + 0.875 (current behavior, slightly below block top).
- **Flow level N (1–7):** Surface at `y + (N / 8.0) * 0.875`. Level 1 = thin layer, level 7 = nearly full.
- **Side faces:** Only render the face if the adjacent block is not also water (or is water with a lower level). The face height must match the flow level.
- **Flow direction arrows in shader:** In the fragment shader, slightly tilt the water normal in the flow direction for visual flow cues. Calculate flow direction on CPU and encode in vertex data.

### 5.6 Falling Sand/Gravel [ADD to block_ticks.py]

While implementing the tick system, add gravity for sand and gravel:

- When a sand/gravel block has air below it, schedule a tick.
- On tick: remove the block, spawn a "falling block" entity (or simply teleport: move block down until it hits solid ground, instant for simplicity).
- Schedule the block above (in case there's a stack of sand).
- Also trigger when a block below sand/gravel is broken.

### 5.7 Integration with World Edits [MODIFY engine/world/world.py, game/game.py]

In `world.set_block()`:
```python
def set_block(self, wx, wy, wz, block_id):
    # ... existing logic ...
    # After modifying the block, notify fluids:
    self.tick_scheduler.on_block_changed(wx, wy, wz)
```

`on_block_changed` checks all 6 neighbors. If any neighbor is a fluid, schedule it for update.

In `game.game.py` main loop:
```python
def _update_gameplay(self, dt):
    # ... existing updates ...
    self.world.tick_scheduler.tick(current_tick)  # Process scheduled fluid/sand updates
```

### 5.8 Generation Changes [MODIFY engine/world/generation.py]

- Remove the "keep ≥5 block roof" constraint in cave generation — floods are now handled by the fluid sim.
- Generate waterfalls: when a cave opening intersects a river or ocean, water will naturally flow in.
- Underground lakes: place water sources in cheese caverns at low altitudes.
- Lava lakes: place lava sources in deep caves (y < 20) as small pools.

---

## Verification

1. Break a block at the bottom of the ocean — water should flow into the hole.
2. Place a block in a river — water should flow around it.
3. Dig a channel from a lake downhill — water should flow down and form a waterfall.
4. Pour water on lava — verify obsidian/cobblestone generation.
5. Build a 2×2 pool (infinite water source) — removing water from the middle should refill.
6. Break a block under sand — sand should fall.
7. Performance: flooding a large area (breaking a dam) should not drop below 30 FPS. Fluid updates are capped at 512/tick.
8. Save/load: fluid state must persist correctly.
