# Phase 4: World Height Expansion & Advanced Generation

**Goal:** Expand the world from 128 to 256+ blocks tall, add new biomes, rivers, improved caves, and ore veins for a richer, more explorable world.

**Depends on:** None (can be done in parallel with graphics phases).

---

## Current State Analysis

**File:** `engine/world/chunk.py` (~30 lines)
- `Chunk.blocks` is a NumPy `uint8` array shaped `(16, 16, 128)` — height hardcoded to 128.
- `sky_light` and `block_light` have the same shape.

**File:** `engine/world/generation.py` (~540 lines)
- Multi-pass pipeline: continents → mountains → climate → biomes → terrain → caves → ores → water → trees → plants.
- 7 biomes: ocean, beach, desert, snowy, mountains, forest, plains.
- Sea level at y=48. Max height ~120 (mountain peaks).
- Caves: spaghetti (worm tunnels) + caverns (large rooms) using 3D noise at half resolution.
- Trees: oak + birch only. Cross-chunk safe via world-coordinate hash.

**File:** `engine/world/noise.py` (~180 lines)
- Pure NumPy simplex noise. `fbm2`, `fbm3`, `ridged2`. No JIT compilation.

---

## Implementation Steps

### 4.1 Expand World Height to 256 [MODIFY chunk.py, generation.py, lighting.py, mesher.py, collision.py]

**Global constant:** Define `CHUNK_HEIGHT = 256` in `engine/world/chunk.py` (or a shared constants module).

**Changes required:**
- `chunk.py`: Change array shapes from `(16, 16, 128)` to `(16, 16, 256)`.
- `generation.py`: Adjust all height references. Mountain peaks can now reach y=200+. Sea level stays at y=64 (raise from 48 for more underwater depth). Bedrock at y=0.
- `lighting.py`: The flood-fill iterates over the full Y range — doubling height doubles lighting time. Optimize by tracking the max solid Y per column and skipping air above.
- `mesher.py`: Same Y range expansion. Optimize by splitting the chunk into vertical sections (sub-chunks of 16×16×16) and only meshing non-empty sections.
- `collision.py`: Update Y bounds checks.
- `save.py`: Version the save format. Old saves with height=128 must be migrated (pad with air above y=127).
- `renderer.py`: Frustum culling already uses mesh-tight vertical bounds — no change needed, but verify.

**Memory impact:** Each chunk goes from ~80 KB to ~160 KB (blocks + 2 light channels). At render distance 16, ~1000 loaded chunks = ~160 MB. Acceptable for modern systems.

### 4.2 Sub-Chunk Optimization [MODIFY chunk.py, mesher.py]

Split each chunk into 16 vertical sections of 16×16×16:

```python
class Chunk:
    def __init__(self, cx, cz):
        self.blocks = np.zeros((16, 16, CHUNK_HEIGHT), dtype=np.uint8)
        self.sections_dirty = [False] * (CHUNK_HEIGHT // 16)  # Track which sections need remeshing
        self.section_empty = [True] * (CHUNK_HEIGHT // 16)    # Skip completely empty sections
```

Benefits:
- Only remesh the section that was edited (not the whole 16×16×256 column).
- Skip meshing/rendering empty sections above terrain height.
- Frustum culling becomes per-section, enabling vertical culling (don't render underground sections when looking at the sky).

### 4.3 New Biomes [MODIFY generation.py]

Expand from 7 to **15+ biomes** using a Whittaker-style temperature × humidity × altitude classification:

| Biome | Temperature | Humidity | Features |
|---|---|---|---|
| Ocean | any | any (deep water) | Deep water, coral (decorative), sand bottom |
| Deep Ocean | any | any (deeper) | Very deep, underwater caves, clay deposits |
| Beach | warm-med | any (shore) | Sand, palm trees (special oak variant) |
| Desert | hot | dry | Sand, sandstone layers, cacti, dead bushes, desert temples |
| Savanna | hot | medium | Tall grass, acacia-style trees (flat top), orange clay |
| Plains | medium | medium | Grass, flowers, villages |
| Forest | medium | humid | Dense oak trees, mushrooms |
| Dark Forest | medium | very humid | Tall oaks, dense canopy, dark ground, giant mushrooms |
| Birch Forest | medium | medium | Birch trees only, lighter feel |
| Taiga | cold | medium | Spruce trees (tall conifers), ferns, wolves |
| Snowy Tundra | cold | dry | Snow, ice, sparse spruce, igloos |
| Mountains | any | any (high altitude) | Extreme height, exposed stone, snow peaks, goats |
| Jungle | hot | very humid | Giant trees (2×2 trunk), vines, dense undergrowth, parrots |
| Swamp | medium | very humid | Shallow water, lily pads, mangrove-style trees, slimes |
| Mushroom Island | special | special | Mycelium surface, giant mushrooms, no hostile mobs |

**Implementation approach:**
- Add new block types as needed: acacia_log, spruce_log, spruce_leaves, jungle_log, jungle_leaves, mycelium, clay, ice, packed_ice, fern, vine, lily_pad.
- Create biome-specific tree generators: `generate_oak`, `generate_birch`, `generate_spruce`, `generate_jungle_giant`, `generate_acacia`.
- Biome assignment uses two noise maps (temperature, humidity) quantized into biome regions — extend the existing system.

### 4.4 River Generation [NEW section in generation.py]

**Algorithm:**
1. Generate a 2D "river noise" using a separate simplex noise seed.
2. Rivers exist where `abs(river_noise) < threshold` (values near zero form thin bands — river beds).
3. Carve the river bed: lower terrain height by 4–6 blocks along the river path. Fill with water up to sea level.
4. River banks: place sand/gravel along the edges (2-block border).
5. River width: varies with noise frequency. Typical: 6–12 blocks wide.

**Challenge:** Rivers must look natural when crossing chunk boundaries. Since the noise function is deterministic, the same river carving will produce identical results on both sides of a chunk border.

### 4.5 Improved Caves [MODIFY generation.py]

**Current issues:** Half-resolution 3D noise creates blocky-looking caves. Only two cave types.

**Improvements:**
1. **Full-resolution 3D noise** for cave carving (requires performance optimization — see Phase 10 for Numba JIT). As a compromise, use full-resolution near the surface (y > 32) and half-resolution deep underground.
2. **Cheese caves**: Large open caverns formed by a separate low-frequency 3D noise. These are the "rooms" you can find deep underground.
3. **Noodle caves**: Thinner, winding tunnels (higher-frequency noise with lower threshold) connecting cheese caves.
4. **Surface caves / ravines**: Occasionally carve deep vertical canyons (ravines) visible from the surface. Generated by a stretched 2D noise check combined with a vertical carving mask.
5. **Dripstone / Stalactites**: In cheese caves, hang pointed stone formations from ceilings and grow them from floors using a heightmap-based algorithm per cave column.
6. **Lush caves**: Below jungles/forests, caves have moss, glow berries (emissive blocks hanging from ceiling), and clay.

### 4.6 Ore Veins [MODIFY generation.py]

Replace uniform random ore scattering with vein-based generation:

```python
# Current: for each block, random chance based on depth → single ore block
# New: generate "vein centers" using noise, then grow clusters around each center

def generate_ore_vein(chunk_blocks, vein_center, ore_type, vein_size):
    """Grows an irregular blob of ore blocks around a center point.
    Uses 3D simplex noise to determine vein shape (organic, not spherical).
    vein_size: number of ore blocks (e.g., coal=8-16, diamond=2-6)."""
```

Ore distribution by depth (256-height world):
| Ore | Y Range | Vein Size | Frequency |
|---|---|---|---|
| Coal | 0–192 | 8–16 | Common |
| Iron | 0–128 | 4–8 | Common |
| Gold | 0–48 | 4–6 | Uncommon |
| Diamond | 0–24 | 2–6 | Rare |
| Lapis Lazuli | 0–48 | 4–8 | Uncommon |
| Emerald | 0–32 (mountains only) | 1–2 | Very Rare |
| Copper | 32–128 | 6–12 | Common |

### 4.7 New Block Types [MODIFY configs/blocks.json, atlas.py]

Add blocks needed by new biomes and features:

**Nature:** spruce_log, spruce_leaves, jungle_log, jungle_leaves, acacia_log, acacia_leaves, vine, lily_pad, fern, mycelium, podzol, clay, mud, moss_block, glow_berry_vine (emissive), dripstone.

**Ores:** lapis_lazuli_ore, emerald_ore, copper_ore, redstone_ore (emissive when walked on).

**Building:** smooth_stone, polished_granite, polished_diorite, polished_andesite, chiseled_stone_bricks, cracked_stone_bricks.

**Ice/Snow:** ice (transparent, slippery), packed_ice (opaque), blue_ice (opaque, very slippery).

Each new block needs: albedo tile + normal map + MRAO map + entry in blocks.json.

---

## Verification

1. Generate a new world with height=256. Fly to y=200 and look down — mountains should be grand and impressive.
2. Walk through multiple biomes — each should feel distinct (different trees, colors, terrain shape).
3. Dig down — caves should have variety (narrow tunnels, large caverns, ravines).
4. Find ore veins — they should be clustered, not single random blocks.
5. Run `tools/logic_test.py` — all tests must pass with new height.
6. Performance: chunk generation time should stay under 25 ms despite height increase (optimize with early Y termination).
