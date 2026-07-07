"""Procedural terrain generation.

The pipeline follows the spec's multi-pass philosophy — every pass is a
separate method so passes can be replaced independently:

    continents -> mountains -> climate -> biomes -> terrain fill
    -> caves -> ores -> water -> trees -> plants -> bedrock

Everything is a pure function of (seed, chunk coords), which makes chunk
generation embarrassingly parallel and border-seamless: trees are scattered
with a stateless world-coordinate hash, so a chunk can independently compute
trees rooted in neighbouring chunks and write only the overhanging parts.
"""

from __future__ import annotations

import numpy as np

from engine.core.log import get_logger
from engine.world.blocks import AIR, BlockRegistry
from engine.world.coords import CHUNK_X, CHUNK_Y, CHUNK_Z
from engine.world.noise import NoiseField, hash01

_log = get_logger("worldgen")

SEA_LEVEL = 64
SNOW_LINE = 150

# Biome ids (internal to the generator).
# Biome ids. The first 7 keep their historical values (save compatibility);
# richer biomes are appended (plan 4.3, Whittaker temperature x humidity).
(OCEAN, BEACH, DESERT, SNOWY, MOUNTAIN, FOREST, PLAINS,
 SAVANNA, TAIGA, JUNGLE, SWAMP, BIRCH_FOREST, DEEP_OCEAN) = range(13)
BIOME_COUNT = 13

# Trees need context from neighbouring columns, so climate/height fields are
# computed on a grid padded by the maximum canopy radius.
_PAD = 3
_EXT = CHUNK_X + 2 * _PAD


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _upsample2(field: np.ndarray) -> np.ndarray:
    """Double resolution by linear interpolation along every axis.

    Input shape (a+1, b+1, c+1) -> output (2a, 2b, 2c).  3D noise for caves
    and ores is sampled at half resolution and upsampled — 8x fewer gradient
    evaluations for shapes that are blobby anyway (profiled: generation
    dropped from ~60 ms to ~20 ms per chunk).
    """
    for axis in range(3):
        lo = field.take(range(field.shape[axis] - 1), axis=axis)
        hi = field.take(range(1, field.shape[axis]), axis=axis)
        mid = (lo + hi) * 0.5
        shape = list(lo.shape)
        shape[axis] *= 2
        out = np.empty(shape, dtype=field.dtype)
        even = [slice(None)] * 3
        odd = [slice(None)] * 3
        even[axis] = slice(0, None, 2)
        odd[axis] = slice(1, None, 2)
        out[tuple(even)] = lo
        out[tuple(odd)] = mid
        field = out
    return field


class WorldGenerator:
    def __init__(self, seed: int, registry: BlockRegistry) -> None:
        self.seed = seed
        self.noise = NoiseField(seed)
        self.registry = registry
        # Set by the game after construction to avoid an import cycle.
        self.structures = None

        ids = registry.id_of
        self._stone = ids("stone")
        self._dirt = ids("dirt")
        self._grass = ids("grass_block")
        self._sand = ids("sand")
        self._gravel = ids("gravel")
        self._water = ids("water")
        self._lava = ids("lava")
        self._log = ids("oak_log")
        self._leaves = ids("oak_leaves")
        self._birch_log = ids("birch_log")
        self._birch_leaves = ids("birch_leaves")
        self._snowy_grass = ids("snowy_grass")
        self._snow_block = ids("snow_block")
        self._sandstone = ids("sandstone")
        self._bedrock = ids("bedrock")
        self._glowstone = ids("glowstone")
        self._tall_grass = ids("tall_grass")
        self._flower_red = ids("flower_red")
        self._flower_yellow = ids("flower_yellow")
        self._dead_bush = ids("dead_bush")
        self._mushroom_red = ids("mushroom_red")
        self._mushroom_brown = ids("mushroom_brown")
        self._cobble = ids("cobblestone")
        self._mossy = ids("mossy_cobblestone")
        self._torch = ids("torch")
        self._spruce_log = ids("spruce_log")
        self._spruce_leaves = ids("spruce_leaves")
        self._jungle_log = ids("jungle_log")
        self._jungle_leaves = ids("jungle_leaves")
        self._acacia_log = ids("acacia_log")
        self._acacia_leaves = ids("acacia_leaves")
        self._ice = ids("ice")
        self._clay = ids("clay")
        self._fern = ids("fern")
        # Ore veins: (id, y_min, y_max, attempts_per_chunk, vein_size).
        self._ore_veins = (
            (ids("coal_ore"), 8, 192, 10, 13),
            (ids("copper_ore"), 32, 120, 6, 10),
            (ids("iron_ore"), 6, 128, 8, 7),
            (ids("lapis_ore"), 6, 48, 3, 6),
            (ids("redstone_ore"), 4, 40, 4, 7),
            (ids("gold_ore"), 4, 40, 3, 5),
            (ids("diamond_ore"), 4, 22, 2, 4),
            (ids("emerald_ore"), 6, 40, 1, 2),
        )

        # Per-biome surface/filler blocks, indexed by biome id (BIOME_COUNT).
        g, d, s, st = self._grass, self._dirt, self._sand, self._stone
        sn, gr = self._snowy_grass, self._gravel
        self._top = np.array(
            [s, s, s, sn, st, g, g,  g, sn, g, g, g, gr], dtype=np.uint8,
        )
        self._filler = np.array(
            [s, s, s, d, st, d, d,  d, d, d, d, d, gr], dtype=np.uint8,
        )
        # Tree spawn chance per biome (0 = no trees).
        self._tree_chance = np.array(
            [0.0, 0.0, 0.0, 0.02, 0.004, 0.06, 0.006,
             0.006, 0.05, 0.075, 0.03, 0.05, 0.0], dtype=np.float64,
        )

    # -- climate & height fields ----------------------------------------------
    def _height_field(self, wx: np.ndarray, wz: np.ndarray) -> np.ndarray:
        n = self.noise
        # Gentle domain warp breaks up the "obviously Perlin" coastlines.
        warp = 28.0 * n.fbm2(wx * 0.004 + 570.0, wz * 0.004 + 570.0, octaves=2)
        cx = wx + warp
        cz = wz - warp

        continents = n.fbm2(cx * 0.0016, cz * 0.0016, octaves=4)
        hills = n.fbm2(wx * 0.010 + 130.0, wz * 0.010 + 130.0, octaves=4) * 9.0

        mountain_region = _smoothstep(
            0.18, 0.55, n.fbm2(wx * 0.0009 + 77.0, wz * 0.0009 + 77.0, octaves=3)
        )
        ridges = n.ridged2(wx * 0.006 + 900.0, wz * 0.006 + 900.0, octaves=4) ** 2

        # 256-tall world: peaks push past y=200 in mountain regions.
        height = 61.0 + continents * 32.0 + hills + mountain_region * ridges * 130.0

        # Rivers: thin bands where a dedicated noise crosses zero carve
        # sea-level channels through any terrain below the high mountains.
        river = np.abs(n.fbm2(wx * 0.0031 + 3300.0, wz * 0.0031 + 3300.0, octaves=3))
        river_depth = (1.0 - _smoothstep(0.0, 0.045, river)) * 9.0
        carvable = height < 118.0
        riverbed = np.minimum(height, float(SEA_LEVEL) - 1.0 - river_depth * 0.5)
        height = np.where(carvable & (river_depth > 0.5), riverbed, height)

        return np.clip(height, 6, CHUNK_Y - 40).astype(np.int32)

    def _climate(
        self, wx: np.ndarray, wz: np.ndarray, height: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        n = self.noise
        temp = 0.5 + 0.5 * n.fbm2(wx * 0.0021 + 1000.0, wz * 0.0021 + 1000.0, octaves=3)
        temp -= np.maximum(0, height - 72) * 0.0045  # altitude cooling
        hum = 0.5 + 0.5 * n.fbm2(wx * 0.0019 + 2000.0, wz * 0.0019 + 2000.0, octaves=3)
        return temp, hum

    def _biomes(
        self, height: np.ndarray, temp: np.ndarray, hum: np.ndarray
    ) -> np.ndarray:
        # Whittaker-style: water depth first, then a temperature x humidity
        # grid for land (plan 4.3).
        conditions = [
            height < 34,
            height < SEA_LEVEL - 1,
            height <= SEA_LEVEL + 1,
            height > 165,
            # -- hot --
            (temp > 0.66) & (hum < 0.32),   # desert
            (temp > 0.62) & (hum < 0.55),   # savanna
            (temp > 0.60) & (hum >= 0.55),  # jungle
            # -- cold --
            (temp < 0.28),                  # snowy tundra
            (temp < 0.42) & (hum > 0.40),   # taiga
            # -- temperate --
            (hum > 0.66),                   # swamp
            (hum > 0.50) & (temp > 0.48),   # forest
            (hum > 0.44),                   # birch forest
        ]
        choices = [DEEP_OCEAN, OCEAN, BEACH, MOUNTAIN,
                   DESERT, SAVANNA, JUNGLE, SNOWY, TAIGA,
                   SWAMP, FOREST, BIRCH_FOREST]
        return np.select(conditions, choices, default=PLAINS).astype(np.uint8)

    def find_spawn(self, search_radius: int = 512, spacing: int = 8) -> tuple[int, int]:
        """Nearest comfortable spawn column: dry land, walkable biome.

        Samples the height/climate fields directly (no chunk generation), so
        the search over thousands of candidate columns costs milliseconds.
        """
        coords = np.arange(-search_radius, search_radius + 1, spacing, dtype=np.float32)
        wx, wz = np.meshgrid(coords, coords, indexing="ij")
        h = self._height_field(wx, wz)
        temp, hum = self._climate(wx, wz, h)
        biome = self._biomes(h, temp, hum)

        good = (
            (h >= SEA_LEVEL + 2)
            & (h <= 96)
            & np.isin(biome, (PLAINS, FOREST, BIRCH_FOREST, SAVANNA, TAIGA))
        )
        if not good.any():
            _log.warning("No comfortable spawn found within %d blocks", search_radius)
            return 0, 0
        dist2 = wx * wx + wz * wz
        dist2[~good] = np.inf
        ix, iz = np.unravel_index(np.argmin(dist2), dist2.shape)
        return int(wx[ix, iz]), int(wz[ix, iz])

    # -- main entry point -------------------------------------------------------
    def generate_chunk(self, cx: int, cz: int) -> np.ndarray:
        """Returns block ids, shape (CHUNK_X, CHUNK_Z, CHUNK_Y), dtype uint8."""
        # Extended-column fields (interior chunk = [_PAD:_PAD+16]).
        ex = np.arange(cx * CHUNK_X - _PAD, cx * CHUNK_X - _PAD + _EXT, dtype=np.float32)
        ez = np.arange(cz * CHUNK_Z - _PAD, cz * CHUNK_Z - _PAD + _EXT, dtype=np.float32)
        wx_e, wz_e = np.meshgrid(ex, ez, indexing="ij")

        h_e = self._height_field(wx_e, wz_e)
        temp_e, hum_e = self._climate(wx_e, wz_e, h_e)
        biome_e = self._biomes(h_e, temp_e, hum_e)

        core = slice(_PAD, _PAD + CHUNK_X)
        h = h_e[core, core]
        biome = biome_e[core, core]

        blocks = self._fill_terrain(h, biome)
        self._carve_caves(blocks, cx, cz, h)
        self._place_ores(blocks, cx, cz)
        self._fill_water(blocks, h)
        self._plant_trees(blocks, cx, cz, h_e, biome_e)
        self._place_ruins(blocks, cx, cz, h_e, biome_e)
        if self.structures is not None:
            self.structures.place(blocks, cx, cz, h)
        self._scatter_plants(blocks, cx, cz, h, biome)
        blocks[:, :, 0] = self._bedrock
        return blocks

    # -- passes ---------------------------------------------------------------
    def _fill_terrain(self, h: np.ndarray, biome: np.ndarray) -> np.ndarray:
        blocks = np.zeros((CHUNK_X, CHUNK_Z, CHUNK_Y), dtype=np.uint8)
        y = np.arange(CHUNK_Y, dtype=np.int32)[None, None, :]
        h3 = h[:, :, None]

        blocks[y <= h3] = self._stone

        # Ocean/river floor material varies with depth.
        top = self._top[biome].copy()
        filler = self._filler[biome].copy()
        deep = np.isin(biome, (OCEAN, DEEP_OCEAN)) & (h < 52)
        top[deep] = self._gravel
        filler[deep] = self._gravel
        # Clay patches on shallow submerged floors and swamp beds.
        clay = ((biome == SWAMP) | (np.isin(biome, (OCEAN, DEEP_OCEAN)) & (h > 48))) & (h < SEA_LEVEL)
        top[clay] = self._clay
        # High peaks get a snow cap regardless of biome table.
        snow_cap = (biome == MOUNTAIN) & (h >= SNOW_LINE)
        top[snow_cap] = self._snow_block

        ix, iz = np.meshgrid(np.arange(CHUNK_X), np.arange(CHUNK_Z), indexing="ij")
        for depth in (1, 2, 3):
            yy = np.maximum(h - depth, 1)
            blocks[ix, iz, yy] = filler
        # Desert gets a sandstone base under the loose sand.
        desert = biome == DESERT
        if desert.any():
            for depth in (4, 5):
                yy = np.maximum(h - depth, 1)
                sel = desert & (yy > 1)
                blocks[ix[sel], iz[sel], yy[sel]] = self._sandstone
        blocks[ix, iz, h] = top
        # Frozen lakes: an ice skin over cold biome water at sea level.
        frozen = np.isin(biome, (SNOWY, TAIGA)) & (h == SEA_LEVEL - 1)
        if frozen.any():
            blocks[ix[frozen], iz[frozen], SEA_LEVEL] = self._ice
        return blocks

    def _half_res_coords(
        self, cx: int, cz: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Half-resolution sample coordinates (one extra sample for lerp)."""
        x3 = np.arange(cx * CHUNK_X, cx * CHUNK_X + CHUNK_X + 1, 2, dtype=np.float32)
        z3 = np.arange(cz * CHUNK_Z, cz * CHUNK_Z + CHUNK_Z + 1, 2, dtype=np.float32)
        y3 = np.arange(0, CHUNK_Y + 1, 2, dtype=np.float32)
        return x3[:, None, None], z3[None, :, None], y3[None, None, :]

    def _carve_caves(
        self, blocks: np.ndarray, cx: int, cz: int, h: np.ndarray
    ) -> None:
        n = self.noise
        x3, z3, y3 = self._half_res_coords(cx, cz)

        # Two independent noise "sheets"; tunnels appear where both are near
        # zero — the classic spaghetti-cave intersection trick.  Sampled at
        # half resolution and upsampled (see _upsample2).
        t1 = _upsample2(n.perlin3(x3 * 0.022, y3 * 0.034, z3 * 0.022))
        t2 = _upsample2(
            n.perlin3(x3 * 0.022 + 400.0, y3 * 0.034 + 400.0, z3 * 0.022 + 400.0)
        )
        spaghetti = (np.abs(t1) < 0.062) & (np.abs(t2) < 0.062)

        cheese = _upsample2(
            n.fbm3(x3 * 0.013 + 800.0, y3 * 0.021 + 800.0, z3 * 0.013 + 800.0, octaves=2)
        )
        y_idx = np.arange(CHUNK_Y, dtype=np.int32)[None, None, :]
        caverns = (cheese > 0.46) & (y_idx < 52)

        # Keep a >=5 block roof so caves never breach the surface (and never
        # puncture the ocean floor — no fluid sim yet to handle the flood).
        depth_ok = (y_idx >= 4) & (y_idx <= (h[:, :, None] - 5))
        carve = (spaghetti | caverns) & self._ravine_mask(cx, cz, h) & depth_ok

        blocks[carve] = AIR
        blocks[carve & (y_idx <= 12)] = self._lava

        # Natural glowstone pockets on cavern ceilings — landmarks that show
        # off block lighting deep underground.
        carved_below = np.zeros_like(carve)
        carved_below[:, :, 1:] = carve[:, :, :-1]
        glow = (
            carved_below
            & (blocks == self._stone)
            & (cheese > 0.60)
            & (y_idx < 48)
        )
        blocks[glow] = self._glowstone

    def _ravine_mask(self, cx: int, cz: int, h: np.ndarray) -> np.ndarray:
        """Full-height carve mask; False inside a rare deep ravine slot.

        Ravines are thin vertical canyons: a stretched 2D noise near zero
        cuts a slit from bedrock up to just under the surface.
        """
        n = self.noise
        wx = np.arange(cx * CHUNK_X, (cx + 1) * CHUNK_X, dtype=np.float32)[:, None]
        wz = np.arange(cz * CHUNK_Z, (cz + 1) * CHUNK_Z, dtype=np.float32)[None, :]
        # Stretch Z so the canyon runs as a long crack, not a dot.
        field = np.abs(n.fbm2(wx * 0.010 + 6100.0, wz * 0.0022 + 6100.0, octaves=2))
        slit = field < 0.014  # (X, Z)
        y_idx = np.arange(CHUNK_Y, dtype=np.int32)[None, None, :]
        in_band = (y_idx >= 10) & (y_idx <= (h[:, :, None] - 4))
        return ~(slit[:, :, None] & in_band)

    def _place_ores(self, blocks: np.ndarray, cx: int, cz: int) -> None:
        """Vein-based ores: scatter centres, grow an organic blob at each
        (plan 4.6). Deterministic per chunk so it never depends on order."""
        stone_mask = blocks == self._stone
        rng = np.random.default_rng(
            ((cx * 341873128) ^ (cz * 132897987) ^ self.seed) & 0x7FFFFFFF
        )
        offsets = np.array(
            [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
             (0, 0, 1), (0, 0, -1), (1, 1, 0), (-1, 0, 1), (1, 0, -1),
             (0, 1, 1), (0, -1, -1), (1, -1, 0), (-1, 1, 0)],
            dtype=np.int32,
        )
        for ore_id, y_min, y_max, attempts, vein_size in self._ore_veins:
            for _ in range(attempts):
                lx = int(rng.integers(0, CHUNK_X))
                lz = int(rng.integers(0, CHUNK_Z))
                ly = int(rng.integers(y_min, y_max))
                grow = offsets[: min(vein_size, len(offsets))]
                px = np.clip(lx + grow[:, 0], 0, CHUNK_X - 1)
                pz = np.clip(lz + grow[:, 1] * 0 + grow[:, 2], 0, CHUNK_Z - 1)
                py = np.clip(ly + grow[:, 1], 1, CHUNK_Y - 1)
                place = stone_mask[px, pz, py]
                blocks[px[place], pz[place], py[place]] = ore_id

    def _fill_water(self, blocks: np.ndarray, h: np.ndarray) -> None:
        y = np.arange(CHUNK_Y, dtype=np.int32)[None, None, :]
        sea = (y > h[:, :, None]) & (y <= SEA_LEVEL) & (blocks == AIR)
        blocks[sea] = self._water

    def _plant_trees(
        self,
        blocks: np.ndarray,
        cx: int,
        cz: int,
        h_e: np.ndarray,
        biome_e: np.ndarray,
    ) -> None:
        wx0 = cx * CHUNK_X - _PAD
        wz0 = cz * CHUNK_Z - _PAD
        gx = np.arange(_EXT)[:, None] + wx0
        gz = np.arange(_EXT)[None, :] + wz0

        roll = hash01(self.seed, gx, gz, salt=11)
        chance = self._tree_chance[biome_e]
        candidates = (roll < chance) & (h_e > SEA_LEVEL)

        trunk_roll = hash01(self.seed, gx, gz, salt=12)
        variant_roll = hash01(self.seed, gx, gz, salt=13)
        for tx, tz in np.argwhere(candidates):
            base_x = int(tx) - _PAD  # chunk-local column of the trunk
            base_z = int(tz) - _PAD
            ground = int(h_e[tx, tz])
            biome = int(biome_e[tx, tz])
            trunk_h = 4 + int(trunk_roll[tx, tz] * 3.0)
            variant = variant_roll[tx, tz]

            if biome == TAIGA:
                self._write_conifer(blocks, base_x, base_z, ground,
                                    5 + int(trunk_roll[tx, tz] * 4.0),
                                    self._spruce_log, self._spruce_leaves)
                continue
            if biome == JUNGLE:
                log_id, leaves_id = self._jungle_log, self._jungle_leaves
                trunk_h = 6 + int(trunk_roll[tx, tz] * 4.0)
            elif biome == SAVANNA:
                log_id, leaves_id = self._acacia_log, self._acacia_leaves
            elif biome == BIRCH_FOREST or (biome == FOREST and variant < 0.30):
                log_id, leaves_id = self._birch_log, self._birch_leaves
            else:
                log_id, leaves_id = self._log, self._leaves
            self._write_tree(blocks, base_x, base_z, ground, trunk_h, log_id, leaves_id)

    def _write_conifer(
        self, blocks: np.ndarray, bx: int, bz: int, ground: int,
        trunk_h: int, log_id: int, leaves_id: int,
    ) -> None:
        """Tapered spruce: leaf rings that shrink towards a pointed top."""
        def put(px, pz, py, bid, only_air):
            if 0 <= px < CHUNK_X and 0 <= pz < CHUNK_Z and 0 < py < CHUNK_Y - 1:
                if not only_air or blocks[px, pz, py] == AIR:
                    blocks[px, pz, py] = bid

        put(bx, bz, ground, self._dirt, False)
        for dy in range(1, trunk_h + 1):
            put(bx, bz, ground + dy, log_id, False)
        radius = 2
        for dy in range(2, trunk_h + 1):
            r = radius if (trunk_h - dy) % 2 == 1 else max(1, radius - 1)
            if dy >= trunk_h - 1:
                r = 1
            for dx in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if abs(dx) + abs(dz) <= r:
                        put(bx + dx, bz + dz, ground + dy, leaves_id, True)
        put(bx, bz, ground + trunk_h + 1, leaves_id, True)

    def _write_tree(
        self,
        blocks: np.ndarray,
        bx: int,
        bz: int,
        ground: int,
        trunk_h: int,
        log_id: int,
        leaves_id: int,
    ) -> None:
        def put(px: int, pz: int, py: int, block_id: int, only_air: bool) -> None:
            if 0 <= px < CHUNK_X and 0 <= pz < CHUNK_Z and 0 < py < CHUNK_Y - 1:
                if not only_air or blocks[px, pz, py] == AIR:
                    blocks[px, pz, py] = block_id

        put(bx, bz, ground, self._dirt, only_air=False)
        for dy in range(1, trunk_h + 1):
            put(bx, bz, ground + dy, log_id, only_air=False)
        # Canopy: two 5x5 layers (corners trimmed), a 3x3, and a plus on top.
        for dy in (trunk_h - 1, trunk_h):
            for dx in range(-2, 3):
                for dz in range(-2, 3):
                    if abs(dx) == 2 and abs(dz) == 2:
                        continue
                    put(bx + dx, bz + dz, ground + dy, leaves_id, only_air=True)
        for dx in range(-1, 2):
            for dz in range(-1, 2):
                if abs(dx) == 1 and abs(dz) == 1:
                    continue
                put(bx + dx, bz + dz, ground + trunk_h + 1, leaves_id, only_air=True)
        put(bx, bz, ground + trunk_h + 1, leaves_id, only_air=True)

    def _place_ruins(
        self,
        blocks: np.ndarray,
        cx: int,
        cz: int,
        h_e: np.ndarray,
        biome_e: np.ndarray,
    ) -> None:
        """Rare ruined towers: crumbled cobblestone rings with a torch inside.

        Stateless like trees: any chunk rebuilds the same ruin from the world
        coordinates alone, so structures cross chunk borders seamlessly.
        """
        wx0 = cx * CHUNK_X - _PAD
        wz0 = cz * CHUNK_Z - _PAD
        gx = np.arange(_EXT)[:, None] + wx0
        gz = np.arange(_EXT)[None, :] + wz0
        roll = hash01(self.seed, gx, gz, salt=31)
        ok_biome = np.isin(biome_e, (PLAINS, FOREST, DESERT))
        candidates = (roll < 0.0011) & ok_biome & (h_e > SEA_LEVEL + 1) & (h_e < 84)

        for tx, tz in np.argwhere(candidates):
            base_x = int(tx) - _PAD
            base_z = int(tz) - _PAD
            ground = int(h_e[tx, tz])
            rng = np.random.default_rng(
                ((int(gx[tx, 0]) * 73856093) ^ (int(gz[0, tz]) * 19349663) ^ self.seed)
                & 0x7FFFFFFF
            )
            height = int(rng.integers(3, 5))

            def put(px: int, pz: int, py: int, block_id: int) -> None:
                if 0 <= px < CHUNK_X and 0 <= pz < CHUNK_Z and 0 < py < CHUNK_Y - 1:
                    blocks[px, pz, py] = block_id

            for dx in range(-2, 3):
                for dz in range(-2, 3):
                    stone = self._mossy if rng.random() < 0.35 else self._cobble
                    put(base_x + dx, base_z + dz, ground, stone)  # floor
                    on_ring = abs(dx) == 2 or abs(dz) == 2
                    if not on_ring:
                        # Clear the interior of grass/trees debris.
                        for dy in range(1, height + 1):
                            put(base_x + dx, base_z + dz, ground + dy, AIR)
                        continue
                    wall_h = int(rng.integers(1, height + 1))
                    if rng.random() < 0.2:
                        wall_h = 0  # crumbled gap
                    for dy in range(1, wall_h + 1):
                        stone = self._mossy if rng.random() < 0.35 else self._cobble
                        put(base_x + dx, base_z + dz, ground + dy, stone)
            put(base_x, base_z, ground + 1, self._torch)

    def _scatter_plants(
        self,
        blocks: np.ndarray,
        cx: int,
        cz: int,
        h: np.ndarray,
        biome: np.ndarray,
    ) -> None:
        ix, iz = np.meshgrid(np.arange(CHUNK_X), np.arange(CHUNK_Z), indexing="ij")
        wx = ix + cx * CHUNK_X
        wz = iz + cz * CHUNK_Z
        roll = hash01(self.seed, wx, wz, salt=21)

        above = np.minimum(h + 1, CHUNK_Y - 1)
        on_grass = (blocks[ix, iz, h] == self._grass) & (blocks[ix, iz, above] == AIR)
        on_sand = (blocks[ix, iz, h] == self._sand) & (blocks[ix, iz, above] == AIR)
        in_forest = biome == FOREST
        in_desert = biome == DESERT

        for plant_id, lo, hi, where in (
            (self._tall_grass, 0.0, 0.10, on_grass),
            (self._flower_red, 0.10, 0.115, on_grass),
            (self._flower_yellow, 0.115, 0.13, on_grass),
            (self._mushroom_red, 0.13, 0.137, on_grass & in_forest),
            (self._mushroom_brown, 0.137, 0.146, on_grass & in_forest),
            (self._dead_bush, 0.0, 0.030, on_sand & in_desert),
        ):
            sel = where & (roll >= lo) & (roll < hi)
            blocks[ix[sel], iz[sel], above[sel]] = plant_id


def surface_height(blocks: np.ndarray, x: int, z: int, registry: BlockRegistry) -> int:
    """Highest solid y in a column — used to find a safe spawn point."""
    column = blocks[x, z, :]
    solid = registry.solid[column]
    ys = np.nonzero(solid)[0]
    return int(ys[-1]) if len(ys) else SEA_LEVEL
