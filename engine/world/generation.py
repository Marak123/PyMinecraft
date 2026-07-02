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

SEA_LEVEL = 48
SNOW_LINE = 96

# Biome ids (internal to the generator).
OCEAN, BEACH, DESERT, SNOWY, MOUNTAIN, FOREST, PLAINS = range(7)

# Trees need context from neighbouring columns, so climate/height fields are
# computed on a grid padded by the maximum canopy radius.
_PAD = 3
_EXT = CHUNK_X + 2 * _PAD


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class WorldGenerator:
    def __init__(self, seed: int, registry: BlockRegistry) -> None:
        self.seed = seed
        self.noise = NoiseField(seed)
        self.registry = registry

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
        self._snowy_grass = ids("snowy_grass")
        self._snow_block = ids("snow_block")
        self._sandstone = ids("sandstone")
        self._bedrock = ids("bedrock")
        self._tall_grass = ids("tall_grass")
        self._flower_red = ids("flower_red")
        self._flower_yellow = ids("flower_yellow")
        self._ores = (
            (ids("coal_ore"), 0.60, 16, 100, 0.110),
            (ids("iron_ore"), 0.615, 6, 70, 0.115),
            (ids("gold_ore"), 0.655, 4, 34, 0.120),
            (ids("diamond_ore"), 0.675, 2, 18, 0.130),
        )

        # Per-biome surface blocks, indexed by biome id.
        self._top = np.array(
            [self._sand, self._sand, self._sand, self._snowy_grass,
             self._stone, self._grass, self._grass],
            dtype=np.uint8,
        )
        self._filler = np.array(
            [self._sand, self._sand, self._sand, self._dirt,
             self._stone, self._dirt, self._dirt],
            dtype=np.uint8,
        )
        self._tree_chance = np.array(
            [0.0, 0.0, 0.0, 0.018, 0.0, 0.055, 0.006], dtype=np.float64
        )

    # -- climate & height fields ----------------------------------------------
    def _height_field(self, wx: np.ndarray, wz: np.ndarray) -> np.ndarray:
        n = self.noise
        # Gentle domain warp breaks up the "obviously Perlin" coastlines.
        warp = 28.0 * n.fbm2(wx * 0.004 + 570.0, wz * 0.004 + 570.0, octaves=2)
        cx = wx + warp
        cz = wz - warp

        continents = n.fbm2(cx * 0.0016, cz * 0.0016, octaves=4)
        hills = n.fbm2(wx * 0.010 + 130.0, wz * 0.010 + 130.0, octaves=4) * 7.0

        mountain_region = _smoothstep(
            0.18, 0.55, n.fbm2(wx * 0.0009 + 77.0, wz * 0.0009 + 77.0, octaves=3)
        )
        ridges = n.ridged2(wx * 0.006 + 900.0, wz * 0.006 + 900.0, octaves=4) ** 2

        height = 46.0 + continents * 26.0 + hills + mountain_region * ridges * 58.0
        return np.clip(height, 6, CHUNK_Y - 6).astype(np.int32)

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
        conditions = [
            height < SEA_LEVEL - 1,
            height <= SEA_LEVEL + 1,
            height > 86,
            (temp > 0.62) & (hum < 0.45),
            temp < 0.30,
            hum > 0.52,
        ]
        choices = [OCEAN, BEACH, MOUNTAIN, DESERT, SNOWY, FOREST]
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
            & (h <= 80)
            & np.isin(biome, (PLAINS, FOREST, SNOWY))
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
        self._scatter_plants(blocks, cx, cz, h, biome)
        blocks[:, :, 0] = self._bedrock
        return blocks

    # -- passes ---------------------------------------------------------------
    def _fill_terrain(self, h: np.ndarray, biome: np.ndarray) -> np.ndarray:
        blocks = np.zeros((CHUNK_X, CHUNK_Z, CHUNK_Y), dtype=np.uint8)
        y = np.arange(CHUNK_Y, dtype=np.int32)[None, None, :]
        h3 = h[:, :, None]

        blocks[y <= h3] = self._stone

        # Ocean floor material varies with depth: shallow sand, deep gravel.
        top = self._top[biome].copy()
        filler = self._filler[biome].copy()
        deep_ocean = (biome == OCEAN) & (h < 40)
        top[deep_ocean] = self._gravel
        filler[deep_ocean] = self._gravel
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
        return blocks

    def _carve_caves(
        self, blocks: np.ndarray, cx: int, cz: int, h: np.ndarray
    ) -> None:
        n = self.noise
        x3 = np.arange(cx * CHUNK_X, (cx + 1) * CHUNK_X, dtype=np.float32)[:, None, None]
        z3 = np.arange(cz * CHUNK_Z, (cz + 1) * CHUNK_Z, dtype=np.float32)[None, :, None]
        y3 = np.arange(CHUNK_Y, dtype=np.float32)[None, None, :]

        # Two independent noise "sheets"; tunnels appear where both are near
        # zero — the classic spaghetti-cave intersection trick.
        t1 = n.perlin3(x3 * 0.022, y3 * 0.034, z3 * 0.022)
        t2 = n.perlin3(x3 * 0.022 + 400.0, y3 * 0.034 + 400.0, z3 * 0.022 + 400.0)
        spaghetti = (np.abs(t1) < 0.062) & (np.abs(t2) < 0.062)

        cheese = n.fbm3(x3 * 0.013 + 800.0, y3 * 0.021 + 800.0, z3 * 0.013 + 800.0, octaves=2)
        caverns = (cheese > 0.46) & (y3 < 40)

        y_idx = np.arange(CHUNK_Y, dtype=np.int32)[None, None, :]
        # Keep a >=5 block roof so caves never breach the surface (and never
        # puncture the ocean floor — no fluid sim yet to handle the flood).
        depth_ok = (y_idx >= 4) & (y_idx <= (h[:, :, None] - 5))
        carve = (spaghetti | caverns) & depth_ok

        blocks[carve] = AIR
        blocks[carve & (y_idx <= 10)] = self._lava

    def _place_ores(self, blocks: np.ndarray, cx: int, cz: int) -> None:
        n = self.noise
        x3 = np.arange(cx * CHUNK_X, (cx + 1) * CHUNK_X, dtype=np.float32)[:, None, None]
        z3 = np.arange(cz * CHUNK_Z, (cz + 1) * CHUNK_Z, dtype=np.float32)[None, :, None]
        y3 = np.arange(CHUNK_Y, dtype=np.float32)[None, None, :]
        stone_mask = blocks == self._stone

        for i, (ore_id, threshold, y_min, y_max, freq) in enumerate(self._ores):
            offset = 1300.0 + i * 137.3
            field = n.perlin3(x3 * freq + offset, y3 * freq + offset, z3 * freq + offset)
            vein = (field > threshold) & (y3 >= y_min) & (y3 <= y_max) & stone_mask
            blocks[vein] = ore_id

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
        for tx, tz in np.argwhere(candidates):
            base_x = int(tx) - _PAD  # chunk-local column of the trunk
            base_z = int(tz) - _PAD
            ground = int(h_e[tx, tz])
            trunk_h = 4 + int(trunk_roll[tx, tz] * 3.0)
            self._write_tree(blocks, base_x, base_z, ground, trunk_h)

    def _write_tree(
        self, blocks: np.ndarray, bx: int, bz: int, ground: int, trunk_h: int
    ) -> None:
        def put(px: int, pz: int, py: int, block_id: int, only_air: bool) -> None:
            if 0 <= px < CHUNK_X and 0 <= pz < CHUNK_Z and 0 < py < CHUNK_Y - 1:
                if not only_air or blocks[px, pz, py] == AIR:
                    blocks[px, pz, py] = block_id

        put(bx, bz, ground, self._dirt, only_air=False)
        for dy in range(1, trunk_h + 1):
            put(bx, bz, ground + dy, self._log, only_air=False)
        # Canopy: two 5x5 layers (corners trimmed), a 3x3, and a plus on top.
        for dy in (trunk_h - 1, trunk_h):
            for dx in range(-2, 3):
                for dz in range(-2, 3):
                    if abs(dx) == 2 and abs(dz) == 2:
                        continue
                    put(bx + dx, bz + dz, ground + dy, self._leaves, only_air=True)
        for dx in range(-1, 2):
            for dz in range(-1, 2):
                if abs(dx) == 1 and abs(dz) == 1:
                    continue
                put(bx + dx, bz + dz, ground + trunk_h + 1, self._leaves, only_air=True)
        put(bx, bz, ground + trunk_h + 1, self._leaves, only_air=True)

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

        for plant_id, lo, hi in (
            (self._tall_grass, 0.0, 0.10),
            (self._flower_red, 0.10, 0.115),
            (self._flower_yellow, 0.115, 0.13),
        ):
            sel = on_grass & (roll >= lo) & (roll < hi)
            blocks[ix[sel], iz[sel], above[sel]] = plant_id


def surface_height(blocks: np.ndarray, x: int, z: int, registry: BlockRegistry) -> int:
    """Highest solid y in a column — used to find a safe spawn point."""
    column = blocks[x, z, :]
    solid = registry.solid[column]
    ys = np.nonzero(solid)[0]
    return int(ys[-1]) if len(ys) else SEA_LEVEL
