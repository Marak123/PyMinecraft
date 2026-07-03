"""Procedural texture tiles.

The game ships zero image assets: every 16x16 tile is painted here with
deterministic per-tile RNG, so the whole look is reproducible and mods can
register new painters (or later, real image files) without engine changes.

Tiles are stacked into a GPU texture *array* (one layer per tile) instead of
a classic atlas — no UV bleeding, mipmaps per layer for free.

Painting helpers favour *organic* looks: smoothed blotch fields for natural
materials, voronoi cells for cobble, and pixel-art masks for UI icons.
"""

from __future__ import annotations

import zlib
from typing import Callable

import numpy as np

from engine.core.log import get_logger

_log = get_logger("atlas")

TILE = 16

Painter = Callable[[np.random.Generator], np.ndarray]
_PAINTERS: dict[str, Painter] = {}


def painter(name: str) -> Callable[[Painter], Painter]:
    def register(fn: Painter) -> Painter:
        _PAINTERS[name] = fn
        return fn
    return register


# -- painting helpers -----------------------------------------------------------
def _flat(rgb: tuple[int, int, int], alpha: int = 255) -> np.ndarray:
    tile = np.zeros((TILE, TILE, 4), dtype=np.uint8)
    tile[:, :, 0], tile[:, :, 1], tile[:, :, 2], tile[:, :, 3] = *rgb, alpha
    return tile


def _speckle(tile: np.ndarray, rng: np.random.Generator, amount: int) -> np.ndarray:
    noise = rng.integers(-amount, amount + 1, size=(TILE, TILE, 1), dtype=np.int16)
    rgb = tile[:, :, :3].astype(np.int16) + noise
    tile[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return tile


def _blotch_field(rng: np.random.Generator, passes: int = 2) -> np.ndarray:
    """Smooth random field in [-1, 1] — organic patches instead of static."""
    field = rng.random((TILE, TILE)).astype(np.float32)
    for _ in range(passes):
        field = sum(
            np.roll(np.roll(field, dx, 0), dz, 1)
            for dx in (-1, 0, 1)
            for dz in (-1, 0, 1)
        ) / 9.0
    field -= field.mean()
    peak = np.abs(field).max() or 1.0
    return field / peak


def _blotch(tile: np.ndarray, rng: np.random.Generator, amount: int) -> np.ndarray:
    field = (_blotch_field(rng) * amount).astype(np.int16)[:, :, None]
    tile[:, :, :3] = np.clip(tile[:, :, :3].astype(np.int16) + field, 0, 255).astype(np.uint8)
    return tile


def _shade_rows(tile: np.ndarray, rows, factor: float) -> np.ndarray:
    tile[rows, :, :3] = (tile[rows, :, :3] * factor).astype(np.uint8)
    return tile


def _blobs(
    tile: np.ndarray,
    rng: np.random.Generator,
    color: tuple[int, int, int],
    clusters: int,
) -> np.ndarray:
    """Ore clusters: a bright core with darker fringe pixels."""
    base = np.array(color, dtype=np.int16)
    for _ in range(clusters):
        cx, cy = rng.integers(2, TILE - 2, size=2)
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            x, y = int(cx + dx), int(cy + dy)
            tile[y, x, :3] = np.clip(base + rng.integers(-14, 15, 3), 0, 255)
        for dx, dy in ((-1, 0), (2, 0), (0, -1), (0, 2), (2, 1), (-1, 1)):
            x, y = int(cx + dx) % TILE, int(cy + dy) % TILE
            tile[y, x, :3] = np.clip(base * 6 // 10 + rng.integers(-10, 11, 3), 0, 255)
    return tile


def _mask_paint(rows: list[str], palette: dict[str, tuple[int, int, int, int]]) -> np.ndarray:
    """Pixel-art painter: 16 strings of 16 chars mapped through a palette."""
    tile = np.zeros((TILE, TILE, 4), dtype=np.uint8)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch in palette:
                tile[y, x] = palette[ch]
    return tile


def _stone_base(rng: np.random.Generator) -> np.ndarray:
    tile = _blotch(_flat((126, 126, 126)), rng, 14)
    tile = _speckle(tile, rng, 7)
    # A few short diagonal cracks give stone its rocky read.
    for _ in range(3):
        x, y = rng.integers(1, TILE - 5, size=2)
        for step in range(rng.integers(3, 6)):
            px, py = int(x + step), int(y + step // 2)
            if px < TILE and py < TILE:
                tile[py, px, :3] = (tile[py, px, :3] * 0.78).astype(np.uint8)
    return tile


# -- terrain -------------------------------------------------------------------
@painter("stone")
def _p_stone(rng: np.random.Generator) -> np.ndarray:
    return _stone_base(rng)


@painter("dirt")
def _p_dirt(rng: np.random.Generator) -> np.ndarray:
    tile = _speckle(_blotch(_flat((134, 96, 67)), rng, 16), rng, 10)
    for _ in range(5):  # tiny embedded pebbles
        x, y = rng.integers(0, TILE, size=2)
        tile[y, x, :3] = (150, 143, 134)
    return tile


@painter("grass_top")
def _p_grass_top(rng: np.random.Generator) -> np.ndarray:
    tile = _blotch(_flat((100, 158, 55)), rng, 18)
    tile = _speckle(tile, rng, 9)
    blades = rng.random((TILE, TILE)) < 0.10  # bright young blades
    tile[blades, :3] = np.clip(
        tile[blades, :3].astype(np.int16) + np.array([12, 26, 8]), 0, 255
    ).astype(np.uint8)
    return tile


@painter("grass_side")
def _p_grass_side(rng: np.random.Generator) -> np.ndarray:
    tile = _p_dirt(rng)
    grass = _p_grass_top(rng)
    tile[0:3] = grass[0:3]
    ragged = rng.random(TILE) < 0.5
    tile[3, ragged] = grass[3, ragged]
    deeper = rng.random(TILE) < 0.18
    tile[4, deeper] = grass[4, deeper]
    return tile


@painter("sand")
def _p_sand(rng: np.random.Generator) -> np.ndarray:
    tile = _blotch(_flat((219, 207, 163)), rng, 8)
    # Wind ripples: soft darker rows drifting across the tile.
    yy = np.arange(TILE)[:, None] + (np.arange(TILE)[None, :] // 5)
    ripple = (np.sin(yy * 1.25) * 7).astype(np.int16)[:, :, None]
    tile[:, :, :3] = np.clip(tile[:, :, :3].astype(np.int16) - ripple, 0, 255).astype(np.uint8)
    return _speckle(tile, rng, 5)


@painter("sandstone")
def _p_sandstone(rng: np.random.Generator) -> np.ndarray:
    tile = _speckle(_blotch(_flat((212, 200, 157)), rng, 7), rng, 5)
    _shade_rows(tile, slice(0, 1), 0.90)
    _shade_rows(tile, slice(5, 6), 0.86)
    _shade_rows(tile, slice(10, 11), 0.88)
    _shade_rows(tile, slice(15, 16), 0.84)
    return tile


@painter("gravel")
def _p_gravel(rng: np.random.Generator) -> np.ndarray:
    tile = _blotch(_flat((131, 127, 126)), rng, 10)
    for _ in range(16):  # loose pebbles of varying tone
        x, y = rng.integers(0, TILE - 1, size=2)
        shade = int(rng.integers(-40, 41))
        patch = np.clip(tile[y : y + 2, x : x + 2, :3].astype(np.int16) + shade, 0, 255)
        tile[y : y + 2, x : x + 2, :3] = patch.astype(np.uint8)
    return tile


@painter("bedrock")
def _p_bedrock(rng: np.random.Generator) -> np.ndarray:
    return _speckle(_blotch(_flat((62, 62, 64)), rng, 26), rng, 20)


@painter("snow")
def _p_snow(rng: np.random.Generator) -> np.ndarray:
    tile = _blotch(_flat((241, 246, 251)), rng, 5)
    sparkle = rng.random((TILE, TILE)) < 0.05
    tile[sparkle, :3] = 255
    return tile


@painter("snow_side")
def _p_snow_side(rng: np.random.Generator) -> np.ndarray:
    tile = _p_dirt(rng)
    tile[0:4] = _p_snow(rng)[0:4]
    return tile


# -- liquids -------------------------------------------------------------------
@painter("water")
def _p_water(rng: np.random.Generator) -> np.ndarray:
    tile = _blotch(_flat((50, 108, 219)), rng, 9)
    highlights = rng.random((TILE, TILE)) < 0.06
    tile[highlights, :3] = np.clip(
        tile[highlights, :3].astype(np.int16) + np.array([25, 30, 30]), 0, 255
    ).astype(np.uint8)
    return tile


@painter("lava")
def _p_lava(rng: np.random.Generator) -> np.ndarray:
    tile = _blotch(_flat((203, 84, 16)), rng, 26)
    field = _blotch_field(rng, passes=1)
    hot = field > 0.35
    tile[hot, :3] = (255, 196, 72)
    crust = field < -0.45
    tile[crust, :3] = (96, 32, 12)
    return tile


# -- wood & plants ------------------------------------------------------------
@painter("log_side")
def _p_log_side(rng: np.random.Generator) -> np.ndarray:
    tile = _speckle(_flat((104, 82, 49)), rng, 7)
    for col in range(TILE):  # vertical bark grain
        dark = 16 if col % 4 == 0 else (8 if col % 4 == 2 else 0)
        if dark:
            tile[:, col, :3] = np.clip(
                tile[:, col, :3].astype(np.int16) - dark - rng.integers(0, 7), 0, 255
            ).astype(np.uint8)
    for _ in range(4):  # bark knots
        x, y = rng.integers(0, TILE, size=2)
        tile[y, x, :3] = (78, 58, 33)
    return tile


@painter("birch_log_side")
def _p_birch_log(rng: np.random.Generator) -> np.ndarray:
    tile = _speckle(_flat((222, 219, 209)), rng, 6)
    for _ in range(7):  # the characteristic dark horizontal dashes
        x = int(rng.integers(0, TILE - 3))
        y = int(rng.integers(0, TILE))
        w = int(rng.integers(2, 5))
        tile[y, x : x + w, :3] = (56, 52, 46)
    return tile


@painter("log_top")
def _p_log_top(rng: np.random.Generator) -> np.ndarray:
    tile = _speckle(_flat((104, 82, 49)), rng, 5)
    yy, xx = np.mgrid[0:TILE, 0:TILE]
    dist = np.sqrt((xx - 7.5) ** 2 + (yy - 7.5) ** 2)
    rings = (dist.astype(np.int32) % 3) == 0
    tile[rings] = np.array([178, 143, 88, 255], dtype=np.uint8)
    tile[dist > 7.4] = np.array([92, 72, 42, 255], dtype=np.uint8)  # bark rim
    return tile


@painter("planks")
def _p_planks(rng: np.random.Generator) -> np.ndarray:
    tile = _speckle(_blotch(_flat((162, 130, 78)), rng, 8), rng, 6)
    for row in (3, 7, 11, 15):  # board gaps
        _shade_rows(tile, slice(row, row + 1), 0.66)
    for row0, col in ((0, 11), (4, 3), (8, 13), (12, 6)):  # board ends
        tile[row0 : row0 + 4, col, :3] = (tile[row0 : row0 + 4, col, :3] * 0.74).astype(np.uint8)
        nail_y = row0 + 1
        tile[nail_y, min(col + 1, TILE - 1), :3] = (96, 82, 58)
    # subtle horizontal grain
    grain = (np.sin(np.arange(TILE) * 2.2)[None, :] * 5).astype(np.int16)[:, :, None] * 0
    tile[:, :, :3] = np.clip(tile[:, :, :3].astype(np.int16) + grain, 0, 255).astype(np.uint8)
    return tile


@painter("leaves")
def _p_leaves(rng: np.random.Generator) -> np.ndarray:
    tile = _blotch(_flat((56, 122, 34)), rng, 26)
    tile = _speckle(tile, rng, 12)
    holes = rng.random((TILE, TILE)) < 0.16
    tile[holes, 3] = 0
    return tile


@painter("birch_leaves")
def _p_birch_leaves(rng: np.random.Generator) -> np.ndarray:
    tile = _blotch(_flat((96, 148, 62)), rng, 24)
    tile = _speckle(tile, rng, 12)
    holes = rng.random((TILE, TILE)) < 0.16
    tile[holes, 3] = 0
    return tile


@painter("tall_grass")
def _p_tall_grass(rng: np.random.Generator) -> np.ndarray:
    tile = _flat((0, 0, 0), alpha=0)
    for _ in range(10):
        x = int(rng.integers(1, TILE - 1))
        height = int(rng.integers(6, 14))
        shade = rng.integers(-24, 25)
        color = np.clip(np.array([90, 150, 54]) + shade, 0, 255)
        tile[TILE - height : TILE, x, :3] = color
        tile[TILE - height : TILE, x, 3] = 255
        if height > 9:  # bend the tip of tall blades
            tip = TILE - height
            bend = x + (1 if rng.random() < 0.5 else -1)
            if 0 <= bend < TILE:
                tile[tip : tip + 2, bend, :3] = color
                tile[tip : tip + 2, bend, 3] = 255
    return tile


def _flower(rng: np.random.Generator, petal: tuple[int, int, int]) -> np.ndarray:
    tile = _flat((0, 0, 0), alpha=0)
    tile[9:16, 8, :3] = (60, 125, 40)
    tile[9:16, 8, 3] = 255
    tile[11, 9, :3] = (60, 125, 40)  # leaf
    tile[11, 9, 3] = 255
    tile[4:8, 7:10, :3] = petal
    tile[4:8, 7:10, 3] = 255
    tile[3, 8, :3] = petal
    tile[3, 8, 3] = 255
    tile[5:7, 8, :3] = np.clip(np.array(petal, dtype=np.int16) - 80, 0, 255).astype(np.uint8)
    return tile


@painter("flower_red")
def _p_flower_red(rng: np.random.Generator) -> np.ndarray:
    return _flower(rng, (214, 48, 38))


@painter("flower_yellow")
def _p_flower_yellow(rng: np.random.Generator) -> np.ndarray:
    return _flower(rng, (240, 222, 62))


@painter("dead_bush")
def _p_dead_bush(rng: np.random.Generator) -> np.ndarray:
    tile = _flat((0, 0, 0), alpha=0)
    color = np.array([123, 92, 51], dtype=np.uint8)
    tile[10:16, 8, :3] = color
    tile[10:16, 8, 3] = 255
    for _ in range(6):  # crooked twigs
        x, y = 8, int(rng.integers(10, 14))
        dx = 1 if rng.random() < 0.5 else -1
        for step in range(int(rng.integers(2, 5))):
            x += dx if rng.random() < 0.7 else 0
            y -= 1
            if 0 <= x < TILE and 0 <= y < TILE:
                jitter = color.astype(np.int16) + rng.integers(-14, 15, 3)
                tile[y, x, :3] = np.clip(jitter, 0, 255).astype(np.uint8)
                tile[y, x, 3] = 255
    return tile


def _mushroom(rng: np.random.Generator, cap: tuple[int, int, int], dots: bool) -> np.ndarray:
    tile = _flat((0, 0, 0), alpha=0)
    tile[11:16, 7:9, :3] = (228, 220, 200)  # stem
    tile[11:16, 7:9, 3] = 255
    tile[8:11, 5:11, :3] = cap
    tile[8:11, 5:11, 3] = 255
    tile[7, 6:10, :3] = cap
    tile[7, 6:10, 3] = 255
    if dots:
        for x, y in ((6, 9), (9, 8), (8, 10)):
            tile[y, x, :3] = (240, 236, 228)
    return tile


@painter("mushroom_red")
def _p_mushroom_red(rng: np.random.Generator) -> np.ndarray:
    return _mushroom(rng, (196, 46, 38), dots=True)


@painter("mushroom_brown")
def _p_mushroom_brown(rng: np.random.Generator) -> np.ndarray:
    return _mushroom(rng, (148, 106, 76), dots=False)


@painter("torch")
def _p_torch(rng: np.random.Generator) -> np.ndarray:
    tile = _flat((0, 0, 0), alpha=0)
    tile[6:16, 7:9, :3] = (110, 86, 52)  # handle
    tile[6:16, 7:9, 3] = 255
    tile[4:6, 7:9, :3] = (255, 216, 96)  # coal head
    tile[4:6, 7:9, 3] = 255
    tile[3, 7:9, :3] = (255, 244, 180)   # flame tip
    tile[3, 7:9, 3] = 255
    tile[4, 6, :3] = (255, 170, 60)
    tile[4, 6, 3] = 255
    tile[5, 9, :3] = (255, 170, 60)
    tile[5, 9, 3] = 255
    return tile


# -- building & ores -------------------------------------------------------------
@painter("cobble")
def _p_cobble(rng: np.random.Generator) -> np.ndarray:
    # Voronoi stones: irregular cells, per-stone tone, dark mortar between.
    points = rng.integers(0, TILE, size=(9, 2))
    tones = rng.integers(96, 152, size=9)
    yy, xx = np.mgrid[0:TILE, 0:TILE]
    # Wrapped distances keep the texture tileable.
    dx = np.abs(xx[None] - points[:, 0, None, None])
    dy = np.abs(yy[None] - points[:, 1, None, None])
    dx = np.minimum(dx, TILE - dx)
    dy = np.minimum(dy, TILE - dy)
    dist = dx * dx + dy * dy
    nearest = np.argsort(dist, axis=0)
    cell = nearest[0]
    tile = _flat((110, 110, 110))
    tile[:, :, 0] = tile[:, :, 1] = tile[:, :, 2] = tones[cell]
    tile = _speckle(tile, rng, 7)
    # Mortar: pixels whose two nearest stones are almost equidistant.
    d0 = np.take_along_axis(dist, nearest[0:1], axis=0)[0]
    d1 = np.take_along_axis(dist, nearest[1:2], axis=0)[0]
    mortar = (d1 - d0) <= 2
    tile[mortar, :3] = (tile[mortar, :3] * 0.55).astype(np.uint8)
    return tile


@painter("bricks")
def _p_bricks(rng: np.random.Generator) -> np.ndarray:
    tile = _speckle(_blotch(_flat((150, 74, 60)), rng, 12), rng, 7)
    mortar = (188, 180, 172)
    for row in (0, 4, 8, 12):  # horizontal mortar lines
        tile[row, :, :3] = mortar
    # Vertical joints, offset every other course (running bond pattern).
    for band_start, offset in ((1, 4), (5, 12), (9, 4), (13, 12)):
        for col in (offset % TILE, (offset + 8) % TILE):
            tile[band_start : band_start + 3, col, :3] = mortar
    return tile


@painter("glass")
def _p_glass(rng: np.random.Generator) -> np.ndarray:
    tile = _flat((0, 0, 0), alpha=0)
    border = np.array([205, 227, 232, 255], dtype=np.uint8)
    tile[0, :] = border
    tile[-1, :] = border
    tile[:, 0] = border
    tile[:, -1] = border
    for i in range(2, 7):  # diagonal shine
        tile[i, 8 - i] = (235, 245, 248, 255)
        tile[i + 1, 8 - i] = (235, 245, 248, 255)
    tile[10, 3] = (225, 240, 245, 150)
    tile[11, 4] = (225, 240, 245, 150)
    return tile


@painter("glowstone")
def _p_glowstone(rng: np.random.Generator) -> np.ndarray:
    tile = _blotch(_flat((188, 148, 92)), rng, 18)
    field = _blotch_field(rng, passes=1)
    bright = field > 0.25
    tile[bright, :3] = (252, 222, 130)
    core = field > 0.55
    tile[core, :3] = (255, 246, 196)
    return tile


@painter("coal_ore")
def _p_coal(rng: np.random.Generator) -> np.ndarray:
    return _blobs(_stone_base(rng), rng, (44, 44, 46), 3)


@painter("iron_ore")
def _p_iron(rng: np.random.Generator) -> np.ndarray:
    return _blobs(_stone_base(rng), rng, (219, 176, 145), 3)


@painter("gold_ore")
def _p_gold(rng: np.random.Generator) -> np.ndarray:
    return _blobs(_stone_base(rng), rng, (250, 230, 78), 3)


@painter("diamond_ore")
def _p_diamond(rng: np.random.Generator) -> np.ndarray:
    return _blobs(_stone_base(rng), rng, (102, 235, 240), 3)


# -- UI icons -----------------------------------------------------------------------
_HEART_ROWS = [
    "................",
    "..XX....XX......",
    ".XRRX..XRRX.....",
    "XRRWRXXRRRRX....",
    "XRRWRRRRRRRX....",
    "XRRRRRRRRRRX....",
    ".XRRRRRRRRX.....",
    "..XRRRRRRX......",
    "...XRRRRX.......",
    "....XRRX........",
    ".....XX.........",
    "................",
    "................",
    "................",
    "................",
    "................",
]


def _heart(palette_r: tuple[int, int, int, int]) -> np.ndarray:
    return _mask_paint(
        _HEART_ROWS,
        {
            "X": (46, 10, 10, 255),
            "R": palette_r,
            "W": (255, 200, 200, 255),
        },
    )


@painter("heart_full")
def _p_heart_full(rng: np.random.Generator) -> np.ndarray:
    return _heart((214, 36, 36, 255))


@painter("heart_half")
def _p_heart_half(rng: np.random.Generator) -> np.ndarray:
    tile = _heart((214, 36, 36, 255))
    empty = _heart((60, 26, 26, 255))
    tile[:, 6:] = empty[:, 6:]
    return tile


@painter("heart_empty")
def _p_heart_empty(rng: np.random.Generator) -> np.ndarray:
    return _heart((60, 26, 26, 255))


@painter("bubble")
def _p_bubble(rng: np.random.Generator) -> np.ndarray:
    tile = _flat((0, 0, 0), alpha=0)
    yy, xx = np.mgrid[0:TILE, 0:TILE]
    dist = np.sqrt((xx - 7.5) ** 2 + (yy - 6.5) ** 2)
    ring = (dist >= 3.4) & (dist <= 5.2)
    tile[ring] = (150, 200, 250, 235)
    tile[5, 5] = (230, 245, 255, 255)  # highlight
    tile[5, 6] = (230, 245, 255, 255)
    return tile


# -- build -----------------------------------------------------------------------
def _error_tile() -> np.ndarray:
    tile = _flat((255, 0, 255))
    tile[0:8, 8:16, :3] = (0, 0, 0)
    tile[8:16, 0:8, :3] = (0, 0, 0)
    return tile


def build_tiles(tile_names: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    """Paint all requested tiles. Returns (layers array (L,16,16,4), name->layer)."""
    layers = np.zeros((len(tile_names), TILE, TILE, 4), dtype=np.uint8)
    mapping: dict[str, int] = {}
    for i, name in enumerate(tile_names):
        fn = _PAINTERS.get(name)
        if fn is None:
            _log.warning("No painter for tile '%s' — using error tile", name)
            layers[i] = _error_tile()
        else:
            # Stable per-tile seed: same tile looks identical across runs.
            rng = np.random.default_rng(zlib.crc32(name.encode("utf-8")))
            layers[i] = fn(rng)
        mapping[name] = i
    _log.info("Painted %d procedural tiles", len(tile_names))
    return layers, mapping
