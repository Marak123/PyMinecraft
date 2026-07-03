"""Procedural texture tiles (32x32).

The game ships zero image assets: every tile is painted here with
deterministic per-tile RNG.  Tiles live in a GPU texture *array* (one layer
per tile) — no UV bleeding, per-layer mipmaps.

Painting toolkit: multi-octave value noise for organic mottling, voronoi
cells for masonry, edge bevels for subtle depth, and 16px pixel-art masks
(upscaled 2x) for items and UI icons.
"""

from __future__ import annotations

import zlib
from typing import Callable

import numpy as np

from engine.core.log import get_logger

_log = get_logger("atlas")

TILE = 32

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


def _add(tile: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Add a (H, W) or (H, W, 3) signed field to the RGB channels."""
    if delta.ndim == 2:
        delta = delta[:, :, None]
    rgb = tile[:, :, :3].astype(np.int16) + delta.astype(np.int16)
    tile[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return tile


def _smooth(field: np.ndarray, passes: int = 1) -> np.ndarray:
    for _ in range(passes):
        field = sum(
            np.roll(np.roll(field, dx, 0), dz, 1)
            for dx in (-1, 0, 1)
            for dz in (-1, 0, 1)
        ) / 9.0
    return field


def _noise(rng: np.random.Generator, octaves: tuple[tuple[int, float], ...]) -> np.ndarray:
    """Multi-octave value noise in [-1, 1]; octave = (cell_size, weight)."""
    out = np.zeros((TILE, TILE), dtype=np.float32)
    total = 0.0
    for cell, weight in octaves:
        n = TILE // cell
        base = rng.random((n, n)).astype(np.float32)
        up = np.kron(base, np.ones((cell, cell), dtype=np.float32))
        out += _smooth(up, passes=1) * weight
        total += weight
    out /= total
    out -= out.mean()
    peak = float(np.abs(out).max()) or 1.0
    return out / peak


def _mottle(tile: np.ndarray, rng: np.random.Generator, amount: int,
            octaves: tuple[tuple[int, float], ...] = ((8, 1.0), (4, 0.6), (2, 0.35))) -> np.ndarray:
    return _add(tile, _noise(rng, octaves) * amount)


def _grain(tile: np.ndarray, rng: np.random.Generator, amount: int) -> np.ndarray:
    return _add(tile, rng.integers(-amount, amount + 1, (TILE, TILE)).astype(np.float32))


def _bevel(tile: np.ndarray, strength: int = 14) -> np.ndarray:
    """Light top/left, dark bottom/right — reads as subtle depth."""
    edge = np.zeros((TILE, TILE), dtype=np.float32)
    edge[0, :] += strength
    edge[:, 0] += strength * 0.6
    edge[-1, :] -= strength
    edge[:, -1] -= strength * 0.6
    edge[1, :] += strength * 0.4
    edge[-2, :] -= strength * 0.4
    return _add(tile, edge)


def _voronoi(rng: np.random.Generator, points: int) -> tuple[np.ndarray, np.ndarray]:
    """Wrapped voronoi: (cell id map, mortar mask) — tileable masonry."""
    pts = rng.integers(0, TILE, size=(points, 2))
    yy, xx = np.mgrid[0:TILE, 0:TILE]
    dx = np.abs(xx[None] - pts[:, 0, None, None])
    dy = np.abs(yy[None] - pts[:, 1, None, None])
    dx = np.minimum(dx, TILE - dx)
    dy = np.minimum(dy, TILE - dy)
    dist = dx * dx + dy * dy
    order = np.argsort(dist, axis=0)
    d0 = np.take_along_axis(dist, order[0:1], axis=0)[0]
    d1 = np.take_along_axis(dist, order[1:2], axis=0)[0]
    return order[0], (d1 - d0) <= 3


def _mask_paint(rows: list[str], palette: dict[str, tuple[int, int, int, int]]) -> np.ndarray:
    """16px pixel-art painter, upscaled to TILE with crisp 2x pixels."""
    art = np.zeros((16, 16, 4), dtype=np.uint8)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch in palette:
                art[y, x] = palette[ch]
    return np.kron(art, np.ones((2, 2, 1), dtype=np.uint8))


def _stone_base(rng: np.random.Generator) -> np.ndarray:
    tile = _mottle(_flat((127, 127, 129)), rng, 16)
    tile = _grain(tile, rng, 5)
    # Hairline cracks: short darkened random walks.
    for _ in range(5):
        x, y = rng.integers(2, TILE - 8, size=2)
        for _ in range(int(rng.integers(5, 11))):
            x += int(rng.integers(0, 2))
            y += int(rng.integers(-1, 2))
            if 0 <= x < TILE and 0 <= y < TILE:
                tile[y, x, :3] = (tile[y, x, :3] * 0.72).astype(np.uint8)
    return tile


# -- terrain -------------------------------------------------------------------
@painter("stone")
def _p_stone(rng):
    return _bevel(_stone_base(rng), 8)


@painter("dirt")
def _p_dirt(rng):
    tile = _mottle(_flat((133, 94, 64)), rng, 20, ((8, 1.0), (4, 0.7), (2, 0.4)))
    tile = _grain(tile, rng, 7)
    for _ in range(9):  # embedded pebbles & roots
        x, y = rng.integers(1, TILE - 1, size=2)
        c = (152, 145, 136) if rng.random() < 0.6 else (96, 66, 44)
        tile[y, x, :3] = c
        if rng.random() < 0.5:
            tile[y, min(x + 1, TILE - 1), :3] = c
    return tile


@painter("grass_top")
def _p_grass_top(rng):
    tile = _mottle(_flat((94, 155, 52)), rng, 22, ((16, 0.8), (8, 1.0), (4, 0.6)))
    tile = _grain(tile, rng, 8)
    # Sparse bright blade tips + a few dark clumps.
    blades = rng.random((TILE, TILE)) < 0.07
    tile[blades, :3] = np.clip(
        tile[blades, :3].astype(np.int16) + np.array([14, 30, 6]), 0, 255
    ).astype(np.uint8)
    return tile


@painter("grass_side")
def _p_grass_side(rng):
    tile = _p_dirt(rng)
    grass = _p_grass_top(rng)
    tile[0:6] = grass[0:6]
    # Ragged overhang: grass dips 1-4 px into the dirt.
    depth = 6 + (rng.random(TILE) * 4).astype(int)
    for x in range(TILE):
        tile[6 : depth[x], x] = grass[6 : depth[x], x]
    return tile


@painter("sand")
def _p_sand(rng):
    tile = _mottle(_flat((221, 208, 164)), rng, 12, ((8, 1.0), (4, 0.5)))
    yy, xx = np.mgrid[0:TILE, 0:TILE]
    ripple = np.sin((yy + xx * 0.22) * 0.65) * 6.0
    tile = _add(tile, ripple)
    return _grain(tile, rng, 4)


@painter("sandstone")
def _p_sandstone(rng):
    tile = _mottle(_flat((214, 201, 158)), rng, 10)
    for row in range(0, TILE, 8):  # strata
        tile[row : row + 1, :, :3] = (tile[row : row + 1, :, :3] * 0.86).astype(np.uint8)
        tile[row + 4 : row + 5, :, :3] = (tile[row + 4 : row + 5, :, :3] * 0.93).astype(np.uint8)
    return _bevel(tile, 8)


@painter("gravel")
def _p_gravel(rng):
    cell, mortar = _voronoi(rng, 26)
    tones = rng.integers(96, 158, size=cell.max() + 1)
    tile = _flat((120, 118, 116))
    tile[:, :, 0] = tile[:, :, 1] = tile[:, :, 2] = tones[cell]
    tile[:, :, 1] = np.clip(tile[:, :, 1].astype(np.int16) - 3, 0, 255).astype(np.uint8)
    tile = _grain(tile, rng, 8)
    tile[mortar, :3] = (tile[mortar, :3] * 0.72).astype(np.uint8)
    return tile


@painter("bedrock")
def _p_bedrock(rng):
    return _grain(_mottle(_flat((60, 60, 63)), rng, 34, ((8, 1.0), (2, 0.8))), rng, 16)


@painter("snow")
def _p_snow(rng):
    tile = _mottle(_flat((242, 247, 252)), rng, 7)
    sparkle = rng.random((TILE, TILE)) < 0.03
    tile[sparkle, :3] = 255
    return tile


@painter("snow_side")
def _p_snow_side(rng):
    tile = _p_dirt(rng)
    tile[0:8] = _p_snow(rng)[0:8]
    return tile


# -- liquids -------------------------------------------------------------------
@painter("water")
def _p_water(rng):
    tile = _mottle(_flat((47, 106, 217)), rng, 13, ((16, 1.0), (8, 0.7)))
    glints = _noise(rng, ((8, 1.0),)) > 0.55
    tile[glints, :3] = np.clip(
        tile[glints, :3].astype(np.int16) + np.array([22, 28, 26]), 0, 255
    ).astype(np.uint8)
    return tile


@painter("lava")
def _p_lava(rng):
    field = _noise(rng, ((16, 1.0), (8, 0.8), (4, 0.4)))
    tile = _flat((94, 30, 12))
    tile = _add(tile, np.clip(field, -1, 0) * 40)
    hot = field > 0.05
    tile[hot, :3] = (214, 92, 20)
    hotter = field > 0.35
    tile[hotter, :3] = (255, 176, 48)
    core = field > 0.6
    tile[core, :3] = (255, 236, 130)
    return tile


# -- wood & foliage ------------------------------------------------------------
@painter("log_side")
def _p_log_side(rng):
    tile = _flat((106, 84, 50))
    xx = np.arange(TILE, dtype=np.float32)[None, :]
    yy = np.arange(TILE, dtype=np.float32)[:, None]
    ridges = np.sin(xx * 1.15 + np.sin(yy * 0.35) * 1.8) * 12.0
    tile = _add(tile, ridges)
    tile = _mottle(tile, rng, 9, ((8, 1.0),))
    for _ in range(4):  # bark knots
        x, y = rng.integers(2, TILE - 2, size=2)
        tile[y - 1 : y + 2, x - 1 : x + 2, :3] = (74, 56, 32)
        tile[y, x, :3] = (128, 102, 62)
    return tile


@painter("birch_log_side")
def _p_birch_log(rng):
    tile = _mottle(_flat((224, 221, 211)), rng, 8)
    for _ in range(11):  # characteristic dark dashes
        x = int(rng.integers(0, TILE - 6))
        y = int(rng.integers(0, TILE))
        w = int(rng.integers(3, 8))
        tile[y : y + 2, x : x + w, :3] = (54, 50, 45)
    return tile


@painter("log_top")
def _p_log_top(rng):
    tile = _mottle(_flat((110, 87, 52)), rng, 7)
    yy, xx = np.mgrid[0:TILE, 0:TILE]
    dist = np.sqrt((xx - 15.5) ** 2 + (yy - 15.5) ** 2)
    wobble = _noise(rng, ((8, 1.0),)) * 1.6
    rings = ((dist + wobble).astype(np.int32) % 5) < 2
    tile[rings] = np.array([182, 147, 92, 255], dtype=np.uint8)
    tile[dist > 14.6] = np.array([88, 68, 40, 255], dtype=np.uint8)  # bark rim
    return tile


@painter("planks")
def _p_planks(rng):
    tile = _mottle(_flat((168, 134, 82)), rng, 10, ((16, 1.0), (4, 0.4)))
    yy = np.arange(TILE, dtype=np.float32)[:, None]
    xx = np.arange(TILE, dtype=np.float32)[None, :]
    grain = np.sin(xx * 0.8 + yy * 0.15) * 6.0  # horizontal wood grain
    tile = _add(tile, grain)
    for row in (7, 15, 23, 31):  # board gaps
        tile[row, :, :3] = (tile[row, :, :3] * 0.55).astype(np.uint8)
        if row - 1 >= 0:
            tile[row - 1, :, :3] = (tile[row - 1, :, :3] * 0.85).astype(np.uint8)
    for row0, col in ((0, 22), (8, 6), (16, 26), (24, 12)):  # joints + nails
        tile[row0 : row0 + 7, col, :3] = (tile[row0 : row0 + 7, col, :3] * 0.7).astype(np.uint8)
        tile[row0 + 2, col + 2 if col + 2 < TILE else col - 2, :3] = (92, 78, 54)
    return tile


def _leaf_tile(rng, base: tuple[int, int, int]) -> np.ndarray:
    field = _noise(rng, ((8, 1.0), (4, 0.9), (2, 0.5)))
    tile = _flat(base)
    tile = _add(tile, field * 34)
    tips = field > 0.42  # lit leaf clusters
    tile[tips, :3] = np.clip(
        tile[tips, :3].astype(np.int16) + np.array([16, 26, 10]), 0, 255
    ).astype(np.uint8)
    holes = field < -0.52
    tile[holes, 3] = 0
    speckle = rng.random((TILE, TILE)) < 0.05
    tile[speckle, 3] = 0
    return tile


@painter("leaves")
def _p_leaves(rng):
    return _leaf_tile(rng, (52, 118, 32))


@painter("birch_leaves")
def _p_birch_leaves(rng):
    return _leaf_tile(rng, (92, 146, 58))


# -- masonry & ores -------------------------------------------------------------
@painter("cobble")
def _p_cobble(rng):
    cell, mortar = _voronoi(rng, 12)
    tones = rng.integers(100, 150, size=cell.max() + 1)
    tile = _flat((110, 110, 110))
    tile[:, :, 0] = tile[:, :, 1] = tile[:, :, 2] = tones[cell]
    tile = _mottle(tile, rng, 8, ((4, 1.0),))
    tile[mortar, :3] = (tile[mortar, :3] * 0.55).astype(np.uint8)
    return _bevel(tile, 6)


@painter("mossy_cobble")
def _p_mossy(rng):
    tile = _p_cobble(rng)
    moss = _noise(rng, ((8, 1.0), (4, 0.6))) > 0.18
    green = np.array([72, 118, 48], dtype=np.int16)
    tile[moss, :3] = ((tile[moss, :3].astype(np.int16) + green * 2) // 3).astype(np.uint8)
    return tile


@painter("stone_bricks")
def _p_stone_bricks(rng):
    tile = _mottle(_flat((122, 122, 124)), rng, 10)
    mortar_shade = 0.55
    for row in (0, 8, 16, 24):
        tile[row, :, :3] = (tile[row, :, :3] * mortar_shade).astype(np.uint8)
    for row0, offset in ((1, 0), (9, 16), (17, 0), (25, 16)):
        for col in (offset % TILE, (offset + 16) % TILE):
            tile[row0 : row0 + 7, col, :3] = (tile[row0 : row0 + 7, col, :3] * mortar_shade).astype(np.uint8)
    return _bevel(tile, 7)


@painter("bricks")
def _p_bricks(rng):
    tile = _mottle(_flat((152, 74, 58)), rng, 14)
    mortar = np.array([190, 182, 174], dtype=np.uint8)
    for row in (0, 8, 16, 24):
        tile[row : row + 2, :, :3] = mortar
    for row0, offset in ((2, 6), (10, 22), (18, 6), (26, 22)):
        for col in (offset % TILE, (offset + 16) % TILE):
            tile[row0 : row0 + 6, col : col + 2, :3] = mortar
    return tile


@painter("crafting_table_top")
def _p_craft_top(rng):
    tile = _p_planks(rng)
    edge = np.array([96, 74, 46], dtype=np.uint8)
    tile[0:3, :, :3] = edge
    tile[-3:, :, :3] = edge
    tile[:, 0:3, :3] = edge
    tile[:, -3:, :3] = edge
    tile[14:18, 3:-3, :3] = (118, 92, 58)  # tool groove
    tile[3:-3, 14:18, :3] = (118, 92, 58)
    return tile


@painter("crafting_table_side")
def _p_craft_side(rng):
    tile = _p_planks(rng)
    tile[0:4, :, :3] = (96, 74, 46)
    tile[8:20, 4:14, :3] = (188, 62, 48)    # painted tool shapes
    tile[8:20, 18:28, :3] = (120, 120, 128)
    return tile


@painter("bookshelf")
def _p_bookshelf(rng):
    tile = _p_planks(rng)
    for shelf_y in (2, 12, 22):
        x = 2
        while x < TILE - 3:
            w = int(rng.integers(2, 5))
            color = [(178, 60, 48), (60, 96, 168), (74, 140, 70), (196, 168, 72),
                     (140, 76, 150)][int(rng.integers(0, 5))]
            tile[shelf_y : shelf_y + 8, x : x + w, :3] = color
            tile[shelf_y : shelf_y + 8, x, :3] = tuple(int(c * 0.75) for c in color)
            x += w + 1
    return tile


@painter("glass")
def _p_glass(rng):
    tile = _flat((0, 0, 0), alpha=0)
    border = np.array([208, 229, 234, 255], dtype=np.uint8)
    tile[0:2, :] = border
    tile[-2:, :] = border
    tile[:, 0:2] = border
    tile[:, -2:] = border
    for i in range(4, 15):  # diagonal streaks
        tile[i, 18 - i] = (238, 246, 250, 200)
        tile[i + 1, 18 - i] = (238, 246, 250, 200)
        tile[i + 10, 30 - i] = (238, 246, 250, 130)
    return tile


@painter("glowstone")
def _p_glowstone(rng):
    field = _noise(rng, ((8, 1.0), (4, 0.7)))
    tile = _flat((172, 132, 82))
    tile = _add(tile, field * 24)
    bright = field > 0.12
    tile[bright, :3] = (250, 216, 122)
    core = field > 0.45
    tile[core, :3] = (255, 246, 190)
    return tile


def _ore(rng, color: tuple[int, int, int]) -> np.ndarray:
    tile = _stone_base(rng)
    base = np.array(color, dtype=np.int16)
    for _ in range(4):  # crystal clusters with shading
        cx, cy = rng.integers(4, TILE - 4, size=2)
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2)):
            if rng.random() < 0.8:
                x, y = int(cx + dx), int(cy + dy)
                tile[y, x, :3] = np.clip(base + rng.integers(-16, 17, 3), 0, 255)
        tile[cy, cx, :3] = np.clip(base + 46, 0, 255)          # glint
        tile[min(cy + 3, TILE - 1), min(cx + 3, TILE - 1), :3] = \
            np.clip(base * 5 // 10, 0, 255)                    # shadow px
    return tile


@painter("coal_ore")
def _p_coal(rng):
    return _ore(rng, (42, 42, 44))


@painter("iron_ore")
def _p_iron(rng):
    return _ore(rng, (221, 178, 146))


@painter("gold_ore")
def _p_gold(rng):
    return _ore(rng, (252, 230, 80))


@painter("diamond_ore")
def _p_diamond(rng):
    return _ore(rng, (104, 236, 242))


@painter("white")
def _p_white(rng):
    return _flat((255, 255, 255))


# -- pixel-art masks (plants, items, UI) --------------------------------------------
_TORCH_ROWS = [
    "................", "................", "................",
    ".......ww.......",
    ".......yy.......",
    "......oyyo......",
    ".......hh.......",
    ".......hh.......",
    ".......hh.......",
    ".......hh.......",
    ".......hh.......",
    ".......hh.......",
    ".......hh.......",
    ".......hh.......",
    "................", "................",
]


@painter("torch")
def _p_torch(rng):
    return _mask_paint(_TORCH_ROWS, {
        "h": (112, 88, 54, 255), "y": (255, 216, 96, 255),
        "w": (255, 246, 190, 255), "o": (255, 168, 58, 255),
    })


def _flower(petal: tuple[int, int, int], dark: tuple[int, int, int]) -> np.ndarray:
    rows = [
        "................", "................", "................",
        ".......pp.......",
        "......pddp......",
        "......pddp......",
        ".......pp.......",
        ".......ss.......",
        ".......ss.......",
        "......lss.......",
        ".......ss.......",
        ".......ss.......",
        ".......ss.......",
        "................", "................", "................",
    ]
    return _mask_paint(rows, {
        "p": (*petal, 255), "d": (*dark, 255),
        "s": (62, 128, 42, 255), "l": (82, 148, 56, 255),
    })


@painter("flower_red")
def _p_flower_red(rng):
    return _flower((216, 48, 38), (130, 22, 18))


@painter("flower_yellow")
def _p_flower_yellow(rng):
    return _flower((242, 224, 62), (168, 148, 30))


@painter("tall_grass")
def _p_tall_grass(rng):
    tile = _flat((0, 0, 0), alpha=0)
    for _ in range(14):
        x = int(rng.integers(2, TILE - 2))
        height = int(rng.integers(12, 27))
        shade = rng.integers(-26, 27)
        color = np.clip(np.array([88, 148, 52]) + shade, 0, 255)
        tile[TILE - height : TILE, x, :3] = color
        tile[TILE - height : TILE, x, 3] = 255
        bend = x + (1 if rng.random() < 0.5 else -1)
        if 0 <= bend < TILE and height > 16:
            tip = TILE - height
            tile[tip : tip + 4, bend, :3] = color
            tile[tip : tip + 4, bend, 3] = 255
    return tile


@painter("dead_bush")
def _p_dead_bush(rng):
    tile = _flat((0, 0, 0), alpha=0)
    color = np.array([124, 92, 50], dtype=np.int16)
    tile[20:32, 15:17, :3] = color
    tile[20:32, 15:17, 3] = 255
    for _ in range(9):
        x, y = 15, int(rng.integers(20, 28))
        dx = 1 if rng.random() < 0.5 else -1
        for _ in range(int(rng.integers(4, 9))):
            x += dx if rng.random() < 0.7 else 0
            y -= 1
            if 0 <= x < TILE and 0 <= y < TILE:
                tile[y, x, :3] = np.clip(color + rng.integers(-14, 15, 3), 0, 255).astype(np.uint8)
                tile[y, x, 3] = 255
    return tile


def _mushroom(cap: tuple[int, int, int], dots: bool) -> np.ndarray:
    rows = [
        "................", "................", "................", "................",
        "................", "................",
        "......cccc......",
        ".....cccccc.....",
        ".....cwccwc....." if dots else ".....cccccc.....",
        ".....cccccc.....",
        ".......ss.......",
        ".......ss.......",
        ".......ss.......",
        "......ssss......",
        "................", "................",
    ]
    return _mask_paint(rows, {
        "c": (*cap, 255), "w": (240, 236, 228, 255), "s": (226, 218, 198, 255),
    })


@painter("mushroom_red")
def _p_mushroom_red(rng):
    return _mushroom((198, 46, 38), dots=True)


@painter("mushroom_brown")
def _p_mushroom_brown(rng):
    return _mushroom((148, 106, 76), dots=False)


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
    "................", "................", "................", "................", "................",
]


def _heart(red: tuple[int, int, int, int]) -> np.ndarray:
    return _mask_paint(_HEART_ROWS, {
        "X": (46, 10, 10, 255), "R": red, "W": (255, 200, 200, 255),
    })


@painter("heart_full")
def _p_heart_full(rng):
    return _heart((214, 36, 36, 255))


@painter("heart_half")
def _p_heart_half(rng):
    tile = _heart((214, 36, 36, 255))
    tile[:, 12:] = _heart((60, 26, 26, 255))[:, 12:]
    return tile


@painter("heart_empty")
def _p_heart_empty(rng):
    return _heart((60, 26, 26, 255))


@painter("bubble")
def _p_bubble(rng):
    tile = _flat((0, 0, 0), alpha=0)
    yy, xx = np.mgrid[0:TILE, 0:TILE]
    dist = np.sqrt((xx - 15.5) ** 2 + (yy - 13.5) ** 2)
    ring = (dist >= 7.0) & (dist <= 10.4)
    tile[ring] = (150, 200, 250, 235)
    tile[9:12, 10:13] = (230, 245, 255, 255)  # highlight
    return tile


# -- build -----------------------------------------------------------------------
def _error_tile() -> np.ndarray:
    tile = _flat((255, 0, 255))
    tile[0:16, 16:32, :3] = (0, 0, 0)
    tile[16:32, 0:16, :3] = (0, 0, 0)
    return tile


def build_tiles(tile_names: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    """Paint all requested tiles. Returns (layers array (L,T,T,4), name->layer)."""
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
    _log.info("Painted %d procedural tiles (%dpx)", len(tile_names), TILE)
    return layers, mapping
