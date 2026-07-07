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

# Painters author tiles at TILE resolution; build_tiles upsamples to
# ATLAS_SIZE and layers extra fine-grain detail on top (plan phase 3,
# adjusted from 128 to 64 for startup time).
TILE = 32
ATLAS_SIZE = 64

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


@painter("copper_ore")
def _p_copper(rng):
    return _ore(rng, (216, 125, 78))


@painter("lapis_ore")
def _p_lapis(rng):
    return _ore(rng, (42, 82, 190))


@painter("redstone_ore")
def _p_redstone(rng):
    return _ore(rng, (222, 38, 28))


@painter("emerald_ore")
def _p_emerald(rng):
    return _ore(rng, (58, 202, 92))


@painter("smooth_stone")
def _p_smooth_stone(rng):
    tile = _mottle(_flat((141, 141, 144)), rng, 8)
    return _bevel(_grain(tile, rng, 3), 6)


@painter("cracked_stone_bricks")
def _p_cracked_bricks(rng):
    tile = _p_stone_bricks(rng)
    for _ in range(4):  # long cracks across the masonry
        x, y = rng.integers(2, TILE - 10, size=2)
        for _ in range(int(rng.integers(8, 16))):
            x += int(rng.integers(0, 2))
            y += int(rng.integers(-1, 2))
            if 0 <= x < TILE and 0 <= y < TILE:
                tile[y, x, :3] = (tile[y, x, :3] * 0.6).astype(np.uint8)
    return tile


@painter("ice")
def _p_ice(rng):
    tile = _mottle(_flat((168, 205, 240)), rng, 12, ((16, 1.0), (8, 0.6)))
    for _ in range(4):  # glassy internal cracks
        x, y = rng.integers(2, TILE - 8, size=2)
        for _ in range(int(rng.integers(5, 10))):
            x += int(rng.integers(0, 2))
            y += int(rng.integers(-1, 2))
            if 0 <= x < TILE and 0 <= y < TILE:
                tile[y, x, :3] = (222, 240, 252)
    return _bevel(tile, 8)


@painter("packed_ice")
def _p_packed_ice(rng):
    return _bevel(_mottle(_flat((126, 172, 224)), rng, 14), 8)


@painter("clay")
def _p_clay(rng):
    return _grain(_mottle(_flat((158, 160, 170)), rng, 10), rng, 4)


@painter("obsidian")
def _p_obsidian(rng):
    tile = _mottle(_flat((22, 18, 34)), rng, 12, ((8, 1.0), (4, 0.6)))
    glint = _noise(rng, ((8, 1.0),)) > 0.55
    tile[glint, :3] = np.clip(tile[glint, :3].astype(np.int16) +
                              np.array([44, 30, 60]), 0, 255).astype(np.uint8)
    return _bevel(tile, 6)


@painter("netherrack")
def _p_netherrack(rng):
    tile = _mottle(_flat((94, 34, 34)), rng, 20, ((8, 1.0), (4, 0.7), (2, 0.4)))
    veins = _noise(rng, ((8, 1.0),)) > 0.4
    tile[veins, :3] = np.clip(tile[veins, :3].astype(np.int16) +
                              np.array([34, 8, 8]), 0, 255).astype(np.uint8)
    return tile


@painter("soul_sand")
def _p_soul_sand(rng):
    tile = _mottle(_flat((84, 66, 54)), rng, 14)
    for _ in range(3):  # hollow "faces" pressed into the sand
        cx, cy = rng.integers(6, TILE - 6, size=2)
        yy, xx = np.mgrid[0:TILE, 0:TILE]
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        tile[(d > 2) & (d < 4), :3] = (48, 36, 30)
    return tile


@painter("nether_bricks")
def _p_nether_bricks(rng):
    tile = _mottle(_flat((52, 26, 30)), rng, 10)
    mortar = (30, 14, 16)
    for row in (0, 8, 16, 24):
        tile[row : row + 1, :, :3] = mortar
    for row0, offset in ((1, 0), (9, 16), (17, 0), (25, 16)):
        for col in (offset % TILE, (offset + 16) % TILE):
            tile[row0 : row0 + 7, col, :3] = mortar
    return tile


@painter("chest_top")
def _p_chest_top(rng):
    tile = _p_planks(rng)
    tile[0:3, :, :3] = (74, 52, 30)
    tile[-3:, :, :3] = (74, 52, 30)
    tile[:, 0:3, :3] = (74, 52, 30)
    tile[:, -3:, :3] = (74, 52, 30)
    tile[14:18, 14:18, :3] = (222, 200, 120)  # latch
    return tile


@painter("chest_side")
def _p_chest_side(rng):
    tile = _p_planks(rng)
    tile[15:17, :, :3] = (60, 42, 24)      # lid seam
    tile[13:20, 14:18, :3] = (222, 200, 120)  # lock plate
    tile[0:3, :, :3] = (74, 52, 30)
    tile[-3:, :, :3] = (74, 52, 30)
    return tile


@painter("cobweb")
def _p_cobweb(rng):
    tile = _flat((0, 0, 0), alpha=0)
    c = (228, 232, 238, 210)
    m = TILE // 2
    for a in range(0, TILE, 4):  # radial + diagonal strands
        tile[m, a] = c
        tile[a, m] = c
        if a < TILE:
            tile[a, a] = c
            tile[a, TILE - 1 - a] = c
    for r in (6, 11):  # concentric rings (diamond)
        for t in range(-r, r + 1):
            for (px, py) in ((m + t, m + (r - abs(t))), (m + t, m - (r - abs(t)))):
                if 0 <= px < TILE and 0 <= py < TILE:
                    tile[py, px] = c
    return tile


@painter("end_stone")
def _p_end_stone(rng):
    return _bevel(_mottle(_flat((220, 224, 168)), rng, 12), 6)


@painter("spawner")
def _p_spawner(rng):
    tile = _flat((0, 0, 0), alpha=0)
    bar = (58, 66, 74, 255)
    for a in range(0, TILE, 6):  # cage bars
        tile[:, a] = bar
        tile[a, :] = bar
    tile[12:20, 12:20, :3] = (90, 40, 40)  # glimpse of the mob inside
    tile[12:20, 12:20, 3] = 255
    return tile


@painter("nether_portal")
def _p_nether_portal(rng):
    tile = _flat((120, 40, 190))
    swirl = _noise(rng, ((16, 1.0), (8, 0.7)))
    tile = _add(tile, swirl * 60)
    bright = swirl > 0.3
    tile[bright, :3] = (200, 130, 240)
    return tile


@painter("glowshroom")
def _p_glowshroom(rng):
    tile = _flat((0, 0, 0), alpha=0)
    for _ in range(10):
        x, y = rng.integers(2, TILE - 2, size=2)
        tile[y, x, :3] = (120, 220, 180)
        tile[y, x, 3] = 255
        if rng.random() < 0.5 and x + 1 < TILE:
            tile[y, x + 1, :3] = (150, 240, 200)
            tile[y, x + 1, 3] = 255
    return tile


def _recolor_log(rng, bark: tuple[int, int, int], knot: tuple[int, int, int]) -> np.ndarray:
    tile = _flat(bark)
    xx = np.arange(TILE, dtype=np.float32)[None, :]
    yy = np.arange(TILE, dtype=np.float32)[:, None]
    ridges = np.sin(xx * 1.15 + np.sin(yy * 0.35) * 1.8) * 12.0
    tile = _add(tile, ridges)
    tile = _mottle(tile, rng, 9, ((8, 1.0),))
    for _ in range(4):
        x, y = rng.integers(2, TILE - 2, size=2)
        tile[y - 1 : y + 2, x - 1 : x + 2, :3] = knot
    return tile


@painter("spruce_log_side")
def _p_spruce_log(rng):
    return _recolor_log(rng, (76, 56, 34), (52, 38, 24))


@painter("jungle_log_side")
def _p_jungle_log(rng):
    tile = _recolor_log(rng, (98, 76, 46), (70, 54, 32))
    moss = _noise(rng, ((8, 1.0),)) > 0.45
    tile[moss, :3] = ((tile[moss, :3].astype(np.int16) +
                       np.array([60, 118, 48]) * 2) // 3).astype(np.uint8)
    return tile


@painter("acacia_log_side")
def _p_acacia_log(rng):
    return _recolor_log(rng, (112, 102, 94), (150, 88, 54))


@painter("spruce_leaves")
def _p_spruce_leaves(rng):
    return _leaf_tile(rng, (38, 84, 40))


@painter("jungle_leaves")
def _p_jungle_leaves(rng):
    return _leaf_tile(rng, (46, 132, 28))


@painter("acacia_leaves")
def _p_acacia_leaves(rng):
    return _leaf_tile(rng, (104, 142, 44))


@painter("fern")
def _p_fern(rng):
    tile = _flat((0, 0, 0), alpha=0)
    for _ in range(8):  # arching fronds
        x = int(rng.integers(4, TILE - 4))
        height = int(rng.integers(10, 20))
        bend = 1 if rng.random() < 0.5 else -1
        color = np.clip(np.array([58, 112, 44]) + rng.integers(-18, 19), 0, 255)
        for step in range(height):
            px = x + bend * (step * step // 90)
            py = TILE - 1 - step
            if 0 <= px < TILE and 0 <= py < TILE:
                tile[py, px, :3] = color
                tile[py, px, 3] = 255
                if step % 3 == 0 and 0 <= px + bend < TILE:
                    tile[py, px + bend, :3] = color
                    tile[py, px + bend, 3] = 255
    return tile


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


_DRUMSTICK_ROWS = [
    "................",
    "..........XX....",
    ".........XMMX...",
    "........XMMMX...",
    ".......XMMMX....",
    "......XMMMX.....",
    ".....XMMMX......",
    "....XMMMX.......",
    "...XMMMXX.......",
    "..XMMMXBB.......",
    "..XMMXBBB.......",
    "..XMMXBB........",
    "...XXBB.........",
    "................", "................", "................",
]


def _drumstick(meat: tuple[int, int, int, int]) -> np.ndarray:
    return _mask_paint(_DRUMSTICK_ROWS, {
        "X": (60, 34, 18, 255), "M": meat, "B": (232, 224, 208, 255),
    })


@painter("food_full")
def _p_food_full(rng):
    return _drumstick((150, 90, 52, 255))


@painter("food_half")
def _p_food_half(rng):
    tile = _drumstick((150, 90, 52, 255))
    tile[:, 16:] = _drumstick((70, 50, 40, 255))[:, 16:]
    return tile


@painter("food_empty")
def _p_food_empty(rng):
    return _drumstick((70, 50, 40, 255))


@painter("bubble")
def _p_bubble(rng):
    tile = _flat((0, 0, 0), alpha=0)
    yy, xx = np.mgrid[0:TILE, 0:TILE]
    dist = np.sqrt((xx - 15.5) ** 2 + (yy - 13.5) ** 2)
    ring = (dist >= 7.0) & (dist <= 10.4)
    tile[ring] = (150, 200, 250, 235)
    tile[9:12, 10:13] = (230, 245, 255, 255)  # highlight
    return tile


# -- item icons (tools, food) ---------------------------------------------------
_TIER_COLOR = {
    "wood": (140, 100, 58), "stone": (120, 120, 124),
    "iron": (216, 216, 220), "diamond": (110, 232, 236),
}
_HANDLE = (112, 82, 48)


def _register_tool(ttype: str, tier: str) -> None:
    head = _TIER_COLOR[tier]

    def paint(rng, ttype=ttype, head=head):
        tile = _flat((0, 0, 0), alpha=0)
        # Diagonal wooden handle from bottom-left to centre.
        for i in range(4, TILE - 6):
            x, y = i, TILE - 2 - i
            if 0 <= x < TILE and 0 <= y < TILE:
                tile[y, x, :3] = _HANDLE
                tile[y, x, 3] = 255
                tile[y, x + 1, :3] = _HANDLE
                tile[y, x + 1, 3] = 255
        hx, hy = TILE - 10, 6  # head region near top-right
        if ttype == "pickaxe":
            for dx in range(-6, 7):
                yy = hy + abs(dx) // 3
                _put(tile, hx + dx, yy, head)
        elif ttype == "sword":
            for dy in range(-1, 9):  # long blade back down the handle line
                _put(tile, hx + dy, hy + dy, head)
            _put(tile, hx - 2, hy + 2, head)
            _put(tile, hx + 2, hy - 2, head)
        elif ttype == "axe":
            for dx in range(0, 5):
                for dy in range(-3, 4):
                    if abs(dy) <= 3 - dx // 2:
                        _put(tile, hx + dx, hy + dy, head)
        else:  # shovel
            for dx in range(-2, 3):
                for dy in range(-1, 4):
                    _put(tile, hx + dx, hy + dy, head)
        return tile

    _PAINTERS[f"{ttype}_{tier}"] = paint


def _put(tile, x, y, rgb):
    if 0 <= x < TILE and 0 <= y < TILE:
        tile[y, x, :3] = rgb
        tile[y, x, 3] = 255


for _tt in ("pickaxe", "axe", "shovel", "sword"):
    for _tr in ("wood", "stone", "iron", "diamond"):
        _register_tool(_tt, _tr)


@painter("stick")
def _p_stick(rng):
    tile = _flat((0, 0, 0), alpha=0)
    for i in range(6, TILE - 6):
        _put(tile, i, TILE - 2 - i, (128, 94, 54))
        _put(tile, i + 1, TILE - 2 - i, (150, 112, 66))
    return tile


@painter("apple")
def _p_apple(rng):
    tile = _flat((0, 0, 0), alpha=0)
    yy, xx = np.mgrid[0:TILE, 0:TILE]
    d = np.sqrt((xx - TILE / 2) ** 2 + (yy - TILE / 2 - 1) ** 2)
    body = d < TILE * 0.32
    tile[body] = (206, 44, 40, 255)
    tile[body & (xx < TILE / 2)] = (226, 70, 60, 255)  # highlight
    tile[TILE // 2 - 8 : TILE // 2 - 4, TILE // 2, :3] = (90, 62, 34)  # stem
    tile[TILE // 2 - 8 : TILE // 2 - 4, TILE // 2, 3] = 255
    return tile


def _meat(rng, color, cooked):
    tile = _flat((0, 0, 0), alpha=0)
    yy, xx = np.mgrid[0:TILE, 0:TILE]
    d = np.sqrt((xx - TILE / 2) ** 2 + (yy - TILE / 2) ** 2)
    body = d < TILE * 0.34
    tile[body] = (*color, 255)
    bone = (yy > TILE * 0.6) & (np.abs(xx - TILE / 2) < 2)
    tile[bone] = (236, 224, 208, 255)
    if cooked:
        tile[body & (d < TILE * 0.2)] = (120, 74, 44, 255)  # seared centre
    return tile


@painter("raw_meat")
def _p_raw_meat(rng):
    return _meat(rng, (208, 96, 104), cooked=False)


@painter("cooked_meat")
def _p_cooked_meat(rng):
    return _meat(rng, (150, 96, 58), cooked=True)


# -- build -----------------------------------------------------------------------
def _error_tile() -> np.ndarray:
    tile = _flat((255, 0, 255))
    tile[0:16, 16:32, :3] = (0, 0, 0)
    tile[16:32, 0:16, :3] = (0, 0, 0)
    return tile


# Per-tile material parameters for the PBR maps (metallic, roughness,
# normal strength). Anything not listed uses the defaults.
_MATERIALS: dict[str, tuple[float, float, float]] = {
    "gold_ore": (0.55, 0.45, 1.4), "iron_ore": (0.4, 0.55, 1.4),
    "diamond_ore": (0.35, 0.3, 1.4), "coal_ore": (0.05, 0.8, 1.4),
    "copper_ore": (0.5, 0.5, 1.4), "lapis_ore": (0.2, 0.5, 1.4),
    "redstone_ore": (0.1, 0.55, 1.4), "emerald_ore": (0.35, 0.35, 1.4),
    "ice": (0.0, 0.15, 0.5), "packed_ice": (0.0, 0.25, 0.6),
    "smooth_stone": (0.0, 0.6, 0.9), "glass": (0.0, 0.12, 0.2), "water": (0.0, 0.15, 0.4),
    "stone": (0.0, 0.72, 1.5), "cobble": (0.0, 0.85, 1.7),
    "mossy_cobble": (0.0, 0.9, 1.7), "stone_bricks": (0.0, 0.75, 1.7),
    "bricks": (0.0, 0.8, 1.7), "snow": (0.0, 0.35, 0.6),
    "leaves": (0.0, 0.95, 0.8), "birch_leaves": (0.0, 0.95, 0.8),
    "glowstone": (0.1, 0.4, 1.2),
}
_DEFAULT_MATERIAL = (0.0, 0.8, 1.0)


def _upscale_detail(tile: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """2x upscale + fine grain so 64px tiles gain real detail, not blur."""
    big = np.kron(tile, np.ones((2, 2, 1), dtype=np.uint8))
    opaque = big[:, :, 3] > 0
    grain = rng.integers(-5, 6, (ATLAS_SIZE, ATLAS_SIZE), dtype=np.int16)
    cells = np.kron(
        rng.integers(-6, 7, (ATLAS_SIZE // 4, ATLAS_SIZE // 4), dtype=np.int16),
        np.ones((4, 4), dtype=np.int16),
    )
    detail = (grain + cells)[:, :, None]
    rgb = big[:, :, :3].astype(np.int16) + np.where(opaque[:, :, None], detail, 0)
    big[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return big


def build_tiles(
    tile_names: list[str], pack_dir=None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Paint all tiles + derive PBR maps.

    Returns (albedo (L,S,S,4), normal (L,S,S,3), mrao (L,S,S,3), name->layer).
    ``pack_dir``: optional external texture pack — PNGs named after tiles
    override the procedural painters (missing ones fall back).
    """
    n = len(tile_names)
    albedo = np.zeros((n, ATLAS_SIZE, ATLAS_SIZE, 4), dtype=np.uint8)
    normal = np.zeros((n, ATLAS_SIZE, ATLAS_SIZE, 3), dtype=np.uint8)
    mrao = np.zeros((n, ATLAS_SIZE, ATLAS_SIZE, 3), dtype=np.uint8)
    mapping: dict[str, int] = {}

    for i, name in enumerate(tile_names):
        rng = np.random.default_rng(zlib.crc32(name.encode("utf-8")))
        packed = _load_pack_tile(pack_dir, name) if pack_dir else None
        if packed is not None:
            albedo[i] = packed
        else:
            fn = _PAINTERS.get(name)
            if fn is None:
                _log.warning("No painter for tile '%s' — using error tile", name)
                albedo[i] = np.kron(_error_tile(), np.ones((2, 2, 1), dtype=np.uint8))
            else:
                albedo[i] = _upscale_detail(fn(rng), rng)
        mapping[name] = i

        met, rough, strength = _MATERIALS.get(name, _DEFAULT_MATERIAL)
        normal[i] = _normal_from_albedo(albedo[i], strength)
        mrao[i] = _mrao_map(albedo[i], met, rough)

    _log.info("Painted %d procedural tiles (%dpx + normal/MRAO maps)", n, ATLAS_SIZE)
    return albedo, normal, mrao, mapping


def _normal_from_albedo(tile: np.ndarray, strength: float) -> np.ndarray:
    """Sobel height-from-luminance normal map (plan 3.2)."""
    lum = tile[:, :, :3].astype(np.float32).mean(axis=2) / 255.0
    dx = (np.roll(lum, -1, axis=1) - np.roll(lum, 1, axis=1)) * strength
    dy = (np.roll(lum, -1, axis=0) - np.roll(lum, 1, axis=0)) * strength
    nz = np.ones_like(lum)
    length = np.sqrt(dx * dx + dy * dy + 1.0)
    out = np.empty((*lum.shape, 3), dtype=np.uint8)
    out[:, :, 0] = ((-dx / length) * 0.5 + 0.5) * 255
    out[:, :, 1] = ((-dy / length) * 0.5 + 0.5) * 255
    out[:, :, 2] = ((nz / length) * 0.5 + 0.5) * 255
    return out


def _mrao_map(tile: np.ndarray, metallic: float, roughness: float) -> np.ndarray:
    """R=metallic, G=roughness, B=micro-AO baked from local darkness."""
    lum = tile[:, :, :3].astype(np.float32).mean(axis=2) / 255.0
    mean = lum.mean() or 1.0
    micro_ao = np.clip(1.0 - np.clip((mean - lum) / max(mean, 1e-3), 0.0, 1.0) * 0.45, 0.0, 1.0)
    out = np.empty((*lum.shape, 3), dtype=np.uint8)
    out[:, :, 0] = int(metallic * 255)
    out[:, :, 1] = int(roughness * 255)
    out[:, :, 2] = (micro_ao * 255).astype(np.uint8)
    return out


def _load_pack_tile(pack_dir, name: str) -> np.ndarray | None:
    """External texture pack override: <pack>/<tile>.png -> ATLAS_SIZE RGBA."""
    from pathlib import Path

    from PIL import Image

    path = Path(pack_dir) / f"{name}.png"
    if not path.exists():
        return None
    try:
        img = Image.open(path).convert("RGBA").resize(
            (ATLAS_SIZE, ATLAS_SIZE), Image.NEAREST
        )
        return np.asarray(img, dtype=np.uint8)
    except Exception as exc:  # noqa: BLE001 - bad pack must not kill startup
        _log.warning("Texture pack tile '%s' failed to load: %s", name, exc)
        return None
