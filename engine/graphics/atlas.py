"""Procedural texture tiles.

The game ships zero image assets: every 16x16 tile is painted here with
deterministic per-tile RNG, so the whole look is reproducible and mods can
register new painters (or later, real image files) without engine changes.

Tiles are stacked into a GPU texture *array* (one layer per tile) instead of
a classic atlas — no UV bleeding, mipmaps per layer for free.
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


def _blobs(
    tile: np.ndarray,
    rng: np.random.Generator,
    color: tuple[int, int, int],
    clusters: int,
) -> np.ndarray:
    """Small plus-shaped ore clusters with slight colour variation."""
    for _ in range(clusters):
        cx, cy = rng.integers(2, TILE - 2, size=2)
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1)):
            x, y = int(cx + dx) % TILE, int(cy + dy) % TILE
            jitter = rng.integers(-18, 19, size=3)
            tile[y, x, :3] = np.clip(np.array(color) + jitter, 0, 255)
    return tile


def _stone_base(rng: np.random.Generator) -> np.ndarray:
    return _speckle(_flat((127, 127, 127)), rng, 11)


# -- terrain -------------------------------------------------------------------
@painter("stone")
def _p_stone(rng: np.random.Generator) -> np.ndarray:
    return _stone_base(rng)


@painter("dirt")
def _p_dirt(rng: np.random.Generator) -> np.ndarray:
    return _speckle(_flat((134, 96, 67)), rng, 15)


@painter("grass_top")
def _p_grass_top(rng: np.random.Generator) -> np.ndarray:
    return _speckle(_flat((104, 157, 58)), rng, 13)


@painter("grass_side")
def _p_grass_side(rng: np.random.Generator) -> np.ndarray:
    tile = _speckle(_flat((134, 96, 67)), rng, 15)
    grass = _speckle(_flat((104, 157, 58)), rng, 13)
    tile[0:3] = grass[0:3]
    ragged = rng.random(TILE) < 0.5
    tile[3, ragged] = grass[3, ragged]
    return tile


@painter("sand")
def _p_sand(rng: np.random.Generator) -> np.ndarray:
    return _speckle(_flat((219, 207, 163)), rng, 9)


@painter("sandstone")
def _p_sandstone(rng: np.random.Generator) -> np.ndarray:
    tile = _speckle(_flat((212, 200, 157)), rng, 7)
    for row in (0, 5, 10, 15):
        tile[row, :, :3] = (tile[row, :, :3] * 0.88).astype(np.uint8)
    return tile


@painter("gravel")
def _p_gravel(rng: np.random.Generator) -> np.ndarray:
    tile = _speckle(_flat((131, 127, 126)), rng, 8)
    for _ in range(14):
        x, y = rng.integers(0, TILE - 1, size=2)
        shade = rng.integers(-38, 39)
        patch = np.clip(tile[y : y + 2, x : x + 2, :3].astype(np.int16) + shade, 0, 255)
        tile[y : y + 2, x : x + 2, :3] = patch.astype(np.uint8)
    return tile


@painter("bedrock")
def _p_bedrock(rng: np.random.Generator) -> np.ndarray:
    return _speckle(_flat((64, 64, 64)), rng, 32)


@painter("snow")
def _p_snow(rng: np.random.Generator) -> np.ndarray:
    return _speckle(_flat((240, 246, 250)), rng, 6)


@painter("snow_side")
def _p_snow_side(rng: np.random.Generator) -> np.ndarray:
    tile = _speckle(_flat((134, 96, 67)), rng, 15)
    tile[0:4] = _speckle(_flat((240, 246, 250)), rng, 6)[0:4]
    return tile


# -- liquids -------------------------------------------------------------------
@painter("water")
def _p_water(rng: np.random.Generator) -> np.ndarray:
    tile = _flat((52, 110, 220))
    rows = np.sin(np.linspace(0.0, 3.0 * np.pi, TILE)) * 5.0
    tile[:, :, :3] = np.clip(
        tile[:, :, :3].astype(np.int16) + rows[:, None, None].astype(np.int16), 0, 255
    ).astype(np.uint8)
    return _speckle(tile, rng, 6)


@painter("lava")
def _p_lava(rng: np.random.Generator) -> np.ndarray:
    tile = _speckle(_flat((205, 90, 18)), rng, 18)
    for _ in range(6):
        x, y = rng.integers(1, TILE - 1, size=2)
        tile[y - 1 : y + 1, x - 1 : x + 1, :3] = (255, 190, 64)
    return tile


# -- wood & plants ------------------------------------------------------------
@painter("log_side")
def _p_log_side(rng: np.random.Generator) -> np.ndarray:
    tile = _speckle(_flat((104, 82, 49)), rng, 8)
    for col in range(0, TILE, 4):
        dark = rng.integers(14, 26)
        tile[:, col, :3] = np.clip(tile[:, col, :3].astype(np.int16) - dark, 0, 255).astype(np.uint8)
    return tile


@painter("log_top")
def _p_log_top(rng: np.random.Generator) -> np.ndarray:
    tile = _speckle(_flat((104, 82, 49)), rng, 6)
    yy, xx = np.mgrid[0:TILE, 0:TILE]
    dist = np.sqrt((xx - 7.5) ** 2 + (yy - 7.5) ** 2)
    rings = (dist.astype(np.int32) % 3 == 0)
    tile[rings] = np.array([178, 143, 88, 255], dtype=np.uint8)
    return tile


@painter("planks")
def _p_planks(rng: np.random.Generator) -> np.ndarray:
    tile = _speckle(_flat((162, 130, 78)), rng, 9)
    for row in (3, 7, 11, 15):
        tile[row, :, :3] = (tile[row, :, :3] * 0.7).astype(np.uint8)
    for row, col in ((0, 11), (4, 3), (8, 13), (12, 6)):
        tile[row : row + 4, col, :3] = (tile[row : row + 4, col, :3] * 0.78).astype(np.uint8)
    return tile


@painter("leaves")
def _p_leaves(rng: np.random.Generator) -> np.ndarray:
    tile = _speckle(_flat((58, 122, 36)), rng, 22)
    holes = rng.random((TILE, TILE)) < 0.18
    tile[holes, 3] = 0
    return tile


@painter("tall_grass")
def _p_tall_grass(rng: np.random.Generator) -> np.ndarray:
    tile = _flat((0, 0, 0), alpha=0)
    for _ in range(9):
        x = int(rng.integers(1, TILE - 1))
        height = int(rng.integers(6, 14))
        shade = rng.integers(-20, 21)
        color = np.clip(np.array([92, 148, 56]) + shade, 0, 255)
        tile[TILE - height : TILE, x, :3] = color
        tile[TILE - height : TILE, x, 3] = 255
    return tile


def _flower(rng: np.random.Generator, petal: tuple[int, int, int]) -> np.ndarray:
    tile = _flat((0, 0, 0), alpha=0)
    tile[9:16, 8, :3] = (60, 125, 40)
    tile[9:16, 8, 3] = 255
    tile[4:8, 7:10, :3] = petal
    tile[4:8, 7:10, 3] = 255
    tile[5:7, 8, :3] = np.clip(np.array(petal, dtype=np.int16) - 70, 0, 255).astype(np.uint8)
    return tile


@painter("flower_red")
def _p_flower_red(rng: np.random.Generator) -> np.ndarray:
    return _flower(rng, (215, 50, 40))


@painter("flower_yellow")
def _p_flower_yellow(rng: np.random.Generator) -> np.ndarray:
    return _flower(rng, (238, 220, 60))


# -- building & ores -------------------------------------------------------------
@painter("cobble")
def _p_cobble(rng: np.random.Generator) -> np.ndarray:
    tile = _speckle(_flat((110, 110, 110)), rng, 8)
    # Rounded stones on a 4x4 grid with dark mortar between them.
    for gy in range(0, TILE, 4):
        for gx in range(0, TILE, 4):
            shade = int(rng.integers(-22, 30))
            stone = np.clip(122 + shade + rng.integers(-6, 7, (4, 4, 1)), 0, 255)
            tile[gy : gy + 4, gx : gx + 4, :3] = stone
    tile[::4, :, :3] = (tile[::4, :, :3] * 0.62).astype(np.uint8)
    tile[:, ::4, :3] = (tile[:, ::4, :3] * 0.62).astype(np.uint8)
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
    return tile


@painter("coal_ore")
def _p_coal(rng: np.random.Generator) -> np.ndarray:
    return _blobs(_stone_base(rng), rng, (46, 46, 46), 3)


@painter("iron_ore")
def _p_iron(rng: np.random.Generator) -> np.ndarray:
    return _blobs(_stone_base(rng), rng, (216, 175, 147), 3)


@painter("gold_ore")
def _p_gold(rng: np.random.Generator) -> np.ndarray:
    return _blobs(_stone_base(rng), rng, (250, 234, 80), 3)


@painter("diamond_ore")
def _p_diamond(rng: np.random.Generator) -> np.ndarray:
    return _blobs(_stone_base(rng), rng, (98, 233, 240), 3)


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
