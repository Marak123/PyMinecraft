"""Voxel lighting: flood-fill sky light and block light.

Two independent 0..15 light fields per chunk:

- **sky light** — full-strength beams travel straight down through air, then
  flood laterally with decay.  This is what produces real voxel shadows:
  dark caves, shade under trees and overhangs.
- **block light** — flood fill from emissive blocks (torches, glowstone,
  lava), attenuated per step.

Both are computed with iterative vectorised dilation (a NumPy-friendly BFS):
each pass takes the max of the six shifted neighbours minus the entry cost,
repeated until stable.  Light travels at most 15 blocks, so computing a
chunk on its 3x3-chunk window yields exact values for the centre chunk.

Edits use :func:`relight_box`: a from-scratch recompute of a small window
around the edit, seeded with the stored light on the window boundary.  This
handles both light addition *and* removal without the classic two-phase
un-lighting BFS.
"""

from __future__ import annotations

import numpy as np

from engine.world.blocks import AIR, BlockRegistry
from engine.world.coords import CHUNK_X, CHUNK_Y, CHUNK_Z

MAX_LIGHT = 15
# Light can travel at most MAX_LIGHT blocks; +1 boundary shell.
RELIGHT_RADIUS = MAX_LIGHT + 1

_I16 = np.int16


def _spread_once(light: np.ndarray, receives: np.ndarray, cost: np.ndarray) -> bool:
    """One dilation pass; returns True if any cell got brighter.

    ``cost`` is the per-cell entry cost (1 + extra attenuation of the
    destination cell).  Opaque cells never receive light but may hold it
    (emissive sources), which is why the mask applies to receiving only.
    """
    best = light.copy()
    for axis in range(3):
        for direction in (1, -1):
            shifted = np.full_like(light, 0)
            if direction == 1:
                shifted[tuple(slice(1, None) if a == axis else slice(None) for a in range(3))] = \
                    light[tuple(slice(None, -1) if a == axis else slice(None) for a in range(3))]
            else:
                shifted[tuple(slice(None, -1) if a == axis else slice(None) for a in range(3))] = \
                    light[tuple(slice(1, None) if a == axis else slice(None) for a in range(3))]
            np.maximum(best, shifted - cost, out=best)
    np.clip(best, 0, MAX_LIGHT, out=best)
    grew = bool((best > light)[receives].any())
    if grew:
        light[receives] = best[receives]
    return grew


def _flood(light: np.ndarray, receives: np.ndarray, cost: np.ndarray) -> np.ndarray:
    for _ in range(MAX_LIGHT + 1):
        if not _spread_once(light, receives, cost):
            break
    return light


def _sky_beams(blocks: np.ndarray, beam_alive_top: np.ndarray | None) -> np.ndarray:
    """Full-strength sunlight columns: 15 where every cell above is air.

    ``beam_alive_top`` marks columns whose beam enters from above the array
    (None means the array top *is* the sky).
    """
    air = blocks == AIR
    if beam_alive_top is not None:
        air = air & np.ones_like(air)  # copy-on-write guard for the cumprod
        air[:, :, -1] &= beam_alive_top
    beam = np.cumprod(air[:, :, ::-1], axis=2, dtype=np.uint8)[:, :, ::-1]
    return beam.astype(_I16) * MAX_LIGHT


def compute_window_light(
    blocks: np.ndarray, registry: BlockRegistry, beam_alive_top: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Sky and block light for a full block array (any shape, y last)."""
    receives = ~registry.opaque[blocks]
    cost = (1 + registry.light_attenuation[blocks]).astype(_I16)

    sky = _sky_beams(blocks, beam_alive_top)
    _flood(sky, receives, cost)

    block = registry.emission[blocks].astype(_I16)
    _flood(block, receives, cost)
    return sky.astype(np.uint8), block.astype(np.uint8)


def relight_box(world, x: int, y: int, z: int) -> set[tuple[int, int]]:
    """Recompute light in a box around an edited block.

    Seeds the box boundary with stored light (valid: the edit cannot affect
    cells further than MAX_LIGHT away), recomputes the interior from scratch
    and writes back only cells that changed.  Returns the chunk keys whose
    light changed (they need remeshing).
    """
    r = RELIGHT_RADIUS
    x0, x1 = x - r, x + r + 1
    z0, z1 = z - r, z + r + 1
    y0, y1 = max(y - r, 0), min(y + r + 1, CHUNK_Y)

    blocks, valid = world.copy_block_box(x0, x1, y0, y1, z0, z1)
    old_sky, old_blk = world.copy_light_box(x0, x1, y0, y1, z0, z1)
    registry = world.registry

    receives = (~registry.opaque[blocks]) & valid
    cost = (1 + registry.light_attenuation[blocks]).astype(_I16)

    # Sky: beams re-enter from the top shell where it was fully lit.
    if y1 >= CHUNK_Y:
        beam_top = None
    else:
        beam_top = old_sky[:, :, -1] == MAX_LIGHT
    sky = _sky_beams(blocks, beam_top)
    # Boundary shells keep their stored values as flood sources.
    _seed_boundary(sky, old_sky)
    _flood(sky, receives, cost)

    blk = registry.emission[blocks].astype(_I16)
    _seed_boundary(blk, old_blk.astype(_I16))
    _flood(blk, receives, cost)

    sky8 = sky.astype(np.uint8)
    blk8 = blk.astype(np.uint8)
    changed = (sky8 != old_sky) | (blk8 != old_blk)
    changed &= valid
    if not changed.any():
        return set()
    return world.write_light_box(x0, x1, y0, y1, z0, z1, sky8, blk8, changed)


def _seed_boundary(light: np.ndarray, stored) -> None:
    stored = np.asarray(stored, dtype=_I16)
    for axis in range(3):
        first = tuple(0 if a == axis else slice(None) for a in range(3))
        last = tuple(-1 if a == axis else slice(None) for a in range(3))
        np.maximum(light[first], stored[first], out=light[first])
        np.maximum(light[last], stored[last], out=light[last])
