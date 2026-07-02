"""Voxel ray casting (Amanatides & Woo DDA).

Steps the ray voxel-by-voxel, which is exact — no missed corners — and
cheap for the short reach distances block interaction needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, inf
from typing import Callable

import numpy as np

from engine.world.world import World


@dataclass
class RayHit:
    block: tuple[int, int, int]     # the voxel that was hit
    previous: tuple[int, int, int]  # the empty voxel the ray came from
    distance: float


def raycast(
    world: World,
    origin: np.ndarray,
    direction: np.ndarray,
    max_distance: float,
    targetable: Callable[[int], bool],
) -> RayHit | None:
    x, y, z = floor(origin[0]), floor(origin[1]), floor(origin[2])
    dx, dy, dz = float(direction[0]), float(direction[1]), float(direction[2])

    step_x = 1 if dx > 0 else -1
    step_y = 1 if dy > 0 else -1
    step_z = 1 if dz > 0 else -1

    # Distance along the ray to the first boundary crossing per axis, and the
    # distance between subsequent crossings.
    t_max_x = ((x + (step_x > 0)) - origin[0]) / dx if dx != 0 else inf
    t_max_y = ((y + (step_y > 0)) - origin[1]) / dy if dy != 0 else inf
    t_max_z = ((z + (step_z > 0)) - origin[2]) / dz if dz != 0 else inf
    t_delta_x = abs(1.0 / dx) if dx != 0 else inf
    t_delta_y = abs(1.0 / dy) if dy != 0 else inf
    t_delta_z = abs(1.0 / dz) if dz != 0 else inf

    prev = (x, y, z)
    t = 0.0
    while t <= max_distance:
        block_id = world.get_block(x, y, z)
        if targetable(block_id):
            return RayHit((x, y, z), prev, t)
        prev = (x, y, z)
        if t_max_x <= t_max_y and t_max_x <= t_max_z:
            t = t_max_x
            t_max_x += t_delta_x
            x += step_x
        elif t_max_y <= t_max_z:
            t = t_max_y
            t_max_y += t_delta_y
            y += step_y
        else:
            t = t_max_z
            t_max_z += t_delta_z
            z += step_z
    return None
