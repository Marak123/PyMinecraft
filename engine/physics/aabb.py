"""AABB-vs-voxel-grid collision (collide and slide).

Movement is resolved one axis at a time — the standard robust approach for
voxel worlds: no tunnelling at survival-level speeds, and sliding along
walls falls out for free.  The swept volume is at most a few dozen voxels,
so plain Python loops are fast enough here.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor

import numpy as np

from engine.world.world import World

_EPS = 1e-4


@dataclass
class MoveResult:
    position: np.ndarray
    on_ground: bool
    hit_head: bool
    hit_wall: bool


def _box_voxels(
    min_x: float, min_y: float, min_z: float, max_x: float, max_y: float, max_z: float
):
    """Integer voxel coords overlapped by an AABB (max bound exclusive-ish)."""
    for x in range(floor(min_x), floor(max_x - _EPS) + 1):
        for y in range(floor(min_y), floor(max_y - _EPS) + 1):
            for z in range(floor(min_z), floor(max_z - _EPS) + 1):
                yield x, y, z


def box_intersects_solid(
    world: World, feet: np.ndarray, half_w: float, height: float
) -> bool:
    """True if the entity box at `feet` overlaps any solid voxel."""
    for x, y, z in _box_voxels(
        feet[0] - half_w, feet[1], feet[2] - half_w,
        feet[0] + half_w, feet[1] + height, feet[2] + half_w,
    ):
        if world.is_solid(x, y, z):
            return True
    return False


def block_intersects_box(
    block: tuple[int, int, int], feet: np.ndarray, half_w: float, height: float
) -> bool:
    """Would placing a full cube at `block` overlap the entity box?"""
    bx, by, bz = block
    return (
        bx + 1 > feet[0] - half_w - _EPS
        and bx < feet[0] + half_w + _EPS
        and by + 1 > feet[1] - _EPS
        and by < feet[1] + height + _EPS
        and bz + 1 > feet[2] - half_w - _EPS
        and bz < feet[2] + half_w + _EPS
    )


def move_with_collisions(
    world: World,
    feet: np.ndarray,
    velocity: np.ndarray,
    dt: float,
    half_w: float,
    height: float,
) -> MoveResult:
    """Move an AABB through the voxel grid, clamping against solids per axis.

    Returns the new feet position plus contact flags; the caller is
    responsible for zeroing velocity components on the axes that collided.
    """
    pos = feet.astype(np.float64).copy()
    delta = velocity * dt
    on_ground = False
    hit_head = False
    hit_wall = False

    # X and Z first so walking into a wall while falling still slides down it,
    # then Y so ground contact reflects the final horizontal position.
    for axis in (0, 2, 1):
        d = float(delta[axis])
        if d == 0.0:
            continue
        pos[axis] += d
        min_x, max_x = pos[0] - half_w, pos[0] + half_w
        min_y, max_y = pos[1], pos[1] + height
        min_z, max_z = pos[2] - half_w, pos[2] + half_w

        collided = False
        for x, y, z in _box_voxels(min_x, min_y, min_z, max_x, max_y, max_z):
            if not world.is_solid(x, y, z):
                continue
            collided = True
            # min/max against every overlapping solid: the *nearest* one wins
            # (never clamp forward past a previously found closer voxel).
            if axis == 0:
                if d > 0:
                    pos[0] = min(pos[0], x - half_w - _EPS)
                else:
                    pos[0] = max(pos[0], x + 1 + half_w + _EPS)
            elif axis == 2:
                if d > 0:
                    pos[2] = min(pos[2], z - half_w - _EPS)
                else:
                    pos[2] = max(pos[2], z + 1 + half_w + _EPS)
            else:
                if d > 0:
                    pos[1] = min(pos[1], y - height - _EPS)
                    hit_head = True
                else:
                    pos[1] = max(pos[1], y + 1 + _EPS)
                    on_ground = True
        if collided and axis != 1:
            hit_wall = True

    return MoveResult(pos, on_ground, hit_head, hit_wall)
