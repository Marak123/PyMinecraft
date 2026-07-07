"""3D A* pathfinding on the voxel grid (plan phase 7.4).

Walkable = feet + head cells clear, a solid floor under the feet, stepping
up at most 1 block and dropping at most 3.  Capped step count keeps searches
for unreachable goals cheap.  Manhattan heuristic (admissible on a grid).
"""

from __future__ import annotations

import heapq
from math import floor

from engine.world.blocks import AIR
from engine.world.world import World

# 8 horizontal moves (4 cardinal + 4 diagonal); vertical handled via step/drop.
_MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def _standable(world: World, x: int, y: int, z: int) -> bool:
    reg = world.registry
    if reg.solid[world.get_block(x, y, z)] or reg.solid[world.get_block(x, y + 1, z)]:
        return False  # body must fit
    return reg.solid[world.get_block(x, y - 1, z)]


def _resolve_floor(world: World, x: int, y: int, z: int) -> int | None:
    """Given a target column, return a standable y stepping up 1 / down 3."""
    for ny in (y + 1, y, y - 1, y - 2, y - 3):
        if _standable(world, x, ny, z):
            return ny
    return None


def find_path(world: World, start, goal, max_steps: int = 180) -> list[tuple[int, int, int]] | None:
    sx, sy, sz = floor(start[0]), floor(start[1]), floor(start[2])
    gx, gy, gz = floor(goal[0]), floor(goal[1]), floor(goal[2])
    start_c = (sx, sy, sz)
    goal_c = (gx, gy, gz)

    def h(c):
        return abs(c[0] - gx) + abs(c[1] - gy) + abs(c[2] - gz)

    open_heap = [(h(start_c), 0, start_c)]
    came: dict[tuple, tuple] = {}
    g_score = {start_c: 0}
    steps = 0

    while open_heap and steps < max_steps:
        steps += 1
        _, cost, cur = heapq.heappop(open_heap)
        if abs(cur[0] - gx) <= 1 and abs(cur[2] - gz) <= 1 and abs(cur[1] - gy) <= 2:
            return _reconstruct(came, cur)
        for dx, dz in _MOVES:
            nx, nz = cur[0] + dx, cur[2] + dz
            ny = _resolve_floor(world, nx, cur[1], nz)
            if ny is None:
                continue
            # Diagonal squeeze check: don't cut through a solid corner.
            if dx != 0 and dz != 0:
                if world.registry.solid[world.get_block(cur[0] + dx, cur[1], cur[2])] and \
                   world.registry.solid[world.get_block(cur[0], cur[1], cur[2] + dz)]:
                    continue
            nc = (nx, ny, nz)
            step_cost = cost + (14 if dx and dz else 10) + abs(ny - cur[1]) * 2
            if step_cost < g_score.get(nc, 1 << 30):
                g_score[nc] = step_cost
                came[nc] = cur
                heapq.heappush(open_heap, (step_cost + h(nc) * 10, step_cost, nc))
    return None


def _reconstruct(came, cur) -> list[tuple[int, int, int]]:
    path = [cur]
    while cur in came:
        cur = came[cur]
        path.append(cur)
    path.reverse()
    return path
