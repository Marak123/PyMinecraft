"""Nether portal detection and dimension travel (plan phase 6.7).

An obsidian frame enclosing an air pocket is ignited into portal blocks;
standing in one for a moment carries the player to the matching dimension,
with an 8:1 horizontal coordinate scale between overworld and Nether.
"""

from __future__ import annotations

from math import floor

from engine.world.blocks import AIR
from engine.world.world import World

_MAX_PORTAL_CELLS = 16
NETHER_SCALE = 8


def find_portal_area(world: World, start: tuple[int, int, int]) -> list[tuple[int, int, int]] | None:
    """Flood-fill an air pocket from `start`; return its cells if fully
    enclosed by obsidian in a single vertical plane, else None."""
    obsidian = world.registry.id_of("obsidian")
    portal = world.registry.id_of("nether_portal")

    def is_air(c):
        b = world.get_block(*c)
        return b == AIR or b == portal

    if not is_air(start):
        return None

    # A portal is planar: figure out whether it extends in X or Z.
    for axis in (0, 2):
        cells: list[tuple[int, int, int]] = []
        stack = [start]
        seen = {start}
        ok = True
        while stack and ok:
            c = stack.pop()
            cells.append(c)
            if len(cells) > _MAX_PORTAL_CELLS:
                ok = False
                break
            # Neighbours within the plane: up/down and along `axis`.
            for d in (
                (0, 1, 0), (0, -1, 0),
                (1, 0, 0) if axis == 0 else (0, 0, 1),
                (-1, 0, 0) if axis == 0 else (0, 0, -1),
            ):
                nc = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
                b = world.get_block(*nc)
                if b == obsidian:
                    continue
                if is_air(nc):
                    if nc not in seen:
                        seen.add(nc)
                        stack.append(nc)
                else:
                    ok = False  # leaks into non-frame material
                    break
        if ok and 2 <= len(cells) <= _MAX_PORTAL_CELLS:
            return cells
    return None


def ignite(world: World, start: tuple[int, int, int]) -> bool:
    cells = find_portal_area(world, start)
    if cells is None:
        return False
    portal = world.registry.id_of("nether_portal")
    changed: set[tuple[int, int]] = set()
    for c in cells:
        if world.set_block(c[0], c[1], c[2], portal):
            changed.add((c[0] >> 4, c[2] >> 4))
    from engine.world import lighting
    for c in cells:
        world.dirty_chunks |= lighting.relight_box(world, *c)
    return bool(cells)


def map_position(pos, to_nether: bool):
    """Scale horizontal coordinates between the two dimensions."""
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    if to_nether:
        return [x / NETHER_SCALE, y, z / NETHER_SCALE]
    return [x * NETHER_SCALE, y, z * NETHER_SCALE]
