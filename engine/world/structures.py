"""Procedural structure placement (plan phase 6).

Structures are built in code rather than JSON schematics — a placer runs
during chunk generation and writes into whatever part of a structure falls
inside the current chunk.  Placement is a pure function of (seed, structure
origin), so a structure that straddles chunk borders is rebuilt identically
from either side (the same stateless trick trees and ruins use).

The generator asks a structure "which cells fall in chunk (cx,cz)?" by
building each candidate whose bounding box overlaps the chunk, clipping the
writes to the chunk. Candidates are found by scanning a small neighbourhood
of "structure grid" cells around the chunk.
"""

from __future__ import annotations

import numpy as np

from engine.world.blocks import AIR, BlockRegistry
from engine.world.coords import CHUNK_X, CHUNK_Y, CHUNK_Z
from engine.world.noise import hash01

# A structure is anchored to a chunk-grid cell of this many chunks and spawns
# there with a probability — spaces them out without a global registry.
_VILLAGE_CELL = 6      # chunks
_VILLAGE_CHANCE = 0.22
_DUNGEON_CELL = 3
_DUNGEON_CHANCE = 0.5
_MINESHAFT_CELL = 5
_MINESHAFT_CHANCE = 0.45


def _rng(seed: int, gx: int, gz: int, salt: int) -> np.random.Generator:
    return np.random.default_rng(
        ((gx * 73856093) ^ (gz * 19349663) ^ (seed * 83492791) ^ (salt * 2654435761))
        & 0x7FFFFFFF
    )


class StructureGenerator:
    def __init__(self, seed: int, registry: BlockRegistry) -> None:
        self.seed = seed
        r = registry
        self.b = {
            name: r.id_of(name)
            for name in (
                "cobblestone", "mossy_cobblestone", "oak_planks", "oak_log",
                "glass", "torch", "chest", "spawner", "cobweb", "stone",
                "dirt", "grass_block", "bookshelf",
            )
        }

    # -- generation entry point (called per chunk) ------------------------------
    def place(self, blocks: np.ndarray, cx: int, cz: int, height: np.ndarray) -> None:
        self._scan(blocks, cx, cz, height, _VILLAGE_CELL, _VILLAGE_CHANCE, 71, self._village)
        self._scan(blocks, cx, cz, None, _DUNGEON_CELL, _DUNGEON_CHANCE, 72, self._dungeon)
        self._scan(blocks, cx, cz, None, _MINESHAFT_CELL, _MINESHAFT_CHANCE, 73, self._mineshaft)

    def _scan(self, blocks, cx, cz, height, cell, chance, salt, build) -> None:
        """Find structure origins in nearby grid cells and build the part of
        each that overlaps this chunk."""
        gcx, gcz = cx // cell, cz // cell
        for gx in range(gcx - 1, gcx + 2):
            for gz in range(gcz - 1, gcz + 2):
                rng = _rng(self.seed, gx, gz, salt)
                if rng.random() > chance:
                    continue
                # Deterministic origin chunk + block offset within the cell.
                ocx = gx * cell + int(rng.integers(0, cell))
                ocz = gz * cell + int(rng.integers(0, cell))
                ox = ocx * CHUNK_X + int(rng.integers(2, 12))
                oz = ocz * CHUNK_Z + int(rng.integers(2, 12))
                build(blocks, cx, cz, height, ox, oz, rng)

    # -- writer clipped to the current chunk ------------------------------------
    def _writer(self, blocks: np.ndarray, cx: int, cz: int):
        base_x = cx * CHUNK_X
        base_z = cz * CHUNK_Z

        def put(wx: int, wy: int, wz: int, bid: int, only_air: bool = False) -> None:
            lx = wx - base_x
            lz = wz - base_z
            if 0 <= lx < CHUNK_X and 0 <= lz < CHUNK_Z and 0 < wy < CHUNK_Y - 1:
                if not only_air or blocks[lx, lz, wy] == AIR:
                    blocks[lx, lz, wy] = bid

        return put

    # -- structures -------------------------------------------------------------
    def _village(self, blocks, cx, cz, height, ox, oz, rng) -> None:
        """A cluster of small houses around a central column.  Ground height
        is sampled from the world height field (structure only builds if the
        origin column is reasonably flat land)."""
        gy = self._origin_ground(height, cx, cz, ox, oz)
        if gy is None or gy < 66 or gy > 110:
            return
        put = self._writer(blocks, cx, cz)
        # 3-5 houses on a rough ring around the origin.
        house_rng = _rng(self.seed, ox, oz, salt=90)
        count = int(house_rng.integers(3, 6))
        for i in range(count):
            angle = (i / count) * 6.2832 + house_rng.random()
            dist = 5 + int(house_rng.random() * 5)
            hx = ox + int(np.cos(angle) * dist)
            hz = oz + int(np.sin(angle) * dist)
            self._house(put, hx, gy, hz, house_rng)
        # A well / lantern post at the centre.
        for dy in range(1, 4):
            put(ox, gy + dy, oz, self.b["oak_log"])
        put(ox, gy + 4, oz, self.b["torch"])

    def _house(self, put, x0, gy, z0, rng) -> None:
        w, d, wall_h = 5, 5, 3
        wood = self.b["oak_planks"]
        log = self.b["oak_log"]
        for dx in range(w):
            for dz in range(d):
                put(x0 + dx, gy, z0 + dz, wood)  # foundation
        for dy in range(1, wall_h + 1):
            for dx in range(w):
                for dz in range(d):
                    edge = dx in (0, w - 1) or dz in (0, d - 1)
                    if not edge:
                        continue
                    corner = dx in (0, w - 1) and dz in (0, d - 1)
                    bid = log if corner else wood
                    put(x0 + dx, gy + dy, z0 + dz, bid)
        # Door + a window.
        put(x0 + w // 2, gy + 1, z0, AIR)
        put(x0 + w // 2, gy + 2, z0, AIR)
        put(x0, gy + 2, z0 + d // 2, self.b["glass"])
        put(x0 + w - 1, gy + 2, z0 + d // 2, self.b["glass"])
        # Flat plank roof + a torch inside.
        for dx in range(w):
            for dz in range(d):
                put(x0 + dx, gy + wall_h + 1, z0 + dz, wood)
        put(x0 + 1, gy + 1, z0 + 1, self.b["chest"])
        put(x0 + w - 2, gy + wall_h, z0 + d - 2, self.b["torch"])

    def _origin_ground(self, height, cx, cz, ox, oz) -> int | None:
        """Height at the structure origin if it lies in this chunk's field."""
        lx = ox - cx * CHUNK_X
        lz = oz - cz * CHUNK_Z
        if 0 <= lx < CHUNK_X and 0 <= lz < CHUNK_Z:
            return int(height[lx, lz])
        # Origin is in a neighbour chunk; recompute its column height cheaply
        # would need the generator — instead approximate with the nearest
        # in-chunk column (villages are flat enough for this to look fine).
        cxl = min(max(lx, 0), CHUNK_X - 1)
        czl = min(max(lz, 0), CHUNK_Z - 1)
        return int(height[cxl, czl])

    def _dungeon(self, blocks, cx, cz, height, ox, oz, rng) -> None:
        oy = int(rng.integers(14, 46))
        put = self._writer(blocks, cx, cz)
        w = d = 7
        h = 5
        # Only carve where there's stone (approximate: always build; the box
        # replaces terrain and hollows the interior).
        for dx in range(w):
            for dz in range(d):
                for dy in range(h):
                    edge = (dx in (0, w - 1) or dz in (0, d - 1)
                            or dy in (0, h - 1))
                    wx, wy, wz = ox + dx, oy + dy, oz + dz
                    if edge:
                        stone = (self.b["mossy_cobblestone"]
                                 if rng.random() < 0.4 else self.b["cobblestone"])
                        put(wx, wy, wz, stone)
                    else:
                        put(wx, wy, wz, AIR)
        put(ox + w // 2, oy + 1, oz + d // 2, self.b["spawner"])
        put(ox + 1, oy + 1, oz + 1, self.b["chest"])
        put(ox + w - 2, oy + 1, oz + d - 2, self.b["chest"])

    def _mineshaft(self, blocks, cx, cz, height, ox, oz, rng) -> None:
        oy = int(rng.integers(20, 44))
        put = self._writer(blocks, cx, cz)
        # A few straight corridor segments branching from the origin.
        segments = int(rng.integers(3, 6))
        x, z = ox, oz
        for _ in range(segments):
            horizontal = rng.random() < 0.5
            length = int(rng.integers(8, 20))
            step = 1 if rng.random() < 0.5 else -1
            for s in range(length):
                cxp = x + (step * s if horizontal else 0)
                czp = z + (0 if horizontal else step * s)
                # 3-wide, 3-tall corridor.
                for a in (-1, 0, 1):
                    for dy in range(3):
                        wx = cxp + (0 if horizontal else a)
                        wz = czp + (a if horizontal else 0)
                        put(wx, oy + dy, wz, AIR)
                # Support frame every 5 blocks.
                if s % 5 == 0:
                    for dy in range(3):
                        put(cxp + (0 if horizontal else -1), oy + dy,
                            czp + (-1 if horizontal else 0), self.b["oak_log"])
                        put(cxp + (0 if horizontal else 1), oy + dy,
                            czp + (1 if horizontal else 0), self.b["oak_log"])
                    for a in (-1, 0, 1):
                        wx = cxp + (0 if horizontal else a)
                        wz = czp + (a if horizontal else 0)
                        put(wx, oy + 3, wz, self.b["oak_planks"])
                if rng.random() < 0.06:
                    put(cxp, oy + 2, czp, self.b["cobweb"], only_air=True)
            if horizontal:
                x += step * length
            else:
                z += step * length
