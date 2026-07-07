"""World manager: chunk storage, block access, edit bookkeeping.

Owns the chunk dictionary and coordinates generation/persistence, but does
not generate terrain or build meshes itself — those live in their own
modules per the separation-of-concerns rule.
"""

from __future__ import annotations

import numpy as np

from engine.core.log import get_logger
from engine.world.blocks import AIR, BlockRegistry
from engine.world.chunk import Chunk
from engine.world.coords import CHUNK_X, CHUNK_Y, CHUNK_Z, world_to_chunk
from engine.world.generation import WorldGenerator
from engine.world.save import WorldStorage

_log = get_logger("world")

# Offsets of the 8 horizontal neighbours (used for meshing dependencies).
NEIGHBOURS_8 = tuple(
    (dx, dz) for dx in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dz) != (0, 0)
)


class World:
    def __init__(
        self, generator: WorldGenerator, registry: BlockRegistry, storage: WorldStorage
    ) -> None:
        self.generator = generator
        self.registry = registry
        self.storage = storage
        self.chunks: dict[tuple[int, int], Chunk] = {}
        # Chunks whose mesh must be rebuilt this frame (player edits).
        self.dirty_chunks: set[tuple[int, int]] = set()
        # Optional callback(x, y, z) fired after any successful set_block —
        # the fluid/falling-block systems subscribe here.
        self.on_change = None

    # -- chunk access -----------------------------------------------------------
    def get_chunk(self, cx: int, cz: int) -> Chunk | None:
        return self.chunks.get((cx, cz))

    def has_all_neighbours(self, cx: int, cz: int) -> bool:
        return all((cx + dx, cz + dz) in self.chunks for dx, dz in NEIGHBOURS_8)

    def produce_chunk_blocks(self, cx: int, cz: int) -> tuple[np.ndarray, bool]:
        """Load from disk or generate. Returns (blocks, was_loaded_from_disk).

        Thread-safe: touches only the generator (pure) and storage (reads).
        """
        saved = self.storage.load_chunk(cx, cz)
        if saved is not None and saved.shape == (CHUNK_X, CHUNK_Z, CHUNK_Y):
            return np.ascontiguousarray(saved, dtype=np.uint8), True
        return self.generator.generate_chunk(cx, cz), False

    def add_chunk(self, chunk: Chunk) -> None:
        self.chunks[chunk.key] = chunk

    def remove_chunk(self, cx: int, cz: int) -> Chunk | None:
        """Detach a chunk. The caller owns persisting it if modified —
        the streamer pushes that I/O onto the worker pool."""
        return self.chunks.pop((cx, cz), None)

    def save_all_modified(self) -> int:
        count = 0
        for chunk in self.chunks.values():
            if chunk.modified:
                self.storage.save_chunk(chunk.cx, chunk.cz, chunk.blocks)
                chunk.modified = False
                count += 1
        return count

    # -- block access -------------------------------------------------------------
    def get_block(self, wx: int, wy: int, wz: int) -> int:
        """Block id at world coords; AIR outside loaded chunks / world height."""
        if wy < 0 or wy >= CHUNK_Y:
            return AIR
        chunk = self.chunks.get(world_to_chunk(wx, wz))
        if chunk is None:
            return AIR
        return int(chunk.blocks[wx & 15, wz & 15, wy])

    def is_solid(self, wx: int, wy: int, wz: int) -> bool:
        """Physics query. Unloaded chunks count as solid so entities can never
        fall into terrain that simply has not streamed in yet."""
        if wy < 0:
            return True
        if wy >= CHUNK_Y:
            return False
        chunk = self.chunks.get(world_to_chunk(wx, wz))
        if chunk is None:
            return True
        return bool(self.registry.solid[chunk.blocks[wx & 15, wz & 15, wy]])

    def set_block(self, wx: int, wy: int, wz: int, block_id: int) -> bool:
        if wy < 1 or wy >= CHUNK_Y - 1:
            return False
        key = world_to_chunk(wx, wz)
        chunk = self.chunks.get(key)
        if chunk is None:
            return False
        lx, lz = wx & 15, wz & 15
        if chunk.blocks[lx, lz, wy] == block_id:
            return False
        chunk.blocks[lx, lz, wy] = block_id
        chunk.modified = True
        self._mark_dirty_around(key, lx, lz)
        if self.on_change is not None:
            self.on_change(wx, wy, wz)
        return True

    def _mark_dirty_around(self, key: tuple[int, int], lx: int, lz: int) -> None:
        """Mark the edited chunk dirty, plus any neighbour whose mesh can see
        the edited block (border blocks affect neighbour AO/visibility)."""
        cx, cz = key
        dxs = [0] + ([-1] if lx == 0 else [1] if lx == CHUNK_X - 1 else [])
        dzs = [0] + ([-1] if lz == 0 else [1] if lz == CHUNK_Z - 1 else [])
        for dx in dxs:
            for dz in dzs:
                nkey = (cx + dx, cz + dz)
                if nkey in self.chunks:
                    self.dirty_chunks.add(nkey)

    # -- lighting support ------------------------------------------------------------
    def has_lit_neighbourhood(self, cx: int, cz: int) -> bool:
        """True when the chunk and all 8 neighbours have light computed."""
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                chunk = self.chunks.get((cx + dx, cz + dz))
                if chunk is None or not chunk.has_light:
                    return False
        return True

    def build_light_window(self, cx: int, cz: int) -> np.ndarray | None:
        """3x3-chunk block snapshot for the lighting job (48x48xY).

        Light travels at most 15 blocks, so this window yields exact light
        for the centre chunk. Returns None if any neighbour is missing.
        """
        if (cx, cz) not in self.chunks or not self.has_all_neighbours(cx, cz):
            return None
        window = np.empty((CHUNK_X * 3, CHUNK_Z * 3, CHUNK_Y), dtype=np.uint8)
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                window[
                    (dx + 1) * CHUNK_X : (dx + 2) * CHUNK_X,
                    (dz + 1) * CHUNK_Z : (dz + 2) * CHUNK_Z,
                    :,
                ] = self.chunks[(cx + dx, cz + dz)].blocks
        return window

    # -- box copy/write (edit relighting) ---------------------------------------------
    def _overlapping_chunks(self, x0: int, x1: int, z0: int, z1: int):
        for ccx in range(x0 >> 4, ((x1 - 1) >> 4) + 1):
            for ccz in range(z0 >> 4, ((z1 - 1) >> 4) + 1):
                # Intersection of the box with this chunk, in both spaces.
                wx0 = max(x0, ccx * CHUNK_X)
                wx1 = min(x1, (ccx + 1) * CHUNK_X)
                wz0 = max(z0, ccz * CHUNK_Z)
                wz1 = min(z1, (ccz + 1) * CHUNK_Z)
                box = (slice(wx0 - x0, wx1 - x0), slice(wz0 - z0, wz1 - z0))
                local = (slice(wx0 - ccx * CHUNK_X, wx1 - ccx * CHUNK_X),
                         slice(wz0 - ccz * CHUNK_Z, wz1 - ccz * CHUNK_Z))
                yield (ccx, ccz), box, local

    def copy_block_box(
        self, x0: int, x1: int, y0: int, y1: int, z0: int, z1: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Blocks in a world-space box + validity mask (False = unloaded)."""
        blocks = np.zeros((x1 - x0, z1 - z0, y1 - y0), dtype=np.uint8)
        valid = np.zeros(blocks.shape, dtype=bool)
        ys = slice(y0, y1)
        for key, box, local in self._overlapping_chunks(x0, x1, z0, z1):
            chunk = self.chunks.get(key)
            if chunk is None:
                continue
            blocks[box[0], box[1], :] = chunk.blocks[local[0], local[1], ys]
            valid[box[0], box[1], :] = True
        return blocks, valid

    def copy_light_box(
        self, x0: int, x1: int, y0: int, y1: int, z0: int, z1: int
    ) -> tuple[np.ndarray, np.ndarray]:
        sky = np.zeros((x1 - x0, z1 - z0, y1 - y0), dtype=np.uint8)
        blk = np.zeros(sky.shape, dtype=np.uint8)
        ys = slice(y0, y1)
        for key, box, local in self._overlapping_chunks(x0, x1, z0, z1):
            chunk = self.chunks.get(key)
            if chunk is None or not chunk.has_light:
                continue
            sky[box[0], box[1], :] = chunk.sky_light[local[0], local[1], ys]
            blk[box[0], box[1], :] = chunk.block_light[local[0], local[1], ys]
        return sky, blk

    def write_light_box(
        self,
        x0: int, x1: int, y0: int, y1: int, z0: int, z1: int,
        sky: np.ndarray, blk: np.ndarray, changed: np.ndarray,
    ) -> set[tuple[int, int]]:
        """Apply changed light cells back to chunk storage.

        Returns the keys of chunks that actually changed (need remeshing).
        """
        touched: set[tuple[int, int]] = set()
        ys = slice(y0, y1)
        for key, box, local in self._overlapping_chunks(x0, x1, z0, z1):
            chunk = self.chunks.get(key)
            if chunk is None or not chunk.has_light:
                continue
            mask = changed[box[0], box[1], :]
            if not mask.any():
                continue
            chunk.sky_light[local[0], local[1], ys][mask] = sky[box[0], box[1], :][mask]
            chunk.block_light[local[0], local[1], ys][mask] = blk[box[0], box[1], :][mask]
            touched.add(key)
        return touched

    # -- meshing support ------------------------------------------------------------
    def build_mesh_input(
        self, cx: int, cz: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Padded (blocks, sky_light, block_light) snapshot for the mesher.

        The 1-block halo is needed for border face visibility, AO and smooth
        light. Returns None until all 8 neighbours exist *and* are lit.
        """
        chunk = self.chunks.get((cx, cz))
        if chunk is None or not chunk.has_light or not self.has_lit_neighbourhood(cx, cz):
            return None

        shape = (CHUNK_X + 2, CHUNK_Z + 2, CHUNK_Y + 2)
        blocks = np.zeros(shape, dtype=np.uint8)
        sky = np.zeros(shape, dtype=np.uint8)
        blk = np.zeros(shape, dtype=np.uint8)
        # Below-world counts as solid rock (culls downward faces of bedrock);
        # above-world stays AIR with full skylight.
        blocks[:, :, 0] = self.registry.id_of("bedrock")
        sky[:, :, -1] = 15

        # Source region per neighbour offset -> destination slices in the pad.
        for dx, dz in ((0, 0), *NEIGHBOURS_8):
            neighbour = self.chunks[(cx + dx, cz + dz)]
            sx = slice(1, -1) if dx == 0 else (slice(0, 1) if dx == -1 else slice(-1, None))
            sz = slice(1, -1) if dz == 0 else (slice(0, 1) if dz == -1 else slice(-1, None))
            nx = slice(None) if dx == 0 else (slice(-1, None) if dx == -1 else slice(0, 1))
            nz = slice(None) if dz == 0 else (slice(-1, None) if dz == -1 else slice(0, 1))
            blocks[sx, sz, 1:-1] = neighbour.blocks[nx, nz, :]
            sky[sx, sz, 1:-1] = neighbour.sky_light[nx, nz, :]
            blk[sx, sz, 1:-1] = neighbour.block_light[nx, nz, :]
        return blocks, sky, blk
