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
        chunk = self.chunks.pop((cx, cz), None)
        if chunk is not None and chunk.modified:
            self.storage.save_chunk(cx, cz, chunk.blocks)
            chunk.modified = False
        return chunk

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

    # -- meshing support ------------------------------------------------------------
    def build_padded_blocks(self, cx: int, cz: int) -> np.ndarray | None:
        """Chunk blocks padded by 1 in every axis with real neighbour data.

        The mesher needs this halo for face visibility and AO at borders.
        Returns None if any of the 8 horizontal neighbours is missing.
        """
        chunk = self.chunks.get((cx, cz))
        if chunk is None or not self.has_all_neighbours(cx, cz):
            return None

        padded = np.zeros((CHUNK_X + 2, CHUNK_Z + 2, CHUNK_Y + 2), dtype=np.uint8)
        padded[1:-1, 1:-1, 1:-1] = chunk.blocks
        # Below-world counts as solid rock (culls downward faces of bedrock);
        # above-world stays AIR.
        padded[:, :, 0] = self.registry.id_of("bedrock")

        neighbours = {
            (dx, dz): self.chunks[(cx + dx, cz + dz)].blocks for dx, dz in NEIGHBOURS_8
        }
        padded[0, 1:-1, 1:-1] = neighbours[(-1, 0)][-1, :, :]
        padded[-1, 1:-1, 1:-1] = neighbours[(1, 0)][0, :, :]
        padded[1:-1, 0, 1:-1] = neighbours[(0, -1)][:, -1, :]
        padded[1:-1, -1, 1:-1] = neighbours[(0, 1)][:, 0, :]
        padded[0, 0, 1:-1] = neighbours[(-1, -1)][-1, -1, :]
        padded[0, -1, 1:-1] = neighbours[(-1, 1)][-1, 0, :]
        padded[-1, 0, 1:-1] = neighbours[(1, -1)][0, -1, :]
        padded[-1, -1, 1:-1] = neighbours[(1, 1)][0, 0, :]
        return padded
