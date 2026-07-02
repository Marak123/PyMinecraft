"""Chunk data container and lifecycle state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np


class ChunkState(Enum):
    GENERATED = auto()   # block data exists, no mesh yet
    MESHING = auto()     # a mesh job is in flight
    READY = auto()       # mesh uploaded (possibly empty)


@dataclass
class Chunk:
    cx: int
    cz: int
    blocks: np.ndarray  # (CHUNK_X, CHUNK_Z, CHUNK_Y) uint8
    state: ChunkState = ChunkState.GENERATED
    modified: bool = field(default=False)  # needs saving on unload
    dirty: bool = field(default=False)     # needs remeshing

    @property
    def key(self) -> tuple[int, int]:
        return (self.cx, self.cz)
