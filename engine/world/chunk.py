"""Chunk data container and lifecycle state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np


class ChunkState(Enum):
    GENERATED = auto()   # block data exists, no light/mesh yet
    LIGHTING = auto()    # a lighting job is in flight
    LIT = auto()         # light arrays ready, no mesh yet
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
    # 0..15 light fields, filled by the lighting stage (None until computed).
    sky_light: np.ndarray | None = None
    block_light: np.ndarray | None = None

    @property
    def key(self) -> tuple[int, int]:
        return (self.cx, self.cz)

    @property
    def has_light(self) -> bool:
        return self.sky_light is not None
