"""Chunk dimensions and coordinate conversions.

Chunk size is a project-wide constant kept in one place so it can be
re-benchmarked later (the design spec explicitly asks not to assume any
particular size is optimal).  16x16x128 keeps single-chunk remesh cost low,
which matters most in Python where block edits remesh the whole chunk.
"""

from __future__ import annotations

CHUNK_X = 16
CHUNK_Z = 16
CHUNK_Y = 128


def world_to_chunk(wx: int, wz: int) -> tuple[int, int]:
    """Block coords -> chunk coords (floor division handles negatives)."""
    return wx >> 4, wz >> 4


def world_to_local(wx: int, wz: int) -> tuple[int, int]:
    return wx & 15, wz & 15
