"""Vectorised chunk mesher.

Face-culling mesher with per-vertex ambient occlusion, fully vectorised in
NumPy — no per-block Python loops.  Emits the compressed 8-byte vertex
format described in :mod:`engine.graphics.cubegeom`.

Runs on worker threads: it only reads immutable registry tables and the
padded block snapshot it is given.

(Greedy meshing is a known future optimisation; it conflicts with per-vertex
AO and needs benchmarking first — see docs/ROADMAP.md.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine.graphics.cubegeom import AO_OFFSETS
from engine.world.blocks import (
    RENDER_CROSS,
    RENDER_CUTOUT,
    RENDER_LIQUID,
    RENDER_SOLID,
    BlockRegistry,
)
from engine.world.coords import CHUNK_X, CHUNK_Y, CHUNK_Z

# Face direction offsets in array axis order (x, z, y) — the block array is
# indexed [x, z, y] while face ids live in world (x, y, z) space.
_DIR_ARRAY_OFFSETS = (
    (1, 0, 0),   # +X
    (-1, 0, 0),  # -X
    (0, 0, 1),   # +Y
    (0, 0, -1),  # -Y
    (0, 1, 0),   # +Z
    (0, -1, 0),  # -Z
)

# Triangulation patterns: A splits along corner diagonal 0-2, B along 1-3.
# The mesher picks per-face whichever avoids the "AO anisotropy" artifact.
_PAT_A = np.array([0, 1, 2, 0, 2, 3], dtype=np.int64)
_PAT_B = np.array([1, 2, 3, 1, 3, 0], dtype=np.int64)

_U32 = np.uint32


@dataclass
class ChunkMeshData:
    """CPU-side mesh streams; each is a flat uint32 array (2 words/vertex)."""

    opaque: np.ndarray | None
    cutout: np.ndarray | None
    water: np.ndarray | None

    @property
    def vertex_count(self) -> int:
        return sum(len(a) // 2 for a in (self.opaque, self.cutout, self.water) if a is not None)


def _neighbour_view(padded: np.ndarray, ox: int, oz: int, oy: int) -> np.ndarray:
    return padded[
        1 + ox : 1 + CHUNK_X + ox,
        1 + oz : 1 + CHUNK_Z + oz,
        1 + oy : 1 + CHUNK_Y + oy,
    ]


def _pack_faces(
    xs: np.ndarray,
    zs: np.ndarray,
    ys: np.ndarray,
    face: int,
    ao: np.ndarray,          # (n, 4) uint32 values 0..3
    tex_layers: np.ndarray,  # (n,)
    emission: np.ndarray,    # (n,)
    flags: np.ndarray | None = None,  # (n,) 0/1
) -> np.ndarray:
    n = len(xs)
    base = (
        xs.astype(_U32)
        | (ys.astype(_U32) << _U32(6))
        | (zs.astype(_U32) << _U32(14))
        | (_U32(face) << _U32(24))
    )
    if flags is not None:
        base |= flags.astype(_U32) << _U32(27)

    corner_ids = np.arange(4, dtype=_U32)
    words0 = base[:, None] | (corner_ids[None, :] << _U32(20)) | (ao << _U32(22))

    # Flip triangulation so the quad diagonal runs through the brighter pair
    # of corners — removes the classic AO "bowtie" artifact.
    flip = (ao[:, 0] + ao[:, 2]) >= (ao[:, 1] + ao[:, 3])
    pattern = np.where(flip[:, None], _PAT_A[None, :], _PAT_B[None, :])
    verts0 = np.take_along_axis(words0, pattern, axis=1)  # (n, 6)

    word1 = (tex_layers.astype(_U32) | (emission.astype(_U32) << _U32(16)))
    out = np.empty((n, 6, 2), dtype=_U32)
    out[:, :, 0] = verts0
    out[:, :, 1] = word1[:, None]
    return out.reshape(-1)


def _corner_ao(
    opaque_padded: np.ndarray,
    xs: np.ndarray,
    zs: np.ndarray,
    ys: np.ndarray,
    face: int,
) -> np.ndarray:
    """Per-corner AO values (n, 4) in 0..3 (3 = fully open)."""
    n = len(xs)
    ao = np.empty((n, 4), dtype=_U32)
    for corner in range(4):
        offs = AO_OFFSETS[face, corner]  # (3 samples, 3 comps) in world (x,y,z)
        samples = []
        for k in range(3):
            ox, oy, oz = offs[k]
            samples.append(
                opaque_padded[xs + 1 + ox, zs + 1 + oz, ys + 1 + oy]
            )
        s1, s2, diag = samples
        occ = s1.astype(np.int32) + s2 + diag
        val = 3 - occ
        # Both edge neighbours solid -> corner fully occluded regardless.
        val[s1 & s2] = 0
        ao[:, corner] = val.astype(_U32)
    return ao


def build_chunk_meshes(padded: np.ndarray, registry: BlockRegistry) -> ChunkMeshData:
    """Build render streams for one chunk from its padded block snapshot."""
    core = padded[1:-1, 1:-1, 1:-1]
    render = registry.render[core]
    opaque_padded = registry.opaque[padded]

    solid_mask = render == RENDER_SOLID
    cutout_mask = render == RENDER_CUTOUT
    liquid_mask = render == RENDER_LIQUID

    opaque_parts: list[np.ndarray] = []
    cutout_parts: list[np.ndarray] = []
    water_parts: list[np.ndarray] = []

    # Liquid surface flag: top edge is lowered unless the same liquid sits above.
    above = _neighbour_view(padded, 0, 0, 1)
    liquid_surface = liquid_mask & (above != core)

    for face, (ox, oz, oy) in enumerate(_DIR_ARRAY_OFFSETS):
        nb = _neighbour_view(padded, ox, oz, oy)
        nb_opaque = registry.opaque[nb]
        same_as_nb = nb == core

        # Solid cubes: hidden only by opaque neighbours.
        vis = solid_mask & ~nb_opaque
        if vis.any():
            xs, zs, ys = np.nonzero(vis)
            ids = core[vis]
            opaque_parts.append(
                _pack_faces(
                    xs, zs, ys, face,
                    _corner_ao(opaque_padded, xs, zs, ys, face),
                    registry.face_layers[ids, face],
                    registry.emission[ids],
                )
            )

        # Cutout cubes (leaves/glass): also cull faces between identical blocks.
        vis = cutout_mask & ~nb_opaque & ~same_as_nb
        if vis.any():
            xs, zs, ys = np.nonzero(vis)
            ids = core[vis]
            cutout_parts.append(
                _pack_faces(
                    xs, zs, ys, face,
                    _corner_ao(opaque_padded, xs, zs, ys, face),
                    registry.face_layers[ids, face],
                    registry.emission[ids],
                )
            )

        # Liquids: no AO (avoids dark seams on the water surface).
        vis = liquid_mask & ~nb_opaque & ~same_as_nb
        if vis.any():
            xs, zs, ys = np.nonzero(vis)
            ids = core[vis]
            n = len(xs)
            water_parts.append(
                _pack_faces(
                    xs, zs, ys, face,
                    np.full((n, 4), 3, dtype=_U32),
                    registry.face_layers[ids, face],
                    registry.emission[ids],
                    flags=liquid_surface[vis].astype(_U32),
                )
            )

    # Cross-shaped plants: two diagonal quads, no culling, drawn in the
    # cutout pass (which renders with backface culling disabled).
    cross = render == RENDER_CROSS
    if cross.any():
        xs, zs, ys = np.nonzero(cross)
        ids = core[cross]
        n = len(xs)
        ao = np.full((n, 4), 3, dtype=_U32)
        for face in (6, 7):
            cutout_parts.append(
                _pack_faces(
                    xs, zs, ys, face, ao,
                    registry.face_layers[ids, 0],
                    registry.emission[ids],
                )
            )

    def _concat(parts: list[np.ndarray]) -> np.ndarray | None:
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else np.concatenate(parts)

    return ChunkMeshData(
        opaque=_concat(opaque_parts),
        cutout=_concat(cutout_parts),
        water=_concat(water_parts),
    )
