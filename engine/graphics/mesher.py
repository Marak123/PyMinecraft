"""Vectorised chunk mesher.

Face-culling mesher with per-vertex ambient occlusion and *smooth lighting*,
fully vectorised in NumPy — no per-block Python loops.  Emits the compressed
8-byte vertex format described in :mod:`engine.graphics.cubegeom`.

Smooth lighting samples the same four cells per face corner that AO uses
(front, two edges, diagonal) and averages their light, which produces the
soft light gradients and voxel shadows the renderer displays.

Runs on worker threads: it only reads immutable registry tables and the
padded snapshots it is given.

(Greedy meshing is a known future optimisation; it conflicts with per-vertex
AO/light and needs benchmarking first — see docs/ROADMAP.md.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine.graphics.cubegeom import AO_OFFSETS, FACE_NORMALS
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
    y_min: int = 0
    y_max: int = CHUNK_Y

    @property
    def vertex_count(self) -> int:
        return sum(len(a) // 2 for a in (self.opaque, self.cutout, self.water) if a is not None)


def _neighbour_view(padded: np.ndarray, ox: int, oz: int, oy: int) -> np.ndarray:
    # Core Y-band is the padded height minus the 2-block halo.
    yb = padded.shape[2] - 2
    return padded[
        1 + ox : 1 + CHUNK_X + ox,
        1 + oz : 1 + CHUNK_Z + oz,
        1 + oy : 1 + yb + oy,
    ]


def _pack_faces(
    xs: np.ndarray,
    zs: np.ndarray,
    ys: np.ndarray,
    face: int,
    ao: np.ndarray,          # (n, 4) uint32 values 0..3
    sky: np.ndarray,         # (n, 4) uint32 values 0..15
    blk: np.ndarray,         # (n, 4) uint32 values 0..15
    tex_layers: np.ndarray,  # (n,)
    emission: np.ndarray,    # (n,)
    flags: np.ndarray | None = None,  # (n,) 0/1
    y_offset: int = 0,       # band-local ys -> world ys
) -> np.ndarray:
    n = len(xs)
    base0 = (
        xs.astype(_U32)
        | ((ys.astype(_U32) + _U32(y_offset)) << _U32(6))
        | (zs.astype(_U32) << _U32(14))
        | (_U32(face) << _U32(24))
    )
    if flags is not None:
        base0 |= flags.astype(_U32) << _U32(27)

    corner_ids = np.arange(4, dtype=_U32)
    words0 = base0[:, None] | (corner_ids[None, :] << _U32(20)) | (ao << _U32(22))

    base1 = tex_layers.astype(_U32) | (emission.astype(_U32) << _U32(16))
    words1 = base1[:, None] | (sky << _U32(20)) | (blk << _U32(24))

    # Flip triangulation so the quad diagonal runs through the brighter pair
    # of corners — removes the classic AO "bowtie" artifact.
    flip = (ao[:, 0] + ao[:, 2]) >= (ao[:, 1] + ao[:, 3])
    pattern = np.where(flip[:, None], _PAT_A[None, :], _PAT_B[None, :])

    out = np.empty((n, 6, 2), dtype=_U32)
    out[:, :, 0] = np.take_along_axis(words0, pattern, axis=1)
    out[:, :, 1] = np.take_along_axis(words1, pattern, axis=1)
    return out.reshape(-1)


def _corner_shading(
    opaque_padded: np.ndarray,
    sky_padded: np.ndarray,
    blk_padded: np.ndarray,
    xs: np.ndarray,
    zs: np.ndarray,
    ys: np.ndarray,
    face: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-corner (ao, sky, block) — AO in 0..3, light in 0..15.

    Light is the average of the four cells touching the corner in the layer
    the face looks into; opaque cells hold light 0, so the average naturally
    darkens corners against walls (soft voxel shadows).
    """
    n = len(xs)
    ao = np.empty((n, 4), dtype=_U32)
    sky = np.empty((n, 4), dtype=_U32)
    blk = np.empty((n, 4), dtype=_U32)

    # The cell directly in front of the face is shared by all four corners.
    fx, fy, fz = FACE_NORMALS[face]
    front = (xs + 1 + fx, zs + 1 + fz, ys + 1 + fy)
    front_sky = sky_padded[front].astype(np.uint16)
    front_blk = blk_padded[front].astype(np.uint16)

    for corner in range(4):
        offs = AO_OFFSETS[face, corner]  # (3 samples, 3 comps) in world (x,y,z)
        idx = [
            (xs + 1 + offs[k, 0], zs + 1 + offs[k, 2], ys + 1 + offs[k, 1])
            for k in range(3)
        ]
        s1, s2, diag = (opaque_padded[i] for i in idx)
        occ = s1.astype(np.int32) + s2 + diag
        val = 3 - occ
        # Both edge neighbours solid -> corner fully occluded regardless.
        val[s1 & s2] = 0
        ao[:, corner] = val.astype(_U32)

        sky_sum = front_sky + sky_padded[idx[0]] + sky_padded[idx[1]] + sky_padded[idx[2]]
        blk_sum = front_blk + blk_padded[idx[0]] + blk_padded[idx[1]] + blk_padded[idx[2]]
        sky[:, corner] = (sky_sum // 4).astype(_U32)
        blk[:, corner] = (blk_sum // 4).astype(_U32)
    return ao, sky, blk


def build_chunk_meshes(
    mesh_input: tuple[np.ndarray, np.ndarray, np.ndarray], registry: BlockRegistry
) -> ChunkMeshData:
    """Build render streams for one chunk from its padded snapshots.

    Only the vertical band that actually holds renderable blocks is meshed —
    on the 256-tall world most columns are air, so clipping to the active
    band (plan 4.2 sub-chunk optimisation) skips the empty sky entirely.
    """
    full_padded, full_sky, full_blk = mesh_input
    full_core = full_padded[1:-1, 1:-1, 1:-1]
    renderable = registry.render[full_core] != 0
    occupied = np.nonzero(renderable.any(axis=(0, 1)))[0]
    if len(occupied) == 0:
        return ChunkMeshData(None, None, None, 0, 1)
    y0 = int(occupied[0])
    y1 = int(occupied[-1]) + 1
    # Slice with a 1-block halo (padded coords are core+1) for neighbour views.
    padded = full_padded[:, :, y0 : y1 + 2]
    sky_padded = full_sky[:, :, y0 : y1 + 2]
    blk_padded = full_blk[:, :, y0 : y1 + 2]
    y_band = y1 - y0

    core = padded[1:-1, 1:-1, 1:-1]
    render = registry.render[core]
    opaque_padded = registry.opaque[padded]

    solid_mask = render == RENDER_SOLID
    cutout_mask = render == RENDER_CUTOUT
    liquid_mask = render == RENDER_LIQUID

    opaque_parts: list[np.ndarray] = []
    cutout_parts: list[np.ndarray] = []
    water_parts: list[np.ndarray] = []
    y_lo, y_hi = CHUNK_Y, 0

    def _track_y(ys: np.ndarray) -> None:
        # ys are band-local; report world coords for tight render bounds.
        nonlocal y_lo, y_hi
        y_lo = min(y_lo, int(ys.min()) + y0)
        y_hi = max(y_hi, int(ys.max()) + 1 + y0)

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
            _track_y(ys)
            ids = core[vis]
            ao, sky, blk = _corner_shading(
                opaque_padded, sky_padded, blk_padded, xs, zs, ys, face
            )
            opaque_parts.append(
                _pack_faces(
                    xs, zs, ys, face, ao, sky, blk,
                    registry.face_layers[ids, face],
                    registry.emission[ids], y_offset=y0,
                )
            )

        # Cutout cubes (leaves/glass): also cull faces between identical blocks.
        vis = cutout_mask & ~nb_opaque & ~same_as_nb
        if vis.any():
            xs, zs, ys = np.nonzero(vis)
            _track_y(ys)
            ids = core[vis]
            ao, sky, blk = _corner_shading(
                opaque_padded, sky_padded, blk_padded, xs, zs, ys, face
            )
            cutout_parts.append(
                _pack_faces(
                    xs, zs, ys, face, ao, sky, blk,
                    registry.face_layers[ids, face],
                    registry.emission[ids], y_offset=y0,
                )
            )

        # Liquids: flat light from the cell itself, no AO (avoids dark seams).
        vis = liquid_mask & ~nb_opaque & ~same_as_nb
        if vis.any():
            xs, zs, ys = np.nonzero(vis)
            _track_y(ys)
            ids = core[vis]
            n = len(xs)
            own = (xs + 1, zs + 1, ys + 1)
            sky = np.broadcast_to(
                sky_padded[own].astype(_U32)[:, None], (n, 4)
            )
            blk = np.broadcast_to(
                blk_padded[own].astype(_U32)[:, None], (n, 4)
            )
            water_parts.append(
                _pack_faces(
                    xs, zs, ys, face,
                    np.full((n, 4), 3, dtype=_U32), sky, blk,
                    registry.face_layers[ids, face],
                    registry.emission[ids],
                    flags=liquid_surface[vis].astype(_U32), y_offset=y0,
                )
            )

    # Cross-shaped plants: two diagonal quads, no culling, drawn in the
    # cutout pass (which renders with backface culling disabled).
    cross = render == RENDER_CROSS
    if cross.any():
        xs, zs, ys = np.nonzero(cross)
        _track_y(ys)
        ids = core[cross]
        n = len(xs)
        own = (xs + 1, zs + 1, ys + 1)
        ao = np.full((n, 4), 3, dtype=_U32)
        sky = np.broadcast_to(sky_padded[own].astype(_U32)[:, None], (n, 4))
        blk = np.broadcast_to(blk_padded[own].astype(_U32)[:, None], (n, 4))
        for face in (6, 7):
            cutout_parts.append(
                _pack_faces(
                    xs, zs, ys, face, ao, sky, blk,
                    registry.face_layers[ids, 0],
                    registry.emission[ids], y_offset=y0,
                )
            )

    def _concat(parts: list[np.ndarray]) -> np.ndarray | None:
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else np.concatenate(parts)

    if y_hi <= y_lo:  # empty chunk mesh
        y_lo, y_hi = 0, 1
    return ChunkMeshData(
        opaque=_concat(opaque_parts),
        cutout=_concat(cutout_parts),
        water=_concat(water_parts),
        y_min=max(y_lo - 1, 0),
        y_max=min(y_hi + 1, CHUNK_Y),
    )
