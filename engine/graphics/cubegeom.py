"""Single source of truth for cube/cross face geometry.

Both the CPU mesher and the GLSL vertex shader consume these tables (the
shader constants are *generated* from them), so winding, UVs and AO sampling
can never drift apart.

Face order everywhere: 0:+X 1:-X 2:+Y 3:-Y 4:+Z 5:-Z, plus two extra
"faces" 6/7 for the diagonal quads of cross-shaped plants.

Vertex format (2x uint32 per vertex, 8 bytes):
    word0: x:6 | y:8 | z:6 | corner:2 | ao:2 | face:3 | flag:1
    word1: texture_layer:16 | emission:4
``flag`` marks liquid surface vertices whose top edge is lowered in the
water shader (Minecraft-style non-full water surface).
"""

from __future__ import annotations

import numpy as np

# Outward normals in world (x, y, z). Faces 6/7 (cross quads) use "up" for
# lighting so plants are lit like the ground they stand on.
FACE_NORMALS = np.array(
    [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
        (0, 1, 0), (0, 1, 0),
    ],
    dtype=np.int32,
)

# Four corner offsets per face, CCW when viewed from outside the block.
FACE_CORNERS = np.array(
    [
        [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],  # +X
        [(0, 0, 1), (0, 1, 1), (0, 1, 0), (0, 0, 0)],  # -X
        [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)],  # +Y
        [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],  # -Y
        [(1, 0, 1), (1, 1, 1), (0, 1, 1), (0, 0, 1)],  # +Z
        [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],  # -Z
        [(0, 0, 0), (0, 1, 0), (1, 1, 1), (1, 0, 1)],  # cross A
        [(1, 0, 0), (1, 1, 0), (0, 1, 1), (0, 0, 1)],  # cross B
    ],
    dtype=np.int32,
)

# Texture coordinates per corner. v grows downward in the tile image, so for
# side-like faces v = 1 - y (tile top row appears at the top of the face).
_SIDE_UV = [(0, 1), (0, 0), (1, 0), (1, 1)]
FACE_UVS = np.array(
    [
        _SIDE_UV,                                  # +X
        _SIDE_UV,                                  # -X
        [(0, 0), (0, 1), (1, 1), (1, 0)],          # +Y (u=x, v=z)
        [(0, 0), (1, 0), (1, 1), (0, 1)],          # -Y
        _SIDE_UV,                                  # +Z
        _SIDE_UV,                                  # -Z
        _SIDE_UV,                                  # cross A
        _SIDE_UV,                                  # cross B
    ],
    dtype=np.float32,
)

# Directional shading factors (classic voxel look: tops bright, bottoms dark).
FACE_SHADE = np.array([0.62, 0.62, 1.0, 0.5, 0.8, 0.8, 0.9, 0.9], dtype=np.float32)


def _build_ao_offsets() -> np.ndarray:
    """AO sample offsets for the 6 cube faces.

    For each (face, corner) the three occluders are the two edge neighbours
    and the diagonal neighbour in the block layer the face looks into.
    Returns int offsets in world (x, y, z), shape (6, 4, 3, 3).
    """
    out = np.zeros((6, 4, 3, 3), dtype=np.int32)
    for face in range(6):
        normal = FACE_NORMALS[face]
        n_axis = int(np.argmax(np.abs(normal)))
        plane_axes = [a for a in range(3) if a != n_axis]
        for corner in range(4):
            c = FACE_CORNERS[face, corner]
            side1 = normal.copy()
            side2 = normal.copy()
            diag = normal.copy()
            a1, a2 = plane_axes
            s1 = 1 if c[a1] == 1 else -1
            s2 = 1 if c[a2] == 1 else -1
            side1[a1] += s1
            side2[a2] += s2
            diag[a1] += s1
            diag[a2] += s2
            out[face, corner, 0] = side1
            out[face, corner, 1] = side2
            out[face, corner, 2] = diag
    return out


AO_OFFSETS = _build_ao_offsets()


def glsl_geometry_tables() -> str:
    """GLSL constant arrays generated from the tables above."""
    corners = ", ".join(
        f"vec3({c[0]}.0, {c[1]}.0, {c[2]}.0)"
        for face in FACE_CORNERS
        for c in face
    )
    uvs = ", ".join(
        f"vec2({u:.1f}, {v:.1f})" for face in FACE_UVS for (u, v) in face
    )
    normals = ", ".join(f"vec3({n[0]}.0, {n[1]}.0, {n[2]}.0)" for n in FACE_NORMALS)
    shade = ", ".join(f"{s:.3f}" for s in FACE_SHADE)
    return f"""
const vec3 FACE_CORNERS[32] = vec3[32]({corners});
const vec2 FACE_UVS[32] = vec2[32]({uvs});
const vec3 FACE_NORMALS[8] = vec3[8]({normals});
const float FACE_SHADE[8] = float[8]({shade});
"""
