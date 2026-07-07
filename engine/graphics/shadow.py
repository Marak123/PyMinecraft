"""Cascaded shadow maps for the sun (plan phase 1).

Three cascades fitted as spheres around camera-frustum slices — the sphere
fit plus texel snapping keeps shadow edges stable while the camera moves
(the classic CSM shimmer fix).  The shadow pass reuses the chunks' vertex
buffers through dedicated depth-only VAOs.
"""

from __future__ import annotations

import numpy as np

from engine.world.coords import CHUNK_Y

# Cascade far distances in blocks (near sharp -> far coarse). The far split
# is kept moderate so the third cascade doesn't redraw the whole loaded world.
CASCADE_SPLITS = (16.0, 44.0, 96.0)


def cascade_matrices(
    camera_pos: np.ndarray,
    camera_forward: np.ndarray,
    sun_dir: np.ndarray,
    shadow_resolution: int,
) -> list[np.ndarray]:
    """Orthographic light view-projection matrix per cascade (row-major)."""
    matrices = []
    up = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(sun_dir, up))) > 0.95:
        up = np.array([0.0, 0.0, 1.0])

    near = 0.0
    for far in CASCADE_SPLITS:
        # Sphere around the frustum slice: centre on the view ray.
        centre = camera_pos + camera_forward * ((near + far) * 0.5)
        radius = far * 0.75 + 4.0

        # Texel snapping: move the centre in whole shadow-texel steps only.
        texel = (radius * 2.0) / shadow_resolution
        right = np.cross(sun_dir, up)
        right /= np.linalg.norm(right)
        true_up = np.cross(right, sun_dir)
        cx = np.floor(np.dot(centre, right) / texel) * texel
        cy = np.floor(np.dot(centre, true_up) / texel) * texel
        cz = np.dot(centre, sun_dir)
        centre = right * cx + true_up * cy + sun_dir * cz

        eye = centre + sun_dir * (radius + CHUNK_Y)
        view = _look_at(eye, centre, up)
        depth_range = radius * 2.0 + CHUNK_Y * 2.0
        proj = _ortho(-radius, radius, -radius, radius, 0.1, depth_range)
        matrices.append(proj @ view)
        near = far
    return matrices


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, fwd)
    m = np.identity(4)
    m[0, :3] = right
    m[1, :3] = true_up
    m[2, :3] = -fwd
    m[0, 3] = -np.dot(right, eye)
    m[1, 3] = -np.dot(true_up, eye)
    m[2, 3] = np.dot(fwd, eye)
    return m


def _ortho(l: float, r: float, b: float, t: float, n: float, f: float) -> np.ndarray:
    m = np.identity(4)
    m[0, 0] = 2.0 / (r - l)
    m[1, 1] = 2.0 / (t - b)
    m[2, 2] = -2.0 / (f - n)
    m[0, 3] = -(r + l) / (r - l)
    m[1, 3] = -(t + b) / (t - b)
    m[2, 3] = -(f + n) / (f - n)
    return m
