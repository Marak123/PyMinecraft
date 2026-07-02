"""Minimal 3D math helpers built on NumPy.

We deliberately avoid an external math dependency (PyGLM etc.): the engine
only needs a handful of matrix builders and a frustum extractor, and keeping
them here means one fewer native wheel to break on user machines.

Convention: matrices are row-major and combined as ``proj @ view``.  OpenGL
expects column-major storage, so upload with ``m.T.astype('f4').tobytes()``
(the renderer centralises this).
"""

from __future__ import annotations

import math

import numpy as np


def perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Right-handed perspective projection matrix (OpenGL clip space)."""
    f = 1.0 / math.tan(math.radians(fov_y_deg) * 0.5)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / max(aspect, 1e-6)
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Right-handed view matrix."""
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, fwd)

    m = np.identity(4, dtype=np.float64)
    m[0, :3] = right
    m[1, :3] = true_up
    m[2, :3] = -fwd
    m[0, 3] = -np.dot(right, eye)
    m[1, 3] = -np.dot(true_up, eye)
    m[2, 3] = np.dot(fwd, eye)
    return m


def ortho_2d(width: float, height: float) -> np.ndarray:
    """Pixel-space orthographic projection with the origin at the TOP-left.

    Y grows downwards, which matches how UI layouts are usually reasoned about.
    """
    m = np.identity(4, dtype=np.float64)
    m[0, 0] = 2.0 / width
    m[1, 1] = -2.0 / height
    m[0, 3] = -1.0
    m[1, 3] = 1.0
    return m


def direction_from_angles(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Unit forward vector for yaw/pitch in degrees (yaw 0 looks along +X)."""
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    cp = math.cos(pitch)
    return np.array(
        [cp * math.cos(yaw), math.sin(pitch), cp * math.sin(yaw)], dtype=np.float64
    )


def extract_frustum_planes(view_proj: np.ndarray) -> np.ndarray:
    """Gribb-Hartmann frustum plane extraction.

    Returns a (6, 4) array of normalised planes (nx, ny, nz, d) such that a
    point p is inside when ``dot(n, p) + d >= 0`` for all planes.
    """
    m = view_proj
    planes = np.array(
        [
            m[3] + m[0],  # left
            m[3] - m[0],  # right
            m[3] + m[1],  # bottom
            m[3] - m[1],  # top
            m[3] + m[2],  # near
            m[3] - m[2],  # far
        ],
        dtype=np.float64,
    )
    lengths = np.linalg.norm(planes[:, :3], axis=1, keepdims=True)
    lengths[lengths == 0.0] = 1.0
    return planes / lengths


def aabbs_in_frustum(
    planes: np.ndarray, centers: np.ndarray, half_extents: np.ndarray
) -> np.ndarray:
    """Vectorised AABB-vs-frustum test.

    ``centers`` is (N, 3), ``half_extents`` is (3,) or (N, 3).
    Returns a boolean mask of AABBs at least partially inside the frustum.
    Uses the p-vertex trick: an AABB is outside iff its most positive vertex
    along the plane normal is still behind the plane.
    """
    normals = planes[:, :3]  # (6, 3)
    d = planes[:, 3]  # (6,)
    # Distance from each center to each plane: (N, 6)
    dist = centers @ normals.T + d
    # Projected radius of the box onto each plane normal: (N, 6)
    radius = np.abs(half_extents) @ np.abs(normals).T
    return np.all(dist + radius >= 0.0, axis=1)
