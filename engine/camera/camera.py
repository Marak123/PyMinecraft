"""First-person camera producing view/projection matrices and a frustum."""

from __future__ import annotations

import numpy as np

from engine.core import mathx


class Camera:
    UP = np.array([0.0, 1.0, 0.0])

    def __init__(self, fov: float, aspect: float, near: float = 0.05, far: float = 1000.0) -> None:
        self.position = np.zeros(3, dtype=np.float64)
        self.yaw = -90.0  # degrees; 0 looks along +X
        self.pitch = 0.0
        self.fov = fov
        self.near = near
        self.far = far
        self._aspect = aspect
        self._proj = mathx.perspective(fov, aspect, near, far)

    def set_aspect(self, aspect: float) -> None:
        if aspect > 0 and abs(aspect - self._aspect) > 1e-9:
            self._aspect = aspect
            self._proj = mathx.perspective(self.fov, aspect, self.near, self.far)

    def set_fov(self, fov: float) -> None:
        """Runtime FOV change (sprint kick). Cheap: one matrix rebuild."""
        if abs(fov - self.fov) > 1e-3:
            self.fov = fov
            self._proj = mathx.perspective(fov, self._aspect, self.near, self.far)

    def rotate(self, d_yaw: float, d_pitch: float) -> None:
        self.yaw = (self.yaw + d_yaw) % 360.0
        # Hard pitch clamp: looking exactly straight up/down degenerates look_at.
        self.pitch = float(np.clip(self.pitch + d_pitch, -89.5, 89.5))

    @property
    def forward(self) -> np.ndarray:
        return mathx.direction_from_angles(self.yaw, self.pitch)

    @property
    def flat_forward(self) -> np.ndarray:
        """Forward projected onto the XZ plane — used for walking movement."""
        f = mathx.direction_from_angles(self.yaw, 0.0)
        f[1] = 0.0
        norm = np.linalg.norm(f)
        return f / norm if norm > 0 else f

    @property
    def right(self) -> np.ndarray:
        f = self.forward
        r = np.cross(f, self.UP)
        return r / np.linalg.norm(r)

    def view_matrix(self) -> np.ndarray:
        return mathx.look_at(self.position, self.position + self.forward, self.UP)

    def view_proj(self) -> np.ndarray:
        return self._proj @ self.view_matrix()

    def frustum_planes(self) -> np.ndarray:
        return mathx.extract_frustum_planes(self.view_proj())
