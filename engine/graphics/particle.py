"""Billboarded particle system (plan phase 9.5).

Particles live in a fixed-size structured NumPy array updated in bulk (no
per-particle Python loop) and drawn as camera-facing instanced quads.  The
game emits bursts for block breaks, explosions and splashes; everything
renders into the HDR scene target so sparks bloom.
"""

from __future__ import annotations

import moderngl
import numpy as np

from engine.graphics import shaders

_DTYPE = np.dtype([
    ("pos", "f4", 3), ("vel", "f4", 3), ("color", "f4", 4),
    ("life", "f4"), ("max_life", "f4"), ("size", "f4"), ("gravity", "f4"),
])

_QUAD = np.array(
    [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)],
    dtype="f4",
)


class ParticleSystem:
    def __init__(self, ctx: moderngl.Context, max_particles: int = 4096) -> None:
        self.ctx = ctx
        self.max = max_particles
        self.p = np.zeros(max_particles, dtype=_DTYPE)
        self.count = 0

        self.prog = ctx.program(shaders.PARTICLE_VERT, shaders.PARTICLE_FRAG)
        self._quad_vbo = ctx.buffer(_QUAD.tobytes())
        # Instance buffer holds pos(3) + color(4) + size(1) = 8 floats each.
        self._inst = ctx.buffer(reserve=max_particles * 8 * 4, dynamic=True)
        self._vao = ctx.vertex_array(self.prog, [
            (self._quad_vbo, "2f4", "in_corner"),
            (self._inst, "3f4 4f4 1f4/i", "in_pos", "in_color", "in_size"),
        ])

    def emit(self, position, color, count=16, speed=2.5, spread=1.0,
             lifetime=0.7, size=0.12, gravity=1.0) -> None:
        n = min(count, self.max - self.count)
        if n <= 0:
            return
        rng = np.random.default_rng()
        sl = slice(self.count, self.count + n)
        self.p["pos"][sl] = position
        dirs = rng.normal(0.0, 1.0, (n, 3))
        dirs[:, 1] = np.abs(dirs[:, 1]) * 0.6 + 0.4  # bias upward
        self.p["vel"][sl] = dirs * speed * spread
        col = np.array(color, dtype="f4")
        self.p["color"][sl] = col
        life = lifetime * rng.uniform(0.6, 1.0, n)
        self.p["life"][sl] = life
        self.p["max_life"][sl] = life
        self.p["size"][sl] = size * rng.uniform(0.6, 1.3, n)
        self.p["gravity"][sl] = gravity
        self.count += n

    def update(self, dt: float) -> None:
        if self.count == 0:
            return
        a = self.p[:self.count]
        a["pos"] += a["vel"] * dt
        a["vel"][:, 1] -= 18.0 * a["gravity"] * dt
        a["vel"] *= (1.0 - min(dt * 2.0, 0.9))  # drag
        a["life"] -= dt
        # Compact: keep only living particles.
        alive = a["life"] > 0.0
        n = int(alive.sum())
        if n != self.count:
            self.p[:n] = a[alive]
            self.count = n

    def render(self, view_proj: np.ndarray, cam_right, cam_up) -> None:
        if self.count == 0:
            return
        a = self.p[:self.count]
        inst = np.empty((self.count, 8), dtype="f4")
        inst[:, 0:3] = a["pos"]
        inst[:, 3:7] = a["color"]
        inst[:, 6] = a["life"] / np.maximum(a["max_life"], 1e-4)  # fade alpha
        inst[:, 7] = a["size"]
        self._inst.write(inst.tobytes())

        ctx = self.ctx
        ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        ctx.disable(moderngl.CULL_FACE)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        try:
            ctx.fbo.depth_mask = False
        except AttributeError:
            pass
        self.prog["u_view_proj"].write(view_proj.T.astype("f4").tobytes())
        self.prog["u_camera_right"].value = tuple(float(v) for v in cam_right)
        self.prog["u_camera_up"].value = tuple(float(v) for v in cam_up)
        self._vao.render(moderngl.TRIANGLES, vertices=6, instances=self.count)
        try:
            ctx.fbo.depth_mask = True
        except AttributeError:
            pass
        ctx.disable(moderngl.BLEND)
