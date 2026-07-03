"""Box-model renderer for entities (mobs, third-person player).

Entities are stacks of tinted boxes (head/body/limbs).  One shared unit-cube
VBO; each part sets a model matrix, tint and light level.  A dozen entities
at ~6 parts each is a trivial number of draw calls.
"""

from __future__ import annotations

import math

import moderngl
import numpy as np

_VERT = """#version 330
uniform mat4 u_view_proj;
uniform mat4 u_model;

in vec3 in_pos;
in vec3 in_normal;

out vec3 v_normal;

void main() {
    v_normal = mat3(u_model) * in_normal;
    gl_Position = u_view_proj * u_model * vec4(in_pos, 1.0);
}
"""

_FRAG = """#version 330
uniform vec3 u_tint;
uniform float u_light;

in vec3 v_normal;
out vec4 f_color;

void main() {
    vec3 n = normalize(v_normal);
    // Simple voxel-style face shading by dominant axis.
    float shade = 0.8;
    if (abs(n.y) > 0.7) shade = n.y > 0.0 ? 1.0 : 0.55;
    else if (abs(n.x) > 0.7) shade = 0.68;
    f_color = vec4(u_tint * u_light * shade, 1.0);
}
"""


def _unit_cube() -> np.ndarray:
    """36 vertices of a unit cube centred at origin, with normals."""
    faces = [
        ((1, 0, 0), [(0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5), (0.5, -0.5, 0.5)]),
        ((-1, 0, 0), [(-0.5, -0.5, 0.5), (-0.5, 0.5, 0.5), (-0.5, 0.5, -0.5), (-0.5, -0.5, -0.5)]),
        ((0, 1, 0), [(-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, -0.5)]),
        ((0, -1, 0), [(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (-0.5, -0.5, 0.5)]),
        ((0, 0, 1), [(0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5), (-0.5, -0.5, 0.5)]),
        ((0, 0, -1), [(-0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (0.5, 0.5, -0.5), (0.5, -0.5, -0.5)]),
    ]
    verts = []
    for normal, corners in faces:
        for i in (0, 1, 2, 0, 2, 3):
            verts.append((*corners[i], *normal))
    return np.array(verts, dtype=np.float32)


def model_matrix(
    position: np.ndarray,
    yaw_deg: float,
    offset: tuple[float, float, float],
    size: tuple[float, float, float],
    pitch_deg: float = 0.0,
) -> np.ndarray:
    """translate(position) @ rotY(yaw) @ translate(offset) @ rotX(pitch) @ scale.

    ``offset`` places the part in entity-local space (rotates with the body);
    ``pitch`` swings limbs around the part's own origin.
    """
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)

    rot_y = np.array([[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]], dtype=np.float64)
    rot_x = np.array([[1, 0, 0, 0], [0, cp, -sp, 0], [0, sp, cp, 0], [0, 0, 0, 1]], dtype=np.float64)
    scale = np.diag([size[0], size[1], size[2], 1.0])
    t_off = np.identity(4)
    t_off[:3, 3] = offset
    t_pos = np.identity(4)
    t_pos[:3, 3] = position
    return t_pos @ rot_y @ t_off @ rot_x @ scale


class BoxRenderer:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self.prog = ctx.program(_VERT, _FRAG)
        cube = _unit_cube()
        self._vbo = ctx.buffer(cube.tobytes())
        self._vao = ctx.vertex_array(
            self.prog, [(self._vbo, "3f4 3f4", "in_pos", "in_normal")]
        )

    def begin(self, view_proj: np.ndarray) -> None:
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        try:
            self.prog["u_view_proj"].write(view_proj.T.astype("f4").tobytes())
        except KeyError:
            pass

    def draw_box(self, model: np.ndarray, tint: tuple[float, float, float], light: float) -> None:
        self.prog["u_model"].write(model.T.astype("f4").tobytes())
        self.prog["u_tint"].value = tint
        self.prog["u_light"].value = light
        self._vao.render(moderngl.TRIANGLES)
