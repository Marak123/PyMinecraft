"""GLSL shader sources.

The chunk/water vertex shaders decode the compressed 8-byte vertex format
(see cubegeom.py) and read geometry from constant tables *generated* from
the same Python data the mesher uses — one source of truth.

Terrain lighting is computed per-vertex (faces are flat, so per-fragment
would buy nothing); AO interpolates across the face for smooth corners.
"""

from __future__ import annotations

from engine.graphics.cubegeom import glsl_geometry_tables

_TABLES = glsl_geometry_tables()

_VERTEX_DECODE = """
    uint w0 = in_data.x;
    uint w1 = in_data.y;
    float bx = float(w0 & 63u);
    float by = float((w0 >> 6u) & 255u);
    float bz = float((w0 >> 14u) & 63u);
    uint corner = (w0 >> 20u) & 3u;
    uint ao_bits = (w0 >> 22u) & 3u;
    uint face = (w0 >> 24u) & 7u;
    uint flag = (w0 >> 27u) & 1u;
    uint idx = face * 4u + corner;
    vec3 corner_off = FACE_CORNERS[idx];
    uint tex_layer = w1 & 65535u;
    float emission = float((w1 >> 16u) & 15u) / 15.0;
"""

_LIGHT_CALC = """
    float diff = max(dot(FACE_NORMALS[face], u_sun_dir), 0.0);
    float sky = u_daylight * (0.35 + 0.65 * diff);
    float light = clamp(0.16 + sky, 0.0, 1.0) * FACE_SHADE[face];
    v_light = max(light, emission);
"""

CHUNK_VERT = f"""#version 330
uniform mat4 u_view_proj;
uniform vec3 u_chunk_origin;
uniform vec3 u_sun_dir;
uniform float u_daylight;
uniform float u_time;

in uvec2 in_data;

out vec3 v_world;
out vec2 v_uv;
out float v_ao;
out float v_light;
flat out uint v_layer;

{_TABLES}

void main() {{
{_VERTEX_DECODE}
    vec3 local = vec3(bx, by, bz) + corner_off;
    vec3 world = u_chunk_origin + local;
    // Cross-plant sway: only the top vertices move, roots stay planted.
    if (face >= 6u && corner_off.y > 0.5) {{
        world.x += sin(u_time * 1.8 + world.x * 0.9 + world.z * 1.3) * 0.05;
        world.z += cos(u_time * 1.6 + world.z * 1.1) * 0.05;
    }}
    v_world = world;
    v_uv = FACE_UVS[idx];
    v_layer = tex_layer;
    v_ao = float(ao_bits) / 3.0;
{_LIGHT_CALC}
    gl_Position = u_view_proj * vec4(world, 1.0);
}}
"""

CHUNK_FRAG = """#version 330
uniform sampler2DArray u_tiles;
uniform vec3 u_fog_color;
uniform vec2 u_fog_range;
uniform vec3 u_camera_pos;
uniform bool u_alpha_test;

in vec3 v_world;
in vec2 v_uv;
in float v_ao;
in float v_light;
flat in uint v_layer;

out vec4 f_color;

void main() {
    vec4 tex = texture(u_tiles, vec3(v_uv, float(v_layer)));
    if (u_alpha_test && tex.a < 0.5) discard;
    float ao = mix(0.52, 1.0, v_ao);
    vec3 color = tex.rgb * (v_light * ao);
    float dist = length(v_world - u_camera_pos);
    float fog = smoothstep(u_fog_range.x, u_fog_range.y, dist);
    f_color = vec4(mix(color, u_fog_color, fog), 1.0);
}
"""

WATER_VERT = f"""#version 330
uniform mat4 u_view_proj;
uniform vec3 u_chunk_origin;
uniform vec3 u_sun_dir;
uniform float u_daylight;
uniform float u_time;

in uvec2 in_data;

out vec3 v_world;
out vec2 v_uv;
out float v_light;
flat out uint v_layer;

{_TABLES}

void main() {{
{_VERTEX_DECODE}
    vec3 local = vec3(bx, by, bz) + corner_off;
    vec3 world = u_chunk_origin + local;
    // Liquid surface sits slightly below the block top and gently waves.
    if (flag == 1u && corner_off.y > 0.5) {{
        world.y -= 0.115;
        world.y += sin(u_time * 1.6 + world.x * 0.7 + world.z * 0.9) * 0.04;
    }}
    v_world = world;
    v_uv = FACE_UVS[idx];
    v_layer = tex_layer;
{_LIGHT_CALC}
    gl_Position = u_view_proj * vec4(world, 1.0);
}}
"""

WATER_FRAG = """#version 330
uniform sampler2DArray u_tiles;
uniform vec3 u_fog_color;
uniform vec2 u_fog_range;
uniform vec3 u_camera_pos;
uniform float u_alpha;

in vec3 v_world;
in vec2 v_uv;
in float v_light;
flat in uint v_layer;

out vec4 f_color;

void main() {
    vec4 tex = texture(u_tiles, vec3(v_uv, float(v_layer)));
    vec3 color = tex.rgb * v_light;
    float dist = length(v_world - u_camera_pos);
    float fog = smoothstep(u_fog_range.x, u_fog_range.y, dist);
    f_color = vec4(mix(color, u_fog_color, fog), u_alpha);
}
"""

SKY_VERT = """#version 330
out vec2 v_ndc;

void main() {
    // Single fullscreen triangle from gl_VertexID — no buffers needed.
    vec2 pos = vec2(
        gl_VertexID == 1 ? 3.0 : -1.0,
        gl_VertexID == 2 ? 3.0 : -1.0
    );
    v_ndc = pos;
    gl_Position = vec4(pos, 0.999999, 1.0);
}
"""

SKY_FRAG = """#version 330
uniform mat4 u_inv_view_proj;
uniform vec3 u_sun_dir;
uniform vec3 u_zenith_color;
uniform vec3 u_horizon_color;
uniform float u_daylight;

in vec2 v_ndc;
out vec4 f_color;

float star_hash(vec3 g) {
    return fract(sin(dot(g, vec3(12.9898, 78.233, 37.719))) * 43758.5453);
}

void main() {
    vec4 p0 = u_inv_view_proj * vec4(v_ndc, -1.0, 1.0);
    vec4 p1 = u_inv_view_proj * vec4(v_ndc, 1.0, 1.0);
    vec3 dir = normalize(p1.xyz / p1.w - p0.xyz / p0.w);

    float up = clamp(dir.y, -1.0, 1.0);
    vec3 col = mix(u_horizon_color, u_zenith_color, pow(max(up, 0.0), 0.55));
    if (up < 0.0) {
        // Below the horizon: gentle haze darkening, terrain fog blends into it.
        col = mix(u_horizon_color, u_horizon_color * 0.82, min(-up * 2.0, 1.0));
    }

    float sun_dot = max(dot(dir, u_sun_dir), 0.0);
    float disc = pow(sun_dot, 1600.0) * 2.4;
    float glow = pow(sun_dot, 9.0) * 0.24;
    col += (disc + glow) * vec3(1.0, 0.88, 0.68) * max(u_daylight, 0.06);

    // Sparse hash-based starfield fades in at night.
    if (u_daylight < 0.5 && up > 0.02) {
        vec3 cell = floor(dir * 110.0);
        float star = step(0.9982, star_hash(cell));
        col += star * (0.5 - u_daylight) * 1.6;
    }
    f_color = vec4(col, 1.0);
}
"""

LINES_VERT = """#version 330
uniform mat4 u_view_proj;
uniform vec3 u_offset;

in vec3 in_pos;

void main() {
    gl_Position = u_view_proj * vec4(in_pos + u_offset, 1.0);
}
"""

LINES_FRAG = """#version 330
uniform vec4 u_color;
out vec4 f_color;

void main() {
    f_color = u_color;
}
"""

UI_COLOR_VERT = """#version 330
uniform mat4 u_proj;

in vec2 in_pos;
in vec4 in_color;

out vec4 v_color;

void main() {
    v_color = in_color;
    gl_Position = u_proj * vec4(in_pos, 0.0, 1.0);
}
"""

UI_COLOR_FRAG = """#version 330
in vec4 v_color;
out vec4 f_color;

void main() {
    f_color = v_color;
}
"""

UI_TEXT_VERT = """#version 330
uniform mat4 u_proj;

in vec2 in_pos;
in vec2 in_uv;

out vec2 v_uv;

void main() {
    v_uv = in_uv;
    gl_Position = u_proj * vec4(in_pos, 0.0, 1.0);
}
"""

UI_TEXT_FRAG = """#version 330
uniform sampler2D u_font;
uniform vec4 u_color;

in vec2 v_uv;
out vec4 f_color;

void main() {
    float alpha = texture(u_font, v_uv).r;
    f_color = vec4(u_color.rgb, u_color.a * alpha);
}
"""

UI_BLOCK_VERT = """#version 330
uniform mat4 u_proj;

in vec2 in_pos;
in vec3 in_uvl;

out vec3 v_uvl;

void main() {
    v_uvl = in_uvl;
    gl_Position = u_proj * vec4(in_pos, 0.0, 1.0);
}
"""

UI_BLOCK_FRAG = """#version 330
uniform sampler2DArray u_tiles;

in vec3 v_uvl;
out vec4 f_color;

void main() {
    vec4 tex = texture(u_tiles, v_uvl);
    if (tex.a < 0.1) discard;
    f_color = tex;
}
"""
