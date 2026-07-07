"""GLSL shader sources.

The chunk/water vertex shaders decode the compressed 8-byte vertex format
(see cubegeom.py) and read geometry from constant tables *generated* from
the same Python data the mesher uses — one source of truth.

Lighting model: per-vertex smooth sky/block light baked by the mesher,
combined per-fragment as ``max(sky * daylight, warm * block)`` with a gamma
curve — bright days, warm torch pools at night, genuinely dark caves.
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
    float sky_raw = float((w1 >> 20u) & 15u) / 15.0;
    float blk_raw = float((w1 >> 24u) & 15u) / 15.0;
"""

# Directional shade: classic voxel face shading; the sun-facing term is
# passed separately so the fragment shader can attenuate it by shadow maps.
_SHADE_CALC = """
    v_shade = FACE_SHADE[face];
    v_sun = max(dot(FACE_NORMALS[face], u_sun_dir), 0.0) * u_daylight;
    v_sky = sky_raw;
    v_blk = blk_raw;
    v_emission = emission;
"""

# Shared fragment light combine (expects v_sky, v_blk, v_emission inputs,
# u_daylight uniform, plus `shadow` and `direct_sun` already computed).
# Direct sun is 40% of sky light and gets shadowed; the rest is ambient.
_LIGHT_COMBINE = """
    float sky_scale = mix(0.06, 1.0, u_daylight);
    float direct = direct_sun * shadow;
    float sl = pow(v_sky, 1.5) * sky_scale * (0.60 + 0.40 * direct);
    float bl = pow(v_blk, 1.5);
    vec3 light = max(vec3(sl), vec3(1.0, 0.82, 0.58) * bl);
    light = max(light, vec3(0.032));
    light = max(light, vec3(v_emission));
"""

# Cascaded-shadow-map sampling: pick the cascade by view distance, project
# into light space, hardware-PCF compare. Fully skippable via u_shadows.
_SHADOW_SAMPLE = """
    float shadow = 1.0;
    float view_dist = length(v_world - u_camera_pos);
    if (u_shadows && v_sun > 0.001) {
        vec4 lp;
        float bias;
        if (view_dist < u_cascade_far.x) {
            lp = u_light_vp[0] * vec4(v_world, 1.0);
            bias = 0.0012;
        } else if (view_dist < u_cascade_far.y) {
            lp = u_light_vp[1] * vec4(v_world, 1.0);
            bias = 0.0020;
        } else if (view_dist < u_cascade_far.z) {
            lp = u_light_vp[2] * vec4(v_world, 1.0);
            bias = 0.0032;
        } else {
            lp = vec4(2.0);
            bias = 0.0;
        }
        vec3 sc = lp.xyz * 0.5 + 0.5;
        if (sc.x > 0.0 && sc.x < 1.0 && sc.y > 0.0 && sc.y < 1.0 && sc.z < 1.0) {
            sc.z -= bias;
            if (view_dist < u_cascade_far.x) {
                shadow = texture(u_shadow0, sc);
            } else if (view_dist < u_cascade_far.y) {
                shadow = texture(u_shadow1, sc);
            } else {
                shadow = texture(u_shadow2, sc);
            }
        }
    }
"""

_SHADOW_UNIFORMS = """
uniform bool u_shadows;
uniform mat4 u_light_vp[3];
uniform vec3 u_cascade_far;
uniform sampler2DShadow u_shadow0;
uniform sampler2DShadow u_shadow1;
uniform sampler2DShadow u_shadow2;
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
out vec3 v_normal;
out float v_ao;
out float v_sky;
out float v_blk;
out float v_shade;
out float v_sun;
out float v_emission;
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
    v_normal = FACE_NORMALS[face];
    v_ao = float(ao_bits) / 3.0;
{_SHADE_CALC}
    gl_Position = u_view_proj * vec4(world, 1.0);
}}
"""

CHUNK_FRAG = """#version 330
uniform sampler2DArray u_tiles;
uniform sampler2DArray u_normal_map;
uniform sampler2DArray u_mrao_map;
uniform vec3 u_fog_color;
uniform vec2 u_fog_range;
uniform vec3 u_camera_pos;
uniform vec3 u_sun_dir;
uniform float u_daylight;
uniform float u_time;
uniform bool u_alpha_test;
""" + _SHADOW_UNIFORMS + """
in vec3 v_world;
in vec2 v_uv;
in vec3 v_normal;
in float v_ao;
in float v_sky;
in float v_blk;
in float v_shade;
in float v_sun;
in float v_emission;
flat in uint v_layer;

out vec4 f_color;

void main() {
    vec3 uvl = vec3(v_uv, float(v_layer));
    vec4 tex = texture(u_tiles, uvl);
    if (u_alpha_test && tex.a < 0.5) discard;

    // Simplified voxel PBR: perturb the axis-aligned face normal with the
    // procedural normal map; MRAO drives specular + micro-occlusion.
    vec3 n = normalize(v_normal);
    vec3 t = abs(n.y) > 0.5 ? vec3(1.0, 0.0, 0.0) : normalize(cross(vec3(0.0, 1.0, 0.0), n));
    vec3 b = cross(n, t);
    vec3 nm = texture(u_normal_map, uvl).rgb * 2.0 - 1.0;
    vec3 pn = normalize(t * nm.x + b * nm.y + n * nm.z);
    vec3 mrao = texture(u_mrao_map, uvl).rgb;
    float direct_sun = max(dot(pn, u_sun_dir), 0.0) * u_daylight;
""" + _SHADOW_SAMPLE + _LIGHT_COMBINE + """
    float ao = mix(0.55, 1.0, v_ao) * mix(1.0, mrao.b, 0.7);
    vec3 color = tex.rgb * light * (v_shade * ao);

    // Sun specular: roughness/metallic from the MRAO map.
    vec3 view = normalize(u_camera_pos - v_world);
    vec3 half_vec = normalize(view + u_sun_dir);
    float spec = pow(max(dot(pn, half_vec), 0.0), mix(180.0, 10.0, mrao.g))
               * (1.0 - mrao.g) * u_daylight * shadow;
    color += spec * mix(vec3(1.0), tex.rgb, mrao.r) * (0.22 + mrao.r * 1.2);

    // Emissive blocks push past 1.0 so bloom picks them up; torches flicker.
    float flicker = 0.8 + 0.2 * sin(u_time * 8.0 + v_world.x * 7.1 + v_world.z * 5.3);
    color *= 1.0 + v_emission * 1.6 * flicker;
    float fog = smoothstep(u_fog_range.x, u_fog_range.y, view_dist);
    f_color = vec4(mix(color, u_fog_color, fog), 1.0);
}
"""

BLOOM_FRAG = """#version 330
uniform sampler2D u_src;
uniform bool u_extract;

in vec2 v_uv;
out vec4 f_color;

void main() {
    vec3 c = texture(u_src, v_uv).rgb;
    if (u_extract) {
        // Keep only HDR overshoot (emissive blocks, sun, speculars).
        float l = max(max(c.r, c.g), c.b);
        float w = max(l - 1.05, 0.0) / max(l, 1e-4);
        c *= w;
    }
    f_color = vec4(c, 1.0);
}
"""

SHADOW_VERT = f"""#version 330
uniform mat4 u_light_vp;
uniform vec3 u_chunk_origin;

in uvec2 in_data;

out vec2 v_uv;
flat out uint v_layer;

{_TABLES}

void main() {{
{_VERTEX_DECODE}
    vec3 world = u_chunk_origin + vec3(bx, by, bz) + corner_off;
    v_uv = FACE_UVS[idx];
    v_layer = tex_layer;
    gl_Position = u_light_vp * vec4(world, 1.0);
}}
"""

SHADOW_FRAG = """#version 330
uniform sampler2DArray u_tiles;
uniform bool u_alpha_test;

in vec2 v_uv;
flat in uint v_layer;

void main() {
    if (u_alpha_test && texture(u_tiles, vec3(v_uv, float(v_layer))).a < 0.5) {
        discard;
    }
}
"""

TONEMAP_VERT = """#version 330
out vec2 v_uv;

void main() {
    vec2 pos = vec2(
        gl_VertexID == 1 ? 3.0 : -1.0,
        gl_VertexID == 2 ? 3.0 : -1.0
    );
    v_uv = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
}
"""

TONEMAP_FRAG = """#version 330
uniform sampler2D u_scene;
uniform sampler2D u_bloom_a;
uniform sampler2D u_bloom_b;
uniform sampler2D u_bloom_c;
uniform float u_bloom_strength;
uniform bool u_underwater;
uniform float u_time;
uniform vec3 u_grade;       // time-of-day colour grading multiplier
uniform vec3 u_sun_screen;  // xy = sun position in UV space, z = ray strength

in vec2 v_uv;
out vec4 f_color;

vec3 aces(vec3 x) {
    // Narkowicz ACES filmic approximation.
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
    vec2 uv = v_uv;
    if (u_underwater) {
        uv += vec2(sin(uv.y * 42.0 + u_time * 2.6), cos(uv.x * 38.0 + u_time * 2.2)) * 0.0035;
    }
    vec3 c = texture(u_scene, uv).rgb;
    c += (texture(u_bloom_a, uv).rgb * 0.5
        + texture(u_bloom_b, uv).rgb * 0.35
        + texture(u_bloom_c, uv).rgb * 0.25) * u_bloom_strength;

    // Cheap god rays: march the mid bloom level towards the sun (5 taps).
    if (u_sun_screen.z > 0.02) {
        vec2 delta = (u_sun_screen.xy - uv) / 5.0;
        vec2 p = uv;
        vec3 rays = vec3(0.0);
        float weight = 1.0;
        for (int i = 0; i < 5; i++) {
            p += delta;
            rays += texture(u_bloom_c, p).rgb * weight;
            weight *= 0.78;
        }
        c += rays * 0.03 * u_sun_screen.z;
    }

    c *= u_grade * 1.06;
    c = aces(c);
    float vig = 1.0 - 0.22 * dot(v_uv - 0.5, v_uv - 0.5) * 4.0 * 0.55;
    f_color = vec4(c * vig, 1.0);
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
out vec3 v_normal;
out float v_sky;
out float v_blk;
out float v_shade;
out float v_sun;
out float v_emission;
flat out uint v_layer;

{_TABLES}

void main() {{
{_VERTEX_DECODE}
    vec3 local = vec3(bx, by, bz) + corner_off;
    vec3 world = u_chunk_origin + local;
    v_normal = FACE_NORMALS[face];
    // Liquid surface sits slightly below the block top; two summed wave
    // trains give a livelier swell than a single sine.
    if (flag == 1u && corner_off.y > 0.5) {{
        world.y -= 0.115;
        world.y += sin(u_time * 1.6 + world.x * 0.7 + world.z * 0.9) * 0.04
                 + sin(u_time * 2.3 + world.x * 1.7 - world.z * 1.1) * 0.022;
    }}
    v_world = world;
    v_uv = FACE_UVS[idx];
    v_layer = tex_layer;
{_SHADE_CALC}
    gl_Position = u_view_proj * vec4(world, 1.0);
}}
"""

WATER_FRAG = """#version 330
uniform sampler2DArray u_tiles;
uniform vec3 u_fog_color;
uniform vec2 u_fog_range;
uniform vec3 u_camera_pos;
uniform float u_daylight;
uniform float u_alpha;
uniform float u_time;
uniform vec3 u_sun_dir;
uniform vec3 u_zenith_color;
uniform vec3 u_horizon_color;

in vec3 v_world;
in vec2 v_uv;
in vec3 v_normal;
in float v_sky;
in float v_blk;
in float v_shade;
in float v_sun;
in float v_emission;
flat in uint v_layer;

out vec4 f_color;

void main() {
    // Animated surface: distort the UVs so liquids visibly flow.
    vec2 uv = v_uv + vec2(
        sin(u_time * 1.1 + v_world.z * 1.3),
        cos(u_time * 0.9 + v_world.x * 1.1)
    ) * 0.05;
    vec4 tex = texture(u_tiles, vec3(uv, float(v_layer)));
    float shadow = 1.0;  // liquids skip CSM sampling (cost/benefit)
    float direct_sun = v_sun;
""" + _LIGHT_COMBINE + """
    vec3 color = tex.rgb * light * v_shade;

    // Fresnel sky reflection + sun specular (plan phase 2 water).
    vec3 n = normalize(v_normal);
    vec3 view = normalize(u_camera_pos - v_world);
    float fresnel = pow(1.0 - max(dot(n, view), 0.0), 3.0);
    vec3 sky_reflect = mix(u_horizon_color, u_zenith_color, 0.5 + 0.5 * n.y);
    color = mix(color, sky_reflect * max(u_daylight, 0.12), fresnel * 0.65);
    vec3 half_vec = normalize(view + u_sun_dir);
    float spec = pow(max(dot(n, half_vec), 0.0), 220.0) * u_daylight;
    color += spec * vec3(1.0, 0.94, 0.78) * 1.8;  // overshoots -> blooms

    float alpha = mix(u_alpha, 0.9, fresnel);
    float dist = length(v_world - u_camera_pos);
    float fog = smoothstep(u_fog_range.x, u_fog_range.y, dist);
    f_color = vec4(mix(color, u_fog_color, fog), alpha);
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

    // Moon: small cold disc opposite the sun.
    float moon_dot = max(dot(dir, -u_sun_dir), 0.0);
    col += pow(moon_dot, 2400.0) * vec3(0.75, 0.78, 0.85) * (1.0 - u_daylight);

    // Sparse hash-based starfield fades in at night.
    if (u_daylight < 0.5 && up > 0.02) {
        vec3 cell = floor(dir * 110.0);
        float star = step(0.9982, star_hash(cell));
        col += star * (0.5 - u_daylight) * 1.6;
    }
    f_color = vec4(col, 1.0);
}
"""

CLOUD_VERT = """#version 330
uniform mat4 u_view_proj;
uniform vec3 u_cloud_origin;

in vec2 in_pos;

out vec3 v_world;

void main() {
    v_world = vec3(in_pos.x, 0.0, in_pos.y) + u_cloud_origin;
    gl_Position = u_view_proj * vec4(v_world, 1.0);
}
"""

CLOUD_FRAG = """#version 330
uniform vec3 u_fog_color;
uniform vec2 u_fog_range;
uniform vec3 u_camera_pos;
uniform float u_daylight;

in vec3 v_world;
out vec4 f_color;

void main() {
    vec3 color = vec3(mix(0.08, 1.0, u_daylight));
    float dist = length(v_world - u_camera_pos);
    float fog = smoothstep(u_fog_range.x, u_fog_range.y, dist);
    f_color = vec4(mix(color, u_fog_color, fog), 0.55 * (1.0 - fog * 0.6));
}
"""

RAIN_VERT = """#version 330
uniform mat4 u_view_proj;
uniform vec3 u_center;
uniform float u_time;

in vec3 in_drop;    // x, z offset in the rain cylinder + phase
in vec3 in_corner;  // corner dx, dy, quad axis (0 = X-facing, 1 = Z-facing)

out float v_alpha;

void main() {
    float fall = fract(in_drop.z - u_time * 0.6);
    vec3 world = u_center + vec3(in_drop.x, fall * 22.0, in_drop.y);
    if (in_corner.z < 0.5) {
        world.x += in_corner.x * 0.05;
    } else {
        world.z += in_corner.x * 0.05;
    }
    world.y += in_corner.y * 0.8;  // streak length
    v_alpha = 0.30 * (1.0 - fall * 0.55);
    gl_Position = u_view_proj * vec4(world, 1.0);
}
"""

RAIN_FRAG = """#version 330
in float v_alpha;
out vec4 f_color;

void main() {
    f_color = vec4(0.62, 0.70, 0.86, v_alpha);
}
"""

PARTICLE_VERT = """#version 330
uniform mat4 u_view_proj;
uniform vec3 u_camera_right;
uniform vec3 u_camera_up;

in vec2 in_corner;    // quad corner (-0.5..0.5)
in vec3 in_pos;       // instance world position
in vec4 in_color;     // instance rgba (a = current alpha)
in float in_size;     // instance billboard size

out vec4 v_color;
out vec2 v_uv;

void main() {
    vec3 world = in_pos + u_camera_right * in_corner.x * in_size
                        + u_camera_up * in_corner.y * in_size;
    v_color = in_color;
    v_uv = in_corner;
    gl_Position = u_view_proj * vec4(world, 1.0);
}
"""

PARTICLE_FRAG = """#version 330
in vec4 v_color;
in vec2 v_uv;
out vec4 f_color;

void main() {
    float d = length(v_uv) * 2.0;
    if (d > 1.0) discard;                     // soft round sprite
    float soft = 1.0 - smoothstep(0.6, 1.0, d);
    f_color = vec4(v_color.rgb, v_color.a * soft);
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
