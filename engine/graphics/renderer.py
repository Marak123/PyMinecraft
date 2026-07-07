"""GPU renderer.

Owns every GL object: programs, the tile texture array, per-chunk VAOs and
the dynamic UI buffers.  Consumes plain data (mesh streams, environment
uniform values) — it knows nothing about gameplay.

Frame structure:
    sky (fullscreen, no depth) -> opaque (front-to-back, culled)
    -> cutout (alpha-tested, no cull) -> water (blended, back-to-front)
    -> clouds (blended, no depth write) -> highlight box -> UI overlay

Chunk buffers are pooled: a remesh reuses the existing GPU buffer when the
new data fits (with slack), avoiding constant allocate/free churn while
editing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import moderngl
import numpy as np

from engine.core import mathx
from engine.core.log import get_logger
from engine.graphics import atlas, shaders
from engine.graphics.font import FontAtlas
from engine.graphics.framebuffer import SceneTargets
from engine.graphics.mesher import ChunkMeshData
from engine.graphics.shadow import CASCADE_SPLITS, cascade_matrices
from engine.world.blocks import BlockRegistry
from engine.world.coords import CHUNK_X, CHUNK_Y, CHUNK_Z
from engine.world.environment import Environment

_log = get_logger("renderer")

# UI-only tiles appended to the block tile atlas (hearts, bubbles...).
_UI_TILES = ("heart_full", "heart_half", "heart_empty", "bubble",
             "food_full", "food_half", "food_empty")

_CLOUD_ALTITUDE = 224.0  # above the tallest 256-world peaks
_CLOUD_CELL = 16.0
_CLOUD_GRID = 32
_CLOUD_PERIOD = _CLOUD_CELL * _CLOUD_GRID


@dataclass
class _Stream:
    vbo: moderngl.Buffer
    vao: moderngl.VertexArray
    svao: moderngl.VertexArray | None  # depth-only VAO for the shadow pass
    count: int      # vertices in use
    capacity: int   # bytes allocated

    def release(self) -> None:
        self.vao.release()
        if self.svao is not None:
            self.svao.release()
        self.vbo.release()


class _ChunkGpu:
    __slots__ = ("origin", "center", "half", "opaque", "cutout", "water")

    def __init__(self, cx: int, cz: int) -> None:
        ox, oz = cx * CHUNK_X, cz * CHUNK_Z
        self.origin = (float(ox), 0.0, float(oz))
        self.center = np.array(
            [ox + CHUNK_X / 2, CHUNK_Y / 2, oz + CHUNK_Z / 2], dtype=np.float64
        )
        self.half = np.array([CHUNK_X / 2, CHUNK_Y / 2, CHUNK_Z / 2], dtype=np.float64)
        self.opaque: _Stream | None = None
        self.cutout: _Stream | None = None
        self.water: _Stream | None = None

    def set_y_bounds(self, y_min: int, y_max: int) -> None:
        # Tight vertical bounds from the mesher make frustum culling reject
        # flat far chunks when looking up/down.
        self.center[1] = (y_min + y_max) / 2
        self.half[1] = max((y_max - y_min) / 2, 1.0)

    def release(self) -> None:
        for stream in (self.opaque, self.cutout, self.water):
            if stream is not None:
                stream.release()
        self.opaque = self.cutout = self.water = None


def _unit_cube_edges() -> np.ndarray:
    corners = [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]
    lines: list[tuple[float, float, float]] = []
    for i, a in enumerate(corners):
        for b in corners[i + 1 :]:
            if sum(abs(a[k] - b[k]) for k in range(3)) == 1:
                lines.extend((a, b))
    verts = np.array(lines, dtype=np.float32)
    # Slight inflation prevents z-fighting with the block's own faces.
    return (verts - 0.5) * 1.004 + 0.5


def _rain_mesh(drops: int = 340) -> np.ndarray:
    """Static rain streaks: two crossed quads per drop, animated in-shader."""
    rng = np.random.default_rng(11)
    angle = rng.random(drops) * 2 * np.pi
    radius = np.sqrt(rng.random(drops)) * 15.0
    x = np.cos(angle) * radius
    z = np.sin(angle) * radius
    phase = rng.random(drops)
    corners = [(-0.5, 0.0), (0.5, 0.0), (0.5, 1.0), (-0.5, 0.0), (0.5, 1.0), (-0.5, 1.0)]
    verts = np.empty((drops, 12, 6), dtype=np.float32)
    for axis in (0, 1):
        for c, (cx, cy) in enumerate(corners):
            i = axis * 6 + c
            verts[:, i, 0] = x
            verts[:, i, 1] = z
            verts[:, i, 2] = phase
            verts[:, i, 3] = cx
            verts[:, i, 4] = cy
            verts[:, i, 5] = axis
    return verts.reshape(-1, 6)


def _cloud_mesh() -> np.ndarray:
    """Static quad field for one cloud tile (periodic, world-anchored)."""
    rng = np.random.default_rng(7)
    field = rng.random((_CLOUD_GRID, _CLOUD_GRID))
    # Two smoothing passes clump the noise into blobby cloud banks.
    for _ in range(2):
        field = sum(
            np.roll(np.roll(field, dx, 0), dz, 1)
            for dx in (-1, 0, 1)
            for dz in (-1, 0, 1)
        ) / 9.0
    mask = field > 0.55
    xs, zs = np.nonzero(mask)
    quads = np.empty((len(xs), 6, 2), dtype=np.float32)
    x0 = xs * _CLOUD_CELL
    z0 = zs * _CLOUD_CELL
    x1 = x0 + _CLOUD_CELL
    z1 = z0 + _CLOUD_CELL
    quads[:, 0] = np.stack([x0, z0], 1)
    quads[:, 1] = np.stack([x1, z0], 1)
    quads[:, 2] = np.stack([x1, z1], 1)
    quads[:, 3] = np.stack([x0, z0], 1)
    quads[:, 4] = np.stack([x1, z1], 1)
    quads[:, 5] = np.stack([x0, z1], 1)
    return quads.reshape(-1, 2)


class Renderer:
    def __init__(
        self,
        ctx: moderngl.Context,
        registry: BlockRegistry,
        font_atlas: FontAtlas | None = None,
        pack_dir: str | None = None,
    ) -> None:
        self.ctx = ctx
        self.registry = registry
        self._chunks: dict[tuple[int, int], _ChunkGpu] = {}

        self.prog_chunk = ctx.program(shaders.CHUNK_VERT, shaders.CHUNK_FRAG)
        self.prog_water = ctx.program(shaders.WATER_VERT, shaders.WATER_FRAG)
        self.prog_shadow = ctx.program(shaders.SHADOW_VERT, shaders.SHADOW_FRAG)
        self.prog_tonemap = ctx.program(shaders.TONEMAP_VERT, shaders.TONEMAP_FRAG)
        self.prog_sky = ctx.program(shaders.SKY_VERT, shaders.SKY_FRAG)
        self.prog_clouds = ctx.program(shaders.CLOUD_VERT, shaders.CLOUD_FRAG)
        self.prog_lines = ctx.program(shaders.LINES_VERT, shaders.LINES_FRAG)
        self.prog_ui_color = ctx.program(shaders.UI_COLOR_VERT, shaders.UI_COLOR_FRAG)
        self.prog_ui_text = ctx.program(shaders.UI_TEXT_VERT, shaders.UI_TEXT_FRAG)
        self.prog_ui_block = ctx.program(shaders.UI_BLOCK_VERT, shaders.UI_BLOCK_FRAG)

        # Procedural tile texture array; layer ids are baked into the registry
        # so the mesher can emit them directly into vertex data.  UI-only
        # tiles (hearts etc.) ride along in the same array.
        tile_names = registry.required_tiles() + [
            t for t in _UI_TILES if t not in registry.required_tiles()
        ]
        albedo, normal, mrao, mapping = atlas.build_tiles(tile_names, pack_dir)
        registry.assign_texture_layers(mapping)
        self._tile_layers = mapping
        size = atlas.ATLAS_SIZE
        layers = albedo.shape[0]
        self.tiles = ctx.texture_array((size, size, layers), 4, albedo.tobytes())
        self.tiles.build_mipmaps()
        self.tiles.filter = (moderngl.NEAREST_MIPMAP_LINEAR, moderngl.NEAREST)
        self.tiles_normal = ctx.texture_array((size, size, layers), 3, normal.tobytes())
        self.tiles_mrao = ctx.texture_array((size, size, layers), 3, mrao.tobytes())
        for tex in (self.tiles_normal, self.tiles_mrao):
            tex.build_mipmaps()
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        try:
            self.tiles.anisotropy = 8.0
            self.tiles_normal.anisotropy = 8.0
            self.tiles_mrao.anisotropy = 8.0
        except Exception:  # noqa: BLE001 - optional GPU feature
            pass

        self.prog_bloom = ctx.program(shaders.TONEMAP_VERT, shaders.BLOOM_FRAG)
        self._sky_vao = ctx.vertex_array(self.prog_sky, [])
        self._tonemap_vao = ctx.vertex_array(self.prog_tonemap, [])
        self._bloom_vao = ctx.vertex_array(self.prog_bloom, [])
        self._set(self.prog_tonemap, "u_scene", 0)
        self._set(self.prog_tonemap, "u_bloom_a", 1)
        self._set(self.prog_tonemap, "u_bloom_b", 2)
        self._set(self.prog_tonemap, "u_bloom_c", 3)
        # HDR scene target + shadow cascades (resized on first frame).
        self.targets = SceneTargets(ctx, 8, 8)
        for i, tex in enumerate(self.targets.shadow_maps):
            tex.use(location=5 + i)
        self._set(self.prog_chunk, "u_shadow0", 5)
        self._set(self.prog_chunk, "u_shadow1", 6)
        self._set(self.prog_chunk, "u_shadow2", 7)
        self._set(self.prog_chunk, "u_cascade_far", CASCADE_SPLITS)
        self._set(self.prog_chunk, "u_normal_map", 8)
        self._set(self.prog_chunk, "u_mrao_map", 9)
        self._box_vbo = ctx.buffer(_unit_cube_edges().tobytes())
        self._box_vao = ctx.vertex_array(
            self.prog_lines, [(self._box_vbo, "3f4", "in_pos")]
        )
        cloud_verts = _cloud_mesh()
        self._cloud_count = len(cloud_verts)
        self._cloud_vbo = ctx.buffer(cloud_verts.tobytes())
        self._cloud_vao = ctx.vertex_array(
            self.prog_clouds, [(self._cloud_vbo, "2f4", "in_pos")]
        )
        self.prog_rain = ctx.program(shaders.RAIN_VERT, shaders.RAIN_FRAG)
        rain_verts = _rain_mesh()
        self._rain_count = len(rain_verts)
        self._rain_vbo = ctx.buffer(rain_verts.tobytes())
        self._rain_vao = ctx.vertex_array(
            self.prog_rain, [(self._rain_vbo, "3f4 3f4", "in_drop", "in_corner")]
        )

        # Dynamic UI buffers, grown on demand.
        self._ui_color_vbo = ctx.buffer(reserve=64 * 1024, dynamic=True)
        self._ui_color_vao = ctx.vertex_array(
            self.prog_ui_color, [(self._ui_color_vbo, "2f4 4f4", "in_pos", "in_color")]
        )
        self._ui_text_vbo = ctx.buffer(reserve=256 * 1024, dynamic=True)
        self._ui_text_vao = ctx.vertex_array(
            self.prog_ui_text, [(self._ui_text_vbo, "2f4 2f4", "in_pos", "in_uv")]
        )
        self._ui_block_vbo = ctx.buffer(reserve=64 * 1024, dynamic=True)
        self._ui_block_vao = ctx.vertex_array(
            self.prog_ui_block, [(self._ui_block_vbo, "2f4 3f4", "in_pos", "in_uvl")]
        )

        self.font: FontAtlas | None = font_atlas
        self.font_tex: moderngl.Texture | None = None
        if font_atlas is not None:
            h, w = font_atlas.image.shape
            self.font_tex = ctx.texture((w, h), 1, font_atlas.image.tobytes())
            self.font_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)

        # Average colour per tile layer — particle tint for block breaks.
        self._tile_avg = (albedo[:, :, :, :3].reshape(layers, -1, 3)
                          .mean(axis=1) / 255.0).astype("f4")

        self._set(self.prog_water, "u_alpha", 0.62)
        _log.info("Renderer ready (%d tile layers @ %dpx)", layers, size)

    def block_color(self, block_id: int) -> tuple[float, float, float]:
        layer = int(self.registry.face_layers[block_id, 4])
        r, g, b = self._tile_avg[layer]
        return (float(r), float(g), float(b))

    def tile_layer(self, name: str) -> int:
        """Texture-array layer of a tile by name (UI icons)."""
        return self._tile_layers[name]

    # -- small uniform helpers ---------------------------------------------------
    @staticmethod
    def _set(prog: moderngl.Program, name: str, value) -> None:
        try:
            prog[name].value = value
        except KeyError:
            pass  # uniform optimised out — harmless

    @staticmethod
    def _set_mat(prog: moderngl.Program, name: str, m: np.ndarray) -> None:
        try:
            prog[name].write(m.T.astype("f4").tobytes())
        except KeyError:
            pass

    # -- chunk mesh lifecycle ------------------------------------------------------
    def _update_stream(
        self, prog: moderngl.Program, stream: _Stream | None, data: np.ndarray | None
    ) -> _Stream | None:
        """Upload a mesh stream, reusing the existing buffer when it fits."""
        if data is None or len(data) == 0:
            if stream is not None:
                stream.release()
            return None
        payload = data.tobytes()
        if stream is not None and len(payload) <= stream.capacity:
            stream.vbo.orphan()
            stream.vbo.write(payload)
            stream.count = len(data) // 2
            return stream
        if stream is not None:
            stream.release()
        capacity = int(len(payload) * 1.25)
        vbo = self.ctx.buffer(reserve=capacity, dynamic=True)
        vbo.write(payload)
        vao = self.ctx.vertex_array(prog, [(vbo, "2u4", "in_data")])
        # Chunk-shader streams get a second, depth-only VAO for shadows.
        svao = None
        if prog is self.prog_chunk:
            svao = self.ctx.vertex_array(
                self.prog_shadow, [(vbo, "2u4", "in_data")]
            )
        return _Stream(vbo, vao, svao, len(data) // 2, capacity)

    def upload_chunk(self, cx: int, cz: int, mesh: ChunkMeshData) -> None:
        key = (cx, cz)
        gpu = self._chunks.get(key)
        if gpu is None:
            gpu = _ChunkGpu(cx, cz)
            self._chunks[key] = gpu
        gpu.opaque = self._update_stream(self.prog_chunk, gpu.opaque, mesh.opaque)
        gpu.cutout = self._update_stream(self.prog_chunk, gpu.cutout, mesh.cutout)
        gpu.water = self._update_stream(self.prog_water, gpu.water, mesh.water)
        gpu.set_y_bounds(mesh.y_min, mesh.y_max)

    def unload_chunk(self, cx: int, cz: int) -> None:
        gpu = self._chunks.pop((cx, cz), None)
        if gpu is not None:
            gpu.release()

    # -- frame ----------------------------------------------------------------------
    def resize(self, width: int, height: int) -> None:
        self.ctx.viewport = (0, 0, width, height)
        self.targets.resize(width, height)

    def _shadow_pass(self, camera, env, gpus, centers) -> list[np.ndarray] | None:
        """Render the CSM cascades; returns light matrices or None (night)."""
        if env.daylight <= 0.02 or not gpus:
            return None
        matrices = cascade_matrices(
            camera.position, camera.forward, env.sun_dir,
            self.targets.shadow_resolution,
        )
        ctx = self.ctx
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE | moderngl.BLEND)
        self.tiles.use(location=0)
        prog = self.prog_shadow
        self._set(prog, "u_tiles", 0)
        origin_uniform = prog["u_chunk_origin"]
        dist2 = np.sum((centers[:, [0, 2]] - camera.position[[0, 2]]) ** 2, axis=1)

        for i, (fbo, matrix) in enumerate(zip(self.targets.shadow_fbos, matrices)):
            fbo.clear(depth=1.0)
            fbo.use()
            self._set_mat(prog, "u_light_vp", matrix)
            # Only chunks near enough to cast into this cascade.
            reach = (CASCADE_SPLITS[i] + CHUNK_X * 2.5) ** 2
            self._set(prog, "u_alpha_test", False)
            casters = [g for g, d in zip(gpus, dist2) if d < reach]
            for gpu in casters:
                if gpu.opaque is not None and gpu.opaque.svao is not None:
                    origin_uniform.value = gpu.origin
                    gpu.opaque.svao.render(moderngl.TRIANGLES, vertices=gpu.opaque.count)
            # Cutout (leaves) shadows only in the two near cascades — the far
            # cascade's alpha-tested overdraw isn't worth the fill cost.
            if i < 2:
                self._set(prog, "u_alpha_test", True)
                for gpu in casters:
                    if gpu.cutout is not None and gpu.cutout.svao is not None:
                        origin_uniform.value = gpu.origin
                        gpu.cutout.svao.render(moderngl.TRIANGLES, vertices=gpu.cutout.count)
        return matrices

    def _finish_frame(
        self,
        out_fbo: moderngl.Framebuffer,
        extra_pass,
        view_proj: np.ndarray,
        env: Environment,
        time_s: float,
        underwater: bool,
        bloom: bool,
    ) -> None:
        """Entity/rain callback into HDR, bloom chain, tonemap to output."""
        if extra_pass is not None:
            extra_pass()
        ctx = self.ctx
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)

        if bloom:
            # Threshold-extract into the 1/2 target, then downsample twice.
            src = self.targets.hdr_color
            for i, (tex, fbo) in enumerate(self.targets.bloom):
                fbo.use()
                src.use(location=0)
                self._set(self.prog_bloom, "u_src", 0)
                self._set(self.prog_bloom, "u_extract", i == 0)
                self._bloom_vao.render(moderngl.TRIANGLES, vertices=3)
                src = tex

        # Sun position in screen UV space for the god-ray march.
        sun_screen = (0.0, 0.0, 0.0)
        if env.daylight > 0.05:
            # w=0: a direction (point at infinity) — camera-translation-proof.
            clip = view_proj @ np.append(env.sun_dir, 0.0)
            if clip[3] > 0.0:
                ndc = clip[:2] / clip[3]
                if abs(ndc[0]) < 1.3 and abs(ndc[1]) < 1.3:
                    fade = max(0.0, 1.0 - (abs(ndc[0]) + abs(ndc[1])) * 0.45)
                    sun_screen = (
                        float(ndc[0] * 0.5 + 0.5),
                        float(ndc[1] * 0.5 + 0.5),
                        float(env.daylight * fade),
                    )

        out_fbo.use()
        self.targets.hdr_color.use(location=0)
        for i, (tex, _fbo) in enumerate(self.targets.bloom):
            tex.use(location=1 + i)
        prog = self.prog_tonemap
        self._set(prog, "u_bloom_strength", 0.85 if bloom else 0.0)
        self._set(prog, "u_underwater", underwater)
        self._set(prog, "u_time", time_s)
        self._set(prog, "u_grade", tuple(env.color_grade))
        self._set(prog, "u_sun_screen", sun_screen)
        self._tonemap_vao.render(moderngl.TRIANGLES, vertices=3)

    def render_world(
        self,
        camera,
        env: Environment,
        time_s: float,
        fog_start: float,
        fog_end: float,
        fog_color: tuple[float, float, float] | None = None,
        highlight: tuple[int, int, int] | None = None,
        shadows: bool = True,
        extra_pass=None,
        underwater: bool = False,
        bloom: bool = True,
    ) -> dict[str, int]:
        ctx = self.ctx
        view_proj = camera.view_proj()
        fog = tuple(fog_color if fog_color is not None else env.fog_color)
        self._daylight_cache = env.daylight

        # The scene renders into the HDR target; whatever framebuffer the
        # caller had bound (screen or a test FBO) receives the tonemapped blit.
        out_fbo = ctx.fbo
        self.targets.resize(*out_fbo.size)

        import time as _t
        gpus_all = list(self._chunks.values())
        centers_all = (
            np.array([g.center for g in gpus_all]) if gpus_all else np.zeros((0, 3))
        )
        _ts = _t.perf_counter()
        shadow_mats = (
            self._shadow_pass(camera, env, gpus_all, centers_all) if shadows else None
        )
        self.last_shadow_ms = (_t.perf_counter() - _ts) * 1000.0

        self.targets.hdr_fbo.use()
        ctx.clear(fog[0], fog[1], fog[2], 1.0, depth=1.0)

        # -- sky --------------------------------------------------------------
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
        self._set_mat(self.prog_sky, "u_inv_view_proj", np.linalg.inv(view_proj))
        self._set(self.prog_sky, "u_sun_dir", tuple(env.sun_dir))
        self._set(self.prog_sky, "u_zenith_color", tuple(env.zenith_color))
        self._set(self.prog_sky, "u_horizon_color", tuple(fog))
        self._set(self.prog_sky, "u_daylight", env.daylight)
        self._sky_vao.render(moderngl.TRIANGLES, vertices=3)

        # -- visibility -----------------------------------------------------------
        stats = {"chunks_loaded": len(gpus_all), "chunks_visible": 0, "vertices": 0}
        if not gpus_all:
            self._render_clouds(camera, view_proj, time_s, fog, fog_start, fog_end)
            self._finish_frame(out_fbo, extra_pass, view_proj, env, time_s, underwater, bloom)
            return stats
        halves = np.array([g.half for g in gpus_all])
        planes = camera.frustum_planes()
        visible_mask = mathx.aabbs_in_frustum(planes, centers_all, halves)
        visible_idx = np.nonzero(visible_mask)[0]
        if len(visible_idx) == 0:
            self._render_clouds(camera, view_proj, time_s, fog, fog_start, fog_end)
            self._finish_frame(out_fbo, extra_pass, view_proj, env, time_s, underwater, bloom)
            return stats
        dist2 = np.sum((centers_all[visible_idx] - camera.position) ** 2, axis=1)
        order = visible_idx[np.argsort(dist2)]  # front-to-back
        visible = [gpus_all[i] for i in order]
        stats["chunks_visible"] = len(visible)

        # -- opaque ------------------------------------------------------------
        ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.tiles.use(location=0)
        self.tiles_normal.use(location=8)
        self.tiles_mrao.use(location=9)
        for i, tex in enumerate(self.targets.shadow_maps):
            tex.use(location=5 + i)
        prog = self.prog_chunk
        self._set(prog, "u_tiles", 0)
        self._set(prog, "u_shadows", shadow_mats is not None)
        if shadow_mats is not None:
            try:
                prog["u_light_vp"].write(
                    np.concatenate(
                        [m.T.astype("f4").reshape(-1) for m in shadow_mats]
                    ).tobytes()
                )
            except KeyError:
                pass
        self._set_mat(prog, "u_view_proj", view_proj)
        self._set(prog, "u_sun_dir", tuple(env.sun_dir))
        self._set(prog, "u_daylight", env.daylight)
        self._set(prog, "u_time", time_s)
        self._set(prog, "u_fog_color", fog)
        self._set(prog, "u_fog_range", (fog_start, fog_end))
        self._set(prog, "u_camera_pos", tuple(camera.position))
        self._set(prog, "u_alpha_test", False)
        origin_uniform = prog["u_chunk_origin"]
        for gpu in visible:
            if gpu.opaque is not None:
                origin_uniform.value = gpu.origin
                gpu.opaque.vao.render(moderngl.TRIANGLES, vertices=gpu.opaque.count)
                stats["vertices"] += gpu.opaque.count

        # -- cutout (leaves, glass, plants): both faces visible ---------------------
        ctx.disable(moderngl.CULL_FACE)
        self._set(prog, "u_alpha_test", True)
        for gpu in visible:
            if gpu.cutout is not None:
                origin_uniform.value = gpu.origin
                gpu.cutout.vao.render(moderngl.TRIANGLES, vertices=gpu.cutout.count)
                stats["vertices"] += gpu.cutout.count

        # -- water (translucent): back-to-front --------------------------------------
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        prog = self.prog_water
        self._set(prog, "u_tiles", 0)
        self._set_mat(prog, "u_view_proj", view_proj)
        self._set(prog, "u_zenith_color", tuple(env.zenith_color))
        self._set(prog, "u_horizon_color", tuple(env.horizon_color))
        self._set(prog, "u_sun_dir", tuple(env.sun_dir))
        self._set(prog, "u_daylight", env.daylight)
        self._set(prog, "u_time", time_s)
        self._set(prog, "u_fog_color", fog)
        self._set(prog, "u_fog_range", (fog_start, fog_end))
        self._set(prog, "u_camera_pos", tuple(camera.position))
        water_origin = prog["u_chunk_origin"]
        for gpu in reversed(visible):
            if gpu.water is not None:
                water_origin.value = gpu.origin
                gpu.water.vao.render(moderngl.TRIANGLES, vertices=gpu.water.count)
                stats["vertices"] += gpu.water.count

        self._render_clouds(camera, view_proj, time_s, fog, fog_start, fog_end)

        # -- targeted block outline ---------------------------------------------------
        if highlight is not None:
            ctx.enable(moderngl.BLEND)
            self._set_mat(self.prog_lines, "u_view_proj", view_proj)
            self._set(self.prog_lines, "u_offset", tuple(float(c) for c in highlight))
            self._set(self.prog_lines, "u_color", (0.0, 0.0, 0.0, 0.7))
            self._box_vao.render(moderngl.LINES)

        ctx.disable(moderngl.BLEND)
        self._finish_frame(out_fbo, extra_pass, view_proj, env, time_s, underwater, bloom)
        return stats

    def _render_clouds(
        self,
        camera,
        view_proj: np.ndarray,
        time_s: float,
        fog: tuple[float, float, float],
        fog_start: float,
        fog_end: float,
    ) -> None:
        ctx = self.ctx
        ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        ctx.disable(moderngl.CULL_FACE)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        try:
            ctx.fbo.depth_mask = False
        except AttributeError:
            pass
        prog = self.prog_clouds
        self._set_mat(prog, "u_view_proj", view_proj)
        self._set(prog, "u_fog_color", fog)
        # Clouds live far above terrain: stretch their fog so they stay
        # visible to the horizon instead of dissolving with the ground fog.
        self._set(prog, "u_fog_range", (fog_start * 1.6, fog_end * 2.2))
        self._set(prog, "u_camera_pos", tuple(camera.position))
        self._set(prog, "u_daylight", getattr(self, "_daylight_cache", 1.0))
        origin = prog["u_cloud_origin"]

        drift = time_s * 1.7
        base_x = math.floor((camera.position[0] - drift) / _CLOUD_PERIOD)
        base_z = math.floor(camera.position[2] / _CLOUD_PERIOD)
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                origin.value = (
                    (base_x + dx) * _CLOUD_PERIOD + drift,
                    _CLOUD_ALTITUDE,
                    (base_z + dz) * _CLOUD_PERIOD,
                )
                self._cloud_vao.render(moderngl.TRIANGLES, vertices=self._cloud_count)
        try:
            ctx.fbo.depth_mask = True
        except AttributeError:
            pass

    def render_rain(self, camera, time_s: float) -> None:
        """Rain streak cylinder around the camera; call while it is raining."""
        ctx = self.ctx
        ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        ctx.disable(moderngl.CULL_FACE)
        try:
            ctx.fbo.depth_mask = False
        except AttributeError:
            pass
        prog = self.prog_rain
        self._set_mat(prog, "u_view_proj", camera.view_proj())
        self._set(prog, "u_time", time_s)
        self._set(prog, "u_center", (
            float(camera.position[0]),
            float(camera.position[1]) - 11.0,
            float(camera.position[2]),
        ))
        self._rain_vao.render(moderngl.TRIANGLES, vertices=self._rain_count)
        try:
            ctx.fbo.depth_mask = True
        except AttributeError:
            pass
        ctx.disable(moderngl.BLEND)

    # -- UI -----------------------------------------------------------------------
    def begin_ui(self, width: int, height: int) -> None:
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        proj = mathx.ortho_2d(width, height)
        self._set_mat(self.prog_ui_color, "u_proj", proj)
        self._set_mat(self.prog_ui_text, "u_proj", proj)
        self._set_mat(self.prog_ui_block, "u_proj", proj)

    def _write_dynamic(self, vbo: moderngl.Buffer, data: bytes) -> moderngl.Buffer:
        if len(data) > vbo.size:
            vbo.orphan(len(data) * 2)
        vbo.write(data)
        return vbo

    def draw_ui_rects(self, verts: np.ndarray) -> None:
        """verts: (N, 6) float32 rows of (x, y, r, g, b, a)."""
        if len(verts) == 0:
            return
        self._write_dynamic(self._ui_color_vbo, verts.astype(np.float32).tobytes())
        self._ui_color_vao.render(moderngl.TRIANGLES, vertices=len(verts))

    def draw_ui_text(self, verts: np.ndarray, color: tuple[float, float, float, float]) -> None:
        """verts: (N, 4) float32 rows of (x, y, u, v) from font.layout_text."""
        if self.font_tex is None or len(verts) == 0:
            return
        self.font_tex.use(location=1)
        self._set(self.prog_ui_text, "u_font", 1)
        self._set(self.prog_ui_text, "u_color", color)
        self._write_dynamic(self._ui_text_vbo, verts.astype(np.float32).tobytes())
        self._ui_text_vao.render(moderngl.TRIANGLES, vertices=len(verts))

    def draw_ui_blocks(self, verts: np.ndarray) -> None:
        """verts: (N, 5) float32 rows of (x, y, u, v, layer)."""
        if len(verts) == 0:
            return
        self.tiles.use(location=0)
        self._set(self.prog_ui_block, "u_tiles", 0)
        self._write_dynamic(self._ui_block_vbo, verts.astype(np.float32).tobytes())
        self._ui_block_vao.render(moderngl.TRIANGLES, vertices=len(verts))

    # -- misc ---------------------------------------------------------------------
    def screenshot(self, path: str) -> None:
        from PIL import Image

        fbo = self.ctx.fbo
        data = fbo.read(components=3)
        img = Image.frombytes("RGB", fbo.size, data).transpose(Image.FLIP_TOP_BOTTOM)
        img.save(path)
        _log.info("Screenshot saved to %s", path)

    def clear_chunks(self) -> None:
        """Drop all chunk meshes (dimension switch) — keeps GL programs/FBOs."""
        for gpu in self._chunks.values():
            gpu.release()
        self._chunks.clear()

    def release(self) -> None:
        self.clear_chunks()
        self.targets.release()
