"""GPU renderer.

Owns every GL object: programs, the tile texture array, per-chunk VAOs and
the dynamic UI buffers.  Consumes plain data (mesh streams, environment
uniform values) — it knows nothing about gameplay.

Frame structure:
    sky (fullscreen, no depth) -> opaque (front-to-back, culled)
    -> cutout (alpha-tested, no cull) -> water (blended, back-to-front)
    -> highlight box -> UI overlay
"""

from __future__ import annotations

from dataclasses import dataclass

import moderngl
import numpy as np

from engine.core import mathx
from engine.core.log import get_logger
from engine.graphics import atlas, shaders
from engine.graphics.font import FontAtlas
from engine.graphics.mesher import ChunkMeshData
from engine.world.blocks import BlockRegistry
from engine.world.coords import CHUNK_X, CHUNK_Y, CHUNK_Z
from engine.world.environment import Environment

_log = get_logger("renderer")


@dataclass
class _Stream:
    vbo: moderngl.Buffer
    vao: moderngl.VertexArray
    count: int

    def release(self) -> None:
        self.vao.release()
        self.vbo.release()


class _ChunkGpu:
    __slots__ = ("origin", "center", "opaque", "cutout", "water")

    def __init__(self, cx: int, cz: int) -> None:
        ox, oz = cx * CHUNK_X, cz * CHUNK_Z
        self.origin = (float(ox), 0.0, float(oz))
        self.center = np.array(
            [ox + CHUNK_X / 2, CHUNK_Y / 2, oz + CHUNK_Z / 2], dtype=np.float64
        )
        self.opaque: _Stream | None = None
        self.cutout: _Stream | None = None
        self.water: _Stream | None = None

    def release(self) -> None:
        for stream in (self.opaque, self.cutout, self.water):
            if stream is not None:
                stream.release()
        self.opaque = self.cutout = self.water = None


_CHUNK_HALF_EXTENTS = np.array([CHUNK_X / 2, CHUNK_Y / 2, CHUNK_Z / 2])


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


class Renderer:
    def __init__(
        self,
        ctx: moderngl.Context,
        registry: BlockRegistry,
        font_atlas: FontAtlas | None = None,
    ) -> None:
        self.ctx = ctx
        self.registry = registry
        self._chunks: dict[tuple[int, int], _ChunkGpu] = {}

        self.prog_chunk = ctx.program(shaders.CHUNK_VERT, shaders.CHUNK_FRAG)
        self.prog_water = ctx.program(shaders.WATER_VERT, shaders.WATER_FRAG)
        self.prog_sky = ctx.program(shaders.SKY_VERT, shaders.SKY_FRAG)
        self.prog_lines = ctx.program(shaders.LINES_VERT, shaders.LINES_FRAG)
        self.prog_ui_color = ctx.program(shaders.UI_COLOR_VERT, shaders.UI_COLOR_FRAG)
        self.prog_ui_text = ctx.program(shaders.UI_TEXT_VERT, shaders.UI_TEXT_FRAG)
        self.prog_ui_block = ctx.program(shaders.UI_BLOCK_VERT, shaders.UI_BLOCK_FRAG)

        # Procedural tile texture array; layer ids are baked into the registry
        # so the mesher can emit them directly into vertex data.
        tiles, mapping = atlas.build_tiles(registry.required_tiles())
        registry.assign_texture_layers(mapping)
        self.tiles = ctx.texture_array(
            (atlas.TILE, atlas.TILE, tiles.shape[0]), 4, tiles.tobytes()
        )
        self.tiles.build_mipmaps()
        self.tiles.filter = (moderngl.NEAREST_MIPMAP_LINEAR, moderngl.NEAREST)
        try:
            self.tiles.anisotropy = 8.0
        except Exception:  # noqa: BLE001 - optional GPU feature
            pass

        self._sky_vao = ctx.vertex_array(self.prog_sky, [])
        self._box_vbo = ctx.buffer(_unit_cube_edges().tobytes())
        self._box_vao = ctx.vertex_array(
            self.prog_lines, [(self._box_vbo, "3f4", "in_pos")]
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

        self._set(self.prog_water, "u_alpha", 0.62)
        _log.info("Renderer ready (%d tile layers)", tiles.shape[0])

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
    def _make_stream(self, prog: moderngl.Program, data: np.ndarray | None) -> _Stream | None:
        if data is None or len(data) == 0:
            return None
        vbo = self.ctx.buffer(data.tobytes())
        vao = self.ctx.vertex_array(prog, [(vbo, "2u4", "in_data")])
        return _Stream(vbo, vao, len(data) // 2)

    def upload_chunk(self, cx: int, cz: int, mesh: ChunkMeshData) -> None:
        key = (cx, cz)
        old = self._chunks.pop(key, None)
        if old is not None:
            old.release()
        gpu = _ChunkGpu(cx, cz)
        gpu.opaque = self._make_stream(self.prog_chunk, mesh.opaque)
        gpu.cutout = self._make_stream(self.prog_chunk, mesh.cutout)
        gpu.water = self._make_stream(self.prog_water, mesh.water)
        self._chunks[key] = gpu

    def unload_chunk(self, cx: int, cz: int) -> None:
        gpu = self._chunks.pop((cx, cz), None)
        if gpu is not None:
            gpu.release()

    # -- frame ----------------------------------------------------------------------
    def resize(self, width: int, height: int) -> None:
        self.ctx.viewport = (0, 0, width, height)

    def render_world(
        self,
        camera,
        env: Environment,
        time_s: float,
        fog_start: float,
        fog_end: float,
        fog_color: tuple[float, float, float] | None = None,
        highlight: tuple[int, int, int] | None = None,
    ) -> dict[str, int]:
        ctx = self.ctx
        view_proj = camera.view_proj()
        fog = tuple(fog_color if fog_color is not None else env.fog_color)

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
        keys = list(self._chunks.keys())
        stats = {"chunks_loaded": len(keys), "chunks_visible": 0, "vertices": 0}
        if not keys:
            return stats
        centers = np.array([self._chunks[k].center for k in keys])
        planes = camera.frustum_planes()
        visible_mask = mathx.aabbs_in_frustum(planes, centers, _CHUNK_HALF_EXTENTS)
        visible_idx = np.nonzero(visible_mask)[0]
        if len(visible_idx) == 0:
            return stats
        dist2 = np.sum((centers[visible_idx] - camera.position) ** 2, axis=1)
        order = visible_idx[np.argsort(dist2)]  # front-to-back
        visible = [self._chunks[keys[i]] for i in order]
        stats["chunks_visible"] = len(visible)

        # -- opaque ------------------------------------------------------------
        ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.tiles.use(location=0)
        prog = self.prog_chunk
        self._set(prog, "u_tiles", 0)
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
                gpu.opaque.vao.render(moderngl.TRIANGLES)
                stats["vertices"] += gpu.opaque.count

        # -- cutout (leaves, glass, plants): both faces visible ---------------------
        ctx.disable(moderngl.CULL_FACE)
        self._set(prog, "u_alpha_test", True)
        for gpu in visible:
            if gpu.cutout is not None:
                origin_uniform.value = gpu.origin
                gpu.cutout.vao.render(moderngl.TRIANGLES)
                stats["vertices"] += gpu.cutout.count

        # -- water (translucent): back-to-front --------------------------------------
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        prog = self.prog_water
        self._set(prog, "u_tiles", 0)
        self._set_mat(prog, "u_view_proj", view_proj)
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
                gpu.water.vao.render(moderngl.TRIANGLES)
                stats["vertices"] += gpu.water.count

        # -- targeted block outline ---------------------------------------------------
        if highlight is not None:
            self._set_mat(self.prog_lines, "u_view_proj", view_proj)
            self._set(self.prog_lines, "u_offset", tuple(float(c) for c in highlight))
            self._set(self.prog_lines, "u_color", (0.0, 0.0, 0.0, 0.7))
            self._box_vao.render(moderngl.LINES)

        ctx.disable(moderngl.BLEND)
        return stats

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

    def release(self) -> None:
        for gpu in self._chunks.values():
            gpu.release()
        self._chunks.clear()
