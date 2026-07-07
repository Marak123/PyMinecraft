"""Framebuffer management for the multi-pass pipeline (plan phase 1).

Owns the HDR scene target (RGBA16F — lets emissive values exceed 1.0 for
future bloom) and the cascaded shadow-map depth targets.  Everything is
recreated on window resize.
"""

from __future__ import annotations

import moderngl

from engine.core.log import get_logger

_log = get_logger("fbo")

SHADOW_CASCADES = 3


class SceneTargets:
    def __init__(self, ctx: moderngl.Context, width: int, height: int,
                 shadow_resolution: int = 1536) -> None:
        self.ctx = ctx
        self.width = 0
        self.height = 0
        self.hdr_color: moderngl.Texture | None = None
        self.hdr_depth: moderngl.Texture | None = None
        self.hdr_fbo: moderngl.Framebuffer | None = None
        self.resize(width, height)

        # One depth texture + FBO per shadow cascade.  Hardware PCF via
        # depth-compare samplers (sampler2DShadow in GLSL).
        self.shadow_resolution = shadow_resolution
        self.shadow_maps: list[moderngl.Texture] = []
        self.shadow_fbos: list[moderngl.Framebuffer] = []
        for _ in range(SHADOW_CASCADES):
            tex = ctx.depth_texture((shadow_resolution, shadow_resolution))
            tex.compare_func = "<="
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = tex.repeat_y = False
            self.shadow_maps.append(tex)
            self.shadow_fbos.append(ctx.framebuffer(depth_attachment=tex))
        _log.info("Scene targets ready (HDR %dx%d, %d shadow cascades @ %d)",
                  width, height, SHADOW_CASCADES, shadow_resolution)

    def resize(self, width: int, height: int) -> None:
        if width == self.width and height == self.height or width == 0 or height == 0:
            return
        for res in (self.hdr_fbo, self.hdr_color, self.hdr_depth):
            if res is not None:
                res.release()
        for tex, fbo in getattr(self, "bloom", []):
            tex.release()
            fbo.release()
        self.width, self.height = width, height
        self.hdr_color = self.ctx.texture((width, height), 4, dtype="f2")
        self.hdr_color.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.hdr_depth = self.ctx.depth_texture((width, height))
        self.hdr_fbo = self.ctx.framebuffer(
            color_attachments=[self.hdr_color], depth_attachment=self.hdr_depth
        )
        # Bloom pyramid: 1/2, 1/4, 1/8 resolution HDR targets (plan phase 2).
        self.bloom: list[tuple[moderngl.Texture, moderngl.Framebuffer]] = []
        for divisor in (2, 4, 8):
            w, h = max(width // divisor, 1), max(height // divisor, 1)
            tex = self.ctx.texture((w, h), 4, dtype="f2")
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = tex.repeat_y = False
            self.bloom.append((tex, self.ctx.framebuffer(color_attachments=[tex])))

    def release(self) -> None:
        for tex, fbo in getattr(self, "bloom", []):
            tex.release()
            fbo.release()
        for res in (self.hdr_fbo, self.hdr_color, self.hdr_depth,
                    *self.shadow_fbos, *self.shadow_maps):
            if res is not None:
                res.release()
