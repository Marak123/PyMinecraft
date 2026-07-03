"""HUD: crosshair, hotbar, hearts, air bubbles, dig progress, debug overlay.

Pure vertex-data assembly — all actual drawing goes through the renderer's
UI helpers, so this module owns zero GL objects.  Hearts and bubbles are
tiles from the same procedural texture array the blocks use.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from engine.graphics.font import FontAtlas, layout_text
from engine.graphics.renderer import Renderer
from engine.world.blocks import BlockRegistry

_SLOT = 46
_ICON = 32
_HEART = 18


@dataclass
class HudState:
    fps: float = 0.0
    frame_ms: float = 0.0
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    chunk: tuple[int, int] = (0, 0)
    stats: dict = field(default_factory=dict)
    selected_slot: int = 0
    debug_visible: bool = False
    paused: bool = False
    underwater: bool = False
    time_of_day: float = 0.0
    seed: int = 0
    hand_label: str = ""
    flying: bool = False
    mode: str = "survival"
    health: float = 20.0
    max_health: float = 20.0
    air: float = 10.0
    max_air: float = 10.0
    breaking: float | None = None
    damage_flash: float = 0.0
    dead: bool = False


def _rect(x: float, y: float, w: float, h: float, color: tuple[float, float, float, float]) -> np.ndarray:
    r, g, b, a = color
    x1, y1 = x + w, y + h
    return np.array(
        [
            (x, y, r, g, b, a), (x1, y, r, g, b, a), (x1, y1, r, g, b, a),
            (x, y, r, g, b, a), (x1, y1, r, g, b, a), (x, y1, r, g, b, a),
        ],
        dtype=np.float32,
    )


def _icon_quad(x: float, y: float, size: float, layer: int) -> np.ndarray:
    x1, y1 = x + size, y + size
    lay = float(layer)
    return np.array(
        [
            (x, y, 0.0, 0.0, lay), (x1, y, 1.0, 0.0, lay), (x1, y1, 1.0, 1.0, lay),
            (x, y, 0.0, 0.0, lay), (x1, y1, 1.0, 1.0, lay), (x, y1, 0.0, 1.0, lay),
        ],
        dtype=np.float32,
    )


class Hud:
    def __init__(
        self,
        renderer: Renderer,
        registry: BlockRegistry,
        font_atlas: FontAtlas,
        hotbar_ids: list[int],
    ) -> None:
        self.renderer = renderer
        self.registry = registry
        self.font = font_atlas
        self.hotbar_ids = hotbar_ids

    # -- text helper with drop shadow --------------------------------------------
    def _text(self, x: float, y: float, text: str, scale: float = 1.0,
              color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)) -> None:
        shadow = layout_text(self.font, x + 1, y + 1, text, scale)
        self.renderer.draw_ui_text(shadow, (0.0, 0.0, 0.0, 0.8 * color[3]))
        main = layout_text(self.font, x, y, text, scale)
        self.renderer.draw_ui_text(main, color)

    def _text_centered(self, cx: float, y: float, text: str, scale: float = 1.0,
                       color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)) -> None:
        w = len(text) * self.font.cell_w * scale
        self._text(cx - w / 2, y, text, scale, color)

    # -- main entry --------------------------------------------------------------
    def draw(self, width: int, height: int, state: HudState) -> None:
        self.renderer.begin_ui(width, height)

        if state.underwater:
            self.renderer.draw_ui_rects(
                _rect(0, 0, width, height, (0.05, 0.18, 0.45, 0.28))
            )
        if state.damage_flash > 0.0:
            alpha = min(state.damage_flash / 0.4, 1.0) * 0.34
            self.renderer.draw_ui_rects(
                _rect(0, 0, width, height, (0.75, 0.05, 0.05, alpha))
            )

        self._draw_crosshair(width, height)
        if state.breaking is not None:
            self._draw_break_progress(width, height, state.breaking)
        self._draw_hotbar(width, height, state)
        if state.mode == "survival":
            self._draw_vitals(width, height, state)
        if state.debug_visible:
            self._draw_debug(state)
        if state.dead:
            self._draw_death(width, height)
        elif state.paused:
            self._draw_pause(width, height)

    # -- pieces ---------------------------------------------------------------
    def _draw_crosshair(self, width: int, height: int) -> None:
        cx, cy = width / 2, height / 2
        color = (1.0, 1.0, 1.0, 0.8)
        rects = np.concatenate(
            [_rect(cx - 8, cy - 1, 16, 2, color), _rect(cx - 1, cy - 8, 2, 16, color)]
        )
        self.renderer.draw_ui_rects(rects)

    def _draw_break_progress(self, width: int, height: int, progress: float) -> None:
        cx, cy = width / 2, height / 2
        bar_w = 64.0
        rects = np.concatenate([
            _rect(cx - bar_w / 2, cy + 18, bar_w, 6, (0.0, 0.0, 0.0, 0.55)),
            _rect(cx - bar_w / 2 + 1, cy + 19, (bar_w - 2) * min(progress, 1.0), 4,
                  (0.95, 0.95, 0.95, 0.9)),
        ])
        self.renderer.draw_ui_rects(rects)

    def _hotbar_origin(self, width: int, height: int) -> tuple[float, float]:
        n = len(self.hotbar_ids)
        return width / 2 - (n * _SLOT) / 2, height - _SLOT - 8

    def _draw_hotbar(self, width: int, height: int, state: HudState) -> None:
        x0, y0 = self._hotbar_origin(width, height)

        rects = []
        icons = []
        for i, block_id in enumerate(self.hotbar_ids):
            x = x0 + i * _SLOT
            if i == state.selected_slot:
                rects.append(_rect(x - 3, y0 - 3, _SLOT + 6, _SLOT + 6, (1, 1, 1, 0.85)))
            rects.append(_rect(x, y0, _SLOT - 2, _SLOT - 2, (0.08, 0.08, 0.08, 0.62)))
            # Icon uses the block's +Z side texture (face index 4).
            layer = int(self.registry.face_layers[block_id, 4])
            pad = (_SLOT - 2 - _ICON) / 2
            icons.append(_icon_quad(x + pad, y0 + pad, _ICON, layer))

        self.renderer.draw_ui_rects(np.concatenate(rects))
        self.renderer.draw_ui_blocks(np.concatenate(icons))

        label = state.hand_label or (
            "CREATIVE (F4)" if state.mode == "creative" and state.flying else ""
        )
        if label:
            self._text_centered(width / 2, y0 - 26, label)

    def _draw_vitals(self, width: int, height: int, state: HudState) -> None:
        x0, y0 = self._hotbar_origin(width, height)
        icons = []

        # Hearts: 10 icons, 2 hp each.
        hearts_y = y0 - _HEART - 6
        full_layer = self.renderer.tile_layer("heart_full")
        half_layer = self.renderer.tile_layer("heart_half")
        empty_layer = self.renderer.tile_layer("heart_empty")
        for i in range(10):
            hp = state.health - i * 2
            layer = full_layer if hp >= 2 else (half_layer if hp >= 1 else empty_layer)
            icons.append(_icon_quad(x0 + i * (_HEART + 2), hearts_y, _HEART, layer))

        # Air bubbles appear only while diving, right-aligned over the hotbar.
        if state.air < state.max_air - 1e-3:
            bubble_layer = self.renderer.tile_layer("bubble")
            bubbles = int(np.ceil(state.air))
            n = len(self.hotbar_ids)
            bx1 = x0 + n * _SLOT - _HEART
            for i in range(bubbles):
                icons.append(
                    _icon_quad(bx1 - i * (_HEART + 2), hearts_y, _HEART, bubble_layer)
                )

        if icons:
            self.renderer.draw_ui_blocks(np.concatenate(icons))

    def _draw_debug(self, state: HudState) -> None:
        s = state.stats
        x, y, z = state.position
        lines = [
            f"PyMinecraft dev | FPS {state.fps:5.0f} | {state.frame_ms:5.1f} ms"
            f" | {state.mode.upper()}" + (" FLY" if state.flying else ""),
            f"XYZ: {x:8.2f} / {y:6.2f} / {z:8.2f}   chunk: {state.chunk[0]}, {state.chunk[1]}",
            f"chunks: {s.get('loaded', 0)} loaded, {s.get('chunks_visible', 0)} visible"
            f" | jobs: gen {s.get('pending_gen', 0)}, light {s.get('pending_light', 0)},"
            f" mesh {s.get('pending_mesh', 0)}",
            f"verts: {s.get('vertices', 0) / 1e6:.2f}M   day: {state.time_of_day:.2f}"
            f"   seed: {state.seed}",
            f"ms: update {s.get('ms_update', 0.0):4.1f} | stream {s.get('ms_stream', 0.0):4.1f}"
            f" | render {s.get('ms_render', 0.0):4.1f}",
        ]
        for i, line in enumerate(lines):
            self._text(8, 8 + i * (self.font.cell_h + 2), line)

    def _draw_pause(self, width: int, height: int) -> None:
        self.renderer.draw_ui_rects(_rect(0, 0, width, height, (0.0, 0.0, 0.0, 0.55)))
        self._text_centered(width / 2, height / 2 - 40, "PAUSED", scale=2.0)
        self._text_centered(
            width / 2, height / 2 + 8, "Press ESC or click to resume",
            color=(0.9, 0.9, 0.9, 1.0),
        )

    def _draw_death(self, width: int, height: int) -> None:
        self.renderer.draw_ui_rects(_rect(0, 0, width, height, (0.35, 0.0, 0.0, 0.5)))
        self._text_centered(width / 2, height / 2 - 40, "YOU DIED", scale=2.0)
        self._text_centered(
            width / 2, height / 2 + 8, "Respawning...", color=(0.95, 0.85, 0.85, 1.0)
        )
