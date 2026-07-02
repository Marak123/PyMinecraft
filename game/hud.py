"""HUD: crosshair, hotbar, debug overlay, pause screen.

Pure vertex-data assembly — all actual drawing goes through the renderer's
UI helpers, so this module owns zero GL objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from engine.graphics.font import FontAtlas, layout_text
from engine.graphics.renderer import Renderer
from engine.world.blocks import BlockRegistry

_SLOT = 46
_ICON = 32


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

    # -- main entry --------------------------------------------------------------
    def draw(self, width: int, height: int, state: HudState) -> None:
        self.renderer.begin_ui(width, height)

        if state.underwater:
            self.renderer.draw_ui_rects(
                _rect(0, 0, width, height, (0.05, 0.18, 0.45, 0.28))
            )

        self._draw_crosshair(width, height)
        self._draw_hotbar(width, height, state)
        if state.debug_visible:
            self._draw_debug(state)
        if state.paused:
            self._draw_pause(width, height)

    # -- pieces ---------------------------------------------------------------
    def _draw_crosshair(self, width: int, height: int) -> None:
        cx, cy = width / 2, height / 2
        color = (1.0, 1.0, 1.0, 0.8)
        rects = np.concatenate(
            [_rect(cx - 8, cy - 1, 16, 2, color), _rect(cx - 1, cy - 8, 2, 16, color)]
        )
        self.renderer.draw_ui_rects(rects)

    def _draw_hotbar(self, width: int, height: int, state: HudState) -> None:
        n = len(self.hotbar_ids)
        x0 = width / 2 - (n * _SLOT) / 2
        y0 = height - _SLOT - 8

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

        if state.hand_label:
            label_w = len(state.hand_label) * self.font.cell_w
            self._text(width / 2 - label_w / 2, y0 - 26, state.hand_label)

    def _draw_debug(self, state: HudState) -> None:
        s = state.stats
        x, y, z = state.position
        lines = [
            f"PyMinecraft dev | FPS {state.fps:5.0f} | {state.frame_ms:5.1f} ms"
            + (" | FLY" if state.flying else ""),
            f"XYZ: {x:8.2f} / {y:6.2f} / {z:8.2f}   chunk: {state.chunk[0]}, {state.chunk[1]}",
            f"chunks: {s.get('loaded', 0)} loaded, {s.get('chunks_visible', 0)} visible"
            f" | jobs: gen {s.get('pending_gen', 0)}, mesh {s.get('pending_mesh', 0)}",
            f"verts: {s.get('vertices', 0) / 1e6:.2f}M   day: {state.time_of_day:.2f}"
            f"   seed: {state.seed}",
        ]
        for i, line in enumerate(lines):
            self._text(8, 8 + i * (self.font.cell_h + 2), line)

    def _draw_pause(self, width: int, height: int) -> None:
        self.renderer.draw_ui_rects(_rect(0, 0, width, height, (0.0, 0.0, 0.0, 0.55)))
        title = "PAUSED"
        hint = "Press ESC or click to resume"
        tw = len(title) * self.font.cell_w * 2
        hw = len(hint) * self.font.cell_w
        self._text(width / 2 - tw / 2, height / 2 - 40, title, scale=2.0)
        self._text(width / 2 - hw / 2, height / 2 + 8, hint, color=(0.9, 0.9, 0.9, 1.0))
