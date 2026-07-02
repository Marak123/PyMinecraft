"""Bitmap font atlas rasterised at startup with Pillow.

Uses the system monospace font (Consolas on Windows) so we get crisp debug
text without shipping font assets.  Glyphs for ASCII 32..126 are packed into
a single-channel grid texture; text is drawn as one quad batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from engine.core.log import get_logger

_log = get_logger("font")

_FIRST = 32
_LAST = 126
_COLS = 16

_FONT_CANDIDATES = (
    "C:/Windows/Fonts/consola.ttf",
    "C:/Windows/Fonts/cour.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
)


@dataclass
class FontAtlas:
    image: np.ndarray  # (H, W) uint8 alpha
    cell_w: int
    cell_h: int
    uv_scale: tuple[float, float]  # cell size in UV space

    def glyph_uv(self, char: str) -> tuple[float, float]:
        """UV origin of a glyph cell (top-left)."""
        code = ord(char)
        if not (_FIRST <= code <= _LAST):
            code = ord("?")
        i = code - _FIRST
        col = i % _COLS
        row = i // _COLS
        return col * self.uv_scale[0], row * self.uv_scale[1]


def build_font_atlas(size: int = 18) -> FontAtlas:
    font = None
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).exists():
            font = ImageFont.truetype(candidate, size)
            break
    if font is None:
        _log.warning("No system monospace font found; using PIL fallback")
        font = ImageFont.load_default()

    ascent, descent = font.getmetrics()
    cell_h = ascent + descent
    cell_w = max(
        int(font.getlength(chr(c))) for c in range(_FIRST, _LAST + 1)
    ) or size // 2

    rows = (_LAST - _FIRST) // _COLS + 1
    img = Image.new("L", (cell_w * _COLS, cell_h * rows), 0)
    draw = ImageDraw.Draw(img)
    for code in range(_FIRST, _LAST + 1):
        i = code - _FIRST
        draw.text(((i % _COLS) * cell_w, (i // _COLS) * cell_h), chr(code), fill=255, font=font)

    atlas = np.asarray(img, dtype=np.uint8)
    _log.info("Font atlas %dx%d (cell %dx%d)", img.width, img.height, cell_w, cell_h)
    return FontAtlas(
        image=atlas,
        cell_w=cell_w,
        cell_h=cell_h,
        uv_scale=(cell_w / img.width, cell_h / img.height),
    )


def layout_text(
    atlas: FontAtlas, x: float, y: float, text: str, scale: float = 1.0
) -> np.ndarray:
    """Build (x, y, u, v) vertices (6 per glyph) for a text run at top-left (x, y)."""
    w = atlas.cell_w * scale
    h = atlas.cell_h * scale
    du, dv = atlas.uv_scale
    verts = np.empty((len(text), 6, 4), dtype=np.float32)
    pen = x
    for i, ch in enumerate(text):
        u, v = atlas.glyph_uv(ch)
        x0, y0, x1, y1 = pen, y, pen + w, y + h
        u0, v0, u1, v1 = u, v, u + du, v + dv
        verts[i] = [
            (x0, y0, u0, v0), (x1, y0, u1, v0), (x1, y1, u1, v1),
            (x0, y0, u0, v0), (x1, y1, u1, v1), (x0, y1, u0, v1),
        ]
        pen += w
    return verts.reshape(-1, 4)
