"""Interactive UI screens: inventory / creative picker / settings.

Immediate-mode style: every frame the screen lays out widgets, tests the
mouse against them, and batches draws in strict order (rects -> icons ->
text) so nothing overdraws its own labels.  No GL objects owned here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from engine.graphics.font import FontAtlas, layout_text
from engine.graphics.renderer import Renderer
from engine.world.blocks import RENDER_NONE, BlockRegistry
from game.inventory import HOTBAR_SLOTS, CraftingBook, Inventory

_SLOT = 44
_ICON = 32
_PANEL_W, _PANEL_H = 640, 400


@dataclass
class Mouse:
    x: float
    y: float
    clicked: bool  # LMB pressed this frame


@dataclass
class _Batch:
    rects: list = field(default_factory=list)
    icons: list = field(default_factory=list)
    texts: list = field(default_factory=list)  # (x, y, s, scale, color)


def _rect(x, y, w, h, color) -> np.ndarray:
    r, g, b, a = color
    x1, y1 = x + w, y + h
    return np.array(
        [(x, y, r, g, b, a), (x1, y, r, g, b, a), (x1, y1, r, g, b, a),
         (x, y, r, g, b, a), (x1, y1, r, g, b, a), (x, y1, r, g, b, a)],
        dtype=np.float32,
    )


def _icon(x, y, size, layer) -> np.ndarray:
    x1, y1 = x + size, y + size
    lay = float(layer)
    return np.array(
        [(x, y, 0, 0, lay), (x1, y, 1, 0, lay), (x1, y1, 1, 1, lay),
         (x, y, 0, 0, lay), (x1, y1, 1, 1, lay), (x, y1, 0, 1, lay)],
        dtype=np.float32,
    )


def _hit(mouse: Mouse, x, y, w, h) -> bool:
    return x <= mouse.x < x + w and y <= mouse.y < y + h


class _UiBase:
    def __init__(self, renderer: Renderer, font: FontAtlas) -> None:
        self.renderer = renderer
        self.font = font

    def _flush(self, batch: _Batch) -> None:
        if batch.rects:
            self.renderer.draw_ui_rects(np.concatenate(batch.rects))
        if batch.icons:
            self.renderer.draw_ui_blocks(np.concatenate(batch.icons))
        for x, y, s, scale, color in batch.texts:
            self.renderer.draw_ui_text(
                layout_text(self.font, x + 1, y + 1, s, scale), (0, 0, 0, 0.8 * color[3])
            )
            self.renderer.draw_ui_text(layout_text(self.font, x, y, s, scale), color)

    @staticmethod
    def _panel(batch: _Batch, x, y, w, h, title: str) -> None:
        batch.rects.append(_rect(x - 2, y - 2, w + 4, h + 4, (0.85, 0.85, 0.9, 0.35)))
        batch.rects.append(_rect(x, y, w, h, (0.07, 0.07, 0.09, 0.94)))
        batch.texts.append((x + 12, y + 10, title, 1.25, (1, 1, 1, 1)))


class InventoryScreen(_UiBase):
    """Survival: 36-slot inventory + recipe list.  Creative: all-block picker."""

    def __init__(self, renderer, font, registry: BlockRegistry,
                 inventory: Inventory, book: CraftingBook) -> None:
        super().__init__(renderer, font)
        self.registry = registry
        self.inventory = inventory
        self.book = book
        self.swap_source: int | None = None
        # Everything selectable in the creative picker: placeable blocks plus
        # item-only entries (tools, food) that have an icon.
        self.placeable = [
            d.id for d in registry.defs
            if d.name not in ("air", "bedrock") and d.textures
        ]

    def _slot_icon(self, block_id: int) -> int:
        return int(self.registry.face_layers[block_id, 4])

    def _draw_slot(self, batch: _Batch, x, y, entry, *, selected=False, source=False) -> None:
        if selected or source:
            border = (1.0, 0.85, 0.2, 0.95) if source else (1.0, 1.0, 1.0, 0.9)
            batch.rects.append(_rect(x - 2, y - 2, _SLOT, _SLOT, border))
        batch.rects.append(_rect(x, y, _SLOT - 4, _SLOT - 4, (0.16, 0.16, 0.19, 0.9)))
        if entry:
            pad = (_SLOT - 4 - _ICON) / 2
            batch.icons.append(_icon(x + pad, y + pad, _ICON, self._slot_icon(entry[0])))
            if entry[1] > 1:
                s = str(entry[1])
                batch.texts.append(
                    (x + _SLOT - 10 - len(s) * self.font.cell_w * 0.8, y + _SLOT - 20, s, 0.8, (1, 1, 1, 1))
                )

    def update(self, width: int, height: int, mouse: Mouse, *,
               creative: bool, selected_slot: int) -> int:
        """Draw + interact. Returns the (possibly changed) selected hotbar slot."""
        px, py = (width - _PANEL_W) / 2, (height - _PANEL_H) / 2
        batch = _Batch()
        self._panel(batch, px, py, _PANEL_W, _PANEL_H, "CREATIVE BLOCKS" if creative else "INVENTORY")

        if creative:
            self._creative_grid(batch, px, py, mouse, selected_slot)
        else:
            self._survival_grid(batch, px, py, mouse)

        # Hotbar row (both modes) at the panel bottom.
        hb_y = py + _PANEL_H - _SLOT - 14
        for i in range(HOTBAR_SLOTS):
            x = px + 20 + i * _SLOT
            if mouse.clicked and _hit(mouse, x, hb_y, _SLOT - 4, _SLOT - 4):
                if not creative and self.swap_source is not None:
                    self.inventory.swap(self.swap_source, i)
                    self.swap_source = None
                else:
                    selected_slot = i
            self._draw_slot(batch, x, hb_y, self.inventory.slot(i), selected=(i == selected_slot))

        self._flush(batch)
        return selected_slot

    def _creative_grid(self, batch: _Batch, px, py, mouse: Mouse, selected_slot: int) -> None:
        cols = 12
        for idx, block_id in enumerate(self.placeable):
            x = px + 20 + (idx % cols) * _SLOT
            y = py + 46 + (idx // cols) * _SLOT
            if mouse.clicked and _hit(mouse, x, y, _SLOT - 4, _SLOT - 4):
                self.inventory.slots[selected_slot] = [block_id, 1]
            hovered = _hit(mouse, x, y, _SLOT - 4, _SLOT - 4)
            self._draw_slot(batch, x, y, (block_id, 1), selected=hovered)
            if hovered:
                batch.texts.append((px + 20, py + _PANEL_H - _SLOT - 44,
                                    self.registry.by_id[block_id].label, 1.0, (1, 1, 0.8, 1)))
        batch.texts.append((px + 320, py + 14, "click block -> selected slot", 0.8, (0.7, 0.7, 0.75, 1)))

    def _survival_grid(self, batch: _Batch, px, py, mouse: Mouse) -> None:
        # Backpack slots 9..35 in a 9x3 grid; click-click to swap stacks.
        for row in range(3):
            for col in range(9):
                index = 9 + row * 9 + col
                x = px + 20 + col * _SLOT
                y = py + 46 + row * _SLOT
                if mouse.clicked and _hit(mouse, x, y, _SLOT - 4, _SLOT - 4):
                    if self.swap_source is None:
                        if self.inventory.slot(index):
                            self.swap_source = index
                    else:
                        self.inventory.swap(self.swap_source, index)
                        self.swap_source = None
                self._draw_slot(batch, x, y, self.inventory.slot(index),
                                source=(self.swap_source == index))

        # Recipe list on the right; hover shows ingredients, click crafts.
        rx = px + 20 + 9 * _SLOT + 18
        batch.texts.append((rx, py + 46, "Crafting", 1.0, (0.92, 0.92, 0.6, 1)))
        ry = py + 72
        hint: str | None = None
        for recipe in self.book.recipes:
            ok = CraftingBook.can_craft(recipe, self.inventory)
            if mouse.clicked and ok and _hit(mouse, rx, ry, 200, 24):
                CraftingBook.craft(recipe, self.inventory)
            bg = (0.15, 0.20, 0.15, 0.9) if ok else (0.10, 0.10, 0.11, 0.6)
            batch.rects.append(_rect(rx, ry, 200, 24, bg))
            batch.icons.append(_icon(rx + 3, ry + 3, 18, self._slot_icon(recipe.output)))
            color = (1, 1, 1, 1) if ok else (0.55, 0.55, 0.55, 1)
            batch.texts.append((rx + 26, ry + 4, f"{recipe.count}x {recipe.label}", 0.8, color))
            if _hit(mouse, rx, ry, 200, 24):
                hint = "Needs: " + ", ".join(
                    f"{n}x {self.registry.by_id[b].label}"
                    for b, n in recipe.ingredients.items()
                )
            ry += 27
        if hint:
            batch.texts.append((px + 20, py + _PANEL_H - _SLOT - 44, hint, 0.8, (0.9, 0.9, 0.7, 1)))


class SettingsScreen(_UiBase):
    """Pause-menu settings: applied live by the game, saved to settings.json."""

    def update(self, width: int, height: int, mouse: Mouse, values: dict) -> str | None:
        """`values`: {key: (label, display, kind)} with kind 'step' | 'toggle'.
        Returns 'key+', 'key-' or 'key!' for the widget clicked this frame."""
        pw = 470
        ph = 100 + len(values) * 44
        px, py = (width - pw) / 2, (height - ph) / 2
        batch = _Batch()
        self._panel(batch, px, py, pw, ph, "PAUSED — SETTINGS")

        action: str | None = None
        y = py + 52
        for key, (label, display, kind) in values.items():
            batch.texts.append((px + 16, y + 8, label, 1.0, (1, 1, 1, 1)))
            if kind == "toggle":
                w = 84
                x = px + pw - w - 16
                on = display == "ON"
                bg = (0.24, 0.44, 0.24, 0.95) if on else (0.4, 0.2, 0.2, 0.95)
                batch.rects.append(_rect(x, y, w, 30, bg))
                batch.texts.append((x + w / 2 - len(display) * self.font.cell_w / 2, y + 7,
                                    display, 1.0, (1, 1, 1, 1)))
                if mouse.clicked and _hit(mouse, x, y, w, 30):
                    action = key + "!"
            else:
                for sign, dx in (("-", pw - 180), ("+", pw - 52)):
                    x = px + dx
                    hover = _hit(mouse, x, y, 36, 30)
                    bg = (0.34, 0.34, 0.42, 1.0) if hover else (0.22, 0.22, 0.27, 0.95)
                    batch.rects.append(_rect(x, y, 36, 30, bg))
                    batch.texts.append((x + 14, y + 7, sign, 1.0, (1, 1, 1, 1)))
                    if mouse.clicked and hover:
                        action = key + sign
                batch.texts.append((px + pw - 136, y + 8, display, 1.0, (0.9, 0.9, 0.95, 1)))
            y += 44
        batch.texts.append((px + 16, py + ph - 30, "ESC resume | E inventory | F11 fullscreen",
                            0.8, (0.7, 0.7, 0.75, 1)))
        self._flush(batch)
        return action
