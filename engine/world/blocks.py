"""Data-driven block registry.

Blocks are *definitions*, not classes (see design manifesto).  The registry
loads ``configs/blocks.json`` and bakes the properties into flat NumPy lookup
tables indexed by block id, because the mesher and the generator consume them
in bulk vectorised operations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from engine.core.log import get_logger

_log = get_logger("blocks")

# Render types (values are baked into the RENDER lookup table).
RENDER_NONE = 0
RENDER_SOLID = 1
RENDER_CUTOUT = 2
RENDER_LIQUID = 3
RENDER_CROSS = 4

_RENDER_BY_NAME = {
    "none": RENDER_NONE,
    "solid": RENDER_SOLID,
    "cutout": RENDER_CUTOUT,
    "liquid": RENDER_LIQUID,
    "cross": RENDER_CROSS,
}

# Face order used across mesher/shaders/registry: +X -X +Y -Y +Z -Z
FACE_TEXTURE_KEYS = ("side", "side", "top", "bottom", "side", "side")

AIR = 0


@dataclass
class BlockDef:
    id: int
    name: str
    label: str
    render: int
    solid: bool
    opaque: bool
    breakable: bool = True
    replaceable: bool = False
    emission: int = 0
    textures: dict[str, str] = field(default_factory=dict)

    def face_tile(self, face: int) -> str | None:
        """Tile name for a face index (see FACE_TEXTURE_KEYS order)."""
        if not self.textures:
            return None
        key = FACE_TEXTURE_KEYS[face]
        return self.textures.get(key) or self.textures.get("all")


class BlockRegistry:
    """Loads block definitions and exposes vectorisable lookup tables."""

    def __init__(self, defs: list[BlockDef]) -> None:
        self.defs = defs
        self.by_name: dict[str, BlockDef] = {d.name: d for d in defs}
        self.by_id: dict[int, BlockDef] = {d.id: d for d in defs}

        n = max(d.id for d in defs) + 1
        self.opaque = np.zeros(n, dtype=bool)
        self.solid = np.zeros(n, dtype=bool)
        self.render = np.zeros(n, dtype=np.uint8)
        self.emission = np.zeros(n, dtype=np.uint8)
        self.replaceable = np.zeros(n, dtype=bool)
        # Texture layer per (block, face); filled by assign_texture_layers().
        self.face_layers = np.zeros((n, 6), dtype=np.uint16)

        for d in defs:
            self.opaque[d.id] = d.opaque
            self.solid[d.id] = d.solid
            self.render[d.id] = d.render
            self.emission[d.id] = d.emission
            self.replaceable[d.id] = d.replaceable

    @classmethod
    def load(cls, path: Path) -> "BlockRegistry":
        data = json.loads(path.read_text(encoding="utf-8"))
        defs: list[BlockDef] = []
        for entry in data["blocks"]:
            defs.append(
                BlockDef(
                    id=int(entry["id"]),
                    name=entry["name"],
                    label=entry.get("label", entry["name"]),
                    render=_RENDER_BY_NAME[entry.get("render", "solid")],
                    solid=bool(entry.get("solid", True)),
                    opaque=bool(entry.get("opaque", True)),
                    breakable=bool(entry.get("breakable", True)),
                    replaceable=bool(entry.get("replaceable", False)),
                    emission=int(entry.get("emission", 0)),
                    textures=dict(entry.get("textures", {})),
                )
            )
        ids = [d.id for d in defs]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate block ids in blocks.json")
        _log.info("Loaded %d block definitions from %s", len(defs), path.name)
        return cls(defs)

    def required_tiles(self) -> list[str]:
        """All tile names referenced by any block face (atlas build input)."""
        tiles: list[str] = []
        for d in self.defs:
            for face in range(6):
                tile = d.face_tile(face)
                if tile and tile not in tiles:
                    tiles.append(tile)
        return tiles

    def assign_texture_layers(self, layer_of_tile: dict[str, int]) -> None:
        """Bake tile-name -> texture-array-layer mapping into face_layers."""
        for d in self.defs:
            for face in range(6):
                tile = d.face_tile(face)
                if tile is not None:
                    if tile not in layer_of_tile:
                        raise KeyError(f"Block '{d.name}' references unknown tile '{tile}'")
                    self.face_layers[d.id, face] = layer_of_tile[tile]

    def id_of(self, name: str) -> int:
        return self.by_name[name].id
