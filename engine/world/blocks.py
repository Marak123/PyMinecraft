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
    hardness: float = 1.0  # seconds to break in survival mode
    light_attenuation: int = 0  # extra light decay per block entered
    needs_support: bool = False  # must stand on a solid block (plants, torch)
    drops: str | None = None  # block dropped when mined; None = itself, "none" = nothing
    fluid: dict | None = None  # {kind: water|lava, level: int, source: bool}
    placeable: bool = True     # False for pure items (tools, food)
    mined_by: str | None = None  # tool type that mines this fast
    min_tier: int = 0          # tool tier needed to get a drop
    tool: dict | None = None   # {type, tier, speed, damage}
    food: dict | None = None   # {hunger}
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
        self.hardness = np.ones(n, dtype=np.float32)
        self.light_attenuation = np.zeros(n, dtype=np.uint8)
        self.needs_support = np.zeros(n, dtype=bool)
        # Texture layer per (block, face); filled by assign_texture_layers().
        self.face_layers = np.zeros((n, 6), dtype=np.uint16)

        # Fluid tables: kind (0 none, 1 water, 2 lava), flow level (0 = source
        # or full), and (kind, level) -> block id for the simulator.
        self.fluid_kind = np.zeros(n, dtype=np.uint8)
        self.fluid_level = np.zeros(n, dtype=np.uint8)
        self.fluid_source = np.zeros(n, dtype=bool)
        self.fluid_by_level: dict[tuple[int, int], int] = {}
        self.max_flow = {1: 0, 2: 0}  # deepest flow level per kind
        for d in defs:
            if d.fluid:
                kind = 1 if d.fluid["kind"] == "water" else 2
                level = int(d.fluid.get("level", 0))
                self.fluid_kind[d.id] = kind
                self.fluid_level[d.id] = level
                self.fluid_source[d.id] = bool(d.fluid.get("source", False))
                self.fluid_by_level[(kind, level)] = d.id
                self.max_flow[kind] = max(self.max_flow[kind], level)

        # Item metadata (plan phase 8): placeability, tool power, food value.
        self.placeable = np.ones(n, dtype=bool)
        self.mined_by = ["" for _ in range(n)]
        self.min_tier = np.zeros(n, dtype=np.uint8)
        self.tool_type = ["" for _ in range(n)]
        self.tool_tier = np.zeros(n, dtype=np.int16)
        self.tool_speed = np.ones(n, dtype=np.float32)
        self.tool_damage = np.zeros(n, dtype=np.uint8)
        self.food_hunger = np.zeros(n, dtype=np.uint8)
        for d in defs:
            self.placeable[d.id] = d.placeable
            self.mined_by[d.id] = d.mined_by or ""
            self.min_tier[d.id] = d.min_tier
            if d.tool:
                self.tool_type[d.id] = d.tool["type"]
                self.tool_tier[d.id] = int(d.tool.get("tier", 0))
                self.tool_speed[d.id] = float(d.tool.get("speed", 1.0))
                self.tool_damage[d.id] = int(d.tool.get("damage", 1))
            if d.food:
                self.food_hunger[d.id] = int(d.food["hunger"])

        # What a block yields when mined (0 = nothing, AIR never drops).
        self.drops = np.zeros(n, dtype=np.uint8)
        for d in defs:
            self.opaque[d.id] = d.opaque
            self.solid[d.id] = d.solid
            self.render[d.id] = d.render
            self.emission[d.id] = d.emission
            self.replaceable[d.id] = d.replaceable
            self.hardness[d.id] = d.hardness
            self.light_attenuation[d.id] = d.light_attenuation
            self.needs_support[d.id] = d.needs_support
        for d in defs:
            if d.drops == "none":
                self.drops[d.id] = 0
            elif d.drops is None:
                self.drops[d.id] = d.id
            else:
                self.drops[d.id] = self.by_name[d.drops].id

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
                    hardness=float(entry.get("hardness", 1.0)),
                    light_attenuation=int(entry.get("light_attenuation", 0)),
                    needs_support=bool(entry.get("needs_support", False)),
                    drops=entry.get("drops"),
                    fluid=entry.get("fluid"),
                    placeable=bool(entry.get("placeable", True)),
                    mined_by=entry.get("mined_by"),
                    min_tier=int(entry.get("min_tier", 0)),
                    tool=entry.get("tool"),
                    food=entry.get("food"),
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
