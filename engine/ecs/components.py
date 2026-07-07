"""ECS component types (plan phase 7.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Transform:
    position: np.ndarray
    yaw: float = 0.0
    pitch: float = 0.0


@dataclass
class Velocity:
    value: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class Collider:
    half_width: float
    height: float
    on_ground: bool = False


@dataclass
class Health:
    current: float
    maximum: float
    hurt_timer: float = 0.0      # red flash / i-frames
    death_timer: float = 0.0     # falls over then despawns


@dataclass
class MobAI:
    mob_type: str
    hostile: bool
    speed: float
    blackboard: dict = field(default_factory=dict)
    think_timer: float = 0.0
    walk_cycle: float = 0.0
    attack_cooldown: float = 0.0


@dataclass
class Drops:
    table: list  # [(block_name, min, max), ...]


@dataclass
class DroppedItem:
    block_id: int
    count: int
    pickup_delay: float = 0.5
    age: float = 0.0
    bob: float = 0.0


@dataclass
class NetherFuse:
    """Creeper priming state."""
    timer: float = 0.0
    lit: bool = False
