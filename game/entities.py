"""Entities: wandering pigs and the third-person player model.

Deliberately pre-ECS: a handful of peaceful mobs with simple wander AI,
sharing the player's AABB physics.  The ECS refactor lands when entity
counts and component variety justify it (see ROADMAP).
"""

from __future__ import annotations

import math
import random
from math import floor

import numpy as np

from engine.graphics.boxrender import BoxRenderer, model_matrix
from engine.physics.aabb import move_with_collisions
from engine.world.coords import CHUNK_Y
from engine.world.world import World

_PIG_TINT = (0.93, 0.62, 0.65)
_PIG_DARK = (0.78, 0.48, 0.52)
_GRAVITY = -27.0


def entity_light(world: World, x: float, y: float, z: float, daylight: float) -> float:
    """Brightness for entity shading, sampled from the chunk light fields."""
    chunk = world.get_chunk(floor(x) >> 4, floor(z) >> 4)
    yy = min(max(int(y), 0), CHUNK_Y - 1)
    if chunk is None or not chunk.has_light:
        return max(0.25, daylight)
    lx, lz = floor(x) & 15, floor(z) & 15
    sky = float(chunk.sky_light[lx, lz, yy]) / 15.0
    blk = float(chunk.block_light[lx, lz, yy]) / 15.0
    return max(sky ** 1.5 * max(daylight, 0.06), blk ** 1.5, 0.1)


class Pig:
    HALF_W = 0.35
    HEIGHT = 0.85

    def __init__(self, position: np.ndarray) -> None:
        self.position = position.astype(np.float64)
        self.velocity = np.zeros(3)
        self.yaw = random.uniform(0.0, 360.0)
        self.on_ground = False
        self.hp = 6
        self.hurt_timer = 0.0
        self.dead = False
        self._think_timer = random.uniform(1.0, 3.0)
        self._walking = False
        self.walk_cycle = 0.0

    def hurt(self, amount: int, knockback: np.ndarray) -> None:
        self.hp -= amount
        self.hurt_timer = 0.35
        self.velocity[0] += knockback[0] * 6.0
        self.velocity[2] += knockback[2] * 6.0
        self.velocity[1] = 4.5
        if self.hp <= 0:
            self.dead = True

    def update(self, dt: float, world: World) -> None:
        self.hurt_timer = max(0.0, self.hurt_timer - dt)
        self._think_timer -= dt
        if self._think_timer <= 0.0:
            self._think_timer = random.uniform(1.5, 5.0)
            self._walking = random.random() < 0.65
            self.yaw += random.uniform(-120.0, 120.0)

        speed = 1.15 if self._walking else 0.0
        rad = math.radians(self.yaw)
        target = np.array([math.cos(rad) * speed, 0.0, math.sin(rad) * speed])
        blend = min(1.0, 8.0 * dt)
        self.velocity[0] += (target[0] - self.velocity[0]) * blend
        self.velocity[2] += (target[2] - self.velocity[2]) * blend
        self.velocity[1] = max(self.velocity[1] + _GRAVITY * dt, -30.0)

        result = move_with_collisions(
            world, self.position, self.velocity, dt, self.HALF_W, self.HEIGHT
        )
        if result.hit_wall and result.on_ground:
            self.velocity[1] = 7.0  # hop over one-block steps
        intended = self.velocity * dt
        actual = result.position - self.position
        self.position = result.position
        self.on_ground = result.on_ground
        for axis in (0, 1, 2):
            if abs(actual[axis] - intended[axis]) > 1e-7 and axis == 1:
                self.velocity[1] = 0.0 if result.on_ground else self.velocity[1]

        if speed > 0:
            self.walk_cycle += dt * 9.0

    def render(self, boxes: BoxRenderer, world: World, daylight: float) -> None:
        light = entity_light(
            world, self.position[0], self.position[1] + 0.5, self.position[2], daylight
        )
        tint = (1.0, 0.35, 0.35) if self.hurt_timer > 0 else _PIG_TINT
        dark = (0.85, 0.3, 0.3) if self.hurt_timer > 0 else _PIG_DARK
        yaw = -self.yaw - 90.0  # model faces +Z locally; entity yaw is world XZ
        swing = math.sin(self.walk_cycle) * 30.0

        def part(offset, size, pitch=0.0, color=tint):
            boxes.draw_box(
                model_matrix(self.position, yaw, offset, size, pitch), color, light
            )

        part((0.0, 0.55, 0.0), (0.62, 0.5, 0.95))              # body
        part((0.0, 0.72, 0.62), (0.44, 0.42, 0.34))            # head
        part((0.0, 0.62, 0.82), (0.16, 0.12, 0.06), color=dark)  # snout
        for sx in (-0.18, 0.18):
            part((sx, 0.18, 0.32), (0.18, 0.38, 0.18), pitch=swing, color=dark)
            part((sx, 0.18, -0.32), (0.18, 0.38, 0.18), pitch=-swing, color=dark)


class PlayerModel:
    """Third-person player: classic blocky humanoid with walk animation."""

    SKIN = (0.85, 0.66, 0.52)
    SHIRT = (0.19, 0.5, 0.66)
    PANTS = (0.24, 0.28, 0.46)

    def __init__(self) -> None:
        self.walk_cycle = 0.0

    def update(self, dt: float, horizontal_speed: float) -> None:
        self.walk_cycle += dt * (4.0 + horizontal_speed * 1.6) * min(horizontal_speed, 1.8)

    def render(
        self,
        boxes: BoxRenderer,
        world: World,
        position: np.ndarray,
        yaw_deg: float,
        daylight: float,
        moving: bool,
    ) -> None:
        light = entity_light(world, position[0], position[1] + 1.0, position[2], daylight)
        yaw = -yaw_deg - 90.0
        swing = math.sin(self.walk_cycle) * (34.0 if moving else 0.0)

        def part(offset, size, pitch=0.0, color=self.SHIRT):
            boxes.draw_box(model_matrix(position, yaw, offset, size, pitch), color, light)

        part((0.0, 1.62, 0.0), (0.46, 0.46, 0.46), color=self.SKIN)   # head
        part((0.0, 1.06, 0.0), (0.5, 0.68, 0.26))                     # torso
        part((-0.34, 1.18, 0.0), (0.17, 0.62, 0.2), pitch=swing, color=self.SKIN)
        part((0.34, 1.18, 0.0), (0.17, 0.62, 0.2), pitch=-swing, color=self.SKIN)
        part((-0.13, 0.38, 0.0), (0.2, 0.74, 0.22), pitch=-swing, color=self.PANTS)
        part((0.13, 0.38, 0.0), (0.2, 0.74, 0.22), pitch=swing, color=self.PANTS)


class MobManager:
    MAX_PIGS = 10
    SPAWN_INTERVAL = 3.0
    DESPAWN_DIST = 64.0

    def __init__(self, world: World, registry) -> None:
        self.world = world
        self.registry = registry
        self.pigs: list[Pig] = []
        self._spawn_timer = 2.0

    def _surface_y(self, x: int, z: int) -> int | None:
        chunk = self.world.get_chunk(x >> 4, z >> 4)
        if chunk is None:
            return None
        column = chunk.blocks[x & 15, z & 15, :]
        solid = self.registry.solid[column]
        ys = np.nonzero(solid)[0]
        if not len(ys):
            return None
        top = int(ys[-1])
        if self.registry.by_id[int(column[top])].name != "grass_block":
            return None
        return top

    def update(self, dt: float, player_pos: np.ndarray) -> None:
        self._spawn_timer -= dt
        if self._spawn_timer <= 0.0:
            self._spawn_timer = self.SPAWN_INTERVAL
            if len(self.pigs) < self.MAX_PIGS:
                angle = random.uniform(0, 2 * math.pi)
                dist = random.uniform(18.0, 34.0)
                x = int(player_pos[0] + math.cos(angle) * dist)
                z = int(player_pos[2] + math.sin(angle) * dist)
                top = self._surface_y(x, z)
                if top is not None and top < CHUNK_Y - 4:
                    self.pigs.append(Pig(np.array([x + 0.5, top + 1.05, z + 0.5])))

        for pig in self.pigs:
            pig.update(dt, self.world)
        self.pigs = [
            p for p in self.pigs
            if not p.dead
            and np.linalg.norm(p.position - player_pos) < self.DESPAWN_DIST
            and p.position[1] > -8
        ]

    def try_attack(self, origin: np.ndarray, direction: np.ndarray, reach: float) -> bool:
        """Ray-vs-AABB against all pigs; hurt the nearest hit."""
        best: tuple[float, Pig] | None = None
        for pig in self.pigs:
            t = _ray_aabb(
                origin, direction,
                pig.position - np.array([pig.HALF_W, 0, pig.HALF_W]),
                pig.position + np.array([pig.HALF_W, pig.HEIGHT, pig.HALF_W]),
            )
            if t is not None and t <= reach and (best is None or t < best[0]):
                best = (t, pig)
        if best is None:
            return False
        flat = np.array([direction[0], 0.0, direction[2]])
        norm = np.linalg.norm(flat)
        best[1].hurt(2, flat / norm if norm > 1e-6 else flat)
        return True

    def render(self, boxes: BoxRenderer, daylight: float) -> None:
        for pig in self.pigs:
            pig.render(boxes, self.world, daylight)


def _ray_aabb(origin, direction, box_min, box_max) -> float | None:
    t_near, t_far = 0.0, math.inf
    for axis in range(3):
        d = direction[axis]
        if abs(d) < 1e-9:
            if not (box_min[axis] <= origin[axis] <= box_max[axis]):
                return None
            continue
        t1 = (box_min[axis] - origin[axis]) / d
        t2 = (box_max[axis] - origin[axis]) / d
        t1, t2 = min(t1, t2), max(t1, t2)
        t_near = max(t_near, t1)
        t_far = min(t_far, t2)
        if t_near > t_far:
            return None
    return t_near
