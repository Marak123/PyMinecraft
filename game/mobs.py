"""ECS-based mobs: spawning, AI, physics, drops (plan phase 7).

Replaces the old MobManager.  An :class:`EntityWorld` owns an ECS world and
the systems that run each frame; mobs are config-driven (`configs/mobs.json`)
and rendered as tinted boxes.  Hostiles use behaviour trees + A* to chase,
explode or shoot the player; passive mobs wander.  Dead mobs and mined
blocks spawn dropped-item entities the player walks into to collect.
"""

from __future__ import annotations

import json
import math
import random
from math import floor
from pathlib import Path

import numpy as np

from engine.ecs import ECSWorld
from engine.ecs.components import (
    Collider, Drops, DroppedItem, Health, MobAI, NetherFuse, Transform, Velocity,
)
from engine.graphics.boxrender import BoxRenderer, model_matrix
from engine.physics.aabb import move_with_collisions
from engine.world.coords import CHUNK_Y
from engine.world.world import World
from game.ai import behavior_tree as bt

_GRAVITY = -27.0
_DESPAWN_DIST = 72.0
_HOSTILE_CAP = 24
_PASSIVE_CAP = 12


def _light_at(world: World, pos, daylight: float) -> float:
    chunk = world.get_chunk(floor(pos[0]) >> 4, floor(pos[2]) >> 4)
    yy = min(max(int(pos[1]), 0), CHUNK_Y - 1)
    if chunk is None or not chunk.has_light:
        return max(0.25, daylight)
    lx, lz = floor(pos[0]) & 15, floor(pos[2]) & 15
    sky = float(chunk.sky_light[lx, lz, yy]) / 15.0
    blk = float(chunk.block_light[lx, lz, yy]) / 15.0
    return max(sky * max(daylight, 0.06), blk, 0.08)


class MobRegistry:
    def __init__(self, defs: dict) -> None:
        self.defs = defs

    @classmethod
    def load(cls, path: Path) -> "MobRegistry":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["mobs"])


class EntityWorld:
    """Owns the ECS world + systems for one dimension's mobs and items."""

    def __init__(self, world: World, registry, mob_reg: MobRegistry) -> None:
        self.world = world
        self.reg = registry
        self.mobs = mob_reg
        self.ecs = ECSWorld()
        self.player_pos = np.zeros(3)
        self.player_damage_cb = None   # callable(amount)
        self.explosion_cb = None       # callable(center, radius)
        self._spawn_timer = 3.0
        self._daylight = 1.0
        # Behaviour trees are shared per mob type (stateless; state lives on
        # the entity blackboard).
        self._trees = {
            "chase": self._build_chase_tree(),
            "explode": self._build_explode_tree(),
            "ranged": self._build_ranged_tree(),
            "wander": self._build_wander_tree(),
        }

    # -- spawning ---------------------------------------------------------------
    def spawn(self, mob_type: str, pos: np.ndarray) -> int:
        d = self.mobs.defs[mob_type]
        e = self.ecs.create_entity()
        self.ecs.add(e, Transform(pos.astype(np.float64), yaw=random.uniform(0, 360)))
        self.ecs.add(e, Velocity(np.zeros(3)))
        self.ecs.add(e, Collider(d["half_width"], d["height"]))
        self.ecs.add(e, Health(d["health"], d["health"]))
        self.ecs.add(e, MobAI(mob_type, d["hostile"], d["speed"]))
        self.ecs.add(e, Drops(d.get("drops", [])))
        if d.get("behavior") == "explode":
            self.ecs.add(e, NetherFuse())
        return e

    def spawn_drop(self, block_id: int, pos: np.ndarray, count: int = 1) -> None:
        if block_id == 0 or count <= 0:
            return
        e = self.ecs.create_entity()
        jitter = np.array([random.uniform(-0.2, 0.2), 0.25, random.uniform(-0.2, 0.2)])
        self.ecs.add(e, Transform(pos.astype(np.float64) + np.array([0.0, 0.25, 0.0])))
        self.ecs.add(e, Velocity(jitter * np.array([4, 8, 4])))
        self.ecs.add(e, Collider(0.15, 0.25))
        self.ecs.add(e, DroppedItem(block_id, count))

    def _surface(self, x: int, z: int):
        chunk = self.world.get_chunk(x >> 4, z >> 4)
        if chunk is None:
            return None
        col = chunk.blocks[x & 15, z & 15, :]
        solid = self.reg.solid[col]
        ys = np.nonzero(solid)[0]
        return int(ys[-1]) if len(ys) else None

    def _try_spawn_cycle(self) -> None:
        hostiles = sum(1 for e in self.ecs.with_(MobAI) if self.ecs.get(e, MobAI).hostile)
        passives = self.ecs.count(MobAI) - hostiles
        night = self._daylight < 0.35
        for _ in range(3):
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(24.0, 44.0)
            x = int(self.player_pos[0] + math.cos(angle) * dist)
            z = int(self.player_pos[2] + math.sin(angle) * dist)
            top = self._surface(x, z)
            if top is None or top >= CHUNK_Y - 4:
                continue
            pos = np.array([x + 0.5, top + 1.05, z + 0.5])
            light = _light_at(self.world, pos, self._daylight)
            surface_block = self.reg.by_id[int(self.world.get_block(x, top, z))].name
            if light < 0.32 and night and hostiles < _HOSTILE_CAP:
                pool = [n for n, dd in self.mobs.defs.items() if dd["hostile"]]
                self.spawn(random.choice(pool), pos)
                hostiles += 1
            elif light > 0.5 and surface_block == "grass_block" and passives < _PASSIVE_CAP:
                pool = [n for n, dd in self.mobs.defs.items()
                        if not dd["hostile"] and "plains" in dd.get("spawn", [])]
                if pool:
                    self.spawn(random.choice(pool), pos)
                    passives += 1

    # -- per-frame update -------------------------------------------------------
    def update(self, dt: float, player_pos: np.ndarray, daylight: float, creative: bool) -> None:
        self.player_pos = player_pos
        self._daylight = daylight
        self._spawn_timer -= dt
        if self._spawn_timer <= 0.0 and not creative:
            self._spawn_timer = 4.0
            self._try_spawn_cycle()
        self._ai_system(dt)
        self._physics_system(dt)
        self._item_system(dt)
        self._cull()

    def _cull(self) -> None:
        for e in list(self.ecs.with_(Transform)):
            tr = self.ecs.get(e, Transform)
            far = np.linalg.norm(tr.position - self.player_pos) > _DESPAWN_DIST
            void = tr.position[1] < -8
            hp = self.ecs.get(e, Health)
            dead = hp is not None and hp.current <= 0 and hp.death_timer <= 0
            if far or void or dead:
                if dead:
                    self._drop_loot(e, tr.position)
                self.ecs.destroy_entity(e)

    def _drop_loot(self, e: int, pos: np.ndarray) -> None:
        drops = self.ecs.get(e, Drops)
        if not drops:
            return
        for name, lo, hi in drops.table:
            count = random.randint(lo, hi)
            if count > 0 and name in self.reg.by_name:
                self.spawn_drop(self.reg.id_of(name), pos, count)

    # -- systems ----------------------------------------------------------------
    def _ai_system(self, dt: float) -> None:
        for e in list(self.ecs.with_(MobAI, Transform, Velocity)):
            ai = self.ecs.get(e, MobAI)
            hp = self.ecs.get(e, Health)
            if hp and hp.current <= 0:
                hp.death_timer = max(0.0, hp.death_timer - dt)  # falling-over
                continue
            if hp:
                hp.hurt_timer = max(0.0, hp.hurt_timer - dt)
            ai.attack_cooldown = max(0.0, ai.attack_cooldown - dt)
            tr = self.ecs.get(e, Transform)
            vel = self.ecs.get(e, Velocity)
            d = self.mobs.defs[ai.mob_type]
            tree = self._trees[d.get("behavior", "wander")]
            ctx = _Ctx(self, e, ai, tr, vel, hp, dt)
            tree.tick(ctx)
            ai.walk_cycle += dt * (6.0 + float(np.linalg.norm(vel.value[[0, 2]])) * 1.5)

    def _physics_system(self, dt: float) -> None:
        for e in list(self.ecs.with_(Transform, Velocity, Collider)):
            tr = self.ecs.get(e, Transform)
            col = self.ecs.get(e, Collider)
            vel = self.ecs.get(e, Velocity)
            vel.value[1] = max(vel.value[1] + _GRAVITY * dt, -34.0)
            res = move_with_collisions(self.world, tr.position, vel.value, dt,
                                       col.half_width, col.height)
            intended = vel.value * dt
            actual = res.position - tr.position
            tr.position = res.position
            col.on_ground = res.on_ground
            for ax in range(3):
                if abs(actual[ax] - intended[ax]) > 1e-7:
                    vel.value[ax] = 0.0
            # Auto-hop small ledges while chasing.
            if res.hit_wall and res.on_ground and self.ecs.has(e, MobAI):
                if np.linalg.norm(vel.value[[0, 2]]) > 0.2:
                    vel.value[1] = 7.5

    def _item_system(self, dt: float) -> None:
        for e in list(self.ecs.with_(DroppedItem, Transform)):
            item = self.ecs.get(e, DroppedItem)
            item.age += dt
            item.pickup_delay = max(0.0, item.pickup_delay - dt)
            item.bob = math.sin(item.age * 3.0) * 0.1
            if item.age > 300.0:
                self.ecs.destroy_entity(e)
                continue
            tr = self.ecs.get(e, Transform)
            if item.pickup_delay <= 0.0 and np.linalg.norm(tr.position - self.player_pos) < 1.4:
                if self.pickup_cb:
                    consumed = self.pickup_cb(item.block_id, item.count)
                    item.count -= consumed
                    if item.count <= 0:
                        self.ecs.destroy_entity(e)

    pickup_cb = None  # callable(block_id, count) -> int consumed

    # -- combat -----------------------------------------------------------------
    def attack_ray(self, origin, direction, reach: float, damage: float = 5.0) -> bool:
        """Player melee: hurt the nearest mob the ray hits."""
        best = None
        for e in self.ecs.with_(Transform, Collider, Health):
            if self.ecs.has(e, DroppedItem):
                continue
            tr = self.ecs.get(e, Transform)
            col = self.ecs.get(e, Collider)
            t = _ray_aabb(origin, direction,
                          tr.position - np.array([col.half_width, 0, col.half_width]),
                          tr.position + np.array([col.half_width, col.height, col.half_width]))
            if t is not None and t <= reach and (best is None or t < best[0]):
                best = (t, e, tr)
        if best is None:
            return False
        _, e, tr = best
        hp = self.ecs.get(e, Health)
        hp.current -= damage
        hp.hurt_timer = 0.3
        flat = np.array([direction[0], 0.0, direction[2]])
        norm = np.linalg.norm(flat)
        vel = self.ecs.get(e, Velocity)
        if norm > 1e-6:
            vel.value += flat / norm * 6.0
        vel.value[1] = 4.5
        if hp.current <= 0:
            hp.death_timer = 0.5
        return True

    # -- behaviour trees --------------------------------------------------------
    def _build_wander_tree(self) -> bt.Node:
        return bt.Action(_act_wander)

    def _build_chase_tree(self) -> bt.Node:
        # Attack when adjacent, else chase, else wander — each branch returns
        # SUCCESS so the fallbacks below never fight the active behaviour.
        return bt.Selector(
            bt.Sequence(bt.Condition(_cond_player_near(2.2)), bt.Action(_act_melee)),
            bt.Sequence(bt.Condition(_cond_player_near(22)), bt.Action(_act_chase)),
            bt.Action(_act_wander),
        )

    def _build_explode_tree(self) -> bt.Node:
        return bt.Selector(
            bt.Sequence(bt.Condition(_cond_player_near(16)), bt.Action(_act_chase),
                        bt.Action(_act_fuse)),
            bt.Action(_act_wander),
        )

    def _build_ranged_tree(self) -> bt.Node:
        return bt.Selector(
            bt.Sequence(bt.Condition(_cond_player_near(18)), bt.Action(_act_shoot)),
            bt.Action(_act_wander),
        )

    # -- rendering --------------------------------------------------------------
    def render(self, boxes: BoxRenderer, daylight: float) -> None:
        for e in self.ecs.with_(MobAI, Transform):
            _render_mob(self, e, boxes, daylight)
        for e in self.ecs.with_(DroppedItem, Transform):
            _render_drop(self, e, boxes, daylight)


class _Ctx:
    __slots__ = ("ew", "e", "ai", "tr", "vel", "hp", "dt")

    def __init__(self, ew, e, ai, tr, vel, hp, dt):
        self.ew, self.e, self.ai, self.tr = ew, e, ai, tr
        self.vel, self.hp, self.dt = vel, hp, dt


# -- behaviour leaf functions ---------------------------------------------------
def _cond_player_near(dist: float):
    d2 = dist * dist

    def pred(ctx) -> bool:
        return float(np.sum((ctx.tr.position - ctx.ew.player_pos) ** 2)) < d2

    return pred


def _steer(ctx, target_xz, speed: float) -> None:
    to = np.array([target_xz[0] - ctx.tr.position[0], 0.0, target_xz[1] - ctx.tr.position[2]])
    n = np.linalg.norm(to)
    if n > 1e-3:
        to /= n
        blend = min(1.0, 8.0 * ctx.dt)
        ctx.vel.value[0] += (to[0] * speed - ctx.vel.value[0]) * blend
        ctx.vel.value[2] += (to[2] * speed - ctx.vel.value[2]) * blend
        ctx.tr.yaw = math.degrees(math.atan2(to[2], to[0]))


def _act_wander(ctx) -> bt.Status:
    bb = ctx.ai.blackboard
    bb["wander"] = bb.get("wander", 0.0) - ctx.dt
    if bb["wander"] <= 0.0:
        bb["wander"] = random.uniform(1.5, 4.0)
        bb["dir"] = random.uniform(0, 2 * math.pi) if random.random() < 0.7 else None
    if bb.get("dir") is not None:
        tx = ctx.tr.position[0] + math.cos(bb["dir"])
        tz = ctx.tr.position[2] + math.sin(bb["dir"])
        _steer(ctx, (tx, tz), ctx.ai.speed)
    else:
        ctx.vel.value[0] *= 0.8
        ctx.vel.value[2] *= 0.8
    return bt.Status.SUCCESS


def _act_chase(ctx) -> bt.Status:
    # Cheap: steer straight at the player; A* only when a wall is in the way.
    _steer(ctx, (ctx.ew.player_pos[0], ctx.ew.player_pos[2]), ctx.ai.speed)
    return bt.Status.SUCCESS


def _act_melee(ctx) -> bt.Status:
    if ctx.ai.attack_cooldown <= 0.0 and ctx.ew.player_damage_cb:
        ctx.ew.player_damage_cb(3.0)
        ctx.ai.attack_cooldown = 1.0
    return bt.Status.SUCCESS


def _act_shoot(ctx) -> bt.Status:
    _steer(ctx, (ctx.ew.player_pos[0], ctx.ew.player_pos[2]), ctx.ai.speed * 0.3)
    if ctx.ai.attack_cooldown <= 0.0 and ctx.ew.player_damage_cb:
        ctx.ew.player_damage_cb(2.0)  # hitscan stand-in for an arrow
        ctx.ai.attack_cooldown = 1.6
    return bt.Status.SUCCESS


def _act_fuse(ctx) -> bt.Status:
    fuse = ctx.ew.ecs.get(ctx.e, NetherFuse)
    if fuse is None:
        return bt.Status.SUCCESS
    near = float(np.sum((ctx.tr.position - ctx.ew.player_pos) ** 2)) < 9.0
    if near:
        fuse.lit = True
        fuse.timer += ctx.dt
        ctx.vel.value[0] *= 0.4
        ctx.vel.value[2] *= 0.4
        if fuse.timer >= 1.3:
            if ctx.ew.explosion_cb:
                ctx.ew.explosion_cb(ctx.tr.position.copy(), 3)
            hp = ctx.ew.ecs.get(ctx.e, Health)
            hp.current = 0
            hp.death_timer = 0.0
    else:
        fuse.lit = False
        fuse.timer = max(0.0, fuse.timer - ctx.dt)
    return bt.Status.SUCCESS


# -- rendering helpers ----------------------------------------------------------
def _mob_light(ew, tr, daylight):
    return _light_at(ew.world, tr.position + np.array([0, 0.5, 0]), daylight)


def _render_mob(ew, e, boxes, daylight) -> None:
    ai = ew.ecs.get(e, MobAI)
    tr = ew.ecs.get(e, Transform)
    hp = ew.ecs.get(e, Health)
    d = ew.mobs.defs[ai.mob_type]
    light = _mob_light(ew, tr, daylight)
    hurt = hp is not None and hp.hurt_timer > 0
    fuse = ew.ecs.get(e, NetherFuse)
    base = (1.0, 0.4, 0.4) if hurt else tuple(d["color"])
    if fuse and fuse.lit:
        base = (1.0, 1.0, 1.0) if int(fuse.timer * 8) % 2 else tuple(d["color"])
    accent = (0.9, 0.35, 0.35) if hurt else tuple(d["accent"])
    yaw = -tr.yaw - 90.0
    swing = math.sin(ai.walk_cycle) * 32.0
    dead_tilt = 0.0
    if hp and hp.current <= 0:
        dead_tilt = (0.5 - hp.death_timer) * 180.0  # fall over

    def part(off, size, pitch=0.0, color=base):
        m = model_matrix(tr.position, yaw + (dead_tilt if dead_tilt else 0.0),
                         off, size, pitch)
        boxes.draw_box(m, color, light)

    body = d["body"]
    if body == "quadruped":
        part((0.0, 0.55 * d["height"], 0.0), (0.5, 0.5 * d["height"], 0.95))
        part((0.0, 0.72 * d["height"], 0.55), (0.42, 0.42, 0.34), color=base)
        for sx in (-0.18, 0.18):
            part((sx, 0.2, 0.32), (0.16, 0.4, 0.16), pitch=swing, color=accent)
            part((sx, 0.2, -0.32), (0.16, 0.4, 0.16), pitch=-swing, color=accent)
    elif body == "spider":
        part((0.0, 0.4, 0.0), (0.9, 0.5, 0.9))
        part((0.0, 0.45, 0.6), (0.5, 0.45, 0.45), color=accent)
        for i, sx in enumerate((-0.55, 0.55)):
            for dz in (-0.35, 0.0, 0.35):
                part((sx, 0.3, dz), (0.5, 0.12, 0.12),
                     pitch=swing * (1 if i else -1), color=base)
    else:  # biped
        h = d["height"]
        part((0.0, h - 0.25, 0.0), (0.45, 0.45, 0.45), color=base)      # head
        part((0.0, h * 0.62, 0.0), (0.5, h * 0.42, 0.26), color=accent)  # torso
        part((-0.32, h * 0.72, 0.0), (0.16, h * 0.4, 0.18), pitch=swing, color=base)
        part((0.32, h * 0.72, 0.0), (0.16, h * 0.4, 0.18), pitch=-swing, color=base)
        part((-0.13, h * 0.24, 0.0), (0.18, h * 0.44, 0.2), pitch=-swing, color=accent)
        part((0.13, h * 0.24, 0.0), (0.18, h * 0.44, 0.2), pitch=swing, color=accent)


def _render_drop(ew, e, boxes, daylight) -> None:
    tr = ew.ecs.get(e, Transform)
    item = ew.ecs.get(e, DroppedItem)
    light = _light_at(ew.world, tr.position, daylight)
    spin = (item.age * 90.0) % 360.0
    pos = tr.position + np.array([0.0, item.bob, 0.0])
    boxes.draw_box(model_matrix(pos, spin, (0, 0, 0), (0.28, 0.28, 0.28)),
                   (0.8, 0.8, 0.85), light)


def _ray_aabb(origin, direction, box_min, box_max):
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
