"""Player controller: movement, swimming, flying, sneaking and survival state.

Owns health/air/fall bookkeeping; world editing lives in the game loop where
the hotbar is.  In creative mode all damage is ignored and flying unlocks.
"""

from __future__ import annotations

from math import floor

import glfw
import numpy as np

from engine.camera import Camera
from engine.core.log import get_logger
from engine.input import InputState
from engine.physics.aabb import move_with_collisions
from engine.world.blocks import RENDER_LIQUID, BlockRegistry
from engine.world.world import World

_log = get_logger("player")

SURVIVAL = "survival"
CREATIVE = "creative"


class Player:
    HALF_W = 0.3
    HEIGHT = 1.8
    EYE = 1.62

    GRAVITY = -27.0
    JUMP_SPEED = 8.6
    WALK_SPEED = 4.4
    SNEAK_SPEED = 1.6
    SPRINT_SPEED = 6.9
    FLY_SPEED = 11.0
    FLY_SPRINT_SPEED = 24.0
    SWIM_SPEED = 3.1

    MAX_HEALTH = 20.0
    MAX_AIR = 10.0          # seconds of breath
    SAFE_FALL = 3.2         # blocks of free fall before damage
    REGEN_DELAY = 5.0       # seconds after damage before regen starts
    REGEN_RATE = 1.0 / 2.5  # hearts-halves per second
    RESPAWN_DELAY = 2.0

    def __init__(self, world: World, registry: BlockRegistry, spawn: np.ndarray) -> None:
        self.world = world
        self.registry = registry
        self.position = spawn.astype(np.float64).copy()
        self.spawn_point = spawn.astype(np.float64).copy()
        self.velocity = np.zeros(3, dtype=np.float64)
        self.on_ground = False
        self.flying = False
        self.sneaking = False
        self.sprinting = False
        self.mode = SURVIVAL
        self.in_fluid = False
        self.in_lava = False
        self.eye_in_fluid_id = 0  # block id at eye level (0 = air) — drives fog

        self.health = self.MAX_HEALTH
        self.air = self.MAX_AIR
        self.dead = False
        self.damage_flash = 0.0  # seconds of red vignette left
        self._respawn_timer = 0.0
        self._regen_cooldown = 0.0
        self._lava_tick = 0.0
        self._drown_tick = 0.0
        self._fall_peak: float | None = None

    # -- queries ------------------------------------------------------------------
    @property
    def eye_position(self) -> np.ndarray:
        return self.position + np.array([0.0, self.EYE, 0.0])

    def _fluid_at(self, x: float, y: float, z: float) -> int:
        bid = self.world.get_block(floor(x), floor(y), floor(z))
        return bid if self.registry.render[bid] == RENDER_LIQUID else 0

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        if mode == SURVIVAL:
            self.flying = False
        _log.info("Game mode: %s", mode)

    # -- damage & survival ----------------------------------------------------------
    def damage(self, amount: float) -> None:
        if self.mode == CREATIVE or self.dead or amount <= 0:
            return
        self.health -= amount
        self.damage_flash = 0.4
        self._regen_cooldown = self.REGEN_DELAY
        if self.health <= 0:
            self.health = 0
            self.dead = True
            self._respawn_timer = self.RESPAWN_DELAY

    def _respawn(self) -> None:
        self.position = self.spawn_point.copy()
        self.velocity[:] = 0.0
        self.health = self.MAX_HEALTH
        self.air = self.MAX_AIR
        self.dead = False
        self._fall_peak = None

    def _update_survival(self, dt: float) -> None:
        self.damage_flash = max(0.0, self.damage_flash - dt)
        if self.mode == CREATIVE:
            self.air = self.MAX_AIR
            return

        # Drowning: breath runs out with the eyes underwater.
        eye_fluid = self.eye_in_fluid_id
        underwater = eye_fluid != 0 and not self.in_lava
        if underwater:
            self.air = max(0.0, self.air - dt)
            if self.air <= 0.0:
                self._drown_tick += dt
                if self._drown_tick >= 1.0:
                    self._drown_tick = 0.0
                    self.damage(2.0)
        else:
            self.air = min(self.MAX_AIR, self.air + dt * 2.5)
            self._drown_tick = 0.0

        # Lava hurts fast.
        if self.in_lava:
            self._lava_tick += dt
            if self._lava_tick >= 0.35:
                self._lava_tick = 0.0
                self.damage(3.0)
        else:
            self._lava_tick = 0.0

        # Slow regeneration when out of danger.
        self._regen_cooldown = max(0.0, self._regen_cooldown - dt)
        if self._regen_cooldown == 0.0 and 0 < self.health < self.MAX_HEALTH:
            self.health = min(self.MAX_HEALTH, self.health + self.REGEN_RATE * dt)

    # -- per-frame update -------------------------------------------------------------
    def update(self, dt: float, inp: InputState, camera: Camera) -> None:
        if self.dead:
            self._respawn_timer -= dt
            self.damage_flash = 0.5
            if self._respawn_timer <= 0.0:
                self._respawn()
            camera.position = self.eye_position
            return

        eye = self.eye_position
        feet_fluid = self._fluid_at(self.position[0], self.position[1] + 0.2, self.position[2])
        self.in_fluid = feet_fluid != 0
        self.in_lava = bool(
            feet_fluid and self.registry.by_id[feet_fluid].name == "lava"
        )
        self.eye_in_fluid_id = self._fluid_at(eye[0], eye[1], eye[2])

        if self.mode == CREATIVE and inp.was_pressed(glfw.KEY_F):
            self.flying = not self.flying
            if self.flying:
                self.velocity[1] = 0.0

        self.sneaking = (
            inp.is_down(glfw.KEY_LEFT_SHIFT) and not self.flying and not self.in_fluid
        )
        wish = self._wish_direction(inp, camera)
        if self.flying:
            self._update_flying(dt, inp, wish)
        else:
            self._update_walking(dt, inp, wish, camera)

        self._track_falling()
        self._apply_movement(dt)
        self._update_survival(dt)
        camera.position = self.eye_position

        # Fell out of the world (should be impossible with bedrock, but a
        # corrupt save must not soft-lock the game).
        if self.position[1] < -16.0:
            self.damage(6.0)
            self.position = self.spawn_point.copy()
            self.velocity[:] = 0.0
            self._fall_peak = None

    # -- movement modes ------------------------------------------------------------
    def _wish_direction(self, inp: InputState, camera: Camera) -> np.ndarray:
        forward = camera.flat_forward
        right = np.cross(forward, Camera.UP)
        wish = np.zeros(3, dtype=np.float64)
        if inp.is_down(glfw.KEY_W):
            wish += forward
        if inp.is_down(glfw.KEY_S):
            wish -= forward
        if inp.is_down(glfw.KEY_D):
            wish += right
        if inp.is_down(glfw.KEY_A):
            wish -= right
        norm = np.linalg.norm(wish)
        return wish / norm if norm > 1e-9 else wish

    @staticmethod
    def _approach(current: float, target: float, rate: float, dt: float) -> float:
        return current + (target - current) * min(1.0, rate * dt)

    def _update_flying(self, dt: float, inp: InputState, wish: np.ndarray) -> None:
        speed = (
            self.FLY_SPRINT_SPEED
            if inp.is_down(glfw.KEY_LEFT_CONTROL)
            else self.FLY_SPEED
        )
        vertical = 0.0
        if inp.is_down(glfw.KEY_SPACE):
            vertical += speed
        if inp.is_down(glfw.KEY_LEFT_SHIFT):
            vertical -= speed
        self.velocity[0] = self._approach(self.velocity[0], wish[0] * speed, 10.0, dt)
        self.velocity[2] = self._approach(self.velocity[2], wish[2] * speed, 10.0, dt)
        self.velocity[1] = self._approach(self.velocity[1], vertical, 10.0, dt)
        self.sprinting = False

    def _update_walking(
        self, dt: float, inp: InputState, wish: np.ndarray, camera: Camera
    ) -> None:
        moving = bool(np.linalg.norm(wish) > 0.1)
        self.sprinting = (
            inp.is_down(glfw.KEY_LEFT_CONTROL) and moving and not self.sneaking
        )
        if self.in_fluid:
            speed = self.SWIM_SPEED * (0.55 if self.in_lava else 1.0)
        elif self.sneaking:
            speed = self.SNEAK_SPEED
        else:
            speed = self.SPRINT_SPEED if self.sprinting else self.WALK_SPEED

        control = 12.0 if (self.on_ground or self.in_fluid) else 4.0
        self.velocity[0] = self._approach(self.velocity[0], wish[0] * speed, control, dt)
        self.velocity[2] = self._approach(self.velocity[2], wish[2] * speed, control, dt)

        if self.in_fluid:
            # Buoyant sink with a swim-up option; lava is extra sluggish.
            self.velocity[1] += self.GRAVITY * 0.28 * dt
            self.velocity[1] = max(self.velocity[1], -3.0)
            if inp.is_down(glfw.KEY_SPACE):
                self.velocity[1] = self._approach(self.velocity[1], 4.2, 9.0, dt)
            if self.on_ground and inp.is_down(glfw.KEY_SPACE):
                self.velocity[1] = self.JUMP_SPEED * 0.72
        else:
            self.velocity[1] += self.GRAVITY * dt
            self.velocity[1] = max(self.velocity[1], -55.0)  # terminal velocity
            if self.on_ground and inp.is_down(glfw.KEY_SPACE):
                self.velocity[1] = self.JUMP_SPEED
                if self.sprinting:
                    # Sprint-jumping carries extra momentum, Minecraft-style.
                    self.velocity[0] += camera.flat_forward[0] * 1.6
                    self.velocity[2] += camera.flat_forward[2] * 1.6

        if self.sneaking and self.on_ground:
            self._clamp_to_ledge(dt)

    def _has_ground_below(self, x: float, z: float) -> bool:
        y = floor(self.position[1] - 0.06)
        for ox in (-self.HALF_W, self.HALF_W):
            for oz in (-self.HALF_W, self.HALF_W):
                if self.world.is_solid(floor(x + ox), y, floor(z + oz)):
                    return True
        return False

    def _clamp_to_ledge(self, dt: float) -> None:
        """Sneaking never walks off an edge: cancel axis moves that would."""
        if self.velocity[0] != 0.0 and not self._has_ground_below(
            self.position[0] + self.velocity[0] * dt, self.position[2]
        ):
            self.velocity[0] = 0.0
        if self.velocity[2] != 0.0 and not self._has_ground_below(
            self.position[0], self.position[2] + self.velocity[2] * dt
        ):
            self.velocity[2] = 0.0

    # -- integration ------------------------------------------------------------
    def _track_falling(self) -> None:
        airborne = not self.on_ground and not self.in_fluid and not self.flying
        if airborne:
            peak = self.position[1] if self._fall_peak is None else self._fall_peak
            self._fall_peak = max(peak, self.position[1])
        elif self.in_fluid or self.flying:
            self._fall_peak = None  # water/flight cancels accumulated fall

    def _apply_movement(self, dt: float) -> None:
        intended = self.velocity * dt
        result = move_with_collisions(
            self.world, self.position, self.velocity, dt, self.HALF_W, self.HEIGHT
        )
        actual = result.position - self.position
        self.position = result.position
        landed = result.on_ground and not self.on_ground
        self.on_ground = result.on_ground

        if landed and self._fall_peak is not None:
            fall = self._fall_peak - self.position[1]
            self._fall_peak = None
            if fall > self.SAFE_FALL:
                self.damage(float(int(fall - self.SAFE_FALL + 0.5)) * 1.0)

        # Zero velocity on any axis that got clamped, so we don't keep
        # accelerating into walls/floor.
        for axis in range(3):
            if abs(actual[axis] - intended[axis]) > 1e-7:
                self.velocity[axis] = 0.0
