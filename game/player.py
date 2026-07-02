"""Player controller: walking, jumping, swimming, flying.

Consumes the engine's input snapshot and physics; owns no rendering and no
world-editing logic (that lives in the game loop where the hotbar is).
"""

from __future__ import annotations

import glfw
import numpy as np

from engine.camera import Camera
from engine.input import InputState
from engine.physics.aabb import move_with_collisions
from engine.world.blocks import RENDER_LIQUID, BlockRegistry
from engine.world.world import World


class Player:
    HALF_W = 0.3
    HEIGHT = 1.8
    EYE = 1.62

    GRAVITY = -27.0
    JUMP_SPEED = 8.6
    WALK_SPEED = 4.4
    SPRINT_SPEED = 6.9
    FLY_SPEED = 11.0
    FLY_SPRINT_SPEED = 24.0
    SWIM_SPEED = 3.1

    def __init__(self, world: World, registry: BlockRegistry, spawn: np.ndarray) -> None:
        self.world = world
        self.registry = registry
        self.position = spawn.astype(np.float64).copy()
        self.spawn_point = spawn.astype(np.float64).copy()
        self.velocity = np.zeros(3, dtype=np.float64)
        self.on_ground = False
        self.flying = False
        self.in_fluid = False
        self.eye_in_fluid_id = 0  # block id at eye level (0 = air) — drives fog

    # -- queries ------------------------------------------------------------------
    @property
    def eye_position(self) -> np.ndarray:
        return self.position + np.array([0.0, self.EYE, 0.0])

    def _fluid_at(self, x: float, y: float, z: float) -> int:
        bid = self.world.get_block(int(np.floor(x)), int(np.floor(y)), int(np.floor(z)))
        return bid if self.registry.render[bid] == RENDER_LIQUID else 0

    # -- per-frame update -------------------------------------------------------------
    def update(self, dt: float, inp: InputState, camera: Camera) -> None:
        eye = self.eye_position
        self.in_fluid = bool(
            self._fluid_at(self.position[0], self.position[1] + 0.2, self.position[2])
        )
        self.eye_in_fluid_id = self._fluid_at(eye[0], eye[1], eye[2])

        if inp.was_pressed(glfw.KEY_F):
            self.flying = not self.flying
            if self.flying:
                self.velocity[1] = 0.0

        wish = self._wish_direction(inp, camera)
        if self.flying:
            self._update_flying(dt, inp, wish)
        else:
            self._update_walking(dt, inp, wish)

        self._apply_movement(dt)
        camera.position = self.eye_position

        # Fell out of the world (should be impossible with bedrock, but a
        # corrupt save must not soft-lock the game).
        if self.position[1] < -16.0:
            self.position = self.spawn_point.copy()
            self.velocity[:] = 0.0

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
    def _approach(current: np.ndarray, target: np.ndarray, rate: float, dt: float) -> np.ndarray:
        return current + (target - current) * min(1.0, rate * dt)

    def _update_flying(self, dt: float, inp: InputState, wish: np.ndarray) -> None:
        speed = (
            self.FLY_SPRINT_SPEED
            if inp.is_down(glfw.KEY_LEFT_CONTROL)
            else self.FLY_SPEED
        )
        target = wish * speed
        vertical = 0.0
        if inp.is_down(glfw.KEY_SPACE):
            vertical += speed
        if inp.is_down(glfw.KEY_LEFT_SHIFT):
            vertical -= speed
        target[1] = vertical
        self.velocity = self._approach(self.velocity, target, 10.0, dt)

    def _update_walking(self, dt: float, inp: InputState, wish: np.ndarray) -> None:
        sprint = inp.is_down(glfw.KEY_LEFT_CONTROL)
        if self.in_fluid:
            speed = self.SWIM_SPEED
        else:
            speed = self.SPRINT_SPEED if sprint else self.WALK_SPEED

        target_h = wish * speed
        control = 12.0 if (self.on_ground or self.in_fluid) else 4.0
        self.velocity[0] = self._approach(
            np.array([self.velocity[0]]), np.array([target_h[0]]), control, dt
        )[0]
        self.velocity[2] = self._approach(
            np.array([self.velocity[2]]), np.array([target_h[2]]), control, dt
        )[0]

        if self.in_fluid:
            # Buoyant sink with a swim-up option; lava is extra sluggish.
            self.velocity[1] += self.GRAVITY * 0.28 * dt
            self.velocity[1] = max(self.velocity[1], -3.0)
            if inp.is_down(glfw.KEY_SPACE):
                self.velocity[1] = self._approach(
                    np.array([self.velocity[1]]), np.array([4.2]), 9.0, dt
                )[0]
            if self.on_ground and inp.is_down(glfw.KEY_SPACE):
                self.velocity[1] = self.JUMP_SPEED * 0.72
        else:
            self.velocity[1] += self.GRAVITY * dt
            self.velocity[1] = max(self.velocity[1], -55.0)  # terminal velocity
            if self.on_ground and inp.is_down(glfw.KEY_SPACE):
                self.velocity[1] = self.JUMP_SPEED

    # -- integration ------------------------------------------------------------
    def _apply_movement(self, dt: float) -> None:
        intended = self.velocity * dt
        result = move_with_collisions(
            self.world, self.position, self.velocity, dt, self.HALF_W, self.HEIGHT
        )
        actual = result.position - self.position
        self.position = result.position
        self.on_ground = result.on_ground
        # Zero velocity on any axis that got clamped, so we don't keep
        # accelerating into walls/floor.
        for axis in range(3):
            if abs(actual[axis] - intended[axis]) > 1e-7:
                self.velocity[axis] = 0.0
