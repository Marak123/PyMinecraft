"""Game assembly and main loop.

Wires engine subsystems together (window, renderer, world, streaming,
physics) and adds the gameplay: block interaction, hotbar, day cycle, HUD.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import glfw
import numpy as np

from engine.camera import Camera
from engine.core.config import load_settings
from engine.core.log import get_logger, init_logging
from engine.core.timing import FrameClock
from engine.graphics.font import build_font_atlas
from engine.graphics.mesher import build_chunk_meshes
from engine.graphics.renderer import Renderer
from engine.physics.aabb import block_intersects_box
from engine.physics.raycast import RayHit, raycast
from engine.window import Window
from engine.world.blocks import (
    AIR,
    RENDER_CROSS,
    RENDER_CUTOUT,
    RENDER_SOLID,
    BlockRegistry,
)
from engine.world.coords import CHUNK_X, CHUNK_Z
from engine.world.environment import Environment
from engine.world.generation import WorldGenerator, surface_height
from engine.world.save import WorldStorage
from engine.world.streaming import ChunkStreamer
from engine.world.world import World
from game.hud import Hud, HudState
from game.player import Player

_log = get_logger("game")

_REACH = 5.5
_EDIT_REPEAT_DELAY = 0.22
_TARGETABLE_RENDERS = (RENDER_SOLID, RENDER_CUTOUT, RENDER_CROSS)


class Game:
    def __init__(self, max_frames: int | None = None, screenshot_path: str | None = None) -> None:
        self.root = Path(__file__).resolve().parents[1]
        init_logging(self.root / "logs")
        self.settings = load_settings(self.root / "configs" / "settings.json")
        self.max_frames = max_frames
        self.screenshot_path = screenshot_path

        self.registry = BlockRegistry.load(self.root / "configs" / "blocks.json")
        self.storage = WorldStorage(self.root / "saves" / self.settings.world_name)
        meta = self.storage.load_meta() or {}

        seed = meta.get("seed")
        if seed is None:
            seed = self.settings.seed
        if seed is None:
            seed = random.randrange(2**31)
        self.seed = int(seed)
        _log.info("World seed: %d", self.seed)

        generator = WorldGenerator(self.seed, self.registry)
        self.world = World(generator, self.registry, self.storage)

        self.window = Window(
            self.settings.window_width,
            self.settings.window_height,
            "PyMinecraft",
            vsync=self.settings.vsync,
            fullscreen=self.settings.fullscreen,
        )
        font_atlas = build_font_atlas(16)
        self.renderer = Renderer(self.window.ctx, self.registry, font_atlas)
        self.streamer = ChunkStreamer(
            self.world,
            mesh_fn=lambda padded: build_chunk_meshes(padded, self.registry),
            upload_fn=self.renderer.upload_chunk,
            unload_fn=self.renderer.unload_chunk,
            render_radius=self.settings.render_distance,
        )

        self.env = Environment(
            self.settings.day_length_seconds,
            start_time=float(meta.get("time_of_day", 0.30)),
        )
        w, h = self.window.framebuffer_size
        self.camera = Camera(self.settings.fov, w / max(h, 1))

        spawn = self._prepare_spawn(meta)
        self.player = Player(self.world, self.registry, spawn)
        if "player_pos" in meta:
            self.player.position = np.array(meta["player_pos"], dtype=np.float64)
            self.camera.yaw = float(meta.get("player_yaw", self.camera.yaw))
            self.camera.pitch = float(meta.get("player_pitch", self.camera.pitch))
            self.player.flying = bool(meta.get("flying", False))
        self.camera.position = self.player.eye_position

        self.hotbar_ids = self._resolve_hotbar()
        self.selected_slot = int(meta.get("selected_slot", 0)) % len(self.hotbar_ids)
        self.hud = Hud(self.renderer, self.registry, font_atlas, self.hotbar_ids)

        self.clock = FrameClock()
        self.paused = False
        self.debug_visible = True
        self.target: RayHit | None = None
        self._edit_timer = 0.0
        self._hand_label_until = 0.0
        self.window.capture_cursor(True)

    # -- setup helpers ------------------------------------------------------------
    def _prepare_spawn(self, meta: dict) -> np.ndarray:
        _log.info("Preparing spawn area...")
        t0 = time.perf_counter()
        if "spawn_pos" in meta:
            sx, sz = int(meta["spawn_pos"][0]), int(meta["spawn_pos"][2])
        else:
            sx, sz = self.world.generator.find_spawn()
        scx, scz = sx >> 4, sz >> 4
        self.streamer.ensure_spawn_area(scx, scz, radius=2)
        chunk = self.world.get_chunk(scx, scz)
        assert chunk is not None
        ground = surface_height(chunk.blocks, sx & 15, sz & 15, self.registry)
        _log.info(
            "Spawn at (%d, %d, %d), ready in %.2f s",
            sx, ground + 1, sz, time.perf_counter() - t0,
        )
        return np.array([sx + 0.5, ground + 1.05, sz + 0.5])

    def _resolve_hotbar(self) -> list[int]:
        ids = []
        for name in self.settings.hotbar:
            if name in self.registry.by_name:
                ids.append(self.registry.id_of(name))
            else:
                _log.warning("Unknown hotbar block '%s' — skipped", name)
        return ids or [self.registry.id_of("stone")]

    # -- main loop ------------------------------------------------------------
    def run(self) -> None:
        frames = 0
        try:
            while not self.window.should_close:
                dt = self.clock.tick()
                self.window.poll()
                self._handle_global_input()

                if not self.paused:
                    self._update_gameplay(dt)

                center = (
                    int(np.floor(self.player.position[0])) >> 4,
                    int(np.floor(self.player.position[2])) >> 4,
                )
                fwd = self.camera.forward
                self.streamer.update(center, (float(fwd[0]), float(fwd[2])))

                self._render()
                self.window.swap()

                frames += 1
                if self.max_frames is not None and frames >= self.max_frames:
                    if self.screenshot_path:
                        self.renderer.screenshot(self.screenshot_path)
                    self.window.request_close()
        finally:
            self._shutdown()

    # -- input ---------------------------------------------------------------------
    def _handle_global_input(self) -> None:
        inp = self.window.input
        if inp.was_pressed(glfw.KEY_ESCAPE):
            self._set_paused(not self.paused)
        if not self.window.focused and not self.paused:
            self._set_paused(True)
        if self.paused and inp.was_button_pressed(glfw.MOUSE_BUTTON_LEFT):
            self._set_paused(False)

        if inp.was_pressed(glfw.KEY_F3):
            self.debug_visible = not self.debug_visible
        if inp.was_pressed(glfw.KEY_F2):
            shots = self.root / "screenshots"
            shots.mkdir(exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self.renderer.screenshot(str(shots / f"shot_{stamp}.png"))

        for i in range(len(self.hotbar_ids)):
            if inp.was_pressed(glfw.KEY_1 + i):
                self._select_slot(i)
        if inp.scroll_dy != 0.0:
            step = -1 if inp.scroll_dy > 0 else 1
            self._select_slot((self.selected_slot + step) % len(self.hotbar_ids))

    def _set_paused(self, paused: bool) -> None:
        self.paused = paused
        self.window.capture_cursor(not paused)

    def _select_slot(self, index: int) -> None:
        self.selected_slot = index
        self._hand_label_until = self.clock.time + 1.8

    # -- gameplay ------------------------------------------------------------------
    def _update_gameplay(self, dt: float) -> None:
        inp = self.window.input
        sens = self.settings.mouse_sensitivity
        self.camera.rotate(inp.mouse_dx * sens, -inp.mouse_dy * sens)

        self.player.update(dt, inp, self.camera)
        self.env.update(dt)

        self.target = raycast(
            self.world,
            self.camera.position,
            self.camera.forward,
            _REACH,
            lambda bid: self.registry.render[bid] in _TARGETABLE_RENDERS,
        )
        self._handle_block_edits(dt, inp)

    def _handle_block_edits(self, dt: float, inp) -> None:
        left = inp.is_button_down(glfw.MOUSE_BUTTON_LEFT)
        right = inp.is_button_down(glfw.MOUSE_BUTTON_RIGHT)
        if not (left or right):
            self._edit_timer = _EDIT_REPEAT_DELAY  # first press acts instantly
        else:
            self._edit_timer += dt

        if inp.was_button_pressed(glfw.MOUSE_BUTTON_MIDDLE) and self.target:
            picked = self.world.get_block(*self.target.block)
            if picked != AIR and picked in self.hotbar_ids:
                self._select_slot(self.hotbar_ids.index(picked))
            elif picked != AIR:
                self.hotbar_ids[self.selected_slot] = picked
                self._select_slot(self.selected_slot)

        act = (
            inp.was_button_pressed(glfw.MOUSE_BUTTON_LEFT)
            or inp.was_button_pressed(glfw.MOUSE_BUTTON_RIGHT)
            or self._edit_timer >= _EDIT_REPEAT_DELAY
        )
        if not act or self.target is None:
            return

        if left:
            block_id = self.world.get_block(*self.target.block)
            if self.registry.by_id[block_id].breakable:
                self.world.set_block(*self.target.block, AIR)
                self._edit_timer = 0.0
        elif right:
            self._try_place(self.target)

    def _try_place(self, hit: RayHit) -> None:
        cell = hit.previous
        new_id = self.hotbar_ids[self.selected_slot]
        current = self.world.get_block(*cell)
        if not self.registry.replaceable[current]:
            return
        # Never let a solid block materialise inside the player.
        if self.registry.solid[new_id] and block_intersects_box(
            cell, self.player.position, Player.HALF_W, Player.HEIGHT
        ):
            return
        if self.world.set_block(*cell, new_id):
            self._edit_timer = 0.0

    # -- rendering ------------------------------------------------------------------
    def _render(self) -> None:
        w, h = self.window.framebuffer_size
        if w == 0 or h == 0:  # minimised
            return
        self.camera.set_aspect(w / h)
        self.renderer.resize(w, h)

        fog_end = self.settings.render_distance * CHUNK_X - 8.0
        fog_start = fog_end * 0.55
        fog_color = None
        underwater = self.player.eye_in_fluid_id != 0
        if underwater:
            fog_start, fog_end = 2.0, 22.0
            d = max(self.env.daylight, 0.12)
            fog_color = (0.045 * d, 0.14 * d, 0.38 * d)

        stats = self.renderer.render_world(
            self.camera,
            self.env,
            self.clock.time,
            fog_start,
            fog_end,
            fog_color=fog_color,
            highlight=self.target.block if self.target else None,
        )
        stats.update(self.streamer.stats())

        hand_id = self.hotbar_ids[self.selected_slot]
        state = HudState(
            fps=self.clock.fps,
            frame_ms=self.clock.delta * 1000.0,
            position=tuple(self.player.position),
            chunk=(
                int(np.floor(self.player.position[0])) >> 4,
                int(np.floor(self.player.position[2])) >> 4,
            ),
            stats=stats,
            selected_slot=self.selected_slot,
            debug_visible=self.debug_visible,
            paused=self.paused,
            underwater=underwater,
            time_of_day=self.env.time_of_day,
            seed=self.seed,
            hand_label=(
                self.registry.by_id[hand_id].label
                if self.clock.time < self._hand_label_until
                else ""
            ),
            flying=self.player.flying,
        )
        self.hud.draw(w, h, state)

    # -- shutdown ------------------------------------------------------------------
    def _shutdown(self) -> None:
        _log.info("Shutting down: saving world...")
        try:
            self.storage.save_meta(
                {
                    "seed": self.seed,
                    "time_of_day": self.env.time_of_day,
                    "spawn_pos": [float(v) for v in self.player.spawn_point],
                    "player_pos": [float(v) for v in self.player.position],
                    "player_yaw": self.camera.yaw,
                    "player_pitch": self.camera.pitch,
                    "flying": self.player.flying,
                    "selected_slot": self.selected_slot,
                }
            )
            saved = self.world.save_all_modified()
            _log.info("Saved %d modified chunks", saved)
        finally:
            self.streamer.shutdown()
            self.renderer.release()
            self.window.close()
