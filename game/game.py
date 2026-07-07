"""Game assembly and main loop.

Wires engine subsystems together (window, renderer, world, streaming,
lighting, physics) and adds the gameplay: block interaction with dig times,
survival state, hotbar, day cycle, HUD.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import glfw
import numpy as np

from engine.camera import Camera
from engine.core.config import load_settings, save_settings
from engine.core.log import get_logger, init_logging
from engine.core.timing import FrameClock
from engine.graphics.font import build_font_atlas
from engine.graphics.mesher import build_chunk_meshes
from engine.graphics.renderer import Renderer
from engine.physics.aabb import block_intersects_box
from engine.physics.raycast import RayHit, raycast
from engine.window import Window
from engine.world import lighting
from engine.world.block_ticks import BlockTickScheduler, FallingBlocks
from engine.world.blocks import (
    AIR,
    RENDER_CROSS,
    RENDER_CUTOUT,
    RENDER_SOLID,
    BlockRegistry,
)
from engine.world.fluids import FluidSimulator
from engine.world.coords import CHUNK_X, CHUNK_Z
from engine.world.dimensions import NetherGenerator, nether_safe_y
from engine.world.environment import Environment
from engine.world.generation import WorldGenerator, surface_height
from engine.world.structures import StructureGenerator
from engine.world.save import WorldStorage
from engine.world.streaming import ChunkStreamer
from engine.world.world import World
from engine.audio import AudioEngine
from engine.graphics.boxrender import BoxRenderer
from engine.graphics.particle import ParticleSystem
from game.entities import PlayerModel
from game.mobs import EntityWorld, MobRegistry
from game import portals
from game.hud import Hud, HudState
from game.inventory import HOTBAR_SLOTS, CraftingBook, Inventory
from game.player import CREATIVE, SURVIVAL, Player
from game.ui import InventoryScreen, Mouse, SettingsScreen

_log = get_logger("game")

_REACH = 5.5
_PLACE_REPEAT_DELAY = 0.22
_CREATIVE_BREAK_DELAY = 0.22
_TARGETABLE_RENDERS = (RENDER_SOLID, RENDER_CUTOUT, RENDER_CROSS)


@dataclass
class _Dimension:
    """A self-contained world bundle the game can switch between."""
    name: str
    world: World
    streamer: ChunkStreamer
    ticks: BlockTickScheduler
    fluids: FluidSimulator
    falling: FallingBlocks
    saved_pos: list | None  # player position when last here


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
        generator.structures = StructureGenerator(self.seed, self.registry)
        self.world = World(generator, self.registry, self.storage)
        self._tick_accum = 0.0

        self.window = Window(
            self.settings.window_width,
            self.settings.window_height,
            "PyMinecraft",
            vsync=self.settings.vsync,
            fullscreen=self.settings.fullscreen,
        )
        font_atlas = build_font_atlas(16)
        self._font = font_atlas
        self.renderer = Renderer(
            self.window.ctx, self.registry, font_atlas,
            pack_dir=self.settings.texture_pack,
        )
        self.streamer = self._make_streamer(self.world)

        # Fluid + falling-block simulation for the active world, 20 TPS.
        self._bind_simulation(self.world)
        # Dimension registry: overworld now, Nether created on first travel.
        self.dimensions: dict[str, _Dimension] = {
            "overworld": _Dimension("overworld", self.world, self.streamer,
                                    self.ticks, self.fluids, self.falling, None)
        }
        self.dim_name = "overworld"
        self._portal_timer = 0.0

        self.env = Environment(
            self.settings.day_length_seconds,
            start_time=float(meta.get("time_of_day", 0.30)),
        )
        w, h = self.window.framebuffer_size
        self.camera = Camera(self.settings.fov, w / max(h, 1))
        self._base_fov = self.settings.fov

        spawn = self._prepare_spawn(meta)
        self.player = Player(self.world, self.registry, spawn)
        self.player.mode = str(meta.get("mode", SURVIVAL))
        self.player.health = float(meta.get("health", Player.MAX_HEALTH))
        self.player.hunger = float(meta.get("hunger", Player.MAX_HUNGER))
        if "player_pos" in meta:
            self.player.position = np.array(meta["player_pos"], dtype=np.float64)
            self.camera.yaw = float(meta.get("player_yaw", self.camera.yaw))
            self.camera.pitch = float(meta.get("player_pitch", self.camera.pitch))
            self.player.flying = bool(meta.get("flying", False)) and self.player.mode == CREATIVE
        self.camera.position = self.player.eye_position

        self.inventory = Inventory()
        if "inventory" in meta:
            self.inventory.load_meta(meta["inventory"])
        elif self.player.mode == CREATIVE:
            self._fill_default_hotbar()
        self.crafting = CraftingBook.load(self.root / "configs" / "recipes.json", self.registry)
        self.selected_slot = int(meta.get("selected_slot", 0)) % HOTBAR_SLOTS
        self.hud = Hud(self.renderer, self.registry, font_atlas, self.inventory)
        self.inv_screen = InventoryScreen(
            self.renderer, font_atlas, self.registry, self.inventory, self.crafting
        )
        self.settings_screen = SettingsScreen(self.renderer, font_atlas)
        self.screen: str | None = None  # None | "inventory"

        self.boxes = BoxRenderer(self.window.ctx)
        self.particles = ParticleSystem(self.window.ctx)
        self.audio = AudioEngine()
        self._prev_health = Player.MAX_HEALTH
        self._step_timer = 0.0
        self.mob_reg = MobRegistry.load(self.root / "configs" / "mobs.json")
        self.entities = self._make_entities(self.world)
        self.player_model = PlayerModel()
        self.third_person = bool(meta.get("third_person", False))

        self.clock = FrameClock()
        self.paused = False
        self.debug_visible = True
        self.target: RayHit | None = None
        self._place_timer = 0.0
        self._break_progress = 0.0
        self._break_target: tuple[int, int, int] | None = None
        self._hand_label_until = 0.0
        self._prof: dict[str, float] = {}
        self.window.capture_cursor(True)

    def _show_loading(self, text: str) -> None:
        """Draw one 'generating world' frame so the window isn't frozen while
        the spawn area streams in (plan 10 loading screen)."""
        w, h = self.window.framebuffer_size
        if w == 0 or h == 0:
            return
        self.window.ctx.screen.use()
        self.renderer.resize(w, h)
        self.window.ctx.clear(0.05, 0.06, 0.09)
        self.renderer.begin_ui(w, h)
        from engine.graphics.font import layout_text
        for label, y, scale, col in (
            ("PyMinecraft", h / 2 - 60, 3.0, (1.0, 1.0, 1.0, 1.0)),
            (text, h / 2 + 20, 1.2, (0.7, 0.8, 0.9, 1.0)),
        ):
            fw = len(label) * self._font.cell_w * scale
            verts = layout_text(self._font, w / 2 - fw / 2, y, label, scale)
            self.renderer.draw_ui_text(verts, col)
        self.window.swap()
        self.window.poll()

    # -- setup helpers ------------------------------------------------------------
    def _prepare_spawn(self, meta: dict) -> np.ndarray:
        _log.info("Preparing spawn area...")
        self._show_loading("Generating world...")
        t0 = time.perf_counter()
        if "spawn_pos" in meta:
            sx, sz = int(meta["spawn_pos"][0]), int(meta["spawn_pos"][2])
        else:
            sx, sz = self.world.generator.find_spawn()
        scx, scz = sx >> 4, sz >> 4
        self.streamer.ensure_spawn_area(scx, scz, radius=1)
        chunk = self.world.get_chunk(scx, scz)
        assert chunk is not None
        ground = surface_height(chunk.blocks, sx & 15, sz & 15, self.registry)
        _log.info(
            "Spawn at (%d, %d, %d), ready in %.2f s",
            sx, ground + 1, sz, time.perf_counter() - t0,
        )
        return np.array([sx + 0.5, ground + 1.05, sz + 0.5])

    def _fill_default_hotbar(self) -> None:
        """Creative starter kit from settings (survival starts empty)."""
        for i, name in enumerate(self.settings.hotbar[:HOTBAR_SLOTS]):
            if name in self.registry.by_name:
                self.inventory.slots[i] = [self.registry.id_of(name), 1]

    # -- main loop ------------------------------------------------------------
    def run(self) -> None:
        frames = 0
        try:
            while not self.window.should_close:
                dt = self.clock.tick()
                self.window.poll()
                self._handle_global_input()

                t0 = time.perf_counter()
                if not self.paused and self.screen is None:
                    self._update_gameplay(dt)
                t1 = time.perf_counter()

                center = (
                    int(np.floor(self.player.position[0])) >> 4,
                    int(np.floor(self.player.position[2])) >> 4,
                )
                fwd = self.camera.forward
                self.streamer.update(center, (float(fwd[0]), float(fwd[2])))
                t2 = time.perf_counter()

                self._render()
                t3 = time.perf_counter()
                self._prof = {
                    "update": (t1 - t0) * 1000.0,
                    "stream": (t2 - t1) * 1000.0,
                    "render": (t3 - t2) * 1000.0,
                }
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
            if self.screen is not None:
                self._set_screen(None)
            else:
                self._set_paused(not self.paused)
        if inp.was_pressed(glfw.KEY_E) and not self.paused and not self.player.dead:
            self._set_screen(None if self.screen else "inventory")
        if not self.window.focused and not self.paused and self.screen is None:
            self._set_paused(True)

        if inp.was_pressed(glfw.KEY_F11):
            self.settings.fullscreen = not self.settings.fullscreen
            self.window.set_fullscreen(self.settings.fullscreen)
            save_settings(self.root / "configs" / "settings.json", self.settings)

        if inp.was_pressed(glfw.KEY_F5):
            self.third_person = not self.third_person
        if inp.was_pressed(glfw.KEY_F3):
            self.debug_visible = not self.debug_visible
        if inp.was_pressed(glfw.KEY_F4):
            self.player.set_mode(
                CREATIVE if self.player.mode == SURVIVAL else SURVIVAL
            )
            self._hand_label_until = 0.0  # make room for the mode banner
        if inp.was_pressed(glfw.KEY_F2):
            shots = self.root / "screenshots"
            shots.mkdir(exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self.renderer.screenshot(str(shots / f"shot_{stamp}.png"))

        for i in range(HOTBAR_SLOTS):
            if inp.was_pressed(glfw.KEY_1 + i):
                self._select_slot(i)
        if inp.scroll_dy != 0.0 and self.screen is None:
            step = -1 if inp.scroll_dy > 0 else 1
            self._select_slot((self.selected_slot + step) % HOTBAR_SLOTS)

    def _set_paused(self, paused: bool) -> None:
        self.paused = paused
        if paused:
            self.screen = None
        self.window.capture_cursor(not paused and self.screen is None)

    def _set_screen(self, screen: str | None) -> None:
        self.screen = screen
        self.inv_screen.swap_source = None
        self.window.capture_cursor(screen is None and not self.paused)

    def _select_slot(self, index: int) -> None:
        self.selected_slot = index
        self._hand_label_until = self.clock.time + 1.8

    # -- gameplay ------------------------------------------------------------------
    def _update_gameplay(self, dt: float) -> None:
        inp = self.window.input
        sens = self.settings.mouse_sensitivity
        self.camera.rotate(inp.mouse_dx * sens, -inp.mouse_dy * sens)

        was_in_fluid = self.player.in_fluid
        self.player.update(dt, inp, self.camera)
        self.env.update(dt)
        self.particles.update(dt)
        self.audio.update(dt)
        self._feedback(dt, was_in_fluid)
        self._tick_fluids(dt)
        self.entities.update(dt, self.player.position, self.env.daylight,
                             self.player.mode == CREATIVE)
        h_speed = float(np.linalg.norm(self.player.velocity[[0, 2]]))
        self.player_model.update(dt, h_speed)
        if self.third_person:
            self._pull_back_camera()

        # Sprint FOV kick eases in and out.
        target_fov = self._base_fov + (7.0 if self.player.sprinting else 0.0)
        self.camera.set_fov(
            self.camera.fov + (target_fov - self.camera.fov) * min(1.0, 12.0 * dt)
        )

        if self.player.dead:
            self.target = None
            return
        # Targeting always originates at the player's eyes, not the camera —
        # in third person the camera sits metres behind the player.
        self.target = raycast(
            self.world,
            self.player.eye_position,
            self.camera.forward,
            _REACH,
            lambda bid: self.registry.render[bid] in _TARGETABLE_RENDERS,
        )
        self._handle_block_edits(dt, inp)

        # G ignites an obsidian frame the player is looking at.
        if inp.was_pressed(glfw.KEY_G) and self.target is not None:
            if self.world.get_block(*self.target.block) == self.registry.id_of("obsidian"):
                above = (self.target.block[0], self.target.block[1] + 1, self.target.block[2])
                if portals.ignite(self.world, above):
                    _log.info("Portal ignited at %s", above)
        self._update_portal(dt)

    def _update_portal(self, dt: float) -> None:
        """Standing in a portal block carries you to the other dimension."""
        eye = self.player.eye_position
        in_portal = self.world.get_block(
            int(np.floor(eye[0])), int(np.floor(eye[1])), int(np.floor(eye[2]))
        ) == self.registry.id_of("nether_portal")
        if self._portal_timer < 0.0:
            self._portal_timer = min(0.0, self._portal_timer + dt)  # grace
            return
        if in_portal:
            self._portal_timer += dt
            if self._portal_timer >= 1.2:
                self.travel("overworld" if self.dim_name == "nether" else "nether")
        else:
            self._portal_timer = 0.0

    def _make_streamer(self, world: World) -> ChunkStreamer:
        return ChunkStreamer(
            world,
            mesh_fn=lambda mesh_input: build_chunk_meshes(mesh_input, self.registry),
            upload_fn=self.renderer.upload_chunk,
            unload_fn=self.renderer.unload_chunk,
            render_radius=self.settings.render_distance,
        )

    def _bind_simulation(self, world: World) -> None:
        """(Re)create the tick/fluid/falling systems for the active world."""
        self.ticks = BlockTickScheduler()
        self.fluids = FluidSimulator(world, self.ticks, self.registry)
        self.falling = FallingBlocks(world, self.ticks)
        world.on_change = self._on_block_changed

    def _make_entities(self, world: World) -> EntityWorld:
        ew = EntityWorld(world, self.registry, self.mob_reg)
        ew.player_damage_cb = self.player.damage if hasattr(self, "player") else None
        ew.pickup_cb = self._pickup_item
        ew.explosion_cb = self._explode
        return ew

    def _pickup_item(self, block_id: int, count: int) -> int:
        """Add dropped items to the inventory; return how many were taken."""
        if self.player.mode == CREATIVE:
            return count  # vacuum without filling in creative
        leftover = self.inventory.add(block_id, count)
        return count - leftover

    def _explode(self, center: np.ndarray, radius: int) -> None:
        self.particles.emit(center + np.array([0, 0.5, 0]), (1.0, 0.9, 0.6, 1.0),
                            count=60, speed=6.0, spread=1.4, size=0.3,
                            lifetime=0.9, gravity=0.3)
        self.particles.emit(center + np.array([0, 0.5, 0]), (0.25, 0.22, 0.2, 1.0),
                            count=40, speed=3.0, size=0.4, lifetime=1.4, gravity=0.1)
        self.audio.play("explode")
        cx, cy, cz = int(center[0]), int(center[1]), int(center[2])
        r2 = radius * radius
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if dx * dx + dy * dy + dz * dz > r2:
                        continue
                    bx, by, bz = cx + dx, cy + dy, cz + dz
                    bid = self.world.get_block(bx, by, bz)
                    if bid != AIR and self.registry.by_id[bid].breakable:
                        self._apply_edit(bx, by, bz, AIR)
        if np.linalg.norm(center - self.player.position) < radius + 2:
            self.player.damage(7.0)

    def _feedback(self, dt: float, was_in_fluid: bool) -> None:
        """Footsteps, hurt cue, splash — audio + particle reactions."""
        if self.player.health < self._prev_health - 0.01:
            self.audio.play("hurt", min_gap=0.3)
        self._prev_health = self.player.health

        if self.player.in_fluid and not was_in_fluid:
            self.audio.play("splash")
            eye = self.player.position + np.array([0.0, 0.5, 0.0])
            self.particles.emit(eye, (0.5, 0.7, 1.0, 1.0), count=14,
                                speed=3.0, size=0.1, lifetime=0.6)

        speed = float(np.linalg.norm(self.player.velocity[[0, 2]]))
        if self.player.on_ground and speed > 1.5:
            self._step_timer -= dt
            if self._step_timer <= 0.0:
                self._step_timer = 0.34
                self.audio.play("step", min_gap=0.2)
        else:
            self._step_timer = 0.0

    def _on_block_changed(self, x: int, y: int, z: int) -> None:
        self.fluids.on_block_changed(x, y, z)
        self.falling.notify(x, y, z)

    # -- dimension travel -------------------------------------------------------
    def _get_dimension(self, name: str) -> _Dimension:
        if name in self.dimensions:
            return self.dimensions[name]
        gen = NetherGenerator(self.seed, self.registry)
        world = World(gen, self.registry, WorldStorage(self.storage.root.parent / name))
        streamer = self._make_streamer(world)
        ticks = BlockTickScheduler()
        fluids = FluidSimulator(world, ticks, self.registry)
        falling = FallingBlocks(world, ticks)
        world.on_change = None  # bound when activated
        dim = _Dimension(name, world, streamer, ticks, fluids, falling, None)
        self.dimensions[name] = dim
        return dim

    def travel(self, target_name: str) -> None:
        """Switch the active dimension, mapping the player's position."""
        cur = self.dimensions[self.dim_name]
        cur.saved_pos = [float(v) for v in self.player.position]
        self.world.save_all_modified()

        to_nether = target_name == "nether"
        dim = self._get_dimension(target_name) if target_name != "overworld" \
            else self.dimensions["overworld"]

        # Swap active references and rebind the per-world systems.
        self.renderer.clear_chunks()
        self.streamer = dim.streamer
        self.ticks, self.fluids, self.falling = dim.ticks, dim.fluids, dim.falling
        self.world = dim.world
        self.world.on_change = self._on_block_changed
        self.player.world = self.world
        self.entities = self._make_entities(self.world)
        self.dim_name = target_name

        if dim.saved_pos is not None and target_name == "overworld":
            pos = dim.saved_pos
        else:
            pos = portals.map_position(self.player.position, to_nether)
        cx, cz = int(pos[0]) >> 4, int(pos[2]) >> 4
        self.streamer.ensure_spawn_area(cx, cz, radius=1)
        chunk = self.world.get_chunk(cx, cz)
        lx, lz = int(pos[0]) & 15, int(pos[2]) & 15
        if to_nether:
            y = nether_safe_y(chunk.blocks, lx, lz, self.registry)
            self._build_return_platform(int(pos[0]), y, int(pos[2]))
        else:
            y = surface_height(chunk.blocks, lx, lz, self.registry) + 1
        self.player.position = np.array([int(pos[0]) + 0.5, y + 0.1, int(pos[2]) + 0.5])
        self.player.velocity[:] = 0.0
        self.camera.position = self.player.eye_position
        self._portal_timer = -1.5  # grace period so you don't bounce back
        _log.info("Traveled to %s at %s", target_name, self.player.position.astype(int))

    def _build_return_platform(self, x: int, y: int, z: int) -> None:
        """A small obsidian pad + portal so Nether arrival isn't into lava."""
        obs = self.registry.id_of("obsidian")
        for dx in range(-1, 3):
            for dz in range(-1, 2):
                self.world.set_block(x + dx, y - 1, z + dz, obs)
        # A 2x3 portal frame the player can step back through.
        for dy in range(3):
            self.world.set_block(x - 1, y + dy, z, obs)
            self.world.set_block(x + 2, y + dy, z, obs)
        for dx in range(0, 2):
            self.world.set_block(x + dx, y - 1 + 3, z, obs)
        portals.ignite(self.world, (x, y, z))

    def _tick_fluids(self, dt: float) -> None:
        # Fixed 20 TPS regardless of frame rate; cap catch-up after a hitch.
        self._tick_accum += dt
        step = 1.0 / 20.0
        budget = 4
        while self._tick_accum >= step and budget > 0:
            self._tick_accum -= step
            budget -= 1
            self.ticks.tick()

    def _pull_back_camera(self) -> None:
        """Third person: camera slides back until terrain blocks it."""
        eye = self.player.eye_position
        back = -self.camera.forward
        distance = 0.3
        while distance < 4.0:
            probe = eye + back * (distance + 0.3)
            if self.world.is_solid(
                int(np.floor(probe[0])), int(np.floor(probe[1])), int(np.floor(probe[2]))
            ):
                break
            distance += 0.3
        self.camera.position = eye + back * distance

    # -- block edits ------------------------------------------------------------
    def _apply_edit(self, x: int, y: int, z: int, block_id: int) -> bool:
        """set_block + incremental relight + remesh scheduling."""
        if not self.world.set_block(x, y, z, block_id):
            return False
        # Breaking the support from under a plant/torch pops the plant too.
        above = self.world.get_block(x, y + 1, z)
        if block_id == AIR and self.registry.needs_support[above]:
            self.world.set_block(x, y + 1, z, AIR)
        self.world.dirty_chunks |= lighting.relight_box(self.world, x, y, z)
        return True

    def _handle_block_edits(self, dt: float, inp) -> None:
        if inp.was_button_pressed(glfw.MOUSE_BUTTON_MIDDLE) and self.target:
            picked = self.world.get_block(*self.target.block)
            if picked != AIR:
                if self.player.mode == CREATIVE:
                    self.inventory.slots[self.selected_slot] = [picked, 1]
                    self._select_slot(self.selected_slot)
                else:
                    # Survival pick: jump to the hotbar slot holding that block.
                    for i in range(HOTBAR_SLOTS):
                        entry = self.inventory.slot(i)
                        if entry and entry[0] == picked:
                            self._select_slot(i)
                            break

        self._handle_breaking(dt, inp)
        self._handle_placing(dt, inp)

    def _held_tool(self):
        """(tool_type, tier, speed) of the selected item, or ('', -1, 1.0)."""
        entry = self.inventory.slot(self.selected_slot)
        if entry and self.registry.tool_type[entry[0]]:
            bid = entry[0]
            return (self.registry.tool_type[bid], int(self.registry.tool_tier[bid]),
                    float(self.registry.tool_speed[bid]))
        return ("", -1, 1.0)

    def _dig_time(self, block_id: int) -> float:
        """Seconds to break a block given the held tool (plan 8.2)."""
        hardness = max(float(self.registry.hardness[block_id]), 0.05)
        tool_type, tier, speed = self._held_tool()
        mined_by = self.registry.mined_by[block_id]
        if mined_by and tool_type == mined_by:
            return hardness / speed
        return hardness * 1.5  # wrong/no tool

    def _break_block(self, block: tuple[int, int, int], block_id: int) -> None:
        center = np.array([block[0] + 0.5, block[1] + 0.5, block[2] + 0.5])
        self.particles.emit(center, (*self.renderer.block_color(block_id), 1.0),
                            count=22, speed=2.2, size=0.11, lifetime=0.6)
        self.audio.play("break")
        self._apply_edit(*block, AIR)
        self.player.add_exhaustion(0.005)
        if self.player.mode == SURVIVAL:
            # Wrong tool tier yields nothing (plan 8.2).
            _, tier, _ = self._held_tool()
            if tier < int(self.registry.min_tier[block_id]):
                return
            drop = int(self.registry.drops[block_id])
            if drop != AIR:
                self.inventory.add(drop)

    def _handle_breaking(self, dt: float, inp) -> None:
        # Attacking a mob wins over starting to dig.
        if inp.was_button_pressed(glfw.MOUSE_BUTTON_LEFT):
            entry = self.inventory.slot(self.selected_slot)
            dmg = 1.0
            if entry and self.registry.tool_damage[entry[0]]:
                dmg = float(self.registry.tool_damage[entry[0]])
            if self.entities.attack_ray(self.player.eye_position, self.camera.forward,
                                        _REACH, dmg):
                self.player.add_exhaustion(0.1)
                self._break_progress = 0.0
                self._break_target = None
                return
        left = inp.is_button_down(glfw.MOUSE_BUTTON_LEFT)
        if not left or self.target is None:
            self._break_progress = 0.0
            self._break_target = None
            return

        block = self.target.block
        block_id = self.world.get_block(*block)
        if not self.registry.by_id[block_id].breakable:
            self._break_progress = 0.0
            self._break_target = None
            return

        if self.player.mode == CREATIVE:
            # Creative: instant, with a small repeat delay while held.
            if inp.was_button_pressed(glfw.MOUSE_BUTTON_LEFT):
                self._break_progress = 1.0
            else:
                self._break_progress += dt / _CREATIVE_BREAK_DELAY
            if self._break_progress >= 1.0:
                self._break_block(block, block_id)
                self._break_progress = 0.0
            return

        # Survival: hold to dig, progress resets when the target changes.
        if block != self._break_target:
            self._break_target = block
            self._break_progress = 0.0
        self._break_progress += dt / self._dig_time(block_id)
        if self._break_progress >= 1.0:
            self._break_block(block, block_id)
            self._break_progress = 0.0
            self._break_target = None

    def _handle_placing(self, dt: float, inp) -> None:
        right = inp.is_button_down(glfw.MOUSE_BUTTON_RIGHT)
        if not right:
            self._place_timer = _PLACE_REPEAT_DELAY  # first press acts instantly
            return
        # RMB on food eats it instead of placing.
        if inp.was_button_pressed(glfw.MOUSE_BUTTON_RIGHT) and self._try_eat():
            return
        self._place_timer += dt
        act = (
            inp.was_button_pressed(glfw.MOUSE_BUTTON_RIGHT)
            or self._place_timer >= _PLACE_REPEAT_DELAY
        )
        if act and self.target is not None and self._try_place(self.target):
            self._place_timer = 0.0

    def _try_eat(self) -> bool:
        entry = self.inventory.slot(self.selected_slot)
        if entry is None:
            return False
        hunger = int(self.registry.food_hunger[entry[0]])
        if hunger <= 0 or not self.player.eat(hunger):
            return False
        self.audio.play("eat")
        if self.player.mode == SURVIVAL:
            self.inventory.consume_slot(self.selected_slot)
        return True

    def _try_place(self, hit: RayHit) -> bool:
        entry = self.inventory.slot(self.selected_slot)
        if entry is None:
            return False
        new_id = entry[0]
        if not self.registry.placeable[new_id]:
            return False  # tools/food are items, not blocks
        cell = hit.previous
        current = self.world.get_block(*cell)
        if not self.registry.replaceable[current]:
            return False
        # Plants and torches need solid ground under them.
        below = self.world.get_block(cell[0], cell[1] - 1, cell[2])
        if self.registry.needs_support[new_id] and not self.registry.solid[below]:
            return False
        # Never let a solid block materialise inside the player.
        if self.registry.solid[new_id] and block_intersects_box(
            cell, self.player.position, Player.HALF_W, Player.HEIGHT
        ):
            return False
        if not self._apply_edit(*cell, new_id):
            return False
        self.audio.play("place")
        if self.player.mode == SURVIVAL:
            self.inventory.consume_slot(self.selected_slot)
        return True

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
        if self.dim_name == "nether":
            # Red murk with a bedrock lid — the Nether has no sky.
            fog_start, fog_end = 6.0, self.settings.render_distance * CHUNK_X * 0.6
            fog_color = (0.32, 0.09, 0.07)
        underwater = self.player.eye_in_fluid_id != 0
        if underwater:
            if self.registry.by_id.get(self.player.eye_in_fluid_id) and \
               self.registry.by_id[self.player.eye_in_fluid_id].name == "lava":
                fog_start, fog_end = 0.5, 6.0
                fog_color = (0.45, 0.12, 0.02)
            else:
                fog_start, fog_end = 2.0, 22.0
                d = max(self.env.daylight, 0.12)
                fog_color = (0.045 * d, 0.14 * d, 0.38 * d)

        def extra_pass() -> None:
            # Entities and weather render into the HDR scene target so they
            # share the terrain's depth buffer and tonemapping.
            self.boxes.begin(self.camera.view_proj())
            self.entities.render(self.boxes, self.env.daylight)
            if self.third_person and not self.player.dead:
                self.player_model.render(
                    self.boxes, self.world, self.player.position,
                    self.camera.yaw, self.env.daylight,
                    moving=float(np.linalg.norm(self.player.velocity[[0, 2]])) > 0.4,
                )
            if self.env.raining:
                self.renderer.render_rain(self.camera, self.clock.time)
            # Particles billboard against the camera; render into the HDR pass.
            right = self.camera.right
            up = np.cross(self.camera.forward, right)
            self.particles.render(self.camera.view_proj(), right, up)

        stats = self.renderer.render_world(
            self.camera,
            self.env,
            self.clock.time,
            fog_start,
            fog_end,
            fog_color=fog_color,
            highlight=self.target.block if self.target else None,
            shadows=self.settings.shadows and self.dim_name != "nether",
            extra_pass=extra_pass,
            underwater=underwater,
            bloom=self.settings.bloom,
        )

        stats.update(self.streamer.stats())
        stats.update({f"ms_{k}": v for k, v in self._prof.items()})
        stats["ms_shadow"] = getattr(self.renderer, "last_shadow_ms", 0.0)

        hand = self.inventory.slot(self.selected_slot)
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
                self.registry.by_id[hand[0]].label
                if hand and self.clock.time < self._hand_label_until
                else ""
            ),
            flying=self.player.flying,
            mode=self.player.mode,
            health=self.player.health,
            max_health=Player.MAX_HEALTH,
            hunger=self.player.hunger,
            max_hunger=Player.MAX_HUNGER,
            air=self.player.air,
            max_air=Player.MAX_AIR,
            breaking=(
                self._break_progress
                if 0.0 < self._break_progress < 1.0 and self.player.mode == SURVIVAL
                else None
            ),
            damage_flash=self.player.damage_flash,
            dead=self.player.dead,
        )
        self.hud.draw(w, h, state)
        self._draw_screens(w, h)

    def _draw_screens(self, width: int, height: int) -> None:
        inp = self.window.input
        mouse = Mouse(*inp.cursor_pos, inp.was_button_pressed(glfw.MOUSE_BUTTON_LEFT))
        if self.screen == "inventory":
            self.selected_slot = self.inv_screen.update(
                width, height, mouse,
                creative=self.player.mode == CREATIVE,
                selected_slot=self.selected_slot,
            )
        elif self.paused:
            s = self.settings
            values = {
                "rd": ("Render distance", str(s.render_distance), "step"),
                "fov": ("Field of view", f"{s.fov:.0f}", "step"),
                "sens": ("Mouse sensitivity", f"{s.mouse_sensitivity:.2f}", "step"),
                "vsync": ("VSync", "ON" if s.vsync else "OFF", "toggle"),
                "full": ("Fullscreen", "ON" if s.fullscreen else "OFF", "toggle"),
                "shad": ("Shadows", "ON" if s.shadows else "OFF", "toggle"),
                "bloom": ("Bloom", "ON" if s.bloom else "OFF", "toggle"),
            }
            action = self.settings_screen.update(width, height, mouse, values)
            if action:
                self._apply_setting(action)

    def _apply_setting(self, action: str) -> None:
        s = self.settings
        if action == "rd+":
            s.render_distance = min(16, s.render_distance + 1)
        elif action == "rd-":
            s.render_distance = max(4, s.render_distance - 1)
        elif action == "fov+":
            s.fov = min(110.0, s.fov + 5.0)
        elif action == "fov-":
            s.fov = max(60.0, s.fov - 5.0)
        elif action == "sens+":
            s.mouse_sensitivity = min(0.30, round(s.mouse_sensitivity + 0.02, 2))
        elif action == "sens-":
            s.mouse_sensitivity = max(0.02, round(s.mouse_sensitivity - 0.02, 2))
        elif action == "vsync!":
            s.vsync = not s.vsync
            self.window.set_vsync(s.vsync)
        elif action == "full!":
            s.fullscreen = not s.fullscreen
            self.window.set_fullscreen(s.fullscreen)
        elif action == "shad!":
            s.shadows = not s.shadows
        elif action == "bloom!":
            s.bloom = not s.bloom
        if action.startswith("rd"):
            for dim in self.dimensions.values():
                dim.streamer.set_render_radius(s.render_distance)
        if action.startswith("fov"):
            self._base_fov = s.fov
            self.camera.set_fov(s.fov)
        save_settings(self.root / "configs" / "settings.json", s)

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
                    "mode": self.player.mode,
                    "health": float(self.player.health),
                    "hunger": float(self.player.hunger),
                    "inventory": self.inventory.to_meta(),
                    "third_person": self.third_person,
                    "dimension": self.dim_name,
                }
            )
            saved = sum(d.world.save_all_modified() for d in self.dimensions.values())
            _log.info("Saved %d modified chunks", saved)
        finally:
            for dim in self.dimensions.values():
                dim.streamer.shutdown()
            self.renderer.release()
            self.window.close()
