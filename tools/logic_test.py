"""Headless integration tests: block edits, persistence, physics, raycast.

Run:  py tools/logic_test.py
No GL context needed — exercises world/physics logic only.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from engine.core.log import init_logging
from engine.physics.aabb import move_with_collisions
from engine.physics.raycast import raycast
from engine.world import lighting
from engine.world.blocks import AIR, BlockRegistry
from engine.world.chunk import Chunk
from engine.world.coords import CHUNK_X, CHUNK_Z
from engine.world.generation import WorldGenerator, surface_height
from engine.world.save import WorldStorage
from engine.world.world import World

SEED = 4242


def make_world(save_dir: Path, radius: int = 1) -> tuple[World, BlockRegistry]:
    root = Path(__file__).resolve().parents[1]
    registry = BlockRegistry.load(root / "configs" / "blocks.json")
    generator = WorldGenerator(SEED, registry)
    world = World(generator, registry, WorldStorage(save_dir))
    for cx in range(-radius, radius + 1):
        for cz in range(-radius, radius + 1):
            blocks, _ = world.produce_chunk_blocks(cx, cz)
            world.add_chunk(Chunk(cx, cz, blocks))
    return world, registry


def light_chunk(world: World, registry: BlockRegistry, cx: int, cz: int) -> None:
    window = world.build_light_window(cx, cz)
    assert window is not None
    sky, blk = lighting.compute_window_light(window, registry)
    centre = (slice(CHUNK_X, 2 * CHUNK_X), slice(CHUNK_Z, 2 * CHUNK_Z))
    chunk = world.get_chunk(cx, cz)
    chunk.sky_light = np.ascontiguousarray(sky[centre])
    chunk.block_light = np.ascontiguousarray(blk[centre])


def test_edit_and_persistence(save_dir: Path) -> None:
    world, registry = make_world(save_dir)
    stone = registry.id_of("stone")

    assert world.set_block(5, 90, 5, stone), "set_block failed"
    assert world.get_block(5, 90, 5) == stone
    assert (0, 0) in world.dirty_chunks, "edited chunk not marked dirty"

    # Border edit must dirty the neighbour too (AO/visibility reaches across).
    world.set_block(0, 90, 0, stone)
    assert (-1, 0) in world.dirty_chunks and (0, -1) in world.dirty_chunks

    saved = world.save_all_modified()
    assert saved == 1, f"expected 1 modified chunk, saved {saved}"

    # Reload through the normal production path: disk beats regeneration.
    world2, _ = make_world(save_dir)
    assert world2.get_block(5, 90, 5) == stone, "edit lost after reload"
    print("[ok] edits + persistence")


def test_physics(save_dir: Path) -> None:
    world, registry = make_world(save_dir)
    chunk = world.get_chunk(0, 0)
    assert chunk is not None
    ground = surface_height(chunk.blocks, 8, 8, registry)

    # Drop from 10 blocks up: must land on the surface, not sink into it.
    pos = np.array([8.5, ground + 10.0, 8.5])
    vel = np.zeros(3)
    on_ground = False
    for _ in range(400):
        vel[1] = max(vel[1] - 27.0 * (1 / 60), -55.0)
        res = move_with_collisions(world, pos, vel, 1 / 60, 0.3, 1.8)
        pos = res.position
        if res.on_ground:
            on_ground = True
            vel[1] = 0.0
    assert on_ground, "never landed"
    assert abs(pos[1] - (ground + 1)) < 0.01, f"landed at {pos[1]}, expected ~{ground + 1}"

    # Walk into a wall: X must clamp, no tunnelling.
    wall_x = int(np.floor(pos[0])) + 2
    stone = registry.id_of("stone")
    for y in range(int(pos[1]), int(pos[1]) + 3):
        world.set_block(wall_x, y, int(np.floor(pos[2])), stone)
    vel = np.array([50.0, 0.0, 0.0])
    hit_wall = False
    for _ in range(60):
        res = move_with_collisions(world, pos, vel, 1 / 60, 0.3, 1.8)
        pos = res.position
        if res.hit_wall:
            hit_wall = True
    assert hit_wall, "wall not detected"
    assert pos[0] < wall_x - 0.29, f"tunnelled into wall (x={pos[0]}, wall={wall_x})"
    print(f"[ok] physics (landed, wall clamped x={pos[0]:.3f} vs wall at {wall_x})")


def test_raycast(save_dir: Path) -> None:
    world, registry = make_world(save_dir)
    chunk = world.get_chunk(0, 0)
    assert chunk is not None
    ground = surface_height(chunk.blocks, 8, 8, registry)

    origin = np.array([8.5, ground + 3.0, 8.5])
    hit = raycast(
        world, origin, np.array([0.0, -1.0, 0.0]), 6.0,
        lambda bid: bool(registry.solid[bid]),
    )
    assert hit is not None, "ray straight down missed the ground"
    assert hit.block[1] == ground, f"hit y={hit.block[1]}, expected {ground}"
    assert hit.previous[1] == ground + 1
    print(f"[ok] raycast (hit {hit.block}, from {hit.previous})")


def test_lighting(save_dir: Path) -> None:
    world, registry = make_world(save_dir, radius=2)
    chunk = world.get_chunk(0, 0)

    # Deterministic scene: a 9x9 stone platform floating above the terrain.
    ground = surface_height(chunk.blocks, 8, 8, registry)
    assert ground < 65, "test expects low terrain at this seed"
    stone = registry.id_of("stone")
    chunk.blocks[4:13, 4:13, 70] = stone

    for cx in range(-1, 2):
        for cz in range(-1, 2):
            light_chunk(world, registry, cx, cz)

    assert chunk.sky_light[8, 8, 72] == 15, "open sky above the platform"
    under = int(chunk.sky_light[8, 8, 69])
    assert 0 < under < 15, f"platform must cast a soft shadow (got {under})"
    assert chunk.block_light[8, 8, 71] == 0, "no block light before the torch"

    # Torch placement: incremental relight must spread warm light around.
    torch = registry.id_of("torch")
    assert world.set_block(8, 71, 8, torch)
    changed = lighting.relight_box(world, 8, 71, 8)
    assert changed, "relight reported no changes"
    assert chunk.block_light[8, 8, 71] == 14, "torch cell should hold light 14"
    assert chunk.block_light[8, 8, 73] == 12, "light must decay 1 per block upward"

    # Removing the torch must darken the area again (removal case).
    world.set_block(8, 71, 8, AIR)
    changed = lighting.relight_box(world, 8, 71, 8)
    assert changed, "un-relight reported no changes"
    assert chunk.block_light[8, 8, 71] == 0, "light must vanish with the torch"
    print(f"[ok] lighting (shadow under platform = {under}, torch spread + removal)")


def test_survival(save_dir: Path) -> None:
    from engine.camera import Camera
    from engine.input import InputState
    from game.player import CREATIVE, Player

    world, registry = make_world(save_dir)
    chunk = world.get_chunk(0, 0)
    # Dry landing pad well above the ocean (water landings cancel fall damage
    # by design, so the drop must end on solid ground).
    chunk.blocks[6:11, 6:11, 70] = registry.id_of("stone")

    camera = Camera(fov=75.0, aspect=16 / 9)
    idle = InputState()

    # Survival: a 12-block drop must hurt but not kill.
    player = Player(world, registry, np.array([8.5, 83.0, 8.5]))
    for _ in range(300):
        player.update(1 / 60, idle, camera)
    assert player.on_ground, "player never landed"
    assert abs(player.position[1] - 71.0) < 0.01, f"landed at {player.position[1]}"
    assert player.health < player.MAX_HEALTH, "fall damage not applied"
    assert player.health > 0, "12-block fall should not be lethal"

    # Creative: the same drop is harmless.
    god = Player(world, registry, np.array([8.5, 83.0, 8.5]))
    god.mode = CREATIVE
    for _ in range(300):
        god.update(1 / 60, idle, camera)
    assert god.health == god.MAX_HEALTH, "creative must ignore fall damage"
    print(f"[ok] survival (fall damage: {player.MAX_HEALTH - player.health:.0f} hp, creative immune)")


def main() -> int:
    init_logging(None)
    tmp = Path(tempfile.mkdtemp(prefix="pymc_test_"))
    try:
        test_edit_and_persistence(tmp / "w1")
        test_physics(tmp / "w2")
        test_raycast(tmp / "w3")
        test_lighting(tmp / "w4")
        test_survival(tmp / "w5")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("LOGIC TESTS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
