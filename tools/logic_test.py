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
from engine.world.blocks import AIR, BlockRegistry
from engine.world.chunk import Chunk
from engine.world.generation import WorldGenerator, surface_height
from engine.world.save import WorldStorage
from engine.world.world import World

SEED = 4242


def make_world(save_dir: Path) -> tuple[World, BlockRegistry]:
    root = Path(__file__).resolve().parents[1]
    registry = BlockRegistry.load(root / "configs" / "blocks.json")
    generator = WorldGenerator(SEED, registry)
    world = World(generator, registry, WorldStorage(save_dir))
    for cx in range(-1, 2):
        for cz in range(-1, 2):
            blocks, _ = world.produce_chunk_blocks(cx, cz)
            world.add_chunk(Chunk(cx, cz, blocks))
    return world, registry


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


def main() -> int:
    init_logging(None)
    tmp = Path(tempfile.mkdtemp(prefix="pymc_test_"))
    try:
        test_edit_and_persistence(tmp / "w1")
        test_physics(tmp / "w2")
        test_raycast(tmp / "w3")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("LOGIC TESTS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
