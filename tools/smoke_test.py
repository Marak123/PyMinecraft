"""Offscreen smoke test: generate terrain, mesh it, render frames to PNG.

Run:  py tools/smoke_test.py [output_dir]

Uses a standalone GL context + FBO, so it works without opening a window.
Doubles as a micro-benchmark for generation and meshing.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import moderngl
import numpy as np

from engine.camera import Camera
from engine.core.log import init_logging
from engine.graphics.mesher import build_chunk_meshes
from engine.graphics.renderer import Renderer
from engine.world.blocks import BlockRegistry
from engine.world.chunk import Chunk
from engine.world.environment import Environment
from engine.world.generation import WorldGenerator, surface_height
from engine.world.save import WorldStorage
from engine.world.world import World

WIDTH, HEIGHT = 1280, 720
SEED = 1337


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    init_logging(None)

    root = Path(__file__).resolve().parents[1]
    registry = BlockRegistry.load(root / "configs" / "blocks.json")
    generator = WorldGenerator(SEED, registry)
    storage = WorldStorage(root / "saves" / "_smoke_test_unused")
    world = World(generator, registry, storage)

    # -- find land and generate 5x5 chunks around it -----------------------------
    sx, sz = generator.find_spawn()
    scx, scz = sx >> 4, sz >> 4
    print(f"[spawn] land at ({sx}, {sz}) -> chunk ({scx}, {scz})")

    t0 = time.perf_counter()
    gen_count = 0
    for cx in range(scx - 2, scx + 3):
        for cz in range(scz - 2, scz + 3):
            world.add_chunk(Chunk(cx, cz, generator.generate_chunk(cx, cz)))
            gen_count += 1
    gen_ms = (time.perf_counter() - t0) * 1000 / gen_count
    print(f"[gen]  {gen_count} chunks, avg {gen_ms:.1f} ms/chunk")

    # -- GL setup ---------------------------------------------------------------
    ctx = moderngl.create_standalone_context(require=330)
    print(f"[gl]   {ctx.info['GL_VERSION']} | {ctx.info['GL_RENDERER']}")
    renderer = Renderer(ctx, registry, font_atlas=None)

    fbo = ctx.framebuffer(
        color_attachments=[ctx.texture((WIDTH, HEIGHT), 4)],
        depth_attachment=ctx.depth_texture((WIDTH, HEIGHT)),
    )
    fbo.use()
    ctx.viewport = (0, 0, WIDTH, HEIGHT)

    # -- mesh the interior 3x3 (all neighbours present) ---------------------------
    t0 = time.perf_counter()
    mesh_count = 0
    total_verts = 0
    for cx in range(scx - 1, scx + 2):
        for cz in range(scz - 1, scz + 2):
            padded = world.build_padded_blocks(cx, cz)
            assert padded is not None, f"padded failed for {cx},{cz}"
            mesh = build_chunk_meshes(padded, registry)
            renderer.upload_chunk(cx, cz, mesh)
            total_verts += mesh.vertex_count
            mesh_count += 1
    mesh_ms = (time.perf_counter() - t0) * 1000 / mesh_count
    print(f"[mesh] {mesh_count} chunks, avg {mesh_ms:.1f} ms/chunk, {total_verts} verts")
    assert total_verts > 1000, "suspiciously empty meshes"

    # -- render a few angles ------------------------------------------------------
    spawn_chunk = world.get_chunk(scx, scz)
    assert spawn_chunk is not None
    ground = surface_height(spawn_chunk.blocks, sx & 15, sz & 15, registry)
    print(f"[world] ground height at spawn: {ground}")

    env = Environment(day_length_seconds=900.0, start_time=0.35)
    camera = Camera(fov=75.0, aspect=WIDTH / HEIGHT)

    px, pz = sx + 0.5, sz + 0.5
    shots = [
        ("smoke_wide.png", (px - 22.0, ground + 26.0, pz - 22.0), 45.0, -24.0),
        ("smoke_ground.png", (px, ground + 1.7, pz), 30.0, -8.0),
    ]
    for name, pos, yaw, pitch in shots:
        camera.position = np.array(pos)
        camera.yaw = yaw
        camera.pitch = pitch
        t0 = time.perf_counter()
        stats = renderer.render_world(
            camera, env, time_s=1.0, fog_start=200.0, fog_end=420.0
        )
        render_ms = (time.perf_counter() - t0) * 1000
        err = ctx.error
        print(f"[draw] {name}: {stats} in {render_ms:.1f} ms, GL error: {err}")
        assert err == "GL_NO_ERROR", err
        renderer.screenshot(str(out_dir / name))

    print("SMOKE TEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
