"""Offscreen smoke test: generate, light and mesh terrain, render PNGs.

Run:  py tools/smoke_test.py [output_dir]

Uses a standalone GL context + FBO, so it works without opening a window.
Doubles as a micro-benchmark for generation, lighting and meshing, and
exercises the full edit pipeline (torch placement + incremental relight)
for the night shot.
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
from engine.world import lighting
from engine.world.blocks import BlockRegistry
from engine.world.chunk import Chunk
from engine.world.coords import CHUNK_X, CHUNK_Z
from engine.world.environment import Environment
from engine.world.generation import WorldGenerator, surface_height
from engine.world.save import WorldStorage
from engine.world.world import World

WIDTH, HEIGHT = 1280, 720
SEED = 1337


def light_chunk(world: World, registry: BlockRegistry, cx: int, cz: int) -> None:
    window = world.build_light_window(cx, cz)
    assert window is not None, f"light window failed for {cx},{cz}"
    sky, blk = lighting.compute_window_light(window, registry)
    centre = (slice(CHUNK_X, 2 * CHUNK_X), slice(CHUNK_Z, 2 * CHUNK_Z))
    chunk = world.get_chunk(cx, cz)
    chunk.sky_light = np.ascontiguousarray(sky[centre])
    chunk.block_light = np.ascontiguousarray(blk[centre])


def mesh_chunk(world: World, registry: BlockRegistry, renderer: Renderer, cx: int, cz: int) -> int:
    mesh_input = world.build_mesh_input(cx, cz)
    assert mesh_input is not None, f"mesh input failed for {cx},{cz}"
    mesh = build_chunk_meshes(mesh_input, registry)
    renderer.upload_chunk(cx, cz, mesh)
    return mesh.vertex_count


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    init_logging(None)

    root = Path(__file__).resolve().parents[1]
    registry = BlockRegistry.load(root / "configs" / "blocks.json")
    generator = WorldGenerator(SEED, registry)
    storage = WorldStorage(root / "saves" / "_smoke_test_unused")
    world = World(generator, registry, storage)

    # -- find land, then gen 7x7 / light 5x5 / mesh 3x3 around it ---------------
    sx, sz = generator.find_spawn()
    scx, scz = sx >> 4, sz >> 4
    print(f"[spawn] land at ({sx}, {sz}) -> chunk ({scx}, {scz})")

    t0 = time.perf_counter()
    gen_count = 0
    for cx in range(scx - 3, scx + 4):
        for cz in range(scz - 3, scz + 4):
            world.add_chunk(Chunk(cx, cz, generator.generate_chunk(cx, cz)))
            gen_count += 1
    gen_ms = (time.perf_counter() - t0) * 1000 / gen_count
    print(f"[gen]   {gen_count} chunks, avg {gen_ms:.1f} ms/chunk")

    t0 = time.perf_counter()
    light_count = 0
    for cx in range(scx - 2, scx + 3):
        for cz in range(scz - 2, scz + 3):
            light_chunk(world, registry, cx, cz)
            light_count += 1
    light_ms = (time.perf_counter() - t0) * 1000 / light_count
    print(f"[light] {light_count} chunks, avg {light_ms:.1f} ms/chunk")

    # -- GL setup ---------------------------------------------------------------
    ctx = moderngl.create_standalone_context(require=330)
    print(f"[gl]    {ctx.info['GL_VERSION']} | {ctx.info['GL_RENDERER']}")
    renderer = Renderer(ctx, registry, font_atlas=None)

    fbo = ctx.framebuffer(
        color_attachments=[ctx.texture((WIDTH, HEIGHT), 4)],
        depth_attachment=ctx.depth_texture((WIDTH, HEIGHT)),
    )
    fbo.use()
    ctx.viewport = (0, 0, WIDTH, HEIGHT)

    t0 = time.perf_counter()
    total_verts = 0
    mesh_count = 0
    for cx in range(scx - 1, scx + 2):
        for cz in range(scz - 1, scz + 2):
            total_verts += mesh_chunk(world, registry, renderer, cx, cz)
            mesh_count += 1
    mesh_ms = (time.perf_counter() - t0) * 1000 / mesh_count
    print(f"[mesh]  {mesh_count} chunks, avg {mesh_ms:.1f} ms/chunk, {total_verts} verts")
    assert total_verts > 1000, "suspiciously empty meshes"

    spawn_chunk = world.get_chunk(scx, scz)
    ground = surface_height(spawn_chunk.blocks, sx & 15, sz & 15, registry)
    print(f"[world] ground height at spawn: {ground}")

    # Sanity: full sun above ground, torch light spreads after an edit.
    sky_at = int(spawn_chunk.sky_light[sx & 15, sz & 15, min(ground + 2, 127)])
    assert sky_at == 15, f"expected full skylight above ground, got {sky_at}"

    camera = Camera(fov=75.0, aspect=WIDTH / HEIGHT)
    px, pz = sx + 0.5, sz + 0.5

    def shoot(name: str, env: Environment, pos, yaw: float, pitch: float) -> None:
        camera.position = np.array(pos)
        camera.yaw = yaw
        camera.pitch = pitch
        t = time.perf_counter()
        stats = renderer.render_world(
            camera, env, time_s=1.0, fog_start=200.0, fog_end=420.0
        )
        ms = (time.perf_counter() - t) * 1000
        err = ctx.error
        print(f"[draw]  {name}: {stats} in {ms:.1f} ms, GL error: {err}")
        assert err == "GL_NO_ERROR", err
        renderer.screenshot(str(out_dir / name))

    day = Environment(day_length_seconds=900.0, start_time=0.35)
    shoot("smoke_wide.png", day, (px - 22.0, ground + 26.0, pz - 22.0), 45.0, -24.0)
    shoot("smoke_ground.png", day, (px, ground + 1.7, pz), 30.0, -8.0)

    # -- torches at night: exercises set_block + incremental relight -------------
    placed = 0
    torch = registry.id_of("torch")
    for dx, dz in ((0, 0), (4, 2), (-3, 3), (2, -4)):
        bx, bz = sx + dx, sz + dz
        chunk = world.get_chunk(bx >> 4, bz >> 4)
        h = surface_height(chunk.blocks, bx & 15, bz & 15, registry)
        if world.get_block(bx, h + 1, bz) == 0:
            world.set_block(bx, h + 1, bz, torch)
            world.dirty_chunks |= lighting.relight_box(world, bx, h + 1, bz)
            placed += 1
    assert placed >= 2, "could not place test torches"
    for key in sorted(world.dirty_chunks):
        if abs(key[0] - scx) <= 1 and abs(key[1] - scz) <= 1:
            mesh_chunk(world, registry, renderer, *key)
    world.dirty_chunks.clear()
    print(f"[edit]  placed {placed} torches + relit + remeshed")

    night = Environment(day_length_seconds=900.0, start_time=0.92)
    print(f"[env]   night daylight = {night.daylight:.3f}")
    shoot("smoke_night.png", night, (px - 10.0, ground + 7.0, pz - 10.0), 45.0, -18.0)

    print("SMOKE TEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
