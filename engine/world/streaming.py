"""Asynchronous chunk streaming.

Pipeline (each stage independent, per the architecture doc):

    request -> generate (worker) -> integrate (main)
            -> light (worker, 3x3 window) -> integrate (main)
            -> mesh (worker) -> upload (main, budgeted) -> visible

Stage radii nest: terrain must exist one ring beyond lighting (light needs
a 3x3 block window) and light one ring beyond meshing (meshing needs a lit
halo), so gen = render + 2, light = render + 1, mesh = render.

Generation, lighting, meshing and chunk saving all run on a shared thread
pool; NumPy releases the GIL for the heavy array work.  All GPU-facing work
(uploads) happens on the main thread via callbacks.
"""

from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

import numpy as np

from engine.core.log import get_logger
from engine.world.chunk import Chunk, ChunkState
from engine.world.coords import CHUNK_X, CHUNK_Z
from engine.world.lighting import compute_window_light
from engine.world.world import World

_log = get_logger("streaming")

# Per-frame budgets: keep the main thread hitch-free.  Submissions are
# capped too because building job snapshots (light windows, mesh halos)
# copies arrays on the main thread.
MAX_GEN_INTEGRATIONS_PER_FRAME = 12
MAX_LIGHT_INTEGRATIONS_PER_FRAME = 10
MAX_MESH_UPLOADS_PER_FRAME = 8
MAX_GEN_SUBMITS_PER_FRAME = 8
MAX_LIGHT_SUBMITS_PER_FRAME = 4
MAX_MESH_SUBMITS_PER_FRAME = 5

MeshFn = Callable[[tuple[np.ndarray, np.ndarray, np.ndarray]], Any]
UploadFn = Callable[[int, int, Any], None]
UnloadFn = Callable[[int, int], None]


def _ring_offsets(radius: int) -> list[tuple[int, int]]:
    """Chunk offsets within a circular radius, sorted nearest-first."""
    offsets = [
        (dx, dz)
        for dx in range(-radius, radius + 1)
        for dz in range(-radius, radius + 1)
        if dx * dx + dz * dz <= (radius + 0.5) ** 2
    ]
    offsets.sort(key=lambda o: o[0] * o[0] + o[1] * o[1])
    return offsets


class ChunkStreamer:
    def __init__(
        self,
        world: World,
        mesh_fn: MeshFn,
        upload_fn: UploadFn,
        unload_fn: UnloadFn,
        render_radius: int,
    ) -> None:
        self.world = world
        self.mesh_fn = mesh_fn
        self.upload_fn = upload_fn
        self.unload_fn = unload_fn
        self.set_render_radius(render_radius)

        workers = max(2, (os.cpu_count() or 4) - 2)
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="chunk")
        self._max_inflight = workers * 2
        self._gen_jobs: dict[tuple[int, int], Future] = {}
        self._light_jobs: dict[tuple[int, int], Future] = {}
        self._mesh_jobs: dict[tuple[int, int], Future] = {}
        _log.info("Chunk streamer using %d worker threads", workers)

    def set_render_radius(self, render_radius: int) -> None:
        """Live render-distance change (settings menu)."""
        self.render_radius = render_radius
        self.light_radius = render_radius + 1
        self.gen_radius = render_radius + 2
        self.unload_radius = render_radius + 4
        self._gen_offsets = _ring_offsets(self.gen_radius)
        self._light_offsets = _ring_offsets(self.light_radius)
        self._render_offsets = _ring_offsets(self.render_radius)

    # -- public -------------------------------------------------------------
    def update(self, center: tuple[int, int], forward_xz: tuple[float, float]) -> None:
        self._integrate_generated()
        self._request_generation(center, forward_xz)
        self._integrate_lighting()
        self._request_lighting(center)
        self._integrate_meshes()
        self._request_meshes(center)
        self._remesh_dirty()
        self._unload_far(center)

    def ensure_spawn_area(self, cx: int, cz: int, radius: int = 1) -> None:
        """Blocking bootstrap so the player never spawns into void."""
        for dx, dz in _ring_offsets(radius + 2):
            key = (cx + dx, cz + dz)
            if key not in self.world.chunks:
                blocks, _ = self.world.produce_chunk_blocks(*key)
                self.world.add_chunk(Chunk(key[0], key[1], blocks))
        for dx, dz in _ring_offsets(radius + 1):
            self._light_now(cx + dx, cz + dz)
        for dx, dz in _ring_offsets(radius):
            self._mesh_now(cx + dx, cz + dz)

    def stats(self) -> dict[str, int]:
        return {
            "loaded": len(self.world.chunks),
            "pending_gen": len(self._gen_jobs),
            "pending_light": len(self._light_jobs),
            "pending_mesh": len(self._mesh_jobs),
        }

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)

    # -- generation ------------------------------------------------------------
    def _request_generation(
        self, center: tuple[int, int], forward_xz: tuple[float, float]
    ) -> None:
        budget = min(self._max_inflight - len(self._gen_jobs), MAX_GEN_SUBMITS_PER_FRAME)
        if budget <= 0:
            return
        cx, cz = center
        fx, fz = forward_xz
        candidates: list[tuple[float, tuple[int, int]]] = []
        for dx, dz in self._gen_offsets:
            key = (cx + dx, cz + dz)
            if key in self.world.chunks or key in self._gen_jobs:
                continue
            # Chunks in front of the camera stream in first.
            dist2 = dx * dx + dz * dz
            facing_bonus = (dx * fx + dz * fz) * 2.0
            candidates.append((dist2 - facing_bonus, key))
        candidates.sort(key=lambda c: c[0])
        for _, key in candidates[:budget]:
            self._gen_jobs[key] = self._pool.submit(
                self.world.produce_chunk_blocks, *key
            )

    def _integrate_generated(self) -> None:
        done = [(k, f) for k, f in self._gen_jobs.items() if f.done()]
        for key, future in done[:MAX_GEN_INTEGRATIONS_PER_FRAME]:
            del self._gen_jobs[key]
            try:
                blocks, _ = future.result()
            except Exception:  # noqa: BLE001 - worker crash must not kill the loop
                _log.exception("Chunk generation failed for %s", key)
                continue
            self.world.add_chunk(Chunk(key[0], key[1], blocks))

    # -- lighting ------------------------------------------------------------
    def _light_job(self, window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sky, blk = compute_window_light(window, self.world.registry)
        centre = (slice(CHUNK_X, 2 * CHUNK_X), slice(CHUNK_Z, 2 * CHUNK_Z))
        return (
            np.ascontiguousarray(sky[centre]),
            np.ascontiguousarray(blk[centre]),
        )

    def _request_lighting(self, center: tuple[int, int]) -> None:
        budget = min(self._max_inflight - len(self._light_jobs), MAX_LIGHT_SUBMITS_PER_FRAME)
        if budget <= 0:
            return
        cx, cz = center
        for dx, dz in self._light_offsets:
            if budget <= 0:
                break
            key = (cx + dx, cz + dz)
            chunk = self.world.get_chunk(*key)
            if chunk is None or chunk.state != ChunkState.GENERATED:
                continue
            window = self.world.build_light_window(*key)
            if window is None:
                continue  # neighbours still streaming in
            chunk.state = ChunkState.LIGHTING
            self._light_jobs[key] = self._pool.submit(self._light_job, window)
            budget -= 1

    def _integrate_lighting(self) -> None:
        done = [(k, f) for k, f in self._light_jobs.items() if f.done()]
        for key, future in done[:MAX_LIGHT_INTEGRATIONS_PER_FRAME]:
            del self._light_jobs[key]
            chunk = self.world.get_chunk(*key)
            try:
                sky, blk = future.result()
            except Exception:  # noqa: BLE001
                _log.exception("Lighting failed for %s", key)
                if chunk is not None:
                    chunk.state = ChunkState.GENERATED
                continue
            if chunk is None:
                continue  # unloaded while lighting
            chunk.sky_light = sky
            chunk.block_light = blk
            chunk.state = ChunkState.LIT

    def _light_now(self, cx: int, cz: int) -> None:
        """Synchronous lighting — spawn bootstrap only."""
        chunk = self.world.get_chunk(cx, cz)
        if chunk is None or chunk.has_light:
            return
        window = self.world.build_light_window(cx, cz)
        if window is None:
            return
        chunk.sky_light, chunk.block_light = self._light_job(window)
        chunk.state = ChunkState.LIT

    # -- meshing ------------------------------------------------------------
    def _request_meshes(self, center: tuple[int, int]) -> None:
        budget = min(self._max_inflight - len(self._mesh_jobs), MAX_MESH_SUBMITS_PER_FRAME)
        if budget <= 0:
            return
        cx, cz = center
        for dx, dz in self._render_offsets:
            if budget <= 0:
                break
            key = (cx + dx, cz + dz)
            chunk = self.world.get_chunk(*key)
            if chunk is None or chunk.state != ChunkState.LIT:
                continue
            mesh_input = self.world.build_mesh_input(*key)
            if mesh_input is None:
                continue  # neighbour light still streaming in
            chunk.state = ChunkState.MESHING
            self._mesh_jobs[key] = self._pool.submit(self.mesh_fn, mesh_input)
            budget -= 1

    def _integrate_meshes(self) -> None:
        done = [(k, f) for k, f in self._mesh_jobs.items() if f.done()]
        for key, future in done[:MAX_MESH_UPLOADS_PER_FRAME]:
            del self._mesh_jobs[key]
            chunk = self.world.get_chunk(*key)
            try:
                mesh = future.result()
            except Exception:  # noqa: BLE001
                _log.exception("Meshing failed for %s", key)
                if chunk is not None:
                    chunk.state = ChunkState.LIT
                continue
            if chunk is None:
                continue  # unloaded while meshing
            self.upload_fn(key[0], key[1], mesh)
            chunk.state = ChunkState.READY

    def _mesh_now(self, cx: int, cz: int) -> None:
        """Synchronous mesh+upload — used for edits and spawn bootstrap."""
        chunk = self.world.get_chunk(cx, cz)
        if chunk is None:
            return
        mesh_input = self.world.build_mesh_input(cx, cz)
        if mesh_input is None:
            chunk.state = (
                ChunkState.LIT if chunk.has_light else ChunkState.GENERATED
            )
            return
        self.upload_fn(cx, cz, self.mesh_fn(mesh_input))
        chunk.state = ChunkState.READY

    def _remesh_dirty(self) -> None:
        if not self.world.dirty_chunks:
            return
        # Player edits remesh synchronously: instant feedback beats budget.
        # Chunks with an async mesh already in flight stay dirty — the stale
        # in-flight result must not silently erase the edit.
        deferred: set[tuple[int, int]] = set()
        for key in sorted(self.world.dirty_chunks):
            if key in self._mesh_jobs:
                deferred.add(key)
            else:
                self._mesh_now(*key)
        self.world.dirty_chunks = deferred

    # -- unloading ------------------------------------------------------------
    def _unload_far(self, center: tuple[int, int]) -> None:
        cx, cz = center
        limit2 = (self.unload_radius + 0.5) ** 2
        to_remove = [
            key
            for key in self.world.chunks
            if (key[0] - cx) ** 2 + (key[1] - cz) ** 2 > limit2
        ]
        for key in to_remove:
            if key in self._mesh_jobs or key in self._light_jobs:
                continue  # let the in-flight job finish first
            chunk = self.world.remove_chunk(*key)
            if chunk is not None and chunk.modified:
                # The chunk is detached — nothing mutates it anymore, so the
                # save can run on the pool without copying.
                chunk.modified = False
                self._pool.submit(
                    self.world.storage.save_chunk, key[0], key[1], chunk.blocks
                )
            self.unload_fn(*key)
