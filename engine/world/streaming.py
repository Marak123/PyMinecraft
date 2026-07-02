"""Asynchronous chunk streaming.

Pipeline (per the architecture doc, each stage independent):

    request -> generate (worker) -> integrate (main) -> mesh (worker)
            -> upload (main, budgeted) -> visible

Generation and meshing run on a shared thread pool; NumPy releases the GIL
for the heavy array work, so workers genuinely overlap.  All GPU-facing work
(uploads) happens on the main thread via callbacks.
"""

from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

import numpy as np

from engine.core.log import get_logger
from engine.world.chunk import Chunk, ChunkState
from engine.world.world import World

_log = get_logger("streaming")

# Per-frame integration budgets: keep the main thread hitch-free.
MAX_GEN_INTEGRATIONS_PER_FRAME = 12
MAX_MESH_UPLOADS_PER_FRAME = 8

MeshFn = Callable[[np.ndarray], Any]
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
        self.render_radius = render_radius
        # Terrain must exist one ring beyond the render radius because
        # meshing needs all 8 neighbours of a chunk.
        self.gen_radius = render_radius + 1
        self.unload_radius = render_radius + 3

        workers = max(2, (os.cpu_count() or 4) - 2)
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="chunk")
        self._max_inflight = workers * 2
        self._gen_jobs: dict[tuple[int, int], Future] = {}
        self._mesh_jobs: dict[tuple[int, int], Future] = {}
        self._gen_offsets = _ring_offsets(self.gen_radius)
        self._render_offsets = _ring_offsets(self.render_radius)
        _log.info("Chunk streamer using %d worker threads", workers)

    # -- public -------------------------------------------------------------
    def update(self, center: tuple[int, int], forward_xz: tuple[float, float]) -> None:
        self._integrate_generated()
        self._request_generation(center, forward_xz)
        self._integrate_meshes()
        self._request_meshes(center)
        self._remesh_dirty()
        self._unload_far(center)

    def ensure_spawn_area(self, cx: int, cz: int, radius: int = 2) -> None:
        """Blocking bootstrap so the player never spawns into void."""
        for dx, dz in _ring_offsets(radius):
            key = (cx + dx, cz + dz)
            if key not in self.world.chunks:
                blocks, _ = self.world.produce_chunk_blocks(*key)
                self.world.add_chunk(Chunk(key[0], key[1], blocks))
        for dx, dz in _ring_offsets(radius - 1):
            self._mesh_now(cx + dx, cz + dz)

    def stats(self) -> dict[str, int]:
        return {
            "loaded": len(self.world.chunks),
            "pending_gen": len(self._gen_jobs),
            "pending_mesh": len(self._mesh_jobs),
        }

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # -- generation ------------------------------------------------------------
    def _request_generation(
        self, center: tuple[int, int], forward_xz: tuple[float, float]
    ) -> None:
        budget = self._max_inflight - len(self._gen_jobs)
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

    # -- meshing ------------------------------------------------------------
    def _request_meshes(self, center: tuple[int, int]) -> None:
        budget = self._max_inflight - len(self._mesh_jobs)
        if budget <= 0:
            return
        cx, cz = center
        for dx, dz in self._render_offsets:
            if budget <= 0:
                break
            key = (cx + dx, cz + dz)
            chunk = self.world.get_chunk(*key)
            if chunk is None or chunk.state != ChunkState.GENERATED:
                continue
            padded = self.world.build_padded_blocks(*key)
            if padded is None:
                continue  # neighbours still streaming in
            chunk.state = ChunkState.MESHING
            self._mesh_jobs[key] = self._pool.submit(self.mesh_fn, padded)
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
                    chunk.state = ChunkState.GENERATED
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
        padded = self.world.build_padded_blocks(cx, cz)
        if padded is None:
            chunk.state = ChunkState.GENERATED
            return
        self.upload_fn(cx, cz, self.mesh_fn(padded))
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
            if key in self._mesh_jobs:
                continue  # let the in-flight job finish first
            self.world.remove_chunk(*key)
            self.unload_fn(*key)
