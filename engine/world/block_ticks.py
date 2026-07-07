"""Scheduled block updates (plan phase 5).

A small deterministic queue processed at a fixed rate.  Fluids and falling
blocks register here; the queue is deduplicated by position and capped per
game tick so breaking a dam floods gradually instead of hitching.
"""

from __future__ import annotations

import heapq
from typing import Callable

Pos = tuple[int, int, int]
UpdateFn = Callable[[int, int, int], None]


class BlockTickScheduler:
    def __init__(self) -> None:
        self._heap: list[tuple[int, int, Pos]] = []
        self._queued: set[Pos] = set()
        self._seq = 0
        self.tick_count = 0
        self._handlers: list[UpdateFn] = []

    def add_handler(self, fn: UpdateFn) -> None:
        """Handlers run for every popped position (fluids, falling blocks)."""
        self._handlers.append(fn)

    def schedule(self, x: int, y: int, z: int, delay: int) -> None:
        pos = (x, y, z)
        if pos in self._queued:
            return
        self._queued.add(pos)
        self._seq += 1
        heapq.heappush(self._heap, (self.tick_count + max(1, delay), self._seq, pos))

    def tick(self, max_updates: int = 512) -> int:
        """Advance one game tick; process due updates. Returns count handled."""
        self.tick_count += 1
        done = 0
        while self._heap and self._heap[0][0] <= self.tick_count and done < max_updates:
            _, _, pos = heapq.heappop(self._heap)
            self._queued.discard(pos)
            for fn in self._handlers:
                fn(*pos)
            done += 1
        return done


class FallingBlocks:
    """Gravity for sand and gravel (plan 5.6).

    On update, an unsupported falling block teleports straight down to the
    first solid surface — simpler than a falling entity and visually fine at
    tick rate.  The column above is rescheduled so stacks collapse.
    """

    def __init__(self, world, scheduler: BlockTickScheduler) -> None:
        self.world = world
        self.scheduler = scheduler
        reg = world.registry
        self._affected = {reg.id_of("sand"), reg.id_of("gravel")}
        scheduler.add_handler(self.update)

    def notify(self, x: int, y: int, z: int) -> None:
        """A block changed: a falling block above the gap may now drop."""
        for dy in range(0, 3):
            bid = self.world.get_block(x, y + dy, z)
            if bid in self._affected:
                self.scheduler.schedule(x, y + dy, z, 2)

    def update(self, x: int, y: int, z: int) -> None:
        from engine.world.blocks import AIR

        bid = self.world.get_block(x, y, z)
        if bid not in self._affected:
            return
        ny = y
        while ny > 1 and self.world.get_block(x, ny - 1, z) == AIR:
            ny -= 1
        if ny == y:
            return
        self.world.set_block(x, y, z, AIR)
        self.world.set_block(x, ny, z, bid)
        # A block might now be unsupported directly above the old spot.
        above = self.world.get_block(x, y + 1, z)
        if above in self._affected:
            self.scheduler.schedule(x, y + 1, z, 2)
