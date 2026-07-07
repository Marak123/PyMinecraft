"""Cellular-automata fluids: flowing water and lava (plan phase 5).

Flow state lives in dedicated block ids (source + a few flow levels) so no
per-block metadata array is needed.  A block updates on a scheduled tick:
it falls into air below, else spreads to lower-level horizontal neighbours,
and dries up if nothing feeds it.  Water meeting lava turns to stone/obsidian.
"""

from __future__ import annotations

from engine.world.blocks import AIR, BlockRegistry
from engine.world.block_ticks import BlockTickScheduler
from engine.world.world import World

_WATER, _LAVA = 1, 2
_HORIZONTAL = ((1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1))
_NEIGHBOURS = _HORIZONTAL + ((0, 1, 0), (0, -1, 0))


class FluidSimulator:
    def __init__(self, world: World, scheduler: BlockTickScheduler,
                 registry: BlockRegistry) -> None:
        self.world = world
        self.scheduler = scheduler
        self.reg = registry
        self._obsidian = registry.id_of("obsidian")
        self._stone = registry.id_of("stone")
        self._cobble = registry.id_of("cobblestone")
        self._delay = {_WATER: 5, _LAVA: 14}
        scheduler.add_handler(self.update)

    # -- edit notification ----------------------------------------------------
    def on_block_changed(self, x: int, y: int, z: int) -> None:
        """A block was placed/broken: wake any fluid touching this cell."""
        for dx, dy, dz in _NEIGHBOURS:
            nx, ny, nz = x + dx, y + dy, z + dz
            kind = self._kind(self.world.get_block(nx, ny, nz))
            if kind:
                self.scheduler.schedule(nx, ny, nz, self._delay[kind])
        # The edited cell itself may need to start flowing/falling.
        if self._kind(self.world.get_block(x, y, z)):
            self.scheduler.schedule(x, y, z, 2)

    # -- helpers ----------------------------------------------------------------
    def _kind(self, bid: int) -> int:
        return int(self.reg.fluid_kind[bid])

    def _level(self, bid: int) -> int:
        return int(self.reg.fluid_level[bid])

    def _is_source(self, bid: int) -> bool:
        return bool(self.reg.fluid_source[bid])

    def _flow_id(self, kind: int, level: int) -> int | None:
        return self.reg.fluid_by_level.get((kind, level))

    def _set(self, x: int, y: int, z: int, bid: int) -> None:
        if self.world.set_block(x, y, z, bid):
            # Waking neighbours here keeps a flood spreading tick by tick.
            for dx, dy, dz in _NEIGHBOURS:
                nb = self.world.get_block(x + dx, y + dy, z + dz)
                k = self._kind(nb)
                if k:
                    self.scheduler.schedule(x + dx, y + dy, z + dz, self._delay[k])

    # -- the per-block rule -----------------------------------------------------
    def update(self, x: int, y: int, z: int) -> None:
        bid = self.world.get_block(x, y, z)
        kind = self._kind(bid)
        if not kind:
            return
        level = self._level(bid)
        source = self._is_source(bid)

        # 1. Flow blocks dry up unless still fed by a source or a shallower
        #    (closer-to-source) neighbour.
        if not source and not self._has_support(x, y, z, kind, level):
            self.world.set_block(x, y, z, AIR)
            self._wake_neighbours(x, y, z)
            return

        below = self.world.get_block(x, y - 1, z)
        below_kind = self._kind(below)

        # 2. Fall straight down into air or shallower fluid.
        if below == AIR or (below_kind == kind and not self._is_source(below)
                            and self._level(below) > 1):
            if below_kind == _WATER and kind == _LAVA or below_kind == _LAVA and kind == _WATER:
                self._interact(x, y - 1, z, kind)
            else:
                falling = self._flow_id(kind, 1)  # full column below a feed
                if falling is not None:
                    self._set(x, y - 1, z, falling)
            return

        # 3. On solid/own fluid ground: spread horizontally one level weaker.
        if level < self.reg.max_flow[kind]:
            next_level = level + 1
            flow_id = self._flow_id(kind, next_level)
            if flow_id is not None:
                for dx, _, dz in _HORIZONTAL:
                    self._spread(x + dx, y, z + dz, kind, next_level, flow_id)

    def _spread(self, x: int, y: int, z: int, kind: int, level: int, flow_id: int) -> None:
        target = self.world.get_block(x, y, z)
        tkind = self._kind(target)
        if target == AIR:
            self._set(x, y, z, flow_id)
        elif tkind and tkind != kind:
            self._interact(x, y, z, kind)
        elif tkind == kind and not self._is_source(target) and self._level(target) > level:
            self._set(x, y, z, flow_id)  # deepen a weaker flow

    def _has_support(self, x: int, y: int, z: int, kind: int, level: int) -> bool:
        # A source directly above always feeds; otherwise a horizontal
        # neighbour of the same kind that is closer to a source (lower level).
        above = self.world.get_block(x, y + 1, z)
        if self._kind(above) == kind:
            return True
        for dx, _, dz in _HORIZONTAL:
            nb = self.world.get_block(x + dx, y, z + dz)
            if self._kind(nb) == kind and (self._is_source(nb) or self._level(nb) < level):
                return True
        return False

    def _interact(self, x: int, y: int, z: int, incoming_kind: int) -> None:
        """Water + lava meet: hot side hardens (plan 5.4)."""
        target = self.world.get_block(x, y, z)
        if incoming_kind == _WATER:
            result = self._obsidian if self._is_source(target) else self._cobble
        else:  # lava flowing into water
            result = self._stone
        self.world.set_block(x, y, z, result)
        self._wake_neighbours(x, y, z)

    def _wake_neighbours(self, x: int, y: int, z: int) -> None:
        for dx, dy, dz in _NEIGHBOURS:
            k = self._kind(self.world.get_block(x + dx, y + dy, z + dz))
            if k:
                self.scheduler.schedule(x + dx, y + dy, z + dz, self._delay[k])
