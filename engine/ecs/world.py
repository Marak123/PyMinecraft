"""Minimal ECS world (plan phase 7.1).

Entities are integer ids; components live in per-type dicts.  No archetype
storage — plain dict lookup is plenty for the few hundred entities a voxel
sandbox keeps active.  Systems are callables run in registration order.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator

System = Callable[["ECSWorld", float], None]


class ECSWorld:
    def __init__(self) -> None:
        self._next_id = 1
        self._components: dict[type, dict[int, Any]] = {}
        self._alive: set[int] = set()
        self._systems: list[System] = []

    # -- entities -------------------------------------------------------------
    def create_entity(self) -> int:
        eid = self._next_id
        self._next_id += 1
        self._alive.add(eid)
        return eid

    def destroy_entity(self, eid: int) -> None:
        self._alive.discard(eid)
        for store in self._components.values():
            store.pop(eid, None)

    def is_alive(self, eid: int) -> bool:
        return eid in self._alive

    # -- components -----------------------------------------------------------
    def add(self, eid: int, component: Any) -> Any:
        self._components.setdefault(type(component), {})[eid] = component
        return component

    def get(self, eid: int, comp_type: type) -> Any | None:
        return self._components.get(comp_type, {}).get(eid)

    def has(self, eid: int, comp_type: type) -> bool:
        return eid in self._components.get(comp_type, {})

    def store(self, comp_type: type) -> dict[int, Any]:
        return self._components.get(comp_type, {})

    def with_(self, *comp_types: type) -> Iterator[int]:
        """Yield entity ids that own every requested component type."""
        if not comp_types:
            return
        stores = [self._components.get(t, {}) for t in comp_types]
        smallest = min(stores, key=len)
        for eid in list(smallest.keys()):
            if all(eid in s for s in stores):
                yield eid

    def count(self, comp_type: type) -> int:
        return len(self._components.get(comp_type, {}))

    # -- systems --------------------------------------------------------------
    def add_system(self, system: System) -> None:
        self._systems.append(system)

    def update(self, dt: float) -> None:
        for system in self._systems:
            system(self, dt)
