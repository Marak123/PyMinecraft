"""Survival inventory (36 slots incl. 9-slot hotbar) and shapeless crafting.

Slots hold ``[block_id, count]`` pairs or None.  The first 9 slots ARE the
hotbar — no duplication of state.  Recipes are data (configs/recipes.json).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from engine.core.log import get_logger
from engine.world.blocks import BlockRegistry

_log = get_logger("inventory")

STACK_SIZE = 64
SLOTS = 36
HOTBAR_SLOTS = 9


class Inventory:
    def __init__(self) -> None:
        self.slots: list[list[int] | None] = [None] * SLOTS

    # -- queries ------------------------------------------------------------
    def slot(self, index: int) -> tuple[int, int] | None:
        entry = self.slots[index]
        return (entry[0], entry[1]) if entry else None

    def count(self, block_id: int) -> int:
        return sum(e[1] for e in self.slots if e and e[0] == block_id)

    # -- mutation ------------------------------------------------------------
    def add(self, block_id: int, amount: int = 1) -> int:
        """Add items, filling existing stacks first. Returns leftover."""
        for entry in self.slots:
            if amount <= 0:
                break
            if entry and entry[0] == block_id and entry[1] < STACK_SIZE:
                take = min(STACK_SIZE - entry[1], amount)
                entry[1] += take
                amount -= take
        for i in range(SLOTS):
            if amount <= 0:
                break
            if self.slots[i] is None:
                take = min(STACK_SIZE, amount)
                self.slots[i] = [block_id, take]
                amount -= take
        return amount

    def remove(self, block_id: int, amount: int = 1) -> bool:
        """Consume items if enough exist anywhere in the inventory."""
        if self.count(block_id) < amount:
            return False
        for i, entry in enumerate(self.slots):
            if amount <= 0:
                break
            if entry and entry[0] == block_id:
                take = min(entry[1], amount)
                entry[1] -= take
                amount -= take
                if entry[1] == 0:
                    self.slots[i] = None
        return True

    def consume_slot(self, index: int) -> bool:
        entry = self.slots[index]
        if not entry:
            return False
        entry[1] -= 1
        if entry[1] <= 0:
            self.slots[index] = None
        return True

    def swap(self, a: int, b: int) -> None:
        self.slots[a], self.slots[b] = self.slots[b], self.slots[a]

    # -- persistence ------------------------------------------------------------
    def to_meta(self) -> list:
        return [list(e) if e else None for e in self.slots]

    def load_meta(self, data: list) -> None:
        for i in range(min(SLOTS, len(data))):
            entry = data[i]
            self.slots[i] = [int(entry[0]), int(entry[1])] if entry else None


@dataclass
class Recipe:
    output: int
    count: int
    ingredients: dict[int, int]  # block_id -> amount
    label: str


class CraftingBook:
    def __init__(self, recipes: list[Recipe]) -> None:
        self.recipes = recipes

    @classmethod
    def load(cls, path: Path, registry: BlockRegistry) -> "CraftingBook":
        data = json.loads(path.read_text(encoding="utf-8"))
        recipes = []
        for entry in data["recipes"]:
            output = registry.id_of(entry["output"])
            if entry.get("type") == "shaped":
                # Flatten the pattern to ingredient counts — the list-based
                # crafting UI is position-independent (3x3 grid deferred).
                counts: dict[int, int] = {}
                key = entry["key"]
                for row in entry["pattern"]:
                    for sym in row:
                        if sym != " " and sym in key:
                            bid = registry.id_of(key[sym])
                            counts[bid] = counts.get(bid, 0) + 1
                ingredients = counts
            else:
                ingredients = {
                    registry.id_of(name): int(n)
                    for name, n in entry["ingredients"].items()
                }
            recipes.append(
                Recipe(
                    output=output,
                    count=int(entry.get("count", 1)),
                    ingredients=ingredients,
                    label=registry.by_id[output].label,
                )
            )
        _log.info("Loaded %d recipes", len(recipes))
        return cls(recipes)

    @staticmethod
    def can_craft(recipe: Recipe, inv: Inventory) -> bool:
        return all(inv.count(bid) >= n for bid, n in recipe.ingredients.items())

    @staticmethod
    def craft(recipe: Recipe, inv: Inventory) -> bool:
        if not CraftingBook.can_craft(recipe, inv):
            return False
        for bid, n in recipe.ingredients.items():
            inv.remove(bid, n)
        leftover = inv.add(recipe.output, recipe.count)
        if leftover:  # inventory full: refund what we can, drop the rest
            _log.info("Inventory full, %d crafted items lost", leftover)
        return True
